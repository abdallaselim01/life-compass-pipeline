"""Self-maintenance and quiet alerts (design decision 2026-11-18, part D: Fully Automated, and no notification headaches).

health_check.py runs this hourly. The order is always: look, repair, and only then (if something is still wrong) tell the user.

  Checks            listener ANSWERING (an HTTP request, not only the port open), the scheduled tasks present and enabled (re-registered from
                    windows_config/scheduled_tasks when not), the database passing SQLite's quick check, free disk space, and the
                    Gemini chain not being down in full, and the Telegram token / Gemini keys not being rejected by their provider.
  Repairs           a listener that is down is started the way the watchdog starts it; a missing or disabled task is registered or
                    enabled again. Anything repaired is written to the log only.
  Alerts            ONE Telegram message for a problem that is still there after the repair attempt (and has lasted as long as its
                    own patience: a listener 'alerts.persist_minutes', a fully failing Gemini chain 'alerts.models_down_hours', the
                    rest at once). A problem is told once, never repeated until it was fixed and came back; at most
                    'alerts.max_per_day' messages a day, several problems going in ONE message; no "all is well" messages, no
                    weekly report, no quiet hours (the user has no fixed sleep times). When Telegram itself cannot be reached the
                    problem stays unannounced and is retried at the next run.
  Maintenance       the weekly yt-dlp update (ytdlp_update.py). Quiet: only the log.

The alert rules are one pure function (apply_policy) so they can be tested without any of the rest.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import tunables
from .logging_setup import get_logger

file_log = get_logger("health")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATE_PATH = PROJECT_ROOT / "data" / "health_state.json"
_DB_PATH = PROJECT_ROOT / "data" / "principles.db"
_TASK_XML_DIR = PROJECT_ROOT / "windows_config" / "scheduled_tasks"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Issue:
    key: str               # stable name of the problem: what "the same problem" means across runs
    text: str              # the one line the user reads (plain Arabic, no technical detail)
    after_minutes: float = 0.0   # how long the problem must have been seen before the user is told


# ---------------------------------------------------------------- the alert rules (pure)
def apply_policy(state: dict, issues: list[Issue], now: float, send: Callable[[str], bool]) -> dict:
    """Update `state` for this run's `issues` and send at most one message. Returns the state.

    state = {"issues": {key: {"first_seen", "alerted_at", "text"}}, "alerts": {"date", "count"}}"""
    known = state.setdefault("issues", {})
    current = {issue.key: issue for issue in issues}
    for key in [k for k in known if k not in current]:
        del known[key]                                            # fixed: forgotten without a word
    for issue in issues:
        known.setdefault(issue.key, {"first_seen": now, "alerted_at": None, "text": issue.text})["text"] = issue.text
    ready = [i for i in issues if known[i.key]["alerted_at"] is None and now - known[i.key]["first_seen"] >= i.after_minutes * 60]
    sent = state.setdefault("alerts", {})
    today = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
    if sent.get("date") != today:
        sent.update(date=today, count=0)
    if ready and sent["count"] < tunables.get("alerts", "max_per_day"):
        message = "⚠️ Life Compass\n" + "\n".join(f"• {i.text}" for i in ready)
        if send(message):
            for issue in ready:
                known[issue.key]["alerted_at"] = now
            sent["count"] += 1
        else:
            file_log.warning("alert not sent (Telegram could not be reached); it is retried at the next run")
    return state


# ---------------------------------------------------------------- checks (each repairs first, reports what is left)
def _powershell(script: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", creationflags=_NO_WINDOW, timeout=timeout)


def check_listener() -> list[Issue]:
    """The listener answers (liveness.ensure starts it when it is down, and restarts it when it holds the port without answering)."""
    from . import liveness
    if liveness.ensure() == "up":
        return []
    for _ in range(12):
        time.sleep(2)
        if liveness.probe() == "up":
            file_log.info("listener is up again")
            return []
    return [Issue("listener_down", "The system is down (the listener is not responding and could not be restarted)", tunables.get("alerts", "persist_minutes"))]


def task_files() -> dict[str, Path]:
    """{task name: its saved XML} for every scheduled task the system has a copy of."""
    import re
    found = {}
    for xml in sorted(_TASK_XML_DIR.glob("*.xml")):
        raw = xml.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")   # the export is UTF-8 whatever its header says
        match = re.search(r"<URI>\\?([^<]+)</URI>", text)
        if match:
            found[match.group(1)] = xml
    return found


def _task_state(name: str) -> str:
    """Ready / Running / Disabled ..., or "" when the task does not exist."""
    r = _powershell(f"(Get-ScheduledTask -TaskName '{name}' -ErrorAction Stop).State")
    return r.stdout.strip() if r.returncode == 0 else ""


def _current_pythonw() -> Path | None:
    """The pythonw.exe next to the Python that is running this (what the scheduled tasks use, so no console window flashes)."""
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return candidate if candidate.exists() else None


def localize_task_xml(text: str, project_root: Path, pythonw: Path | None) -> str:
    """The saved task XML with its paths made current: the working directory and the script in the Arguments follow the project
    folder wherever it is now, and the Command follows the Python that is installed now. A task saved when the project lived
    elsewhere (or under another Python) is therefore registered correctly instead of pointing at nothing."""
    from xml.sax.saxutils import escape
    root = escape(str(project_root))
    text = re.sub(r"<WorkingDirectory>[^<]*</WorkingDirectory>", lambda m: f"<WorkingDirectory>{root}</WorkingDirectory>", text)
    text = re.sub(r'<Arguments>"[^"<]*[\\/]scripts[\\/]([^"\\/<]+)"</Arguments>',
                  lambda m: f'<Arguments>"{root}\\scripts\\{m.group(1)}"</Arguments>', text)
    if pythonw is not None:
        text = re.sub(r"<Command>[^<]*pythonw\.exe</Command>", lambda m: f"<Command>{escape(str(pythonw))}</Command>", text)
    return text


def _register_task(name: str, xml: Path) -> None:
    """Registers (or replaces) the task from its saved XML with the paths of today (see localize_task_xml). The XML reaches
    PowerShell through a scratch file inside the project that is removed right after."""
    raw = xml.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    scratch = PROJECT_ROOT / "data" / "_task_register.xml"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(localize_task_xml(text, PROJECT_ROOT, _current_pythonw()), encoding="utf-8")
    try:
        _powershell(f"Register-ScheduledTask -TaskName '{name}' -Xml (Get-Content -Raw -LiteralPath '{scratch}') -Force | Out-Null")
    finally:
        scratch.unlink(missing_ok=True)


def _task_action_is_broken(name: str) -> bool:
    """True when the task's program or script no longer exists on disk (the project folder or Python was moved or upgraded). A task
    whose action cannot be read counts as fine: only a proven broken path triggers a new registration."""
    r = _powershell(f"$a = (Get-ScheduledTask -TaskName '{name}' -ErrorAction Stop).Actions[0]; $a.Execute; $a.Arguments")
    if r.returncode != 0:
        return False
    lines = r.stdout.splitlines()
    program = lines[0].strip().strip('"') if lines else ""
    arguments = lines[1] if len(lines) > 1 else ""
    targets = ([program] if program else []) + re.findall(r'"([^"]+)"', arguments)
    return any(Path(t).is_absolute() and "%" not in t and not Path(t).exists() for t in targets)


def ensure_tasks() -> list[Issue]:
    """Every task of windows_config/scheduled_tasks exists, points at files that exist, and is enabled; re-registered (with the
    current project and Python paths) or enabled when not."""
    issues = []
    for name, xml in task_files().items():
        state = _task_state(name)
        if state == "":
            file_log.warning(f"scheduled task '{name}' is missing: registering it from {xml.name}")
            _register_task(name, xml)
            state = _task_state(name)
        elif _task_action_is_broken(name):
            file_log.warning(f"scheduled task '{name}' points at a file that no longer exists (the project or Python moved): "
                             f"registering it again from {xml.name} with the current paths")
            _register_task(name, xml)
            state = _task_state(name)
        if state == "Disabled":
            file_log.warning(f"scheduled task '{name}' is disabled: enabling it")
            _powershell(f"Enable-ScheduledTask -TaskName '{name}' | Out-Null")
            state = _task_state(name)
        if state in ("", "Disabled"):
            issues.append(Issue(f"task:{name}", f"Scheduled task '{name}' is not running and could not be repaired", tunables.get("alerts", "persist_minutes")))
    return issues


def check_database() -> list[Issue]:
    """A failed quick_check is damage and is told at once. A database that cannot be opened for the check (a writer holds it for a
    moment: the journal mode is the default one, where a writer blocks readers) proves nothing, so it is told only when it lasts
    (audit 2026-11-22: it used to raise a false alarm at once)."""
    try:
        with sqlite3.connect(_DB_PATH.as_uri() + "?mode=ro", uri=True, timeout=30) as con:
            result = con.execute("PRAGMA quick_check").fetchone()[0]
        if result == "ok":
            return []
    except sqlite3.OperationalError as e:                  # locked, or the file cannot be opened right now
        file_log.warning(f"database check could not run: {type(e).__name__}: {e}")
        return [Issue("db_unreadable", "The database cannot be opened for checking (for the last 15 minutes)", 15)]
    except Exception as e:  # noqa: BLE001 -- e.g. "file is not a database": that is damage
        file_log.warning(f"database check failed: {type(e).__name__}: {e}")
    return [Issue("db_integrity", "The database has a problem (the integrity check failed)")]


def check_disk() -> list[Issue]:
    import shutil
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    if free_gb < tunables.get("alerts", "min_free_gb"):
        return [Issue("disk_low", "Free disk space is almost gone")]
    return []


def check_models() -> list[Issue]:
    """Every model of the chain failing in a row (a quota limit never counts; 'models.failures_to_count_as_down' failures without an
    answer in between) for long enough. Only a message: it never changes the order of the chain."""
    from . import modelchain
    health = modelchain._read(modelchain._HEALTH_PATH).get("models", {})
    chain = modelchain.chain()
    need = int(tunables.get("models", "failures_to_count_as_down"))
    if chain and all(int((health.get(m) or {}).get("failures", 0) or 0) >= need for m in chain):
        return [Issue("models_down", "All Gemini models are failing (not because of quota)", tunables.get("alerts", "models_down_hours") * 60)]
    return []


# ---------------------------------------------------------------- credentials the provider has rejected (maintenance 2026-11-25)
# Only a CLEAR rejection counts (Telegram 401/404 for the token; Gemini 400/401/403 with the provider's own key wording). No answer at all
# (network down, timeout, 5xx) proves nothing and is never an issue. Nothing secret is ever logged or put in a message.
_KEY_REJECTED_WORDS = re.compile(r"API key|API_KEY|PERMISSION_DENIED|leaked|expired", re.IGNORECASE)


def telegram_token_rejected(status: int) -> bool:
    return status in (401, 404)


def gemini_key_rejected(status: int, body: str) -> bool:
    return status in (400, 401, 403) and bool(_KEY_REJECTED_WORDS.search(body or ""))


def check_credentials() -> list[Issue]:
    import requests
    issues: list[Issue] = []
    wait = tunables.get("alerts", "persist_minutes")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if token:
        try:
            if telegram_token_rejected(requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=15).status_code):
                issues.append(Issue("telegram_token_dead", "The Telegram bot token is rejected (revoked or changed); messages will not be delivered", wait))
        except requests.RequestException:
            pass                                              # no answer proves nothing (and the error text would carry the token)
    for label, name in (("main", "GEMINI_API_KEY"), ("backup", "GEMINI_API_KEY_BACKUP")):
        key = os.environ.get(name, "")
        if not key:
            continue
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", headers={"x-goog-api-key": key}, timeout=15)
            if gemini_key_rejected(r.status_code, r.text):
                issues.append(Issue(f"gemini_key_dead:{name}", f"The {label} Gemini key was rejected by Google (revoked or leaked)", wait))
        except requests.RequestException:
            pass
    return issues


_TOAST_SCRIPT = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$t = $xml.GetElementsByTagName('text')
$t.Item(0).AppendChild($xml.CreateTextNode('Life Compass')) | Out-Null
$t.Item(1).AppendChild($xml.CreateTextNode($env:LC_TOAST_TEXT)) | Out-Null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""


def show_toast(text: str) -> bool:
    """A quiet Windows notification (no Telegram): the only way to reach the user when the Telegram token itself is dead."""
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _TOAST_SCRIPT], capture_output=True, text=True,
                           timeout=30, creationflags=_NO_WINDOW, env={**os.environ, "LC_TOAST_TEXT": text})
        return r.returncode == 0
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"toast could not be shown: {type(e).__name__}")
        return False


def telegram_failures_last_day(now: float | None = None) -> int:
    """How many Telegram sends gave up (every attempt failed) in the last 24 hours, counted from the master log. Only for the log:
    never an alert (no notification headaches, and Telegram is the very channel that is failing)."""
    from .logging_setup import ARCHIVE_DIR, MASTER_LOG_PATH
    now = time.time() if now is None else now
    paths = [MASTER_LOG_PATH]
    try:                                    # a log rotated within the last day still holds part of the 24 hours
        recent = [p for p in ARCHIVE_DIR.glob(f"{MASTER_LOG_PATH.stem}_*{MASTER_LOG_PATH.suffix}") if now - p.stat().st_mtime < 24 * 3600]
        paths = sorted(recent) + paths
    except OSError:
        pass
    count = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "| ERROR" not in line or not ("Telegram send failed after" in line or "failed after retries" in line):
                        continue
                    try:
                        stamp = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp()
                    except ValueError:
                        continue
                    if now - 24 * 3600 <= stamp <= now + 60:
                        count += 1
        except OSError:
            continue
    return count


_SECRET_COPY = re.compile(r"^\.env($|\.bak)")      # .env and .env.bak*, not .env.example


def _env_copy_is_locked(path: Path, user: str) -> bool:
    """True when only the user, SYSTEM and Administrators are on the file's ACL and nothing is inherited from the folder."""
    r = subprocess.run(["icacls", str(path)], capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        return True                                   # cannot tell: leave it alone rather than touch it blindly
    out = r.stdout.replace(str(path), "")
    if "(I)" in out:
        return False
    names = {m.group(1).split("\\")[-1].lower() for m in re.finditer(r"(\S+):\(", out)}
    return names <= {user.lower(), "system", "administrators"}


def lock_env_copies(backups: Path | None = None) -> list[str]:
    """Copies of .env that a backup made inherit the folder's wider permissions (audit 2026-11-21: another group could read old keys
    and the Telegram token). Locks each such copy like the live .env. Returns what it fixed; never deletes or moves anything."""
    root = backups or Path(__file__).resolve().parents[2] / "backups"
    user = os.environ.get("USERNAME") or ""
    if not user:
        return []
    fixed = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not _SECRET_COPY.match(name):
                continue
            path = Path(dirpath) / name
            if _env_copy_is_locked(path, user):
                continue
            r = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(F)", "SYSTEM:(F)", "Administrators:(F)"],
                               capture_output=True, text=True, timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode == 0:
                fixed.append(str(path.relative_to(root)))
    return fixed


def collect_issues() -> list[Issue]:
    issues: list[Issue] = []
    for check in (check_listener, ensure_tasks, check_database, check_disk, check_models, check_credentials):
        try:
            issues += check()
        except Exception as e:  # noqa: BLE001 -- one broken check must not hide the others
            file_log.warning(f"{check.__name__} could not run: {type(e).__name__}: {e}")
    return issues


# ---------------------------------------------------------------- the run
def logging_setup_module():
    from . import logging_setup
    return logging_setup


def _read_state() -> dict:
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def run(send: Callable[[str], bool] | None = None, now: float | None = None) -> dict:
    """One health run. Returns the state (what the user was told about and what is being watched)."""
    now = time.time() if now is None else now
    issues = collect_issues()
    if send is None:
        from .publisher import send_message
        token_dead = any(i.key == "telegram_token_dead" for i in issues)

        def send(text: str) -> bool:                              # Telegram first; a token Telegram itself rejected can only be told by a toast
            return bool(send_message(text, log=lambda *_: None)) or (token_dead and show_toast(text))
    state = apply_policy(_read_state(), issues, now, send)
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        from . import ytdlp_update
        ytdlp_update.maybe_update()
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"yt-dlp maintenance did not run: {type(e).__name__}: {e}")
    try:
        gave_up = telegram_failures_last_day(now)
        if gave_up:
            file_log.warning(f"telegram: {gave_up} send(s) gave up in the last 24 hours (the network was down); each is retried by "
                             f"its own task, no alert")
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"telegram failure count did not run: {type(e).__name__}: {e}")
    try:
        from . import logging_setup
        archived = logging_setup.rotate_if_large()
        if archived:
            file_log.info(f"the master log was over {logging_setup.MAX_LOG_BYTES // (1024 * 1024)} MB: moved to logs\\archive\\{archived}")
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"log rotation did not run: {type(e).__name__}: {e}")
    try:
        resurface_log = logging_setup_module().LOGS_DIR / "resurface.log"
        archived = logging_setup_module().rotate_if_large(resurface_log, max_bytes=1024 * 1024)
        if archived:
            file_log.info(f"resurface.log was over 1 MB: moved to logs\\archive\\{archived}")
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"resurface log rotation did not run: {type(e).__name__}: {e}")
    try:
        fixed = lock_env_copies()
        if fixed:
            file_log.info(f"locked the permissions of {len(fixed)} .env copy(ies) in backups: {fixed}")
    except Exception as e:  # noqa: BLE001
        file_log.warning(f".env copy lock did not run: {type(e).__name__}: {e}")
    try:
        from . import snapshots
        snapshots.take(now)                                       # a daily database snapshot (only when the database changed); log only
    except Exception as e:  # noqa: BLE001 -- housekeeping must never break the health run
        file_log.warning(f"db snapshot did not run: {type(e).__name__}: {e}")
    try:
        from . import retention
        retention.sweep(now)
    except Exception as e:  # noqa: BLE001 -- housekeeping must never break the health run
        file_log.warning(f"retention did not run: {type(e).__name__}: {e}")
    file_log.info(f"health run: {len(issues)} open issue(s): {[i.key for i in issues]}")
    return state
