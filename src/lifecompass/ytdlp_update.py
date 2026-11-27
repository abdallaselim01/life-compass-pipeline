"""yt-dlp keeps itself up to date (design decision 2026-11-18, part D: Fully Automated).

yt-dlp breaks whenever YouTube changes something, and the fix is always a newer yt-dlp. So the system keeps its OWN copy, in
vendor/ inside the project (nothing is installed outside the folder, no virtual environment, no change to the scheduled tasks'
interpreter): audio.py puts vendor/ first on the path of every yt-dlp it starts, so a newer copy there wins over the one installed
for Python.

Two triggers:
  * weekly (maintenance.ytdlp_check_days), from health_check.py: if PyPI has a newer version, it is installed into a staging
    folder and tried before it replaces anything;
  * when a yt-dlp command fails (audio.py asks `update_after_failure`), at most once per maintenance.ytdlp_failure_retry_hours: the
    same update, and the command runs once more with the newer copy.

A new version replaces the current one only if it passes the same smoke test the current one is held to (it reports its version and
can read the video page of the last processed video without downloading it) and the current one does not do better. The copy it
replaces is moved to backups/ytdlp_vendor_<version>_<time> (never deleted); the staging folder of a version that failed is the one
thing this module deletes, because it created it seconds earlier and nothing else ever points to it.

Quiet by design: a failed update is only written to the log; the next trigger tries again.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import tunables
from .logging_setup import get_logger

file_log = get_logger("ytdlp_update")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT_ROOT / "vendor"
_STAGING = PROJECT_ROOT / "vendor.staging"
_STATE_PATH = PROJECT_ROOT / "data" / "ytdlp_state.json"
_DB_PATH = PROJECT_ROOT / "data" / "principles.db"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def env(vendor: Path | None = None) -> dict:
    """The environment a yt-dlp process runs in: UTF-8, and vendor/ (or the given folder) first on the module path when it exists."""
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    folder = vendor if vendor is not None else VENDOR
    if folder.is_dir():
        environment["PYTHONPATH"] = str(folder) + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    return environment


def _run(args: list, vendor: Path | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          creationflags=_NO_WINDOW, env=env(vendor), timeout=timeout)


def _read_state() -> dict:
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _write_state(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def version_key(version: str) -> tuple:
    """2026.10.26 -> (2026, 10, 26): yt-dlp versions are dates, compared number by number."""
    return tuple(int(part) for part in re.findall(r"\d+", version or ""))


def current_version(vendor: Path | None = None) -> str:
    r = _run(["yt_dlp", "--version"], vendor)
    return r.stdout.strip() if r.returncode == 0 else ""


def latest_version() -> str:
    import requests
    response = requests.get("https://pypi.org/pypi/yt-dlp/json", timeout=20)
    response.raise_for_status()
    return str(response.json()["info"]["version"])


def _smoke_video_id() -> str:
    """The last processed video: a real, public video of the speaker to read the page of. Empty when there is none."""
    try:
        with sqlite3.connect(_DB_PATH.as_uri() + "?mode=ro", uri=True) as con:
            row = con.execute("SELECT video_id FROM processed_videos ORDER BY rowid DESC LIMIT 1").fetchone()
        return row[0] if row else ""
    except Exception:  # noqa: BLE001
        return ""


def smoke_test(vendor: Path | None = None) -> bool:
    """The copy starts, and (when there is a processed video) reads that video's page without downloading anything."""
    try:
        if not current_version(vendor):
            return False
        video_id = _smoke_video_id()
        if not video_id:
            return True
        r = _run(["yt_dlp", "--skip-download", "--no-warnings", "--print", "id", f"https://www.youtube.com/watch?v={video_id}"],
                 vendor, timeout=180)
        return r.returncode == 0 and r.stdout.strip() == video_id
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"yt-dlp smoke test could not run: {type(e).__name__}: {e}")
        return False


def update() -> str:
    """Install the newest yt-dlp if it is newer than the one in use and passes the smoke test. Returns a one-line note about what
    changed, or "" when nothing did (already the newest, the check could not run, or the newer one did not pass)."""
    state = _read_state()
    state["last_check"] = time.time()
    try:
        latest, current = latest_version(), current_version()
        if not current or version_key(latest) <= version_key(current):
            _write_state(state)
            return ""
        if _STAGING.exists():
            shutil.rmtree(_STAGING)                    # left over from a run that was cut off: ours, never used
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(_STAGING), "--no-deps", "--no-cache-dir",
                            "--disable-pip-version-check", f"yt-dlp=={latest}"], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", creationflags=_NO_WINDOW, timeout=300)
        if r.returncode != 0:
            file_log.warning(f"yt-dlp {latest}: pip install failed: {(r.stderr or r.stdout).strip()[-200:]}")
            _write_state(state)
            return ""
        new_ok, old_ok = smoke_test(_STAGING), smoke_test()
        if not new_ok:
            file_log.warning(f"yt-dlp {latest} did not pass the smoke test (the current {current} {'passes' if old_ok else 'fails too'}): kept {current}")
            shutil.rmtree(_STAGING, ignore_errors=True)
            _write_state(state)
            return ""
        if VENDOR.exists():
            base = PROJECT_ROOT / "backups" / f"ytdlp_vendor_{current}_{time.strftime('%Y%m%d_%H%M%S')}"
            aside, n = base, 1
            while aside.exists():                      # never into (or over) a folder that is already there
                n += 1
                aside = base.with_name(f"{base.name}_{n}")
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(VENDOR), str(aside))
        shutil.move(str(_STAGING), str(VENDOR))
        state.update(last_update=time.time(), version=latest)
        _write_state(state)
        note = f"yt-dlp {current} -> {latest}" + ("" if old_ok else " (the old one could no longer read the video page)")
        file_log.info(note)
        return note
    except Exception as e:  # noqa: BLE001 -- never raises into a download or a health check
        file_log.warning(f"yt-dlp update did not run: {type(e).__name__}: {e}")
        try:
            _write_state(state)
        except Exception:  # noqa: BLE001
            pass
        return ""


def maybe_update() -> str:
    """The weekly check (called by health_check.py): does nothing until maintenance.ytdlp_check_days have passed."""
    if time.time() - float(_read_state().get("last_check", 0) or 0) < tunables.get("maintenance", "ytdlp_check_days") * 86400:
        return ""
    return update()


def update_after_failure() -> bool:
    """A yt-dlp command failed: try to get a newer yt-dlp, at most once per maintenance.ytdlp_failure_retry_hours. True when a newer
    copy is now in place (the caller runs its command once more)."""
    state = _read_state()
    if time.time() - float(state.get("last_failure_attempt", 0) or 0) < tunables.get("maintenance", "ytdlp_failure_retry_hours") * 3600:
        return False
    state["last_failure_attempt"] = time.time()
    _write_state(state)
    return bool(update())
