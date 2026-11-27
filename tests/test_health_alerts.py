"""Part D (design decision 2026-11-18): self-maintenance with quiet alerts, and yt-dlp that keeps itself up to date."""
import json
import sqlite3
import time
from pathlib import Path

import pytest

from lifecompass import health, modelchain, tunables, ytdlp_update
from lifecompass.health import Issue, apply_policy

HOUR = 3600.0
_REAL_UPDATE_FUNCTIONS = (ytdlp_update.maybe_update, ytdlp_update.update_after_failure)   # the conftest replaces them with do-nothings
DAY0 = time.mktime((2026, 11, 18, 12, 0, 0, 0, 0, -1))


class Phone:
    """A Telegram that records what it was sent, and can be unreachable."""
    def __init__(self):
        self.messages, self.up = [], True

    def __call__(self, text):
        if self.up:
            self.messages.append(text)
        return self.up


def test_nothing_wrong_means_nothing_is_ever_sent():
    phone, state = Phone(), {}
    for hour in range(48):
        apply_policy(state, [], DAY0 + hour * HOUR, phone)
    assert phone.messages == []


def test_a_problem_is_told_only_after_it_lasted_as_long_as_its_patience_and_then_only_once():
    phone, state = Phone(), {}
    listener = [Issue("listener_down", "السيستم واقف", 60)]
    apply_policy(state, listener, DAY0, phone)
    assert phone.messages == []                                                          # just seen: maybe the watchdog fixes it
    apply_policy(state, listener, DAY0 + 30 * 60, phone)
    assert phone.messages == []
    apply_policy(state, listener, DAY0 + 61 * 60, phone)
    assert len(phone.messages) == 1 and "السيستم واقف" in phone.messages[0]
    for hour in range(2, 30):
        apply_policy(state, listener, DAY0 + hour * HOUR, phone)
    assert len(phone.messages) == 1                                                      # never repeated while it stays broken


def test_a_fixed_problem_is_forgotten_silently_and_told_again_only_if_it_comes_back_and_lasts():
    phone, state = Phone(), {}
    listener = [Issue("listener_down", "السيستم واقف", 60)]
    apply_policy(state, listener, DAY0, phone)
    apply_policy(state, listener, DAY0 + 2 * HOUR, phone)
    assert len(phone.messages) == 1
    apply_policy(state, [], DAY0 + 3 * HOUR, phone)                                       # fixed: no "all is well" message
    assert len(phone.messages) == 1 and state["issues"] == {}
    apply_policy(state, listener, DAY0 + 30 * HOUR, phone)                                # back, patience starts again
    assert len(phone.messages) == 1
    apply_policy(state, listener, DAY0 + 32 * HOUR, phone)
    assert len(phone.messages) == 2


def test_problems_that_are_ready_together_go_in_one_message():
    phone, state = Phone(), {}
    apply_policy(state, [Issue("disk_low", "المساحة قربت تخلص"), Issue("db_integrity", "الداتا بيز فيها مشكلة")], DAY0, phone)
    assert len(phone.messages) == 1
    assert "المساحة قربت تخلص" in phone.messages[0] and "الداتا بيز فيها مشكلة" in phone.messages[0]


def test_at_most_two_messages_a_day_and_the_rest_wait_for_tomorrow():
    phone, state = Phone(), {}
    for i, key in enumerate(("a", "b", "c")):
        apply_policy(state, [Issue(k, f"مشكلة {k}") for k in ("a", "b", "c")[: i + 1]], DAY0 + i * HOUR, phone)
    assert len(phone.messages) == 2                                                      # c is held back today
    apply_policy(state, [Issue(k, f"مشكلة {k}") for k in ("a", "b", "c")], DAY0 + 24 * HOUR, phone)
    assert len(phone.messages) == 3 and "مشكلة c" in phone.messages[2] and "مشكلة a" not in phone.messages[2]


def test_when_telegram_is_unreachable_the_problem_stays_untold_and_is_retried():
    phone, state = Phone(), {}
    phone.up = False
    apply_policy(state, [Issue("disk_low", "المساحة قربت تخلص")], DAY0, phone)
    assert state["issues"]["disk_low"]["alerted_at"] is None and state["alerts"]["count"] == 0
    phone.up = True
    apply_policy(state, [Issue("disk_low", "المساحة قربت تخلص")], DAY0 + HOUR, phone)
    assert len(phone.messages) == 1


def test_there_are_no_quiet_hours_in_the_alert_rules():
    phone, state = Phone(), {}
    three_am = time.mktime((2026, 11, 18, 3, 0, 0, 0, 0, -1))
    apply_policy(state, [Issue("db_integrity", "الداتا بيز فيها مشكلة")], three_am, phone)
    assert len(phone.messages) == 1


def test_the_database_check(tmp_path, monkeypatch):
    db = tmp_path / "principles.db"
    sqlite3.connect(db).execute("CREATE TABLE t (x)").connection.close()
    monkeypatch.setattr(health, "_DB_PATH", db)
    assert health.check_database() == []
    db.write_bytes(b"this is not a database" * 200)
    assert [i.key for i in health.check_database()] == ["db_integrity"]
    # a file that cannot be opened right now (a writer holds it, or it is missing) is told only when it lasts (audit 2026-11-22)
    monkeypatch.setattr(health, "_DB_PATH", tmp_path / "missing.db")
    issues = health.check_database()
    assert [i.key for i in issues] == ["db_unreadable"] and issues[0].after_minutes == 15


def test_the_disk_check(monkeypatch):
    import shutil
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100e9, 97e9, 3e9))
    assert [i.key for i in health.check_disk()] == ["disk_low"]
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(100e9, 50e9, 50e9))
    assert health.check_disk() == []


def test_the_models_check_speaks_only_when_every_model_keeps_failing(tmp_path, monkeypatch):
    monkeypatch.setattr(modelchain, "_HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(modelchain, "chain", lambda: ["m1", "m2"])
    modelchain._HEALTH_PATH.write_text(json.dumps({"models": {"m1": {"failures": 3}}}), encoding="utf-8")
    assert health.check_models() == []                                                    # one model still works
    modelchain._HEALTH_PATH.write_text(json.dumps({"models": {"m1": {"failures": 3}, "m2": {"failures": 1}}}), encoding="utf-8")
    assert health.check_models() == []                                                    # one failure is not a pattern
    modelchain._HEALTH_PATH.write_text(json.dumps({"models": {"m1": {"failures": 3}, "m2": {"failures": 2}}}), encoding="utf-8")
    issues = health.check_models()
    assert [i.key for i in issues] == ["models_down"] and issues[0].after_minutes == 12 * 60   # patience: half a day


def test_one_broken_check_does_not_hide_the_others(monkeypatch):
    def boom():
        raise RuntimeError("x")
    monkeypatch.setattr(health, "check_listener", boom)
    monkeypatch.setattr(health, "ensure_tasks", lambda: [])
    monkeypatch.setattr(health, "check_database", lambda: [Issue("db_integrity", "d")])
    monkeypatch.setattr(health, "check_disk", lambda: [])
    monkeypatch.setattr(health, "check_models", lambda: [])
    assert [i.key for i in health.collect_issues()] == ["db_integrity"]


def test_a_run_saves_its_state_and_sends_through_the_given_phone(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(health, "collect_issues", lambda: [Issue("disk_low", "المساحة قربت تخلص")])
    monkeypatch.setattr(ytdlp_update, "maybe_update", lambda: "")
    phone = Phone()
    health.run(send=phone, now=DAY0)
    assert len(phone.messages) == 1
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["issues"]["disk_low"]["alerted_at"] == DAY0
    health.run(send=phone, now=DAY0 + HOUR)
    assert len(phone.messages) == 1


def test_every_saved_task_copy_has_a_name_and_the_health_task_is_one_of_them():
    names = set(health.task_files())
    assert {"LifeCompass Listener Watchdog", "LifeCompass Random Resurface", "LifeCompass Retry Pending",
            "LifeCompass Health Check"} <= names


# ------------------------------------------------------------------ yt-dlp keeps itself up to date
@pytest.fixture
def yt(tmp_path, monkeypatch):
    """ytdlp_update on scratch folders, with a fake pip, a fake PyPI and smoke tests the test decides."""
    monkeypatch.setattr(ytdlp_update, "VENDOR", tmp_path / "vendor")
    monkeypatch.setattr(ytdlp_update, "_STAGING", tmp_path / "vendor.staging")
    monkeypatch.setattr(ytdlp_update, "_STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(ytdlp_update, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ytdlp_update, "maybe_update", _REAL_UPDATE_FUNCTIONS[0])
    monkeypatch.setattr(ytdlp_update, "update_after_failure", _REAL_UPDATE_FUNCTIONS[1])
    box = {"latest": "2026.12.1", "current": "2026.10.26", "new_ok": True, "old_ok": True, "pip_ok": True}
    monkeypatch.setattr(ytdlp_update, "latest_version", lambda: box["latest"])
    monkeypatch.setattr(ytdlp_update, "current_version", lambda vendor=None: box["current"])
    monkeypatch.setattr(ytdlp_update, "smoke_test", lambda vendor=None: box["new_ok"] if vendor else box["old_ok"])

    class Done:
        def __init__(self, ok):
            self.returncode, self.stdout, self.stderr = (0 if ok else 1), "", "pip said no"

    def fake_pip(cmd, **kw):
        if box["pip_ok"]:
            staging = Path(cmd[cmd.index("--target") + 1])
            staging.mkdir(parents=True)
            (staging / "NEW.txt").write_text(box["latest"], encoding="utf-8")
        return Done(box["pip_ok"])
    monkeypatch.setattr(ytdlp_update.subprocess, "run", fake_pip)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "OLD.txt").write_text(box["current"], encoding="utf-8")
    return box


def test_version_keys_compare_dates_number_by_number():
    assert ytdlp_update.version_key("2026.10.26") > ytdlp_update.version_key("2026.8.19") > ytdlp_update.version_key("2026.7.4")
    assert ytdlp_update.version_key("2026.12.1") > ytdlp_update.version_key("2026.10.26")


def test_a_newer_yt_dlp_that_passes_replaces_the_old_one_and_the_old_one_is_kept_aside(yt, tmp_path):
    note = ytdlp_update.update()
    assert "2026.10.26 -> 2026.12.1" in note
    assert (tmp_path / "vendor" / "NEW.txt").exists() and not (tmp_path / "vendor.staging").exists()
    aside = list((tmp_path / "backups").glob("ytdlp_vendor_2026.10.26_*"))
    assert len(aside) == 1 and (aside[0] / "OLD.txt").exists()                           # never deleted, only moved aside


def test_a_newer_yt_dlp_that_fails_its_smoke_test_changes_nothing(yt, tmp_path):
    yt["new_ok"] = False
    assert ytdlp_update.update() == ""
    assert (tmp_path / "vendor" / "OLD.txt").exists() and not (tmp_path / "vendor.staging").exists()
    assert not (tmp_path / "backups").exists()


def test_when_the_old_one_can_no_longer_read_videos_and_the_new_one_can_the_note_says_so(yt):
    yt["old_ok"] = False
    assert "could no longer read the video page" in ytdlp_update.update()


def test_nothing_happens_when_already_the_newest_or_when_pip_fails(yt, tmp_path):
    yt["latest"] = "2026.10.26"
    assert ytdlp_update.update() == ""
    yt["latest"], yt["pip_ok"] = "2026.12.1", False
    assert ytdlp_update.update() == ""
    assert (tmp_path / "vendor" / "OLD.txt").exists()


def test_the_weekly_check_waits_a_week_and_the_failure_retry_waits_hours(yt):
    assert ytdlp_update.maybe_update() != ""                                              # first time: checks
    yt["latest"] = "2027.1.1"
    assert ytdlp_update.maybe_update() == ""                                              # asked again at once: not yet
    state = json.loads(ytdlp_update._STATE_PATH.read_text(encoding="utf-8"))
    state["last_check"] -= 8 * 86400
    ytdlp_update._STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    assert ytdlp_update.maybe_update() != ""                                              # a week later: checks again
    yt["latest"] = "2027.2.1"
    assert ytdlp_update.update_after_failure() is True                                    # a failed command: tries at once
    yt["latest"] = "2027.3.1"
    assert ytdlp_update.update_after_failure() is False                                   # but not again within hours


def test_the_environment_of_a_yt_dlp_process_puts_vendor_first(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    monkeypatch.setattr(ytdlp_update, "VENDOR", vendor)
    monkeypatch.setenv("PYTHONPATH", "elsewhere")
    assert "PYTHONPATH" in ytdlp_update.env() and ytdlp_update.env()["PYTHONPATH"] == "elsewhere"   # no vendor folder: untouched
    vendor.mkdir()
    assert ytdlp_update.env()["PYTHONPATH"].startswith(str(vendor))
    assert ytdlp_update.env()["PYTHONPATH"].endswith("elsewhere")


def test_audio_runs_yt_dlp_with_vendor_and_retries_once_after_a_successful_update(monkeypatch):
    from lifecompass import audio
    calls = []

    class Result:
        def __init__(self, code):
            self.returncode, self.stdout, self.stderr = code, "", ""

    def fake_run(cmd, **kw):
        calls.append(kw["env"].get("PYTHONPATH"))
        return Result(1 if len(calls) == 1 else 0)
    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(ytdlp_update, "update_after_failure", lambda: True)
    r = audio._run([audio.sys.executable, "-m", "yt_dlp", "--version"])
    assert r.returncode == 0 and len(calls) == 2                                          # failed, updated, ran once more
    calls.clear()
    monkeypatch.setattr(audio.subprocess, "run", lambda cmd, **kw: (calls.append(1), Result(1))[1])
    monkeypatch.setattr(ytdlp_update, "update_after_failure", lambda: False)
    assert audio._run([audio.sys.executable, "-m", "yt_dlp", "--version"]).returncode == 1 and len(calls) == 1
    calls.clear()
    assert audio._run(["ffmpeg", "-i", "x"]).returncode == 1 and len(calls) == 1          # not a yt-dlp command: no update, no retry
