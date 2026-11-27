"""Maintenance 2026-11-25: the daily database snapshot, the rejected-credential check with its toast, and the single yt-dlp info call."""
import json
import os
import sqlite3
import sys
import time
from types import SimpleNamespace

import pytest
import requests

from lifecompass import audio, health, snapshots
from lifecompass.health import Issue
from lifecompass.health import check_credentials as real_check_credentials   # captured before the conftest stub replaces the attribute

DAY = 86400.0
T0 = time.mktime((2026, 11, 25, 12, 0, 0, 0, 0, -1))


def _make_db(path, rows=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS principles (id INTEGER PRIMARY KEY, text TEXT)")
    for i in range(rows):
        conn.execute("INSERT INTO principles (text) VALUES (?)", (f"مبدأ {i}",))
    conn.commit()
    conn.close()


def _touch_forward(path, seconds=5):
    t = path.stat().st_mtime + seconds
    os.utime(path, (t, t))


# ---- the daily snapshot -------------------------------------------------------------------------------------------
def test_first_snapshot_is_a_checked_copy_with_a_manifest(tmp_path):
    db, root = tmp_path / "data" / "principles.db", tmp_path / "backups" / "db_daily"
    _make_db(db, rows=3)
    made = snapshots.take(T0, db, root)
    assert made is not None and made.parent == root
    copy = sqlite3.connect(made / "principles.db")
    assert copy.execute("SELECT COUNT(*) FROM principles").fetchone()[0] == 3
    copy.close()
    manifest = json.loads((made / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == {"principles": 3} and manifest["source_mtime_ns"] == db.stat().st_mtime_ns
    assert snapshots.list_snapshots(root) == [made]


def test_one_a_day_and_none_when_nothing_changed(tmp_path):
    db, root = tmp_path / "principles.db", tmp_path / "backups" / "db_daily"
    _make_db(db)
    assert snapshots.take(T0, db, root) is not None
    assert snapshots.take(T0 + 3600, db, root) is None                 # same day
    assert snapshots.take(T0 + DAY, db, root) is None                  # next day, database unchanged
    _make_db(db)                                                       # one more row
    _touch_forward(db)
    assert snapshots.take(T0 + DAY, db, root) is not None              # next day, database changed
    assert len(snapshots.list_snapshots(root)) == 2


def test_older_snapshots_are_moved_aside_never_deleted(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text("[snapshots]\nkeep = 2\n", encoding="utf-8")
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    db, root = tmp_path / "principles.db", tmp_path / "backups" / "db_daily"
    _make_db(db)
    for day in range(3):
        _make_db(db)
        _touch_forward(db, 10 * (day + 1))
        assert snapshots.take(T0 + day * DAY, db, root) is not None
    kept = snapshots.list_snapshots(root)
    assert len(kept) == 2
    retired = tmp_path / "backups" / f"retired_{time.strftime('%Y%m%d', time.localtime(T0 + 2 * DAY))}" / "db_daily"
    moved = list(retired.iterdir())
    assert len(moved) == 1 and (moved[0] / "principles.db").exists()   # the oldest is intact, only moved


def test_a_bad_copy_never_counts_and_is_kept_aside(tmp_path, monkeypatch):
    db, root = tmp_path / "principles.db", tmp_path / "backups" / "db_daily"
    _make_db(db)

    def broken(src, dest):
        dest.write_bytes(b"half")
        raise RuntimeError("the copy failed SQLite's quick_check")

    monkeypatch.setattr(snapshots, "_copy_and_check", broken)
    assert snapshots.take(T0, db, root) is None
    assert snapshots.list_snapshots(root) == []
    assert any((root / "_failed").iterdir())                           # kept, not deleted


def test_no_database_no_snapshot(tmp_path):
    assert snapshots.take(T0, tmp_path / "missing.db", tmp_path / "backups" / "db_daily") is None


def test_the_snapshot_folder_is_protected_from_retention():
    from lifecompass import retention
    assert retention.is_protected("db_daily")


# ---- rejected credentials -----------------------------------------------------------------------------------------
def test_only_a_clear_rejection_counts():
    assert health.telegram_token_rejected(401) and health.telegram_token_rejected(404)
    assert not health.telegram_token_rejected(200) and not health.telegram_token_rejected(502)
    assert health.gemini_key_rejected(400, '{"error": {"message": "API key not valid. Please pass a valid API key."}}')
    assert health.gemini_key_rejected(403, "PERMISSION_DENIED: your API key was reported as leaked")
    assert not health.gemini_key_rejected(400, '{"error": {"message": "Invalid value for pageSize"}}')
    assert not health.gemini_key_rejected(429, "quota exceeded for this API key")
    assert not health.gemini_key_rejected(503, "API key backend unavailable")


def test_check_credentials_reports_rejections_and_ignores_silence(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP", "key-two")

    def fake_get(url, headers=None, timeout=None):
        if "telegram" in url:
            return SimpleNamespace(status_code=401, text="")
        if headers["x-goog-api-key"] == "key-one":
            return SimpleNamespace(status_code=400, text="API key not valid")
        raise requests.ConnectionError("network down: " + url)          # the backup key: no answer

    monkeypatch.setattr(requests, "get", fake_get)
    assert [i.key for i in real_check_credentials()] == ["telegram_token_dead", "gemini_key_dead:GEMINI_API_KEY"]


def test_a_working_setup_has_no_credential_issue(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:SECRET")
    monkeypatch.setenv("GEMINI_API_KEY", "key-one")
    monkeypatch.delenv("GEMINI_API_KEY_BACKUP", raising=False)
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: SimpleNamespace(status_code=200, text="{}"))
    assert real_check_credentials() == []

def test_a_dead_telegram_token_is_told_once_by_a_toast(monkeypatch, tmp_path):
    from lifecompass import publisher
    monkeypatch.setattr(health, "collect_issues", lambda: [Issue("telegram_token_dead", "التوكن مرفوض")])
    monkeypatch.setattr(publisher, "send_message", lambda text, log=print: False)
    shown = []
    monkeypatch.setattr(health, "show_toast", lambda text: shown.append(text) or True)
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "state.json")
    health.run(now=T0)
    health.run(now=T0 + 3600)
    assert len(shown) == 1 and "التوكن" in shown[0]                     # told once, not again while the problem stays


def test_other_problems_never_use_the_toast(monkeypatch, tmp_path):
    from lifecompass import publisher
    monkeypatch.setattr(health, "collect_issues", lambda: [Issue("disk_low", "المساحة قربت تخلص")])
    monkeypatch.setattr(publisher, "send_message", lambda text, log=print: False)
    shown = []
    monkeypatch.setattr(health, "show_toast", lambda text: shown.append(text) or True)
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "state.json")
    health.run(now=T0)
    assert shown == []                                                  # Telegram is merely unreachable: retried later, no toast


# ---- one yt-dlp call for the title and the length -----------------------------------------------------------------
VID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VID}"


def test_the_title_and_the_length_cost_one_youtube_call(monkeypatch):
    calls = []
    monkeypatch.setattr(audio, "_run", lambda cmd, timeout=None, **kw: calls.append(cmd) or SimpleNamespace(
        returncode=0, stdout=json.dumps({"title": " عنوان ", "duration": 1200.4}), stderr=""))
    assert audio.video_info(URL) == (VID, "عنوان")
    assert audio.video_duration(VID) == 1200.4
    assert len(calls) == 1


def test_a_failed_info_call_is_not_remembered(monkeypatch):
    results = iter([SimpleNamespace(returncode=1, stdout="", stderr="network"),
                    SimpleNamespace(returncode=0, stdout=json.dumps({"title": "t", "duration": 5}), stderr="")])
    monkeypatch.setattr(audio, "_run", lambda cmd, timeout=None, **kw: next(results))
    with pytest.raises(RuntimeError):
        audio.video_info(URL)
    assert audio.video_info(URL) == (VID, "t")


def test_an_old_info_answer_is_asked_again(monkeypatch):
    calls = []
    monkeypatch.setattr(audio, "_run", lambda cmd, timeout=None, **kw: calls.append(1) or SimpleNamespace(
        returncode=0, stdout=json.dumps({"title": "t", "duration": 5}), stderr=""))
    audio.video_info(URL)
    t, info = audio._info_cache[VID]
    audio._info_cache[VID] = (t - audio._INFO_TTL_SECONDS - 1, info)
    audio.video_info(URL)
    assert len(calls) == 2
