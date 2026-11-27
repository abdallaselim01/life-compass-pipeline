import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Tests write to logs/tests.log, never to the real logs/lifecompass.log.
os.environ["LIFECOMPASS_LOG_NAME"] = "tests.log"
# Tests never read the real config.toml: every tunable is its built-in default (a file that does not exist).
os.environ["LIFECOMPASS_CONFIG"] = str(ROOT / "tests" / "_no_config.toml")
# The test log is moved (never deleted) to logs/archive once it is over 1 MB, at the start of a session (every run adds about 260 KB).
from lifecompass import logging_setup  # noqa: E402  (after the two settings above, which it reads)
logging_setup.rotate_if_large(logging_setup.MASTER_LOG_PATH, max_bytes=1024 * 1024)


@pytest.fixture(autouse=True)
def _isolated_call_timings(tmp_path, monkeypatch):
    """The recorded call times of the tests never touch (or read) the real data/call_timings.json."""
    from lifecompass import llm
    monkeypatch.setattr(llm, "_TIMINGS_PATH", tmp_path / "call_timings.json")
    yield


@pytest.fixture(autouse=True)
def _isolated_model_chain(tmp_path, monkeypatch):
    """The model list, its cache and the models' health never touch the real data/ files, and Google is never asked."""
    from lifecompass import modelchain
    monkeypatch.setattr(modelchain, "_CACHE_PATH", tmp_path / "models_cache.json")
    monkeypatch.setattr(modelchain, "_HEALTH_PATH", tmp_path / "model_health.json")
    monkeypatch.setattr(modelchain, "_list_remote", lambda: [])
    yield


@pytest.fixture(autouse=True)
def _isolated_log_archive(tmp_path, monkeypatch):
    """A test that rotates the log (the health run does) moves it into a temporary archive, never into the real logs/archive."""
    from lifecompass import logging_setup
    monkeypatch.setattr(logging_setup, "ARCHIVE_DIR", tmp_path / "log_archive")
    yield


@pytest.fixture(autouse=True)
def _isolated_maintenance(tmp_path, monkeypatch):
    """Health and yt-dlp state never touch the real data/ files, and a test never reaches PyPI or installs anything."""
    from lifecompass import health, ytdlp_update
    monkeypatch.setattr(health, "_STATE_PATH", tmp_path / "health_state.json")
    monkeypatch.setattr(ytdlp_update, "_STATE_PATH", tmp_path / "ytdlp_state.json")
    monkeypatch.setattr(ytdlp_update, "update_after_failure", lambda: False)
    monkeypatch.setattr(ytdlp_update, "maybe_update", lambda: "")
    yield


@pytest.fixture(autouse=True)
def _isolated_housekeeping(tmp_path, monkeypatch):
    """The daily DB snapshot and the retention sweep of a health run work in a temporary folder, never on the real data/ and backups/; the
    credential check never reaches Telegram or Google; the yt-dlp info memory starts empty in every test."""
    from lifecompass import audio, health, retention, snapshots
    monkeypatch.setattr(snapshots, "_DB_PATH", tmp_path / "hk" / "principles.db")
    monkeypatch.setattr(snapshots, "_SNAP_DIR", tmp_path / "hk" / "backups" / "db_daily")
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path / "hk")
    monkeypatch.setattr(health, "check_credentials", lambda: [])
    audio._info_cache.clear()
    yield
    audio._info_cache.clear()


@pytest.fixture(autouse=True)
def _restore_real_model_backend():
    """Every test starts and ends with the real model backend restored."""
    from lifecompass import llm
    llm.set_test_backend()
    yield
    llm.set_test_backend()
