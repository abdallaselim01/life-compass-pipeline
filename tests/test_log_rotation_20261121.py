"""Audit 2026-11-21, P3: the shared master log survives being rotated while several processes write to it."""
import os
import subprocess
import sys
import time
from pathlib import Path

from lifecompass import logging_setup, retention
from lifecompass.logging_setup import SharedFileHandler, rotate_if_large

ROOT = Path(__file__).resolve().parents[1]


def _record(msg):
    import logging
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def test_handler_appends_and_never_holds_the_file_open(tmp_path):
    path = tmp_path / "x.log"
    h = SharedFileHandler(path)
    h.setFormatter(__import__("logging").Formatter("%(message)s"))
    h.emit(_record("one"))
    os.replace(path, tmp_path / "moved.log")               # would fail on Windows if the handler kept the file open
    h.emit(_record("two"))
    assert (tmp_path / "moved.log").read_text(encoding="utf-8") == "one\n"
    assert path.read_text(encoding="utf-8") == "two\n"


def test_rotate_moves_a_large_log_and_never_deletes(tmp_path):
    path, archive = tmp_path / "lifecompass.log", tmp_path / "archive"
    path.write_text("x" * 100, encoding="utf-8")
    assert rotate_if_large(path, max_bytes=1000, archive_dir=archive) is None and path.exists()
    name = rotate_if_large(path, max_bytes=50, archive_dir=archive)
    assert name and not path.exists()
    assert (archive / name).read_text(encoding="utf-8") == "x" * 100


def test_rotate_leaves_a_file_another_process_holds_open(tmp_path):
    path = tmp_path / "lifecompass.log"
    path.write_text("x" * 100, encoding="utf-8")
    with open(path, "a"):                                   # another process has the file open: the move is refused, nothing breaks
        assert rotate_if_large(path, max_bytes=10, archive_dir=tmp_path / "a") is None
    assert path.exists()


def test_three_processes_write_while_the_log_is_rotated(tmp_path):
    path, archive = tmp_path / "lifecompass.log", tmp_path / "archive"
    code = (
        "import logging,sys;from lifecompass.logging_setup import SharedFileHandler\n"
        "log=logging.getLogger('w');log.setLevel(10);h=SharedFileHandler(sys.argv[1]);h.setFormatter(logging.Formatter('%(message)s'));log.addHandler(h)\n"
        "[log.info(f'{sys.argv[2]}-{i} ' + 'y'*200) for i in range(300)]\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), LIFECOMPASS_LOG_NAME="tests.log")
    procs = [subprocess.Popen([sys.executable, "-B", "-c", code, str(path), name], env=env) for name in "abc"]
    rotations = 0
    while any(p.poll() is None for p in procs):
        if rotate_if_large(path, max_bytes=20_000, archive_dir=archive):
            rotations += 1
        time.sleep(0.05)
    assert [p.returncode for p in procs] == [0, 0, 0]
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in list(archive.glob("*.log")) + [path] if f.exists())
    assert rotations >= 1
    assert lines == 900                                     # the cross-process lock: not one line is lost to a rotation or to another writer


def test_retention_moves_old_archived_logs(tmp_path):
    logs, backups = tmp_path / "logs", tmp_path / "backups"
    (logs / "archive").mkdir(parents=True)
    backups.mkdir()
    old, fresh = logs / "archive" / "lifecompass_20260101_000000.log", logs / "archive" / "lifecompass_20261120_000000.log"
    old.write_text("o"), fresh.write_text("f")
    long_ago = time.time() - 90 * 86400
    os.utime(old, (long_ago, long_ago))
    moved = retention.sweep(time.time(), backups_dir=backups, logs_dir=logs)
    assert moved == ["logs\\archive\\lifecompass_20260101_000000.log"]
    assert fresh.exists() and not old.exists()
    assert any(backups.glob("retired_*/run_logs/lifecompass_20260101_000000.log"))
