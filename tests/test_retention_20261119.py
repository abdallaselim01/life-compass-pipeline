"""Automatic retention: old backups and run logs are moved (never deleted), the protected ones never move."""
import os
import time

from lifecompass import retention

DAY = 86400


def _age(path, days, now):
    t = now - days * DAY
    os.utime(path, (t, t))


def _setup(tmp_path, now):
    backups, logs = tmp_path / "backups", tmp_path / "logs"
    backups.mkdir()
    logs.mkdir()
    for name, days in [("old_patch_20260801_before", 90), ("new_patch_20261110_before", 9), ("auto", 400),
                       ("db_old_before", 300), ("restructure_20261006_before", 200), ("system_full_20261024_013225", 200),
                       ("retired_20261114", 300)]:
        (backups / name).mkdir()
        (backups / name / "f.txt").write_text("x")
        _age(backups / name, days, now)
    for name, days in [("20260801_101010.log", 90), ("20261115_101010.log", 4), ("lifecompass.log", 400), ("tests.log", 400)]:
        (logs / name).write_text("x")
        _age(logs / name, days, now)
    return backups, logs


def test_old_backup_and_old_run_log_are_moved_not_deleted(tmp_path):
    now = time.time()
    backups, logs = _setup(tmp_path, now)
    moved = retention.sweep(now, backups, logs)
    assert sorted(moved) == ["backups\\old_patch_20260801_before", "logs\\20260801_101010.log"]
    retired = next(p for p in backups.iterdir() if p.name.startswith("retired_") and p.name != "retired_20261114")
    assert (retired / "old_patch_20260801_before" / "f.txt").read_text() == "x"      # the content is intact
    assert (retired / "run_logs" / "20260801_101010.log").exists()
    assert not (backups / "old_patch_20260801_before").exists()


def test_recent_items_and_non_run_logs_stay(tmp_path):
    now = time.time()
    backups, logs = _setup(tmp_path, now)
    retention.sweep(now, backups, logs)
    assert (backups / "new_patch_20261110_before").exists()
    assert (logs / "20261115_101010.log").exists()
    assert (logs / "lifecompass.log").exists() and (logs / "tests.log").exists()     # never run logs, whatever their age


def test_protected_backups_never_move_whatever_their_age(tmp_path):
    now = time.time()
    backups, logs = _setup(tmp_path, now)
    retention.sweep(now, backups, logs)
    for name in ("auto", "db_old_before", "restructure_20261006_before", "system_full_20261024_013225", "retired_20261114"):
        assert (backups / name / "f.txt").exists(), name


def test_a_second_sweep_moves_nothing_and_a_name_clash_is_not_overwritten(tmp_path):
    now = time.time()
    backups, logs = _setup(tmp_path, now)
    retention.sweep(now, backups, logs)
    assert retention.sweep(now, backups, logs) == []
    (backups / "old_patch_20260801_before").mkdir()                                  # same name again, also old
    (backups / "old_patch_20260801_before" / "g.txt").write_text("y")
    _age(backups / "old_patch_20260801_before", 90, now)
    assert retention.sweep(now, backups, logs) == ["backups\\old_patch_20260801_before"]
    retired = next(p for p in backups.iterdir() if p.name == f"retired_{time.strftime('%Y%m%d', time.localtime(now))}")
    assert (retired / "old_patch_20260801_before" / "f.txt").exists()               # the first one is untouched
    assert (retired / "old_patch_20260801_before_2" / "g.txt").exists()


def test_the_limit_comes_from_config_and_a_missing_folder_is_not_an_error(tmp_path, monkeypatch):
    now = time.time()
    backups, logs = _setup(tmp_path, now)
    cfg = tmp_path / "c.toml"
    cfg.write_text("[retention]\nbackups_days = 5.0\nrun_logs_days = 365.0\n")
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    moved = retention.sweep(now, backups, logs)
    assert "backups\\new_patch_20261110_before" in moved                              # 9 days old > 5
    assert not any(m.startswith("logs\\") for m in moved)                            # 90 days < 365
    assert retention.sweep(now, tmp_path / "nope", tmp_path / "nope2") == []