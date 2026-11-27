"""Automatic retention (design decision 2026-11-19, Fully Automated): old backups and old run logs move aside by themselves.

Nothing is deleted here, ever. An item older than its limit is MOVED to backups\\retired_<date> inside the project, so it can always be
brought back by hand. The limits live in config.toml [retention] (days, 60 by default). The protected backups (the ones the user
said to keep, every db_* and anything already retired) are never touched, whatever their age.

Runs once per health run (hourly) and logs only; it never sends a notification.
"""
from __future__ import annotations

import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from . import tunables
from .logging_setup import get_logger

file_log = get_logger("retention")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROTECTED_BACKUPS = frozenset({"auto", "restructure_20261006_before", "system_full_20261024_013225"})
PROTECTED_PREFIXES = ("db_", "retired_")
_RUN_LOG = re.compile(r"^\d{8}_\d{6}\.log$")      # logs\<YYYYMMDD_HHMMSS>.log; lifecompass.log, resurface.log, tests.log never match


def is_protected(name: str) -> bool:
    return name in PROTECTED_BACKUPS or name.startswith(PROTECTED_PREFIXES)


def _free_name(dest: Path) -> Path:
    if not dest.exists():
        return dest
    n = 2
    while True:
        candidate = dest.with_name(f"{dest.stem}_{n}{dest.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _move(src: Path, folder: Path) -> str | None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(_free_name(folder / src.name)))
        return src.name
    except OSError as e:
        file_log.warning(f"retention: could not move {src.name} ({type(e).__name__}: {e}); left where it is")
        return None


def sweep(now: float | None = None, backups_dir: Path | None = None, logs_dir: Path | None = None) -> list[str]:
    """Move what is older than its limit. Returns the names that were moved (empty when nothing was due)."""
    now = time.time() if now is None else now
    backups = Path(backups_dir) if backups_dir else PROJECT_ROOT / "backups"
    logs = Path(logs_dir) if logs_dir else PROJECT_ROOT / "logs"
    backup_limit = tunables.get("retention", "backups_days") * 86400
    log_limit = tunables.get("retention", "run_logs_days") * 86400
    retired = backups / f"retired_{datetime.fromtimestamp(now):%Y%m%d}"
    moved: list[str] = []
    try:
        entries = sorted(backups.iterdir())
    except OSError:
        entries = []
    for p in entries:
        if is_protected(p.name):
            continue
        try:
            due = now - p.stat().st_mtime > backup_limit
        except OSError:
            continue
        if due and (name := _move(p, retired)):
            moved.append(f"backups\\{name}")
    try:
        log_files = sorted(logs.iterdir())
    except OSError:
        log_files = []
    for p in log_files:
        if not (p.is_file() and _RUN_LOG.match(p.name)):
            continue
        try:
            due = now - p.stat().st_mtime > log_limit
        except OSError:
            continue
        if due and (name := _move(p, retired / "run_logs")):
            moved.append(f"logs\\{name}")
    try:
        archived = sorted((logs / "archive").iterdir())          # master logs rotated by the health run (logging_setup.rotate_if_large)
    except OSError:
        archived = []
    for p in archived:
        if not p.is_file():
            continue
        try:
            due = now - p.stat().st_mtime > log_limit
        except OSError:
            continue
        if due and (name := _move(p, retired / "run_logs")):
            moved.append(f"logs\\archive\\{name}")
    if moved:
        file_log.info(f"retention: moved {len(moved)} old item(s) to {retired.name}: {moved}")
    return moved