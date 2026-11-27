"""A daily snapshot of the database, by itself (maintenance 2026-11-25, Fully Automated).

Until now the database was copied only by hand or just before a specific operation, so a damaged or overwritten database had no recent
copy to come back to. The hourly health run now calls take(): at most one snapshot per day, and only when the database has changed since
the last snapshot (an unchanged database is not copied again and again).

  backups\\db_daily\\<YYYYMMDD_HHMMSS>\\principles.db    a consistent copy (SQLite online backup, safe while the listener writes)
  backups\\db_daily\\<YYYYMMDD_HHMMSS>\\manifest.json    when it was taken, the source's modification time, the rows per table

A snapshot is checked before it counts (SQLite's quick_check, same tables as the source); one that fails the check is moved to
db_daily\\_failed, never kept as a good copy and never deleted. Only the newest [snapshots] keep (14) snapshots stay: older ones are MOVED
(never deleted) to backups\\retired_<date>\\db_daily. The folder name starts with "db_", so retention.py leaves the folder itself alone.

Logs only; it never sends a notification. Everything stays inside the project folder.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from . import retention, tunables
from .logging_setup import get_logger

file_log = get_logger("snapshots")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DB_PATH = PROJECT_ROOT / "data" / "principles.db"
_SNAP_DIR = PROJECT_ROOT / "backups" / "db_daily"
_FAILED = "_failed"


def list_snapshots(snap_dir: Path | None = None) -> list[Path]:
    """The snapshot folders, oldest first (a folder counts only when it holds a database file)."""
    root = Path(snap_dir) if snap_dir else _SNAP_DIR
    try:
        return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "principles.db").exists())
    except OSError:
        return []


def _manifest(folder: Path) -> dict:
    try:
        data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- no usable manifest is an empty one
        return {}


def _copy_and_check(db_path: Path, dest: Path) -> dict:
    """SQLite online backup of the live database to `dest`, then the checks. Returns {table: rows} of the copy; raises when the copy is bad."""
    src = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
            if dst.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("the copy failed SQLite's quick_check")
            tables = sorted(r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'"))
            copied = sorted(r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'"))
            if tables != copied:
                raise RuntimeError(f"the copy has other tables than the database ({tables} vs {copied})")
            return {t: dst.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in copied}
        finally:
            dst.close()
    finally:
        src.close()


def _prune(now: float, snap_dir: Path) -> list[str]:
    keep = int(tunables.get("snapshots", "keep"))
    old = list_snapshots(snap_dir)[:-keep] if keep > 0 else []
    retired = snap_dir.parent / f"retired_{datetime.fromtimestamp(now):%Y%m%d}" / "db_daily"
    return [name for p in old if (name := retention._move(p, retired))]


def take(now: float | None = None, db_path: Path | None = None, snap_dir: Path | None = None) -> Path | None:
    """One look at the snapshots. Returns the new snapshot's folder, or None when none was due (one already today, or the database has
    not changed since the last one) or it could not be made (logged)."""
    now = time.time() if now is None else now
    db = Path(db_path) if db_path else _DB_PATH
    root = Path(snap_dir) if snap_dir else _SNAP_DIR
    try:
        mtime_ns = db.stat().st_mtime_ns
    except OSError:
        return None                                              # no database yet: nothing to copy
    name = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
    existing = list_snapshots(root)
    if existing:
        last = existing[-1]
        if last.name[:8] == name[:8]:
            return None                                          # one a day
        if int(_manifest(last).get("source_mtime_ns", -1)) >= mtime_ns:
            return None                                          # unchanged since the last snapshot
    final = root / name
    work = root / f".{name}.part"
    try:
        work.mkdir(parents=True, exist_ok=True)
        rows = _copy_and_check(db, work / "principles.db")       # the source's modification time was read BEFORE the copy: a write
        (work / "manifest.json").write_text(                     # during the copy makes the next look take another snapshot
            json.dumps({"taken_at": datetime.fromtimestamp(now).isoformat(timespec="seconds"), "source_mtime_ns": mtime_ns,
                        "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        work.rename(final)
    except Exception as e:  # noqa: BLE001 -- a failed snapshot is tried again at the next run, and never stops the health run
        file_log.warning(f"snapshot failed ({type(e).__name__}: {e}); the half-made copy is kept in db_daily\\{_FAILED}")
        if work.exists():
            retention._move(work, root / _FAILED)
        return None
    moved = _prune(now, root)
    file_log.info(f"db snapshot taken: db_daily\\{name} ({sum(rows.values())} rows in {len(rows)} tables)"
                  + (f"; older ones moved aside (not deleted): {moved}" if moved else ""))
    return final
