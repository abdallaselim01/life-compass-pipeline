#!/usr/bin/env python
"""
Restore data/principles.db from a backup folder made by scripts/reset_db.py (user request 2026-11-08: the stage-rerun
agreement -- redo only the changed stage and what depends on it -- needs the stored transcript and fingerprints back).

  1. backs up the CURRENT database (SQLite backup API) into backups/db_before_restore_<stamp>/ and checks the copy;
  2. copies the backup's database into data/principles.db with SQLite's backup API, then checks integrity_check,
     user_version and the same row count in every table as the backup.
Nothing else is touched (no image files, no other tables outside the database). The listener must be stopped first.

  python scripts/restore_db_from_backup.py <backup_folder>          # dry run
  python scripts/restore_db_from_backup.py <backup_folder> --go
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "principles.db"


def tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0] for t in tables(conn)}


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    go = "--go" in sys.argv
    assert len(args) == 1, "usage: restore_db_from_backup.py <backup_folder> [--go]"
    src_path = Path(args[0])
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    src_db = src_path / "principles.db" if src_path.is_dir() else src_path
    assert src_db.is_file(), f"no database at {src_db}"
    assert ROOT in src_db.resolve().parents, "backup must live inside the project folder"

    src = sqlite3.connect(f"file:{src_db.as_posix()}?mode=ro", uri=True)
    assert src.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "backup failed integrity_check"
    want = counts(src)
    want_uv = src.execute("PRAGMA user_version").fetchone()[0]
    cur = sqlite3.connect(str(DB))
    now = counts(cur)
    print("source:", src_db, "user_version", want_uv)
    for t in sorted(set(want) | set(now)):
        print(f"  {t}: now {now.get(t)} -> backup {want.get(t)}")
    if not go:
        print("dry run -- nothing written (use --go)")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = ROOT / "backups" / f"db_before_restore_{stamp}"
    folder.mkdir(parents=True)
    dst = sqlite3.connect(str(folder / "principles.db"))
    cur.backup(dst)
    dst.close()
    chk = sqlite3.connect(str(folder / "principles.db"))
    assert chk.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "pre-restore backup failed integrity_check"
    assert counts(chk) == now, "pre-restore backup row counts differ"
    chk.close()
    print("current database backed up and verified:", folder)

    src.backup(cur)
    cur.commit()
    after = counts(cur)
    assert after == want, (after, want)
    assert cur.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert cur.execute("PRAGMA user_version").fetchone()[0] == want_uv
    print("restored: row counts equal the backup, integrity ok, user_version", want_uv)


if __name__ == "__main__":
    main()
