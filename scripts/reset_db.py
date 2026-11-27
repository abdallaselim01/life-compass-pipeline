#!/usr/bin/env python
"""
Zero the database, everything back to how it was before the first bookmarklet press, as one command instead of hand-written SQL.

What it does (nothing is emptied before a verified backup exists):
  1. stops the listener (nothing may write to the database meanwhile);
  2. backs up data/principles.db with SQLite's own backup API into backups/db_zero_<stamp>_before/ and checks the copy
     (integrity_check + the same row count in every table), together with data/model_health.json and the folder
     data/enhanced_images (with replaced/), because a re-dropped image would otherwise overwrite a file with the same name;
  3. empties every table EXCEPT
       settings  holds the image-ingestion checkpoint (without it the next run would take "first run" for a fresh install);
       images    the dropped pictures stay, as they were at the press; only `standalone_caption`, the one column a run writes, is
                 cleared (ids, paths, times_shown and last_shown_at stay);
     resets the AUTOINCREMENT counters of the emptied tables so ids start again at 1, empties data/model_health.json, then VACUUM;
  4. starts the listener again (so it picks up the zeroed state).
The schema and user_version stay as they are. No file other than the database content and model_health.json is changed.

  python scripts/reset_db.py                 # dry run: row counts, writes nothing
  python scripts/reset_db.py --go            # stop the listener, backup, verify, empty, start the listener
  python scripts/reset_db.py --go --no-restart   # the same, without stopping/starting the listener (for a test on a scratch copy)
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DB = ROOT / "data" / "principles.db"
ENHANCED = ROOT / "data" / "enhanced_images"
HEALTH = ROOT / "data" / "model_health.json"
KEEP = {"settings", "images"}
IMAGE_COLUMNS_A_RUN_WRITES = ("standalone_caption",)


def tables(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0] for t in tables(conn)}


def _images_rest(conn: sqlite3.Connection) -> list[tuple]:
    """The images table without the columns a run writes: what must be identical before and after."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(images)") if r[1] not in IMAGE_COLUMNS_A_RUN_WRITES]
    return conn.execute(f"SELECT {', '.join(cols)} FROM images ORDER BY id").fetchall() if cols else []


def _stop_listener() -> None:
    from lifecompass import liveness
    for pid in liveness.listening_pids():
        print("stopping the listener, pid", pid)
        liveness.kill(pid)
    for _ in range(20):
        if liveness.probe() == "down":
            return
        time.sleep(1)
    raise SystemExit("the listener did not stop: nothing was changed")


def main() -> None:
    go = "--go" in sys.argv
    restart = "--no-restart" not in sys.argv
    conn = sqlite3.connect(str(DB))
    before = counts(conn)
    uv = conn.execute("PRAGMA user_version").fetchone()[0]
    print("user_version:", uv)
    for t, n in before.items():
        print(f"  {t}: {n}" + ("   (kept)" if t in KEEP else ""))
    if not go:
        print("dry run -- nothing written (use --go)")
        return

    if restart:
        _stop_listener()
    try:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = ROOT / "backups" / f"db_zero_{stamp}_before"
        folder.mkdir(parents=True)
        dst = sqlite3.connect(str(folder / "principles.db"))
        conn.backup(dst)
        dst.close()
        chk = sqlite3.connect(str(folder / "principles.db"))
        assert chk.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "backup failed integrity_check"
        assert counts(chk) == before, "backup row counts differ from the database"
        assert chk.execute("PRAGMA user_version").fetchone()[0] == uv
        chk.close()
        if HEALTH.is_file():
            shutil.copy2(HEALTH, folder / "model_health.json")
        if ENHANCED.is_dir():
            shutil.copytree(ENHANCED, folder / "enhanced_images")
            n_src = sum(1 for p in ENHANCED.rglob("*") if p.is_file())
            n_dst = sum(1 for p in (folder / "enhanced_images").rglob("*") if p.is_file())
            assert n_src == n_dst, "image backup file count differs"
        print("backup verified:", folder)

        images_rest_before = _images_rest(conn) if "images" in before else []
        conn.execute("PRAGMA foreign_keys=OFF")
        wiped = [t for t in before if t not in KEEP]
        with conn:
            for t in wiped:
                conn.execute(f"DELETE FROM [{t}]")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='sqlite_sequence'").fetchone():
                conn.executemany("DELETE FROM sqlite_sequence WHERE name = ?", [(t,) for t in wiped])
            if "images" in before:
                for col in IMAGE_COLUMNS_A_RUN_WRITES:
                    conn.execute(f"UPDATE images SET {col} = ''")
        conn.execute("VACUUM")
        after = counts(conn)
        assert all(after[t] == 0 for t in wiped), after
        assert all(after[t] == before[t] for t in KEEP), "a kept table changed its row count"
        if "images" in before:
            assert _images_rest(conn) == images_rest_before, "an image row changed beyond its caption"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == uv
        HEALTH.parent.mkdir(parents=True, exist_ok=True)
        HEALTH.write_text(json.dumps({"models": {}}), encoding="utf-8")
        print("emptied:", ", ".join(wiped))
        print("kept:", {t: after[t] for t in KEEP}, "(image captions cleared)")
        print("done: integrity ok, user_version", uv)
    finally:
        conn.close()
        if restart:
            from lifecompass import liveness
            liveness.launch()
            print("listener started again")


if __name__ == "__main__":
    main()
