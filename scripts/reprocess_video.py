#!/usr/bin/env python
"""
Process a video again from scratch, replacing what is stored for it.

Use it after the pipeline itself changed (a restructure, a better prompt) to
redo a video that was already processed -- the bookmarklet skips videos it has
already done. Dry-run by default: it only prints what it would delete.

  python scripts/reprocess_video.py VIDEO_ID_OR_URL            # show what would change
  python scripts/reprocess_video.py VIDEO_ID_OR_URL --yes      # do it

What it does with --yes:
  1. backs the database up to backups/db_<timestamp>_before_reprocess/
  2. deletes the video's principles (and their image links), its
     stored transcript/document/progress, any queued retry, and its
     "already processed" mark. Screenshots (images) are kept.
  3. runs the normal pipeline on the video (so Telegram receives the result).
"""
from __future__ import annotations

import argparse
import datetime
import sqlite3
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)

from lifecompass import store  # noqa: E402
from lifecompass.audio import extract_video_id, watch_url  # noqa: E402


def plan(conn: sqlite3.Connection, video_id: str) -> dict:
    ids = [r[0] for r in conn.execute("SELECT id FROM principles WHERE video_id = ?", (video_id,))]
    marks = ",".join("?" * len(ids)) or "NULL"
    return {
        "principle_ids": ids,
        "image_links": conn.execute(
            f"SELECT COUNT(*) FROM principle_images WHERE principle_id IN ({marks})", ids).fetchone()[0],
        "transcript": conn.execute("SELECT COUNT(*) FROM transcripts WHERE video_id = ?", (video_id,)).fetchone()[0],
        "document": conn.execute("SELECT COUNT(*) FROM documents WHERE video_id = ?", (video_id,)).fetchone()[0],
        "pending": conn.execute("SELECT COUNT(*) FROM pending_retries WHERE video_id = ?", (video_id,)).fetchone()[0],
        "processed": conn.execute("SELECT COUNT(*) FROM processed_videos WHERE video_id = ?", (video_id,)).fetchone()[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video id or YouTube URL")
    ap.add_argument("--yes", action="store_true", help="actually delete and reprocess")
    args = ap.parse_args()

    video_id = args.video if len(args.video) == 11 and "/" not in args.video else extract_video_id(args.video)
    store.count_all()                      # make sure the schema is current
    conn = sqlite3.connect(store.DB_PATH)
    p = plan(conn, video_id)
    print(f"video {video_id}: {len(p['principle_ids'])} principle(s), "
          f"{p['image_links']} image link(s), transcript={p['transcript']}, document={p['document']}, "
          f"pending={p['pending']}, processed-mark={p['processed']}")
    if not args.yes:
        print("dry run -- nothing changed. Re-run with --yes to back up, delete and reprocess.")
        return

    backup = ROOT / "backups" / f"db_{datetime.datetime.now():%Y%m%d_%H%M%S}_before_reprocess"
    backup.mkdir(parents=True, exist_ok=True)
    dst = sqlite3.connect(backup / "principles.db")
    conn.backup(dst)
    dst.close()
    print("backup:", backup)

    ids = p["principle_ids"]
    marks = ",".join("?" * len(ids)) or "NULL"
    with conn:
        conn.execute(f"DELETE FROM principle_images WHERE principle_id IN ({marks})", ids)
        conn.execute("DELETE FROM principles WHERE video_id = ?", (video_id,))
        for table in ("transcripts", "documents", "pending_retries"):
            conn.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM processed_videos WHERE video_id = ?", (video_id,))
    conn.close()
    print("deleted. Reprocessing ...")

    from lifecompass.pipeline import process_video
    result = process_video(watch_url(video_id))
    print(("OK: " if result.ok else "PENDING: ") + result.message)


if __name__ == "__main__":
    main()
