#!/usr/bin/env python
"""Read-only audit of data/principles.db: integrity, row counts, orphans, missing image files. Changes nothing."""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
c = sqlite3.connect(f"file:{(ROOT / 'data' / 'principles.db').as_posix()}?mode=ro", uri=True)
print("integrity:", c.execute("pragma integrity_check").fetchone()[0],
      "| user_version:", c.execute("pragma user_version").fetchone()[0],
      "| foreign_key_check rows:", len(c.execute("pragma foreign_key_check").fetchall()))
for (t,) in c.execute("select name from sqlite_master where type='table' order by name").fetchall():
    print(f"  {t}: {c.execute(f'select count(*) from {t}').fetchone()[0]}")
checks = {
    "image links to a missing principle": "select count(*) from principle_images where principle_id not in (select id from principles)",
    "image links to a missing image": "select count(*) from principle_images where image_id not in (select id from images)",
    "principles whose video has no videos row": "select count(*) from principles where video_id not in (select video_id from videos)",
    "principles with an empty domain": "select count(*) from principles where domain=''",
    "principles with empty text": "select count(*) from principles where trim(principle)=''",
    "processed videos without stage fingerprints": "select count(*) from processed_videos where video_id not in (select video_id from stage_fingerprints)",
    "documents without a processed mark": "select count(*) from documents where video_id not in (select video_id from processed_videos)",
    "pending_retries rows": "select count(*) from pending_retries",
    "principles whose doc_hash differs from their video's document": "select count(*) from principles p join documents d on d.video_id=p.video_id where p.doc_hash!=d.doc_hash",
}
for name, sql in checks.items():
    print(f"{name}: {c.execute(sql).fetchone()[0]}")
print("domains:", c.execute("select domain,count(*) from principles group by domain").fetchall())
for i, p, cap in c.execute("select id,enhanced_path,standalone_caption from images").fetchall():
    f = Path(p) if Path(p).is_absolute() else ROOT / p
    print(f"  image {i}: {p} {'exists' if f.exists() else 'MISSING'} caption_len={len(cap)}")
