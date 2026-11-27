#!/usr/bin/env python
"""
READ-ONLY check (user rule 2026-11-08/09): every stored principle must be one continuous passage of its video's transcript, with
stutter fragments (a lone letter, or the bare article, left hanging before the full word) taken out.

  python scripts/verify_principles_literal.py [path\\to\\principles.db]

For each stored principle it uses the same mechanical test as the extractor (`_conform`). A principle that is not already such a
passage is listed, with what the guard would turn it into (stutter removed / repaired) or that nothing in the transcript matches it.
Nothing is written. Exit code 1 when any stored principle is not such a passage.
"""
from __future__ import annotations

import sqlite3
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lifecompass.document import SourceDocument                      # noqa: E402
from lifecompass.principle_extractor import _clean_view, _conform    # noqa: E402


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "principles.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    total = bad = 0
    for video_id, blob in con.execute("SELECT video_id, doc_json FROM documents").fetchall():
        view = _clean_view(SourceDocument.from_json(blob))
        rows = con.execute("SELECT id, principle FROM principles WHERE video_id = ? ORDER BY id", (video_id,)).fetchall()
        offenders = []
        for pid, text in rows:
            total += 1
            fixed = _conform(view, text)
            if fixed is None or fixed[1] != "as is":
                offenders.append((pid, text, fixed))
        bad += len(offenders)
        print(f"{video_id}: {len(rows)} principle(s), {len(rows) - len(offenders)} fine, {len(offenders)} NOT")
        for pid, text, fixed in offenders:
            verdict = f"the guard would make it ({fixed[1]}): {fixed[0]}" if fixed else "nothing in the transcript matches it"
            print(f"  #{pid}: {text}\n      -> {verdict}")
    print(f"total: {total} principle(s), {bad} not a continuous passage of the transcript")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
