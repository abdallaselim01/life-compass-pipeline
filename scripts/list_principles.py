"""Quick inspector: prints every stored principle, grouped by domain,
Not part of the runtime pipeline --
a debug/browse tool, same spirit as the language pipeline's peek_*.py
scripts."""
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lifecompass import store

with store._connect() as conn:
    rows = conn.execute("SELECT * FROM principles ORDER BY domain, created_at").fetchall()

if not rows:
    print("No principles stored yet.")
    sys.exit(0)

by_domain: dict[str, list] = {}
for r in rows:
    by_domain.setdefault(r["domain"], []).append(r)

for domain, items in by_domain.items():
    print(f"\n=== {domain} ({len(items)}) ===")
    for r in items:
        p = store._row_to_principle(r)
        print(f"  [{p.id}] {p.principle}")
        print(f"      from: {p.video_title}")
        print(f"      shown {p.times_shown} time(s)")

print(f"\nTotal: {len(rows)} principles")
