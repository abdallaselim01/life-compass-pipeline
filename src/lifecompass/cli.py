"""Manual CLI entry point: python -m lifecompass.cli <youtube_url>"""
from __future__ import annotations

import sys

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from .pipeline import process_video


def main():
    load_dotenv(override=True)
    if len(sys.argv) < 2:
        print("Usage: python -m lifecompass.cli <youtube_url>")
        sys.exit(1)
    url = sys.argv[1]
    result = process_video(url)
    print(f"\n[{'OK' if result.ok else 'FAILED'}] {result.message}")
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
