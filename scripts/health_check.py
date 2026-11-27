"""
Hourly self-maintenance (scheduled task "LifeCompass Health Check"): looks, repairs, and tells the user only about a problem that is
still there after the repair, once (see src/lifecompass/health.py). Also keeps yt-dlp up to date (src/lifecompass/ytdlp_update.py).
Everything it does is in logs/lifecompass.log. Run it by hand to see the same thing: python scripts/health_check.py
"""
import os
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

if __name__ == "__main__":
    # pythonw has no console: give print() somewhere to go
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from lifecompass import health


def main() -> None:
    state = health.run()
    open_issues = state.get("issues", {})
    print(f"open issues: {sorted(open_issues) or 'none'}")


if __name__ == "__main__":
    main()
