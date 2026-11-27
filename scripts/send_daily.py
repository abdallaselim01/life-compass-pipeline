"""
Task Scheduler entrypoint for hourly resurfacing.

Runs under pythonw.exe (not python.exe) specifically so no console window
ever flashes on screen -- this used to run under python.exe, which briefly
showed/hid a console window on every hourly tick (the user explicitly does
not want to see this, even briefly).

pythonw.exe has NO console attached at all: sys.stdout and sys.stderr are
both None (not just redirected -- literally None), unlike a normal
console app. Any code that assumes sys.stdout exists (a bare print(), a
logging.StreamHandler() with its default target, an encoding check like
`sys.stdout.encoding`) will crash with AttributeError the instant it
runs under pythonw. This bit the project directly: switching to pythonw
to fix the window-flash caused every hourly run to fail silently
(LastTaskResult=1, zero log lines written -- the crash happened before
any logging call was even reached, at module-import time).

Fix: redirect sys.stdout/sys.stderr to os.devnull BEFORE any other
import, so every downstream print()/logging call has a real (if
throwaway) file object to write to instead of None. This must happen
first, before importing anything from the lifecompass package.
"""
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
import os

if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from lifecompass.resurface import run_random_resurface

if __name__ == "__main__":
    run_random_resurface()
