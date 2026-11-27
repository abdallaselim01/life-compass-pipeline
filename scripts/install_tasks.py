"""
Registers or repairs every scheduled task of the system from its saved copy in windows_config/scheduled_tasks (the same repair the
hourly health check does by itself). Safe to run any time: a task that exists and is enabled is left alone.

    python scripts/install_tasks.py
"""
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lifecompass import health

for name in health.task_files():
    print(f"{name}: {health._task_state(name) or 'missing'}")
left = health.ensure_tasks()
print()
for name in health.task_files():
    print(f"{name}: {health._task_state(name) or 'missing'}")
print("all tasks are in place" if not left else "could not repair: " + ", ".join(issue.key for issue in left))
