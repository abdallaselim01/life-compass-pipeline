"""
Watchdog for lifecompass.listener -- checks whether the local HTTP listener
(port 8935) is actually up, and relaunches it via the existing silent VBS
launcher if it isn't.

Why this exists: the listener process was observed to disappear
unexpectedly (no exception in listener_err.log, so not a Python crash --
likely an external cause such as a Windows update, a transient system
event, or an accidental manual close) after a very long-running video
processing request. Since the listener is meant to run continuously in
the background at all times, this script is scheduled to run every few
minutes via Task Scheduler (see "LifeCompass Listener Watchdog") and
silently restarts it whenever it's found to be down -- with no user
action required, consistent with the project's zero-conscious-effort
principle.

Since 2026-11-21 it asks whether the listener ANSWERS (an HTTP request), not only whether the port is open, and restarts one that
holds the port without answering (src/lifecompass/liveness.py, standard library only, so the check stays independent of the
'requests' library the listener/pipeline use).
"""
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lifecompass import liveness   # standard library only inside: the watchdog stays independent of the rest of the package


def main() -> None:
    # Up: nothing to do (the expected common case). Down: the silent launcher is started. Holding the port without answering for
    # three probes in a row (about 15 minutes): the stuck process is stopped and the launcher started (audit 2026-11-21, P2).
    liveness.ensure()


if __name__ == "__main__":
    main()
