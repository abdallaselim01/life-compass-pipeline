"""Is the listener really answering? (audit 2026-11-21, P2)

The watchdog (every 5 minutes) and the health check (hourly) used to ask only "is something listening on the port?". A listener that
is stuck but still holds the port answers that question with yes, forever, and is never restarted. This asks the better question: an
HTTP request is sent (OPTIONS /, which the listener answers with 204 and does nothing else, writes no log line) and an answer must
come back within a few seconds.

  up      it answers.
  down    nothing listens on the port: the launcher is started.
  hung    it holds the port but does not answer. Only after HUNG_AFTER such probes in a row (the watchdog runs every 5 minutes, so about
          15 minutes of silence) is its process stopped and the launcher started. Work in progress is not hurt by a slow listener:
          every request has its own thread, so a listener busy with a video still answers.

Standard library only (the watchdog uses it too, and must stay independent of the rest of the package). The state (how many hung
probes in a row, when it was last started) is data/listener_liveness.json."""
from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from . import tunables

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST, PORT = "127.0.0.1", 8935
LAUNCHER = PROJECT_ROOT / "start_listener_silent.vbs"
STATE_PATH = PROJECT_ROOT / "data" / "listener_liveness.json"
HUNG_AFTER = tunables.get("liveness", "hung_after")                    # hung probes in a row before the listener is restarted (config.toml [liveness])
LAUNCH_GRACE_SECONDS = tunables.get("liveness", "launch_grace_seconds")   # a listener started less than this long ago is not started again (the watchdog and the health run may meet)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _log(level: str, text: str) -> None:
    """Events only (a restart is rare). Never raises: the watchdog must not fail because logging could not start."""
    try:
        from .logging_setup import get_logger
        getattr(get_logger("liveness"), level)(text)
    except Exception:  # noqa: BLE001
        pass


def probe(timeout: float = 5.0) -> str:
    """"up", "down" or "hung" (see the module docstring)."""
    try:
        with socket.create_connection((HOST, PORT), timeout=2):
            pass
    except OSError:
        return "down"
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
        try:
            conn.request("OPTIONS", "/")
            conn.getresponse().read()
        finally:
            conn.close()
        return "up"
    except (OSError, http.client.HTTPException):
        return "hung"


def _read_state() -> dict:
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- no usable state is an empty state
        return {}


def _write_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


def listening_pids() -> list:
    """Process ids that listen on the listener's port."""
    r = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, creationflags=_NO_WINDOW, timeout=30)
    pids = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3].upper() == "LISTENING" and re.search(rf"[:.]{PORT}$", parts[1]):
            if parts[4].isdigit() and int(parts[4]) > 0:
                pids.append(int(parts[4]))
    return sorted(set(pids))


def kill(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=_NO_WINDOW, timeout=30)


def launch() -> None:
    subprocess.Popen(["wscript.exe", str(LAUNCHER)], shell=False, creationflags=_NO_WINDOW)


def ensure(now: float | None = None) -> str:
    """One look at the listener, and the repair when it is needed. Returns "up", "hung" (silent, not yet restarted),
    "launched" (it was down) or "restarted" (it was hung for HUNG_AFTER probes in a row)."""
    now = time.time() if now is None else now
    status = probe()
    state = _read_state()
    if status == "up":
        if state.get("hung"):
            state["hung"] = 0
            _write_state(state)
        return "up"
    recently = now - float(state.get("launched_at", 0) or 0) < LAUNCH_GRACE_SECONDS
    if status == "down":
        state["hung"] = 0
        if recently:
            return "up" if probe() == "up" else "launched"
        state["launched_at"] = now
        _write_state(state)
        _log("warning", "listener is down: starting it")
        launch()
        return "launched"
    state["hung"] = int(state.get("hung", 0) or 0) + 1
    _log("warning", f"listener holds port {PORT} but does not answer ({state['hung']} of {HUNG_AFTER} probes)")
    if state["hung"] < HUNG_AFTER or recently:
        _write_state(state)
        return "hung"
    for pid in listening_pids():
        _log("warning", f"listener (pid {pid}) did not answer {HUNG_AFTER} probes in a row: stopping it")
        kill(pid)
    for _ in range(15):                                   # wait until the port is free, then start it again
        time.sleep(1)
        if probe() == "down":
            break
    state.update(hung=0, launched_at=now)
    _write_state(state)
    launch()
    return "restarted"
