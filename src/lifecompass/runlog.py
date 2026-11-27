"""
The run detail record. Every module writes what it sent, what came back and what it decided through `detail`
and `block`; while a run is open (see pipeline._RunLogger) those lines land in that run's own log file
(logs/<timestamp>.log), complete and never shortened, so any later question about a run -- what exactly the model
saw, what exactly it answered, what each stage did with it -- can be answered from that one file. Outside a run
they go to the master debug log. Secrets are masked by logging_setup.redact. Thinking text is never recorded: only
the token counts the API reports.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
import threading
from datetime import datetime
from pathlib import Path

from .logging_setup import get_logger, redact

_master = get_logger("runlog")
_sinks: list = []
_lock = threading.Lock()

# Settings worth knowing when reading a log later (names only; keys are never written, only whether they are set).
_ENV_NAMES = (
    "GEMINI_MODELS", "GEMINI_MODELS_LISTEN", "GEMINI_USE_STRUCTURED_OUTPUT", "DAILY_RESURFACE_COUNT",
    "MANUAL_ENHANCED_DIR",
)


def set_sink(fn) -> None:
    with _lock:
        _sinks.append(fn)


def clear_sink(fn) -> None:
    with _lock:
        if fn in _sinks:
            _sinks.remove(fn)


def detail(text) -> None:
    text = redact(str(text))
    with _lock:
        sink = _sinks[-1] if _sinks else None
    if sink is None:
        _master.debug(text)
        return
    try:
        sink(text)
    except Exception:  # noqa: BLE001 -- recording must never break a run
        _master.debug("run-log sink failed for: " + text[:200], exc_info=True)


def block(name: str, text) -> None:
    text = "" if text is None else str(text)
    detail(f"<<<<<< {name} ({len(text)} chars) >>>>>>\n{text}\n<<<<<< end of {name} >>>>>>")


def sha16(data) -> str:
    b = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def snapshot_lines(root: Path) -> list:
    """The state of the system at the moment a run starts: interpreter, library, settings and a fingerprint of
    every source file and prompt, so a log can always be tied back to the exact code and prompts that produced it."""
    root = Path(root)
    lines = [f"RUN ENVIRONMENT at {datetime.now().isoformat(timespec='seconds')}: python={sys.version.split()[0]} "
             f"platform={platform.platform()} pid={os.getpid()} cwd={os.getcwd()} project_root={root}"]
    try:
        import importlib.metadata as md
        lines.append(f"  google-genai={md.version('google-genai')} pydantic={md.version('pydantic')}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  library versions unavailable: {e}")
    for name in _ENV_NAMES:
        lines.append(f"  env {name}={os.environ.get(name)!r}")
    lines.append(f"  env GEMINI_API_KEY set={bool(os.environ.get('GEMINI_API_KEY'))} "
                 f"GEMINI_API_KEY_BACKUP set={bool(os.environ.get('GEMINI_API_KEY_BACKUP'))}")
    for folder, pattern in (("src/lifecompass", "*.py"), ("prompts", "*.txt")):
        for f in sorted((root / folder).glob(pattern)):
            try:
                st = f.stat()
                lines.append(f"  file {folder}/{f.name} sha256[:16]={sha16(f.read_bytes())} size={st.st_size} "
                             f"modified={datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')}")
            except OSError as e:
                lines.append(f"  file {folder}/{f.name} unreadable: {e}")
    return lines
