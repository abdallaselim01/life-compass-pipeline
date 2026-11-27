"""
Centralized logging setup. One file (logs/lifecompass.log) that every
module in the project writes to at DEBUG level -- every step, every
external call (Gemini, yt-dlp, Telegram), every exception with full
traceback -- so any future problem can be diagnosed from this one file
instead of guessing. Several processes write to it (the listener and every scheduled task), so no process keeps the file open
between records and nobody rotates it while writing: the hourly health run moves it to logs/archive once it is over 5 MB
(rotate_if_large; audit 2026-11-21, P3 -- a RotatingFileHandler cannot roll a file over on Windows while another process has it open).
Archived files are moved aside by retention.py after 60 days, never deleted.

Usage in any module:
    from .logging_setup import get_logger
    log = get_logger(__name__)
    log.debug("...")
    log.info("...")
    log.warning("...")
    log.exception("...")   # inside an except block -- includes traceback automatically
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
# Tests set LIFECOMPASS_LOG_NAME (see tests/conftest.py) so their fake errors never land in the real log.
MASTER_LOG_PATH = LOGS_DIR / (os.environ.get("LIFECOMPASS_LOG_NAME") or "lifecompass.log")
ARCHIVE_DIR = LOGS_DIR / "archive"
MAX_LOG_BYTES = 5 * 1024 * 1024

# Secret redaction -- added 2026-09-29 after a maintenance audit found the
# Telegram bot token written in plain text into the logs (publisher logged
# the full request URL, which embeds the token, and requests' exception
# messages can embed it too). Applied at the FORMATTER level so it covers
# every message AND every traceback, from every module, in both the file
# and console handlers -- no call site has to remember to sanitize.
_SECRET_PATTERNS = [
    (re.compile(r"bot\d{6,}:[A-Za-z0-9_-]{20,}"), "bot<redacted>"),
    (re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "<redacted-google-key>"),
    (re.compile(r"\bAQ\.[0-9A-Za-z_-]{30,}"), "<redacted-google-key>"),          # the current key format (audit 2026-11-22)
]


def redact(text: str) -> str:
    """Masks known secret shapes (Telegram bot token, Google API key)
    in `text`. Safe to call on any string."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))



class _LogLock:
    """A Windows named mutex shared by every process of the system (same session): the one that appends a record, or moves the file
    aside, holds it for that moment only. Without it, two processes appending at once overwrite each other's lines (measured: 3
    processes writing flat out lost 2% to 25% of their lines), and a move can meet a file somebody has open. If the mutex cannot be
    had (not Windows, or it is held for over 2 s) the caller goes on without it: logging must never block or fail a run."""

    _handle = None
    _tried = False

    @classmethod
    def _get(cls):
        if not cls._tried:
            cls._tried = True
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.CreateMutexW.restype = ctypes.c_void_p
                kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
                cls._handle = kernel32.CreateMutexW(None, False, "Local\\LifeCompassLogMutex")
            except Exception:  # noqa: BLE001
                cls._handle = None
        return cls._handle

    def __enter__(self):
        self.held = False
        handle = self._get()
        if handle:
            try:
                import ctypes
                self.held = ctypes.windll.kernel32.WaitForSingleObject(handle, 2000) in (0, 0x80)   # signalled, or abandoned by a dead process
            except Exception:  # noqa: BLE001
                self.held = False
        return self

    def __exit__(self, *exc):
        if self.held:
            try:
                import ctypes
                ctypes.windll.kernel32.ReleaseMutex(self._handle)
            except Exception:  # noqa: BLE001
                pass
        return False


class SharedFileHandler(logging.Handler):
    """Appends each record to the file and closes it again at once, under the cross-process lock above, so the file is never held
    open between records: another process can move it (rotate_if_large) and the next record simply creates the new one."""

    def __init__(self, path, level=logging.NOTSET):
        super().__init__(level)
        self.path = Path(path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record) + "\n"
            for attempt in range(5):
                try:
                    with _LogLock():
                        with open(self.path, "a", encoding="utf-8") as f:
                            f.write(line)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            raise PermissionError(f"could not append to {self.path}")
        except Exception:  # noqa: BLE001
            self.handleError(record)


def rotate_if_large(path=None, max_bytes: int = MAX_LOG_BYTES, archive_dir=None) -> str | None:
    """Moves the log (never deletes it) to logs/archive/<name>_<time> once it is over max_bytes. Meant for ONE caller, the hourly
    health run. Returns the archived file's name, or None (small enough, or a process had the file open at that instant: the next
    run tries again)."""
    path = Path(path) if path else MASTER_LOG_PATH
    archive = Path(archive_dir) if archive_dir else ARCHIVE_DIR
    try:
        if path.stat().st_size <= max_bytes:
            return None
        archive.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        dest, n = archive / f"{path.stem}_{stamp}{path.suffix}", 1
        while dest.exists():                                  # never write over an earlier archive, even within the same second
            n += 1
            dest = archive / f"{path.stem}_{stamp}_{n}{path.suffix}"
        with _LogLock():
            os.replace(path, dest)
        return dest.name
    except OSError:
        return None

_configured = False


def _configure_once() -> None:
    global _configured
    if _configured:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("lifecompass")
    root.setLevel(logging.DEBUG)

    file_handler = SharedFileHandler(MASTER_LOG_PATH)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_RedactingFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    # Also echo INFO+ to console/whatever redirects stdout (listener.log,
    # terminal when run via CLI) -- keeps the existing print-based UX intact
    # while the file handler captures everything, including DEBUG detail
    # print() calls never carried.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_RedactingFormatter("%(message)s"))
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(f"lifecompass.{name}")
