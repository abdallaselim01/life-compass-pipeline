"""Every tunable number of the system, in one file: config.toml at the project root (design decision 2026-11-17: Fully Dynamic).

Each key has a built-in default here (identical to the value the code used before this file existed); config.toml only needs the
keys the user wants to change. The file is re-read whenever it changes (modification time), and a value that is not a number of
the right kind or is out of range is ignored with one warning, so a typo can never stop the system.

What is NOT here, on purpose: anything that is part of a stage fingerprint (the prompts, thinking level, output limits, the
structured-output setting) -- changing those re-runs stages, so they stay in code where a change is a visible patch -- and secrets
and keys (.env).

A module that copies a value into a constant at import time (most of them) sees a change when its process starts: the scheduled
tasks start a new process every run, and the listener is restarted by the watchdog.
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

from .logging_setup import get_logger

file_log = get_logger("tunables")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# (default, lowest, highest) per key. The default's type (int or float) is the type of the value that comes back.
_TABLE = {
    "resurface": {
        "standalone_image_probability": (0.15, 0.0, 1.0),
        "runs_per_day": (24, 1, 1440),
        "quiet_hours_start": (0, 0, 24),
        "quiet_hours_end": (0, 0, 24),
        "daily_count": (5, 1, 1000),
    },
    "telegram": {
        "max_attempts": (3, 1, 20),
        "retry_delay_seconds": (3.0, 0, 600),
        "network_retries": (4, 0, 20),
        "network_retry_base_seconds": (5.0, 0.0, 600.0),
        "network_retry_max_seconds": (300.0, 1.0, 86400.0),
        "album_min_upload_kbps": (40.0, 1.0, 100000.0),
        "album_timeout_margin_seconds": (30.0, 0.0, 3600.0),
    },
    "audio": {
        "duration_tolerance_seconds": (10.0, 0.0, 600.0),
        "info_timeout_seconds": (300.0, 10.0, 86400.0),
        "download_timeout_seconds": (3600.0, 60.0, 86400.0),
        "probe_timeout_seconds": (60.0, 5.0, 3600.0),
    },
    "transcript": {
        "min_words_per_minute": (40, 0, 400),
        "min_seconds_for_rate_check": (120, 0, 7200),
    },
    "extract_guard": {
        "max_repair_growth": (2, 1, 20),
        "min_repair_coverage": (0.5, 0.0, 1.0),
    },
    "waits": {
        "timeout_margin": (2.0, 1.0, 100.0),
        "timeout_floor_seconds": (120.0, 1.0, 86400.0),
        "long_wait_floor_seconds": (450.0, 1.0, 86400.0),
        "stream_stall_seconds": (120.0, 1.0, 86400.0),
        "exhaustion_recheck_seconds": (1800.0, 1.0, 86400.0),
        "long_wait_seconds_per_audio_minute": (20.0, 0.0, 3600.0),
    },
    "retry": {
        "max_attempts": (36, 1, 10000),
        "max_delivery_attempts": (288, 1, 100000),
    },
    "liveness": {
        "hung_after": (3, 1, 100),
        "launch_grace_seconds": (60, 0, 3600),
    },
    "models": {
        "chain_length": (3, 1, 10),
        "refresh_hours": (6.0, 0.0, 720.0),
        "failures_to_count_as_down": (2, 1, 100),
    },
    "alerts": {
        "max_per_day": (2, 1, 20),
        "persist_minutes": (60, 0, 10080),
        "models_down_hours": (12.0, 0.0, 720.0),
        "min_free_gb": (5.0, 0.0, 100000.0),
    },
    "retention": {
        "backups_days": (60.0, 1.0, 3650.0),
        "run_logs_days": (60.0, 1.0, 3650.0),
    },
    "snapshots": {
        "keep": (14, 1, 3650),
    },
    "maintenance": {
        "ytdlp_check_days": (7.0, 0.1, 365.0),
        "ytdlp_failure_retry_hours": (6.0, 0.0, 168.0),
        "orphan_run_minutes": (20.0, 5.0, 1440.0),
        "orphan_max_hours": (6.0, 0.5, 168.0),
        "heartbeat_seconds": (300, 10, 3600),
    },
}

# Text settings (name-like words only, so a typo can never become something odd): (default)
_TEXT = {
    "models": {
        "family": "hydrogen",                  # which model family the chain is made of: gemini-<major>[.<minor>]-<family> (2026-11-23)
        "last_resort": "gemini-4-hydrogen",    # the chain when Google could never be asked (no stored list at all)
    },
}
# Lists of plain model names (same word rule as the text settings): default []
_WORDS = {
    "models": {
        "after": [],                           # exact model names tried after the family chain, in this order (2026-11-24)
    },
}
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]*")

_cache: dict = {"path": None, "mtime": None, "data": {}}
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        file_log.warning(message)


def _path() -> Path:
    return Path(os.environ.get("LIFECOMPASS_CONFIG") or PROJECT_ROOT / "config.toml")


def _load() -> dict:
    path = _path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return {}                                     # no file: every default
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["data"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 -- a file that cannot be read is no file
        _warn_once(f"file:{mtime}", f"{path.name} could not be read ({type(e).__name__}); using the defaults")
        data = {}
    _cache.update(path=path, mtime=mtime, data=data)
    return data


def get_text(section: str, key: str) -> str:
    """The text value of [section] key: lower-case letters, digits, dots and dashes only; anything else is ignored with one warning."""
    default = _TEXT[section][key]
    value = (_load().get(section) or {}).get(key, default)
    if isinstance(value, str) and _WORD.fullmatch(value):
        return value
    _warn_once(f"{section}.{key}:{value!r}", f"config.toml [{section}] {key} = {value!r} is not a plain model-style word; using the default {default!r}")
    return default


def get_words(section: str, key: str) -> list[str]:
    """The list of plain model-style words of [section] key, in the written order, duplicates dropped. A value that is not a list of such
    words is ignored as a whole, with one warning, and the default (an empty list) is used."""
    default = list(_WORDS[section][key])
    value = (_load().get(section) or {}).get(key, default)
    if isinstance(value, list) and all(isinstance(v, str) and _WORD.fullmatch(v) for v in value):
        return list(dict.fromkeys(value))
    _warn_once(f"{section}.{key}:{value!r}", f"config.toml [{section}] {key} = {value!r} is not a list of plain model-style words; "
                                             f"using the default {default!r}")
    return default


def get(section: str, key: str):
    """The value of [section] key. KeyError for a key that does not exist (a programming error, not a config error)."""
    default, low, high = _TABLE[section][key]
    value = (_load().get(section) or {}).get(key, default)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high
    if ok and isinstance(default, int) and float(value) != int(value):
        ok = False                                    # a whole number is expected (2.5 attempts makes no sense)
    if not ok:
        _warn_once(f"{section}.{key}:{value!r}", f"config.toml [{section}] {key} = {value!r} is not a number between {low} and {high}; "
                                                 f"using the default {default}")
        return default
    return int(value) if isinstance(default, int) else float(value)
