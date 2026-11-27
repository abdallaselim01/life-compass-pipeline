"""The Gemini model chain: found by asking Google, newest first (design decision 2026-11-17: Fully Dynamic; 2026-11-21: the order never changes;
2026-11-23: Hydrogen instead of Flash).

The only thing named here is the model FAMILY, and it is a setting (config.toml [models] family, "hydrogen" since 2026-11-23, "flash"
before). Three things are involved in the chain:

  1. WHICH models exist: the key is asked which models it can call (`models.list`, a metadata call that spends no quota). The
     candidates are the stable models of the family, i.e. names of the form gemini-<major>[.<minor>]-<family> exactly, so no lite,
     image, tts, live, preview, omni or alias names. They are ranked newest first, and the first `models.chain_length` of them are the
     chain (today that is gemini-4-hydrogen; a gemini-5-hydrogen would come first with gemini-4-hydrogen behind it). Behind the family
     come the exact names of `models.after` (config.toml), in the order written: today gemini-3.8-flash then gemini-3.6-flash. The answer is
     kept in data/models_cache.json for `models.refresh_hours`; when Google cannot be asked the stale copy is used, and with no copy at
     all the chain is `models.last_resort` (config.toml).
  2. THE ORDER NEVER CHANGES, for any reason (design decision 2026-11-21): the chain is always those models, newest first. Nothing that
     happens to a model (a timeout, an error, an unusable reply, a quota limit, a success) moves it. Every attempt of the fallback walk
     (llm._generate_with_fallback) still reports "ok" or "fail" here, only for the record in data/model_health.json (the failures in
     a row, the last failure, the last answer): it feeds health.py's "every model keeps failing" message and scripts/list_models.py.
     A quota limit is never counted as a failure: it says nothing about the model, and llm.py already remembers it.
  3. WHAT the user pinned: GEMINI_MODELS (and GEMINI_MODELS_LISTEN) in .env still win when they are set, as an explicit override
     (llm._get_models); with none set, this module decides.

The model is not part of any stage fingerprint, so a model change re-runs nothing.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from . import tunables
from .logging_setup import get_logger

file_log = get_logger("modelchain")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_PATH = PROJECT_ROOT / "data" / "models_cache.json"
_HEALTH_PATH = PROJECT_ROOT / "data" / "model_health.json"
_lock = threading.Lock()
_refresh_failed_at = 0.0                  # when Google last could not be asked (this process): not asked again for a few minutes
_RETRY_AFTER_FAILURE_SECONDS = 300.0


def _version(name: str):
    """(major, minor) of a stable name of the configured family, None for anything else."""
    family = tunables.get_text("models", "family")
    m = re.match(rf"^gemini-(\d+)(?:\.(\d+))?-{re.escape(family)}$", name or "")
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 -- a missing or broken file is an empty state
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _list_remote() -> list[str]:
    """The one place the chain asks Google (tests replace it, so a test never reaches the network)."""
    return _ask_google()


def _ask_google() -> list[str]:
    """Names of the models the key can call for generateContent. Main key first, the backup key when the main one cannot list."""
    from google import genai
    last_error = None
    for env_name in ("GEMINI_API_KEY", "GEMINI_API_KEY_BACKUP"):
        key = os.environ.get(env_name)
        if not key:
            continue
        try:
            client = genai.Client(api_key=key)             # kept in a variable: a client nobody holds is closed at once, and the list is lazy
            names = [(m.name or "").replace("models/", "") for m in client.models.list()
                     if "generateContent" in (m.supported_actions or [])]
            if names:
                return names
        except Exception as e:  # noqa: BLE001
            last_error = e
            file_log.warning(f"models.list failed on {env_name}: {type(e).__name__}")
    if last_error is not None:
        raise last_error
    return []


def candidates() -> list[str]:
    """Every stable model of the configured family the key can call, newest first (not cut to the chain length)."""
    global _refresh_failed_at
    now = time.time()
    cache = _read(_CACHE_PATH)
    names = cache.get("names") if isinstance(cache.get("names"), list) else None
    fresh = names is not None and now - float(cache.get("fetched_at", 0) or 0) < tunables.get("models", "refresh_hours") * 3600
    if not fresh and now - _refresh_failed_at >= _RETRY_AFTER_FAILURE_SECONDS:
        try:
            listed = _list_remote()
            if listed:
                names = listed
                _write(_CACHE_PATH, {"fetched_at": now, "names": sorted(set(listed))})
        except Exception as e:  # noqa: BLE001 -- the stale copy (or the alias) carries on
            _refresh_failed_at = now
            file_log.warning(f"model list not refreshed, using {'the stale copy' if names else 'the last resort'}: {type(e).__name__}")
    return sorted({n for n in (names or []) if _version(n)}, key=_version, reverse=True)


def chain() -> list[str]:
    """The models to try, in order: the newest `models.chain_length` models of the family, newest first, then the exact names written in
    `models.after` (config.toml), in the order written (design decision 2026-11-24: gemini-4-hydrogen, then gemini-3.8-flash, then
    gemini-3.6-flash). The order is fixed: nothing a model does, and nothing recorded about it, ever moves it."""
    found = candidates()[: int(tunables.get("models", "chain_length"))] or [tunables.get_text("models", "last_resort")]
    return found + [m for m in tunables.get_words("models", "after") if m not in found]


def record(model: str, outcome: str) -> None:
    """Report one attempt: outcome is "ok" or "fail". A record only: it never changes the order of the chain. Never raises:
    bookkeeping must not break a call."""
    try:
        with _lock:
            data = _read(_HEALTH_PATH)
            models = data.setdefault("models", {})
            entry = models.setdefault(model, {"failures": 0})
            entry.pop("cooldown_until", None)                  # the old cool-down is gone: nothing moves a model any more
            stamp = datetime.now().isoformat(timespec="seconds")
            if outcome == "ok":
                if entry.get("failures"):
                    file_log.info(f"{model} answered again: the failures in a row were cleared")
                entry.update(failures=0, last_ok=stamp)
            else:
                entry["failures"] = int(entry.get("failures", 0) or 0) + 1
                entry["last_failure"] = stamp
            _write(_HEALTH_PATH, data)
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"model health not recorded: {type(e).__name__}: {e}")
