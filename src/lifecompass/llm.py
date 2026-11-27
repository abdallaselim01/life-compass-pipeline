"""
Every model call the pipeline makes goes through here.

Two layers:
  * the infrastructure: the Gemini fallback chain (newest model first: discovered from Google and ordered by what works in
    modelchain.py, or pinned by GEMINI_MODELS in .env), per-model timeouts, daily-quota bookkeeping on
    disk, and strict/lenient structured parsing.
    It was hardened through real incidents (a misleading retryDelay causing a
    ~22-minute run; a model that was merely slow blowing any time budget), so
    its retry/timeout rules are documented on `_generate_with_fallback`.
  * a small typed surface over it for the stages:
                generate_audio_text(...)  -> (text, model_name)   the model hears audio
        generate_json(...)        -> (parsed_schema, model_name)
    All raise StageSkippedError when no model could produce a usable answer,
    so the orchestrator queues the video for a retry instead of storing
    something degraded.

Tests replace the network with `set_test_backend(...)`.
"""
from __future__ import annotations

import os
import re
import hashlib
import json
import time
import traceback
import threading
import typing
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from google import genai
from google.genai import types
from pydantic import BaseModel

from . import modelchain, tunables
from .logging_setup import get_logger
from .runlog import detail as _detail, block as _block

file_log = get_logger("llm")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class StageSkippedError(RuntimeError):
    """Raised by a pipeline stage when the stage
    could not actually run to completion (every model in the fallback
    chain exhausted/unavailable, or the response was unusable) -- unlike
    the old non-fatal behavior (silently returning the input unchanged),
    this propagates up so pipeline.py can refuse to publish a result that
    skipped a mandatory review stage, per an explicit design decision
    2026-09-09: "I don't want anything reaching me that didn't go through
    every stage it was designed to go through." The
    image-matching/captioning passes are NOT covered by this -- those
    remain non-fatal/supplementary, only the listen and extract stages are.

    Since 2026-09-29 match_principles_to_images and caption_standalone_image raise it on a total model outage or an
    unparseable reply (they used to return {} / "" and the images silently never got linked or captioned). It is
    NON-BLOCKING for the video itself: pipeline.py catches it, publishes normally, and queues only the missing
    supplementary work for retry (pending_retries stage "supplementary": image-to-principle matching, standalone
    image captions).
    """


def _get_models(context: str | None = None) -> list[str]:
    """The Gemini model chain, in the order they are tried.

    Pinned in .env (GEMINI_MODELS) it is exactly that list. With nothing pinned (the normal case since 2026-11-17, Fully Dynamic)
    it is the chain modelchain.py discovers from Google and orders by what has been working.

    The listen stage may have its own order (GEMINI_MODELS_LISTEN): the
    stage is read from the call's `context` prefix ("listen[1/5]").
    Anything else -- and a listen without its own setting -- uses
    GEMINI_MODELS. Order is the only thing a stage can change; quota
    bookkeeping and fallback are per model exactly as before.
    """
    raw = None
    if context:
        stage = re.split(r"[\[\s]", context, maxsplit=1)[0].strip().upper()
        if stage == "LISTEN":
            raw = (os.environ.get(f"GEMINI_MODELS_{stage}") or "").strip()
    if not raw:
        raw = (os.environ.get("GEMINI_MODELS") or "").strip()
    if raw:                                              # an explicit pin in .env wins
        return [m.strip() for m in raw.split(",") if m.strip()]
    return modelchain.chain()                            # nothing pinned: discovered and health-ordered (modelchain.py)


# --- Timeout-and-fallback machinery -----------------------------------------
# Redesigned per explicit user instruction after two real, observed
# problems: (1) treating a 429 as "wait and retry the same model" caused a
# ~22-minute run when Google's suggested retryDelay was misleading for a
# fully daily-exhausted quota, and (2) a model that's merely SLOW to reply
# (gemini-3.7-flash took 4+ minutes to eventually return a 503 under high
# demand) blew past any reasonable budget even without retrying it at all,
# because we were passively waiting for Google's own network call to
# return before we could even classify the failure.
#
# The rule is simple and uniform for every Gemini call in this module (design decision 2026-10-14):
#   - The chain is walked ONCE, strongest model first, and each model is asked
#     exactly ONE time. A model that returns an error of any kind (quota, 503,
#     400, 404 ...) does not answer within the active timeout, or answers with nothing usable (empty / unparseable), is left at once
#     and the next model is tried -- never a second attempt on the same model.
#   - Only when the WHOLE chain has failed is anything repeated: the stage is
#     queued (StageSkippedError -> store.pending_retries) and
#     scripts/retry_pending.py runs the chain again from its first model.
#   - There is no fallback outside GEMINI_MODELS.
# A hard timeout is enforced ourselves (via a worker thread) rather than
# relying on the SDK's own timeout behavior, since the observed 4-minute
# stall proved the default behavior cannot be trusted to fail fast.
#
# IMPORTANT: the timeout is a CEILING, not a fixed wait -- _call_with_timeout
# below uses a daemon thread's join(timeout=X), which returns the instant
# the model actually responds, whatever that takes (confirmed 2026-09-18 in
# a real run: gemini-3.6-flash answered extract_principles at 75.5s, well
# under the 120s ceiling, and the pipeline moved on immediately at that
# point, not at 120s). Raising a timeout only widens how long we're willing
# to wait before giving up on a genuinely slow/stuck attempt; it never
# forces the call to wait its full duration when the model replies sooner.

_MODEL_TIMEOUT_SECONDS = 900.0   # generous safety net (15 min) -- not a "thinking time" cap. Raised
                                  # 2026-09-20 after review_quality on a real 30-item batch legitimately
                                  # took 77s and succeeded -- the old 45s cap was killing genuine,
                                  # working responses across every model in the chain, not catching real
                                  # hangs. This only protects against a TRUE dead connection; it never
                                  # forces a wait once the model actually answers (see comment above).

# extract_principles gets its own, longer timeout -- added 2026-09-17 after
# direct REST tests (bypassing this whole module) proved the 45s default was
# too tight specifically for THIS call, not a Google outage: a minimal
# "reply OK" request on this account's key succeeded on gemini-3.8-flash in
# 8.8s, gemini-3.6-flash in 1.8s, and gemini-3.5-flash in 19.5s (only
# gemini-3.7-flash genuinely 503'd even on a minimal call -- a real,
# separate Google-side issue for that one model). The SAME key, SAME
# models, but with extract_principles' actual request shape (the ~10K-char
# extract_prompt.txt as system_instruction, a full ~20K+-char
# transcript, response_schema-constrained JSON output, and dynamic
# thinking_budget=-1) reliably exceeded 25+ seconds without responding at
# all. That combination -- long structured-output generation under a big
# context -- is inherently far slower than a bare text reply, and is
# exactly the kind of expensive request Google is likeliest to delay or
# shed first under any real demand pressure, even while light calls sail
# through. recall_check has a very similar shape (also full transcript +
# response_schema + dynamic thinking) and may deserve the same treatment
# later, but per an explicit scope decision this change is extract_
# principles only for now. Confirmed working as intended 2026-09-18: a
# real run had gemini-3.6-flash answer at 75.5s, which the old 45s cap
# would have killed but the new 120s cap accepted normally.


def _is_daily_quota_exhausted(error: Exception) -> bool:
    """True when the failure is specifically a per-DAY quota limit. Used
    only to decide whether to remember this model as exhausted for the
    rest of THIS video's processing (see _exhausted_models_this_run) --
    it no longer affects retry behavior directly, since every error
    response (this one included) now moves to the next model immediately
    regardless of type. Detected via Google's own quotaId naming in the
    error's details: any violated quota whose id contains "PerDay" means
    this specific model is done until tomorrow."""
    return "PerDay" in str(error)


# Models confirmed daily-exhausted DURING THE CURRENT VIDEO'S PROCESSING.
# One video's pipeline run makes several separate Gemini calls (listen,
# extract principles, match images to principles) -- without
# this, each call independently re-discovers "oh, this model is exhausted
# today" from scratch, wasting a real API round-trip per call just to
# re-learn a fact already established earlier in the SAME run.
# pipeline.py calls reset_exhausted_models() once at the start of each
# video so this doesn't leak stale state into the NEXT video/day, when the
# quota may have reset.
_exhausted_models_this_run: set[str] = set()

# Disk-persisted per-model daily exhaustion, for this documented reason: scripts/retry_pending.py starts a brand-new
# Python process every ~5 minutes, so an in-memory-only flag is wiped
# before the next scheduled run and each fresh process re-discovers "oh,
# this model is exhausted today" from scratch -- burning a real API
# round-trip (and, for a model that is simultaneously 503-slow rather
# than a clean fast 429, up to its whole timeout of
# wasted wall-clock time) on every single 5-minute cycle for the rest of
# the day. Added 2026-09-19 after exactly this pattern was observed live.
# No new model is introduced by this -- it only remembers, across
# process restarts, what _is_daily_quota_exhausted already established
# about the SAME models already in GEMINI_MODELS.
_GEMINI_STATE_PATH = PROJECT_ROOT / "data" / "gemini_exhausted_state.json"


# How long a model flagged "daily quota used up" is skipped before it is asked again.
# Google puts a reset delay in that 429, but it is only a hint: on 2026-10-15 a model answered
# hours before the delay it had announced, and trusting the hint blindly kept a working model
# blocked for the whole day while the retry queue spent all its attempts without calling anything.
_EXHAUSTION_RECHECK_SECONDS = tunables.get("waits", "exhaustion_recheck_seconds")   # config.toml [waits]

_api_calls = [0]

# Stall detection for streamed calls. A streamed reply keeps sending small pieces while the model works,
# so "no piece for a while" means the call is stuck, while a slow reply that keeps sending is left alone.
# The first piece may take long (the model listens/thinks before it writes), later gaps are short.
# _MODEL_TIMEOUT_SECONDS stays as the outer safety net.
# 2026-10-19: 600 -> 360. 3.8's thinking is silent (thinking_budget=-1, no thought summaries), so this is the longest a
# healthy call may stay silent. Measured over 48 healthy 3.8 calls: median 184s, max 304s (one call above 300s). Of the 8
# calls that stayed silent past 304s and were left to run (7 cut at 600s, 1 at 1500s), none ever answered.
# 2026-11-05: 360 -> 900 (= _MODEL_TIMEOUT_SECONDS). Every request now asks for thinking_level HIGH (design decision "all
# settings Max"), so a healthy call stays silent longer than the 48 calls measured above (those ran with thinking_budget=-1)
# and 360 would cut healthy calls. The cost: a call that really hangs (8 of those 56) now holds the chain up to 900s before the
# next model is tried. The first-piece time of every call is in the run log, so the limit can be tuned on real numbers.
_STREAM_FIRST_CHUNK_SECONDS = 900.0
_STREAM_STALL_SECONDS = tunables.get("waits", "stream_stall_seconds")   # config.toml [waits]
_beat_local = threading.local()

# Set when any model in the current call hit a quota limit (429 / daily exhausted); read by the key failover.


class _PerThreadFlag:
    """Same `flag[0]` interface as a one-item list, but every thread has its own value, so one thread's failover decision
    is never reset by another thread's call."""

    def __init__(self):
        self._local = threading.local()

    def __getitem__(self, _i):
        return getattr(self._local, "value", False)

    def __setitem__(self, _i, value):
        self._local.value = value


_quota_seen = _PerThreadFlag()


def api_calls_made() -> int:
    """How many real requests this process has sent to a model so far. The retry queue uses it to
    tell an attempt that really tried (and failed) from one that only found every model flagged
    as out of quota, which spent nothing and must not use up one of the limited attempts."""
    return _api_calls[0]


def _retry_delay_seconds(error) -> float | None:
    """The reset delay Google put in a 429 (RetryInfo.retryDelay / "retry in Ns"), or None."""
    text = str(error)
    for pattern in (r"'retryDelay': '(\d+(?:\.\d+)?)s'", r"retry in (\d+(?:\.\d+)?)s"):
        found = re.search(pattern, text)
        if found:
            return float(found.group(1))
    return None


def _read_exhausted_until() -> dict:
    try:
        data = json.loads(_GEMINI_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    until = data.get("until") if isinstance(data, dict) else None
    if not isinstance(until, dict):
        return {}
    now = time.time()
    return {m: t for m, t in until.items() if isinstance(t, (int, float)) and t > now}


def _load_gemini_exhausted_today() -> set[str]:
    """Models flagged as out of daily quota RIGHT NOW (read back from disk so a fresh process
    does not have to re-learn it the hard way). A flag lasts at most _EXHAUSTION_RECHECK_SECONDS."""
    return set(_read_exhausted_until())


def _mark_gemini_exhausted_on_disk(model_name: str, retry_seconds: float | None = None) -> None:
    wait = _EXHAUSTION_RECHECK_SECONDS if retry_seconds is None else min(retry_seconds, _EXHAUSTION_RECHECK_SECONDS)
    until = _read_exhausted_until()
    until[model_name] = time.time() + max(wait, 1.0)
    try:
        _GEMINI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GEMINI_STATE_PATH.write_text(json.dumps({"until": until}), encoding="utf-8")
    except OSError as e:
        file_log.warning(f"_mark_gemini_exhausted_on_disk: failed to persist state for {model_name}: {e}")


def reset_exhausted_models() -> None:
    """Call once at the very start of processing a new video (see
    pipeline.process_video) so a model's exhaustion, once confirmed, is
    remembered for every remaining Gemini call in THIS run, without ever
    carrying over incorrectly into a later run/day where the quota may
    have reset."""
    _exhausted_models_this_run.clear()
    _exhausted_models_this_run.update(_load_gemini_exhausted_today())


def _call_with_timeout(build_call: Callable[[str], object], model_name: str, timeout: float):
    """Runs build_call(model_name) on a worker thread and gives up after
    `timeout` seconds if it hasn't returned, raising TimeoutError.

    Rewritten 2026-09-18 to use a plain `threading.Thread(daemon=True)`
    instead of `concurrent.futures.ThreadPoolExecutor` -- a REAL incident
    proved this distinction matters, not just style: ThreadPoolExecutor's
    worker threads are NOT daemon threads, so when a call times out and
    the underlying network request never returns on its own (observed:
    a model request that simply hangs, with no exception
    and no response, for hours), that worker thread keeps the WHOLE
    PYTHON PROCESS alive indefinitely, even after every video has
    finished processing and the script's own code has returned. This
    doesn't matter for the always-on listener process, but it silently
    broke scripts/retry_pending.py: Task Scheduler's "IgnoreNew"
    multiple-instances policy saw that old, technically-still-alive
    process and refused to start any new one for roughly TWO HOURS,
    even though the script's own logic had already finished and logged
    "run finished" -- a real self-inflicted incident (2026-09-18) that
    looked like retries had silently stopped, when actually a single
    orphaned thread from an earlier timeout was blocking every
    subsequent scheduled attempt from ever starting. A daemon thread
    does NOT keep the process alive -- once main() returns, the whole
    process exits immediately regardless of what any daemon thread is
    still doing, so this failure mode is now structurally impossible.

    Python still can't forcibly kill the thread itself (it keeps running
    in the background, same as before, until the underlying call
    eventually returns or the process exits) -- that part of the
    tradeoff is unchanged and still fine for this tool's usage pattern."""
    result_box: dict = {}
    beat = {"watching": False, "last": time.time(), "pulses": 0}

    def _run() -> None:
        _beat_local.beat = beat
        try:
            result_box["value"] = build_call(model_name)
        except BaseException as e:  # noqa: BLE001 -- re-raised on the calling thread below, whatever it is
            result_box["error"] = e

    worker = threading.Thread(target=_run, daemon=True, name=f"call-timeout-{model_name}")
    worker.start()
    started = time.time()
    while True:
        worker.join(1.0)
        if not worker.is_alive():
            break
        now = time.time()
        if now - started >= timeout:
            raise TimeoutError(f"{model_name} did not respond within {timeout:.0f}s")
        if beat["watching"]:
            first_piece = max(_STREAM_FIRST_CHUNK_SECONDS, timeout) if timeout > _MODEL_TIMEOUT_SECONDS else _STREAM_FIRST_CHUNK_SECONDS
            limit = _STREAM_STALL_SECONDS if beat["pulses"] else first_piece       # a wait raised above the ceiling (long recording) lifts it too
            if now - beat["last"] > limit:
                what = "stopped sending data" if beat["pulses"] else "sent nothing at all"
                raise TimeoutError(f"{model_name} {what} for {limit:.0f}s")
    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]


# ---- how long to wait for a model (design decision 2026-11-07) -------------------------------------------------------------
# Every call used to wait up to _MODEL_TIMEOUT_SECONDS (900 s). On 2026-11-06 three calls to gemini-3.8-flash hung for the full
# 900 s (45 minutes lost) and none of them ever answered; gemini-3.6-flash answered the same kinds of call in 7.6 to 35 s. So the
# wait limit of each KIND of call is now learned from the calls that really succeeded: the slowest success of that kind ever
# recorded (any model) is kept in data/call_timings.json, and the limit is
#     min(ceiling, max(_TIMEOUT_FLOOR, _TIMEOUT_MARGIN x slowest success)).
# A kind with no recorded success keeps the old ceiling. Only successes are recorded, so a limit only moves up when a slower
# call succeeds; a call cut by the limit is not a success and never lowers it. One attempt per model, never a second attempt on
# the same model: the quota is not spent on anything new.
_TIMINGS_PATH = PROJECT_ROOT / "data" / "call_timings.json"
# Design decision 2026-11-08: the margin is 2 x the slowest success (it was 4 x), for every kind of call.
_TIMEOUT_MARGIN = tunables.get("waits", "timeout_margin")   # config.toml [waits]
_TIMEOUT_FLOOR = tunables.get("waits", "timeout_floor_seconds")
# The big, quality-critical calls never get a limit below 450 s (design decision 2026-11-07): gemini-3.8-flash needed 204 s for extract,
# 304 s was the slowest healthy call of 48 measured, and none of the 8 calls that stayed silent past 304 s ever answered.
_LONG_WAIT_KINDS = ("listen", "extract")
_LONG_WAIT_FLOOR = tunables.get("waits", "long_wait_floor_seconds")
# Design decision 2026-11-24: the wait of a listen or extract call also grows with the length of the recording, because the model's work
# grows with it (all the limits above were measured on one 34-minute video; on 2026-11-19 an extract call on it was cut at 450 s and
# its first piece would have come at 534 s). The least wait is minutes x this many seconds, and the 900 s ceiling rises with it.
_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE = tunables.get("waits", "long_wait_seconds_per_audio_minute")   # config.toml [waits]; 0 = off
_audio_local = threading.local()
_timings_lock = threading.Lock()


def call_kind(context: str) -> str:
    """The kind of a call: its context without the bracket / parenthesis suffix ("listen[2/7]" -> "listen")."""
    return re.split(r"[\[(]", context or "", maxsplit=1)[0].strip() or "unknown"


def _read_timings() -> dict:
    """{kind: {"max": slowest success in seconds, "n": successes, "updated": ...}}; {} when the file is missing or unreadable."""
    try:
        data = json.loads(_TIMINGS_PATH.read_text(encoding="utf-8"))
        kinds = data.get("kinds", {})
        return kinds if isinstance(kinds, dict) else {}
    except Exception:  # noqa: BLE001 -- no usable record means "no data yet", never a failed call
        return {}


def record_success_time(context: str, seconds: float) -> None:
    """Remembers how long a call that really succeeded took (only the slowest per kind is kept). Never raises."""
    kind = call_kind(context)
    try:
        with _timings_lock:
            kinds = _read_timings()
            old = kinds.get(kind) or {"max": 0.0, "n": 0}
            kinds[kind] = {"max": round(max(float(old.get("max", 0.0)), float(seconds)), 1),
                           "n": int(old.get("n", 0)) + 1,
                           "updated": datetime.now().isoformat(timespec="seconds")}
            _TIMINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _TIMINGS_PATH.with_name(_TIMINGS_PATH.name + ".tmp")
            tmp.write_text(json.dumps({"kinds": kinds}, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, _TIMINGS_PATH)
    except Exception as e:  # noqa: BLE001
        file_log.warning(f"could not record the call time for {kind}: {e}")


def adaptive_timeout(context: str, ceiling: float) -> float:
    """How long to wait for a call of this kind: min(ceiling, max(floor, margin x slowest recorded success)); the ceiling
    itself when no success of this kind is recorded (or the record cannot be read)."""
    try:
        kind = call_kind(context)
        entry = _read_timings().get(kind)
        if not entry:
            return ceiling
        floor = _LONG_WAIT_FLOOR if kind in _LONG_WAIT_KINDS else _TIMEOUT_FLOOR
        return min(ceiling, max(floor, _TIMEOUT_MARGIN * float(entry["max"])))
    except Exception:  # noqa: BLE001
        return ceiling


def set_audio_minutes(minutes) -> None:
    """Tells this thread how long (in minutes) the recording it is processing is; None (or 0) = unknown. The pipeline sets it when a
    run starts and the listener's transcription step sets it from the real file, so scaled_long_wait can size the waits."""
    _audio_local.minutes = float(minutes) if minutes else None


def scaled_long_wait(context: str) -> float:
    """The least wait a listen or extract call gets for the recording being processed: its minutes x
    waits.long_wait_seconds_per_audio_minute. 0.0 for any other kind of call, for an unknown length, or when the setting is 0."""
    minutes = getattr(_audio_local, "minutes", None)
    if not minutes or call_kind(context) not in _LONG_WAIT_KINDS:
        return 0.0
    return minutes * _LONG_WAIT_SECONDS_PER_AUDIO_MINUTE


def _generate_with_fallback(
    models: list[str],
    build_call: Callable[[str], object],
    log=print,
    context: str = "",
    timeout: float = _MODEL_TIMEOUT_SECONDS,
    key_tag: str = "",
):
    """Walks the chain once, strongest model first, and asks each model exactly
    ONE time. A model that errors (any reason), does not answer within
    `timeout`, or answers with nothing usable (empty / unparseable) is left
    at once for the next one; a model already known
    daily-exhausted is skipped with no API call spent.

    Returns (response, model_name) from the first model that answers. If every
    model failed, raises the last captured error: the caller turns that into
    StageSkippedError, the video is queued, and the whole chain is run again
    from its first model on the next scheduled retry (scripts/retry_pending.py).
    """
    last_error: Exception | None = None
    for model_name in models:
        state_key = key_tag + model_name   # the backup key has its own quota, so its own flag
        if state_key in _exhausted_models_this_run:
            log(f"    ({context}: {model_name} already known daily-exhausted this run, skipping -- no API call spent)")
            file_log.debug(f"{context}: model={model_name} skipped, already confirmed daily-exhausted earlier this run")
            _detail(f"ATTEMPT SKIPPED [{context}] model={state_key}: already known daily-exhausted this run (no call made)")
            _quota_seen[0] = True
            continue
        _api_calls[0] += 1
        _detail(f"ATTEMPT [{context}] model={state_key} call_number_in_process={_api_calls[0]} "
                f"at {datetime.now().isoformat(timespec='milliseconds')}")
        scaled = scaled_long_wait(context)                   # a long recording gets a longer wait, never a shorter one
        ceiling = max(timeout, scaled)
        limit = max(adaptive_timeout(context, ceiling), scaled)
        _detail(f"WAIT LIMIT [{context}] kind={call_kind(context)} limit={limit:.0f}s (ceiling {ceiling:.0f}s, "
                f"{scaled:.0f}s from the length of the recording)")
        started = time.time()
        try:
            response = _call_with_timeout(build_call, model_name, limit)
            took = time.time() - started
            record_success_time(context, took)
            _detail(f"ATTEMPT OK [{context}] model={state_key} took={took:.1f}s")
            modelchain.record(model_name, "ok")
            return response, model_name
        except TimeoutError as e:
            last_error = e
            _detail(f"ATTEMPT FAILED [{context}] model={state_key}: TimeoutError: {e}")
            log(f"    x {context}: {e} -- moving to next model")
            file_log.warning(f"{context}: {e}, moving to next model")
            modelchain.record(model_name, "fail")
        except Exception as e:  # noqa: BLE001 -- any error response moves to the next model
            last_error = e
            _detail(f"ATTEMPT FAILED [{context}] model={state_key}: {type(e).__name__}: {e}")
            _block(f"TRACEBACK [{context}] model={state_key}", traceback.format_exc())
            if _is_daily_quota_exhausted(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                _quota_seen[0] = True
            if _is_daily_quota_exhausted(e):
                _exhausted_models_this_run.add(state_key)
                _mark_gemini_exhausted_on_disk(state_key, _retry_delay_seconds(e))
            if not (_is_daily_quota_exhausted(e) or "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                modelchain.record(model_name, "fail")      # a quota limit says nothing about the model itself
            log(f"    x {context} failed on {model_name}: {e} -- moving to next model")
            file_log.warning(f"{context}: model={model_name} failed, moving to next model: {e}")
            file_log.debug(f"{context}: full traceback for above failure on {model_name}:", exc_info=True)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{context}: no models were attempted (all already known exhausted, or empty model list)")


def _lenient_parse_items(data: dict, schema: type[BaseModel]) -> BaseModel | None:
    """schema.model_validate(data) is STRICT and ALL-OR-NOTHING: a single
    malformed item anywhere in a large `items`/`matches`-style array
    rejects the entire response, discarding every other item that parsed
    perfectly fine along with it. Added 2026-09-17 after a real failure:
    a weaker model returned 15 well-formed items and ONE with a typo'd key
    (`"principile"` instead of `"principle"`) -- the whole 60-second,
    quota-costing request was thrown away over that single item.

    Tries the strict parse first (covers every schema used in this
    module, whether or not it has an `items` list). Only on failure does
    it fall back to opening `data["items"]` (if present) and validating
    each entry individually, keeping only the ones that succeed and
    silently dropping the rest -- a partial result is far better than
    none for a fallback this expensive to retry. Schemas without an
    `items` list (ImageMatchResult, CaptionResult) simply
    return None here if the strict parse fails, same as before -- this keeps the
    function honest about what it can and can't recover."""
    # 2026-10-03: the quality-review/recall/correction prompts say "return a
    # JSON array" (and an EMPTY array is now a valid answer), so a model may
    # reply with a bare list instead of {"items": [...]}. Wrap it, otherwise a
    # perfectly valid empty/bare-list answer would be read as unparseable and
    # retried forever.
    if isinstance(data, list) and "items" in schema.model_fields:
        data = {"items": data}
    # a reply wrapped under another key ({"principles": [...]}) is read too, when that is the only list in it
    # (never for a dict that carries schema keywords: the echoed schema has one list, "required": ["items"], which is not an answer)
    if (isinstance(data, dict) and "items" in schema.model_fields and "items" not in data
            and not any(k in data for k in ("$defs", "$schema", "properties", "required", "title", "type"))):
        only_lists = [v for v in data.values() if isinstance(v, list)]
        if len(only_lists) == 1:
            data = {"items": only_lists[0]}

    try:
        return schema.model_validate(data)
    except Exception:
        pass

    items_field = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items_field, list):
        return None

    fields = schema.model_fields
    if "items" not in fields:
        return None
    item_args = typing.get_args(fields["items"].annotation)
    if not item_args:
        return None
    item_cls = item_args[0]

    valid_items = []
    for raw_item in items_field:
        try:
            valid_items.append(item_cls.model_validate(raw_item))
        except Exception as e:
            # never silently: the whole item is in the run log (design decision 2026-11-07, no principle is lost unseen)
            _detail(f"PARSE: item dropped, it does not fit {item_cls.__name__} ({type(e).__name__}): {raw_item!r}")
            continue  # drop only this one malformed item, keep the rest
    if not valid_items:
        return None
    return schema.model_validate({"items": valid_items})

_SCHEMA_ENABLED = os.environ.get("GEMINI_USE_STRUCTURED_OUTPUT", "0") == "1"
# Switch, not deletion -- added 2026-09-27 with hard evidence: a controlled,
# interleaved A/B test (same model, same prompt, same thinking_budget=-1,
# only response_schema toggled) measured 1/4 success WITH the schema vs 4/4
# WITHOUT it. response_schema forces token-level grammar-constrained
# decoding (see the 2026-09-05 comment on extract_principles's build_call),
# which is real extra server-side load Google appears to shed first under
# any demand. Defaults OFF given that evidence; flip
# GEMINI_USE_STRUCTURED_OUTPUT=1 in .env to restore the old guaranteed-
# shape behavior if this ever needs to be A/B'd again or the tradeoff
# changes. Reading it once at import time (not per-call) is deliberate:
# this is an operator decision, not something that should silently drift
# mid-run if the env var changes while a long pipeline is already going.


# Maximum-quality request settings (design decision 2026-11-05: "all settings Max"), the same for every Gemini call.
# gemini-3.6-flash and gemini-3.8-flash both take thinking_level HIGH and a 65,536-token output limit (Google's model pages).
_MAX_OUTPUT_TOKENS = 65536


# The API settings each stage sends (design decision 2026-11-06: a change of a stage's settings redoes that stage and every stage
# after it, exactly like a change of its prompt). ONE source of truth: request_settings(stage) is a plain dict; the config that
# is sent is built from it (max_quality_config_kwargs, media_resolution_level) and the stage's settings fingerprint is a hash of
# the same dict (settings_fp), so what is fingerprinted and what is sent cannot drift apart.
STAGES = ("listen", "extract", "match", "caption")
_JSON_STAGES = ("extract", "match", "caption")      # stages whose answer is JSON
_IMAGE_STAGES = ("match", "caption")          # stages whose request carries the picture itself (2026-11-10)


def request_settings(stage: str | None = None) -> dict:
    """The API settings of one stage, as plain values. Everything in here changes what the model answers. Not in here, on
    purpose: the model name (chosen at run time by the fallback chain), the timeouts, the key and the pause between attempts
    -- they protect the run, they do not change the answer. temperature / top_p / top_k are not sent at all (deprecated)."""
    if stage is not None and stage not in STAGES:      # a misspelt stage name must not quietly get the generic settings
        raise ValueError(f"unknown stage {stage!r}; the stages are {', '.join(STAGES)}")
    settings = {"thinking_level": "HIGH", "max_output_tokens": _MAX_OUTPUT_TOKENS}
    if stage in _IMAGE_STAGES:
        settings["media_resolution"] = "ULTRA_HIGH"
    if stage in _JSON_STAGES:
        settings["structured_output"] = bool(_SCHEMA_ENABLED)
    return settings


def settings_fp(stage: str | None = None) -> str:
    """SHA-256[:16] of the stage's settings dict (keys sorted)."""
    return _sha16(json.dumps(request_settings(stage), sort_keys=True))


def max_quality_config_kwargs(stage: str | None = None) -> dict:
    """GenerateContentConfig settings built from request_settings(stage): thinking_level and the output limit (it also counts
    the thinking tokens). No temperature / top_p / top_k: Google deprecated them on these models (ignored today, an HTTP 400
    on future generations). No thinking_budget: legacy, replaced by thinking_level."""
    s = request_settings(stage)
    return {
        "thinking_config": types.ThinkingConfig(thinking_level=getattr(types.ThinkingLevel, s["thinking_level"])),
        "max_output_tokens": s["max_output_tokens"],
    }


def media_resolution_level(stage: str):
    """The media_resolution a stage sends for its images, from request_settings(stage)."""
    return getattr(types.PartMediaResolutionLevel, "MEDIA_RESOLUTION_" + request_settings(stage)["media_resolution"])


def _gemini_config_kwargs(schema: type[BaseModel]) -> dict:
    """What to merge into GenerateContentConfig(...) for a Gemini native
    call. WITH the switch on: forces the SDK's typed response_schema
    (guaranteed-shape JSON, at the real server-load cost documented
    above). OFF: empty dict -- the prompt itself must state the reply
    shape (prompts/extract_prompt.txt does), and the result is read back
    with _parse_structured (below) rather than response.parsed."""
    if not _SCHEMA_ENABLED:
        return {}
    return {"response_mime_type": "application/json", "response_schema": schema}


def _recover_after_schema_echo(raw: str):
    """A model may begin its answer by repeating the JSON schema it was given and add the answer after it, e.g.
    {"$defs": ..., "type": "object"}, "items": [...]} -- not valid JSON as a whole (seen on gemini-3.8-flash at extract[1] on
    2026-11-07: 22 good principles were thrown away). When the text starts with such a schema echo, returns the answer that follows
    it as a dict; None for anything else (a reply that is not a schema echo is rejected exactly as before)."""
    try:
        first, end = json.JSONDecoder().raw_decode(raw)
    except Exception:  # noqa: BLE001
        return None
    if not (isinstance(first, dict) and ("$defs" in first or "properties" in first)):
        return None
    rest = raw[end:].lstrip().lstrip(",").lstrip()
    if not rest:
        return None
    try:
        data = json.loads("{" + rest)
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _salvage_items(raw: str, schema: type[BaseModel]):
    """Last resort for a schema with an `items` list: reads EVERY finished item object found anywhere in the text -- inside a
    reply cut off in the middle of the list, among prose, after a schema echo, under a wrong wrapper key -- and returns them in the
    order they appear (an identical object repeated is kept once). Only objects that fit the item model are taken (the echoed
    schema itself never does), so nothing is invented. None when the schema has no `items` list or nothing fits."""
    fields = schema.model_fields
    if "items" not in fields:
        return None
    args = typing.get_args(fields["items"].annotation)
    if not args:
        return None
    item_cls = args[0]
    decoder = json.JSONDecoder()
    found, seen, i = [], set(), 0
    while True:
        i = raw.find("{", i)
        if i < 0:
            break
        try:
            obj, end = decoder.raw_decode(raw, i)
        except Exception:  # noqa: BLE001 -- an unfinished or broken object: look inside it from the next character
            i += 1
            continue
        try:
            item = item_cls.model_validate(obj)
        except Exception:  # noqa: BLE001 -- a wrapper or the schema echo: look at what is inside it
            i += 1
            continue
        key = item.model_dump_json()
        if key not in seen:
            seen.add(key)
            found.append(item)
        i = end
    return schema.model_validate({"items": found}) if found else None


def _parse_structured(response, schema: type[BaseModel]) -> BaseModel | None:
    """Reads a Gemini native response back into `schema`, working the
    same way regardless of whether _SCHEMA_ENABLED forced the shape or
    not. response.parsed only ever populates when response_schema was
    actually sent; otherwise (or if the SDK's own parse failed anyway)
    this falls back to the same lenient, partial-recovery JSON parse
    relied on whenever the schema is not sent -- see
    _lenient_parse_items's docstring above for why strict
    all-or-nothing parsing is the wrong default for a many-item batch."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return parsed
    raw = (getattr(response, "text", None) or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except Exception:
        data = _recover_after_schema_echo(raw)
    result = _lenient_parse_items(data, schema) if data is not None else None
    if result is None:
        result = _salvage_items(raw, schema)      # last resort: every finished item found anywhere in the text
    return result


_TEST_JSON = None   # callable(prompt, schema, system, context) -> schema instance
_TEST_AUDIO = None  # callable(audio_bytes, prompt, system, context) -> str


def set_test_backend(json=None, audio=None) -> None:
    """Install (or, with no arguments, remove) fake model backends."""
    global _TEST_JSON, _TEST_AUDIO
    _TEST_JSON, _TEST_AUDIO = json, audio


def _client(backup: bool = False):
    name = "GEMINI_API_KEY_BACKUP" if backup else "GEMINI_API_KEY"
    api_key = os.environ.get(name)
    if not api_key:
        raise RuntimeError(f"{name} is not set (check your .env)")
    return genai.Client(api_key=api_key)


def _describe_empty(response) -> str:
    """Why an answer came back with no text (finish reason, block reason, token counts), for the log."""
    bits = []
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            if reason is not None:
                bits.append(f"finish_reason={getattr(reason, 'name', reason)}")
        else:
            bits.append("no candidates")
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None) if feedback else None
        if block is not None:
            bits.append(f"block_reason={getattr(block, 'name', block)}")
        usage = getattr(response, "usage_metadata", None)
        for key in ("prompt_token_count", "candidates_token_count", "thoughts_token_count"):
            value = getattr(usage, key, None) if usage else None
            if value is not None:
                bits.append(f"{key}={value}")
    except Exception:  # noqa: BLE001 -- diagnostics must never break the call
        pass
    return ", ".join(bits)


def _finish_reason_name(response) -> str | None:
    """How the model says it stopped ("STOP", "MAX_TOKENS", "PROHIBITED_CONTENT", ...), or None when the reply
    carries no such information."""
    reason = getattr(response, "finish_reason", None)
    if reason is None:
        candidates = getattr(response, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return None
    name = (getattr(reason, "name", None) or str(reason)).rsplit(".", 1)[-1]
    return None if name == "FINISH_REASON_UNSPECIFIED" else name


def _require_complete(response) -> None:
    """A reply the model cut short (blocked content, token limit, ...) is not a finished answer, whatever text arrived
    before the cut. Raising here makes the chain move on to the next model, and when none of them finishes the stage
    is queued for a retry, so nothing partial is ever stored as if it were complete (2026-11-03: a transcript cut by
    PROHIBITED_CONTENT used to be saved as a success). A reply that says nothing about how it ended is accepted."""
    name = _finish_reason_name(response)
    if name is None or name == "STOP":
        return
    received = len(getattr(response, "text", None) or "")
    detail = _describe_empty(response)
    raise RuntimeError(f"the reply was cut short, finish_reason={name} ({received} characters received before the cut)"
                       + (f" [{detail}]" if detail else ""))


def _non_empty_text(response) -> str:
    """A model that answers with nothing, or whose answer was cut short, has failed: raising here lets the chain
    move on to the next model."""
    _require_complete(response)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        detail = _describe_empty(response)
        raise RuntimeError("returned an empty response" + (f" ({detail})" if detail else ""))
    return text


def _stream_response(client, model_name: str, contents, config):
    """One model call, sent as a stream and joined back into a single answer. Same model, prompt, audio
    and settings as a plain call; the only difference is that pieces arrive while the model works, so
    _call_with_timeout can tell a stuck call (no pieces) from a slow one (pieces still coming)."""
    beat = getattr(_beat_local, "beat", None)
    if beat is not None:
        beat["watching"] = True
        beat["last"] = time.time()
    parts: list[str] = []
    last = None
    finish_seen = None      # how the model says it stopped, from whichever piece carries it (not always the last one)
    started = time.time()
    first_piece_s = None
    pieces = 0
    for chunk in client.models.generate_content_stream(model=model_name, contents=contents, config=config):
        if beat is not None:
            beat["pulses"] += 1
            beat["last"] = time.time()
        pieces += 1
        if first_piece_s is None:
            first_piece_s = time.time() - started
        last = chunk
        cands = getattr(chunk, "candidates", None)
        if cands and getattr(cands[0], "finish_reason", None) is not None:
            finish_seen = cands[0].finish_reason
        try:
            piece = chunk.text
        except Exception:  # noqa: BLE001 -- a piece without text is fine, it still shows the call is alive
            piece = None
        if piece:
            parts.append(piece)
    return SimpleNamespace(
        text="".join(parts), parsed=None,
        candidates=getattr(last, "candidates", None), finish_reason=finish_seen,
        usage_metadata=getattr(last, "usage_metadata", None),
        prompt_feedback=getattr(last, "prompt_feedback", None),
        first_piece_s=first_piece_s, pieces=pieces, total_s=time.time() - started,
    )


def _stream_logged(client, model_name: str, contents, config, log):
    """_stream_response plus one run-log line with the timing, so the stall limits can be tuned on real numbers."""
    response = _stream_response(client, model_name, contents, config)
    if response.first_piece_s is None:
        log(f"    {model_name}: the reply had no pieces ({response.total_s:.0f}s)")
    else:
        log(f"    {model_name}: first piece after {response.first_piece_s:.0f}s, "
            f"{response.pieces} piece(s), {response.total_s:.0f}s in all")
    return response


def _sha16(data) -> str:
    b = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def _detail_request(kind: str, context: str, models: list, system, contents_text: str, *,
                    audio: bytes | None = None, mime: str | None = None, schema=None) -> None:
    """Everything sent to the model, written in full to the run log (see runlog.py)."""
    _detail(f"MODEL CALL [{context}] kind={kind} at {datetime.now().isoformat(timespec='milliseconds')}")
    base = request_settings()
    _detail(f"  chain={models} thinking_level={base['thinking_level']} max_output_tokens={base['max_output_tokens']} "
            f"structured_output={_SCHEMA_ENABLED} schema={getattr(schema, '__name__', None)} "
            f"timeout={_MODEL_TIMEOUT_SECONDS}s first_chunk_limit={_STREAM_FIRST_CHUNK_SECONDS}s "
            f"stall_limit={_STREAM_STALL_SECONDS}s")
    if audio is not None:
        _detail(f"  audio: {len(audio)} bytes, mime={mime}, sha256[:16]={_sha16(audio)}")
    _block(f"SYSTEM PROMPT [{context}] sha256[:16]={_sha16(system or '')}", system or "(none)")
    _block(f"USER CONTENT [{context}] sha256[:16]={_sha16(contents_text)}", contents_text)


def _detail_response(context: str, model_name: str, response) -> None:
    """Everything that came back from one model attempt: the raw reply in full, timing, finish reason and the
    token counts the API reports (never any thinking text)."""
    usage = getattr(response, "usage_metadata", None)
    counts = {}
    for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count",
              "cached_content_token_count"):
        v = getattr(usage, k, None)
        if v is not None:
            counts[k] = v
    finish = [str(getattr(c, "finish_reason", None)) for c in (getattr(response, "candidates", None) or [])]
    text = getattr(response, "text", "") or ""
    _detail(f"MODEL RESPONSE [{context}] model={model_name} first_piece_s={getattr(response, 'first_piece_s', None)} "
            f"pieces={getattr(response, 'pieces', None)} total_s={getattr(response, 'total_s', None)} "
            f"finish_reason={finish} usage={counts} prompt_feedback={getattr(response, 'prompt_feedback', None)}")
    _block(f"RAW REPLY [{context}] model={model_name} sha256[:16]={_sha16(text)}", text if text else "(empty)")


def _chain_with_key_failover(make_build_call, models: list[str], *, log, context: str, timeout: float):
    """Runs the model chain on the main key. Only if the chain failed AND a quota limit was involved,
    and a backup key exists, runs the whole chain once more on the backup key (its quota is separate)
    before the caller gives up and the video is queued."""
    _quota_seen[0] = False
    try:
        return _generate_with_fallback(models, make_build_call(_client()), log=log, context=context, timeout=timeout)
    except Exception:  # noqa: BLE001 -- decided below
        if not (_quota_seen[0] and os.environ.get("GEMINI_API_KEY_BACKUP")):
            raise
    log(f"    {context}: quota hit on the main key -- trying the whole chain once on the backup key")
    file_log.warning(f"{context}: quota hit on the main key, retrying the chain once on the backup key")
    return _generate_with_fallback(models, make_build_call(_client(backup=True)), log=log, context=context,
                                   timeout=timeout, key_tag="backup:")


# The request limit is 20 MB in all, and inline audio travels base64-encoded (4/3 of its size), so a recording above this goes through the
# Files API instead of being cut into pieces (design decision 2026-11-09: a recording is never cut).
INLINE_AUDIO_MAX_BYTES = 14_000_000
_UPLOAD_READY_SECONDS = 300.0


def _upload_audio(client, audio: bytes, mime: str):
    """Uploads the recording once (Files API: up to 2 GB per file, kept 48 hours) and waits until the service has processed it. Returns the
    part to put in the request."""
    import io
    uploaded = client.files.upload(file=io.BytesIO(audio), config=types.UploadFileConfig(mime_type=mime))
    deadline = time.monotonic() + _UPLOAD_READY_SECONDS
    while uploaded.state == types.FileState.PROCESSING and time.monotonic() < deadline:
        time.sleep(2.0)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state not in (None, types.FileState.ACTIVE):
        raise RuntimeError(f"the uploaded recording is not usable (state={uploaded.state})")
    return types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type or mime)


def generate_audio_text(audio: bytes, prompt: str, *, system: str | None = None, context: str,
                        log=print, timeout: float | None = None,
                        mime: str = "audio/mp3") -> tuple:
    """Ask an audio-capable model about a recording."""
    if _TEST_AUDIO is not None:
        return _TEST_AUDIO(audio, prompt, system, context), "test-model"

    models = _get_models(context)
    big = len(audio) > INLINE_AUDIO_MAX_BYTES
    contents = None if big else [types.Part.from_bytes(data=audio, mime_type=mime), prompt]
    config = types.GenerateContentConfig(system_instruction=system, **max_quality_config_kwargs())
    _detail_request("audio", context, models, system, prompt, audio=audio, mime=mime)

    def make_build_call(client):
        uploaded = []                      # one upload per API key: a file belongs to the key that uploaded it

        def build_call(model_name: str):
            log(f"  trying {model_name} ({len(audio) / 1e6:.1f} MB of audio) ...")
            call_contents = contents
            if big:
                if not uploaded:
                    log(f"  {len(audio) / 1e6:.1f} MB is too big to send inline: uploading it once (Files API) ...")
                    uploaded.append(_upload_audio(client, audio, mime))
                call_contents = [uploaded[0], prompt]
            resp = _stream_logged(client, model_name, call_contents, config, log)
            _detail_response(context, model_name, resp)
            return _non_empty_text(resp)
        return build_call

    try:
        result, model_name = _chain_with_key_failover(
            make_build_call, models, log=log, context=context,
            timeout=timeout or _MODEL_TIMEOUT_SECONDS,
        )
    except Exception as e:
        raise StageSkippedError(f"{context}: all models failed: {e}") from e

    return result, model_name


def generate_json(prompt: str, schema, *, system: str | None = None, context: str,
                  log=print, timeout: float | None = None) -> tuple:
    """One JSON call through the model chain; returns (parsed, model_name). The prompt is sent as it is: the reply shape is stated by
    the caller's own prompt file, so it is part of the prompt text that the stage fingerprint covers
   . The reply is read in a tolerant way. (The JSON-Schema hint that used to be appended
    for callers that did not state the shape was removed 2026-11-24: no caller used it.)"""
    if _TEST_JSON is not None:
        return _TEST_JSON(prompt, schema, system, context), "test-model"

    models = _get_models(context)
    contents = prompt
    config = types.GenerateContentConfig(
        system_instruction=system,
        **_gemini_config_kwargs(schema),
        **max_quality_config_kwargs(),
    )
    _detail_request("json", context, models, system, contents, schema=schema)
    partial: list = []      # replies cut short (token limit, blocked content ...) that still held finished items: (n, model, result)

    def make_build_call(client):
        def build_call(model_name: str):
            log(f"  trying {model_name} ...")
            resp = _stream_logged(client, model_name, contents, config, log)
            _detail_response(context, model_name, resp)
            try:
                _require_complete(resp)
            except Exception:
                # the chain still moves on to the next model (a finished answer beats a cut one), but what was finished before
                # the cut is kept as a last resort, so it is never lost
                cut = _parse_structured(resp, schema)
                n_cut = len(getattr(cut, "items", None) or [])
                if n_cut:
                    partial.append((n_cut, model_name, cut))
                    _detail(f"PARTIAL [{context}] model={model_name}: the reply was cut short, {n_cut} finished item(s) were read "
                            f"and are kept as a last resort")
                raise
            parsed = _parse_structured(resp, schema)
            _detail(f"PARSE [{context}] model={model_name} schema={schema.__name__}: "
                    + ("FAILED, could not parse the reply" if parsed is None
                       else f"ok, {len(getattr(parsed, 'items', []) or [])} item(s)"))
            if parsed is None:
                raise RuntimeError(f"could not parse the response into {schema.__name__}")
            return parsed
        return build_call

    try:
        result, model_name = _chain_with_key_failover(
            make_build_call, models, log=log, context=context,
            timeout=timeout or _MODEL_TIMEOUT_SECONDS,
        )
    except Exception as e:
        if partial:             # no model gave a finished answer: the fullest cut reply is used, its finished items are not lost
            n_cut, model_name, result = max(partial, key=lambda entry: entry[0])
            log(f"    {context}: no model finished its answer -- using the {n_cut} finished item(s) of the cut reply of {model_name}")
            _detail(f"PARTIAL [{context}] used: {n_cut} item(s) from the cut reply of {model_name} (every model failed: {e})")
            return result, model_name
        raise StageSkippedError(f"{context}: all models failed: {e}") from e

    return result, model_name
