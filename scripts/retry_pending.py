"""
Task Scheduler entrypoint: resumes any video whose pipeline
(listen / extract) previously hit a
StageSkippedError (typically every Gemini model's daily quota exhausted,
or a Google-side 503 overload) and got queued in store.pending_retries
instead of being lost. Added 2026-09-11 per explicit user request: a
failure should retry itself automatically from where it left off, not
require the user to notice and re-click the bookmarklet.

Same pythonw-safe stdout/stderr handling as send_daily.py -- see that
file's docstring for why this must happen before any other import.
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

import dataclasses
import html
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from lifecompass import store, tunables
from lifecompass.pipeline import (run_stages, PRODUCTION_STAGES, _RunLogger, file_log, retry_supplementary, parse_supplementary_payload,
                                  DELIVER_STAGE, parse_delivery_payload, run_is_live, title_is_placeholder)
from lifecompass.audio import video_info
from lifecompass.llm import reset_exhausted_models, api_calls_made
from lifecompass.publisher import send_message, send_photo_group

def _h(value) -> str:
    """Text for a Telegram message sent with parse_mode=HTML (a title or an error text may hold < > &)."""
    return html.escape(str(value), quote=False)


MAX_ATTEMPTS = tunables.get("retry", "max_attempts")  # 36 x 5 minutes = 3 hours before giving up and notifying (config.toml [retry])
MAX_DELIVERY_ATTEMPTS = tunables.get("retry", "max_delivery_attempts")  # 288 x 5 minutes = a day of waiting for Telegram before giving up (quietly: see _retry_delivery)


def _retry_supplementary(p) -> None:
    """Handles the "supplementary" queue stage (added 2026-09-29). The video itself
    already finished and was published -- only work that must never block it
    is outstanding: image-to-principle matching
    and standalone image captions (see pipeline.retry_supplementary and
    parse_supplementary_payload for the items_json shape). Needs neither the
    transcript nor the item list, so it bypasses the stage-resume logic in
    main(). Quiet on Telegram until it gives up: nothing here changes what
    the user already received."""
    try:
        payload = parse_supplementary_payload(p.items_json)
    except Exception:
        file_log.exception(f"retry_pending: could not deserialize supplementary payload for video_id={p.video_id}, dropping")
        store.delete_pending_retry(p.id)
        return

    if p.attempts >= MAX_ATTEMPTS:
        file_log.warning(f"retry_pending: giving up on supplementary retry for video_id={p.video_id} after {p.attempts} attempts: {payload}")
        missing = []
        if payload["match"]:
            missing.append("linking the images to the principles")
        if payload["captions"]:
            missing.append(f"captioning {len(payload['captions'])} image(s)")
        send_message(
            f"⚠️ <b>Some extra steps did not finish</b>\nVideo: {_h(p.title)}\nMissing: {' + '.join(missing)}\nThe principles and images themselves were saved and are shown as usual, but these steps were not completed after {p.attempts} attempts.",
            log=print,
        )
        store.delete_pending_retry(p.id)
        return

    reset_exhausted_models()
    calls_before = api_calls_made()
    rlog = _RunLogger(p.url, print)
    rlog(f"-> Retrying supplementary steps for '{p.title}' {payload} (attempt {p.attempts + 1}/{MAX_ATTEMPTS}) ...")
    try:
        remaining = retry_supplementary(p.video_id, payload, rlog)
    except Exception as e:
        file_log.exception(f"retry_pending: unhandled exception in supplementary retry for video_id={p.video_id}")
        store.save_pending_retry(p.video_id, p.url, p.title, "supplementary", json.dumps(payload), p.model_used, f"{type(e).__name__}: {e}")
        rlog.close("supplementary retry crashed")
        print(f"[FAILED] supplementary retry crashed for video_id={p.video_id}: {e}")
        return

    if remaining:
        store.save_pending_retry(p.video_id, p.url, p.title, "supplementary", json.dumps(remaining), p.model_used, "supplementary: models still failing",
                                count_attempt=api_calls_made() > calls_before)
        rlog.close(f"supplementary retry: still pending {remaining}")
        print(f"[PENDING] supplementary work still outstanding for video_id={p.video_id}: {remaining}")
    else:
        store.delete_pending_retry(p.id)
        rlog.close("ok (supplementary steps done)")
        print(f"[OK] supplementary retry finished for video_id={p.video_id}")


def _retry_delivery(p) -> None:
    """Handles the "deliver" queue stage: what could not be sent to Telegram when the video finished (Telegram or the network was
    down) is sent now, in the original order: the album first, then the messages, each one only after the one before it was taken.
    The principles are already stored. No Telegram message about a failed delivery (Telegram is the channel that is down): after
    MAX_DELIVERY_ATTEMPTS it only writes to the log and stops, and the principles still come back through the random resurfacing."""
    try:
        payload = parse_delivery_payload(p.items_json)
    except Exception:
        file_log.exception(f"retry_pending: could not read the delivery payload of {p.video_id}, dropping it")
        store.delete_pending_retry(p.id)
        return
    if p.attempts >= MAX_DELIVERY_ATTEMPTS:
        file_log.warning(f"retry_pending: giving up on delivering {p.title!r} to Telegram after {p.attempts} attempts")
        store.delete_pending_retry(p.id)
        return

    photos = [x for x in payload["photos"] if Path(x).exists()]   # a picture whose file is gone cannot be sent
    chunks = list(payload["chunks"])

    def keep() -> None:
        store.save_pending_retry(p.video_id, p.url, p.title, DELIVER_STAGE, json.dumps({"photos": photos, "chunks": chunks}, ensure_ascii=False),
                                 p.model_used, "telegram could not be reached")
        print(f"[PENDING] {len(chunks)} message(s){' and the album' if photos else ''} still waiting for Telegram: {p.title}")

    try:
        if photos:
            if send_photo_group([Path(x) for x in photos], caption=_h(p.title), log=print) is False:
                keep()
                return
            photos = []
        while chunks:
            if send_message(chunks[0], log=print) is False:
                keep()
                return
            chunks.pop(0)
    except Exception:
        file_log.exception(f"retry_pending: unhandled exception delivering video_id={p.video_id}")
        keep()
        return
    store.delete_pending_retry(p.id)
    print(f"[OK] delivered to Telegram: {p.title}")


def main() -> None:
    pending = store.get_pending_retries()
    if not pending:
        return

    for p in pending:
        if p.stage == DELIVER_STAGE:
            _retry_delivery(p)
            continue

        if p.stage == "supplementary":
            _retry_supplementary(p)
            continue

        if run_is_live(p):
            continue            # a run is working on this video right now (its heartbeat is recent, or its process is still there)

        if p.attempts >= MAX_ATTEMPTS:
            file_log.warning(f"retry_pending: giving up on video_id={p.video_id} after {p.attempts} attempts")
            send_message(
                f"❌ <b>The video was given up on after {p.attempts} failed attempts</b>\nVideo: {_h(p.title)}\nLast error: {_h(p.last_error)}\nSend the bookmarklet again manually if you still want it.",
                log=print,
            )
            store.delete_pending_retry(p.id)
            continue

        if title_is_placeholder(p):
            try:                                    # the first step (reading the video's page) failed when it was started
                p = dataclasses.replace(p, title=video_info(p.url)[1] or p.title)
            except Exception as e:  # noqa: BLE001
                file_log.warning(f"retry_pending: still cannot read the video page of {p.video_id}: {e}")
                store.save_pending_retry(p.video_id, p.url, p.title, p.stage, p.items_json, p.model_used, f"{type(e).__name__}: {e}")
                continue

        reset_exhausted_models()
        rlog = _RunLogger(p.url, print)
        rlog(f"-> Resuming video '{p.title}' from stage '{p.stage}' (attempt {p.attempts + 1}/{MAX_ATTEMPTS}) ...")
        # A row queued under an older stage name restarts at the beginning.
        stage = p.stage if p.stage in PRODUCTION_STAGES else "listen"

        try:
            result = run_stages(p.video_id, p.title, p.url, rlog, stage=stage)
            print(f"[{'OK' if result.ok else 'PENDING'}] {result.message}")
        except Exception as e:
            # Final safety net, mirrors listener.py's -- a bug here (not a
            # model/StageSkippedError, something run_stages itself
            # didn't anticipate) would otherwise fail this scheduled task
            # silently every 5 minutes forever with no way to find out.
            file_log.exception(f"retry_pending: unhandled exception resuming video_id={p.video_id}")
            print(f"[FAILED] Unhandled exception resuming video_id={p.video_id}: {e}")
            # Count this toward MAX_ATTEMPTS too -- otherwise a crash that
            # ISN'T a StageSkippedError (which run_stages already
            # counts via save_pending_retry) would retry every 5 minutes
            # forever without ever hitting the give-up threshold.
            store.save_pending_retry(p.video_id, p.url, p.title, p.stage, p.items_json, p.model_used, f"{type(e).__name__}: {e}")
            try:
                send_message(
                    f"⚠️ <b>Unexpected error during the retry</b>\nVideo: {_h(p.title)}\nReason: {_h(type(e).__name__)}: {_h(e)}",
                    log=print,
                )
            except Exception:
                file_log.exception("Failed to send the unhandled-exception Telegram alert itself")


if __name__ == "__main__":
    main()
