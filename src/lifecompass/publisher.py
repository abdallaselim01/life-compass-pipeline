"""
Sends messages to you on Telegram via a bot (direct message, not a
channel -- see README for why that was the final choice). Uses the raw
Bot API over HTTPS, no SDK dependency needed for something this simple.

Retries server-side failures (5xx / 429) up to 3 times with a short delay, and a network that is down (DNS, a refused
socket) with longer, growing waits (config.toml [telegram], 2026-11-19) -- but
NOT blindly on every kind of network exception. A real production bug
(caught via direct log evidence, see HANDOFF.md Section 19/20) showed
this matters: a `requests.exceptions.ReadTimeout` means the request was
already SENT and Telegram may have already processed it -- we just
didn't get the response back in time. Retrying that case unconditionally
caused a real duplicate message (Telegram delivered it twice, our own
code only logged one "failed" + one "succeeded" and had no idea a
duplicate had gone out). A `ConnectTimeout`/`ConnectionError`, by
contrast, means the request never reached the server at all -- safe to
retry.

So: only ConnectTimeout/ConnectionError are retried automatically.
ReadTimeout is treated as "outcome unknown, do not risk a duplicate" --
logged clearly and NOT retried. This trades a slightly higher chance of
under-sending (the principle simply stays eligible for the next hourly
tick, since store.mark_shown is never called on a False return) for
eliminating a much worse failure mode: an unnoticed, unlogged duplicate
message actually reaching the user.
"""
from __future__ import annotations

import os
import time
import json
import requests
from pathlib import Path

from . import tunables
from .logging_setup import get_logger, redact

log_file = get_logger("publisher")

_MAX_ATTEMPTS = tunables.get("telegram", "max_attempts")   # config.toml [telegram]
_RETRY_DELAY_SECONDS = tunables.get("telegram", "retry_delay_seconds")
# A network that is down (DNS fails, a socket is refused) usually comes back within minutes: such failures get their own, longer,
# growing waits -- 5, 15, 45, 135 seconds by default, about 3.5 minutes in all -- before a send gives up.
_NETWORK_RETRIES = tunables.get("telegram", "network_retries")
_NETWORK_RETRY_BASE_SECONDS = tunables.get("telegram", "network_retry_base_seconds")
_NETWORK_RETRY_MAX_SECONDS = tunables.get("telegram", "network_retry_max_seconds")   # config.toml [telegram]

# Telegram's own limit for the caption of a photo or an album (Bot API: 0-1024 characters after the formatting is parsed). It is the
# platform's number, not a setting of ours.
TELEGRAM_CAPTION_LIMIT = 1024
# An album is a big upload: a fixed 30 s limit cannot hold on a slow line (6.6 MB at about 100 KB/s took over a minute on 2026-11-19,
# and the principles wait for their album). Its limit grows with what is sent: the bytes at the slowest rate still waited for, plus a
# margin, never below 30 s (config.toml [telegram]).
_ALBUM_MIN_UPLOAD_KBPS = tunables.get("telegram", "album_min_upload_kbps")
_ALBUM_TIMEOUT_MARGIN_SECONDS = tunables.get("telegram", "album_timeout_margin_seconds")


def _album_timeout_seconds(total_bytes: int) -> float:
    return max(30.0, total_bytes / (_ALBUM_MIN_UPLOAD_KBPS * 1024) + _ALBUM_TIMEOUT_MARGIN_SECONDS)


def _backoff_seconds(network_failures: int) -> float:
    """The wait before the next try after the n-th network failure in a row: the base, then three times more each time, capped."""
    return min(_NETWORK_RETRY_BASE_SECONDS * 3 ** (network_failures - 1), _NETWORK_RETRY_MAX_SECONDS)


def send_message(text: str, log=print) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        msg = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env -- skipping send."
        log(f"  ! {msg}")
        log_file.warning(msg)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    log_file.debug(f"POST sendMessage chat_id={chat_id} text_len={len(text)} preview={text[:80]!r}")

    last_error_summary = ""
    attempt = 0
    network_failures = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=15,
            )
            log_file.debug(f"attempt {attempt}: HTTP status={resp.status_code} body={resp.text[:500]}")
            data = resp.json()
            if data.get("ok"):
                log_file.info(f"Telegram send OK on attempt {attempt} (message_id={data.get('result', {}).get('message_id')})")
                return True

            last_error_summary = str(data)
            # 4xx (bad token, chat not found, bad request) won't fix itself on
            # retry -- only retry on 5xx / rate-limit-shaped failures.
            if resp.status_code < 500 and resp.status_code != 429:
                log(f"  ! Telegram send failed (non-retryable): {data}")
                log_file.error(f"Telegram API returned non-retryable error: {data}")
                return False
            log_file.warning(f"attempt {attempt} got retryable error: {data}")
        except requests.exceptions.ReadTimeout as e:
            # The request WAS sent -- Telegram may have already processed it.
            # Retrying here risks a real duplicate delivery (this exact
            # scenario happened in production once -- see docstring above).
            # Treat as unknown outcome, do not retry.
            last_error_summary = f"ReadTimeout (request may have been delivered, not retrying): {e}"
            log(f"  ! Telegram send timed out waiting for response -- NOT retrying "
                f"(request may have already been delivered): {redact(str(e))}")
            log_file.warning(f"attempt {attempt} got ReadTimeout -- outcome ambiguous, aborting retries to avoid duplicate: {e}")
            return False
        except requests.exceptions.RequestException as e:
            # ConnectTimeout, ConnectionError, etc. -- the request never
            # reached the server, safe to retry, with the longer growing waits of a network that is down.
            last_error_summary = str(e)
            network_failures += 1
            log_file.warning(f"attempt {attempt} raised network exception (safe to retry): {e}")
            if network_failures > _NETWORK_RETRIES:
                break
            time.sleep(_backoff_seconds(network_failures))
            continue

        if attempt - network_failures >= _MAX_ATTEMPTS:   # server-side errors (5xx / 429) keep the short retries
            break
        time.sleep(_RETRY_DELAY_SECONDS)

    log(f"  ! Telegram send failed after {attempt} attempts: {redact(last_error_summary)}")
    log_file.error(f"Telegram send failed after {attempt} attempts: {last_error_summary}")
    return False


def send_photo_group(photo_paths: list[Path], caption: str = "", log=print) -> bool:
    """Sends a Telegram media group (album) of local image files -- the
    'Facebook-post-style' attachment the user asked for: one scrollable
    album, not a caption/link under each individual principle. Two real API
    constraints shape this function (see design discussion in README):

    1. A single sendMediaGroup call is capped at 10 items -- extra photos
       beyond that are sent as additional album call(s).
    2. Telegram only renders a caption on the FIRST item of an album -- so
       `caption` (the video title) is attached only to the very first photo
       of the very first chunk; later chunks (10+ images) carry no caption
       to avoid repeating the title mid-album.

    Uses the same retryable-vs-not distinction as send_message (ConnectTimeout/
    ConnectionError retried, ReadTimeout treated as an ambiguous outcome and
    NOT retried, to avoid risking a duplicate album delivery)."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        msg = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in .env -- skipping photo send."
        log(f"  ! {msg}")
        log_file.warning(msg)
        return False
    if not photo_paths:
        return True

    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    ALBUM_LIMIT = 10
    chunks = [photo_paths[i:i + ALBUM_LIMIT] for i in range(0, len(photo_paths), ALBUM_LIMIT)]

    all_ok = True
    for chunk_idx, chunk in enumerate(chunks):
        # Read bytes up front (not open file handles) so the same payload
        # can be safely resent across retry attempts without stream-exhaustion
        # issues.
        file_bytes = [p.read_bytes() for p in chunk]
        album_timeout = _album_timeout_seconds(sum(len(d) for d in file_bytes))

        media = []
        files = {}
        for i, (path, data) in enumerate(zip(chunk, file_bytes)):
            attach_key = f"photo{i}"
            media_item = {"type": "photo", "media": f"attach://{attach_key}"}
            if chunk_idx == 0 and i == 0 and caption:
                media_item["caption"] = caption
                media_item["parse_mode"] = "HTML"
            media.append(media_item)
            files[attach_key] = (path.name, data, "image/png")

        ok = False
        last_error_summary = ""
        attempt = 0
        network_failures = 0
        while True:
            attempt += 1
            try:
                resp = requests.post(
                    url,
                    data={"chat_id": chat_id, "media": json.dumps(media)},
                    files=files,
                    timeout=album_timeout,
                )
                data_json = resp.json()
                if data_json.get("ok"):
                    ok = True
                    log_file.info(f"sendMediaGroup OK for chunk {chunk_idx} ({len(chunk)} photos)")
                    break
                last_error_summary = str(data_json)
                if resp.status_code < 500 and resp.status_code != 429:
                    log_file.error(f"sendMediaGroup non-retryable error on chunk {chunk_idx}: {data_json}")
                    break
                log_file.warning(f"sendMediaGroup chunk {chunk_idx} attempt {attempt} retryable error: {data_json}")
            except requests.exceptions.ReadTimeout as e:
                last_error_summary = f"ReadTimeout (album may have been delivered, not retrying): {e}"
                log_file.warning(f"sendMediaGroup chunk {chunk_idx} got ReadTimeout -- outcome ambiguous, aborting retries: {e}")
                break
            except requests.exceptions.RequestException as e:
                last_error_summary = str(e)
                network_failures += 1
                log_file.warning(f"sendMediaGroup chunk {chunk_idx} attempt {attempt} network exception (safe to retry): {e}")
                if network_failures > _NETWORK_RETRIES:
                    break
                time.sleep(_backoff_seconds(network_failures))
                continue

            if attempt - network_failures >= _MAX_ATTEMPTS:
                break
            time.sleep(_RETRY_DELAY_SECONDS)

        if not ok:
            log(f"  ! Failed to send photo album chunk {chunk_idx}: {redact(last_error_summary)}")
            log_file.error(f"sendMediaGroup chunk {chunk_idx} failed after retries: {last_error_summary}")
            all_ok = False

    return all_ok
