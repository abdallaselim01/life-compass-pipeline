"""
Random resurfacing: meant to run frequently (every hour, via Task
Scheduler) rather than once a day at a fixed time. Each run is a coin
flip -- send one principle or do nothing -- calibrated so the *expected*
count over 24 hourly runs matches DAILY_RESURFACE_COUNT, but the actual
send times land unpredictably across the day. That unpredictability is
the point: a fixed daily time trains you to expect it (and mentally
brace for/dismiss it), where scattered, un-schedulable pings are what
actually keeps the ideas ambient rather than anticipated.

Each run sends at most ONE principle (not a batch of five) -- five
separate pings through the day, each one small and easy to actually
read, rather than one wall of text you skim once and forget.
"""
from __future__ import annotations

import html
import os
import sys
import random
from datetime import datetime
from pathlib import Path

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from . import store, tunables
from .publisher import TELEGRAM_CAPTION_LIMIT, send_message, send_photo_group
from .logging_setup import get_logger

file_log = get_logger("resurface")

# Message content, per user request/clarification:
# - the lead line is the stored `principle` text --
#   changed 2026-09-07 from always-source_quote to always-principle, per
#   direct request, so the random resurface now matches pipeline.py's
#   immediate post-video notification (also `principle`-led) instead of
#   diverging from it.

# Fraction of resurfacing turns that send a standalone captioned image
# (see run_random_resurface) instead of a principle -- see
# store.pick_standalone_images_for_resurfacing / extractor.caption_standalone_image.
# Kept modest since these are a supplementary content type, not the main event.
STANDALONE_IMAGE_PROBABILITY = tunables.get("resurface", "standalone_image_probability")   # config.toml [resurface]

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "resurface.log"

RUNS_PER_DAY = tunables.get("resurface", "runs_per_day")  # this script is scheduled hourly -- see README (config.toml)

# Quiet hours: OFF (design decision 2026-11-18: no fixed sleep and wake times at the moment). config.toml [resurface] has the same
# start and end hour (0 and 0), an empty window: nothing is skipped and all 24 hours are active. To bring a window back, set
# quiet_hours_start and quiet_hours_end there: the hourly Task Scheduler tick still fires in it, but run_random_resurface()
# exits without spending a probability roll, and the probability is computed against the ACTIVE hours only (so the daily
# average still lands near DAILY_RESURFACE_COUNT instead of being diluted by hours that could never send anyway).
QUIET_HOURS_START = tunables.get("resurface", "quiet_hours_start")   # 0 = 00:00 (config.toml)
QUIET_HOURS_END = tunables.get("resurface", "quiet_hours_end")       # end hour, exclusive (equal to the start = no quiet hours)
ACTIVE_HOURS_PER_DAY = RUNS_PER_DAY - (QUIET_HOURS_END - QUIET_HOURS_START)


def _in_quiet_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    return QUIET_HOURS_START <= now.hour < QUIET_HOURS_END


def _log(line: str) -> None:
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {line}\n")


def _format_message(p: store.Principle) -> str:
    """The message of one principle: its stored `principle` text in bold and nothing else (the same lead as pipeline.py's
    immediate post-video notification). The generated context line, the related-principles list and the stance tag are
    not shown (user requests 2026-09-07 and 2026-09-21)."""
    return f"<b>{html.escape(p.principle, quote=False)}</b>"


def _send_pictures_with_text(paths: list[Path], text_html: str, visible_text: str) -> bool:
    """Pictures with their text. One message with the text as the caption, as always, and only a text longer than Telegram's caption
    limit (design decision 2026-11-24) goes as the album first and the text right after it as its own message: a caption over the limit
    is refused by Telegram for good, and a principle refused for good is never marked as shown, so it would stay first in the queue
    and keep every other principle from ever being sent. True only when everything was sent."""
    if len(visible_text) <= TELEGRAM_CAPTION_LIMIT:
        return bool(send_photo_group(paths, caption=text_html, log=_log))
    file_log.info(f"text of {len(visible_text)} characters is over the caption limit ({TELEGRAM_CAPTION_LIMIT}): album first, text after it")
    if not send_photo_group(paths, caption="", log=_log):
        return False
    return bool(send_message(text_html, log=_log))


def run_random_resurface() -> None:
    load_dotenv(override=True)
    daily_target = tunables.get("resurface", "daily_count")   # config.toml [resurface] (was DAILY_RESURFACE_COUNT in .env until 2026-11-24)

    if _in_quiet_hours():
        _log(f"Quiet hours ({QUIET_HOURS_START:02d}:00-{QUIET_HOURS_END:02d}:00) -- skipping, no roll spent.")
        file_log.debug("skipped: within quiet hours, no probability roll consumed")
        return

    probability = daily_target / ACTIVE_HOURS_PER_DAY
    file_log.info(f"run_random_resurface tick: daily_target={daily_target} probability={probability:.3f} (active_hours={ACTIVE_HOURS_PER_DAY})")

    total = store.count_all()
    if total == 0:
        _log("No principles stored yet -- nothing to resurface.")
        file_log.info("store is empty -- nothing to resurface")
        return

    roll = random.random()
    if roll >= probability:
        _log(f"Skip this run (roll={roll:.3f} >= p={probability:.3f}).")
        file_log.debug(f"skipped: roll={roll:.3f} p={probability:.3f}")
        return

    # A small fraction of resurfacing turns send a standalone captioned
    # image instead of a principle (see store.pick_standalone_images_for_resurfacing
    # and extractor.caption_standalone_image) -- these are user-curated
    # screenshots that illustrate a fact/context the speaker explained but
    # that never became a life-principle of their own. Only taken if at
    # least one such image actually exists; otherwise falls straight
    # through to the normal principle pick below.
    if random.random() < STANDALONE_IMAGE_PROBABILITY:
        # a picture whose file is gone cannot be sent and is passed over (it would stay first in the rotation and block the rest)
        standalone = [i for i in store.pick_standalone_images_for_resurfacing(5) if Path(i.enhanced_path).exists()][:1]
        if standalone:
            img = standalone[0]
            file_log.info(f"selected standalone image id={img.id} times_shown={img.times_shown}")
            try:
                ok = _send_pictures_with_text([Path(img.enhanced_path)], html.escape(img.standalone_caption, quote=False),
                                              img.standalone_caption)
            except Exception:
                ok = False
                file_log.exception(f"sending standalone image id={img.id} failed (non-fatal)")
            if ok:
                store.mark_image_shown(img.id)
                _log(f"Sent standalone image #{img.id}.")
                file_log.info(f"sent + marked shown: standalone image id={img.id}")
            else:
                _log("Standalone image send failed -- NOT marking as shown.")
                file_log.error(f"send failed for standalone image id={img.id}, not marking as shown")
            return

    picked = store.pick_for_resurfacing(1)
    if not picked:
        _log("pick_for_resurfacing returned nothing (unexpected, DB non-empty).")
        file_log.error("pick_for_resurfacing returned nothing despite non-empty store")
        return

    p = picked[0]
    file_log.info(f"selected principle id={p.id} times_shown={p.times_shown} last_shown_at={p.last_shown_at}")

    # If this principle has supporting image(s) (see images.py /
    # store.link_principle_image), send them as ONE Telegram message with
    # the principle text attached as the photo's caption -- not as two
    # separate messages (album, then a standalone text bubble). Telegram
    # only renders a caption on the album's first photo (see
    # send_photo_group's docstring), which is fine here since there's only
    # ever one caption to show per principle anyway. User rule 2026-11-19: a
    # principle that has pictures is never sent without them. If the pictures
    # cannot be sent now, nothing is sent and the principle is NOT marked as
    # shown, so it can be picked again. A picture whose file is gone cannot be
    # sent and does not count: a principle left with no picture file goes out
    # as text.
    images = [img for img in (store.get_images_for_principle(p.id) if p.id else []) if Path(img.enhanced_path).exists()]
    if images:
        try:
            ok = _send_pictures_with_text([Path(img.enhanced_path) for img in images], _format_message(p), p.principle)
        except Exception:
            ok = False
            file_log.exception(f"send_photo_group failed for principle id={p.id} (non-fatal)")
    else:
        ok = send_message(_format_message(p), log=_log)
    if ok:
        store.mark_shown([p.id])
        _log(f"Sent principle #{p.id} ({total} total in store).")
        file_log.info(f"sent + marked shown: principle id={p.id}")
    else:
        _log("Send failed -- NOT marking as shown, will retry next time it's picked.")
        file_log.error(f"send failed for principle id={p.id}, not marking as shown")


if __name__ == "__main__":
    run_random_resurface()
