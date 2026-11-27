"""
Ingest pipeline: one video URL in, extracted+stored principles out.
Mirrors the language pipeline's pipeline.py structure (same per-run logging
pattern to logs/), but the steps after extraction are different: dedup +
SQLite insert, instead of TTS + AnkiConnect push.
"""
from __future__ import annotations

import html
import os
import time
import threading
import traceback
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import audio as _audio
from .audio import video_info, extract_video_id
from .enrichment import (
    match_principles_to_images,
    caption_standalone_image,
    post_stage_prompt_fps,
)
from .llm import reset_exhausted_models, StageSkippedError, api_calls_made, set_audio_minutes
from .document import SourceDocument, DOC_VERSION
from .production import run_production
from .runlog import block as _block, clear_sink, detail as _detail, set_sink, snapshot_lines
from .images import process_pending_images, read_image
from .publisher import send_message, send_photo_group
from .logging_setup import get_logger
from . import stage_fingerprints as sf
from . import store, tunables

file_log = get_logger("pipeline")

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
# Automatic database copies taken before a stage is redone because its prompt changed.
AUTO_BACKUPS_DIR = Path(__file__).resolve().parent.parent.parent / "backups" / "auto"

# pending_retries stage for work that happens AFTER a video is published and
# must never block it: image-to-principle matching and standalone image
# captions. One queue row per video (the table
# is UNIQUE on video_id), so the row's items_json is a JSON object:
#   {"match": bool, "captions": [image ids]}
# (the keys of older versions -- "links", "describe" -- and a bare JSON list are ignored: those steps no longer exist).
SUPPLEMENTARY_STAGE = "supplementary"

# Telegram (or the network) could not take the album or the messages when the video finished: the rest waits in a row of its own
# (the key carries the suffix, because the table is UNIQUE on video_id and the video may also have a "supplementary" row) and
# scripts/retry_pending.py sends it later. items_json: {"photos": [file paths], "chunks": [message texts]}.
DELIVER_STAGE = "deliver"
DELIVER_KEY_SUFFIX = "#deliver"


@dataclass
class PipelineResult:
    ok: bool
    message: str
    added: int = 0
    skipped: int = 0


class _RunLogger:
    def __init__(self, url: str, outer_log):
        self._outer_log = outer_log
        self._t0 = time.monotonic()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOGS_DIR / f"{stamp}.log"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._write_raw(f"=== run started {datetime.now().isoformat()} ===")
        self._write_raw(f"url={url}")
        # From here every module's run detail (runlog.detail / runlog.block) lands in this file: the full prompts
        # and replies of every model call, each stage's input and output, every decision and every store write.
        self._sink = self.detail
        set_sink(self._sink)
        for line in snapshot_lines(Path(__file__).resolve().parents[2]):
            self.detail(line)

    def _write_raw(self, line: str) -> None:
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def detail(self, line: str) -> None:
        """Log-file only (never echoed to the console/listener): for long
        per-item dumps (full principle/source_quote text) that would flood the
        live output but are exactly what is needed to audit what each stage
        produced or dropped. Added 2026-10-03."""
        elapsed = time.monotonic() - self._t0
        self._write_raw(f"[+{elapsed:6.1f}s] {line}")

    def __call__(self, line: str) -> None:
        self._outer_log(line)
        elapsed = time.monotonic() - self._t0
        self._write_raw(f"[+{elapsed:6.1f}s] {line}")

    def exception(self, context: str, exc: Exception) -> None:
        self._write_raw(f"--- exception during {context} ---")
        self._write_raw(traceback.format_exc())

    def close(self, outcome: str) -> None:
        total = time.monotonic() - self._t0
        clear_sink(self._sink)
        self._write_raw(f"=== run finished: {outcome} (total {total:.1f}s) ===\n")
        self._fh.close()

def _format_retry_error(error) -> str:
    """HTML-escape and box a raw exception string in a <pre> block so it
    renders as an isolated, LTR monospace block in Telegram instead of
    jumbling into the surrounding Arabic RTL prose. 2026-09-26, after the
    raw inline dict-repr error text made the retry-queue notification
    genuinely hard to read (mixed-direction wrapping). Truncated so one
    huge error body cannot blow out the whole message."""
    text = str(error).strip()
    if len(text) > 500:
        text = text[:500] + " …"
    return f"<pre>{html.escape(text)}</pre>"



_STAGE_LABELS = {
    "listen": "listening to the recording and writing down the speaker's words (listen)",
    "extract": "extracting the principles (extract)",
}
PRODUCTION_STAGES = tuple(_STAGE_LABELS)


def _log_items(rlog, label: str, items: list) -> None:
    """Writes every item (its domain and full text) to this run's log file under `label`, so a missing idea can be traced to the stage
    that produced or dropped it."""
    rlog.detail(f"--- {label}: {len(items)} item(s) ---")
    for i, it in enumerate(items, 1):
        rlog.detail(f"  {i}. [{it.domain}] {it.principle}")


def _log_production(rlog, result) -> None:
    rlog.detail(f"--- document: {json.dumps(result.doc.stats, ensure_ascii=False)} ---")
    _log_items(rlog, "extracted principles (exactly as the model wrote them)", result.items)


HEARTBEAT_SECONDS = tunables.get("maintenance", "heartbeat_seconds")    # a run touches its queue row this often (store.touch_run; config.toml [maintenance])
ORPHAN_MAX_HOURS = tunables.get("maintenance", "orphan_max_hours")      # a "running" row older than this is an orphan whatever its process id says (ids are reused; config.toml [maintenance])


def _pid_alive(pid: int) -> bool:
    """Is a process with this id still running? (Windows API: os.kill(pid, 0) would END the process there.)"""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))) and code.value == 259     # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except Exception:  # noqa: BLE001 -- cannot tell: treat as alive, the age limit still frees the row
        return True


def run_is_live(p, now: datetime | None = None) -> bool:
    """True when the queue row `p` marks a run that is still going, so the retry task must leave it alone. A "running" row whose
    heartbeat is recent is live; an older one is an orphan (taken over by the retry task) once the process that wrote it is gone, or
    after ORPHAN_MAX_HOURS whatever. A machine that slept for an hour keeps its run: the process is still there when it wakes."""
    if p.last_error != store.RUNNING:
        return False
    now = now or datetime.now()
    try:
        age_minutes = (now - datetime.fromisoformat(p.last_attempt_at or p.created_at)).total_seconds() / 60
    except ValueError:
        age_minutes = float("inf")
    if age_minutes < tunables.get("maintenance", "orphan_run_minutes"):
        return True
    if age_minutes > ORPHAN_MAX_HOURS * 60:
        return False
    try:
        pid = int(json.loads(p.items_json).get("pid", 0))
    except (ValueError, AttributeError, TypeError):
        pid = 0
    return bool(pid) and _pid_alive(pid)


def run_stages(video_id: str, video_title: str, url: str, rlog,
               stage: str = "listen", rerun_note: str = "") -> PipelineResult:
    """The run itself (_run_stages) under a queue row that says it is running, with a heartbeat: a run that dies with its process
    (restart, crash, power loss) is found by scripts/retry_pending.py and finished, instead of being lost without a word
    (audit 2026-11-22). A normal ending replaces or deletes the row."""
    owns_row = store.mark_run_started(video_id, url, video_title, stage, os.getpid())
    stop = threading.Event()
    if owns_row:
        def beat() -> None:
            while not stop.wait(HEARTBEAT_SECONDS):
                try:
                    store.touch_run(video_id)
                except Exception:  # noqa: BLE001 -- a missed beat is not an error: the next one comes
                    pass
        threading.Thread(target=beat, daemon=True).start()
    try:
        return _run_stages(video_id, video_title, url, rlog, stage=stage, rerun_note=rerun_note)
    finally:
        stop.set()


def _note_audio_length(video_id: str) -> None:
    """Tells llm how long the recording is when it is already on disk (a run that resumes at 'extract'); the listen step tells it itself
    from the file it has just downloaded. An unknown length (no file) clears the last value: the wait limits then stay as they are.
    Never raises."""
    minutes = None
    try:
        path = _audio.AUDIO_ROOT / video_id / "full.mp3"
        if path.exists():
            minutes = _audio.audio_duration(path) / 60
    except Exception as e:  # noqa: BLE001 -- the length only sizes a wait: a failure here must never stop a run
        file_log.warning(f"could not read the length of the recording for {video_id}: {type(e).__name__}: {e}")
    set_audio_minutes(minutes)


def _run_stages(video_id: str, video_title: str, url: str, rlog,
                stage: str = "listen", rerun_note: str = "") -> PipelineResult:
    """listen -> extract, then store / images / Telegram.

    `stage` is where to start: a video that failed partway resumes from the stage that failed (the whole document is stored as soon as the
    recording is transcribed). On a StageSkippedError the stage is saved to store.pending_retries and a "queued for retry" Telegram message
    is sent -- failures retry themselves.
    """
    calls_before = api_calls_made()
    _note_audio_length(video_id)
    _detail(f"RUN STAGES start: video_id={video_id} title={video_title!r} url={url} start_stage={stage!r}")
    if stage == "listen" and not store.is_video_processed(video_id):
        # The recording was already transcribed and its document saved (a run that stopped before storing, or principles
        # cleared while the transcript stayed): reuse that exact text, so every run sees the same paragraphs and no
        # transcription call is spent. A document of another format version is not reused.
        saved = store.get_document_json(video_id)
        try:
            reusable = bool(saved) and json.loads(saved).get("version") == DOC_VERSION
        except (ValueError, AttributeError):
            reusable = False
        if reusable:
            rlog("  A saved transcript document exists for this video and it is not stored yet: "
                 "starting at 'extract' (no new transcription).")
            stage = "extract"
    doc = None
    if stage != "listen":
        blob = store.get_document_json(video_id)
        if blob:
            doc = SourceDocument.from_json(blob)
        else:
            stage = "listen"      # nothing saved to resume from: start over
    # The prompt fingerprints this run works with, taken now (before any stage starts) from the prompt files, chained on
    # what is recorded for the stages this run does not redo. {} when nothing can be recorded (unknown stage name).
    prompt_fps = {s_: sf.prompt_fp(s_) for s_ in sf.PRINCIPLE_STAGES}
    stored_eff = {s_: r_["effective_fp"] for s_, r_ in store.get_stage_fingerprints(video_id).items()}
    fingerprints = sf.effective_chain(prompt_fps, stage, stored_eff)
    for s_, (pfp_, efp_) in fingerprints.items():
        _detail(f"STAGE FINGERPRINT {s_}: prompt_fp={pfp_} settings_fp={sf.settings_fp(s_)} effective_fp={efp_}")

    def on_document(d) -> None:
        store.save_document(video_id, d.to_json(), d.doc_hash)
        if "listen" in fingerprints:                    # the transcript now exists: record which prompt wrote it
            try:
                store.save_stage_fingerprint(video_id, "listen", fingerprints["listen"][0], fingerprints["listen"][1],
                                             d.doc_hash, str(getattr(rlog, "path", "")),
                                             settings_fp=sf.settings_fp("listen"))
            except Exception as e:
                rlog(f"x Failed to record the listen fingerprint (non-fatal): {e}")
        try:
            store.insert_transcript(video_id, d.full_text())
        except Exception as e:
            rlog(f"x Failed to save the transcript text (non-fatal): {e}")

    def queue_retry(err: Exception, failed_stage: str) -> PipelineResult:
        label = _STAGE_LABELS.get(failed_stage, failed_stage)
        rlog(f"x {label} could not run: {err}")
        rlog.exception(failed_stage, err)
        file_log.exception(f"{failed_stage} could not run for url={url}")
        previous = next((p for p in store.get_pending_retries() if p.video_id == video_id), None)
        if previous is not None and previous.last_error == store.RUNNING:
            previous = None                      # only the "running" mark of this very run: the user has not been told yet
        # an attempt that sent nothing to any model (every one flagged out of quota) does not use up a retry
        store.save_pending_retry(video_id, url, video_title, failed_stage, "{}", "", str(err),
                                 count_attempt=api_calls_made() > calls_before)
        if previous is None or previous.stage != failed_stage:      # tell the user once, not on every 5-minute retry
            send_message(
                f"⏳ <b>The video is in the retry queue</b>\nVideo: {html.escape(video_title, quote=False)}\nStage: {label}\nReason:\n{_format_retry_error(err)}\nIt will be retried automatically later from the same stage; there is no need to press the bookmarklet again.",
                log=rlog,
            )
        rlog.close(f"queued for retry ({failed_stage})")
        return PipelineResult(ok=False, message=f"{failed_stage} could not run, queued for retry: {err}")

    rlog(f"-> Producing principles (listen -> extract), starting at '{stage}' ...")
    try:
        result = run_production(video_id, video_title, log=rlog, doc=doc, on_document=on_document)
    except StageSkippedError as e:
        return queue_retry(e, getattr(e, "stage", "listen"))
    except Exception as e:
        return queue_retry(e, stage)
    _log_production(rlog, result)
    rlog(f"  Models: {result.model_used}")
    rlog(f"  Principles extracted: {len(result.items)}")
    outcome = _store_and_publish(video_id, video_title, result.doc.full_text(), url, result.items,
                                 result.model_used, result.doc, rlog,
                                 fingerprints=fingerprints, rerun_note=rerun_note)
    # The recording is kept after success (design decision 2026-10-16): it may be needed again, and
    # audio.download_audio reuses it from disk instead of downloading again, even after the DB is zeroed.
    return outcome


def _store_and_publish(
    video_id: str,
    video_title: str,
    raw_text: str,
    url: str,
    items: list,
    model_used: str,
    doc,
    rlog,
    fingerprints: dict | None = None,
    rerun_note: str = "",
) -> PipelineResult:
    """Everything after the principles exist: image matching, storing,
    captions, Telegram. `raw_text` is the transcript text, used as
    context for image captions."""
    if not items:
        # a video can legitimately end with nothing to store (the model found no principle in the transcript)
        rlog("  No principle was extracted -- nothing will be stored for this video.")
        try:
            send_message(
                f"ℹ️ <b>{html.escape(video_title, quote=False)}</b>\nThe video was processed, but no principle was extracted from it, so nothing was stored.",
                log=rlog,
            )
        except Exception as e:
            rlog(f"x Failed to send the empty-result notice (non-fatal): {e}")
    _log_items(rlog, "final items (to be stored)", items)
    _detail(f"STORE phase start: {len(items)} item(s), model_used={model_used}, doc_hash={doc.doc_hash}")

    # The principles exist -- fetch this video's images (stored
    # during the original run, before extraction) fresh from the DB rather
    # than threading them through pending_retries, since they don't change
    # across retries.
    images, image_inputs = _readable_images(store.get_unlinked_images_for_video(video_id), rlog)
    image_ids = [img.id for img in images]
    _detail(f"STORE: {len(images)} unlinked image(s) available for this video, sent to the model as pictures: ids={image_ids}")

    image_matches: dict[int, list[int]] = {}
    match_failed = False
    if image_ids and items:
        try:
            image_matches = match_principles_to_images(
                [item.principle for item in items],
                image_inputs,
                log=rlog,
            )
            if image_matches:
                rlog(f"  Matched {len(image_matches)} principle(s) to at least one image.")
        except StageSkippedError as e:
            # 2026-09-29: used to be swallowed as "no matches" -- every image
            # stayed unlinked forever (observed: 0 of 5 linked after a 503
            # outage). Stays non-blocking; the matching is retried later.
            match_failed = True
            rlog(f"x Image matching could not run (principles save without image links for now, will retry): {e}")
            image_matches = {}
        except Exception as e:
            rlog(f"x Image matching failed (non-fatal, principles will save without image links): {e}")
            file_log.exception("match_principles_to_images failed")
            image_matches = {}

    _detail(f"STORE: image_matches (principle index -> image positions)={image_matches} match_failed={match_failed}")
    existing_texts = store.get_existing_principle_texts()
    added = 0
    skipped = 0

    for idx, item in enumerate(items):
        norm = " ".join(item.principle.strip().split()).lower()
        if norm in existing_texts:
            skipped += 1
            _detail(f"STORE: item {idx + 1} skipped, exact duplicate (normalized text) of a stored principle")
            continue
        existing_texts.add(norm)

        new_id = store.insert_principle(
            store.Principle(
                id=None,
                principle=item.principle,
                domain=item.domain,
                doc_hash=doc.doc_hash,
                video_title=video_title,
                video_url=url,
                video_id=video_id,
            )
        )
        added += 1
        _detail(f"STORE: item {idx + 1} stored as principle id={new_id} domain={item.domain} doc_hash={doc.doc_hash}")

        for img_idx in image_matches.get(idx, []):
            if img_idx < len(image_ids):
                store.link_principle_image(new_id, image_ids[img_idx])
                _detail(f"STORE: principle id={new_id} linked to image id={image_ids[img_idx]}")

    rlog(f"Added {added} new principles, skipped {skipped} duplicates.")

    failed_caption_ids: list[int] = []
    if images:
        unlinked = store.get_unlinked_images_for_video(video_id)
        if unlinked and match_failed:
            rlog("-> Skipping standalone captions for now: image matching failed, so it is not known yet which images stay unmatched (queued for retry).")
        elif unlinked:
            rlog(f"-> Captioning {len(unlinked)} unmatched image(s) for standalone resurfacing ...")
            for img in unlinked:
                try:
                    caption = caption_standalone_image(read_image(img.enhanced_path), raw_text, log=rlog)
                    if caption:
                        store.set_image_caption(img.id, caption)
                        _block(f"STORE: standalone caption saved for image id={img.id}", caption)
                    else:
                        rlog(f"  image id={img.id}: no caption stored (no passage of the transcript explains it, or what the model wrote was not from the transcript)")
                except StageSkippedError as e:
                    failed_caption_ids.append(img.id)
                    rlog(f"x caption_standalone_image could not run for image id={img.id} (will retry later): {e}")
                except Exception as e:
                    rlog(f"x caption_standalone_image failed for image id={img.id} (non-fatal): {e}")
                    file_log.exception(f"caption_standalone_image failed for image id={img.id}")

    file_log.info(f"process_video result: url={url} title={video_title!r} added={added} skipped={skipped} model_used={model_used}")
    _detail(f"STORE: video marked processed: added={added} skipped={skipped} model_used={model_used} doc_hash={doc.doc_hash}")
    for stage_name, (pfp_, efp_) in (fingerprints or {}).items():
        try:
            store.save_stage_fingerprint(video_id, stage_name, pfp_, efp_, doc.doc_hash, str(getattr(rlog, "path", "")),
                                         settings_fp=sf.settings_fp(stage_name))
            _detail(f"STORE: stage fingerprint recorded: {stage_name} prompt_fp={pfp_} effective_fp={efp_}")
        except Exception as e:
            rlog(f"x Failed to record the {stage_name} fingerprint (non-fatal): {e}")
    store.mark_video_processed(video_id, url, video_title, added, ok=True, model_used=model_used)
    # Successfully finished -- clear any queued retry for this video.
    for pending in store.get_pending_retries():
        if pending.video_id == video_id:
            store.delete_pending_retry(pending.id)

    # Supplementary work that could not run (image
    # matching, standalone captions) gets one queue entry so
    # scripts/retry_pending.py completes it later. Done AFTER the cleanup
    # above, which would otherwise delete it. Nothing here blocks or delays
    # what the user receives.
    if match_failed or failed_caption_ids:
        payload = {"match": match_failed, "captions": failed_caption_ids}
        store.save_pending_retry(
            video_id, url, video_title, SUPPLEMENTARY_STAGE, json.dumps(payload), model_used,
            "supplementary step failed (models unavailable)",
        )
        rlog(f"  Queued for later: image matching={'yes' if match_failed else 'no'}, {len(failed_caption_ids)} caption(s).")

    # match / caption are recorded as done only for what really finished here; a stage that was queued is recorded
    # by retry_supplementary when it completes it.
    done_post = []
    if not match_failed:
        done_post.append("match")
        if not failed_caption_ids:
            done_post.append("caption")
    record_post_fingerprints(video_id, tuple(done_post), rlog)

    undelivered: dict = {"photos": [], "chunks": []}   # what Telegram could not take now (a False answer, never a fake's None)
    if added > 0:
        if images:
            _detail(f"TELEGRAM photo album: {[str(img.enhanced_path) for img in images]} caption={video_title!r}")
            try:
                album_ok = send_photo_group(
                    [Path(img.enhanced_path) for img in images],  # DB stores this as TEXT/str -- send_photo_group needs Path objects (2026-09-26 fix)
                    caption=html.escape(video_title, quote=False),     # the caption is sent with parse_mode=HTML
                    log=rlog,
                )
            except Exception as e:
                album_ok = False
                rlog(f"x Failed to send photo album (non-fatal): {e}")
                file_log.exception("send_photo_group failed")
            if album_ok is False:
                undelivered["photos"] = [str(img.enhanced_path) for img in images]

        header = f"<b>{html.escape(video_title, quote=False)}</b>"
        if rerun_note:
            header += "\n" + html.escape(rerun_note, quote=False)
        items_text = [
            f"{i}. {html.escape(item.principle, quote=False)}"
            for i, item in enumerate(items, 1)
        ]

        TELEGRAM_LIMIT = 4000
        chunks: list[str] = []
        current = header
        for item_text in items_text:
            candidate = current + "\n\n" + item_text
            if len(candidate) > TELEGRAM_LIMIT:
                chunks.append(current)
                current = item_text
            else:
                current = candidate
        chunks.append(current)

        for k, chunk in enumerate(chunks, 1):
            if undelivered["photos"] or undelivered["chunks"]:   # the album (the principles never go out without their pictures,
                                                                 # user rule 2026-11-19) or an earlier part could not be sent:
                                                                 # the rest wait behind it, in order
                undelivered["chunks"].append(chunk)
                continue
            _block(f"TELEGRAM message {k}/{len(chunks)} (sent exactly as shown)", chunk)
            if send_message(chunk, log=rlog) is False:
                undelivered["chunks"].append(chunk)

    if undelivered["photos"] or undelivered["chunks"]:
        queue_delivery(video_id, url, video_title, undelivered, model_used, rlog)

    rlog.close(f"ok (added {added}, skipped {skipped})")
    return PipelineResult(
        ok=True,
        message=f"'{video_title}': added {added}, skipped {skipped}.",
        added=added,
        skipped=skipped,
    )


def _post_current(video_id: str) -> dict:
    """{stage: StageFp} of match / caption for the video as they would be recorded now ({} when the video has no
    recorded fingerprint yet for its last principle stage, so nothing can be said)."""
    last = sf.PRINCIPLE_STAGES[-1]
    recorded = store.get_stage_fingerprints(video_id)
    if last not in recorded:
        return {}
    return sf.post_current(recorded[last]["effective_fp"], post_stage_prompt_fps())


def _post_plan(video_id: str) -> sf.PostPlan:
    """Which of match / caption of a processed video are stale (prompt or API settings changed, or a stage before
    them, or an image description). A stage with no recorded fingerprint counts as stale (redone once)."""
    current = _post_current(video_id)
    if not current:
        return sf.PostPlan()
    recorded = store.get_stage_fingerprints(video_id)
    stored = {s: recorded[s]["effective_fp"] for s in sf.POST_STAGES if s in recorded}
    return sf.post_plan(stored, current, {s: recorded[s] for s in sf.POST_STAGES if s in recorded})


def record_post_fingerprints(video_id: str, stages=sf.POST_STAGES, rlog=print) -> None:
    """Records that the given post stages (match / caption) of the video were done with the prompts and API settings
    that are current now. Non-fatal: a failure only means the stage is done once more at the next check."""
    try:
        current = _post_current(video_id)
        for stage in stages:
            if stage in current:
                fp = current[stage]
                store.save_stage_fingerprint(video_id, stage, fp.prompt_fp, fp.effective_fp, "", "",
                                             settings_fp=fp.settings_fp)
    except Exception as e:
        rlog(f"x Failed to record the match / caption fingerprints (non-fatal): {e}")


def queue_delivery(video_id: str, url: str, video_title: str, undelivered: dict, model_used: str, rlog) -> None:
    """Telegram could not take everything when the video finished (the network or Telegram was down even after the longer waits of
    publisher.py): what is left is queued in its own row and scripts/retry_pending.py sends it later, in order. The principles are
    already stored. A row left over from an earlier run of the same video is replaced, so an old message never arrives after a
    newer one."""
    key = video_id + DELIVER_KEY_SUFFIX
    for pending in store.get_pending_retries():
        if pending.video_id == key:
            store.delete_pending_retry(pending.id)
    store.save_pending_retry(key, url, video_title, DELIVER_STAGE, json.dumps(undelivered, ensure_ascii=False), model_used,
                             "telegram could not be reached")
    rlog(f"  Telegram could not be reached: {len(undelivered['chunks'])} message(s)"
         f"{' and the photo album' if undelivered['photos'] else ''} queued, sent when Telegram answers again.")


def parse_delivery_payload(items_json: str) -> dict:
    """items_json of a "deliver" queue row -> {"photos": [file path, ...], "chunks": [message text, ...]}."""
    data = json.loads(items_json)
    if not isinstance(data, dict):
        data = {}
    return {"photos": [str(x) for x in data.get("photos", [])], "chunks": [str(x) for x in data.get("chunks", [])]}


def parse_supplementary_payload(items_json: str) -> dict:
    """items_json of a "supplementary" queue row -> normalized dict {"match", "captions"}. The keys of older versions
    ("links", "describe") and a bare list are ignored: those steps no longer exist."""
    data = json.loads(items_json)
    if not isinstance(data, dict):
        data = {}
    return {
        "match": bool(data.get("match", False)),
        "captions": [int(x) for x in data.get("captions", [])],
    }


def _readable_images(images: list, rlog) -> tuple[list, list]:
    """The images whose file can be read, and their (bytes, mime type) in the same order: the picture itself is what the
    match and caption calls are sent (no text stands in for it). A row whose file is
    gone cannot be sent to a model: it is logged and left out of matching and captioning, never silently."""
    kept, pictures = [], []
    for img in images:
        try:
            pictures.append(read_image(img.enhanced_path))
        except OSError as e:
            rlog(f"x image id={img.id}: its file cannot be read ({e}) -- left out of matching and captioning")
            continue
        kept.append(img)
    return kept, pictures


def _images_only_pass(video_id: str, url: str, rlog) -> str | None:
    """For a video whose stages are up to date: images dropped again under the same name replace the stored ones, and
    the replaced images are then matched to the video's principles from the picture itself. The transcript, the principles
    and Telegram are not touched. Returns a one-line summary, or None when nothing changed (or the video is waiting in
    the retry queue, which owns its images then)."""
    if any(p.video_id == video_id for p in store.get_pending_retries()):
        return None
    changed: list[int] = []
    try:
        for img in process_pending_images(video_id, log=rlog, replacements_only=True):
            store.replace_image(img.replaces_image_id, str(img.enhanced_path), "the same image was dropped again")
            changed.append(img.replaces_image_id)
    except Exception as e:
        rlog(f"x Replacing dropped images failed (non-fatal): {e}")
        rlog.exception("replace images", e)
    post = _post_plan(video_id)             # match / caption whose prompt, settings or inputs changed
    if not changed and not post.redo:
        return None
    redo = tuple(post.redo)
    if redo:
        rlog(f"  Stages to redo: {', '.join(redo)} ({post.reason_text()})")
        moved = store.archive_post_results(video_id, redo, post.reason_text())
        rlog(f"  Archived (not deleted): {moved}")
    payload = {"match": bool(changed) or "match" in redo, "captions": []}
    if "caption" in redo and not payload["match"]:
        payload["captions"] = [im.id for im in store.get_unlinked_images_for_video(video_id)]
    leftover = retry_supplementary(video_id, payload, rlog)
    if redo:            # the redone stages that finished are recorded here (retry_supplementary only records what its payload covered)
        left = leftover or {"match": False, "captions": []}
        done_now = [s_ for s_ in redo
                    if (s_ == "match" and not left["match"])
                    or (s_ == "caption" and not left["match"] and not left["captions"])]
        record_post_fingerprints(video_id, tuple(done_now), rlog)
    if leftover:
        store.save_pending_retry(video_id, url, store.get_processed_video_title(video_id), SUPPLEMENTARY_STAGE,
                                 json.dumps(leftover), "", "image refresh step failed (models unavailable)")
        rlog("  Queued for later: matching or captioning of the refreshed images.")
    if changed:
        return f"{len(set(changed))} image(s) refreshed and matched again (transcript, principles and Telegram untouched)."
    return f"{', '.join(redo)} redone ({post.reason_text()}); transcript, principles and Telegram untouched."


def retry_supplementary(video_id: str, payload: dict, rlog) -> dict | None:
    """Completes what parse_supplementary_payload describes. Returns the
    payload of whatever is STILL outstanding, or None when everything is
    done. Order: image matching; then standalone
    captions for the images that stay unmatched (captions are only attempted
    once matching is settled, so no quota is spent captioning an image that
    is about to be linked to a principle). The models are sent the pictures
    themselves; an image whose file cannot be read is left out, logged.
    Added 2026-09-29."""
    match_pending = payload["match"]
    caption_ids = list(payload["captions"])

    if match_pending:
        unlinked, pictures = _readable_images(store.get_unlinked_images_for_video(video_id), rlog)
        principles = store.get_principles_for_video(video_id)
        if unlinked and principles:
            try:
                matches = match_principles_to_images([p.principle for p in principles], pictures, log=rlog)
            except StageSkippedError as e:
                rlog(f"x image matching still failing: {e}")
            else:
                match_pending = False
                for p_idx, img_idxs in matches.items():
                    for i in img_idxs:
                        if 0 <= p_idx < len(principles) and 0 <= i < len(unlinked):
                            store.link_principle_image(principles[p_idx].id, unlinked[i].id)
                if matches:
                    rlog(f"  Matched {len(matches)} principle(s) to at least one image.")
        else:
            match_pending = False  # nothing left to match
        if not match_pending:
            caption_ids = [im.id for im in store.get_unlinked_images_for_video(video_id) if not im.standalone_caption]

    if not match_pending and caption_ids:
        raw_text = store.get_transcript(video_id) or ""
        by_id = {im.id: im for im in store.get_unlinked_images_for_video(video_id)}
        still: list[int] = []
        for cid in caption_ids:
            im = by_id.get(cid)
            if im is None or im.standalone_caption:
                continue  # linked meanwhile or already captioned
            try:
                caption = caption_standalone_image(read_image(im.enhanced_path), raw_text, log=rlog)
            except StageSkippedError as e:
                rlog(f"x caption_standalone_image still failing for image id={cid}: {e}")
                still.append(cid)
                continue
            except OSError as e:
                rlog(f"x image file for id={cid} cannot be read ({e}) -- dropped from the caption retry")
                continue
            if caption:
                store.set_image_caption(im.id, caption)
            else:
                rlog(f"  image id={cid}: no caption stored (no passage of the transcript explains it, or what the model wrote was not from the transcript)")
        caption_ids = still

    done = []
    if payload["match"] and not match_pending:
        done.append("match")
        if not caption_ids:
            done.append("caption")
    elif payload["captions"] and not match_pending and not caption_ids:
        done.append("caption")
    record_post_fingerprints(video_id, tuple(done), rlog)

    if match_pending or caption_ids:
        return {"match": match_pending, "captions": caption_ids}
    return None


def _queue_unreadable_video(url: str, error: Exception, rlog) -> None:
    """The very first step failed (YouTube or yt-dlp did not answer): the video goes into the retry queue like a failed stage, and
    the user is told once (audit 2026-11-22: it used to end without a word). The title is not known yet, so the video id stands in
    for it; scripts/retry_pending.py reads the real title first (title_is_placeholder)."""
    try:
        vid = extract_video_id(url)
    except ValueError:
        return                                    # not a video address: nothing to retry
    try:
        previous = next((p for p in store.get_pending_retries() if p.video_id == vid), None)
        store.save_pending_retry(vid, url, vid, "listen", "{}", "", str(error))
        if previous is None or previous.last_error == store.RUNNING:
            send_message(
                f"⏳ <b>The video is in the retry queue</b>\nVideo: {html.escape(url, quote=False)}\n"
                f"Stage: reading the video data from YouTube\nReason:\n{_format_retry_error(error)}\n"
                f"It will be retried automatically later; there is no need to press the bookmarklet again.",
                log=rlog,
            )
    except Exception as e:  # noqa: BLE001 -- the safety net must not fail the run it protects
        rlog(f"x Could not queue the unreadable video: {e}")


def title_is_placeholder(p) -> bool:
    """A queue row made by _queue_unreadable_video: its title is still the video id."""
    return p.title == p.video_id


def process_video(url: str, log=print, skip_if_processed: bool = True, trigger_info: str = "") -> PipelineResult:
    rlog = _RunLogger(url, log)
    rlog.detail(f"PROCESS VIDEO: url={url} skip_if_processed={skip_if_processed} trigger={trigger_info or 'direct call'}")
    file_log.info(f"process_video called: url={url} skip_if_processed={skip_if_processed}")

    # Fresh per-video: a model confirmed daily-exhausted earlier in THIS
    # run stays skipped for every remaining Gemini call below (extraction,
    # image matching, captions) without re-discovering it
    # from scratch each time -- but must be reset here so it doesn't
    # incorrectly carry over into a LATER video/day where the quota may
    # have reset (see extractor.reset_exhausted_models).
    reset_exhausted_models()

    rerun_plan = None
    if skip_if_processed:
        try:
            vid = extract_video_id(url)
        except ValueError:
            vid = None
        if vid and store.is_video_processed(vid):
            # A processed video is skipped only while every stage's recorded prompt fingerprint still matches the
            # prompt files; otherwise it is redone from the first stage whose prompt changed.
            recorded_fps = store.get_stage_fingerprints(vid)
            stored_eff = {s_: r_["effective_fp"] for s_, r_ in recorded_fps.items()}
            rerun_plan = sf.plan(stored_eff, sf.current_fingerprints(), recorded_fps)
            for line in rerun_plan.lines():
                rlog.detail(line)
            if not rerun_plan.needed:
                refreshed = _images_only_pass(vid, url, rlog)
                if refreshed:
                    rlog(f"  Video {vid} is up to date; {refreshed}")
                    rlog.close("ok (images refreshed)")
                    return PipelineResult(ok=True, message=f"Already processed -- {refreshed}")
                rlog(f"  Video {vid} already processed with the current prompts -- skipping (no Gemini calls spent).")
                rlog.close("ok (already processed)")
                return PipelineResult(ok=True, message="Already processed -- skipped.")
            rlog(f"  Video {vid} was processed, but: {rerun_plan.reason_text()}")
        # A video already queued in pending_retries has no processed_videos
        # row yet (that's only written on full success), so the check
        # above alone would miss it and re-run extraction from scratch --
        # wasting quota and risking duplicate/conflicting state with
        # whatever scripts/retry_pending.py is about to resume. Added
        # 2026-09-11 alongside the retry-queue feature itself.
        if vid:
            for pending in store.get_pending_retries():
                if pending.video_id == vid:
                    rlog(f"  Video {vid} is already queued for automatic retry (stage: {pending.stage}, attempt {pending.attempts}) -- skipping, not re-extracting.")
                    rlog.close("ok (already queued for retry)")
                    return PipelineResult(ok=True, message=f"Already queued for retry at stage '{pending.stage}' -- skipped.")

    try:
        rlog(f"-> Reading video info for {url} ...")
        video_id, video_title = video_info(url)
        rlog(f"  Video: {video_title}")
        rlog.detail(f"VIDEO info: id={video_id} title={video_title!r}")
    except Exception as e:
        rlog(f"x Could not read the video: {e}")
        rlog.exception("fetch", e)
        file_log.exception(f"Reading video info failed for url={url}")
        _queue_unreadable_video(url, e, rlog)
        rlog.close("failed (video info) -- queued for retry")
        return PipelineResult(ok=False, message=f"Could not read the video, queued for retry: {e}")

    # Manually-enhanced images (see images.py): the user's workflow is to
    # screenshot during the video, enhance each one by hand with any AI
    # image tool, and drop the finished results into one incoming folder
    # BEFORE clicking this bookmarklet -- so by the time this runs, every
    # image for this video is already complete (see README's strategic
    # review: Google's image models turned out to have no free tier at
    # all, confirmed against their official pricing page, and no other
    # provider checked had a free tier reliable enough to trust for
    # preserving exact numbers/text, so manual enhancement was accepted).
    # Non-fatal by design -- an image problem should never block the text
    # pipeline that has worked reliably all along.
    try:
        processed_images = process_pending_images(video_id, log=rlog)
    except Exception as e:
        rlog(f"x Image processing failed (non-fatal, continuing without images): {e}")
        rlog.exception("images", e)
        file_log.exception(f"Image processing failed for video_id={video_id}")
        processed_images = []

    # Images are stored regardless of whether any principle ends up linked
    # to them (insertion itself never fails silently the way a downstream
    # match might) -- image_ids[i] corresponds to processed_images[i].
    image_ids: list[int] = []
    for img in processed_images:
        if img.replaces_image_id is not None:       # the same image dropped again: the stored row is replaced in place
            store.replace_image(img.replaces_image_id, str(img.enhanced_path), "the same image was dropped again")
            image_ids.append(img.replaces_image_id)
        else:
            image_ids.append(store.insert_image(video_id, str(img.enhanced_path)))

    rlog.detail(f"IMAGES processed in this run: {len(processed_images)} new, stored with ids {image_ids}")
    for img, iid in zip(processed_images, image_ids):
        rlog.detail(f"IMAGE id={iid} path={img.enhanced_path}")
    start_stage, note = "listen", ""
    if rerun_plan is not None and rerun_plan.needed:
        start_stage, note = rerun_plan.rerun_from, rerun_plan.note()
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = store.backup_database(AUTO_BACKUPS_DIR / f"db_{stamp}_before_rerun_{video_id}_{start_stage}")
            rlog(f"  Database backed up before redoing from '{start_stage}': {backup}")
        except Exception as e:
            rlog(f"x Could not back the database up -- nothing was changed and the video was not redone: {e}")
            rlog.exception("auto backup", e)
            file_log.exception(f"auto backup failed for video_id={video_id}")
            rlog.close("failed (backup before redo)")
            return PipelineResult(ok=False, message=f"Could not back the database up, nothing changed: {e}")
        counts = store.archive_video_results(video_id, rerun_plan.reason_text(),
                                             archive_document=(start_stage == sf.PRINCIPLE_STAGES[0]))
        rlog(f"  Old results archived (not deleted): {counts}")
        try:
            old_captions = store.archive_post_results(video_id, ("caption",), rerun_plan.reason_text())
            rlog(f"  Old standalone captions archived (not deleted): {old_captions['captions']}")
        except Exception as e:
            rlog(f"x Archiving the old captions failed (non-fatal): {e}")
    return run_stages(video_id, video_title, url, rlog, stage=start_stage, rerun_note=note)
