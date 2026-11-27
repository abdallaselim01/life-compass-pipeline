"""
Manually-enhanced screenshot ingestion.

Design context (see README.md's strategic review on visual reinforcement for
the full trail): Google's image-generation models (Gemini's "Nano Banana"
family) turned out to have NO free tier at all -- confirmed against
Google's own official pricing page, not just a quota hiccup -- and every
other provider checked (Hugging Face, Cloudflare, OpenAI) either has no
real daily-free image-editing quota either, or the free one available
(Cloudflare's Stable Diffusion 1.5) was judged too unreliable at preserving
exact numbers/text to trust for this use case. The accepted alternative:
the USER does the "redraw as a clean flat graphic" step themselves, by hand,
using whichever AI image tool they like (Gemini app, ChatGPT, etc.), and
this module's only job is to pick up whatever finished images they drop in
one folder and match them to principles semantically -- see
enrichment.match_principles_to_images (the model is sent the picture itself).

Deliberately simple by design (this was originally a more complex
per-video folder + raw/enhanced pairing scheme, simplified after feedback):
ONE flat incoming folder, no naming convention required (the model's own
output filename is used as-is), and the existing time-checkpoint pattern
(see the original screenshot-checkpoint design) tells "images for the
video just watched" apart from anything older. The user's workflow is:
screenshot during the video (whenever), manually enhance each one on their
own time via any AI tool, save the results into this one folder, THEN
click the "finished" bookmarklet -- so by the time the pipeline runs,
everything is already complete and the milk-message photo album is never
sent partial.
"""
from __future__ import annotations

import filecmp
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .logging_setup import get_logger
from . import store

file_log = get_logger("images")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENHANCED_DIR = PROJECT_ROOT / "data" / "enhanced_images"
# Files an image replaced (the same image dropped again) are kept here, never deleted.
REPLACED_DIR = ENHANCED_DIR / "replaced"

_CHECKPOINT_KEY = "manual_enhanced_checkpoint"


def _incoming_dir() -> Path:
    """Where the user drops their manually-enhanced images. A single flat
    folder inside the project (not Downloads directly) -- see README:
    the user moves finished images here themselves once they're done
    enhancing all of them for a video, then clicks the bookmarklet.
    Overridable via MANUAL_ENHANCED_DIR in .env."""
    configured = os.environ.get("MANUAL_ENHANCED_DIR")
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "data" / "manual_enhanced_incoming"


@dataclass
class ProcessedImage:
    enhanced_path: Path
    # Set when the file has the same name as an image already stored for the video: that row is replaced in place
    # (by pipeline.py) instead of a second row being added. (Images are no longer described: matching and captioning are
    # sent the picture itself.)
    replaces_image_id: int | None = None


def _dest_name(video_id: str, file_name: str) -> str:
    """The name a dropped file gets in ENHANCED_DIR: '<video_id>_<name>'. A file that already starts with that prefix
    (taken back out of ENHANCED_DIR and dropped again) keeps its name instead of getting the prefix twice."""
    prefix = f"{video_id}_"
    return file_name if file_name.startswith(prefix) else prefix + file_name


def read_image(path) -> tuple[bytes, str]:
    """(bytes, mime type) of an image file: the picture itself, as the match and caption calls receive it. One mime rule for
    every call (png is png, anything else is sent as jpeg). Raises OSError when the file cannot be read."""
    path = Path(path)
    return path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _discover_new_images() -> list[Path]:
    """Everything in the incoming folder newer than the last video's
    checkpoint. Uses each file's **creation time** (st_ctime), not
    modification time (st_mtime) -- discovered the hard way: Windows
    preserves a file's original LastWriteTime (st_mtime) across copy/paste,
    so an image the user just pasted into this folder can still carry a
    modification timestamp from whenever it was first saved elsewhere
    (e.g. days earlier in Downloads), making it look "old" to a checkpoint
    based on mtime and get silently skipped. Creation time, in contrast,
    genuinely updates to "now" the moment a file lands in a given folder on
    Windows (NTFS), which is what "did the user just put this here" needs.
    (Note: st_ctime means something different on Linux -- metadata change
    time, not creation time -- but this project is Windows-only per
    HANDOFF.md, so that portability caveat doesn't apply here.)

    Same time-based separation logic as the original screenshot-checkpoint
    design otherwise: the user handles one video at a time, so this is
    enough, and a missed/misattributed file is a minor, non-destructive
    loss.

    On the very first run ever, the checkpoint is initialized to "now" and
    this returns empty -- so a fresh install doesn't sweep any leftover
    files already sitting in the folder from before this feature existed."""
    folder = _incoming_dir()
    folder.mkdir(parents=True, exist_ok=True)

    checkpoint_str = store.get_setting(_CHECKPOINT_KEY)
    if checkpoint_str is None:
        store.set_setting(_CHECKPOINT_KEY, datetime.now().isoformat())
        file_log.info("_discover_new_images: no checkpoint yet -- initializing to now, returning no results")
        return []

    checkpoint = datetime.fromisoformat(checkpoint_str)

    found = []
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            ctime = datetime.fromtimestamp(f.stat().st_ctime)
        except OSError:
            continue
        if ctime > checkpoint:
            found.append(f)
    found.sort(key=lambda p: p.stat().st_ctime)
    file_log.info(f"_discover_new_images: checkpoint={checkpoint_str} found={[f.name for f in found]}")
    return found


def _discover_replacements(video_id: str) -> list[Path]:
    """Files in the incoming folder, of any age, whose destination name matches an image already stored for this
    video: the user dropped the same image again to replace it. Age does not matter: moving a file inside a drive
    keeps its creation time, which the checkpoint rule above would take for "old"."""
    folder = _incoming_dir()
    if not folder.is_dir():
        return []
    found = []
    for f in folder.iterdir():
        if not f.is_file() or f.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        if store.get_image_by_path(video_id, str(ENHANCED_DIR / _dest_name(video_id, f.name))) is not None:
            found.append(f)
    found.sort(key=lambda p: p.name)
    return found


def process_pending_images(video_id: str, log=print, replacements_only: bool = False) -> list[ProcessedImage]:
    """Picks up every manually-enhanced image dropped since the last video and archives it under a video-scoped name. No
    model is called here: matching and captioning are sent the picture itself later (pipeline.py). No enhancement happens
    here -- that step is done by the user, by hand, before they click the bookmarklet (see module docstring). Never raises --
    an image that cannot be moved is left in place for the next run; they are images the user deliberately curated, so none
    is dropped.

    An image dropped again under the name of one already stored for this video REPLACES it (the old file is kept
    under replaced/, the caller replaces the row in place: ProcessedImage.replaces_image_id), whatever the file's age.
    With replacements_only, only such files are taken and the checkpoint is left alone (used for a video that is
    already up to date, so images meant for the next video are never swept into it)."""
    now = datetime.now()
    images = [] if replacements_only else _discover_new_images()
    for f in _discover_replacements(video_id):
        if f not in images:
            images.append(f)

    if not images:
        if not replacements_only:
            store.set_setting(_CHECKPOINT_KEY, now.isoformat())
        return []

    log(f"-> Found {len(images)} manually-enhanced image(s) for this video, processing ...")
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ProcessedImage] = []
    for img in images:
        try:
            dest_name = _dest_name(video_id, img.name)
            dest_path = ENHANCED_DIR / dest_name
            existing = store.get_image_by_path(video_id, str(dest_path))
            if existing is not None and dest_path.exists() and filecmp.cmp(str(img), str(dest_path), shallow=False):
                # Dropped again byte for byte unchanged (audit 2026-11-22: 12 such copies had piled up): the stored file already is
                # this image, so no second copy is kept in replaced/ (nothing is lost); the incoming duplicate is simply taken in.
                img.unlink()
                log(f"  = {img.name} is identical to the stored image id={existing.id} (no archive copy needed)")
            elif existing is not None and dest_path.exists():
                # The same image dropped again, but changed: the file it replaces is kept (never deleted) under replaced/.
                REPLACED_DIR.mkdir(parents=True, exist_ok=True)
                kept = REPLACED_DIR / f"{now:%Y%m%d_%H%M%S}_{dest_name}"
                shutil.copy2(str(dest_path), str(kept))
                shutil.copy2(str(img), str(dest_path))
                img.unlink()
                log(f"  ~ {img.name} replaces image id={existing.id} (the old file is kept as {kept.name})")
            else:
                shutil.move(str(img), str(dest_path))
                log(f"  + {img.name} -> {dest_name}")

            results.append(ProcessedImage(enhanced_path=dest_path,
                                          replaces_image_id=existing.id if existing is not None else None))
        except Exception as e:
            file_log.exception(f"failed processing manually-enhanced image {img}")
            log(f"  x {img.name}: unexpected error ({e}), leaving in place for next run")

    if not replacements_only:
        store.set_setting(_CHECKPOINT_KEY, now.isoformat())
    log(f"  Processed {len(results)} image(s).")
    return results
