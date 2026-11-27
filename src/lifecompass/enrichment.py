"""
Enrichment around the principles: matching manually curated images to them (the model is sent the
picture itself) and captioning a standalone image from the transcript passage that explains it.
(Linking principles to each other was removed 2026-11-16 by design decision; the links already stored stay as data.)

None of this decides what a principle is -- that is the production core
(production.py). These steps run after the principles exist.
"""
from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel

from .llm import (
    StageSkippedError,
    _MODEL_TIMEOUT_SECONDS,
    _chain_with_key_failover,
    _gemini_config_kwargs,
    _get_models,
    max_quality_config_kwargs,
    media_resolution_level,
    _parse_structured,
)
from .logging_setup import get_logger
from .principle_extractor import conform_to_transcript, split_at_unclear
from .runlog import block as _block, detail as _detail
from .stage_fingerprints import sha16

file_log = get_logger("enrichment")


def _parsed_or_raise(response, schema):
    """An answer that cannot be read as `schema` is a failure of that model: raising lets the chain try the next one."""
    parsed = _parse_structured(response, schema)
    if parsed is None:
        raise RuntimeError(f"could not parse the response into {schema.__name__}")
    return parsed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Screenshot images (see images.py) --------------------------------------
# Added after a strategic review of visual reinforcement (see README): the
# system has no way to know WHEN a supporting paper/drawing appeared in the
# video (no transcript cue), so the user takes the screenshot themselves.
# The match and caption calls below are sent the IMAGE ITSELF and link it to the
# right principle(s) by MEANING, not by timing. (Until 2026-11-10 a separate
# "describe" call turned each image into text first; that step lost content the
# picture really had and is gone. The old
# AI "redraw" prompt and enhance_image() were removed 2026-10-01: Google's image
# models have no free tier, and enhancement is done manually -- see
# MANUAL_ENHANCED_DIR.)

# The reply shape is stated in the prompt itself, so it is part of the stage fingerprint: no JSON-Schema text is appended to the
# match and caption requests (as for extract).
IMAGE_MATCH_PROMPT_TEMPLATE = """\
These are a list of principles extracted from a video and a list of images taken from the same video (the images themselves are attached
below in order, and the number of each image is written before it). Your task: for each principle, decide which image(s) from the list (if any)
have content that is really related to the content of that principle (the same topic, the same number/idea being discussed). One image may be
linked to more than one principle if it really serves more than one idea. Do not link an image to a principle unless the relation in content
is clear - not every image has to be linked to something, and not every principle has to have an image.

Return ONLY a valid JSON object in this shape: {{"matches": [{{"principle_index": 0, "image_indices": [0, 1]}}]}}, using the numbers printed before each principle and each image. A principle with no image is left out.

Principles:
{principles}

Images (attached in order):
"""

# --- Standalone image captioning (dedicated, single-purpose) ---------------
# Added 2026-09-07: an image can be manually curated by the user but end up
# matched to NO principle (see match_principles_to_images) -- it may still
# illustrate real informational content (an economic/historical/social fact
# the speaker explained) worth surfacing on its own during random resurface,
# just not as a "life principle". This pass finds the transcript passage
# that actually explains the image and captions it in the speaker's own
# words -- same fidelity standard as everywhere else in this system, never
# Gemini's own invented explanation of the image.
STANDALONE_IMAGE_CAPTION_PROMPT = (PROJECT_ROOT / "prompts" / "standalone_image_caption_prompt.txt").read_text(encoding="utf-8")


class ImageMatchItem(BaseModel):
    principle_index: int
    image_indices: list[int] = []


class ImageMatchResult(BaseModel):
    matches: list[ImageMatchItem] = []


class CaptionResult(BaseModel):
    caption: str = ""


ImageInput = tuple[bytes, str]      # (the picture's bytes, its mime type): see images.read_image


def match_principles_to_images(
    principle_texts: list[str], images: list[ImageInput], log=print
) -> dict[int, list[int]]:
    """One call per video (not per principle) -- cheap, since it's a single
    prompt: the principles as text and the IMAGES THEMSELVES (not a text that
    stands in for them), each one labelled with its number. Returns
    {principle_index: [image_index, ...]}, many-to-many, based purely on content
    match. An empty dict means the model answered and matched nothing.

    Raises StageSkippedError if the model could not be reached (every model
    in the chain failed) or its reply was unusable, so the caller can tell
    "nothing matches" apart from "matching could not be computed".
    """
    if not principle_texts or not images:
        return {}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (check your .env)")
    models = _get_models()

    principles_text = "\n".join(f"{i}. {t}" for i, t in enumerate(principle_texts))
    prompt = IMAGE_MATCH_PROMPT_TEMPLATE.format(principles=principles_text)
    file_log.debug(f"match_principles_to_images start: n_principles={len(principle_texts)} n_images={len(images)}")

    def make_build_call(client):
        def build_call(model_name: str):
            contents = [prompt]
            for i, (data, mime_type) in enumerate(images):
                contents.append(f"Image {i}:")
                contents.append(types.Part.from_bytes(
                    data=data, mime_type=mime_type, media_resolution=media_resolution_level("match")))
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    **_gemini_config_kwargs(ImageMatchResult),
                    **max_quality_config_kwargs("match"),
                ),
            )
            return _parsed_or_raise(response, ImageMatchResult)
        return build_call

    try:
        parsed, model_name = _chain_with_key_failover(make_build_call, models, log=log, context="match_principles_to_images",
            timeout=_MODEL_TIMEOUT_SECONDS)
    except Exception as e:
        file_log.warning(f"match_principles_to_images: all models failed: {e}")
        raise StageSkippedError(f"match_principles_to_images: all models failed: {e}") from e

    result: dict[int, list[int]] = {}
    for m in parsed.matches:
        if 0 <= m.principle_index < len(principle_texts):
            valid = [i for i in m.image_indices if 0 <= i < len(images)]
            if valid:
                result[m.principle_index] = valid
    file_log.info(f"match_principles_to_images result via {model_name}: {result}")
    return result


def caption_standalone_image(image: ImageInput, transcript_text: str, log=print) -> str:
    """For an image that matched no principle: the model is sent the picture itself together with the transcript, finds
    the transcript passage that actually explains it and captions it in the speaker's own words. See the module-level
    comment above STANDALONE_IMAGE_CAPTION_PROMPT for why this exists.

    The caption goes through the same literal-text guard as the principles (principle_extractor.conform_to_transcript):
    it is stored as one continuous passage of the transcript (stutter taken out, a small gap repaired to the transcript's own
    words); a text with no real anchor in the transcript is NOT stored and is written whole in the run log. Before 2026-11-10
    the caption was stored exactly as the model wrote it.

    Returns "" when there is nothing to caption (no picture, no API key, an empty caption, or one that is not from the
    transcript). The unclear-speech marker is never stored: a caption that contains it keeps its longest side (a caption is one
    passage), whatever is cut is in the run log.

    Raises StageSkippedError when the model could not be reached (all
    models failed) or its reply was unusable. Before 2026-09-29 this
    returned "" in that case too, which left the image uncaptioned -- and
    an uncaptioned image is never surfaced by resurface.py
    (pick_standalone_images_for_resurfacing only selects captioned ones) --
    forever, silently. Now the caller queues it for retry instead.
    """
    image_bytes, mime_type = image
    if not image_bytes:
        return ""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        file_log.error("caption_standalone_image called but GEMINI_API_KEY is not set")
        return ""
    models = _get_models()

    prompt = STANDALONE_IMAGE_CAPTION_PROMPT.format(transcript=transcript_text)
    file_log.debug(f"caption_standalone_image start: picture of {len(image_bytes)} bytes")

    def make_build_call(client):
        def build_call(model_name: str):
            log(f"  trying {model_name} ...")
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes, mime_type=mime_type, media_resolution=media_resolution_level("caption")),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    **_gemini_config_kwargs(CaptionResult),
                    **max_quality_config_kwargs("caption"),
                ),
            )
            return _parsed_or_raise(response, CaptionResult)
        return build_call

    try:
        parsed, model_name = _chain_with_key_failover(make_build_call, models, log=log, context="caption_standalone_image",
            timeout=_MODEL_TIMEOUT_SECONDS)
    except Exception as e:
        log(f"    x caption_standalone_image: all models failed: {e}")
        file_log.warning(f"caption_standalone_image: all models failed: {e}")
        raise StageSkippedError(f"caption_standalone_image: all models failed: {e}") from e

    caption = parsed.caption.strip()
    file_log.info(f"caption_standalone_image result via {model_name}: {caption[:80]!r}")
    if not caption:
        return ""
    fixed = conform_to_transcript(transcript_text, caption)
    if fixed is None:
        log("  ! The caption the model wrote is not from the transcript, so it is not stored (its full text is in the run log).")
        file_log.warning("caption_standalone_image: the caption is not from the transcript -- not stored")
        _detail(f"CAPTION LEFT OUT: fewer than half of its words are in the transcript ({len(caption.split())} words); it is NOT stored.")
        _block("CAPTION LEFT OUT (as the model wrote it)", caption)
        return ""
    new_text, how = fixed
    if how != "as is":
        _detail(f"CAPTION {how.upper()}: {len(caption.split())} word(s) as the model wrote it -> {len(new_text.split())} word(s) of the transcript")
        _block(f"CAPTION as the model wrote it (before: {how})", caption)
    pieces = split_at_unclear(new_text)
    if pieces != [new_text]:        # the marker for speech nobody could make out is never part of a caption (a caption is ONE passage)
        _detail(f"CAPTION contains the unclear-speech marker -> {len(pieces)} passage(s) on its sides"
                + ("; the longest is kept, the marker is not stored" if pieces else "; nothing is left, no caption"))
        _block("CAPTION as it was before the split at the unclear-speech marker", new_text)
        if not pieces:
            return ""
        new_text = max(pieces, key=lambda piece: len(piece.split()))
    return new_text


def post_stage_prompt_fps() -> dict:
    """Prompt fingerprints (SHA-256[:16], CRLF -> LF) of the stages that run after the principles exist: match,
    caption."""
    def norm(text: str) -> str:
        return text.replace("\r\n", "\n")

    return {
        "match": sha16(norm(IMAGE_MATCH_PROMPT_TEMPLATE)),
        "caption": sha16(norm(STANDALONE_IMAGE_CAPTION_PROMPT)),
    }
