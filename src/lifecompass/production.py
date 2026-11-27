"""
The principle-production core: a video in, verified principles out.

    audio  --listen-->   source document (paragraphs: what he SAID)
           --extract-->  principles (one prompt, prompts/extract_prompt.txt: each principle one continuous passage of the transcript),
                         through the literal-text guard in principle_extractor

(The earlier "staged" chain -- select, edit and a cold reader over numbered units -- was removed.)

No database, no Telegram: the caller passes a callback that persists the transcript as soon as it exists, so a failed stage resumes where it
stopped. Any stage may raise llm.StageSkippedError, tagged with `.stage` ("listen" | "extract").
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import audio as audio_mod
from . import llm
from .document import SourceDocument
from .principle_extractor import extract_principles
from .runlog import detail as _detail
from .transcriber import transcribe_video


@dataclass
class PrincipleItem:
    principle: str
    domain: str


@dataclass
class ProductionResult:
    doc: SourceDocument
    items: list
    model_used: str


def _tag(stage: str, err: Exception) -> Exception:
    err.stage = stage
    return err


def run_production(video_id: str, title: str, *, log=print,
                   doc: SourceDocument | None = None,
                   audio_path: Path | None = None,
                   on_document=None) -> ProductionResult:
    if doc is None:
        try:
            _detail(f"STAGE listen start: video_id={video_id} title={title!r}")
            t_listen = time.monotonic()
            path = audio_path or audio_mod.download_audio(video_id)
            if not audio_path and getattr(audio_mod, "last_note", ""):
                log(f"  Audio: {audio_mod.last_note}")
            _detail(f"  audio file in use: {path} ({Path(path).stat().st_size if Path(path).exists() else 'n/a'} bytes); "
                    f"note={getattr(audio_mod, 'last_note', '')!r}")
            doc = transcribe_video(video_id, title, path, log=log)
            _detail(f"STAGE listen finished in {time.monotonic() - t_listen:.1f}s: {len(doc.units)} paragraph(s), "
                    f"stats={doc.stats}")
        except llm.StageSkippedError as e:
            raise _tag("listen", e)
        except (RuntimeError, OSError) as e:       # download / ffmpeg trouble: retry later
            raise _tag("listen", llm.StageSkippedError(f"audio: {e}")) from e
        if on_document:
            on_document(doc)
    if doc.stats.get("low_rate"):
        log("  ! far less speech than expected came back for this recording (reported, not acted on).")

    _detail(f"STAGE extract start: {len(doc.units)} paragraph(s), doc_hash={doc.doc_hash}")
    t_extract = time.monotonic()
    try:
        found, model = extract_principles(doc, title, log=log)
    except llm.StageSkippedError as e:
        _detail(f"STAGE extract FAILED after {time.monotonic() - t_extract:.1f}s: {e}")
        raise _tag("extract", e)
    _detail(f"STAGE extract finished in {time.monotonic() - t_extract:.1f}s: {len(found)} principle(s) by {model}")
    items = [PrincipleItem(text, domain) for text, domain in found]
    models = sorted(set(doc.stats.get("models", []) + [model]))
    return ProductionResult(doc, items, "|".join(models) or "unknown")
