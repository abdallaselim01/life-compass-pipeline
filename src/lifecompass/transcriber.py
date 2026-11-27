"""
Stage 1 -- listen. Turn the speaker's audio into the source document.

A multimodal model hears the WHOLE recording, in one request, and writes what was said, as paragraphs (one paragraph per line). The
recording is never cut (design decision 2026-10-14 and 2026-11-09: the speaker does not stop talking, so a cut lands inside his flow, and the
whole talk keeps his flow, terms and context); one too big for an inline request is uploaded instead (llm.generate_audio_text).
There is no text-repair step after it: the model heard the speech, so it does not have to guess at recognition errors. The one mechanical
check is that the reply is not empty (if it is, the same ask is made once more). The amount of speech is only measured and reported: code
does not decide whether a transcript is "enough", nor which of two transcripts is better.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import audio as audio_mod
from . import llm
from .arabic import strip_diacritics
from .document import SourceDocument, Unit
from . import tunables
from .runlog import detail as _detail, block as _block, sha16 as _sha16

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_WORDS_PER_MINUTE = tunables.get("transcript", "min_words_per_minute")   # below this, the model almost certainly skipped speech (config.toml [transcript])
MIN_SECONDS_FOR_RATE_CHECK = tunables.get("transcript", "min_seconds_for_rate_check")

_MARKDOWN = re.compile(r"[*_`#>]+")


def _load_prompt() -> str:
    return (PROJECT_ROOT / "prompts" / "transcribe_prompt.txt").read_text(encoding="utf-8")


def parse_transcript(text: str) -> list:
    """The model's lines as clean strings: one unit per non-empty line, without
    code fences, markdown marks or vowel signs."""
    out: list = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        body = _MARKDOWN.sub("", strip_diacritics(line))
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            out.append(body)
    return out


def _words(lines: list) -> int:
    return sum(len(t.split()) for t in lines)


def _read(name: str) -> str:
    """A prompt file of the listen stage without the line break at its end (every one of them is part of the stage fingerprint)."""
    return (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8").rstrip("\n")


def _contents(title: str, reminder: bool) -> str:
    """The request that goes with the recording: the title and the instruction (prompts/listen_request.txt), plus the reminder
    (prompts/listen_retry_note.txt) on the second attempt only."""
    text = _read("listen_request.txt").format(title=title)
    if reminder:
        text += "\n" + _read("listen_retry_note.txt")
    return text


def transcribe_video(video_id: str, title: str, audio_path: Path, *, log=print) -> SourceDocument:
    system = _load_prompt()
    seconds = audio_mod.audio_duration(audio_path)
    llm.set_audio_minutes(seconds / 60)                  # the wait limits of listen and extract grow with the recording (llm.scaled_long_wait)
    data = audio_mod.recording_bytes(audio_path)
    log(f"  Listening: the whole recording in one piece, {seconds / 60:.1f} min of audio.")
    _detail(f"LISTEN start: video_id={video_id} title={title!r} audio={audio_path} ({len(data)} bytes sha256[:16]={_sha16(data)}) "
            f"duration={seconds:.1f}s")

    lines, model = [], ""
    for attempt in (1, 2):
        prompt = _contents(title, reminder=attempt == 2)
        text, model = llm.generate_audio_text(data, prompt, system=system, context="listen", log=log)
        lines = parse_transcript(text)
        _detail(f"LISTEN attempt {attempt}: model={model}: raw reply {len(text)} chars, "
                f"{len(text.splitlines())} line(s) -> {len(lines)} paragraph(s) kept by parse_transcript "
                f"(code fences, markdown marks and vowel signs stripped, blank lines dropped)")
        if lines:
            break
        log("    ! the reply had no text lines" + (" -- asking again" if attempt == 1 else ""))

    words = _words(lines)
    wpm = words / (seconds / 60) if seconds else 0
    low_rate = seconds >= MIN_SECONDS_FOR_RATE_CHECK and wpm < MIN_WORDS_PER_MINUTE
    if low_rate:
        log(f"    ! only {wpm:.0f} words/min came back -- worth a look (reported, not acted on).")
    log(f"    {len(lines)} lines, {words} words ({wpm:.0f}/min).")
    _block(f"LISTEN paragraphs as kept (document entries 0-{len(lines) - 1})",
           "\n".join(f"[{k}] {t}" for k, t in enumerate(lines)))

    units = [Unit(i, s) for i, s in enumerate(lines)]
    stats = {
        "minutes": round(seconds / 60, 1),
        "units": len(units),
        "words": words,
        "words_per_minute": round(wpm),
        "unclear_markers": sum(u.text.count("[غير واضح]") for u in units),
        "low_rate": low_rate,
        "models": sorted({model}),
    }
    doc = SourceDocument(video_id=video_id, units=units, stats=stats)
    _detail(f"LISTEN done: stats={stats} doc_hash={doc.doc_hash}")
    _block("FULL TRANSCRIPT DOCUMENT (every entry exactly as stored in the database)",
           "\n".join(f"[{u.index}] {u.text}" for u in doc.units))
    return doc
