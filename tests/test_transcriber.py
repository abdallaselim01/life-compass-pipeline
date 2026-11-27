"""Stage 1 (listen): parsing the model's lines, hearing the WHOLE recording in one request, and failure handling -- with the audio model
replaced by a fake. The recording is never cut (design decision 2026-10-14 and 2026-11-09)."""
import inspect
from pathlib import Path

import pytest

from lifecompass import audio as audio_mod
from lifecompass import llm
from lifecompass.transcriber import parse_transcript, transcribe_video


@pytest.fixture(autouse=True)
def recording(monkeypatch):
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 900.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"the-whole-recording")


def reply(n_lines=40, words_per_line=12, tag="س"):
    return "\n".join(" ".join(f"{tag}{i}_{w}" for w in range(words_per_line)) + "." for i in range(n_lines))


# ---------------------------------------------------------------- parsing
def test_each_line_becomes_a_unit():
    assert parse_transcript("اول.\nتاني.") == ["اول.", "تاني."]


def test_decorations_and_blank_lines_are_dropped():
    assert parse_transcript("```\n**كلام** مهم.\n\n```") == ["كلام مهم."]


def test_vowel_marks_are_stripped():
    assert parse_transcript("بَدْخُل هِنا.") == ["بدخل هنا."]


# ------------------------------------------------------------- transcribing
def test_the_whole_recording_is_heard_in_one_request_and_its_lines_become_units():
    seen = []

    def fake_audio(audio, prompt, system, context):
        seen.append((audio, context))
        return reply(tag="ص")

    llm.set_test_backend(audio=fake_audio)
    doc = transcribe_video("v", "عنوان", Path("x.mp3"))
    assert seen == [(b"the-whole-recording", "listen")]
    assert len(doc.units) == 40 and [u.index for u in doc.units] == list(range(40))
    assert doc.stats["minutes"] == 15.0 and "chunks" not in doc.stats


def test_the_listener_gets_only_the_title_and_the_instruction_and_there_is_no_glossary():
    """The speaker glossary was removed on 2026-11-18 (design decision). The request is the title and the instruction to transcribe,
    plus the reminder on the second attempt only."""
    import importlib.util

    from lifecompass import transcriber
    assert transcriber._contents("عنوان", reminder=False) == "Video title: عنوان\nTranscribe the attached recording in full."
    assert transcriber._contents("{x}", reminder=False).startswith("Video title: {x}\n")
    assert transcriber._contents("عنوان", reminder=True).startswith(transcriber._contents("عنوان", reminder=False) + "\n[Note:")
    assert importlib.util.find_spec("lifecompass.glossary") is None


def test_little_speech_is_only_reported_never_asked_again_or_replaced():
    calls = []

    def fake(audio, prompt, system, context):
        calls.append(prompt)
        return reply(n_lines=2, words_per_line=3)

    llm.set_test_backend(audio=fake)
    doc = transcribe_video("v", "t", Path("x.mp3"))
    assert len(calls) == 1                                # code does not judge "enough" and re-ask
    assert len(doc.units) == 2 and doc.stats["low_rate"] is True


def test_enough_speech_is_not_flagged():
    llm.set_test_backend(audio=lambda *a: reply(n_lines=60, words_per_line=20))
    assert transcribe_video("v", "t", Path("x.mp3")).stats["low_rate"] is False


def test_an_empty_reply_is_asked_for_once_more():
    replies = iter(["```\n```", reply()])
    calls = []

    def fake(audio, prompt, system, context):
        calls.append(prompt)
        return next(replies)

    llm.set_test_backend(audio=fake)
    doc = transcribe_video("v", "t", Path("x.mp3"))
    assert len(calls) == 2 and "[Note:" in calls[1] and "[Note:" not in calls[0]
    assert len(doc.units) == 40


def test_the_first_parseable_answer_is_used_as_it_is():
    """No "longer one wins": whatever the model returned first is the transcript."""
    replies = iter([reply(n_lines=5, words_per_line=4), reply()])
    llm.set_test_backend(audio=lambda *a: next(replies))
    assert len(transcribe_video("v", "t", Path("x.mp3")).units) == 5


def test_unclear_markers_are_counted():
    llm.set_test_backend(audio=lambda *a: "كلام [غير واضح] تاني.")
    assert transcribe_video("v", "t", Path("x.mp3")).stats["unclear_markers"] == 1


def test_there_is_no_way_to_cut_the_recording():
    for name in ("plan_chunks", "cut_chunk", "choose_cuts", "chunk_spans", "detect_silences", "chunk_seconds"):
        assert not hasattr(audio_mod, name), f"audio.{name} still exists"
    assert not {"spans", "cutter", "progress", "on_progress"} & set(inspect.signature(transcribe_video).parameters)
