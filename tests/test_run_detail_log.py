"""The run log must hold every detail of a run (what each model was sent and answered, each stage's input and
output, every decision and store write) so a run can be studied later from that one file."""
import re
from types import SimpleNamespace

import pytest

from lifecompass import audio as audio_mod
from lifecompass import llm, pipeline, runlog, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult



@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: None)
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(audio_mod, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    yield
    store._prepared.clear()


def heard(audio, prompt, system, context):
    # what the two pieces used to say, now said by one recording
    return "\n\n".join(f"فقرة {idx} فيها كلام.\n\nفقرة {idx} تانية فيها مبدأ." for idx in (1, 2))


def extract_ok(prompt, schema, system, context):
    return ExtractionResult(items=[
        ExtractedPrinciple(text="فقرة 2 تانية فيها مبدأ.", domain="Faith"),
        ExtractedPrinciple(text="فقرة 1 فيها كلام.", domain="NotADomain"),
        ExtractedPrinciple(text="  ", domain="Self"),
    ])


def test_the_llm_request_and_reply_are_recorded_in_full():
    seen = []
    runlog.set_sink(seen.append)
    try:
        llm._detail_request("json", "extract[1]", ["m1", "m2"], "SYSTEM TEXT", "USER TEXT",
                            schema=ExtractionResult)
        resp = SimpleNamespace(
            text="RAW REPLY TEXT",
            usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=22,
                                           thoughts_token_count=33, total_token_count=66),
            candidates=[SimpleNamespace(finish_reason="STOP")], prompt_feedback=None,
            first_piece_s=1.5, pieces=3, total_s=4.0)
        llm._detail_response("extract[1]", "m1", resp)
    finally:
        runlog.clear_sink(seen.append)
    log = "\n".join(seen)
    for needle in ["extract[1]", "chain=['m1', 'm2']", "thinking_level=HIGH", "max_output_tokens=65536", "SYSTEM TEXT",
                   "USER TEXT", "RAW REPLY TEXT", "'thoughts_token_count': 33", "finish_reason=['STOP']",
                   "first_piece_s=1.5", "total_s=4.0"]:
        assert needle in log, needle


def test_secrets_never_reach_the_run_log():
    seen = []
    runlog.set_sink(seen.append)
    try:
        runlog.detail("key AIza" + "x" * 35 + " token bot123456789:" + "y" * 30)
    finally:
        runlog.clear_sink(seen.append)
    assert "AIza" not in seen[0] and "bot123456789" not in seen[0]


def test_a_run_writes_every_stage_input_and_output_to_its_own_log(env):
    llm.set_test_backend(audio=heard, json=extract_ok)
    rl = pipeline._RunLogger("http://u", lambda line: None)
    res = pipeline.run_stages("vid", "عنوان", "http://u", rl)      # closes its own run log
    assert res.ok
    log = rl.path.read_text(encoding="utf-8")
    for needle in [
        "RUN ENVIRONMENT", "python=", "env GEMINI_MODELS",
        "file src/lifecompass/pipeline.py sha256[:16]=", "file prompts/extract_prompt.txt sha256[:16]=",
        "file prompts/extract_request.txt sha256[:16]=", "file prompts/listen_request.txt sha256[:16]=",
        "RUN STAGES start: video_id=vid", "STAGE listen start", "LISTEN start", "(5 bytes sha256[:16]=",
        "LISTEN attempt 1", "paragraphs as kept", "LISTEN done", "FULL TRANSCRIPT DOCUMENT",
        "STAGE listen finished", "STAGE extract start", "EXTRACT input", "EXTRACT reply parsed: 3 item(s)",
        "EXTRACT item 1: domain='Faith'; 5 words; found as one continuous passage in paragraph 3",
        "EXTRACT item 2: domain='NotADomain' -> Other (not in the list)", "found as one continuous passage in paragraph 0",
        "EXTRACT item 3: empty text -> skipped", "EXTRACT item 1 text", "STAGE extract finished",
        "STORE phase start: 2 item(s)", "STORE: item 1 stored as principle id=", "STORE: item 2 stored as principle id=",
        "STORE: video marked processed: added=2", "TELEGRAM message 1/1", "فقرة 2 تانية فيها مبدأ.",
    ]:
        assert needle in log, needle
    assert "=== run finished: " in log


def test_the_sink_is_released_when_the_run_closes(env):
    rl = pipeline._RunLogger("http://u", lambda line: None)
    rl.close("test")
    runlog.detail("after the run")      # goes to the master debug log, not to the closed file
    assert "after the run" not in rl.path.read_text(encoding="utf-8")


def test_a_principle_may_run_across_a_paragraph_break_and_the_log_says_so():
    from lifecompass.document import SourceDocument, Unit
    from lifecompass.principle_extractor import _locate, extract_principles
    doc = SourceDocument(video_id="v", units=[Unit(0, "أول جملة. وتكملتها"), Unit(1, "في الفقرة اللي بعدها. وبعدين"),
                                              Unit(2, "فقرة تالتة.")], stats={})
    crossing = "وتكملتها\n\nفي الفقرة اللي بعدها."
    assert _locate(doc, crossing) == (0, 10, 1, 21)
    assert _locate(doc, "وتكملتها في   الفقرة اللي بعدها.") == (0, 10, 1, 21)    # spaces and breaks are layout
    assert _locate(doc, "فقرة تالتة.") == (2, 0, 2, 11)
    assert _locate(doc, "كلام مش موجود") is None
    llm.set_test_backend(json=lambda p, s, sys, c: ExtractionResult(items=[ExtractedPrinciple(text=crossing, domain="Self")]))
    seen = []
    runlog.set_sink(seen.append)
    try:
        items, model = extract_principles(doc, "t", log=lambda s: None)
    finally:
        runlog.clear_sink(seen.append)
    assert items == [(crossing, "Self")]                       # kept exactly as the model wrote it
    assert "from paragraph 0 char 10 to paragraph 1 char 21 (crosses 1 paragraph break(s))" in "\n".join(seen)