"""The production core: listen writes paragraphs, one prompt extracts the principles; nothing else cuts, edits or reviews."""
import re
from pathlib import Path

import pytest

from lifecompass import audio as audio_mod
from lifecompass import llm, pipeline, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult
from lifecompass.production import run_production

PARAS = ["فقرة أولى كلام كتير هنا.", "فقرة تانية فيها مبدأ مهم جدا.", "فقرة تالتة."]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    sent = []
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text))
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(audio_mod, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    yield sent
    store._prepared.clear()


def rlog():
    return pipeline._RunLogger("http://u", lambda line: None)


def heard(audio, prompt, system, context):
    # what the three pieces used to say, now said by one recording
    return "\n\n".join(f"فقرة {idx} فيها كلام.\n\nفقرة {idx} تانية فيها مبدأ." for idx in (1, 2, 3))


def extract_ok(prompt, schema, system, context):
    assert schema is ExtractionResult
    return ExtractionResult(items=[ExtractedPrinciple(text="فقرة 2 تانية فيها مبدأ.", domain="Faith")])


def test_extract_gets_the_whole_paragraph_transcript_in_one_call_and_its_text_is_kept_as_written(monkeypatch):
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    llm.set_test_backend(audio=lambda a, p, s, c: "\n\n".join(PARAS))
    seen = []

    def fake(prompt, schema, system, context):
        seen.append((prompt, schema, system, context))
        return ExtractionResult(items=[ExtractedPrinciple(text=" فقرة تانية فيها مبدأ مهم جدا. ", domain="Self"),
                                       ExtractedPrinciple(text="   ", domain="Self"),
                                       ExtractedPrinciple(text="فقرة تالتة.", domain="NotADomain")])

    llm.set_test_backend(audio=llm._TEST_AUDIO, json=fake)
    res = run_production("vid", "عنوان", audio_path=Path("x.mp3"))
    assert [(i.principle, i.domain) for i in res.items] == [("فقرة تانية فيها مبدأ مهم جدا.", "Self"),
                                                            ("فقرة تالتة.", "Other")]
    assert len(seen) == 1
    prompt, schema, system, context = seen[0]
    assert schema is ExtractionResult and context == "extract[1]"
    assert "\n\n".join(PARAS) in prompt
    assert "Verbatim" in system


def test_a_video_goes_from_audio_to_stored_principles(env):
    llm.set_test_backend(audio=heard, json=extract_ok)
    res = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert res.ok and res.added == 1
    p = store.get_principles_for_video("vid")[0]
    assert p.principle == "فقرة 2 تانية فيها مبدأ." and p.domain == "Faith"
    assert any(p.principle in text for text in env)
    assert store.get_pending_retries() == [] and store.is_video_processed("vid")


def test_a_failure_in_extract_resumes_without_listening_again(env):
    listened = []

    def counting(audio, prompt, system, context):
        listened.append(context)
        return heard(audio, prompt, system, context)

    def extract_fails(prompt, schema, system, context):
        raise llm.StageSkippedError("no model")

    llm.set_test_backend(audio=counting, json=extract_fails)
    assert not pipeline.run_stages("vid", "عنوان", "http://u", rlog()).ok
    pend = store.get_pending_retries()[0]
    assert pend.stage == "extract" and len(listened) == 1

    llm.set_test_backend(audio=counting, json=extract_ok)
    again = pipeline.run_stages("vid", "عنوان", "http://u", rlog(), stage=pend.stage)
    assert again.ok and again.added == 1
    assert len(listened) == 1
    assert store.get_pending_retries() == []