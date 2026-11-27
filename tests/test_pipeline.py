"""The whole path through the pipeline orchestration: stages, storage, resume
after a failure, and what gets sent -- with the models, ffmpeg and Telegram
replaced by fakes and a scratch database."""
import pytest

from lifecompass import audio as audio_mod
from lifecompass import llm, pipeline, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult


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


def heard(audio, prompt, system, context):
    # one paragraph per line, said by one recording
    return "\n".join(f"جملة {idx}-{j} كلمة تانية." for idx in (1, 2, 3) for j in range(5))


PRINCIPLE = "جملة 2-1 كلمة تانية. جملة 2-2 كلمة تانية."


def extract_one(prompt, schema, system, context):
    assert schema is ExtractionResult
    return ExtractionResult(items=[ExtractedPrinciple(text=PRINCIPLE, domain="Self")])


def rlog():
    return pipeline._RunLogger("http://u", lambda line: None)


def test_a_video_goes_from_audio_to_stored_principles(env):
    llm.set_test_backend(audio=heard, json=extract_one)
    res = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert res.ok and res.added == 1
    p = store.get_principles_for_video("vid")[0]
    assert p.principle == PRINCIPLE
    assert p.doc_hash
    assert store.get_document_json("vid")
    assert "جملة 2-1" in store.get_transcript("vid")
    assert any(p.principle in text and "youtu.be" not in text for text in env)
    assert store.get_pending_retries() == []
    assert store.is_video_processed("vid")


def test_a_failure_while_listening_queues_a_retry_that_hears_the_whole_recording_again(env):
    calls = []

    def fails_once(audio, prompt, system, context):
        calls.append(context)
        if len(calls) == 1:
            raise llm.StageSkippedError("no model")
        return heard(audio, prompt, system, context)

    llm.set_test_backend(audio=fails_once, json=extract_one)
    res = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert not res.ok
    pend = store.get_pending_retries()
    assert [(p.video_id, p.stage) for p in pend] == [("vid", "listen")]
    assert store.get_document_json("vid") is None                 # nothing half-heard is kept

    again = pipeline.run_stages("vid", "عنوان", "http://u", rlog(), stage=pend[0].stage)
    assert again.ok and again.added == 1
    assert calls == ["listen", "listen"]                          # the whole recording, heard again from the start
    assert store.get_pending_retries() == []


def test_a_video_with_no_principles_is_recorded_not_retried(env):
    llm.set_test_backend(audio=heard, json=lambda prompt, schema, system, context: ExtractionResult(items=[]))
    res = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert res.ok and res.added == 0
    assert store.is_video_processed("vid") and store.get_pending_retries() == []
    assert any("no principle was extracted" in text for text in env)    # the user is told, in words that are true now


def _clear_principles_keep_document():
    import sqlite3
    c = sqlite3.connect(store.DB_PATH)
    for t in ("principle_images", "principles", "processed_videos"):
        c.execute(f"delete from {t}")
    c.commit(); c.close()


def test_a_saved_document_is_reused_when_the_video_is_not_stored_yet(env):
    listened = []

    def counting(audio, prompt, system, context):
        listened.append(context)
        return heard(audio, prompt, system, context)

    llm.set_test_backend(audio=counting, json=extract_one)
    assert pipeline.run_stages("vid", "عنوان", "http://u", rlog()).ok
    assert len(listened) == 1
    _clear_principles_keep_document()
    assert not store.is_video_processed("vid") and store.get_document_json("vid")

    again = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert again.ok and again.added == 1
    assert len(listened) == 1                                          # the same text, no new transcription
    assert store.is_video_processed("vid")


def test_a_saved_document_of_another_version_is_not_reused(env):
    import json, sqlite3
    listened = []

    def counting(audio, prompt, system, context):
        listened.append(context)
        return heard(audio, prompt, system, context)

    llm.set_test_backend(audio=counting, json=extract_one)
    assert pipeline.run_stages("vid", "عنوان", "http://u", rlog()).ok
    _clear_principles_keep_document()
    c = sqlite3.connect(store.DB_PATH)
    d = json.loads(c.execute("select doc_json from documents").fetchone()[0]); d["version"] = 1
    c.execute("update documents set doc_json=?", (json.dumps(d, ensure_ascii=False),)); c.commit(); c.close()
    assert pipeline.run_stages("vid", "عنوان", "http://u", rlog()).ok
    assert len(listened) == 2                                          # listened again: the old document was not trusted
