import time
import types

import pytest

from lifecompass import llm


def _chunk(text="hello"):
    return types.SimpleNamespace(text=text, candidates=[], usage_metadata=None, prompt_feedback=None)


def _fresh(monkeypatch):
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "a,b")


def test_stalled_stream_moves_to_next_model(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(llm, "_STREAM_FIRST_CHUNK_SECONDS", 1.0)
    monkeypatch.setattr(llm, "_STREAM_STALL_SECONDS", 1.0)

    def stream(model, **kw):
        if model == "a":
            def gen():
                yield _chunk("part")
                time.sleep(30)      # sends one piece, then goes silent
                yield _chunk("never")
            return gen()
        return iter([_chunk("good "), _chunk("answer")])

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(llm, "_client", lambda backup=False: client)
    started = time.time()
    text, model = llm.generate_audio_text(b"audio", "x", context="listen[1/1]")
    assert (text, model) == ("good answer", "b")
    assert time.time() - started < 15


def test_slow_but_streaming_is_not_cut(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setattr(llm, "_STREAM_STALL_SECONDS", 2.5)

    def stream(model, **kw):
        def gen():
            for i in range(4):
                time.sleep(1.2)     # slow, but a piece always arrives before the stall limit
                yield _chunk(str(i))
        return gen()

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(llm, "_client", lambda backup=False: client)
    text, model = llm.generate_audio_text(b"audio", "x", context="listen[1/1]")
    assert (text, model) == ("0123", "a")


def test_backup_key_used_once_after_quota_failure(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP", "dummy")
    used = []

    def primary(**kw):
        used.append("primary")
        raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    def backup(**kw):
        used.append("backup")
        return iter([_chunk("from backup")])

    clients = {False: types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=primary)),
               True: types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=backup))}
    monkeypatch.setattr(llm, "_client", lambda backup=False: clients[backup])
    text, model = llm.generate_audio_text(b"audio", "x", context="listen[1/1]")
    assert text == "from backup"
    assert used == ["primary", "primary", "backup"]    # both models on the main key, then backup once


def test_no_backup_for_non_quota_failure(monkeypatch):
    _fresh(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP", "dummy")
    used = []

    def primary(**kw):
        used.append("primary")
        raise RuntimeError("500 internal")

    clients = {False: types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=primary))}
    monkeypatch.setattr(llm, "_client", lambda backup=False: clients[backup])
    with pytest.raises(llm.StageSkippedError):
        llm.generate_audio_text(b"audio", "x", context="listen[1/1]")
    assert used == ["primary", "primary"]