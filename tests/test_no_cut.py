"""A recording is never cut (design decision 2026-11-09). One too big for an inline request goes to the model through the Files API instead.
No real model or service is called: the client is a fake."""
from types import SimpleNamespace

import pytest
from google.genai import types

from lifecompass import llm

URI = "https://files.example/abc"


class FakeFiles:
    def __init__(self, states):
        self.states = list(states)
        self.uploads = []
        self.gets = 0

    def _file(self, state):
        return SimpleNamespace(name="files/abc", uri=URI, mime_type="audio/mp3", state=state)

    def upload(self, *, file, config):
        self.uploads.append((file.read(), config.mime_type))
        return self._file(self.states.pop(0))

    def get(self, *, name):
        self.gets += 1
        return self._file(self.states.pop(0))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def test_the_upload_waits_until_the_service_has_processed_the_recording():
    files = FakeFiles([types.FileState.PROCESSING, types.FileState.PROCESSING, types.FileState.ACTIVE])
    part = llm._upload_audio(SimpleNamespace(files=files), b"audio-bytes", "audio/mp3")
    assert files.uploads == [(b"audio-bytes", "audio/mp3")] and files.gets == 2
    assert part.file_data.file_uri == URI


def test_a_failed_upload_is_an_error_and_nothing_is_cut_instead():
    files = FakeFiles([types.FileState.PROCESSING, types.FileState.FAILED])
    with pytest.raises(RuntimeError, match="not usable"):
        llm._upload_audio(SimpleNamespace(files=files), b"x", "audio/mp3")


def test_a_recording_that_is_usable_at_once_is_not_polled():
    files = FakeFiles([None])
    llm._upload_audio(SimpleNamespace(files=files), b"x", "audio/mp3")
    assert files.gets == 0


def run(monkeypatch, audio, limit=None):
    sent, fake = [], SimpleNamespace(files=FakeFiles([types.FileState.ACTIVE]))
    if limit is not None:
        monkeypatch.setattr(llm, "INLINE_AUDIO_MAX_BYTES", limit)
    monkeypatch.setattr(llm, "_TEST_AUDIO", None)
    monkeypatch.setattr(llm, "_get_models", lambda context=None: ["m1", "m2"])
    monkeypatch.setattr(llm, "_detail_request", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_detail_response", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_non_empty_text", lambda resp: "text")

    def stream(client, model_name, contents, config, log):
        sent.append((model_name, contents))
        if model_name == "m1":
            raise RuntimeError("the first model fails")
        return object()

    def chain(make_build_call, models, *, log, context, timeout):
        build = make_build_call(fake)
        last = None
        for model in models:
            try:
                return build(model), model
            except RuntimeError as e:
                last = e
        raise last

    monkeypatch.setattr(llm, "_stream_logged", stream)
    monkeypatch.setattr(llm, "_chain_with_key_failover", chain)
    assert llm.generate_audio_text(audio, "prompt", context="listen[1/1]", log=lambda *a: None) == ("text", "m2")
    return sent, fake


def test_a_recording_within_the_inline_limit_is_sent_inline_as_before(monkeypatch):
    sent, fake = run(monkeypatch, b"x" * 100, limit=100)             # exactly at the limit is still inline
    assert fake.files.uploads == []
    assert all(contents[0].inline_data.data == b"x" * 100 and contents[1] == "prompt" for _model, contents in sent)


def test_a_bigger_recording_is_uploaded_once_for_all_the_models_and_never_cut(monkeypatch):
    sent, fake = run(monkeypatch, b"x" * 101, limit=100)
    assert fake.files.uploads == [(b"x" * 101, "audio/mp3")]          # one upload although two models were tried
    assert [model for model, _contents in sent] == ["m1", "m2"]
    assert all(contents[0].file_data.file_uri == URI and contents[1] == "prompt" for _model, contents in sent)
