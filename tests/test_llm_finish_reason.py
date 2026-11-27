"""A model reply that was cut short (blocked content, token limit, ...) is never accepted as a finished answer."""
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from lifecompass import llm
from google.genai import types

STOP = types.FinishReason.STOP
CUT = types.FinishReason.PROHIBITED_CONTENT


def chunk(text, finish=None):
    cands = [SimpleNamespace(finish_reason=finish)] if finish is not None else None
    return SimpleNamespace(text=text, candidates=cands, usage_metadata=None, prompt_feedback=None)


class FakeClient:
    """models.generate_content_stream(model=...) yields the pieces prepared for that model."""

    def __init__(self, replies):
        self.calls = []
        self.models = SimpleNamespace(generate_content_stream=self._stream)
        self._replies = replies

    def _stream(self, model, contents, config):
        self.calls.append(model)
        return iter(self._replies[model])


@pytest.fixture
def chain(monkeypatch):
    llm.set_test_backend()
    monkeypatch.setenv("GEMINI_MODELS_LISTEN", "model-a,model-b")
    monkeypatch.setenv("GEMINI_MODELS", "model-a,model-b")
    monkeypatch.delenv("GEMINI_API_KEY_BACKUP", raising=False)

    def install(replies):
        client = FakeClient(replies)
        monkeypatch.setattr(llm, "_client", lambda backup=False: client)
        return client

    llm._exhausted_models_this_run.clear()
    return install


def reply(text, finish):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)], finish_reason=None)


def test_a_reply_that_stopped_normally_is_accepted():
    assert llm._non_empty_text(reply("نص كامل", STOP)) == "نص كامل"


def test_a_reply_cut_by_blocked_content_is_refused_even_though_it_has_text():
    with pytest.raises(RuntimeError, match="PROHIBITED_CONTENT.*7 characters"):
        llm._non_empty_text(reply("نص ناقص", CUT))


@pytest.mark.parametrize("name", ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER", "BLOCKLIST"])
def test_every_finish_reason_other_than_stop_is_refused(name):
    with pytest.raises(RuntimeError, match=name):
        llm._non_empty_text(reply("جزء من الرد", SimpleNamespace(name=name)))


def test_a_reply_that_says_nothing_about_how_it_ended_is_accepted_as_before():
    assert llm._non_empty_text(SimpleNamespace(text="رد", candidates=None)) == "رد"
    assert llm._non_empty_text(SimpleNamespace(text="رد", candidates=[SimpleNamespace(finish_reason=None)])) == "رد"
    unspecified = types.FinishReason.FINISH_REASON_UNSPECIFIED
    assert llm._non_empty_text(reply("رد", unspecified)) == "رد"


def test_the_finish_reason_is_found_even_when_the_last_piece_does_not_carry_it():
    client = FakeClient({"m": [chunk("a"), chunk("b", CUT), chunk("")]})     # the final piece has no candidates
    resp = llm._stream_response(client, "m", "x", None)
    assert resp.text == "ab" and llm._finish_reason_name(resp) == "PROHIBITED_CONTENT"
    with pytest.raises(RuntimeError, match="cut short"):
        llm._non_empty_text(resp)


def test_audio_cut_on_the_first_model_moves_to_the_next_one_and_the_cut_text_is_not_returned(chain):
    client = chain({"model-a": [chunk("نص ناقص", CUT)], "model-b": [chunk("نص "), chunk("كامل", STOP)]})
    text, model = llm.generate_audio_text(b"audio", "p", context="listen[1/1]", log=lambda *a: None)
    assert (text, model) == ("نص كامل", "model-b") and client.calls == ["model-a", "model-b"]


def test_when_every_model_is_cut_the_stage_is_skipped_so_nothing_partial_is_stored(chain):
    client = chain({"model-a": [chunk("جزء", CUT)], "model-b": [chunk("جزء", types.FinishReason.MAX_TOKENS)]})
    with pytest.raises(llm.StageSkippedError, match="cut short"):
        llm.generate_audio_text(b"audio", "p", context="listen[1/1]", log=lambda *a: None)
    assert client.calls == ["model-a", "model-b"]


class Box(BaseModel):
    items: list[str] = []


def test_a_json_reply_cut_short_is_refused_even_when_its_start_parses(chain):
    client = chain({"model-a": [chunk('{"items": ["a"]}', CUT)], "model-b": [chunk('{"items": ["a", "b"]}', STOP)]})
    parsed, model = llm.generate_json("p", Box, context="extract[1]", log=lambda *a: None)
    assert model == "model-b" and parsed.items == ["a", "b"] and client.calls == ["model-a", "model-b"]
