"""The model chain: each model once per pass, then the next; nothing outside GEMINI_MODELS."""
import types

import pytest

from lifecompass import llm


@pytest.fixture(autouse=True)
def direct_calls(monkeypatch):
    monkeypatch.setattr(llm, "_call_with_timeout", lambda build_call, model, timeout: build_call(model))
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda model: None)


def walk(models, behaviour):
    calls = []

    def build(model):
        calls.append(model)
        result = behaviour[model]
        if isinstance(result, Exception):
            raise result
        return result

    return calls, build


def test_the_first_model_that_answers_ends_the_pass():
    calls, build = walk(["a", "b"], {"a": "ok-a", "b": "ok-b"})
    assert llm._generate_with_fallback(["a", "b"], build, log=lambda *_: None, context="t") == ("ok-a", "a")
    assert calls == ["a"]


def test_an_error_moves_to_the_next_model_without_asking_the_same_one_again():
    calls, build = walk(["a", "b"], {"a": RuntimeError("503 overloaded"), "b": "ok-b"})
    assert llm._generate_with_fallback(["a", "b"], build, log=lambda *_: None, context="t") == ("ok-b", "b")
    assert calls == ["a", "b"]


def test_no_answer_in_time_also_moves_on_with_no_second_attempt():
    calls, build = walk(["a", "b"], {"a": TimeoutError("no answer"), "b": "ok-b"})
    assert llm._generate_with_fallback(["a", "b"], build, log=lambda *_: None, context="t")[1] == "b"
    assert calls == ["a", "b"]


def test_when_the_whole_chain_fails_every_model_was_asked_exactly_once():
    calls, build = walk(["a", "b"], {"a": TimeoutError("slow"), "b": RuntimeError("503 overloaded")})
    with pytest.raises(RuntimeError):
        llm._generate_with_fallback(["a", "b"], build, log=lambda *_: None, context="t")
    assert calls == ["a", "b"]


def test_a_stage_whose_chain_failed_is_queued_and_there_is_no_other_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    calls = []

    def generate_content(**kw):
        calls.append(kw["model"])
        raise RuntimeError("503 overloaded")

    monkeypatch.setattr(llm, "_client", lambda: types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content, generate_content_stream=lambda **kw: iter([generate_content(**kw)]))))
    with pytest.raises(llm.StageSkippedError):
        llm.generate_audio_text(b"x", "hi", context="listen[1/1]", log=lambda *_: None)
    assert calls == ["a", "b"]
    assert not hasattr(llm, "_call_openrouter") and not hasattr(llm, "requests")

def _fake_client(answers):
    calls = []

    def generate_content(**kw):
        calls.append(kw["model"])
        return types.SimpleNamespace(text=answers[kw["model"]])

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content, generate_content_stream=lambda **kw: iter([generate_content(**kw)])))
    return client, calls


def test_an_empty_answer_counts_as_a_failure_and_the_next_model_is_asked(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    client, calls = _fake_client({"a": "", "b": "كلام"})
    monkeypatch.setattr(llm, "_client", lambda: client)
    assert llm.generate_audio_text(b"x", "p", context="listen[1/1]", log=lambda *_: None) == ("كلام", "b")
    assert calls == ["a", "b"]
    client, calls = _fake_client({"a": "   ", "b": "تمام"})
    monkeypatch.setattr(llm, "_client", lambda: client)
    assert llm.generate_audio_text(b"x", "p", context="listen[1/1]", log=lambda *_: None) == ("تمام", "b")
    assert calls == ["a", "b"]


def test_empty_answers_from_every_model_queue_the_stage_after_one_ask_each(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    client, calls = _fake_client({"a": "", "b": ""})
    monkeypatch.setattr(llm, "_client", lambda: client)
    with pytest.raises(llm.StageSkippedError):
        llm.generate_audio_text(b"x", "p", context="listen[1/1]", log=lambda *_: None)
    assert calls == ["a", "b"]


def test_an_answer_that_cannot_be_parsed_moves_on_to_the_next_model(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: int = 1

    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    client, calls = _fake_client({"a": "garbage", "b": "{}"})
    monkeypatch.setattr(llm, "_client", lambda: client)
    monkeypatch.setattr(llm, "_parse_structured", lambda response, schema: None if response.text == "garbage" else Out())
    parsed, model = llm.generate_json("p", Out, context="extract[1]", log=lambda *_: None)
    assert model == "b" and calls == ["a", "b"]


def test_an_empty_answer_says_why_in_the_error():
    response = types.SimpleNamespace(
        text=None,
        candidates=[types.SimpleNamespace(finish_reason=types.SimpleNamespace(name="MAX_TOKENS"))],
        prompt_feedback=None,
        usage_metadata=types.SimpleNamespace(prompt_token_count=100, candidates_token_count=0, thoughts_token_count=9000),
    )
    with pytest.raises(RuntimeError) as err:
        llm._non_empty_text(response)
    assert "MAX_TOKENS" in str(err.value) and "thoughts_token_count=9000" in str(err.value)
