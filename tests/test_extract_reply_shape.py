"""The extract request states its reply shape inside the prompt file (so the stage fingerprint covers it) and carries no JSON-Schema text
. generate_json appends nothing to any prompt (the unused hint was removed 2026-11-24)."""
import json
import types as pytypes

import pytest

from lifecompass import llm, principle_extractor
from lifecompass.document import SourceDocument, Unit
from lifecompass.principle_extractor import ExtractionResult

SHAPE = '{"items": [{"text": "<the principle>", "domain": "<one domain from the DOMAIN list>"}]}'
PASSAGE = "فقرة فيها كلام مهم جدا يا جماعة."
DOC = SourceDocument("abcdefghijk", [Unit(0, PASSAGE)])


def _chunk(text):
    return pytypes.SimpleNamespace(text=text, candidates=[], usage_metadata=None, prompt_feedback=None)


@pytest.fixture
def seen(monkeypatch):
    """A fake Gemini client: every request is recorded, the reply is whatever `reply["text"]` holds."""
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)
    monkeypatch.setattr(llm, "_SCHEMA_ENABLED", False)          # the text hint exists only while the API-enforced schema is off
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls, reply = [], {"text": '{"items": []}'}

    def stream(model, **kw):
        calls.append(kw)
        return iter([_chunk(reply["text"])])

    client = pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(llm, "_client", lambda backup=False: client)
    return pytypes.SimpleNamespace(calls=calls, reply=reply)


def test_the_prompt_file_states_the_reply_shape_and_no_longer_points_at_a_hint_in_the_message():
    text = (principle_extractor.PROJECT_ROOT / "prompts" / "extract_prompt.txt").read_text(encoding="utf-8")
    assert SHAPE in text
    assert "requested with this message" not in text


def test_the_extract_request_carries_no_schema_text_and_the_shape_comes_from_the_prompt_file(seen):
    seen.reply["text"] = '{"items": [{"text": "%s", "domain": "Faith"}]}' % PASSAGE
    out, _model = principle_extractor.extract_principles(DOC, "عنوان", log=lambda *a: None)
    assert out == [(PASSAGE, "Faith")]
    [kw] = seen.calls
    assert json.dumps(ExtractionResult.model_json_schema(), ensure_ascii=False) not in kw["contents"]   # no schema text in this request
    assert "$defs" not in kw["contents"] and "properties" not in kw["contents"]
    assert SHAPE in kw["config"].system_instruction                                  # the shape is in the system prompt instead


def test_generate_json_sends_the_prompt_as_it_is(seen):
    seen.reply["text"] = '{"items": [{"text": "x", "domain": "Faith"}]}'
    llm.generate_json("PROMPT", ExtractionResult, context="extract[1]", log=lambda *a: None)
    [kw] = seen.calls
    assert kw["contents"] == "PROMPT"
    assert not hasattr(llm, "_schema_text_hint")                                      # the dead hint is gone


def test_a_reply_that_echoes_the_schema_or_wraps_the_list_is_still_read(seen):
    """The tolerant reader is unchanged: whatever shape comes back, the finished principles are kept."""
    echo = json.dumps(ExtractionResult.model_json_schema(), ensure_ascii=False)
    seen.reply["text"] = echo + '\n{"items": [{"text": "%s", "domain": "Faith"}]}' % PASSAGE
    out, _model = principle_extractor.extract_principles(DOC, "عنوان", log=lambda *a: None)
    assert out == [(PASSAGE, "Faith")]
