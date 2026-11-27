"""(1) A reply that begins with the JSON schema the model was given and then adds the answer is read, not thrown away
(seen on gemini-3.8-flash at extract[1] on 2026-11-07). (2) The big quality-critical calls never get a wait limit below 450 s.
No model is called."""
import json
import types as pytypes

import pytest

from lifecompass import llm
from lifecompass.enrichment import CaptionResult, ImageMatchResult
from lifecompass.principle_extractor import ExtractionResult

ITEMS = [{"text": "الوعي أمر ضروري.", "domain": "Learning"}, {"text": "بلاش استسهال.", "domain": "Self"}]


def reply(text):
    return pytypes.SimpleNamespace(text=text, parsed=None)


def echo(schema):
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)


# ---- the schema echo --------------------------------------------------------------------------------------------
def test_an_answer_that_follows_an_echoed_schema_is_read():
    text = echo(ExtractionResult) + ', "items": ' + json.dumps(ITEMS, ensure_ascii=False) + "}"
    with pytest.raises(ValueError):
        json.loads(text)                                                    # not valid JSON as a whole: the old parse said no
    got = llm._parse_structured(reply(text), ExtractionResult)
    assert [(i.text, i.domain) for i in got.items] == [(i["text"], i["domain"]) for i in ITEMS]


def test_the_echo_works_for_the_other_answer_shapes_too():
    text = echo(CaptionResult) + ', "caption": "كابشن"}'
    assert llm._parse_structured(reply(text), CaptionResult).caption == "كابشن"
    text = echo(ImageMatchResult) + ', "matches": [{"principle_index": 0, "image_indices": [1]}]}'
    assert llm._parse_structured(reply(text), ImageMatchResult).matches[0].image_indices == [1]


def test_an_echo_inside_the_same_object_and_a_fenced_answer_still_work():
    inside = {"$defs": {}, "properties": {}, "items": ITEMS}
    assert len(llm._parse_structured(reply(json.dumps(inside, ensure_ascii=False)), ExtractionResult).items) == 2
    fenced = "```json\n" + json.dumps({"items": ITEMS}, ensure_ascii=False) + "\n```"
    assert len(llm._parse_structured(reply(fenced), ExtractionResult).items) == 2


def test_a_reply_that_is_not_a_schema_echo_is_rejected_as_before():
    assert llm._recover_after_schema_echo('{"a": 1}, "items": []}') is None            # first object is not a schema
    assert llm._recover_after_schema_echo("not json at all") is None
    assert llm._recover_after_schema_echo(echo(ExtractionResult)) is None               # an echo with no answer after it
    assert llm._recover_after_schema_echo(echo(ExtractionResult) + ', "items": [') is None   # a broken answer after it
    assert llm._parse_structured(reply('{"items": [oops'), ExtractionResult) is None


def test_the_real_shape_of_the_reply_that_was_lost():
    """The reply gemini-3.8-flash gave at extract[1]: the schema object, then ', "items": [...]}'."""
    schema = ('{"$defs": {"ExtractedPrinciple": {"properties": {"text": {"title": "Text", "type": "string"}, "domain": '
              '{"title": "Domain", "type": "string"}}, "required": ["text", "domain"], "title": "ExtractedPrinciple", '
              '"type": "object"}}, "properties": {"items": {"items": {"$ref": "#/$defs/ExtractedPrinciple"}, "title": "Items", '
              '"type": "array"}}, "required": ["items"], "title": "ExtractionResult", "type": "object"}')
    text = schema + ', "items": [{"text": "الوعي أمر ضروري.", "domain": "Learning"}]}'
    assert [i.text for i in llm._parse_structured(reply(text), ExtractionResult).items] == ["الوعي أمر ضروري."]


# ---- the wait limit of the big calls ----------------------------------------------------------------------------
@pytest.mark.parametrize("context", ["listen[1/3]", "extract[1]"])
def test_the_big_calls_never_wait_less_than_450_seconds(context):
    llm.record_success_time(context, 57.0)                                  # 2 x 57 = 114 would have been the limit
    assert llm.adaptive_timeout(context, 900.0) == 450.0


def test_the_big_calls_still_wait_longer_after_a_slower_success_and_never_past_the_ceiling():
    llm.record_success_time("extract[1]", 300.0)
    assert llm.adaptive_timeout("extract[1]", 900.0) == 600.0               # 2 x 300
    llm.record_success_time("extract[1]", 500.0)
    assert llm.adaptive_timeout("extract[1]", 900.0) == 900.0               # capped at the ceiling


def test_the_small_calls_keep_the_120_second_floor_and_a_kind_with_no_success_the_ceiling():
    llm.record_success_time("describe_image", 10.0)
    assert llm.adaptive_timeout("describe_image", 900.0) == 120.0
    assert llm.adaptive_timeout("listen[1/1]", 900.0) == 900.0
