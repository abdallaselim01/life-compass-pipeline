"""No finished principle is lost because of the SHAPE of a model's reply (design decision 2026-11-07). Every shape below used to
throw the reply, or part of it, away. No model is called."""
import json
import types as pytypes

import pytest

from lifecompass import llm
from lifecompass.enrichment import CaptionResult
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

A = {"text": "الوعي أمر ضروري.", "domain": "Learning"}
B = {"text": "بلاش استسهال.", "domain": "Self"}
C = {"text": "الحياة قاسية.", "domain": "Faith"}


def reply(text, finish=None):
    return pytypes.SimpleNamespace(text=text, parsed=None, finish_reason=finish, candidates=None)


def read(text):
    got = llm._parse_structured(reply(text), ExtractionResult)
    return None if got is None else [(i.text, i.domain) for i in got.items]


def pairs(*items):
    return [(i["text"], i["domain"]) for i in items]


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


# ---- the shape of the whole reply -------------------------------------------------------------------------------
def test_a_clean_reply_is_read_as_before():
    assert read(dumps({"items": [A, B]})) == pairs(A, B)


def test_a_reply_cut_off_in_the_middle_of_the_list_keeps_every_finished_item():
    text = '{"items": [' + dumps(A) + ", " + dumps(B) + ', {"text": "نص ناقص ما خلص'
    assert read(text) == pairs(A, B)


def test_prose_before_and_after_the_json_does_not_hide_the_items():
    assert read("تمام، دي المبادئ:\n" + dumps({"items": [A, B]}) + "\nأي استفسار؟") == pairs(A, B)


def test_a_schema_echo_followed_by_a_cut_off_answer_keeps_the_finished_items():
    text = json.dumps(ExtractionResult.model_json_schema()) + ', "items": [' + dumps(A) + ", " + dumps(B) + ', {"te'
    assert read(text) == pairs(A, B)


def test_a_wrong_wrapper_key_is_read():
    assert read(dumps({"principles": [A, B]})) == pairs(A, B)
    assert read(dumps({"principles": [A, B], "note": "تمام"})) == pairs(A, B)
    assert read(dumps([A, B])) == pairs(A, B)                                     # a bare list


def test_items_scattered_in_text_without_any_wrapper_are_all_found():
    assert read(dumps(A) + "\n\n" + dumps(B) + "\n" + dumps(C)) == pairs(A, B, C)


def test_an_object_repeated_twice_is_kept_once_only_when_it_has_to_be_salvaged():
    assert read(dumps(A) + dumps(A) + dumps(B)) == pairs(A, B)


# ---- the shape of one item --------------------------------------------------------------------------------------
def test_an_item_with_no_domain_a_bare_string_or_another_key_for_the_text_is_kept():
    text = dumps({"items": [{"text": "بدون مجال."}, "نص عادي كسطر.", {"principle": "تحت اسم تاني.", "domain": "Self"},
                            {"quote": "اقتباس.", "domain": None}, A]})
    assert read(text) == [("بدون مجال.", "Other"), ("نص عادي كسطر.", "Other"), ("تحت اسم تاني.", "Self"),
                          ("اقتباس.", "Other"), (A["text"], A["domain"])]


def test_the_schema_shown_to_the_model_still_says_the_domain_is_required():
    assert ExtractionResult.model_json_schema()["$defs"]["ExtractedPrinciple"]["required"] == ["text", "domain"]
    assert ExtractedPrinciple(text="x", domain="Self").domain == "Self"


def test_an_item_that_does_not_fit_is_written_to_the_log_not_dropped_in_silence(monkeypatch):
    logged = []
    monkeypatch.setattr(llm, "_detail", lambda line: logged.append(line))
    got = llm._lenient_parse_items({"items": [A, {"nothing": "here"}, 7]}, ExtractionResult)
    assert [i.text for i in got.items] == [A["text"]]
    assert sum("item dropped" in line for line in logged) == 2 and any("nothing" in line for line in logged)


def test_nothing_is_invented_from_a_reply_with_no_item_in_it():
    assert read("no json here") is None
    assert read(dumps({"items": []})) is not None and read(dumps({"items": []})) == []
    assert read(json.dumps(ExtractionResult.model_json_schema())) is None              # the echoed schema alone holds no item
    assert read(json.dumps(ExtractionResult.model_json_schema()) + ', "items": []}') == []    # ... and an empty answer stays empty
    assert llm._parse_structured(reply('{"caption": 5, "x": ['), CaptionResult) is None      # a schema with no items list


# ---- a reply the model cut short (finish_reason other than STOP) ------------------------------------------------
@pytest.fixture
def chain(monkeypatch):
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(llm, "_client", lambda backup=False: object())

    def install(by_model):
        def fake(client, model_name, contents, config, log):
            answer = by_model[model_name]
            if isinstance(answer, Exception):
                raise answer
            return answer
        monkeypatch.setattr(llm, "_stream_logged", fake)
    return install


def run_json():
    return llm.generate_json("prompt", ExtractionResult, context="extract[1]", log=lambda *a, **k: None)


def test_when_every_model_fails_the_finished_items_of_the_fullest_cut_reply_are_used(chain):
    cut_a = reply('{"items": [' + dumps(A) + ", " + dumps(B) + ', {"te', finish="MAX_TOKENS")
    cut_b = reply('{"items": [' + dumps(A) + ', {"te', finish="MAX_TOKENS")
    chain({"a": cut_a, "b": cut_b})
    result, model = run_json()
    assert model == "a" and [(i.text, i.domain) for i in result.items] == pairs(A, B)


def test_a_finished_answer_from_the_next_model_wins_over_a_cut_one(chain):
    cut = reply('{"items": [' + dumps(A) + ', {"te', finish="MAX_TOKENS")
    done = reply(dumps({"items": [A, B, C]}), finish="STOP")
    chain({"a": cut, "b": done})
    result, model = run_json()
    assert model == "b" and len(result.items) == 3


def test_a_cut_reply_with_no_finished_item_still_fails_the_stage_so_it_is_queued_for_retry(chain):
    chain({"a": reply('{"items": [{"te', finish="MAX_TOKENS"), "b": RuntimeError("503 UNAVAILABLE")})
    with pytest.raises(llm.StageSkippedError):
        run_json()
