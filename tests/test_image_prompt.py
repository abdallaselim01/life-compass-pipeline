"""Images are matched and captioned from the picture itself: there is no describe stage."""
import dataclasses
import json
import types as pytypes

import pytest
from google.genai import types

from lifecompass import enrichment, images, llm, pipeline, store

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
TRANSCRIPT = "كلمة أولى في الكلام. الجملة اللي بتشرح الصورة هي دي بالظبط يا جماعة. وبعدها كلام تاني خالص."


def quiet(*args, **kwargs):
    return None


def test_there_is_no_describe_stage_any_more():
    assert "describe" not in llm.STAGES
    for name in ("describe_image", "describe_text_fp", "describe_prompt_fp", "IMAGE_DESCRIBE_PROMPT"):
        assert not hasattr(enrichment, name)
    assert not hasattr(images, "describe_image_file") and not hasattr(pipeline, "refresh_stale_image_descriptions")
    assert not hasattr(pipeline, "stale_description_ids")
    for name in ("set_image_description", "get_undescribed_images_for_video", "get_stale_description_images",
                 "set_image_describe_fp", "get_image_describe_fps"):
        assert not hasattr(store, name)
    assert [f.name for f in dataclasses.fields(images.ProcessedImage)] == ["enhanced_path", "replaces_image_id"]


def test_the_prompts_say_the_pictures_are_attached_and_mention_no_description():
    match = enrichment.IMAGE_MATCH_PROMPT_TEMPLATE
    assert "{images}" not in match and "description" not in match.lower() and "attached" in match
    assert match.count("{principles}") == 1
    caption = enrichment.STANDALONE_IMAGE_CAPTION_PROMPT
    assert "{image_description}" not in caption and "attached" in caption and "{transcript}" in caption


def test_read_image_gives_the_bytes_and_the_one_mime_rule_of_every_call(tmp_path):
    (tmp_path / "a.png").write_bytes(b"1")
    (tmp_path / "b.JPG").write_bytes(b"2")
    (tmp_path / "c.webp").write_bytes(b"3")
    assert images.read_image(tmp_path / "a.png") == (b"1", "image/png")
    assert images.read_image(tmp_path / "b.JPG") == (b"2", "image/jpeg")
    assert images.read_image(tmp_path / "c.webp") == (b"3", "image/jpeg")
    with pytest.raises(OSError):
        images.read_image(tmp_path / "missing.png")


@pytest.fixture
def chain(monkeypatch):
    """A fake Gemini client for the enrichment calls: every request is recorded, the reply is whatever `reply["text"]` holds."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODELS", "a")
    monkeypatch.delenv("GEMINI_API_KEY_BACKUP", raising=False)
    sent, reply = [], {"text": '{"matches": []}'}

    def make(api_key=None):
        def generate_content(**kw):
            sent.append(kw)
            return pytypes.SimpleNamespace(text=reply["text"])
        return pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(generate_content=generate_content))

    monkeypatch.setattr(enrichment.genai, "Client", make)
    monkeypatch.setattr(enrichment, "_parse_structured", lambda response, schema: schema.model_validate_json(response.text))
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    return pytypes.SimpleNamespace(sent=sent, reply=reply)


def test_matching_sends_every_picture_labelled_with_its_number_and_maps_the_answer_back(chain):
    chain.reply["text"] = ('{"matches": [{"principle_index": 0, "image_indices": [1, 7]}, '
                           '{"principle_index": 5, "image_indices": [0]}]}')
    got = enrichment.match_principles_to_images(["مبدأ أول", "مبدأ تاني"], [(b"AAA", "image/png"), (b"BBB", "image/jpeg")],
                                                log=quiet)
    assert got == {0: [1]}                  # an image number that does not exist, and a principle that does not exist, are dropped
    [kw] = chain.sent
    prompt, label0, pic0, label1, pic1 = kw["contents"][:5]
    assert "0. مبدأ أول" in prompt and "1. مبدأ تاني" in prompt
    assert (label0, label1) == ("Image 0:", "Image 1:")
    assert (pic0.inline_data.data, pic0.inline_data.mime_type) == (b"AAA", "image/png")
    assert (pic1.inline_data.data, pic1.inline_data.mime_type) == (b"BBB", "image/jpeg")
    for pic in (pic0, pic1):
        assert pic.media_resolution.level == types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH


def test_nothing_is_sent_when_there_is_no_picture(chain):
    assert enrichment.match_principles_to_images(["مبدأ"], [], log=quiet) == {}
    assert enrichment.match_principles_to_images([], [(PNG, "image/png")], log=quiet) == {}
    assert enrichment.caption_standalone_image((b"", "image/png"), TRANSCRIPT, log=quiet) == ""
    assert chain.sent == []


def caption_of(chain, written):
    chain.reply["text"] = json.dumps({"caption": written}, ensure_ascii=False)
    return enrichment.caption_standalone_image((PNG, "image/png"), TRANSCRIPT, log=quiet)


def test_the_caption_request_holds_the_picture_first_and_then_the_prompt_with_the_transcript(chain):
    assert caption_of(chain, "") == ""
    [kw] = chain.sent
    picture, text = kw["contents"]
    assert picture.inline_data.data == PNG and TRANSCRIPT in text


def test_a_caption_that_is_a_passage_of_the_transcript_is_kept_as_it_is(chain):
    assert caption_of(chain, "الجملة اللي بتشرح الصورة هي دي بالظبط يا جماعة.") == "الجملة اللي بتشرح الصورة هي دي بالظبط يا جماعة."


def test_stutter_in_a_caption_is_taken_out(chain):
    assert caption_of(chain, "الجملة اللي بتشرح الصورة ه هي دي بالظبط") == "الجملة اللي بتشرح الصورة هي دي بالظبط"


def test_a_caption_with_a_word_left_out_is_repaired_to_the_transcripts_own_words(chain):
    assert caption_of(chain, "الجملة اللي بتشرح الصورة دي بالظبط يا جماعة") == "الجملة اللي بتشرح الصورة هي دي بالظبط يا جماعة."


def test_a_caption_that_is_not_from_the_transcript_is_not_stored(chain):
    assert caption_of(chain, "ده كلام مختلف تماما ومش له أي علاقة بالمكتوب هنا خالص") == ""


def test_the_match_and_caption_requests_carry_no_schema_text_because_their_prompts_state_the_reply_shape(chain):
    enrichment.match_principles_to_images(["مبدأ"], [(PNG, "image/png")], log=quiet)
    caption_of(chain, "")
    assert len(chain.sent) == 2
    for kw in chain.sent:
        flat = " ".join(part for part in kw["contents"] if isinstance(part, str))
        assert "$defs" not in flat and "properties" not in flat and "مهم جدًا" not in flat
    assert '{"matches": [{"principle_index": 0, "image_indices": [0, 1]}]}' in enrichment.IMAGE_MATCH_PROMPT_TEMPLATE.format(principles="x")
    assert '{{"caption": "..."}}' in enrichment.STANDALONE_IMAGE_CAPTION_PROMPT

