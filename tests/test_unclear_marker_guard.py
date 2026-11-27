"""The unclear-speech marker [غير واضح] is never stored in a principle or a caption: a text that contains it is
split at it; each side of a principle is kept, a caption keeps its longest side. The caption prompt asks for the speaker's own words in whatever
register he used."""
import json

import pytest

from lifecompass import enrichment
from lifecompass import principle_extractor
from lifecompass.document import SourceDocument, Unit
from lifecompass.principle_extractor import split_at_unclear

# the fake-client fixtures of the sibling test modules (they are fixtures there, imported by name)
from test_extract_reply_shape import seen  # noqa: F401
from test_image_prompt import chain  # noqa: F401

MARK = "[غير واضح]"
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
LEFT = "كلام اول مفيد قوي"
RIGHT = "كلام تاني مفيد جدا يا جماعة."
TRANSCRIPT = f"{LEFT} {MARK} {RIGHT}"


def quiet(*args, **kwargs):
    return None


def test_split_at_unclear_keeps_both_sides_in_order_and_never_the_marker():
    assert split_at_unclear("كلام عادي بدون علامة") == ["كلام عادي بدون علامة"]
    assert split_at_unclear(f"{LEFT} {MARK} {RIGHT}") == [LEFT, RIGHT]
    assert split_at_unclear(f"{MARK} {RIGHT}") == [RIGHT]
    assert split_at_unclear(f"{LEFT} {MARK}") == [LEFT]
    assert split_at_unclear(MARK) == []
    assert split_at_unclear(f"{LEFT} {MARK}، {RIGHT}") == [LEFT, RIGHT]                 # punctuation that clung to the marker is not part of a side
    assert split_at_unclear(f"{LEFT} {MARK} وسط {MARK} {RIGHT}") == [LEFT, "وسط", RIGHT]  # a side of any length is kept: no size shaping


def extract(seen, written):
    seen.reply["text"] = json.dumps({"items": [{"text": written, "domain": "Faith"}]}, ensure_ascii=False)
    doc = SourceDocument("abcdefghijk", [Unit(0, TRANSCRIPT)])
    out, _model = principle_extractor.extract_principles(doc, "عنوان", log=quiet)
    return out


def test_a_principle_that_contains_the_marker_is_split_and_both_sides_are_kept(seen):
    assert extract(seen, TRANSCRIPT) == [(LEFT, "Faith"), (RIGHT, "Faith")]


def test_a_principle_whose_repair_pulls_the_marker_back_in_is_split_too(seen):
    # the model joined the two sides and left the marker out; the literal guard repairs it to the transcript's own passage (marker included),
    # and the split then runs on the repaired text
    assert extract(seen, f"{LEFT} {RIGHT}") == [(LEFT, "Faith"), (RIGHT, "Faith")]


def test_a_principle_without_the_marker_is_untouched(seen):
    assert extract(seen, RIGHT) == [(RIGHT, "Faith")]


def caption(chain, written):
    chain.reply["text"] = json.dumps({"caption": written}, ensure_ascii=False)
    return enrichment.caption_standalone_image((PNG, "image/png"), TRANSCRIPT, log=quiet)


def test_a_caption_that_contains_the_marker_keeps_its_longest_side(chain):
    assert caption(chain, TRANSCRIPT) == RIGHT        # RIGHT has 6 words, LEFT 4


def test_a_caption_that_is_only_the_marker_is_no_caption(chain):
    assert caption(chain, MARK) == ""


def test_a_caption_without_the_marker_is_untouched(chain):
    assert caption(chain, LEFT) == LEFT


def test_the_caption_prompt_asks_for_the_speakers_words_in_whatever_register_and_never_the_marker():
    prompt = enrichment.STANDALONE_IMAGE_CAPTION_PROMPT
    assert "Under no circumstances" not in prompt and "Modern Standard Arabic" not in prompt     # the old ban on the standard register is gone
    assert "whatever register he used" in prompt
    assert MARK in prompt and "ONE continuous passage" in prompt
    assert "attached above" in prompt
    filled = prompt.format(transcript="نص التفريغ")                  # the braces of the JSON line are escaped: the prompt still formats
    assert "نص التفريغ" in filled and '{"caption": "..."}' in filled
