"""Prompt review 2026-11-21: the request texts are prompt files that the stage fingerprints cover,
the prompts say what they now say, and the literal-text guard works on whole words. No model is called."""
from lifecompass import principle_extractor as pe
from lifecompass import stage_fingerprints as sf
from lifecompass.document import SourceDocument, Unit

PROMPTS = sf.PROJECT_ROOT / "prompts"


def test_every_prompt_file_is_in_a_stage_fingerprint_except_the_caption_prompt_that_post_stage_fingerprints_cover():
    used = {name for names in sf.STAGE_PROMPTS.values() for name in names}
    assert {p.name for p in PROMPTS.glob("*.txt")} == used | {"standalone_image_caption_prompt.txt"}


def test_the_extract_request_is_the_title_then_the_marker_then_the_paragraphs():
    doc = SourceDocument("v", [Unit(0, "فقرة أولى."), Unit(1, "فقرة تانية.")])
    assert pe.render(doc, "{عنوان}") == "Video title: {عنوان}\n\n=== TRANSCRIPT ===\nفقرة أولى.\n\nفقرة تانية."


def test_the_extract_prompt_says_what_the_review_decided():
    text = (PROMPTS / "extract_prompt.txt").read_text(encoding="utf-8")
    assert '"=== TRANSCRIPT ==="' in text and "never copy from them" in text
    assert "buried inside a story" not in text and "never says in words" in text
    assert "should not be in the principle, end" not in text and "including an aside that interrupts the thought" in text
    assert "A principle here is any passage" in text
    assert "keep both in one passage" in text and "if a screenshot of it was taken" in text


def test_the_caption_prompt_no_longer_claims_that_the_image_is_not_a_principle():
    text = (PROMPTS / "standalone_image_caption_prompt.txt").read_text(encoding="utf-8")
    assert "did NOT match" not in text and "was not linked to any extracted principle" in text


DOC = SourceDocument("v", [Unit(0, "فلو أنت عملت كده هتتعب. الحياة جميلة جدا يا جماعة. هو بيعمل حاجة كبيرة قوي.")])
VIEW = pe._clean_view(DOC)


def test_a_passage_may_leave_out_the_attached_connective_letter_only():
    assert pe._conform(VIEW, "لو أنت عملت كده هتتعب.") == ("لو أنت عملت كده هتتعب.", "as is")
    assert pe._conform(VIEW, "يعمل حاجة كبيرة قوي.") == ("بيعمل حاجة كبيرة قوي.", "repaired")


def test_a_passage_that_starts_or_stops_inside_a_word_is_widened_to_whole_words():
    assert pe._conform(VIEW, "ياة جميلة جدا يا جماعة.") == ("الحياة جميلة جدا يا جماعة.", "repaired")
    assert pe._conform(VIEW, "الحياة جميلة جدا يا جماع") == ("الحياة جميلة جدا يا جماعة.", "repaired")


def test_a_passage_without_the_punctuation_that_clings_to_its_last_word_is_still_as_it_is():
    assert pe._conform(VIEW, "الحياة جميلة جدا يا جماعة") == ("الحياة جميلة جدا يا جماعة", "as is")
