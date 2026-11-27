"""The verbatim guard (design decision 2026-11-08 and 2026-11-09): every principle that leaves extract_principles is one continuous passage
of the transcript with its stutter fragments taken out (a lone letter, or the bare article, left hanging before the full word).
What the model tidied or changed is REPAIRED to the transcript's own words; only a text with no real anchor in the transcript is left
out and written whole in the run log. No model is called."""
from lifecompass import llm, pipeline, store
from lifecompass import principle_extractor as pe
from lifecompass.document import SourceDocument, Unit
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult
from test_run_detail_log import env, heard  # noqa: F401  (fixtures and helpers of the sibling module)

P0 = "الكلام ده قبل كده كتير. الكتاب ده بيقول لازم تقرأي أولا."
P1 = "أنت ماشي تروح معاهم بدري وتشرب م من اللي أنت بتحبه وتقدمه على غيرك. وبعدين بيكمل كلام تاني هنا."
P2 = 'اذهب ل ل لمن هم فعلا خبراء. وقال: "الوقت لا ينتظر المتأخرين"، وكلمة متأخرين يعني تأخيرا.'
DOC = SourceDocument("v", [Unit(0, P0), Unit(1, P1), Unit(2, P2)])


def run(monkeypatch, texts, doc=DOC):
    blocks, details, lines = [], [], []
    monkeypatch.setattr(pe, "_block", lambda title, body: blocks.append((title, body)))
    monkeypatch.setattr(pe, "_detail", lambda msg: details.append(msg))
    items = [ExtractedPrinciple(text=t, domain="Self") for t in texts]
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: (ExtractionResult(items=items), "m"))
    out, _model = pe.extract_principles(doc, "t", log=lines.append)
    return [t for t, _d in out], blocks, details, lines


def test_a_continuous_passage_is_kept_exactly_as_it_is(monkeypatch):
    text = "أنت ماشي تروح معاهم"
    out, blocks, _details, _lines = run(monkeypatch, [text])
    assert out == [text] and blocks == [("EXTRACT item 1 text", text)]


def test_a_passage_across_a_paragraph_break_is_continuous_too(monkeypatch):
    text = "الكتاب ده بيقول لازم تقرأي أولا. أنت ماشي تروح معاهم"
    assert run(monkeypatch, [text])[0] == [text]


def test_a_start_at_a_later_letter_of_the_first_word_is_widened_to_the_whole_word(monkeypatch):
    text = "نت ماشي تروح معاهم"                      # only the attached connective letter (rule 5: ف / و) may be left out of a word
    assert run(monkeypatch, [text])[0] == ["أنت ماشي تروح معاهم"]


def test_a_stutter_letter_in_the_models_text_is_taken_out_and_both_texts_are_logged(monkeypatch):
    model = "أنت ماشي تروح معاهم بدري وتشرب م من اللي أنت بتحبه وتقدمه على غيرك."
    out, blocks, details, lines = run(monkeypatch, [model])
    assert out == ["أنت ماشي تروح معاهم بدري وتشرب من اللي أنت بتحبه وتقدمه على غيرك."]
    assert ("EXTRACT item 1 as the model wrote it (before: stutter removed)", model) in blocks
    assert any("STUTTER REMOVED" in d for d in details)
    assert "stutter removed from 1" in lines[-1]


def test_a_stutter_the_model_already_left_out_is_accepted_as_it_is(monkeypatch):
    model = "أنت ماشي تروح معاهم بدري وتشرب من اللي"       # the transcript has "وتشرب م من": the stutter letter is not part of the passage
    out, blocks, _details, _lines = run(monkeypatch, [model])
    assert out == [model] and blocks == [("EXTRACT item 1 text", model)]


def test_repeated_stutter_letters_and_a_hanging_article_are_taken_out(monkeypatch):
    doc = SourceDocument("w", [Unit(0, "هي كتب ال كتب وأوراق."), Unit(1, "في البيت و و و و أنا صحيح.")])
    out = run(monkeypatch, ["كتب ال كتب وأوراق.", "في البيت و و و و أنا"], doc)[0]
    assert out[:2] == ["كتب كتب وأوراق.", "في البيت و أنا"]
    assert run(monkeypatch, ["اذهب ل ل لمن هم فعلا خبراء."])[0] == ["اذهب لمن هم فعلا خبراء."]


def test_dropped_quotation_marks_are_repaired_to_the_transcripts_own_words(monkeypatch):
    model = "الوقت لا ينتظر المتأخرين، وكلمة متأخرين يعني تأخيرا."
    out, blocks, _details, lines = run(monkeypatch, [model])
    assert out == ['"الوقت لا ينتظر المتأخرين"، وكلمة متأخرين يعني تأخيرا.']
    assert ("EXTRACT item 1 as the model wrote it (before: repaired)", model) in blocks
    assert "1 repaired to the transcript's own words" in lines[-1]


def test_a_word_the_speaker_never_said_is_repaired_by_taking_it_out(monkeypatch):
    assert run(monkeypatch, ["أنت ماشي تروح معاهم دايما بدري وتشرب"])[0] == ["أنت ماشي تروح معاهم بدري وتشرب"]


def test_a_few_words_the_model_skipped_are_put_back_inside_the_size_limit(monkeypatch):
    model = "أنت ماشي من اللي أنت بتحبه"               # skips تروح معاهم بدري وتشرب: the passage is 10 words, the limit is 2 x 6
    assert run(monkeypatch, [model])[0] == ["أنت ماشي تروح معاهم بدري وتشرب من اللي أنت بتحبه"]


def test_a_repair_never_bridges_to_another_place_of_the_talk(monkeypatch):
    model = "أنت ماشي بتحبه"                            # the gap would make the passage 10 words, the limit is 2 x 3
    assert run(monkeypatch, [model])[0] == ["أنت ماشي"]
    assert run(monkeypatch, ["الكتاب ده بيقول لازم تقرأي أولا. اذهب ل لمن هم فعلا خبراء."])[0] == ["الكتاب ده بيقول لازم تقرأي أولا."]


def test_only_a_text_with_no_anchor_in_the_transcript_is_left_out_and_written_whole_in_the_log(monkeypatch):
    invented = "مبدأ جديد تماما ومخترع"
    out, blocks, details, lines = run(monkeypatch, [invented])
    assert out == []
    assert ("EXTRACT item 1 LEFT OUT (as the model wrote it)", invented) in blocks
    assert any("LEFT OUT" in d and "NOT stored" in d for d in details)
    assert "1 left out (not in the transcript)" in lines[-1] and any("was left out" in ln for ln in lines[:-1])
    assert pe.MAX_REPAIR_GROWTH == 2 and pe.MIN_REPAIR_COVERAGE == 0.5


def test_nothing_that_is_not_one_continuous_passage_ever_leaves_the_function(monkeypatch):
    texts = [
        "أنت ماشي تروح معاهم بدري وتشرب م من اللي أنت بتحبه وتقدمه على غيرك.",      # stutter removed
        "اذهب ل لمن هم فعلا خبراء.",                                                  # stutter removed
        "كلمة مش موجودة خالص في الكلام ده",                                           # left out
        "الكلام ده قبل كده كتير.",                                                    # continuous
        "اذهب ل ل لمن هم فعلا خبراء. الكتاب ده",                                     # repaired to the first piece
    ]
    out, _blocks, _details, lines = run(monkeypatch, texts)
    joined = pe._clean_view(DOC)[2]
    assert len(out) == 4 and all(" ".join(t.split()) in joined for t in out)
    assert all(pe._conform(pe._clean_view(DOC), t) == (t, "as is") for t in out)
    assert "stutter removed from 2" in lines[-1] and "1 repaired" in lines[-1] and "1 left out" in lines[-1]


def test_an_empty_text_is_still_skipped_without_counting_as_left_out(monkeypatch):
    out, _blocks, _details, lines = run(monkeypatch, ["   ", "الكلام ده قبل كده كتير."])
    assert out == ["الكلام ده قبل كده كتير."] and "left out" not in lines[-1]


def test_a_run_stores_and_sends_only_passages_of_the_transcript_and_its_log_has_the_left_out_text(env, monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text))
    invented = "مبدأ جديد تماما ومخترع"
    llm.set_test_backend(audio=heard, json=lambda *a: ExtractionResult(items=[
        ExtractedPrinciple(text="فقرة 2 تانية فيها مبدأ.", domain="Faith"), ExtractedPrinciple(text=invented, domain="Faith")]))
    rl = pipeline._RunLogger("http://u", lambda line: None)
    assert pipeline.run_stages("vid", "عنوان", "http://u", rl).ok
    assert [p.principle for p in store.get_principles_for_video("vid")] == ["فقرة 2 تانية فيها مبدأ."]
    log = rl.path.read_text(encoding="utf-8")
    assert "EXTRACT item 2: LEFT OUT" in log and invented in log and "1 left out (not in the transcript)" in log
    assert sent and all("ومخترع" not in text for text in sent)
