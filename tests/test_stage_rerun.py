"""A processed video is redone only from the first stage whose prompt changed; the old results are archived, not deleted."""
import re
import sqlite3

import pytest

from lc_test_helpers import get_image
from lifecompass import audio as audio_mod
from lifecompass import llm, pipeline, store
from lifecompass import stage_fingerprints as sf
from lifecompass.document import SourceDocument, Unit
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult
from lifecompass.runlog import sha16

VID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VID}"
OLD_DOC = SourceDocument(VID, [Unit(0, "فقرة قديمة فيها كلام."), Unit(1, "فقرة تانية قديمة فيها مبدأ.")])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    sent = []
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text))
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "caption_standalone_image", lambda *a, **k: "")
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "AUTO_BACKUPS_DIR", tmp_path / "auto")
    monkeypatch.setattr(pipeline, "video_info", lambda url: (VID, "عنوان"))
    monkeypatch.setattr(pipeline, "process_pending_images", lambda video_id, log=None, **kw: [])
    monkeypatch.setattr(audio_mod, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    yield sent
    store._prepared.clear()


def rlog():
    return pipeline._RunLogger(URL, lambda line: None)


def write_prompts(root, listen="listen prompt", extract="extract prompt", eol="\n"):
    folder = root / "prompts"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "transcribe_prompt.txt").write_bytes(listen.replace("\n", eol).encode("utf-8"))
    (folder / "extract_prompt.txt").write_bytes(extract.replace("\n", eol).encode("utf-8"))
    for name in ("listen_request.txt", "listen_retry_note.txt", "extract_request.txt"):      # the request texts are part of a stage's prompt
        (folder / name).write_bytes(f"request text {name}".encode("utf-8"))
    return root


def seed(stale=None):
    """A video processed earlier: document, transcript, one principle (linked to an image, and another video's principle next to it), the processed mark and the fingerprints of the current prompts, except for the stage named in `stale`."""
    store.save_document(VID, OLD_DOC.to_json(), OLD_DOC.doc_hash)
    store.insert_transcript(VID, OLD_DOC.full_text())
    old = store.insert_principle(store.Principle(id=None, principle="مبدأ قديم", domain="Self", doc_hash=OLD_DOC.doc_hash,
                                                 video_id=VID, video_title="عنوان", video_url=URL))
    other = store.insert_principle(store.Principle(id=None, principle="مبدأ من فيديو تاني", domain="Money", video_id="otherothr1"))
    image = store.insert_image(VID, "x.png")
    store.link_principle_image(old, image)
    store.mark_video_processed(VID, URL, "عنوان", 1)
    cur = sf.current_fingerprints()
    for stage, fp in cur.items():
        eff = "f" * 16 if stage == stale else fp.effective_fp
        store.save_stage_fingerprint(VID, stage, "0" * 16 if stage == stale else fp.prompt_fp, eff, OLD_DOC.doc_hash,
                                     settings_fp=fp.settings_fp)
    pipeline.record_post_fingerprints(VID)
    return old, other, image


def heard(audio, prompt, system, context):
    # what the two pieces used to say, now said by one recording
    return "\n\n".join(f"فقرة {idx} جديدة فيها كلام.\n\nفقرة {idx} تانية فيها مبدأ." for idx in (1, 2))


def extract_ok(prompt, schema, system, context):
    assert schema is ExtractionResult
    last_paragraph = prompt.split("\n\n")[-1]            # the model must copy a passage that is really in the transcript
    return ExtractionResult(items=[ExtractedPrinciple(text=last_paragraph, domain="Faith")])


def extract_fails(prompt, schema, system, context):
    raise llm.StageSkippedError("no model")


def rows(sql, *args):
    con = sqlite3.connect(store.DB_PATH)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


# ---- fingerprints and the plan ---------------------------------------------------------------------------------
def test_a_change_of_line_endings_alone_does_not_change_a_fingerprint(tmp_path):
    lf = write_prompts(tmp_path / "a", listen="سطر\nتاني")
    crlf = write_prompts(tmp_path / "b", listen="سطر\nتاني", eol="\r\n")
    assert sf.prompt_fp("listen", lf) == sf.prompt_fp("listen", crlf)
    assert sf.prompt_fp("listen", lf) != sf.prompt_fp("listen", write_prompts(tmp_path / "c", listen="سطر\nتالت"))


def test_the_fingerprint_of_a_stage_is_over_the_hashes_of_its_lf_prompt_files_the_run_log_prints():
    for stage, names in sf.STAGE_PROMPTS.items():
        raws = [(sf.PROJECT_ROOT / "prompts" / name).read_bytes() for name in names]
        assert all(b"\r\n" not in raw for raw in raws)
        parts = [sha16(raw) for raw in raws]
        assert sf.prompt_fp(stage) == (parts[0] if len(parts) == 1 else sf.sha16("\n".join(parts)))


def test_a_new_listen_prompt_changes_extract_too_but_a_new_extract_prompt_leaves_listen_alone(tmp_path):
    base = sf.current_fingerprints(write_prompts(tmp_path / "base"))
    new_listen = sf.current_fingerprints(write_prompts(tmp_path / "l", listen="other"))
    new_extract = sf.current_fingerprints(write_prompts(tmp_path / "e", extract="other"))
    assert new_listen["listen"].effective_fp != base["listen"].effective_fp
    assert new_listen["extract"].prompt_fp == base["extract"].prompt_fp
    assert new_listen["extract"].effective_fp != base["extract"].effective_fp
    assert new_extract["listen"].effective_fp == base["listen"].effective_fp
    assert new_extract["extract"].effective_fp != base["extract"].effective_fp


def stored_of(cur):
    return {s: fp.effective_fp for s, fp in cur.items()}


def test_plan_nothing_to_redo_when_every_fingerprint_matches(tmp_path):
    cur = sf.current_fingerprints(write_prompts(tmp_path))
    p = sf.plan(stored_of(cur), cur)
    assert not p.needed and p.rerun_from is None and p.reuse == ("listen", "extract")


def test_plan_a_changed_extract_prompt_reuses_the_transcript(tmp_path):
    old = sf.current_fingerprints(write_prompts(tmp_path / "o"))
    new = sf.current_fingerprints(write_prompts(tmp_path / "n", extract="better"))
    p = sf.plan(stored_of(old), new)
    assert (p.rerun_from, p.rerun, p.reuse) == ("extract", ("extract",), ("listen",))
    assert p.note() == "Reprocessing: the prompt of stage extract changed"


def test_plan_a_changed_listen_prompt_redoes_listen_and_everything_after_it(tmp_path):
    old = sf.current_fingerprints(write_prompts(tmp_path / "o"))
    new = sf.current_fingerprints(write_prompts(tmp_path / "n", listen="better"))
    p = sf.plan(stored_of(old), new)
    assert (p.rerun_from, p.rerun, p.reuse) == ("listen", ("listen", "extract"), ())
    assert p.reasons["extract"] == "follows listen"


def test_plan_a_stage_with_no_recorded_fingerprint_is_redone(tmp_path):
    cur = sf.current_fingerprints(write_prompts(tmp_path))
    assert sf.plan({}, cur).rerun_from == "listen"
    p = sf.plan({"listen": cur["listen"].effective_fp}, cur)
    assert p.rerun_from == "extract" and p.note() == "Reprocessing: stage extract has no recorded fingerprint"


def test_a_run_that_starts_at_extract_builds_on_the_recorded_listen_fingerprint(tmp_path):
    pfps = {s: fp.prompt_fp for s, fp in sf.current_fingerprints(write_prompts(tmp_path)).items()}
    got = sf.effective_chain(pfps, "extract", {"listen": "a" * 16})
    assert got == {"extract": (pfps["extract"], sf.chain(sf.with_settings("extract", pfps["extract"]), "a" * 16))}
    assert sf.effective_chain(pfps, "extract", {}) == {}
    assert list(sf.effective_chain(pfps, "listen", {})) == ["listen", "extract"]


# ---- storage ---------------------------------------------------------------------------------------------------
def test_new_tables_exist_and_preparing_the_database_again_is_harmless(env):
    store.count_all()
    store._prepared.clear()
    store.count_all()
    names = {r[0] for r in rows("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"stage_fingerprints", "documents_history", "principles_archive",
            "principle_images_archive"} <= names
    assert not {"principle_links", "principle_links_archive"} & names                  # dropped at schema 8


def test_a_stage_fingerprint_is_saved_replaced_and_read_back(env):
    store.save_stage_fingerprint("v", "listen", "p1", "e1", "d1", "a.log")
    store.save_stage_fingerprint("v", "listen", "p2", "e2", "d2", "b.log")
    store.save_stage_fingerprint("v", "extract", "p3", "e3", "d2", "b.log")
    got = store.get_stage_fingerprints("v")
    assert set(got) == {"listen", "extract"}
    assert (got["listen"]["prompt_fp"], got["listen"]["effective_fp"], got["listen"]["log_file"]) == ("p2", "e2", "b.log")
    assert store.get_stage_fingerprints("nobody") == {}


def test_archiving_moves_principles_and_image_links_and_leaves_the_document(env):
    old, other, image = seed()
    counts = store.archive_video_results(VID, "why")
    assert counts == {"principles": 1, "image_links": 1, "documents": 0}
    assert store.get_principles_for_video(VID) == []
    assert rows("SELECT id FROM principles WHERE id = ?", other) == [(other,)]                       # another video's principle is untouched
    assert store.get_images_for_principle(old) == []
    assert rows("SELECT id, principle, reason FROM principles_archive") == [(old, "مبدأ قديم", "why")]
    assert rows("SELECT principle_id, image_id FROM principle_images_archive") == [(old, image)]
    assert store.get_document_json(VID) is not None                     # a redo from extract reuses it
    assert get_image(image) is not None                           # the screenshot itself stays


def test_archiving_with_the_document_keeps_the_old_document_and_transcript_in_history(env):
    seed()
    assert store.archive_video_results(VID, "why", archive_document=True)["documents"] == 1
    assert store.get_document_json(VID) is None
    (vid, doc_json, doc_hash, transcript, reason), = rows(
        "SELECT video_id, doc_json, doc_hash, transcript_text, reason FROM documents_history")
    assert (vid, doc_json, doc_hash, reason) == (VID, OLD_DOC.to_json(), OLD_DOC.doc_hash, "why")
    assert transcript == OLD_DOC.full_text()


def test_a_database_backup_is_a_complete_copy(env, tmp_path):
    seed()
    target = store.backup_database(tmp_path / "bk")
    con = sqlite3.connect(target)
    try:
        assert con.execute("SELECT COUNT(*) FROM principles").fetchone()[0] == store.count_all()
        assert con.execute("SELECT COUNT(*) FROM stage_fingerprints").fetchone()[0] == 4      # 2 principle stages + match, caption
    finally:
        con.close()


# ---- the bookmarklet path: process_video ------------------------------------------------------------------------
def test_a_video_whose_prompts_have_not_changed_is_skipped_with_no_model_call(env, monkeypatch):
    seed()
    monkeypatch.setattr(pipeline, "video_info", lambda url: pytest.fail("the video must not even be read"))
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=lambda *a: pytest.fail("extracted"))
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and res.message == "Already processed -- skipped."
    assert len(store.get_principles_for_video(VID)) == 1 and rows("SELECT * FROM principles_archive") == []


def test_a_changed_extract_prompt_redoes_extract_only_and_archives_the_old_principles(env, tmp_path):
    old, _, _ = seed(stale="extract")
    listened = []
    llm.set_test_backend(audio=lambda *a: listened.append(a), json=extract_ok)
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and res.added == 1 and listened == []                 # the saved transcript was reused
    assert [p.principle for p in store.get_principles_for_video(VID)] == ["فقرة تانية قديمة فيها مبدأ."]
    assert rows("SELECT id FROM principles_archive") == [(old,)]
    assert store.get_document_json(VID) == OLD_DOC.to_json() and rows("SELECT * FROM documents_history") == []
    cur = sf.current_fingerprints()
    fps = store.get_stage_fingerprints(VID)
    assert fps["extract"]["effective_fp"] == cur["extract"].effective_fp
    assert fps["listen"]["effective_fp"] == cur["listen"].effective_fp
    assert any("Reprocessing: the prompt of stage extract changed" in text and "فيها مبدأ" in text for text in env)
    assert (tmp_path / "auto").is_dir() and len(list((tmp_path / "auto").glob("*before_rerun*extract*/principles.db"))) == 1


def test_a_changed_listen_prompt_listens_again_and_keeps_the_old_document_in_history(env):
    seed(stale="listen")
    listened = []

    def counting(audio, prompt, system, context):
        listened.append(context)
        return heard(audio, prompt, system, context)

    llm.set_test_backend(audio=counting, json=extract_ok)
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and res.added == 1 and len(listened) == 1
    assert [r[0] for r in rows("SELECT doc_json FROM documents_history")] == [OLD_DOC.to_json()]
    assert store.get_document_json(VID) != OLD_DOC.to_json()
    cur = sf.current_fingerprints()
    fps = store.get_stage_fingerprints(VID)
    assert fps["listen"]["effective_fp"] == cur["listen"].effective_fp
    assert fps["extract"]["effective_fp"] == cur["extract"].effective_fp
    assert any("Reprocessing: the prompt of stage listen changed" in text for text in env)


def test_a_failed_redo_keeps_the_old_results_archived_and_resumes_from_extract(env):
    old, _, _ = seed(stale="extract")
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=extract_fails)
    assert not pipeline.process_video(URL, log=lambda line: None).ok
    pend = store.get_pending_retries()[0]
    assert pend.video_id == VID and pend.stage == "extract"
    assert store.get_principles_for_video(VID) == [] and rows("SELECT id FROM principles_archive") == [(old,)]
    assert store.get_stage_fingerprints(VID)["extract"]["effective_fp"] == "f" * 16      # still the old one

    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=extract_ok)
    again = pipeline.run_stages(VID, "عنوان", URL, rlog(), stage=pend.stage)
    assert again.ok and again.added == 1 and store.get_pending_retries() == []
    assert store.get_stage_fingerprints(VID)["extract"]["effective_fp"] == sf.current_fingerprints()["extract"].effective_fp
    assert pipeline.process_video(URL, log=lambda line: None).message == "Already processed -- skipped."


def test_a_video_queued_for_retry_is_not_touched_even_when_its_prompts_changed(env):
    seed(stale="extract")
    store.save_pending_retry(VID, URL, "عنوان", "extract", "{}", "", "earlier error")
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=lambda *a: pytest.fail("extracted"))
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and "queued for retry" in res.message
    assert len(store.get_principles_for_video(VID)) == 1 and rows("SELECT * FROM principles_archive") == []


def test_when_the_backup_fails_nothing_is_archived_and_nothing_is_redone(env, monkeypatch):
    seed(stale="extract")

    def broken(dest):
        raise OSError("disk full")

    monkeypatch.setattr(store, "backup_database", broken)
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=lambda *a: pytest.fail("extracted"))
    res = pipeline.process_video(URL, log=lambda line: None)
    assert not res.ok and "back the database up" in res.message
    assert len(store.get_principles_for_video(VID)) == 1 and rows("SELECT * FROM principles_archive") == []
    assert store.get_document_json(VID) == OLD_DOC.to_json()


def test_a_new_video_records_the_fingerprint_of_listen_and_of_extract(env):
    llm.set_test_backend(audio=heard, json=extract_ok)
    assert pipeline.run_stages(VID, "عنوان", URL, rlog()).ok
    cur = sf.current_fingerprints()
    fps = store.get_stage_fingerprints(VID)
    assert {s: (r["prompt_fp"], r["effective_fp"]) for s, r in fps.items() if s in cur} == {
        s: (fp.prompt_fp, fp.effective_fp) for s, fp in cur.items()}
    assert fps["listen"]["doc_hash"] == fps["extract"]["doc_hash"] != ""
    assert fps["extract"]["log_file"].endswith(".log")
    assert {"match", "caption"} <= set(fps) and "links" not in fps                       # the stages after the principles are recorded too
    assert fps["match"]["settings_fp"] == llm.settings_fp("match") != ""
    assert fps["listen"]["settings_fp"] == llm.settings_fp("listen") != ""
