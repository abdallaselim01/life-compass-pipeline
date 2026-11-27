"""An image dropped again replaces the stored one in place (the picture; images are no longer described)."""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lc_test_helpers import get_image
from lifecompass import images, llm, pipeline, store
from lifecompass import stage_fingerprints as sf
from lifecompass.document import SourceDocument, Unit
from lifecompass.images import ProcessedImage
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

VID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VID}"
DOC = SourceDocument(VID, [Unit(0, "فقرة قديمة فيها كلام."), Unit(1, "فقرة تانية قديمة فيها مبدأ.")])


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """A scratch database, an 'enhanced' folder and an 'incoming' folder."""
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    enhanced, incoming = tmp_path / "enhanced", tmp_path / "incoming"
    enhanced.mkdir()
    incoming.mkdir()
    monkeypatch.setattr(images, "ENHANCED_DIR", enhanced)
    monkeypatch.setattr(images, "REPLACED_DIR", enhanced / "replaced")
    monkeypatch.setattr(images, "_incoming_dir", lambda: incoming)
    yield SimpleNamespace(enhanced=enhanced, incoming=incoming)
    store._prepared.clear()


def set_checkpoint(when):
    store.set_setting(images._CHECKPOINT_KEY, when.isoformat())


def rows(sql, *args):
    con = sqlite3.connect(store.DB_PATH)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def stored_image(folders, name="pic.png", content=b"old"):
    """An image already stored for VID: its file in the enhanced folder and its row."""
    path = folders.enhanced / f"{VID}_{name}"
    path.write_bytes(content)
    return store.insert_image(VID, str(path)), path


# ---- storage ---------------------------------------------------------------------------------------------------
def test_replacing_an_image_keeps_its_id_and_archives_what_it_was(folders):
    iid, path = stored_image(folders)
    pid = store.insert_principle(store.Principle(id=None, principle="مبدأ", domain="Self", video_id=VID))
    store.link_principle_image(pid, iid)
    store.set_image_caption(iid, "كابشن قديم")
    store.mark_image_shown(iid)
    counts = store.replace_image(iid, str(path), "why")
    assert counts == {"links": 1}
    now = get_image(iid)
    assert (now.standalone_caption, now.times_shown) == ("", 1)
    assert store.get_images_for_principle(pid) == []                       # re-matched later
    assert rows("SELECT image_id, standalone_caption, reason FROM images_history") == [(iid, "كابشن قديم", "why")]
    assert rows("SELECT principle_id, image_id, reason FROM principle_images_archive") == [(pid, iid, "why")]
    with pytest.raises(KeyError):
        store.replace_image(999, "x.png", "why")


# ---- intake: the same image dropped again -----------------------------------------------------------------------
def test_a_name_that_already_has_the_video_prefix_does_not_get_it_twice():
    assert images._dest_name(VID, "pic.png") == f"{VID}_pic.png"
    assert images._dest_name(VID, f"{VID}_pic.png") == f"{VID}_pic.png"


def test_a_new_image_is_moved_into_the_enhanced_folder_without_any_model_call(folders):
    set_checkpoint(datetime.now() - timedelta(days=1))
    (folders.incoming / "new.png").write_bytes(b"fresh")
    (res,) = images.process_pending_images(VID, log=lambda *a: None)
    assert res.replaces_image_id is None
    assert (folders.enhanced / f"{VID}_new.png").read_bytes() == b"fresh" and not any(folders.incoming.iterdir())


def test_the_same_image_dropped_again_replaces_it_even_when_the_file_looks_older_than_the_checkpoint(folders):
    iid, path = stored_image(folders)
    set_checkpoint(datetime.now() + timedelta(days=1))                    # every file in the folder counts as "old"
    (folders.incoming / "pic.png").write_bytes(b"new")
    (res,) = images.process_pending_images(VID, log=lambda *a: None)
    assert res.replaces_image_id == iid and res.enhanced_path == path
    assert path.read_bytes() == b"new" and not any(folders.incoming.iterdir())
    (kept,) = list((folders.enhanced / "replaced").iterdir())               # the old file is kept, never deleted
    assert kept.name.endswith(f"_{VID}_pic.png") and kept.read_bytes() == b"old"


def test_a_file_dropped_with_the_prefix_already_on_it_replaces_without_doubling_it(folders):
    iid, path = stored_image(folders)
    set_checkpoint(datetime.now() + timedelta(days=1))
    (folders.incoming / f"{VID}_pic.png").write_bytes(b"new")
    (res,) = images.process_pending_images(VID, log=lambda *a: None)
    assert res.replaces_image_id == iid and path.read_bytes() == b"new"
    assert sorted(p.name for p in folders.enhanced.iterdir() if p.is_file()) == [f"{VID}_pic.png"]


def test_a_row_whose_file_went_missing_is_healed_by_dropping_the_image_again(folders):
    iid, path = stored_image(folders)
    path.unlink()
    set_checkpoint(datetime.now() + timedelta(days=1))
    (folders.incoming / "pic.png").write_bytes(b"new")
    (res,) = images.process_pending_images(VID, log=lambda *a: None)
    assert res.replaces_image_id == iid and path.read_bytes() == b"new"
    assert not (folders.enhanced / "replaced").exists()


def test_replacements_only_takes_replacements_and_leaves_new_images_and_the_checkpoint_alone(folders):
    stored_image(folders)
    before = datetime.now() - timedelta(days=1)
    set_checkpoint(before)
    (folders.incoming / "pic.png").write_bytes(b"new")
    (folders.incoming / "for_the_next_video.png").write_bytes(b"x")
    results = images.process_pending_images(VID, log=lambda *a: None, replacements_only=True)
    assert [r.enhanced_path.name for r in results] == [f"{VID}_pic.png"]
    assert (folders.incoming / "for_the_next_video.png").exists()
    assert store.get_setting(images._CHECKPOINT_KEY) == before.isoformat()


# ---- the pipeline ----------------------------------------------------------------------------------------------
@pytest.fixture
def pipe(folders, tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text))
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "caption_standalone_image", lambda *a, **k: "")
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(pipeline, "AUTO_BACKUPS_DIR", tmp_path / "auto")
    monkeypatch.setattr(pipeline, "video_info", lambda url: pytest.fail("an up-to-date video must not even be read"))
    monkeypatch.setattr(pipeline, "process_pending_images", lambda video_id, log=None, **kw: [])
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"), json=lambda *a: pytest.fail("extracted"))
    return SimpleNamespace(sent=sent, monkeypatch=monkeypatch)


def seed_video(folders):
    """A video processed earlier with the current prompts, one principle linked to one stored image."""
    store.save_document(VID, DOC.to_json(), DOC.doc_hash)
    store.insert_transcript(VID, DOC.full_text())
    pid = store.insert_principle(store.Principle(id=None, principle="مبدأ", domain="Self", doc_hash=DOC.doc_hash,
                                                 video_id=VID, video_title="عنوان", video_url=URL))
    iid, path = stored_image(folders)
    store.link_principle_image(pid, iid)
    store.mark_video_processed(VID, URL, "عنوان", 1)
    for stage, fp in sf.current_fingerprints().items():
        store.save_stage_fingerprint(VID, stage, fp.prompt_fp, fp.effective_fp, DOC.doc_hash, settings_fp=fp.settings_fp)
    pipeline.record_post_fingerprints(VID)
    return pid, iid, path


def test_dropping_the_same_image_again_on_an_up_to_date_video_replaces_it_and_matches_the_new_picture(folders, pipe):
    pid, iid, path = seed_video(folders)
    pipe.monkeypatch.setattr(pipeline, "process_pending_images", images.process_pending_images)
    pictures_seen = []

    def match(principles, pictures, log=print):
        pictures_seen.append(pictures)
        return {0: [0]}

    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", match)
    set_checkpoint(datetime.now() + timedelta(days=1))
    (folders.incoming / "pic.png").write_bytes(b"new")
    (folders.incoming / "for_the_next_video.png").write_bytes(b"x")
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and "1 image(s) refreshed" in res.message
    assert path.read_bytes() == b"new" and pictures_seen == [[(b"new", "image/png")]]    # the matcher was sent the new picture itself
    assert [im.id for im in store.get_images_for_principle(pid)] == [iid]
    assert (folders.incoming / "for_the_next_video.png").exists()                 # not swept into the old video
    assert len(rows("SELECT * FROM images_history")) == 1
    assert len(rows("SELECT * FROM images")) == 1 and pipe.sent == []


def test_when_the_stages_are_redone_a_dropped_image_replaces_the_row_and_the_others_are_left_alone(folders, pipe):
    pid, iid, path = seed_video(folders)
    other, other_path = stored_image(folders, "other.png", b"o")
    store.save_stage_fingerprint(VID, "extract", "f" * 16, "f" * 16, DOC.doc_hash)      # the extract prompt changed
    pipe.monkeypatch.setattr(pipeline, "video_info", lambda url: (VID, "عنوان"))
    dropped = ProcessedImage(enhanced_path=path, replaces_image_id=iid)
    pipe.monkeypatch.setattr(pipeline, "process_pending_images", lambda video_id, log=None, **kw: [dropped])
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: {})
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"),
                         json=lambda *a: ExtractionResult(items=[ExtractedPrinciple(text="فقرة تانية قديمة فيها مبدأ.", domain="Faith")]))
    res = pipeline.process_video(URL, log=lambda line: None)
    assert res.ok and res.added == 1
    assert Path(get_image(iid).enhanced_path) == path                    # the same row, now holding the dropped file
    assert Path(get_image(other).enhanced_path) == other_path            # untouched
    assert len(rows("SELECT * FROM images")) == 2 and len(rows("SELECT * FROM images_history")) == 1
