"""Matching and captioning that could not run right now are queued for retry_pending.py; the models are sent the pictures themselves."""
import sqlite3

import pytest

from lc_test_helpers import get_image
from lifecompass import llm, pipeline, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

# fixtures and helpers of the sibling test module
from test_image_replace import DOC, URL, VID, folders, pipe, rows, seed_video, stored_image  # noqa: F401


def quiet(*args, **kwargs):
    return None


def unavailable(*args, **kwargs):
    raise llm.StageSkippedError("no model")


def queued():
    return [(r.stage, pipeline.parse_supplementary_payload(r.items_json)) for r in store.get_pending_retries()]


def unlink(image_id):
    con = sqlite3.connect(store.DB_PATH)
    con.execute("DELETE FROM principle_images WHERE image_id = ?", (image_id,))
    con.commit()
    con.close()


def test_a_payload_written_by_an_older_version_still_reads_and_its_dead_steps_are_ignored():
    assert pipeline.parse_supplementary_payload('{"links": [1], "match": true, "captions": [], "describe": [5]}') == {
        "match": True, "captions": []}
    assert pipeline.parse_supplementary_payload("[3, 4]") == {"match": False, "captions": []}


def test_the_retry_matches_an_unlinked_image_from_its_picture(folders, pipe):
    pid, iid, _ = seed_video(folders)
    unlink(iid)
    pictures_seen = []

    def match(principles, pictures, log=print):
        pictures_seen.append(pictures)
        return {0: [0]}

    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", match)
    assert pipeline.retry_supplementary(VID, {"match": True, "captions": []}, quiet) is None
    assert pictures_seen == [[(b"old", "image/png")]]
    assert [im.id for im in store.get_images_for_principle(pid)] == [iid]


def test_matching_that_still_fails_stays_in_the_outstanding_work(folders, pipe):
    _, iid, _ = seed_video(folders)
    unlink(iid)
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", unavailable)
    left = pipeline.retry_supplementary(VID, {"match": True, "captions": []}, quiet)
    assert left == {"match": True, "captions": []}
    assert store.get_images_for_principle(store.get_principles_for_video(VID)[0].id) == []


def test_an_image_that_stays_unmatched_gets_its_caption_from_the_picture(folders, pipe):
    _, iid, _ = seed_video(folders)
    unlink(iid)
    given = []

    def caption(image, transcript, log=print):
        given.append((image, transcript))
        return "كابشن"

    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: {})
    pipe.monkeypatch.setattr(pipeline, "caption_standalone_image", caption)
    assert pipeline.retry_supplementary(VID, {"match": True, "captions": []}, quiet) is None
    assert given == [((b"old", "image/png"), DOC.full_text())]
    assert get_image(iid).standalone_caption == "كابشن"


def test_a_caption_that_still_fails_stays_in_the_outstanding_work(folders, pipe):
    _, iid, _ = seed_video(folders)
    unlink(iid)
    pipe.monkeypatch.setattr(pipeline, "caption_standalone_image", unavailable)
    left = pipeline.retry_supplementary(VID, {"match": False, "captions": [iid]}, quiet)
    assert left == {"match": False, "captions": [iid]}


def test_an_image_whose_file_is_gone_is_left_out_of_the_retry_and_not_queued_forever(folders, pipe):
    _, iid, path = seed_video(folders)
    unlink(iid)
    path.unlink()
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: pytest.fail("sent a missing picture"))
    pipe.monkeypatch.setattr(pipeline, "caption_standalone_image", lambda *a, **k: pytest.fail("captioned a missing picture"))
    assert pipeline.retry_supplementary(VID, {"match": True, "captions": []}, quiet) is None
    assert pipeline.retry_supplementary(VID, {"match": False, "captions": [iid]}, quiet) is None


def test_a_full_run_whose_matching_cannot_run_queues_it_and_holds_the_captions_back(folders, pipe):
    seed_video(folders)
    store.save_stage_fingerprint(VID, "extract", "f" * 16, "f" * 16, DOC.doc_hash)      # the extract prompt changed
    pipe.monkeypatch.setattr(pipeline, "video_info", lambda url: (VID, "عنوان"))
    pipe.monkeypatch.setattr(pipeline, "match_principles_to_images", unavailable)
    pipe.monkeypatch.setattr(pipeline, "caption_standalone_image", lambda *a, **k: pytest.fail("captioned before matching was settled"))
    llm.set_test_backend(audio=lambda *a: pytest.fail("listened"),
                         json=lambda *a: ExtractionResult(items=[ExtractedPrinciple(text="فقرة تانية قديمة فيها مبدأ.", domain="Faith")]))
    res = pipeline.process_video(URL, log=quiet)
    assert res.ok and res.added == 1
    assert queued() == [("supplementary", {"match": True, "captions": []})]


def test_a_video_whose_images_are_all_fine_queues_nothing(folders, pipe):
    seed_video(folders)
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."
    assert queued() == []


def test_a_row_whose_file_is_gone_is_not_queued(folders, pipe):
    seed_video(folders)
    store.insert_image(VID, str(folders.enhanced / f"{VID}_gone.png"))      # no file on disk
    pipeline.record_post_fingerprints(VID)
    assert pipeline.process_video(URL, log=quiet).message == "Already processed -- skipped."
    assert queued() == []
