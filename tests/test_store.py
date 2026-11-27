"""Storage: fresh schema, in-place migration of the real pre-restructure
database, timed principles, documents."""
import shutil
import sqlite3
from pathlib import Path

import pytest

from lifecompass import store


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    monkeypatch.setattr(store, "DB_PATH", path)
    store._prepared.clear()
    yield path
    store._prepared.clear()


def columns(path, table):
    con = sqlite3.connect(path)
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        con.close()


def version(path):
    con = sqlite3.connect(path)
    try:
        return con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()


def find_legacy_backup():
    for parent in Path(__file__).resolve().parents:
        cand = parent / "backups" / "restructure_20261006_before" / "data" / "principles.db"
        if cand.exists():
            return cand
    return None


def test_fresh_database_has_the_new_schema_and_version(scratch_db):
    store.count_all()
    cols = columns(scratch_db, "principles")
    assert "doc_hash" in cols
    assert not ({"source_quote", "trigger", "t_start", "unit_start", "unit_end"} & cols)
    assert "documents" in tables(scratch_db)
    assert "normalize_progress" not in tables(scratch_db)
    assert version(scratch_db) == store._SCHEMA_VERSION


def test_principle_roundtrip_keeps_its_document_hash(scratch_db):
    pid = store.insert_principle(store.Principle(
        id=None, principle="مفيش حد بيغير حد.", domain="Relationships",
        doc_hash="abc", video_id="vid",
        video_title="t", video_url="u"))
    p = store.get_principles_for_video("vid")[0]
    assert p.doc_hash == "abc"
    assert [x.id for x in store.get_principles_for_video("vid")] == [pid]


def test_duplicate_principle_text_is_not_inserted_twice(scratch_db):
    a = store.insert_principle(store.Principle(id=None, principle="كلام  مهم", video_id="v"))
    b = store.insert_principle(store.Principle(id=None, principle="كلام مهم", video_id="v"))
    assert a == b


def test_documents_save_and_replace(scratch_db):
    store.save_document("vid", '{"a": 1}', "h1")
    assert store.get_document_json("vid") == '{"a": 1}'
    store.save_document("vid", '{"a": 2}', "h2")
    assert store.get_document_json("vid") == '{"a": 2}'




def test_the_real_old_database_migrates_in_place_without_losing_anything(scratch_db):
    legacy = find_legacy_backup()
    if legacy is None:
        pytest.skip("pre-restructure backup not present")
    shutil.copy(legacy, scratch_db)
    before = sqlite3.connect(scratch_db)
    n_principles = before.execute("SELECT COUNT(*) FROM principles").fetchone()[0]
    texts = {r[0] for r in before.execute("SELECT principle FROM principles")}
    assert "source_quote" in columns(scratch_db, "principles")
    before.close()

    assert store.count_all() == n_principles            # first connect migrates
    cols = columns(scratch_db, "principles")
    assert not ({"source_quote", "trigger", "unit_start", "unit_end"} & cols)
    assert "doc_hash" in cols
    assert version(scratch_db) == store._SCHEMA_VERSION

    after = sqlite3.connect(scratch_db)
    assert not {"principle_links", "principle_links_archive"} & {r[0] for r in after.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {r[0] for r in after.execute("SELECT principle FROM principles")} == texts
    after.close()

    store._prepared.clear()                             # a second process: nothing left to do
    assert store.count_all() == n_principles


def test_queued_retries_under_old_stage_names_restart_under_the_new_ones(scratch_db):
    store.count_all()
    con = sqlite3.connect(scratch_db)
    con.execute("PRAGMA user_version = 1")             # pretend it is still an old schema
    for stage in ("review_quality", "normalize", "judge", "select"):
        con.execute("""INSERT INTO pending_retries
            (video_id, url, title, stage, items_json, model_used, attempts, created_at)
            VALUES (?, 'u', 't', ?, '[{"old": true}]', '', 0, 'now')""", (f"v_{stage}", stage))
    con.commit()
    con.close()
    store._prepared.clear()
    got = {p.video_id: (p.stage, p.items_json) for p in store.get_pending_retries()}
    assert got["v_review_quality"] == ("listen", "{}")
    assert got["v_normalize"] == ("listen", "{}")
    assert got["v_judge"][0] == "edit"
    assert got["v_select"][0] == "select"


def test_the_old_shortlist_functions_are_gone(scratch_db):
    assert not hasattr(store, "find_by_text_fuzzy")
