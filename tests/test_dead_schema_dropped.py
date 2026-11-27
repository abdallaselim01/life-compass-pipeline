"""The dead parts of the database are gone (design decision 2026-11-13, schema 6): images carry no description, principles no t_start, and the
empty transcribe_progress table is not kept. A database from before is migrated in place without losing a row."""
import sqlite3

import pytest

from lifecompass import store

DEAD_COLUMNS = {"images": ("description", "description_failed", "describe_fp"),
                "images_history": ("description", "description_failed", "describe_fp"),
                "principles": ("t_start", "unit_start", "unit_end"),
                "principles_archive": ("unit_start", "unit_end")}


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    yield tmp_path / "t.db"
    store._prepared.clear()


def columns(db, table):
    con = sqlite3.connect(db)
    try:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    finally:
        con.close()


def tables(db):
    con = sqlite3.connect(db)
    try:
        return {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        con.close()


def test_a_new_database_has_none_of_the_dead_parts(scratch_db):
    store.insert_image("vid", "x.png")
    for table, dead in DEAD_COLUMNS.items():
        assert not set(dead) & columns(scratch_db, table), table
    assert "transcribe_progress" not in tables(scratch_db)
    assert sqlite3.connect(scratch_db).execute("PRAGMA user_version").fetchone()[0] == 8
    assert not {"principle_links", "principle_links_archive"} & tables(scratch_db)
    assert not hasattr(store.Image(id=1, video_id="v", enhanced_path="p"), "description")


def test_a_database_at_schema_5_is_migrated_in_place_and_loses_no_row(scratch_db):
    iid = store.insert_image("vid", "x.png")
    store.set_image_caption(iid, "كابشن")
    pid = store.insert_principle(store.Principle(id=None, principle="مبدأ", domain="Self", video_id="vid"))
    con = sqlite3.connect(scratch_db)                        # make it look like a schema-5 database
    con.execute("ALTER TABLE images ADD COLUMN description TEXT NOT NULL DEFAULT ''")
    con.execute("ALTER TABLE images ADD COLUMN description_failed INTEGER NOT NULL DEFAULT 0")
    con.execute("ALTER TABLE images ADD COLUMN describe_fp TEXT NOT NULL DEFAULT ''")
    con.execute("ALTER TABLE images_history ADD COLUMN description TEXT NOT NULL DEFAULT ''")
    con.execute("ALTER TABLE images_history ADD COLUMN description_failed INTEGER NOT NULL DEFAULT 0")
    con.execute("ALTER TABLE images_history ADD COLUMN describe_fp TEXT NOT NULL DEFAULT ''")
    con.execute("ALTER TABLE principles ADD COLUMN t_start INTEGER")
    con.execute("CREATE TABLE transcribe_progress (video_id TEXT NOT NULL, chunk_idx INTEGER NOT NULL, chunk_json TEXT NOT NULL, created_at TEXT NOT NULL)")
    con.execute("UPDATE images SET description = 'وصف قديم', describe_fp = 'abc' WHERE id = ?", (iid,))
    con.execute("INSERT INTO images_history (image_id, video_id, enhanced_path, standalone_caption, replaced_at, description) "
                "VALUES (?, 'vid', 'old.png', 'كابشن قديم', 'now', 'وصف أقدم')", (iid,))
    con.execute("PRAGMA user_version = 5")
    con.commit()
    con.close()

    store._prepared.clear()                                  # the next connection migrates
    image = store.get_unlinked_images_for_video("vid")[0]
    assert (image.id, image.standalone_caption) == (iid, "كابشن")
    assert store.get_principles_for_video("vid")[0].id == pid
    for table, dead in DEAD_COLUMNS.items():
        assert not set(dead) & columns(scratch_db, table), table
    assert "transcribe_progress" not in tables(scratch_db)
    con = sqlite3.connect(scratch_db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 8
    assert con.execute("SELECT image_id, enhanced_path, standalone_caption FROM images_history").fetchall() == [(iid, "old.png", "كابشن قديم")]
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()
