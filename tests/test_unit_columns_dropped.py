"""Schema 7: a principle is the speaker's passage, not a range of numbered units, and the stage
rows of the removed stages are gone. A schema-6 database is migrated in place and keeps every row of what stays."""
import sqlite3

import pytest

from lifecompass import store
from lifecompass.document import SourceDocument


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    yield tmp_path / "t.db"
    store._prepared.clear()


def test_the_unit_range_is_gone_from_the_code():
    principle = store.Principle(id=None, principle="x")
    assert not hasattr(principle, "unit_start") and not hasattr(principle, "unit_end")
    assert not hasattr(SourceDocument("v", []), "span_text")


def test_a_schema_6_database_is_migrated_in_place_and_keeps_every_row_that_stays(scratch_db):
    pid = store.insert_principle(store.Principle(id=None, principle="مبدأ", domain="Self", video_id="vid", doc_hash="h"))
    con = sqlite3.connect(scratch_db)
    for table in ("principles", "principles_archive"):                  # make it look like a schema-6 database
        con.execute(f"ALTER TABLE {table} ADD COLUMN unit_start INTEGER")
        con.execute(f"ALTER TABLE {table} ADD COLUMN unit_end INTEGER")
    con.execute("UPDATE principles SET unit_start = 3, unit_end = 4")
    con.execute("INSERT INTO principles_archive (id, principle, domain, video_id, doc_hash, created_at, archived_at, reason, unit_start) "
                "VALUES (99, 'قديم', 'Self', 'vid', 'h', 'c', 'a', 'why', 1)")
    for stage in ("listen", "extract", "match", "caption", "links", "select", "edit"):
        con.execute("INSERT INTO stage_fingerprints (video_id, stage, prompt_fp, effective_fp, produced_at) "
                    "VALUES ('vid', ?, 'p', 'e', 'now')", (stage,))
    for stage in ("find_related", "match"):
        con.execute("INSERT INTO pending_retries (video_id, url, title, stage, items_json, model_used, attempts, created_at) "
                    "VALUES (?, 'u', 't', ?, '{}', '', 0, 'now')", ("v_" + stage, stage))
    con.execute("PRAGMA user_version = 6")
    con.commit()
    con.close()

    store._prepared.clear()                                             # the next connection migrates
    principle = store.get_principles_for_video("vid")[0]
    assert (principle.id, principle.principle, principle.doc_hash) == (pid, "مبدأ", "h")
    con = sqlite3.connect(scratch_db)
    for table in ("principles", "principles_archive"):
        assert not {"unit_start", "unit_end"} & {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
    assert con.execute("SELECT id, principle, reason FROM principles_archive").fetchall() == [(99, "قديم", "why")]
    assert sorted(row[0] for row in con.execute("SELECT stage FROM stage_fingerprints")) == ["caption", "extract", "listen", "match"]
    assert [row[0] for row in con.execute("SELECT stage FROM pending_retries")] == ["match"]
    assert con.execute("PRAGMA user_version").fetchone()[0] == 8
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()
