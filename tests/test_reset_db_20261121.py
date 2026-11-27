"""Audit 2026-11-21, P6: scripts/reset_db.py zeroes the database the way it is really done (images and settings kept, captions cleared)."""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("reset_db_under_test", ROOT / "scripts" / "reset_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scratch(tmp_path):
    (tmp_path / "data" / "enhanced_images").mkdir(parents=True)
    (tmp_path / "data" / "enhanced_images" / "a.png").write_bytes(b"png")
    (tmp_path / "data" / "model_health.json").write_text(json.dumps({"models": {"x": {"failures": 7}}}))
    db = tmp_path / "data" / "principles.db"
    c = sqlite3.connect(db)
    c.executescript("""
      CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE images(id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT, enhanced_path TEXT, standalone_caption TEXT NOT NULL DEFAULT '', times_shown INTEGER, last_shown_at TEXT);
      CREATE TABLE principles(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT);
      INSERT INTO settings VALUES('manual_enhanced_checkpoint','2026-11-21T00:33:57');
      INSERT INTO images(video_id,enhanced_path,standalone_caption,times_shown,last_shown_at) VALUES('v','p1','a caption',3,'2026-11-12'),('v','p2','',0,NULL);
      INSERT INTO principles(text) VALUES('p'),('q');
    """)
    c.execute("PRAGMA user_version=7")
    c.commit()
    c.close()
    return db


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    db = scratch(tmp_path)
    mod = load()
    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(sys, "argv", ["reset_db.py"])
    mod.main()
    assert "dry run" in capsys.readouterr().out
    assert sqlite3.connect(db).execute("select count(*) from principles").fetchone()[0] == 2


def test_go_empties_everything_but_settings_and_images_and_clears_only_the_captions(tmp_path, monkeypatch):
    db = scratch(tmp_path)
    mod = load()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "DB", db)
    monkeypatch.setattr(mod, "ENHANCED", tmp_path / "data" / "enhanced_images")
    monkeypatch.setattr(mod, "HEALTH", tmp_path / "data" / "model_health.json")
    monkeypatch.setattr(sys, "argv", ["reset_db.py", "--go", "--no-restart"])
    mod.main()
    c = sqlite3.connect(db)
    assert c.execute("select count(*) from principles").fetchone()[0] == 0
    assert c.execute("select value from settings").fetchone()[0] == "2026-11-21T00:33:57"
    assert c.execute("select id,enhanced_path,standalone_caption,times_shown,last_shown_at from images order by id").fetchall() == \
        [(1, "p1", "", 3, "2026-11-12"), (2, "p2", "", 0, None)]
    assert c.execute("pragma user_version").fetchone()[0] == 7
    assert json.loads((tmp_path / "data" / "model_health.json").read_text()) == {"models": {}}
    backup = next((tmp_path / "backups").glob("db_zero_*_before"))
    assert sqlite3.connect(backup / "principles.db").execute("select count(*) from principles").fetchone()[0] == 2
    assert (backup / "model_health.json").is_file() and (backup / "enhanced_images" / "a.png").is_file()
    c.execute("insert into principles(text) values('z')")
    assert c.execute("select id from principles").fetchone()[0] == 1            # the AUTOINCREMENT counter starts again at 1
