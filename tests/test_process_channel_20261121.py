"""Audit 2026-11-21, P8: process_channel.py stops when a stage is queued for retry (the old check matched a message that no longer exists)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("process_channel_under_test", ROOT / "scripts" / "process_channel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_batch(monkeypatch, results):
    mod = load()
    videos = [{"id": f"v{i}", "title": f"t{i}", "url": f"https://www.youtube.com/watch?v=v{i}"} for i in range(len(results))]
    monkeypatch.setattr(mod, "list_channel_videos", lambda url: videos)
    monkeypatch.setattr(mod.store, "count_processed_videos", lambda: 0)
    monkeypatch.setattr(mod.store, "is_video_processed", lambda vid: False)
    monkeypatch.setattr(mod.store, "count_all", lambda: 0)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    called = []
    it = iter(results)
    monkeypatch.setattr(mod, "process_video", lambda url, log=None: called.append(url) or next(it))
    monkeypatch.setattr(sys, "argv", ["process_channel.py", "https://www.youtube.com/@x/videos", "--delay", "0"])
    mod.main()
    return called


def ok():
    return SimpleNamespace(ok=True, message="done", added=2)


def test_stops_at_the_first_video_queued_for_retry(monkeypatch):
    queued = SimpleNamespace(ok=False, message="listen could not run, queued for retry: all models failed", added=0)
    assert len(run_batch(monkeypatch, [ok(), queued, ok(), ok()])) == 2


def test_other_failures_do_not_stop_the_batch(monkeypatch):
    broken = SimpleNamespace(ok=False, message="video unavailable", added=0)
    assert len(run_batch(monkeypatch, [broken, ok(), ok()])) == 3


def test_uses_the_projects_own_yt_dlp():
    text = (ROOT / "scripts" / "process_channel.py").read_text(encoding="utf-8")
    assert '"yt-dlp",' not in text and "sys.executable" in text
