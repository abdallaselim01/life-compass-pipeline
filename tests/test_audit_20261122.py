"""Audit 2026-11-22: command time limits, escaping in Telegram texts, key redaction, a run that is recorded while it runs (and found
when its process dies), a first step that fails is queued, identical re-dropped images, the test log archive."""
import json
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lifecompass import audio, images, llm, logging_setup, pipeline, resurface, store, tunables
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

VID = "abcdefghijk"
URL = f"https://www.youtube.com/watch?v={VID}"


# ---- command time limits ------------------------------------------------------------------------------------------
def test_a_hung_command_is_stopped_and_reported_as_a_failed_command():
    r = audio._run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert r.returncode == 124 and "timed out" in r.stderr
    ok = audio._run([sys.executable, "-c", "print('fine')"], timeout=30)
    assert ok.returncode == 0 and ok.stdout.strip() == "fine"


def test_every_yt_dlp_and_ffmpeg_call_has_a_limit():
    assert tunables.get("audio", "info_timeout_seconds") == 300.0
    assert tunables.get("audio", "download_timeout_seconds") == 3600.0
    assert tunables.get("audio", "probe_timeout_seconds") == 60.0
    seen = []
    real = audio._run
    audio._run = lambda cmd, timeout=None, **kw: (seen.append(timeout), SimpleNamespace(returncode=1, stdout="", stderr="x"))[1]
    try:
        for call in (lambda: audio.video_info(URL), lambda: audio.video_duration(VID)):
            with pytest.raises(RuntimeError):
                call()
    finally:
        audio._run = real
    assert seen == [300.0, 300.0]


# ---- redaction ----------------------------------------------------------------------------------------------------
def test_the_current_google_key_format_is_masked():
    key = "AQ." + "Ab3_-" * 10
    assert key not in logging_setup.redact(f"error calling with key {key} at once")
    assert "<redacted-google-key>" in logging_setup.redact(key)
    assert "AIza" + "x" * 35 not in logging_setup.redact("AIza" + "x" * 35)


# ---- Telegram HTML ------------------------------------------------------------------------------------------------
def test_a_principle_with_special_characters_is_escaped_for_telegram():
    p = SimpleNamespace(principle="لو a < b & c > d")
    assert resurface._format_message(p) == "<b>لو a &lt; b &amp; c &gt; d</b>"


# ---- the test log archive -----------------------------------------------------------------------------------------
def test_tests_never_rotate_into_the_real_log_archive():
    assert "logs" not in str(logging_setup.ARCHIVE_DIR).replace("log_archive", "")
    assert logging_setup.ARCHIVE_DIR.name == "log_archive"


# ---- the pipeline: a run is recorded while it runs ----------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    sent = []
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text))
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(audio, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio, "recording_bytes", lambda p: b"audio")
    yield sent
    store._prepared.clear()


def rlog():
    return pipeline._RunLogger("http://u", lambda line: None)


def heard(audio_bytes, prompt, system, context):
    return "\n".join(f"جملة {idx}-{j} كلمة تانية." for idx in (1, 2, 3) for j in range(5))


PRINCIPLE = "جملة 2-1 كلمة تانية. جملة 2-2 كلمة تانية."


def extract_one(prompt, schema, system, context):
    return ExtractionResult(items=[ExtractedPrinciple(text=PRINCIPLE, domain="Self")])


def test_a_running_video_has_a_queue_row_that_a_normal_ending_removes(env):
    seen = []

    def listening(audio_bytes, prompt, system, context):
        seen.append([(p.video_id, p.stage, p.last_error, json.loads(p.items_json)) for p in store.get_pending_retries()])
        return heard(audio_bytes, prompt, system, context)

    llm.set_test_backend(audio=listening, json=extract_one)
    res = pipeline.run_stages("vid", "عنوان", "http://u", rlog())
    assert res.ok and res.added == 1
    assert seen == [[("vid", "listen", store.RUNNING, {"pid": os.getpid()})]]         # recorded while it ran
    assert store.get_pending_retries() == []                                          # gone when it finished


def test_a_failure_of_a_running_video_still_tells_the_user_once(env):
    def fails(audio_bytes, prompt, system, context):
        raise llm.StageSkippedError("no model")

    llm.set_test_backend(audio=fails, json=extract_one)
    assert not pipeline.run_stages("vid", "عنوان", "http://u", rlog()).ok
    pend = store.get_pending_retries()
    assert [(p.video_id, p.stage) for p in pend] == [("vid", "listen")] and pend[0].last_error != store.RUNNING
    assert sum("retry queue" in text for text in env) == 1
    # the retry (a row exists already) fails again at the same stage: no second message
    assert not pipeline.run_stages("vid", "عنوان", "http://u", rlog(), stage="listen").ok
    assert sum("retry queue" in text for text in env) == 1


def test_a_row_marking_a_live_run_is_left_alone_and_an_orphan_is_taken_over(env):
    now = datetime(2026, 11, 22, 12, 0, 0)

    def row(minutes_ago, pid=os.getpid(), error=store.RUNNING):
        stamp = (now - timedelta(minutes=minutes_ago)).isoformat()
        return SimpleNamespace(last_error=error, last_attempt_at=stamp, created_at=stamp, items_json=json.dumps({"pid": pid}))

    assert pipeline.run_is_live(row(5), now)                                  # recent heartbeat
    assert pipeline.run_is_live(row(90), now)                                 # the machine slept: its process is still here
    assert not pipeline.run_is_live(row(90, pid=3999999), now)                # its process is gone: an orphan
    assert not pipeline.run_is_live(row(7 * 60), now)                         # far too old, whatever the id says
    assert not pipeline.run_is_live(row(1, error="a failure"), now)           # an ordinary queued retry is never "live"
    assert not pipeline.run_is_live(SimpleNamespace(last_error=store.RUNNING, last_attempt_at="x", created_at="x", items_json="{}"), now)


def test_the_heartbeat_only_touches_a_row_that_still_says_running(env):
    assert store.mark_run_started("vid", "u", "t", "listen", 1) is True
    assert store.mark_run_started("vid", "u", "t", "listen", 2) is False      # a row exists: left as it is
    first = store.get_pending_retries()[0].last_attempt_at
    store.touch_run("vid")
    assert store.get_pending_retries()[0].last_attempt_at >= first
    store.save_pending_retry("vid", "u", "t", "listen", "{}", "", "a failure")
    before = store.get_pending_retries()[0].last_attempt_at
    store.touch_run("vid")
    assert store.get_pending_retries()[0].last_attempt_at == before


# ---- the first step fails ---------------------------------------------------------------------------------------
def test_a_video_whose_page_cannot_be_read_is_queued_and_the_user_is_told_once(env, monkeypatch):
    def broken(url):
        raise RuntimeError("could not read video info: network down")

    monkeypatch.setattr(pipeline, "video_info", broken)
    res = pipeline.process_video(URL)
    assert not res.ok and "queued for retry" in res.message
    pend = store.get_pending_retries()
    assert [(p.video_id, p.stage, p.title) for p in pend] == [(VID, "listen", VID)]
    assert pipeline.title_is_placeholder(pend[0])
    assert sum("retry queue" in text and "&lt;" not in text for text in env) == 1
    again = pipeline.process_video(URL)                                         # pressed again: already queued, not told again
    assert again.ok and sum("retry queue" in text for text in env) == 1


def test_an_address_that_is_not_a_video_is_not_queued(env, monkeypatch):
    monkeypatch.setattr(pipeline, "video_info", lambda url: (_ for _ in ()).throw(ValueError("no video id")))
    pipeline.process_video("https://www.youtube.com/")
    assert store.get_pending_retries() == [] and env == []


# ---- images: the same image dropped again unchanged ----------------------------------------------------------------
@pytest.fixture
def folders(tmp_path, monkeypatch):
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


def _store_image(folders, content):
    path = folders.enhanced / f"{VID}_pic.png"
    path.write_bytes(content)
    return store.insert_image(VID, str(path)), path


def test_an_identical_image_dropped_again_makes_no_archive_copy(folders):
    iid, path = _store_image(folders, b"same bytes")
    (folders.incoming / "pic.png").write_bytes(b"same bytes")
    done = images.process_pending_images(VID, replacements_only=True)
    assert [d.replaces_image_id for d in done] == [iid]
    assert not (folders.enhanced / "replaced").exists() or list((folders.enhanced / "replaced").iterdir()) == []
    assert path.read_bytes() == b"same bytes" and list(folders.incoming.iterdir()) == []


def test_a_changed_image_dropped_again_still_keeps_the_old_file(folders):
    iid, path = _store_image(folders, b"old bytes")
    (folders.incoming / "pic.png").write_bytes(b"new bytes")
    images.process_pending_images(VID, replacements_only=True)
    kept = list((folders.enhanced / "replaced").iterdir())
    assert len(kept) == 1 and kept[0].read_bytes() == b"old bytes" and path.read_bytes() == b"new bytes"
