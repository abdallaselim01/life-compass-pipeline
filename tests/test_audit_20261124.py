"""2026-11-24 audit changes (all approved by the user):
A. a principle (or standalone image text) longer than Telegram's caption limit goes as the album first and the text right after it;
   a standalone image whose file is gone is passed over;
B. the wait limit of listen and extract grows with the length of the recording (and the ceiling and the first-piece limit with it);
C. the numbers that were hard-coded now come from config.toml with the same defaults;
D. the dead code that was removed stays removed."""
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifecompass import liveness, llm, pipeline, publisher, resurface, tunables

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


# ------------------------------------------------------------------ A: long text with pictures
@pytest.fixture
def resurfacing(tmp_path, monkeypatch):
    sent, shown, albums = [], [], []
    real_get = resurface.tunables.get
    monkeypatch.setattr(resurface.tunables, "get", lambda s, k: 100000 if (s, k) == ("resurface", "daily_count") else real_get(s, k))
    monkeypatch.setattr(resurface, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(resurface, "LOG_PATH", tmp_path / "resurface.log")
    monkeypatch.setattr(resurface, "_in_quiet_hours", lambda now=None: False)
    monkeypatch.setattr(resurface.random, "random", lambda: 0.999)                    # never the standalone-image branch by default
    monkeypatch.setattr(resurface.store, "count_all", lambda: 1)
    monkeypatch.setattr(resurface.store, "mark_shown", lambda ids: shown.extend(ids))
    monkeypatch.setattr(resurface, "send_message", lambda text, log=print: sent.append(text) or True)
    monkeypatch.setattr(resurface, "send_photo_group", lambda paths, caption="", log=print: albums.append(caption) or True)
    picture = tmp_path / "a.png"
    picture.write_bytes(PNG)
    monkeypatch.setattr(resurface.store, "get_images_for_principle", lambda pid: [SimpleNamespace(enhanced_path=str(picture))])
    return SimpleNamespace(sent=sent, shown=shown, albums=albums, picture=picture, monkeypatch=monkeypatch)


def _principle(text):
    return SimpleNamespace(id=7, principle=text, times_shown=0, last_shown_at=None)


def test_a_principle_over_the_caption_limit_goes_as_the_album_then_its_text(resurfacing):
    text = "كلمة " * 400                                                                # 2000 characters, over the limit
    resurfacing.monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [_principle(text)])
    resurface.run_random_resurface()
    assert resurfacing.albums == [""]                                                    # the album carries no caption
    assert len(resurfacing.sent) == 1 and text.strip() in resurfacing.sent[0]          # the whole text follows as a message
    assert resurfacing.shown == [7]


def test_a_principle_within_the_caption_limit_stays_one_message_with_its_caption(resurfacing):
    resurfacing.monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [_principle("مبدأ قصير.")])
    resurface.run_random_resurface()
    assert len(resurfacing.albums) == 1 and "مبدأ قصير." in resurfacing.albums[0]
    assert resurfacing.sent == [] and resurfacing.shown == [7]


def test_the_limit_is_telegrams_own_and_the_boundary_is_inclusive(resurfacing):
    assert publisher.TELEGRAM_CAPTION_LIMIT == 1024
    resurfacing.monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [_principle("ا" * 1024)])
    resurface.run_random_resurface()
    assert len(resurfacing.albums) == 1 and resurfacing.sent == []                     # exactly the limit still fits as a caption


def test_a_long_principle_whose_album_fails_sends_no_text_and_is_not_marked_shown(resurfacing):
    resurfacing.monkeypatch.setattr(resurface, "send_photo_group", lambda *a, **k: False)
    resurfacing.monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [_principle("كلمة " * 400)])
    resurface.run_random_resurface()
    assert resurfacing.sent == [] and resurfacing.shown == []


def test_a_standalone_image_whose_file_is_gone_is_passed_over(resurfacing, tmp_path):
    gone = SimpleNamespace(id=1, enhanced_path=str(tmp_path / "gone.png"), standalone_caption="نص", times_shown=0)
    resurfacing.monkeypatch.setattr(resurface.random, "random", lambda: 0.0)            # the standalone-image branch
    resurfacing.monkeypatch.setattr(resurface.store, "pick_standalone_images_for_resurfacing", lambda n: [gone])
    resurfacing.monkeypatch.setattr(resurface.store, "mark_image_shown", lambda i: pytest.fail("nothing was sent"))
    resurfacing.monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [_principle("مبدأ قصير.")])
    resurface.run_random_resurface()
    assert resurfacing.shown == [7]                                                     # it fell through to a principle


def test_a_standalone_image_send_that_raises_is_not_fatal_and_not_marked(resurfacing, tmp_path):
    img = SimpleNamespace(id=1, enhanced_path=str(resurfacing.picture), standalone_caption="نص", times_shown=0)
    resurfacing.monkeypatch.setattr(resurface.random, "random", lambda: 0.0)
    resurfacing.monkeypatch.setattr(resurface.store, "pick_standalone_images_for_resurfacing", lambda n: [img])
    resurfacing.monkeypatch.setattr(resurface.store, "mark_image_shown", lambda i: pytest.fail("not sent"))

    def boom(*a, **k):
        raise OSError("disk")
    resurfacing.monkeypatch.setattr(resurface, "send_photo_group", boom)
    resurface.run_random_resurface()                                                    # no exception escapes


# ------------------------------------------------------------------ B: the wait grows with the recording
def test_the_wait_of_listen_and_extract_grows_with_the_recording_and_nothing_else_does(monkeypatch):
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 20.0)
    llm.set_audio_minutes(120)
    try:
        assert llm.scaled_long_wait("listen") == 2400.0
        assert llm.scaled_long_wait("extract[1]") == 2400.0
        assert llm.scaled_long_wait("match") == 0.0 and llm.scaled_long_wait("caption") == 0.0
        llm.set_audio_minutes(None)
        assert llm.scaled_long_wait("listen") == 0.0                                    # unknown length: as before
    finally:
        llm.set_audio_minutes(None)
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 0.0)                # 0 = off
    llm.set_audio_minutes(120)
    try:
        assert llm.scaled_long_wait("listen") == 0.0
    finally:
        llm.set_audio_minutes(None)


def _limit_of(monkeypatch, context):
    seen = []
    monkeypatch.setattr(llm, "_call_with_timeout", lambda build_call, model, timeout: seen.append(timeout) or build_call(model))
    llm._generate_with_fallback(["m"], lambda model: "ok", context=context)
    return seen[0]


def test_a_two_hour_recording_gets_a_wait_above_the_old_ceiling_and_a_short_one_keeps_its_wait(monkeypatch):
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 20.0)
    try:
        llm.set_audio_minutes(None)
        assert _limit_of(monkeypatch, "listen") == 900.0                                # nothing recorded: the old ceiling
        llm.set_audio_minutes(120)
        assert _limit_of(monkeypatch, "listen") == 2400.0                               # the ceiling rose with the recording
        assert _limit_of(monkeypatch, "match") == 900.0                                 # another kind of call: unchanged
        llm.set_audio_minutes(10)
        assert _limit_of(monkeypatch, "extract[1]") == 900.0                            # a short recording never gets less than before
    finally:
        llm.set_audio_minutes(None)


def test_a_learned_limit_is_never_lowered_by_the_scaling(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 20.0)
    llm.record_success_time("listen", 387.3)                                            # 2 x 387.3 = 774.6, above 34 x 20 = 680
    try:
        llm.set_audio_minutes(34)
        assert _limit_of(monkeypatch, "listen") == pytest.approx(774.6)
        llm.set_audio_minutes(120)
        assert _limit_of(monkeypatch, "listen") == 2400.0                               # the recording's length wins when it is larger
    finally:
        llm.set_audio_minutes(None)


def _chunk(text="x"):
    return types.SimpleNamespace(text=text, candidates=[], usage_metadata=None, prompt_feedback=None)


def _slow_first_piece(monkeypatch):
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setattr(llm, "_MODEL_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(llm, "_STREAM_FIRST_CHUNK_SECONDS", 1.0)

    def stream(model, **kw):
        def gen():
            if model == "a":
                time.sleep(2.2)                                                         # the first piece takes longer than 1 s
            yield _chunk("from " + model)
        return gen()

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content_stream=stream))
    monkeypatch.setattr(llm, "_client", lambda backup=False: client)


def test_without_a_raised_wait_the_first_piece_limit_stays_as_it_was(monkeypatch):
    _slow_first_piece(monkeypatch)
    llm.set_audio_minutes(None)
    text, model = llm.generate_audio_text(b"audio", "p", context="listen[1/1]")
    assert (text, model) == ("from b", "b")                                             # a was cut at 1 s, as before


def test_a_wait_raised_for_a_long_recording_lifts_the_first_piece_limit_too(monkeypatch):
    _slow_first_piece(monkeypatch)
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 1.0)
    llm.set_audio_minutes(10)                                                           # scaled wait 10 s: above the 3 s ceiling
    try:
        text, model = llm.generate_audio_text(b"audio", "p", context="listen[1/1]")
    finally:
        llm.set_audio_minutes(None)
    assert (text, model) == ("from a", "a")                                             # the slow first piece was waited for


def test_the_pipeline_tells_llm_the_length_of_a_recording_that_is_on_disk(tmp_path, monkeypatch):
    mp3 = tmp_path / "vid" / "full.mp3"
    mp3.parent.mkdir()
    mp3.write_bytes(b"x")
    monkeypatch.setattr(pipeline._audio, "AUDIO_ROOT", tmp_path)
    monkeypatch.setattr(pipeline._audio, "audio_duration", lambda p: 7200.0)
    monkeypatch.setattr(llm, "_LONG_WAIT_SECONDS_PER_AUDIO_MINUTE", 20.0)
    pipeline._note_audio_length("vid")
    try:
        assert llm.scaled_long_wait("extract[1]") == 2400.0
        pipeline._note_audio_length("no-such-video")                                   # no file: the length is unknown again
        assert llm.scaled_long_wait("extract[1]") == 0.0
    finally:
        llm.set_audio_minutes(None)


def test_a_failure_reading_the_length_never_stops_a_run(tmp_path, monkeypatch):
    (tmp_path / "vid").mkdir()
    (tmp_path / "vid" / "full.mp3").write_bytes(b"x")
    monkeypatch.setattr(pipeline._audio, "AUDIO_ROOT", tmp_path)

    def broken(path):
        raise RuntimeError("no ffmpeg")
    monkeypatch.setattr(pipeline._audio, "audio_duration", broken)
    pipeline._note_audio_length("vid")
    assert llm.scaled_long_wait("listen") == 0.0


# ------------------------------------------------------------------ C: the numbers come from config.toml, with the same defaults
def test_the_numbers_that_were_hard_coded_keep_their_values_as_defaults():
    assert tunables.get("retry", "max_attempts") == 36
    assert tunables.get("retry", "max_delivery_attempts") == 288
    assert tunables.get("liveness", "hung_after") == 3
    assert tunables.get("liveness", "launch_grace_seconds") == 60
    assert tunables.get("maintenance", "heartbeat_seconds") == 300
    assert tunables.get("maintenance", "orphan_max_hours") == 6.0
    assert tunables.get("telegram", "network_retry_max_seconds") == 300.0
    assert tunables.get("resurface", "daily_count") == 5
    assert tunables.get("waits", "long_wait_seconds_per_audio_minute") == 20.0


def test_the_modules_use_those_numbers():
    assert (pipeline.HEARTBEAT_SECONDS, pipeline.ORPHAN_MAX_HOURS) == (300, 6.0)
    assert (liveness.HUNG_AFTER, liveness.LAUNCH_GRACE_SECONDS) == (3, 60)
    assert publisher._NETWORK_RETRY_MAX_SECONDS == 300.0


def test_the_retry_task_reads_its_caps_from_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("retry_pending_cfg", Path(__file__).resolve().parents[1] / "scripts" / "retry_pending.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.MAX_ATTEMPTS, mod.MAX_DELIVERY_ATTEMPTS) == (36, 288)


def test_the_daily_count_is_not_read_from_the_environment_any_more(monkeypatch):
    monkeypatch.setenv("DAILY_RESURFACE_COUNT", "99")
    monkeypatch.setattr(resurface, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(resurface, "_in_quiet_hours", lambda now=None: False)
    seen = []
    monkeypatch.setattr(resurface.store, "count_all", lambda: seen.append("counted") or 0)
    monkeypatch.setattr(resurface, "_log", lambda line: None)
    resurface.run_random_resurface()                                                    # runs with daily_count = 5 from tunables
    assert seen == ["counted"]
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "DAILY_RESURFACE_COUNT=" not in example


# ------------------------------------------------------------------ D: dead code stays removed
def test_the_dead_code_is_gone():
    assert not hasattr(llm, "_EXTRACTION_MODEL_TIMEOUT_SECONDS")
    assert not hasattr(llm, "_schema_text_hint")
    import inspect
    assert "schema_hint" not in inspect.signature(llm.generate_json).parameters
    from lifecompass import logging_setup
    assert "openrouter" not in " ".join(p[1] for p in logging_setup._SECRET_PATTERNS).lower()
    assert logging_setup.redact("sk-or-" + "a" * 30) == "sk-or-" + "a" * 30            # no longer a secret shape of this system


def test_the_retry_task_file_has_no_bom_and_no_escaped_text():
    raw = (Path(__file__).resolve().parents[1] / "scripts" / "retry_pending.py").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\\u0" not in raw


def test_the_launcher_moves_a_big_listener_log_and_never_deletes_it():
    text = (Path(__file__).resolve().parents[1] / "start_listener_silent.vbs").read_text(encoding="utf-8")
    assert "MoveFile" in text and "logs\\archive" in text and "DeleteFile" not in text and ".Delete" not in text
    assert "On Error Resume Next" in text and "On Error GoTo 0" in text                 # a move that fails never shows an error box
