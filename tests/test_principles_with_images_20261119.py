"""2026-11-19 (user rule): a principle that has pictures is never sent without them. The messages wait behind the album, a failed
photo send never falls back to text alone, and the album's time limit grows with its size so that the waiting can end."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifecompass import audio as audio_mod
from lifecompass import llm, pipeline, publisher, resurface, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32
PRINCIPLE = "جملة 2-1 كلمة تانية. جملة 2-2 كلمة تانية."


def quiet(*args, **kwargs):
    return None


# ------------------------------------------------------------------ pipeline: the messages wait behind the album
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(audio_mod, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    monkeypatch.setattr(pipeline, "match_principles_to_images", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "caption_standalone_image", lambda *a, **k: "")
    picture = tmp_path / "a.png"
    picture.write_bytes(PNG)
    store.insert_image("vid", str(picture))
    yield
    store._prepared.clear()


def heard(audio, prompt, system, context):
    return "\n".join(f"جملة {idx}-{j} كلمة تانية." for idx in (1, 2, 3) for j in range(5))


def extract_one(prompt, schema, system, context):
    return ExtractionResult(items=[ExtractedPrinciple(text=PRINCIPLE, domain="Self")])


def _run(monkeypatch, album):
    sent = []
    monkeypatch.setattr(pipeline, "send_photo_group", album)
    monkeypatch.setattr(pipeline, "send_message", lambda text, log=print: sent.append(text) or True)
    llm.set_test_backend(audio=heard, json=extract_one)
    res = pipeline.run_stages("vid", "عنوان", "http://u", pipeline._RunLogger("http://u", quiet))
    return res, sent


def test_when_the_album_could_not_be_sent_no_principle_message_goes_out_and_everything_is_queued(env, monkeypatch):
    res, sent = _run(monkeypatch, lambda *a, **k: False)
    assert res.ok and res.added == 1 and store.is_video_processed("vid")             # the principles are stored regardless
    assert sent == []                                                                # but not one of them was sent without its pictures
    [row] = [r for r in store.get_pending_retries() if r.stage == pipeline.DELIVER_STAGE]
    payload = pipeline.parse_delivery_payload(row.items_json)
    assert [Path(x).name for x in payload["photos"]] == ["a.png"]
    assert len(payload["chunks"]) == 1 and PRINCIPLE in payload["chunks"][0]


def test_when_the_album_went_out_the_messages_follow_and_nothing_is_queued(env, monkeypatch):
    res, sent = _run(monkeypatch, lambda *a, **k: True)
    assert len(sent) == 1 and PRINCIPLE in sent[0]
    assert [r for r in store.get_pending_retries() if r.stage == pipeline.DELIVER_STAGE] == []


# ------------------------------------------------------------------ publisher: the album's time limit follows its size
class _Reply:
    status_code = 200

    def json(self):
        return {"ok": True}


def test_the_album_time_limit_grows_with_what_is_sent(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    seen = []
    monkeypatch.setattr(publisher.requests, "post", lambda url, **kw: seen.append(kw["timeout"]) or _Reply())
    small, big = tmp_path / "s.png", tmp_path / "b.png"
    small.write_bytes(PNG)
    big.write_bytes(b"0" * 6_000_000)
    assert publisher.send_photo_group([small], log=quiet) is True
    assert publisher.send_photo_group([big], log=quiet) is True
    assert 30.0 <= seen[0] < 31.0             # a small album keeps (practically) the old 30 s limit
    assert seen[1] > 150                      # 6 MB at 40 KB/s is about 146 s, plus the margin


# ------------------------------------------------------------------ resurface: never a principle as text alone when it has pictures
@pytest.fixture
def resurfacing(tmp_path, monkeypatch):
    sent, shown = [], []
    principle = SimpleNamespace(id=7, principle="مبدأ", times_shown=0, last_shown_at=None)
    monkeypatch.setattr(resurface, "load_dotenv", lambda *a, **k: None)
    real_get = resurface.tunables.get
    monkeypatch.setattr(resurface.tunables, "get", lambda s, k: 100000 if (s, k) == ("resurface", "daily_count") else real_get(s, k))   # the coin always lands on "send"
    monkeypatch.setattr(resurface, "LOG_PATH", tmp_path / "resurface.log")
    monkeypatch.setattr(resurface, "_in_quiet_hours", lambda now=None: False)
    monkeypatch.setattr(resurface.random, "random", lambda: 0.999)                     # never the standalone-image branch
    monkeypatch.setattr(resurface.store, "count_all", lambda: 1)
    monkeypatch.setattr(resurface.store, "pick_for_resurfacing", lambda n: [principle])
    monkeypatch.setattr(resurface.store, "mark_shown", lambda ids: shown.extend(ids))
    monkeypatch.setattr(resurface, "send_message", lambda text, log=print: sent.append(text) or True)
    return SimpleNamespace(sent=sent, shown=shown)


def _pictures(monkeypatch, *paths):
    monkeypatch.setattr(resurface.store, "get_images_for_principle", lambda pid: [SimpleNamespace(enhanced_path=str(p)) for p in paths])


def test_a_principle_with_pictures_is_not_sent_as_text_when_the_pictures_could_not_be_sent(resurfacing, tmp_path, monkeypatch):
    picture = tmp_path / "p.png"
    picture.write_bytes(PNG)
    _pictures(monkeypatch, picture)
    monkeypatch.setattr(resurface, "send_photo_group", lambda *a, **k: False)
    resurface.run_random_resurface()
    assert resurfacing.sent == [] and resurfacing.shown == []                        # nothing sent, not marked: it can be picked again


def test_a_principle_with_pictures_goes_out_with_them_and_is_marked_shown(resurfacing, tmp_path, monkeypatch):
    picture = tmp_path / "p.png"
    picture.write_bytes(PNG)
    _pictures(monkeypatch, picture)
    monkeypatch.setattr(resurface, "send_photo_group", lambda *a, **k: True)
    resurface.run_random_resurface()
    assert resurfacing.sent == [] and resurfacing.shown == [7]                       # the photo carries the text as its caption


def test_a_picture_whose_file_is_gone_does_not_hold_the_principle_back(resurfacing, tmp_path, monkeypatch):
    _pictures(monkeypatch, tmp_path / "gone.png")
    monkeypatch.setattr(resurface, "send_photo_group", lambda *a, **k: pytest.fail("there is no picture file to send"))
    resurface.run_random_resurface()
    assert resurfacing.sent == ["<b>مبدأ</b>"] and resurfacing.shown == [7]


def test_a_principle_with_no_pictures_is_sent_as_text(resurfacing, monkeypatch):
    _pictures(monkeypatch)
    resurface.run_random_resurface()
    assert resurfacing.sent == ["<b>مبدأ</b>"] and resurfacing.shown == [7]