"""2026-11-19 (Fully Automated, Fully Dynamic): a Telegram outage is bridged by longer waits, what Telegram could not take is queued and
sent later (and never silently lost), the scheduled tasks follow the project and Python wherever they are, and no bytecode is written."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import requests

from lifecompass import audio as audio_mod
from lifecompass import health, llm, logging_setup, pipeline, publisher, store
from lifecompass.principle_extractor import ExtractedPrinciple, ExtractionResult

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ publisher: a network that is down is waited out
class Reply:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


OK = Reply(200, {"ok": True, "result": {"message_id": 1}})


@pytest.fixture
def telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:abcdefghijklmnopqrstuvwxyz0123456789")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    sleeps = []
    monkeypatch.setattr(publisher.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def _post_that(monkeypatch, *outcomes):
    """requests.post answering with the given outcomes in turn (an exception is raised, a Reply is returned); the last one repeats."""
    calls = []

    def post(url, **kw):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(1)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(publisher.requests, "post", post)
    return calls


def test_a_network_that_is_down_is_waited_out_with_growing_pauses(monkeypatch, telegram):
    down = requests.exceptions.ConnectionError("getaddrinfo failed")
    calls = _post_that(monkeypatch, down, down, down, OK)
    assert publisher.send_message("hi", log=lambda *_: None) is True
    assert len(calls) == 4 and telegram == [5.0, 15.0, 45.0]


def test_a_network_that_stays_down_gives_up_after_the_last_wait(monkeypatch, telegram):
    calls = _post_that(monkeypatch, requests.exceptions.ConnectionError("WinError 10013"))
    assert publisher.send_message("hi", log=lambda *_: None) is False
    assert len(calls) == 5 and telegram == [5.0, 15.0, 45.0, 135.0]


def test_a_read_timeout_is_still_never_retried(monkeypatch, telegram):
    calls = _post_that(monkeypatch, requests.exceptions.ReadTimeout("slow"))
    assert publisher.send_message("hi", log=lambda *_: None) is False
    assert len(calls) == 1 and telegram == []                  # the request may have been delivered: no duplicate risk


def test_server_side_errors_keep_the_short_retries(monkeypatch, telegram):
    calls = _post_that(monkeypatch, Reply(502, {"ok": False, "description": "bad gateway"}))
    assert publisher.send_message("hi", log=lambda *_: None) is False
    assert len(calls) == 3 and telegram == [3.0, 3.0]


def test_a_client_error_is_not_retried_at_all(monkeypatch, telegram):
    calls = _post_that(monkeypatch, Reply(400, {"ok": False, "description": "chat not found"}))
    assert publisher.send_message("hi", log=lambda *_: None) is False
    assert len(calls) == 1 and telegram == []


def test_the_photo_album_waits_out_a_network_that_is_down_too(monkeypatch, telegram, tmp_path):
    photo = tmp_path / "a.png"
    photo.write_bytes(b"png")
    down = requests.exceptions.ConnectionError("getaddrinfo failed")
    calls = _post_that(monkeypatch, down, OK)
    assert publisher.send_photo_group([photo], caption="t", log=lambda *_: None) is True
    assert len(calls) == 2 and telegram == [5.0]
    calls = _post_that(monkeypatch, down)
    telegram.clear()
    assert publisher.send_photo_group([photo], caption="t", log=lambda *_: None) is False
    assert len(calls) == 5 and telegram == [5.0, 15.0, 45.0, 135.0]


# ------------------------------------------------------------------ pipeline: what Telegram could not take is queued, not lost
@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    monkeypatch.setattr(pipeline, "send_photo_group", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(audio_mod, "download_audio", lambda vid: tmp_path / "full.mp3")
    monkeypatch.setattr(audio_mod, "audio_duration", lambda p: 60.0)
    monkeypatch.setattr(audio_mod, "recording_bytes", lambda p: b"audio")
    yield
    store._prepared.clear()


def heard(audio, prompt, system, context):
    return "\n".join(f"جملة {idx}-{j} كلمة تانية." for idx in (1, 2, 3) for j in range(5))


PRINCIPLE = "جملة 2-1 كلمة تانية. جملة 2-2 كلمة تانية."


def extract_one(prompt, schema, system, context):
    return ExtractionResult(items=[ExtractedPrinciple(text=PRINCIPLE, domain="Self")])


def _run(monkeypatch, send):
    monkeypatch.setattr(pipeline, "send_message", send)
    llm.set_test_backend(audio=heard, json=extract_one)
    return pipeline.run_stages("vid", "عنوان", "http://u", pipeline._RunLogger("http://u", lambda line: None))


def test_a_telegram_that_could_not_be_reached_queues_the_messages_and_loses_nothing(env, monkeypatch):
    res = _run(monkeypatch, lambda text, log=print: False)
    assert res.ok and res.added == 1 and store.is_video_processed("vid")             # the principles are stored regardless
    rows = store.get_pending_retries()
    assert [(r.video_id, r.stage) for r in rows] == [("vid" + pipeline.DELIVER_KEY_SUFFIX, pipeline.DELIVER_STAGE)]
    payload = pipeline.parse_delivery_payload(rows[0].items_json)
    assert payload["photos"] == [] and len(payload["chunks"]) == 1 and PRINCIPLE in payload["chunks"][0]


def test_a_telegram_that_answers_queues_nothing(env, monkeypatch):
    sent = []
    _run(monkeypatch, lambda text, log=print: sent.append(text) or True)
    assert sent and store.get_pending_retries() == []


def test_a_delivery_row_does_not_collide_with_the_supplementary_row_of_the_same_video(env):
    store.save_pending_retry("vid", "u", "t", pipeline.SUPPLEMENTARY_STAGE, "{}", "", "models down")
    pipeline.queue_delivery("vid", "u", "t", {"photos": [], "chunks": ["x"]}, "m", lambda line: None)
    assert sorted(r.stage for r in store.get_pending_retries()) == [pipeline.DELIVER_STAGE, pipeline.SUPPLEMENTARY_STAGE]


def test_queueing_again_replaces_a_stale_delivery_row(env):
    pipeline.queue_delivery("vid", "u", "t", {"photos": [], "chunks": ["old"]}, "m", lambda line: None)
    pipeline.queue_delivery("vid", "u", "t", {"photos": [], "chunks": ["new"]}, "m", lambda line: None)
    rows = store.get_pending_retries()
    assert len(rows) == 1 and rows[0].attempts == 0 and pipeline.parse_delivery_payload(rows[0].items_json)["chunks"] == ["new"]


# ------------------------------------------------------------------ retry_pending sends what was queued, in order
@pytest.fixture
def retry(env, monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)               # the real .env (keys) must never load in a test
    spec = importlib.util.spec_from_file_location("retry_pending_under_test", ROOT / "scripts" / "retry_pending.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_stages", lambda *a, **k: pytest.fail("a delivery row must not restart the pipeline"))
    return module


def test_the_queued_messages_are_sent_in_order_and_a_failure_keeps_the_rest_in_place(retry, monkeypatch):
    pipeline.queue_delivery("vid", "u", "عنوان", {"photos": [], "chunks": ["a", "b", "c"]}, "m", lambda line: None)
    sent, up = [], {"b": False}

    def phone(text, log=print):
        if up.get(text, True) is False:
            return False
        sent.append(text)
        return True
    monkeypatch.setattr(retry, "send_message", phone)
    retry.main()
    assert sent == ["a"]                                                              # b failed: c waits behind it
    row = store.get_pending_retries()[0]
    assert pipeline.parse_delivery_payload(row.items_json)["chunks"] == ["b", "c"] and row.attempts == 1
    up["b"] = True
    retry.main()
    assert sent == ["a", "b", "c"] and store.get_pending_retries() == []


def test_the_album_goes_first_and_only_once(retry, monkeypatch, tmp_path):
    photo = tmp_path / "p.png"
    photo.write_bytes(b"png")
    pipeline.queue_delivery("vid", "u", "عنوان", {"photos": [str(photo)], "chunks": ["a"]}, "m", lambda line: None)
    order = []
    monkeypatch.setattr(retry, "send_photo_group", lambda paths, caption="", log=print: order.append("album") or True)

    def failing(text, log=print):
        order.append(text)
        return False
    monkeypatch.setattr(retry, "send_message", failing)
    retry.main()                                                                       # the message fails: the album is not repeated
    assert order == ["album", "a"]
    assert pipeline.parse_delivery_payload(store.get_pending_retries()[0].items_json) == {"photos": [], "chunks": ["a"]}


def test_after_a_day_of_waiting_it_gives_up_quietly(retry, monkeypatch):
    pipeline.queue_delivery("vid", "u", "عنوان", {"photos": [], "chunks": ["a"]}, "m", lambda line: None)
    with store._connect() as conn:
        conn.execute("UPDATE pending_retries SET attempts = ?", (retry.MAX_DELIVERY_ATTEMPTS,))
    monkeypatch.setattr(retry, "send_message", lambda *a, **k: pytest.fail("no Telegram message about a failed delivery"))
    retry.main()
    assert store.get_pending_retries() == []

# ------------------------------------------------------------------ health: the failure count (log only) and task paths
def test_the_failure_count_covers_the_last_day_only_and_only_gave_up_sends(tmp_path, monkeypatch):
    now = time.mktime((2026, 11, 19, 12, 0, 0, 0, 0, -1))

    def line(hours_ago, level, text):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now - hours_ago * 3600)) + f" | {level:<8} | lifecompass.publisher | {text}"
    log = tmp_path / "master.log"
    log.write_text("\n".join([
        line(2, "ERROR", "Telegram send failed after 5 attempts: x"),
        line(3, "ERROR", "sendMediaGroup chunk 0 failed after retries: x"),
        line(30, "ERROR", "Telegram send failed after 5 attempts: too old"),
        line(1, "WARNING", "attempt 1 raised network exception (safe to retry): x"),
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(logging_setup, "MASTER_LOG_PATH", log)
    assert health.telegram_failures_last_day(now) == 2
    monkeypatch.setattr(logging_setup, "MASTER_LOG_PATH", tmp_path / "missing.log")
    assert health.telegram_failures_last_day(now) == 0


TASK_XML = r'''<Task><Actions Context="Author"><Exec><Command>C:\Old\Python313\pythonw.exe</Command><Arguments>"D:\Elsewhere\life-compass-pipeline\scripts\retry_pending.py"</Arguments><WorkingDirectory>D:\Elsewhere\life-compass-pipeline</WorkingDirectory></Exec></Actions></Task>'''


def test_a_saved_task_is_registered_with_the_paths_of_today():
    out = health.localize_task_xml(TASK_XML, Path(r"C:\Project\life-compass-pipeline"), Path(r"C:\Python314\pythonw.exe"))
    assert r"<Command>C:\Python314\pythonw.exe</Command>" in out
    assert r'<Arguments>"C:\Project\life-compass-pipeline\scripts\retry_pending.py"</Arguments>' in out
    assert r"<WorkingDirectory>C:\Project\life-compass-pipeline</WorkingDirectory>" in out
    assert "D:" not in out and "Python313" not in out


def test_without_a_pythonw_the_program_is_left_as_saved():
    out = health.localize_task_xml(TASK_XML, Path(r"C:\Project\life-compass-pipeline"), None)
    assert r"C:\Old\Python313\pythonw.exe" in out and r"C:\Project\life-compass-pipeline\scripts\retry_pending.py" in out


def test_every_saved_task_copy_keeps_its_shape_when_localized():
    for name, xml in health.task_files().items():
        raw = xml.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
        out = health.localize_task_xml(text, ROOT, health._current_pythonw())
        assert out.count("<Exec>") == text.count("<Exec>"), name
        assert f"<WorkingDirectory>{ROOT}</WorkingDirectory>" in out, name
        assert f'"{ROOT}\\scripts\\' in out, name


class Done:
    def __init__(self, out, code=0):
        self.stdout, self.returncode = out, code


def test_a_task_that_points_at_nothing_is_recognised_and_one_that_cannot_be_read_is_left_alone(tmp_path, monkeypatch):
    real = tmp_path / "pythonw.exe"
    script = tmp_path / "x.py"
    real.write_bytes(b"")
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(health, "_powershell", lambda cmd, timeout=60: Done(f'{real}\n"{script}"\n'))
    assert health._task_action_is_broken("T") is False
    monkeypatch.setattr(health, "_powershell", lambda cmd, timeout=60: Done(f'{real}\n"{tmp_path / "gone.py"}"\n'))
    assert health._task_action_is_broken("T") is True
    monkeypatch.setattr(health, "_powershell", lambda cmd, timeout=60: Done("C:\\nowhere\\pythonw.exe\n"))
    assert health._task_action_is_broken("T") is True
    monkeypatch.setattr(health, "_powershell", lambda cmd, timeout=60: Done("", 1))
    assert health._task_action_is_broken("T") is False                                  # unreadable is not "proven broken"


def test_ensure_tasks_registers_a_broken_task_again_and_leaves_a_healthy_one_alone(tmp_path, monkeypatch):
    xml = tmp_path / "t.xml"
    xml.write_text(TASK_XML, encoding="utf-8")
    monkeypatch.setattr(health, "task_files", lambda: {"T": xml})
    monkeypatch.setattr(health, "_task_state", lambda name: "Ready")
    registered = []
    monkeypatch.setattr(health, "_register_task", lambda name, path: registered.append(name))
    monkeypatch.setattr(health, "_task_action_is_broken", lambda name: False)
    assert health.ensure_tasks() == [] and registered == []
    monkeypatch.setattr(health, "_task_action_is_broken", lambda name: True)
    assert health.ensure_tasks() == [] and registered == ["T"]


def test_registering_a_task_hands_powershell_the_localized_xml_and_cleans_up(tmp_path, monkeypatch):
    xml = tmp_path / "t.xml"
    xml.write_text(TASK_XML, encoding="utf-8")
    monkeypatch.setattr(health, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(health, "_current_pythonw", lambda: None)
    seen = {}

    def fake(cmd, timeout=60):
        scratch = tmp_path / "data" / "_task_register.xml"
        seen["cmd"], seen["xml"] = cmd, scratch.read_text(encoding="utf-8")
        return Done("")
    monkeypatch.setattr(health, "_powershell", fake)
    health._register_task("T", xml)
    assert "Register-ScheduledTask" in seen["cmd"] and "-Force" in seen["cmd"]
    assert f"<WorkingDirectory>{tmp_path}</WorkingDirectory>" in seen["xml"]
    assert not (tmp_path / "data" / "_task_register.xml").exists()


# ------------------------------------------------------------------ no bytecode
def test_no_entry_point_writes_bytecode():
    import lifecompass
    assert sys.dont_write_bytecode is True
    assert "sys.dont_write_bytecode = True" in Path(lifecompass.__file__).read_text(encoding="utf-8")