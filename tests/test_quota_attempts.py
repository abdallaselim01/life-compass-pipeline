"""Quota flags, the real-call counter, and retry attempts that must not be wasted."""
import json
import time
import types

import pytest

from lifecompass import llm, store


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setattr(llm, "_GEMINI_STATE_PATH", path)
    return path


def test_google_reset_delay_is_read_from_the_error():
    err = "429 RESOURCE_EXHAUSTED {'details': [{'@type': 'x', 'retryDelay': '64091s'}]}"
    assert llm._retry_delay_seconds(err) == 64091.0
    assert llm._retry_delay_seconds("Please retry in 22.13s.") == 22.13
    assert llm._retry_delay_seconds("503 UNAVAILABLE") is None


def test_a_flag_is_short_even_when_google_announces_a_very_long_delay(state_file):
    llm._mark_gemini_exhausted_on_disk("m1", 64091.0)
    until = json.loads(state_file.read_text())["until"]["m1"]
    assert until - time.time() <= llm._EXHAUSTION_RECHECK_SECONDS + 5
    assert llm._load_gemini_exhausted_today() == {"m1"}


def test_a_flag_expires_and_an_old_format_file_is_ignored(state_file):
    state_file.write_text(json.dumps({"until": {"m1": time.time() - 5, "m2": time.time() + 600}}))
    assert llm._load_gemini_exhausted_today() == {"m2"}
    state_file.write_text(json.dumps({"date_utc": "2026-10-15", "models": ["m1", "m2"]}))
    assert llm._load_gemini_exhausted_today() == set()


def test_only_real_requests_are_counted(monkeypatch):
    monkeypatch.setattr(llm, "_call_with_timeout", lambda build_call, model, timeout: build_call(model))
    monkeypatch.setattr(llm, "_exhausted_models_this_run", {"a", "b"})
    before = llm.api_calls_made()
    with pytest.raises(RuntimeError):
        llm._generate_with_fallback(["a", "b"], lambda m: "x", log=lambda *_: None, context="t")
    assert llm.api_calls_made() == before                      # every model was flagged: nothing was sent
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    assert llm._generate_with_fallback(["a", "b"], lambda m: "x", log=lambda *_: None, context="t") == ("x", "a")
    assert llm.api_calls_made() == before + 1


def test_an_attempt_that_called_nothing_is_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.db")
    store._prepared.clear()
    store.save_pending_retry("v", "u", "t", "supplementary", "{}", "", "first")
    attempts = lambda: store.get_pending_retries()[0].attempts
    assert attempts() == 0
    store.save_pending_retry("v", "u", "t", "supplementary", "{}", "", "again", count_attempt=False)
    assert attempts() == 0
    store.save_pending_retry("v", "u", "t", "supplementary", "{}", "", "again", count_attempt=True)
    assert attempts() == 1
    store._prepared.clear()
