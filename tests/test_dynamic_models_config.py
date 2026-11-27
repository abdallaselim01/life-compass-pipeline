"""Fully Dynamic (design decision 2026-11-17): the model chain is discovered and ordered by what works, and every tunable number lives in
config.toml with a safe default."""
import json
import time

import pytest

from lifecompass import llm, modelchain, tunables


@pytest.fixture
def mc(tmp_path, monkeypatch):
    """modelchain on scratch files, with a fake Google listing that counts its calls."""
    monkeypatch.setattr(modelchain, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(modelchain, "_HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(modelchain, "_refresh_failed_at", 0.0)
    # these tests are about the mechanics of the chain, written for the Flash family (the family is a setting: see the Hydrogen tests)
    monkeypatch.setattr(tunables, "get_text", lambda section, key: {"family": "flash", "last_resort": "gemini-flash-latest"}[key])
    calls = {"n": 0, "names": ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-2.5-flash",
                               "gemini-3.10-flash", "gemini-3.8-flash-tts", "gemini-3.5-flash-lite", "gemini-flash-latest",
                               "gemini-3-flash-preview", "gemini-2.5-flash-image", "gemini-3.8-live", "gemini-omni-1.2-flash",
                               "gemini-4-argon", "gemini-3.1-pro-preview"], "fail": False}

    def fake_list():
        calls["n"] += 1
        if calls["fail"]:
            raise RuntimeError("no network")
        return list(calls["names"])
    monkeypatch.setattr(modelchain, "_list_remote", fake_list)
    return calls


def test_only_stable_flash_models_are_candidates_and_the_newest_comes_first(mc):
    assert modelchain.candidates() == ["gemini-3.10-flash", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                                       "gemini-3.5-flash", "gemini-2.5-flash"]            # 3.10 is newer than 3.8, not older


def test_the_chain_is_the_newest_few_and_follows_a_newer_model_without_any_change(mc):
    assert modelchain.chain() == ["gemini-3.10-flash", "gemini-3.8-flash", "gemini-3.7-flash"]
    mc["names"].append("gemini-4.1-flash")
    modelchain._CACHE_PATH.unlink()
    assert modelchain.chain()[0] == "gemini-4.1-flash"


def test_google_is_asked_once_per_refresh_period_and_a_stale_copy_survives_an_outage(mc, monkeypatch):
    modelchain.chain()
    modelchain.chain()
    assert mc["n"] == 1
    cache = json.loads(modelchain._CACHE_PATH.read_text(encoding="utf-8"))
    cache["fetched_at"] = time.time() - 10 * 3600                                          # older than refresh_hours
    modelchain._CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    mc["fail"] = True
    assert modelchain.chain()[0] == "gemini-3.10-flash"                                  # the stale copy is used
    assert mc["n"] == 2


def test_with_no_listing_and_no_copy_the_chain_is_googles_own_alias(mc):
    mc["fail"] = True
    assert modelchain.chain() == ["gemini-flash-latest"]


def test_after_a_failed_refresh_google_is_not_hammered(mc):
    mc["fail"] = True
    modelchain.chain()
    modelchain.chain()
    modelchain.chain()
    assert mc["n"] == 1                                                                     # asked again only after a few minutes


def test_the_client_that_lists_the_models_stays_open_while_the_list_is_read(monkeypatch):
    """Regression (2026-11-17): a client nobody holds is closed at once, and the list is lazy."""
    import types
    import weakref
    from google import genai

    class FakeClient:
        def __init__(self, api_key):
            self.closed = False
            ref = weakref.ref(self)

            def gen():
                client = ref()
                if client is None or client.closed:
                    raise RuntimeError("Cannot send a request, as the client has been closed.")
                yield types.SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"])
                yield types.SimpleNamespace(name="models/gemini-embedding-001", supported_actions=["embedContent"])
            self.models = types.SimpleNamespace(list=gen)

        def __del__(self):
            self.closed = True

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("GEMINI_API_KEY_BACKUP", raising=False)
    assert modelchain._ask_google() == ["gemini-3.8-flash"]


def test_the_order_of_the_chain_never_changes_whatever_happens_to_a_model(mc):
    order = modelchain.chain()
    for _ in range(9):
        modelchain.record(order[0], "fail")
    assert modelchain.chain() == order                                                       # failures never move a model
    modelchain._HEALTH_PATH.write_text(json.dumps({"models": {order[0]: {"failures": 9, "cooldown_until": time.time() + 99999}}}),
                                       encoding="utf-8")
    assert modelchain.chain() == order                                                       # an old cool-down in the file is ignored
    modelchain.record(order[0], "ok")
    assert modelchain.chain() == order


def test_a_failure_is_only_recorded_and_a_success_clears_the_count(mc):
    m = "gemini-3.10-flash"
    for n in (1, 2, 3):
        modelchain.record(m, "fail")
        entry = json.loads(modelchain._HEALTH_PATH.read_text(encoding="utf-8"))["models"][m]
        assert entry["failures"] == n and "cooldown_until" not in entry and "last_failure" in entry
    modelchain.record(m, "ok")
    entry = json.loads(modelchain._HEALTH_PATH.read_text(encoding="utf-8"))["models"][m]
    assert entry["failures"] == 0 and "last_ok" in entry


def test_a_broken_health_file_is_an_empty_state_and_recording_never_raises(mc):
    modelchain._HEALTH_PATH.write_text("not json", encoding="utf-8")
    assert modelchain.chain()[0] == "gemini-3.10-flash"
    modelchain.record("m", "fail")                                                          # rewrites it cleanly
    assert json.loads(modelchain._HEALTH_PATH.read_text(encoding="utf-8"))["models"]["m"]["failures"] == 1


def test_a_pin_in_env_wins_and_with_none_the_discovered_chain_is_used(monkeypatch):
    monkeypatch.setattr(modelchain, "chain", lambda: ["found-1", "found-2"])
    monkeypatch.delenv("GEMINI_MODELS", raising=False)
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    assert llm._get_models("extract[1]") == ["found-1", "found-2"]
    assert llm._get_models("listen[1/1]") == ["found-1", "found-2"]
    monkeypatch.setenv("GEMINI_MODELS", "p1,p2")
    assert llm._get_models("extract[1]") == ["p1", "p2"]
    monkeypatch.setenv("GEMINI_MODELS_LISTEN", "q1")
    assert llm._get_models("listen[1/1]") == ["q1"]


def test_the_fallback_walk_reports_failures_and_successes_but_not_quota_limits(monkeypatch):
    seen = []
    monkeypatch.setattr(modelchain, "record", lambda model, outcome: seen.append((model, outcome)))
    monkeypatch.setattr(llm, "_call_with_timeout", lambda build_call, model, timeout: build_call(model))
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)

    def build(model):
        if model == "broken":
            raise RuntimeError("503 UNAVAILABLE")
        if model == "limited":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "answer"
    assert llm._generate_with_fallback(["limited", "broken", "good"], build, log=lambda *_: None, context="t") == ("answer", "good")
    assert seen == [("broken", "fail"), ("good", "ok")]


def test_tunables_fall_back_to_the_default_and_follow_the_file(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    assert tunables.get("resurface", "quiet_hours_end") == 0                                # no file: the default
    cfg.write_text("[resurface]\nquiet_hours_end = 7\nstandalone_image_probability = 0.5\n", encoding="utf-8")
    assert tunables.get("resurface", "quiet_hours_end") == 7
    assert tunables.get("resurface", "standalone_image_probability") == 0.5
    assert tunables.get("resurface", "quiet_hours_start") == 0                              # a key the file lacks: the default
    cfg.write_text("[resurface]\nquiet_hours_end = 9\n", encoding="utf-8")
    import os
    os.utime(cfg, ns=(time.time_ns() + 5_000_000_000, time.time_ns() + 5_000_000_000))      # a new modification time
    assert tunables.get("resurface", "quiet_hours_end") == 9                                # re-read on change


def test_a_wrong_value_is_ignored_not_fatal(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    cfg.write_text('[resurface]\nquiet_hours_end = "late"\nstandalone_image_probability = 3\nruns_per_day = 2.5\n'
                   '[telegram]\nmax_attempts = true\n', encoding="utf-8")
    assert tunables.get("resurface", "quiet_hours_end") == 0                                # not a number
    assert tunables.get("resurface", "standalone_image_probability") == 0.15                # out of range
    assert tunables.get("resurface", "runs_per_day") == 24                                  # a whole number is expected
    assert tunables.get("telegram", "max_attempts") == 3                                    # a boolean is not a number
    cfg.write_text("[resurface\nbroken", encoding="utf-8")
    import os
    os.utime(cfg, ns=(time.time_ns() + 9_000_000_000, time.time_ns() + 9_000_000_000))
    assert tunables.get("resurface", "quiet_hours_end") == 0                                # an unreadable file: all defaults
    with pytest.raises(KeyError):
        tunables.get("resurface", "no_such_key")


def test_the_config_file_that_ships_says_exactly_what_the_defaults_say(monkeypatch):
    import tomllib
    from pathlib import Path
    shipped = tomllib.loads((Path(tunables.__file__).resolve().parents[2] / "config.toml").read_text(encoding="utf-8"))
    for section, keys in tunables._TABLE.items():
        for key, (default, _lo, _hi) in keys.items():
            assert shipped[section][key] == default, f"config.toml [{section}] {key} differs from the built-in default"
    for section, keys in tunables._TEXT.items():
        for key, default in keys.items():
            assert shipped[section][key] == default, f"config.toml [{section}] {key} differs from the built-in default"
    for section, keys in tunables._WORDS.items():                                   # the user's own list (2026-11-24): valid words, as written
        for key in keys:
            assert tunables.get_words.__name__ and shipped[section][key] == list(dict.fromkeys(shipped[section][key]))
            assert all(tunables._WORD.fullmatch(v) for v in shipped[section][key]), f"config.toml [{section}] {key} holds a bad name"
    for section, keys in shipped.items():
        for key in keys:
            assert (key in tunables._TABLE[section] or key in tunables._TEXT.get(section, {}) or key in tunables._WORDS.get(section, {})), \
                f"config.toml has [{section}] {key}, which the code does not know"


def test_quiet_hours_are_off_and_all_24_hours_are_active():
    """Design decision 2026-11-18: no fixed sleep and wake times, so no hour is skipped."""
    from datetime import datetime

    from lifecompass import resurface
    assert resurface.ACTIVE_HOURS_PER_DAY == 24
    assert not any(resurface._in_quiet_hours(datetime(2026, 11, 18, hour)) for hour in range(24))


# ---- Hydrogen instead of Flash (design decision 2026-11-23): the family is a setting, the mechanics are the same
def test_the_chain_is_made_of_the_hydrogen_family_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(modelchain, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(modelchain, "_HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(modelchain, "_refresh_failed_at", 0.0)
    names = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-4-hydrogen", "gemini-4-hydrogen-lite", "gemini-4-hydrogen-preview",
             "gemini-4-argon", "gemini-4-carbon", "gemini-4-neon", "gemini-flash-latest", "gemini-hydrogen-latest"]
    monkeypatch.setattr(modelchain, "_list_remote", lambda: list(names))
    assert tunables.get_text("models", "family") == "hydrogen"
    assert modelchain.candidates() == ["gemini-4-hydrogen"]                       # only the stable name of the family
    assert modelchain.chain() == ["gemini-4-hydrogen"]
    names.append("gemini-5-hydrogen")                                             # a newer one comes first, the old one stays behind it
    modelchain._CACHE_PATH.unlink()
    assert modelchain.chain() == ["gemini-5-hydrogen", "gemini-4-hydrogen"]


def test_with_no_list_at_all_the_last_resort_is_hydrogen(tmp_path, monkeypatch):
    monkeypatch.setattr(modelchain, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(modelchain, "_refresh_failed_at", 0.0)

    def down():
        raise RuntimeError("no network")
    monkeypatch.setattr(modelchain, "_list_remote", down)
    assert modelchain.chain() == ["gemini-4-hydrogen"]


def test_the_family_setting_accepts_plain_words_only(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    cfg.write_text('[models]\nfamily = "flash"\nlast_resort = "gemini-flash-latest"\n', encoding="utf-8")
    assert tunables.get_text("models", "family") == "flash" and tunables.get_text("models", "last_resort") == "gemini-flash-latest"
    cfg.write_text('[models]\nfamily = "Hydrogen; drop table"\nlast_resort = 7\n', encoding="utf-8")
    assert tunables.get_text("models", "family") == "hydrogen" and tunables.get_text("models", "last_resort") == "gemini-4-hydrogen"


def test_the_real_config_asks_for_hydrogen():
    import tomllib
    from pathlib import Path
    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "config.toml").read_text(encoding="utf-8"))
    assert cfg["models"]["family"] == "hydrogen" and cfg["models"]["last_resort"] == "gemini-4-hydrogen"


# ---- the fallback after Hydrogen (design decision 2026-11-24): the family chain first, then the names written in [models] after
def test_the_chain_is_hydrogen_then_the_written_fallbacks_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(modelchain, "_CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(modelchain, "_HEALTH_PATH", tmp_path / "health.json")
    monkeypatch.setattr(modelchain, "_refresh_failed_at", 0.0)
    monkeypatch.setattr(modelchain, "_list_remote", lambda: ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-4-hydrogen"])
    monkeypatch.setattr(tunables, "get_words", lambda section, key: ["gemini-3.8-flash", "gemini-3.6-flash"])
    assert modelchain.chain() == ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.6-flash"]
    monkeypatch.setattr(tunables, "get_words", lambda section, key: ["gemini-3.8-flash", "gemini-4-hydrogen"])   # no duplicate of a family model
    assert modelchain.chain() == ["gemini-4-hydrogen", "gemini-3.8-flash"]


def test_the_fallback_list_is_read_from_config_and_a_bad_value_is_ignored(tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    monkeypatch.setenv("LIFECOMPASS_CONFIG", str(cfg))
    cfg.write_text('[models]\nafter = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.8-flash"]\n', encoding="utf-8")
    assert tunables.get_words("models", "after") == ["gemini-3.8-flash", "gemini-3.6-flash"]
    cfg.write_text('[models]\nafter = ["Bad Name; drop"]\n', encoding="utf-8")
    assert tunables.get_words("models", "after") == []
    cfg.write_text('[models]\nafter = 5\n', encoding="utf-8")
    assert tunables.get_words("models", "after") == []
