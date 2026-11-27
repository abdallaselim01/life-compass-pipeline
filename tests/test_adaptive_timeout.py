"""The wait limit of a model call is learned per kind of call from the calls that really succeeded (design decision 2026-11-07):
min(ceiling, max(120 s, 2 x slowest success)), the old ceiling while a kind has no recorded success. No model is called."""
import json

import pytest

from lifecompass import llm


def quiet(*args, **kwargs):
    return None


def timings():
    return json.loads(llm._TIMINGS_PATH.read_text(encoding="utf-8"))["kinds"]


def test_the_kind_of_a_call_is_its_context_without_the_suffix():
    assert llm.call_kind("listen[2/7]") == "listen"
    assert llm.call_kind("extract[1] (coverage)") == "extract"
    assert llm.call_kind("describe_image") == "describe_image"
    assert llm.call_kind("") == "unknown"


def test_a_kind_with_no_recorded_success_keeps_the_ceiling():
    assert llm.adaptive_timeout("describe_image", 900.0) == 900.0


def test_the_limit_is_two_times_the_slowest_success_within_the_floor_and_the_ceiling():
    llm.record_success_time("describe_image", 10.6)
    assert llm.adaptive_timeout("describe_image", 900.0) == 120.0            # 2 x 10.6 is under the floor
    llm.record_success_time("describe_image", 80.0)
    assert llm.adaptive_timeout("describe_image", 900.0) == 160.0            # 2 x 80
    llm.record_success_time("describe_image", 500.0)
    assert llm.adaptive_timeout("describe_image", 900.0) == 900.0            # 2 x 500 is over the ceiling
    assert llm.adaptive_timeout("describe_image", 60.0) == 60.0              # a ceiling below the floor stays the ceiling


def test_only_the_slowest_success_is_kept_and_every_success_is_counted():
    for seconds in (50.0, 10.0, 30.0):
        llm.record_success_time("match_principles_to_images", seconds)
    entry = timings()["match_principles_to_images"]
    assert (entry["max"], entry["n"]) == (50.0, 3)


def test_the_kinds_do_not_affect_each_other_and_the_bracket_suffix_does_not_split_a_kind():
    llm.record_success_time("listen[1/3]", 300.0)
    llm.record_success_time("listen[3/3]", 20.0)
    assert llm.adaptive_timeout("listen[2/3]", 900.0) == 600.0
    assert llm.adaptive_timeout("extract[1]", 900.0) == 900.0


def test_an_unreadable_record_means_no_data_and_a_failed_write_never_breaks_a_call(tmp_path, monkeypatch):
    llm._TIMINGS_PATH.write_text("not json", encoding="utf-8")
    assert llm.adaptive_timeout("describe_image", 900.0) == 900.0
    broken = tmp_path / "a_directory"
    broken.mkdir()
    monkeypatch.setattr(llm, "_TIMINGS_PATH", broken)                       # cannot be written as a file
    llm.record_success_time("describe_image", 5.0)                           # must not raise
    assert llm.adaptive_timeout("describe_image", 900.0) == 900.0


def test_a_successful_attempt_is_recorded_and_the_next_one_of_that_kind_waits_less(monkeypatch):
    waited = []
    real = llm._call_with_timeout

    def watching(build_call, model_name, timeout):
        waited.append(timeout)
        return real(build_call, model_name, timeout)

    monkeypatch.setattr(llm, "_call_with_timeout", watching)
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    for _ in range(2):
        resp, model = llm._generate_with_fallback(["m"], lambda name: "ok", log=quiet, context="describe_image", timeout=900.0)
        assert (resp, model) == ("ok", "m")
    assert waited == [900.0, 120.0]                                          # first call: no data; second: learned
    assert timings()["describe_image"]["n"] == 2


def test_a_failed_or_cut_attempt_is_never_recorded(monkeypatch):
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())

    def fails(name):
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(RuntimeError):
        llm._generate_with_fallback(["m"], fails, log=quiet, context="caption_standalone_image", timeout=900.0)
    monkeypatch.setattr(llm, "_call_with_timeout", lambda b, m, t: (_ for _ in ()).throw(TimeoutError("cut")))
    with pytest.raises(TimeoutError):
        llm._generate_with_fallback(["m"], lambda name: "x", log=quiet, context="caption_standalone_image", timeout=900.0)
    assert not llm._TIMINGS_PATH.exists()


def test_the_chain_still_moves_to_the_next_model_when_the_learned_limit_cuts_a_hung_call(monkeypatch):
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_TIMEOUT_FLOOR", 0.3)
    llm.record_success_time("describe_image", 0.01)                         # learned limit: 0.3 s
    import time

    def build(name):
        if name == "slow":
            time.sleep(5)                                                    # the hung call
        return name

    start = time.time()
    resp, model = llm._generate_with_fallback(["slow", "fast"], build, log=quiet, context="describe_image", timeout=900.0)
    assert (resp, model) == ("fast", "fast") and time.time() - start < 4
