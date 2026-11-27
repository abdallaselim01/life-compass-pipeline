"""Stage routing of the Gemini chain (order only)."""
from lifecompass import llm


def test_the_listen_stage_has_its_own_order_and_everything_else_the_general_chain(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setenv("GEMINI_MODELS_LISTEN", "b,a")
    assert llm._get_models("listen[1/5]") == ["b", "a"]
    assert llm._get_models("extract[1]") == ["a", "b"]                     # no own setting -> general chain
    assert llm._get_models("match_principles_to_images") == ["a", "b"]
    assert llm._get_models(None) == ["a", "b"]
    assert llm._get_models() == ["a", "b"]


def test_a_leftover_setting_for_a_removed_stage_is_ignored(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setenv("GEMINI_MODELS_EDIT", "c")
    monkeypatch.setenv("GEMINI_MODELS_SELECT", "c")
    assert llm._get_models("edit[1]") == ["a", "b"]
    assert llm._get_models("select[round 2]") == ["a", "b"]


def test_blank_stage_setting_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    monkeypatch.setenv("GEMINI_MODELS_LISTEN", "  ")
    assert llm._get_models("listen[1/1]") == ["a", "b"]
