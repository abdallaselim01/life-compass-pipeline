"""The image calls (match, caption) use the backup key when the main key's model is out of daily quota, and an answer that cannot be
read moves on to the next model. (These cases lived in test_links_batch.py until the links stage was removed, 2026-11-16.)"""
import json
import types

from lifecompass import enrichment, llm

IMAGES = [(b"x", "image/png")]


def _two_key_clients(asked, reply):
    """A fake genai.Client factory: which key a request was made with is recorded in `asked`."""
    def make(api_key=None):
        def generate_content(**kw):
            asked.append(api_key)
            return types.SimpleNamespace(text=json.dumps(reply))
        return types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    return make


def _strict(response, schema):
    return schema.model_validate_json(response.text)


def test_image_matching_uses_the_backup_key_when_the_main_key_is_out_of_daily_quota(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "main")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP", "backup")
    monkeypatch.setenv("GEMINI_MODELS", "a")
    asked = []
    monkeypatch.setattr(enrichment.genai, "Client", _two_key_clients(asked, {"matches": []}))
    monkeypatch.setattr(enrichment, "_parse_structured", _strict)
    monkeypatch.setattr(llm, "_exhausted_models_this_run", {"a"})        # the main key's model is already flagged out of quota
    assert enrichment.match_principles_to_images(["مبدأ"], IMAGES, log=lambda *_: None) == {}
    assert asked == ["backup"]                                          # nothing was spent on the main key


def test_without_a_quota_problem_only_the_main_key_is_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "main")
    monkeypatch.setenv("GEMINI_API_KEY_BACKUP", "backup")
    monkeypatch.setenv("GEMINI_MODELS", "a")
    asked = []
    monkeypatch.setattr(enrichment.genai, "Client", _two_key_clients(asked, {"matches": []}))
    monkeypatch.setattr(enrichment, "_parse_structured", _strict)
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    assert enrichment.match_principles_to_images(["مبدأ"], IMAGES, log=lambda *_: None) == {}
    assert asked == ["main"]


def test_an_unreadable_answer_moves_to_the_next_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODELS", "a,b")
    asked = []

    def generate_content(**kw):
        asked.append(kw["model"])
        return types.SimpleNamespace(text="garbage" if kw["model"] == "a" else json.dumps({"matches": []}))

    client = types.SimpleNamespace(models=types.SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(enrichment.genai, "Client", lambda api_key=None: client)

    def parse(response, schema):
        try:
            return schema.model_validate_json(response.text)
        except ValueError:
            return None

    monkeypatch.setattr(enrichment, "_parse_structured", parse)
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    assert enrichment.match_principles_to_images(["مبدأ"], IMAGES, log=lambda *_: None) == {}
    assert asked == ["a", "b"]
