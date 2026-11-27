"""Every Gemini request asks for the maximum-quality settings (design decision 2026-11-05): thinking_level HIGH and the
65,536-token output limit; no temperature / top_p / top_k (deprecated on these models) and no thinking_budget (legacy);
images at ULTRA_HIGH resolution."""
import types as pytypes

import pytest
from google.genai import types

from lifecompass import enrichment, llm
from lifecompass.principle_extractor import ExtractionResult


def check(cfg):
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.HIGH
    assert cfg.thinking_config.thinking_budget is None
    assert cfg.max_output_tokens == 65536
    assert cfg.temperature is None and cfg.top_p is None and cfg.top_k is None


def _chunk(text):
    return pytypes.SimpleNamespace(text=text, candidates=[], usage_metadata=None, prompt_feedback=None)


@pytest.fixture
def seen(monkeypatch):
    """A fake Gemini client that records every request it is sent."""
    monkeypatch.setattr(llm, "_exhausted_models_this_run", set())
    monkeypatch.setattr(llm, "_mark_gemini_exhausted_on_disk", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_MODELS_LISTEN", raising=False)
    monkeypatch.setenv("GEMINI_MODELS", "a")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls = []

    def stream(model, **kw):
        calls.append(kw)
        return iter([_chunk('{"items": [{"text": "x", "domain": "Faith"}]}')])

    def once(model, **kw):
        calls.append(kw)
        return _chunk('{"items": []}')

    client = pytypes.SimpleNamespace(models=pytypes.SimpleNamespace(generate_content_stream=stream, generate_content=once))
    monkeypatch.setattr(llm, "_client", lambda backup=False: client)
    return calls


def test_the_shared_settings_are_the_maximum():
    check(types.GenerateContentConfig(**llm.max_quality_config_kwargs()))


def test_audio_and_json_calls_ask_for_the_maximum_settings(seen):
    llm.generate_audio_text(b"audio", "prompt", context="listen[1/1]")
    llm.generate_json("prompt", ExtractionResult, context="extract[1]")
    assert len(seen) == 2
    for kw in seen:
        check(kw["config"])


def test_match_and_caption_send_the_pictures_themselves_at_ultra_high_resolution_with_the_maximum_settings(seen):
    quiet = lambda *a, **k: None
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    for call in (lambda: enrichment.match_principles_to_images(["مبدأ"], [(png, "image/png")], log=quiet),
                 lambda: enrichment.caption_standalone_image((png, "image/png"), "نص", log=quiet)):
        try:
            call()
        except Exception:       # the fake reply may not fit the schema; only the request matters here
            pass
    match_kw = next(kw for kw in seen if isinstance(kw["contents"][0], str))        # the prompt first, then the pictures
    caption_kw = next(kw for kw in seen if not isinstance(kw["contents"][0], str))   # the picture first, then the prompt
    check(match_kw["config"])
    check(caption_kw["config"])
    for part in (match_kw["contents"][2], caption_kw["contents"][0]):               # (prompt, "صورة 0:", picture) / (picture, prompt)
        assert part.inline_data.data == png
        assert part.media_resolution.level == types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH


def test_the_first_piece_limit_is_not_shorter_than_the_ceiling():
    assert llm._STREAM_FIRST_CHUNK_SECONDS == llm._MODEL_TIMEOUT_SECONDS == 900.0
