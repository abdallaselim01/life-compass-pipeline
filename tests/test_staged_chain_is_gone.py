"""The staged chain (select, edit, cold reader) is gone (design decision 2026-11-14): its modules, prompts, mode switch and settings."""
import importlib.util
from pathlib import Path

import pytest

from lifecompass import llm, production
from lifecompass import stage_fingerprints as sf

ROOT = Path(__file__).resolve().parents[1]


def test_the_staged_modules_and_prompts_are_gone():
    for name in ("selector", "editor"):
        assert importlib.util.find_spec(f"lifecompass.{name}") is None, name
    for name in ("select_prompt.txt", "edit_prompt.txt", "reader_prompt.txt"):
        assert not (ROOT / "prompts" / name).exists(), name


def test_there_is_no_production_mode_any_more():
    assert not hasattr(production, "production_mode") and not hasattr(production, "stage_after_listen")
    assert not hasattr(sf, "MODE_STAGES")
    assert sf.PRINCIPLE_STAGES == ("listen", "extract") and set(sf.STAGE_PROMPTS) == {"listen", "extract"}


def test_the_stage_vocabulary_has_no_select_or_edit():
    assert "select" not in llm.STAGES and "edit" not in llm.STAGES
    for stage in ("select", "edit"):
        with pytest.raises(ValueError, match="unknown stage"):
            llm.request_settings(stage)
