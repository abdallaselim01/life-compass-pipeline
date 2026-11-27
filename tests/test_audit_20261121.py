"""Audit 2026-11-21, P1: the project's own .env wins over a same-named variable set elsewhere in Windows.

python-dotenv's load_dotenv() keeps an existing environment variable by default. A user-level GEMINI_API_KEY therefore made every
process ignore the project's .env key. Every entry point now passes override=True, and the listener reads the .env before it
imports the modules that read the environment when they are imported (llm.py: GEMINI_USE_STRUCTURED_OUTPUT)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sources():
    return list((ROOT / "src" / "lifecompass").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))


def test_every_load_dotenv_call_overrides():
    offenders = []
    for path in _sources():
        for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if re.match(r"^\s*load_dotenv\(", line) and "override=True" not in line:
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, offenders


def test_listener_reads_the_env_before_importing_the_pipeline():
    text = (ROOT / "src" / "lifecompass" / "listener.py").read_text(encoding="utf-8")
    assert text.index("load_dotenv(override=True)") < text.index("from .pipeline import")
