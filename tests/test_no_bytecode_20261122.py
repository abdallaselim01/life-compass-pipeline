"""Audit 2026-11-22: no entry point writes bytecode (the package's own __init__ cannot stop its own .pyc, so the flag is set before it)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_script_that_imports_the_package_sets_the_flag_first():
    offenders = []
    for f in (ROOT / "scripts").glob("*.py"):
        text = f.read_text(encoding="utf-8-sig")
        if "lifecompass" not in text:
            continue
        flag = text.find("sys.dont_write_bytecode = True")
        first_import = min([i for i in (text.find("from lifecompass"), text.find("import lifecompass")) if i != -1] or [10**9])
        if flag == -1 or flag > first_import:
            offenders.append(f.name)
    assert not offenders, offenders


def test_the_listener_launchers_run_python_with_dash_b():
    for name in ("start_listener_silent.vbs", "start_listener.bat"):
        assert re.search(r"python -B -u -m lifecompass\.listener", (ROOT / name).read_text(encoding="utf-8"))
