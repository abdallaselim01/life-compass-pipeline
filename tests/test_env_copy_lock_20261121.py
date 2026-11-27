"""Audit 2026-11-21, P4: copies of .env inside backups are locked like the live .env (automatically, hourly)."""
import subprocess
from types import SimpleNamespace

from lifecompass import health


def test_only_unlocked_env_copies_are_fixed(tmp_path, monkeypatch):
    (tmp_path / "b1").mkdir()
    (tmp_path / "b2").mkdir()
    open_copy, locked_copy = tmp_path / "b1" / ".env", tmp_path / "b2" / ".env.bak_1"
    for f in (open_copy, locked_copy, tmp_path / "b1" / ".env.example", tmp_path / "b1" / "notes.txt"):
        f.write_text("x")
    monkeypatch.setenv("USERNAME", "alice")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        path = cmd[1]
        if len(cmd) == 2:                               # icacls <file>: show the ACL
            if path == str(open_copy):
                out = f"{path} BUILTIN\\Administrators:(F)\\n  SomeOtherGroup:(I)(RX)\\n  PC\\\\alice:(I)(F)\\n"
            else:
                out = f"{path} BUILTIN\\Administrators:(F)\\n  NT AUTHORITY\\SYSTEM:(F)\\n  PC\\\\alice:(F)\\n"
            return SimpleNamespace(returncode=0, stdout=out)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    fixed = health.lock_env_copies(tmp_path)
    assert fixed == [str(open_copy.relative_to(tmp_path))]
    grants = [c for c in calls if len(c) > 2]
    assert len(grants) == 1 and "/inheritance:r" in grants[0] and grants[0][1] == str(open_copy)
    assert not any(c[1].endswith((".env.example", "notes.txt")) for c in calls)
