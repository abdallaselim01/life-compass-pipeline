from pathlib import Path

import pytest

from lifecompass import audio


def _setup(monkeypatch, tmp_path, expected, durations, existing=False):
    monkeypatch.setattr(audio, "AUDIO_ROOT", tmp_path)
    monkeypatch.setattr(audio, "video_duration", lambda vid: expected)
    seq = iter(durations)
    monkeypatch.setattr(audio, "audio_duration", lambda p: next(seq))
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        (tmp_path / "vid" / "full.mp3").write_bytes(b"new")
        return None

    monkeypatch.setattr(audio, "_run", fake_run)
    if existing:
        (tmp_path / "vid").mkdir()
        (tmp_path / "vid" / "full.mp3").write_bytes(b"old")
        (tmp_path / "vid" / "chunk_0_10.mp3").write_bytes(b"oldchunk")
    return calls


def test_matching_file_is_reused_without_download(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, 2392.0, [2392.52], existing=True)
    assert audio.download_audio("vid").read_bytes() == b"old"
    assert calls == []


def test_mismatched_file_is_moved_aside_and_downloaded_again(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, 2392.0, [1500.0, 2392.4], existing=True)
    out = audio.download_audio("vid")
    assert out.read_bytes() == b"new" and len(calls) == 1
    aside = [p for p in tmp_path.iterdir() if p.name.startswith("vid.mismatch_")]
    assert len(aside) == 1 and (aside[0] / "full.mp3").read_bytes() == b"old"
    assert not (tmp_path / "vid" / "chunk_0_10.mp3").exists()   # stale chunk of the bad file is not reused


def test_fresh_download_with_wrong_length_is_an_error(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, 2392.0, [1000.0])
    with pytest.raises(RuntimeError, match="1000s but the video is 2392s"):
        audio.download_audio("vid")