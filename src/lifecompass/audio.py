"""
Getting the speaker's voice: download the audio and keep it on disk whole.

Everything here is signal-level and deterministic. The recording is never cut.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

from . import tunables, ytdlp_update

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = PROJECT_ROOT / "data" / "audio"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The recording is NEVER cut: it goes to the model whole (design decision 2026-10-14 and 2026-11-09: the speaker does not stop talking, so a
# cut lands inside his flow, and the whole talk keeps his flow, terms and context). One too big for an inline request is uploaded instead
# (llm.generate_audio_text). There is no cutting code, no pause detection and no setting for either.

# The recording must be as long as the video. YouTube reports whole seconds and the mp3 encoder adds a
# fraction of a second, so a healthy file differs by under one second; a cut-off download differs by far more.
DURATION_TOLERANCE_SECONDS = tunables.get("audio", "duration_tolerance_seconds")   # config.toml [audio]

# What the last download_audio call did ("reused ...", "downloaded ..."); the pipeline prints it in the run log.
last_note = ""

_VIDEO_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url: str) -> str:
    m = _VIDEO_ID.search(url)
    if not m:
        raise ValueError(f"no video id in URL: {url}")
    return m.group(1)


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _run(cmd: list, timeout: float | None = None, **kw) -> subprocess.CompletedProcess:
    """Runs a command. A yt-dlp command runs with the project's own up-to-date copy (vendor/, see ytdlp_update.py) first on the
    path, and when it fails the system tries to get a newer yt-dlp (at most once every few hours) and runs it once more.
    Every call has a time limit (audit 2026-11-22): a command that hangs is stopped and reported as a failed command (return code
    124), which the callers already treat as an error the retry queue picks up, instead of blocking the run (or the scheduled retry
    task, which never starts a second copy while one is running) for good."""
    is_ytdlp = list(cmd[:3]) == [sys.executable, "-m", "yt_dlp"]
    env = ytdlp_update.env() if is_ytdlp else dict(os.environ, PYTHONIOENCODING="utf-8")

    def run():
        try:
            return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  creationflags=_NO_WINDOW, env=env, timeout=timeout, **kw)
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout:.0f}s and was stopped")
    r = run()
    if is_ytdlp and r.returncode != 0 and ytdlp_update.update_after_failure():
        env = ytdlp_update.env()
        r = run()
    return r


# One YouTube question serves both the title (the start of a run) and the length (the download, minutes later in the same run):
# the answer is kept in memory for a short while, so a video costs one yt-dlp call instead of two (maintenance 2026-11-25).
# Only a successful answer is kept; a failure is asked again as before.
_INFO_TTL_SECONDS = 600.0
_info_cache: dict = {}


def _info(video_id: str, what: str) -> dict:
    """{"title": str, "duration": float | None} of the video, from the memory of the last ten minutes or from YouTube (yt-dlp)."""
    hit = _info_cache.get(video_id)
    if hit and time.monotonic() - hit[0] < _INFO_TTL_SECONDS:
        return hit[1]
    r = _run([sys.executable, "-m", "yt_dlp", "--skip-download", "--no-warnings",
              "--dump-single-json", watch_url(video_id)], timeout=tunables.get("audio", "info_timeout_seconds"))
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"could not read video {what}: {r.stderr.strip()[-300:]}")
    raw = json.loads(r.stdout)
    info = {"title": (raw.get("title") or "").strip(), "duration": raw.get("duration")}
    _info_cache[video_id] = (time.monotonic(), info)
    for stale in [k for k, (t, _) in list(_info_cache.items()) if time.monotonic() - t >= _INFO_TTL_SECONDS]:
        _info_cache.pop(stale, None)                # the memory never grows: an entry older than the limit goes (threads may race: pop, never del)
    return info


def video_info(url: str) -> tuple:
    """(video_id, title) without downloading anything."""
    vid = extract_video_id(url)
    return vid, _info(vid, "info")["title"]


def video_duration(video_id: str) -> float:
    """The video's length in seconds as YouTube reports it (no download)."""
    duration = _info(video_id, "duration")["duration"]
    if not duration:
        raise RuntimeError("YouTube did not report a duration for this video")
    return float(duration)


def download_audio(video_id: str) -> Path:
    """Mono 16 kHz 32 kbps mp3 (about 0.25 MB per minute of speech). A recording already on disk is reused
    only if its length matches the video's; otherwise the folder is moved aside (never deleted) and the audio
    is downloaded again. A fresh download that does not match the video's length is an error."""
    global last_note
    expected = video_duration(video_id)
    dest = AUDIO_ROOT / video_id
    out = dest / "full.mp3"
    moved = ""
    if out.exists() and out.stat().st_size > 0:
        have = audio_duration(out)
        if abs(have - expected) <= DURATION_TOLERANCE_SECONDS:
            last_note = f"reused the recording already on disk ({have:.0f}s, the video is {expected:.0f}s)"
            return out
        moved = f"the recording on disk was {have:.0f}s but the video is {expected:.0f}s, so it was moved aside; "
        # The recording does not match the video: the whole folder goes aside (moved, never deleted).
        shutil.move(str(dest), str(AUDIO_ROOT / f"{video_id}.mismatch_{time.strftime('%Y%m%d_%H%M%S')}"))
    dest.mkdir(parents=True, exist_ok=True)
    r = _run([sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3",
              "--ffmpeg-location", FFMPEG, "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -b:a 32k",
              "--no-warnings", "-o", str(dest / "full.%(ext)s"), watch_url(video_id)],
             timeout=tunables.get("audio", "download_timeout_seconds"))
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"audio download failed: {(r.stderr or r.stdout).strip()[-400:]}")
    actual = audio_duration(out)
    if abs(actual - expected) > DURATION_TOLERANCE_SECONDS:
        raise RuntimeError(f"downloaded audio is {actual:.0f}s but the video is {expected:.0f}s")
    last_note = moved + f"downloaded now ({actual:.0f}s, the video is {expected:.0f}s)"
    return out


def audio_duration(path: Path) -> float:
    r = _run([FFMPEG, "-hide_banner", "-i", str(path)], timeout=tunables.get("audio", "probe_timeout_seconds"))
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise RuntimeError("could not read audio duration")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def recording_bytes(path: Path) -> bytes:
    """The whole recording, as it is on disk."""
    return Path(path).read_bytes()
