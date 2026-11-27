"""
Backlog processor: feeds every video from a channel through the ingest
pipeline, oldest-first, with progress tracking (via store.processed_videos)
so an interrupted run resumes exactly where it left off -- no wasted
Gemini quota re-processing videos already done.

Usage:
    python scripts/process_channel.py <channel_url_or_handle> [--limit N] [--delay SECONDS]

Example:
    python scripts/process_channel.py "https://www.youtube.com/@SomeChannel/videos" --delay 20

Why oldest-first: gives a coherent picture in the store as early as possible and matches "process the whole backlog once, then
just keep up with new uploads" rather than cherry-picking recent videos first.

Why a delay between videos: each video is several Gemini calls (listen, extract, image match and captions), and the free-tier quota
is a real, hard limit (20 requests per day per model). A deliberate pause between videos spreads the request rate out instead of
bursting the whole channel in a few minutes and hitting 429s on most of it.

When a stage cannot run (every model out of quota, the network down): the pipeline saves that video in the retry queue, and the
5-minute "LifeCompass Retry Pending" task finishes it by itself later. This script then stops the batch, because the next video would
hit the same wall; progress so far is saved, and rerunning the script later skips every video already processed.

This is a manual tool (a video is started by the bookmarklet; the system does not process a channel by itself).
"""
from __future__ import annotations

import argparse
import sys
sys.dont_write_bytecode = True   # BEFORE the package is imported: its own __init__ cannot stop the first .pyc being written (audit 2026-11-22)
import time
import json
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from lifecompass.audio import _run
from lifecompass.pipeline import process_video
from lifecompass import store


def list_channel_videos(channel_url: str) -> list[dict]:
    """Uses yt-dlp --flat-playlist (metadata only, no downloads) to list
    every video on a channel, oldest first. Returns dicts with at least
    'id', 'title', 'url'."""
    cmd = [
        sys.executable, "-m", "yt_dlp",           # the project's own up-to-date yt-dlp (vendor/), like every other yt-dlp call
        "--flat-playlist",
        "--dump-single-json",
        "--playlist-reverse",  # oldest-first (yt-dlp's default is newest-first)
        channel_url,
    ]
    proc = _run(cmd, timeout=1800)         # listing a whole channel: generous, but never forever
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp failed to list channel videos:\n{proc.stderr}")
    data = json.loads(proc.stdout)
    entries = data.get("entries", [])
    videos = []
    for e in entries:
        vid = e.get("id")
        if not vid:
            continue
        videos.append({
            "id": vid,
            "title": e.get("title", vid),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("channel_url", help="Channel URL, e.g. https://www.youtube.com/@handle/videos")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to process this run")
    parser.add_argument("--delay", type=float, default=15.0, help="Seconds to wait between videos")
    args = parser.parse_args()

    print(f"Listing videos on channel: {args.channel_url} ...")
    videos = list_channel_videos(args.channel_url)
    print(f"Found {len(videos)} videos total (oldest first).")

    already = store.count_processed_videos()
    print(f"{already} already processed in a previous run -- those will be skipped instantly.\n")

    to_run = videos if args.limit is None else videos[:args.limit]

    processed_this_run = 0
    added_this_run = 0
    for i, v in enumerate(to_run, 1):
        if store.is_video_processed(v["id"]):
            continue  # instant skip, no Gemini/yt-dlp calls spent

        print(f"[{i}/{len(to_run)}] {v['title'][:70]} ...")
        result = process_video(v["url"], log=lambda l: None)  # quiet -- per-run detail is in logs/
        if result.ok:
            print(f"  OK: {result.message}")
            added_this_run += result.added
        else:
            print(f"  FAILED: {result.message}")
            if "queued for retry" in result.message:
                print("\nA stage could not run (quota or network), so this video is in the retry queue "
                      "and finishes by itself. Stopping the batch here -- progress so far is saved, rerun "
                      "this script later (e.g. tomorrow) to continue from this exact point.")
                break

        processed_this_run += 1
        time.sleep(args.delay)

    total_videos = store.count_processed_videos()
    total_principles = store.count_all()
    print(f"\nThis run: {processed_this_run} videos processed, {added_this_run} new principles added.")
    print(f"Overall: {total_videos} videos processed, {total_principles} principles stored.")


if __name__ == "__main__":
    main()
