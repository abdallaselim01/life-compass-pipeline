# Architecture

## Principle

The goal is the speaker's own words, verbatim (formal Arabic stays formal, dialect stays dialect), with maximum quality and coverage and the simplest possible system. YouTube's automatic captions are a weak recognition of the dialect and any textual repair afterwards is guesswork, so the design starts from **listening**: a model listens to the audio and writes down what was said, a second model extracts the principles from that text, and code is responsible only for the mechanical parts. Any judgment about meaning belongs to a model.

## Flow

```
YouTube video (bookmarklet -> listener on port 8935)
  -> audio.py                  downloads the whole audio as one file (mono, about 10 MB for 40 minutes). It is never cut.
  -> transcriber.py (listen)   a model (the newest one in the discovered chain, the others as fallback) listens to the whole recording
                               in one call and writes what was said in paragraphs (an empty line between paragraphs), without
                               stuttering, with [غير واضح] where speech could not be made out. Audio larger than 14 MB is uploaded once
                               with the Files API.
  -> principle_extractor.py    a model reads the whole transcript and returns the principles; each principle is a continuous passage
        (extract)              of the transcript. The reply shape is one line inside prompts/extract_prompt.txt (no schema text is sent).
        + verbatim guard       (mechanical): the passage as it is, or stutter removed, or repaired by alignment with the transcript, or
                               rejected if fewer than half of its words are found in the transcript. The unclear-speech marker splits a
                               principle into two sides (each side is a principle). Every change is logged with the original text.
  -> store.py                  SQLite: principles, transcript, document, stage fingerprints, retry queue.
  -> enrichment.py             after storing, without blocking delivery: matches images to principles (match: the model sees the
                               picture itself) and captions an unmatched image (caption: in the speaker's own words, with the same
                               verbatim guard).
  -> publisher.py              Telegram: an album of the video's images, then the principles message.
```

- **`principle` is the passage text after the guard.** The system keeps no timestamps, no source quote and no unit ranges.
- **Code does only what has no opinion in it:** downloading, storing, checking that a passage is verbatim (and repairing stutter and a small gap), splitting at the unclear marker, fingerprints, time limits, the retry queue. It never judges where a principle starts or ends or which phrasing is best: there are no word lists, thresholds or punctuation rules for that.
- **Stage fingerprints** (`stage_fingerprints.py`): each AI stage (listen, extract, match, caption) has a fingerprint = its prompt + its API settings. When either changes, that stage and every stage after it is redone, but only when the video is sent for processing again, with no Telegram message, and after a backup (`backups/auto`). The model itself is not part of the fingerprint, because it is chosen at runtime by the fallback chain.
- **Time limits:** for each kind of call the wait limit is `min(900, max(floor, 2 x the slowest recorded success))` seconds, kept in `data/call_timings.json`. The limit of listen and extract also grows with the length of the recording.
- **Failure and retry:** a failed call (503, quota, timeout) goes into the `pending_retries` queue and `scripts/retry_pending.py` retries it every 5 minutes. A failed listen restarts from the beginning (there is no cutting and no resuming piece by piece).
- **A run is recorded while it runs:** as soon as a video starts processing, a row is written in `pending_retries` (`last_error = running`, its process id, and `last_attempt_at` as a heartbeat every 5 minutes). If the process dies (restart, crash, power cut) and there has been no heartbeat for 20 minutes (`[maintenance] orphan_run_minutes`) and its process no longer exists, `retry_pending.py` finishes it by itself; if it is still alive (even if the machine slept) it is left alone. If the very first step (reading the video page) fails, the video is queued as well and one message is sent.
- **Time limits for yt-dlp and ffmpeg commands:** `[audio] info_timeout_seconds / download_timeout_seconds / probe_timeout_seconds` (300 / 3600 / 60 seconds). A command that hangs is stopped and counted as a failure that goes to the queue.
- **Short tasks:** the watchdog, the health check and the resurface task are limited to one hour (`ExecutionTimeLimit`); the retry task keeps the default because a long run after the machine wakes is legitimate.
- **Images:** whenever a video has images, its principles are never sent to Telegram without them. The principle messages wait behind the album; a principle that has pictures is never sent as text alone. A principle longer than 1024 characters goes as the album first and the text right after it as its own message.
- **Logging** (`logs/YYYYMMDD_HHMMSS.log`): every request, every reply, every decision, and every change the guard made, with the original text.

## Database

`principles` (and `principles_archive`), `documents` (and history), `transcripts`, `videos`, `processed_videos`, `images` (and `images_history`), `principle_images` (and archive), `stage_fingerprints`, `pending_retries`, `settings`. The version is in `PRAGMA user_version` (currently 8) and the migrations run once per process, so a restored old backup migrates itself. Images carry no description: the model sees the picture itself.

## Rules the system follows

- The speaker is the authority: any directive from him is extracted, in any field. The criterion is how general the sentence is, not its topic.
- A principle, not a procedure: only a step tied to a circumstance, a price, a tool or a law of the time of the recording is dropped, and so are dates, numbers and examples with no rule in them.
- His words verbatim, with no softening or inference.
- A complete idea with a standalone start and an uncut tail; one idea per item; the strongest wording when he repeats himself.
- The prompts contain no examples taken from particular videos.
- No patching: an item that loses its value is dropped, and a video can yield zero principles.
- Nothing dead stays in the system: whatever no entry point reaches is removed (or moved aside after a backup).