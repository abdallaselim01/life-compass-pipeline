# Life Compass Pipeline

Turns the YouTube videos of a speaker into **principles**: verbatim passages in which the speaker states an idea, a directive, advice or a warning. The principles are stored in a local database and delivered to you on Telegram in two separate ways:

1. **Immediately**: right after a video is processed, one message with all the principles extracted from it, under its title.
2. **Randomly, all day long**: single principles drawn from the whole accumulated archive at unpredictable times, so the ideas keep surrounding you instead of becoming a scheduled event you wait for.

The pipeline is built for Arabic speech. The prompts, the verbatim guard and the test data deal with Arabic text on purpose; everything else (code comments, messages, documentation, configuration) is in English.

## How it works

```
YouTube video  (bookmarklet -> local listener on port 8935)
  -> audio.py                  download the whole audio as ONE file (never cut)
  -> transcriber.py (listen)   a Gemini model listens to the whole recording in one call and writes down what was said
  -> principle_extractor.py    a model reads the transcript and returns the principles, each one a continuous passage of the transcript;
                               a mechanical verbatim guard checks, repairs or rejects every passage
  -> store.py                  SQLite: principles, transcript, document, stage fingerprints, retry queue
  -> enrichment.py             matches images you drop in a folder to the principles and captions the unmatched ones
  -> publisher.py              Telegram: an album of the video's images, then the principles message
```

Design rule: whatever an AI model is more competent at stays with the model (what an idea is, where it starts and ends); whatever code is more competent at stays with code (downloading, storing, checking that a passage really is verbatim, fingerprints, time limits, retries). Code never judges meaning.

More detail: [docs/architecture.md](docs/architecture.md).

## Project layout

```
life-compass-pipeline/
├── .env                      # your real secrets: copy .env.example (not tracked by git)
├── config.toml               # every tunable number, with safe defaults (no model names in it except the chain family)
├── data/                     # the database, audio, model health and cache files (not tracked)
├── logs/                     # a detailed log of every run (not tracked)
├── backups/                  # backups taken before changes and before reprocessing (not tracked)
├── docs/                     # architecture, setup and configuration
├── prompts/                  # the prompts and the request texts (all part of the stage fingerprints)
├── src/lifecompass/          # the code
├── scripts/                  # entry points: scheduled tasks, maintenance, diagnostics
├── tests/                    # tests that need no network
├── windows_config/           # saved Task Scheduler tasks and notes
├── vendor/                   # the yt-dlp copy the system keeps up to date by itself (not tracked)
├── start_listener.bat        # start the listener in a visible window (debugging)
└── start_listener_silent.vbs # the real, silent start (what the watchdog runs)
```

## Quick start

1. Python 3.10+ and `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and fill in the Gemini key and the Telegram bot token and chat id.
3. Start the listener (`start_listener.bat`) and register the scheduled tasks (`python scripts\install_tasks.py`).
4. Save the bookmarklet from [docs/setup.md](docs/setup.md) and press it on a video page.

Full instructions: [docs/setup.md](docs/setup.md). Configuration, models, self-maintenance and diagnostics: [docs/configuration.md](docs/configuration.md).

## Tests

```
python -m pytest tests -q -p no:cacheprovider
```

The tests need no network and no ffmpeg. Read-only checks of a live database: `python scripts/verify_principles_literal.py` and `python scripts/audit_db.py`.

## Known limits

- It must run on your own machine (a residential IP): YouTube blocks scraping from datacenter addresses.
- A video that was already processed (recorded in `processed_videos`) is skipped at once, with no yt-dlp or Gemini call.
- The availability of free Gemini models changes over time. The model chain is discovered by asking Google (see the configuration document); `scripts/list_models.py --probe` tries each model with one real call (it spends real quota).
- The free Gemini tier allows roughly 20 requests per day per model. One video costs about: listen (1) + extract (1) + match (1 if there are images) + one caption per unmatched image.

## License

MIT, see [LICENSE](LICENSE).
