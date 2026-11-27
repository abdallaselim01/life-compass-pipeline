# Setup

## One-time setup

1. **Python 3.10+**, then from the project folder:
   ```
   pip install -r requirements.txt
   ```
   There is no `pip install -e .`: the project runs straight from the `src` folder. The launchers set `PYTHONPATH` and the scripts add `src` to `sys.path` themselves, so nothing of the project is written into `site-packages`.

2. **Gemini API key**: free at https://aistudio.google.com/apikey

3. **Telegram bot**:
   - Open Telegram, find **BotFather**, send `/newbot` and follow the steps.
   - You get a **Bot Token** (a long string such as `123456:ABC-DEF...`).
   - Find **@userinfobot**, send it any message, and it gives you your **Chat ID**.
   - **Important**: you must start a conversation with your own bot first (send it `/start` or any message). Telegram does not let a bot message anyone who has not started a conversation with it.

4. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (and optionally `GEMINI_API_KEY_BACKUP`, a key from a second Google project with its own quota). The number of principles per day is not in `.env`: it is `daily_count` in `config.toml` (default 5).

5. **The bookmarklet** (the way a video is started):
   ```
   javascript:(function(){fetch('http://127.0.0.1:8935/process?url='+encodeURIComponent(location.href)).then(r=>r.text()).then(t=>alert(t)).catch(e=>alert('Listener not running? '+e));})();
   ```
   Save it as a browser bookmark (for example "-> Compass").

6. **Start the listener automatically at login**: create a shortcut (not a copy) to `start_listener_silent.vbs` in the Windows Startup folder `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\` (target: `wscript.exe "<project folder>\start_listener_silent.vbs"`). The VBS script works out the project folder from its own location, so do not copy it out of the folder. To start it by hand the first time: `wscript.exe "<project folder>\start_listener_silent.vbs"`.

7. **Scheduled tasks**: run
   ```
   python scripts\install_tasks.py
   ```
   It registers every task of the system from its saved copy in `windows_config\scheduled_tasks` (Retry Pending, Random Resurface, Listener Watchdog, Health Check), with the paths of the current project folder and the current Python. The Health Check task repairs missing or disabled tasks by itself every hour afterwards.

   The **Random Resurface** task must run **every hour, not once a day**, so that the randomness is real: each run rolls a die with probability `daily_count / 24` (from `config.toml`) to send one principle, so most runs send nothing. That is expected and intended.

## Daily use

After you finish a video, press the bookmarklet. The video is processed in the background (the time depends on the video length and on model load; follow the progress in `listener.log` and in `logs/`):

1. the whole audio is downloaded as one file (no cutting);
2. listening: a Gemini model writes down what was said;
3. the principles are extracted in the speaker's exact words (with the verbatim guard);
4. everything is stored in `data/principles.db`;
5. images (if you dropped any) are matched to the principles, and an unmatched image gets a caption;
6. **immediate delivery** on Telegram of all the principles extracted from that video, under its title.

All day long the Task Scheduler tries every hour (24 times a day), and most attempts send nothing, because the target is an average of 5 messages a day spread at random, not 24. Each message holds a single principle, preferring principles that have not been shown yet or have not been shown for a long time.