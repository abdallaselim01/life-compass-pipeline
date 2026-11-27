"""
SQLite storage for extracted principles. Deliberately not Anki -- this
system is passive resurfacing, not spaced-repetition testing, and needs
a "related_principles" relationship Anki's card model has no place for.

Schema:
  principles(id, principle, domain, video_id, doc_hash,
             created_at, last_shown_at, times_shown)
             -- `principle` is one continuous passage of the speaker's words, copied from the video's
             -- transcript (kept in documents / document.py); doc_hash says which version of the document
             -- it was checked against. There is no separate quote column and no unit range.
  documents(video_id, doc_json, doc_hash, created_at)
             -- the verified, normalized transcript (one row per video)

No ORM -- this is a personal single-user tool at small scale (dozens to a
few hundred rows), raw sqlite3 is simpler and has zero dependency weight.

`context`, `weight`, `stance`, `confidence` were extracted fields in an
earlier version of this system, REMOVED 2026-09-07 after a deliberate
review against the system's actual goal: each one's only designed use was
either a frontend display the user explicitly doesn't want (`context`) or
backend scheduling logic that was already permanently disabled
(`weight`/`stance`), or had no use anywhere at all (`confidence`) -- pure
cost on every single extraction call (the model has to correctly fill
these for every item) for zero present benefit. `pick_for_resurfacing`
now uses plain times_shown priority.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# LIFECOMPASS_DB lets tests and evaluation runs use a scratch database.
DB_PATH = Path(os.environ.get("LIFECOMPASS_DB") or (_PROJECT_ROOT / "data" / "principles.db"))

_SCHEMA = """
-- One row per source video -- title/url live here now, not duplicated
-- onto every principle row (normalized 2026-09-20; principles used to
-- carry its own video_title/video_url on every row, which is why old
-- rows might still show up via the migration below rather than this
-- CREATE, but every row -- old and new -- resolves through this table
-- now via principles.video_id).
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS principles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    video_id TEXT NOT NULL DEFAULT '',
    doc_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_shown_at TEXT,
    times_shown INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

-- Case-insensitive, whitespace-normalized dedup index on the principle text
CREATE INDEX IF NOT EXISTS idx_principle_text ON principles (principle);

-- Videos whose review pipeline (recall_check / review_quality /
-- correct_text) hit a StageSkippedError (see extractor.py) partway
-- through -- e.g. every model's daily quota exhausted at once. Instead of
-- discarding the work already done (extraction, and whichever of the
-- three review stages already succeeded), the in-progress item list is
-- saved here so scripts/retry_pending.py can resume from the exact stage
-- that failed, on a later run, rather than starting the whole video over.
CREATE TABLE IF NOT EXISTS pending_retries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    stage TEXT NOT NULL,
    items_json TEXT NOT NULL,
    model_used TEXT NOT NULL DEFAULT '',
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_attempt_at TEXT,
    last_error TEXT NOT NULL DEFAULT ''
);

-- Screenshot images (see images.py) -- one row per screenshot the user took while watching a video and dropped into the incoming folder.
-- The match and caption calls are sent the picture itself; there is no description of it (schema 6 dropped those columns).
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL DEFAULT '',
    enhanced_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    standalone_caption TEXT NOT NULL DEFAULT '',
    times_shown INTEGER NOT NULL DEFAULT 0,
    last_shown_at TEXT
);

-- Many-to-many: one image can support several principles (the speaker
-- reused the same paper for more than one idea) and one principle can be
-- supported by several images -- see extractor.match_principles_to_images
-- for how these links are decided (semantic match, not timestamp).
CREATE TABLE IF NOT EXISTS principle_images (
    principle_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    PRIMARY KEY (principle_id, image_id),
    FOREIGN KEY (principle_id) REFERENCES principles(id),
    FOREIGN KEY (image_id) REFERENCES images(id)
);

-- Generic key/value settings store. Currently holds one key,
-- 'screenshot_checkpoint' (see images.py) -- the timestamp boundary used
-- to tell "screenshots for the video just watched" apart from ordinary,
-- unrelated screenshots sitting in the same default Windows folder.
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- The verified, normalized transcript of a video (see document.py): the
-- units that principles are ranges of. Rewritten whenever a video is
-- re-normalized; principles carry the doc_hash they were cut from.
CREATE TABLE IF NOT EXISTS documents (
    video_id TEXT PRIMARY KEY,
    doc_json TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Raw, unprocessed transcript text for every video the bookmarklet is run
-- on. Kept in its own table (not a column on processed_videos) for two
-- reasons: (1) transcripts can be tens of thousands of characters, and
-- processed_videos is read on every single ingest run (is_video_processed)
-- -- keeping that row lean matters more than saving a JOIN here; (2) it's
-- a separate concern (raw source data vs. processing status). video_id is
-- the PRIMARY KEY rather than a separate surrogate id since the
-- relationship to processed_videos is strictly one-to-one and this also
-- gives dedup-on-insert for free.
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES processed_videos(video_id)
);

-- Which prompt text produced each stage of a video (see stage_fingerprints.py). One row per (video, stage); written
-- after the stage's result is stored. prompt_fp = fingerprint of the stage's prompt; effective_fp also covers the
-- stages before it.
CREATE TABLE IF NOT EXISTS stage_fingerprints (
    video_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    prompt_fp TEXT NOT NULL,
    effective_fp TEXT NOT NULL,
    doc_hash TEXT NOT NULL DEFAULT '',
    produced_at TEXT NOT NULL,
    log_file TEXT NOT NULL DEFAULT '',
    settings_fp TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (video_id, stage)
);

-- Results replaced because a stage's prompt changed (archived, never deleted). A video's old principles leave the
-- live table so duplicate checks, resurfacing and statistics only see the current ones.
CREATE TABLE IF NOT EXISTS documents_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    doc_json TEXT NOT NULL,
    doc_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    transcript_text TEXT,
    archived_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS principles_archive (
    archive_id INTEGER PRIMARY KEY AUTOINCREMENT,
    id INTEGER NOT NULL,
    principle TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT '',
    video_id TEXT NOT NULL DEFAULT '',
    doc_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_shown_at TEXT,
    times_shown INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS principle_images_archive (
    principle_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    archived_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);

-- What an image row said before it was replaced in place (the same image dropped again, or its caption redone). The row keeps its id; this is
-- the record of what it used to be. Never deleted.
CREATE TABLE IF NOT EXISTS images_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    video_id TEXT NOT NULL DEFAULT '',
    enhanced_path TEXT NOT NULL,
    standalone_caption TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    replaced_at TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
"""

# `weight`/`stance`-based scheduling priority (and the `context` frontend
# display it was paired with) were REMOVED 2026-09-07 after a deliberate
# review: their only designed uses were either a frontend display the user
# explicitly doesn't want (context) or scheduling logic that was already
# permanently disabled (WEIGHT_BASED_PRIORITY_ENABLED = False) -- pure cost
# on every extraction call with zero present benefit. pick_for_resurfacing
# below now always uses plain times_shown priority; domain interleaving and
# the longest-since-shown tie-break are unaffected.


@dataclass
class Principle:
    id: int | None
    principle: str
    domain: str = ""
    doc_hash: str = ""
    video_title: str = ""
    video_url: str = ""
    video_id: str = ""
    created_at: str = ""
    last_shown_at: str | None = None
    times_shown: int = 0


@dataclass
class Image:
    id: int | None
    video_id: str
    # Absolute path, resolved against the CURRENT project folder when the row
    # is read (the DB stores it project-relative -- see resolve_image_path).
    enhanced_path: str
    created_at: str = ""
    standalone_caption: str = ""
    times_shown: int = 0
    last_shown_at: str | None = None


_SCHEMA_VERSION = 8          # 8: the empty, unused principle_links and principle_links_archive tables dropped; 7: unit_start/unit_end and the stage rows of removed stages dropped; 6: dead columns/table dropped; 5: stage_fingerprints.settings_fp; 3: stage names listen/select/edit
_prepared: set = set()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout=30: several processes (listener, retry_pending every 5 min,
    # send_daily hourly) share this file; sqlite3's default busy timeout is 5s.
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    key = str(DB_PATH)
    if key not in _prepared:          # schema work happens once per process, not per query
        _prepare(conn)
        _prepared.add(key)
    return conn


def _prepare(conn: sqlite3.Connection) -> None:
    """Create missing tables, then bring an older database up to
    _SCHEMA_VERSION. Versioned (PRAGMA user_version) instead of the old
    "try the ALTER on every connect and swallow the error" pattern. A brand
    new database is stamped with the current version directly."""
    fresh = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='principles'"
    ).fetchone() is None
    conn.executescript(_SCHEMA)
    if fresh:
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        return
    if conn.execute("PRAGMA user_version").fetchone()[0] >= _SCHEMA_VERSION:
        return
    conn.execute("BEGIN IMMEDIATE")      # one process migrates; the others wait, then see the new version
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            _migrate_to_v1(conn)
        if version < 2:
            _migrate_to_v2(conn)
        if version < 3:
            _migrate_to_v3(conn)
        if version < 5:
            _migrate_to_v5(conn)
        if version < 6:
            _migrate_to_v6(conn)
        if version < 7:
            _migrate_to_v7(conn)
        if version < 8:
            _migrate_to_v8(conn)
        conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


def _migrate_to_v8(conn: sqlite3.Connection) -> None:
    """The links between principles are gone (the links stage was removed 2026-11-16, nothing has written to these two tables since,
    and the live ones were empty): the tables and the code that read them are dropped (design decision 2026-11-22). A database restored from
    an older backup loses only its old link rows, which nothing reads."""
    conn.execute("DROP TABLE IF EXISTS principle_links")
    conn.execute("DROP TABLE IF EXISTS principle_links_archive")


def _migrate_to_v7(conn: sqlite3.Connection) -> None:
    """Dead parts dropped: the principles.unit_start/unit_end columns (a principle is the
    speaker's passage itself, not a range of numbered units; they were NULL in every row), their copies in principles_archive, the
    stage_fingerprints rows of stages that no longer exist (links, select, edit) and the queued retries of the removed links stage
    (find_related). Safe on any older database: a column that is not there is skipped."""
    for table in ("principles", "principles_archive"):
        for column in ("unit_start", "unit_end"):
            if _has_column(conn, table, column):
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.execute("DELETE FROM stage_fingerprints WHERE stage IN ('links', 'select', 'edit')")
    conn.execute("DELETE FROM pending_retries WHERE stage = 'find_related'")


def _migrate_to_v6(conn: sqlite3.Connection) -> None:
    """Dead parts dropped (design decision 2026-11-13: nothing dead stays): the empty transcribe_progress table (chunked listening is gone),
    the principles.t_start column (nothing has written or read it since timestamps were dropped; migration 2 dropped its siblings but
    missed it) and the image description columns of images and images_history (images are no longer described: the match and caption
    calls are sent the picture itself). Safe on any older database: a column or table that is not there is skipped."""
    conn.execute("DROP TABLE IF EXISTS transcribe_progress")
    for table, column in (("principles", "t_start"),
                          ("images", "description"), ("images", "description_failed"), ("images", "describe_fp"),
                          ("images_history", "description"), ("images_history", "description_failed"), ("images_history", "describe_fp")):
        if _has_column(conn, table, column):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


def _migrate_to_v5(conn: sqlite3.Connection) -> None:
    """Each stage fingerprint records the API settings fingerprint of the stage ('' for rows from before)."""
    if not _has_column(conn, "stage_fingerprints", "settings_fp"):
        conn.execute("ALTER TABLE stage_fingerprints ADD COLUMN settings_fp TEXT NOT NULL DEFAULT ''")


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    """Everything that used to run (guarded) on every connect, for databases
    created before it existed."""
    for table, column, ddl in (
        ("images", "standalone_caption", "TEXT NOT NULL DEFAULT ''"),
        ("images", "times_shown", "INTEGER NOT NULL DEFAULT 0"),
        ("images", "last_shown_at", "TEXT"),
        ("principles", "video_id", "TEXT NOT NULL DEFAULT ''"),
    ):
        if not _has_column(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_principles_video_id ON principles (video_id)")
    if _has_column(conn, "principles", "video_url"):
        # Rows from before video_id existed: recover it from the URL, then
        # split title/url out into the videos table (normalized 2026-09-20).
        conn.execute("""
            UPDATE principles
            SET video_id = substr(video_url, instr(video_url, 'v=') + 2, 11)
            WHERE video_id = '' AND video_url LIKE '%v=%'
        """)
        conn.execute("""
            INSERT OR IGNORE INTO videos (video_id, title, url, created_at)
            SELECT video_id, MIN(video_title), MIN(video_url), MIN(created_at)
            FROM principles WHERE video_id != '' GROUP BY video_id
        """)
        conn.execute("ALTER TABLE principles DROP COLUMN video_title")
        conn.execute("ALTER TABLE principles DROP COLUMN video_url")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    """The pipeline's stages were renamed (the source is now the audio)."""
    conn.execute("""UPDATE pending_retries SET stage = 'listen', items_json = '{}'
                    WHERE stage IN ('normalize', 'extract_principles', 'recall_check',
                                    'review_quality', 'correct_text')""")
    conn.execute("UPDATE pending_retries SET stage = 'edit' WHERE stage = 'judge'")


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    """Principles become ranges of a stored, verified document. The
    source_quote and trigger columns go: nothing reads them, and a second
    copy of the text only invites the two to disagree."""
    for column, ddl in (("doc_hash", "TEXT NOT NULL DEFAULT ''"),):
        if not _has_column(conn, "principles", column):
            conn.execute(f"ALTER TABLE principles ADD COLUMN {column} {ddl}")
    for gone in ("source_quote", "trigger"):
        if _has_column(conn, "principles", gone):
            conn.execute(f"ALTER TABLE principles DROP COLUMN {gone}")
    # In-flight retries from the old stage names restart under the new pipeline.
    conn.execute("""UPDATE pending_retries SET stage = 'normalize', items_json = '{}'
                    WHERE stage IN ('extract_principles', 'recall_check',
                                    'review_quality', 'correct_text')""")


def _norm(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def get_existing_principle_texts() -> set[str]:
    """Normalized-text set for dedup, same pattern as the language
    pipeline's get_existing_fronts()."""
    with _connect() as conn:
        rows = conn.execute("SELECT principle FROM principles").fetchall()
    return {_norm(r["principle"]) for r in rows}


def insert_principle(p: Principle) -> int:
    """Inserts a principle, with a normalized-text dedup check as a safety
    net (defense in depth): pipeline.py already checks
    get_existing_principle_texts before calling this, but insert_principle
    had no protection of its own -- any future caller that forgets that
    external check could silently duplicate data. If an exact
    (whitespace/case-normalized) match already exists, returns that
    existing row's id instead of inserting a duplicate."""
    norm = _norm(p.principle)
    with _connect() as conn:
        rows = conn.execute("SELECT id, principle FROM principles").fetchall()
        for r in rows:
            if _norm(r["principle"]) == norm:
                return r["id"]

        now = datetime.now().isoformat()
        if p.video_id:
            conn.execute(
                """INSERT INTO videos (video_id, title, url, created_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(video_id) DO NOTHING""",
                (p.video_id, p.video_title, p.video_url, now),
            )
        cur = conn.execute(
            """INSERT INTO principles
               (principle, domain, video_id, doc_hash, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                p.principle, p.domain, p.video_id, p.doc_hash, now,
            ),
        )
        return cur.lastrowid


def _row_to_principle(r: sqlite3.Row) -> Principle:
    """video_title/video_url come from a LEFT JOIN with videos (see each
    caller below) -- normalized out of the principles table itself
    2026-09-20. Guarded with keys-checks so this also tolerates being
    handed a bare principles row (no join) without raising a KeyError."""
    keys = r.keys()
    return Principle(
        id=r["id"], principle=r["principle"], domain=r["domain"],
        doc_hash=r["doc_hash"] if "doc_hash" in keys else "",
        video_id=r["video_id"] if "video_id" in keys else "",
        video_title=r["video_title"] if "video_title" in keys else "",
        video_url=r["video_url"] if "video_url" in keys else "",
        created_at=r["created_at"], last_shown_at=r["last_shown_at"],
        times_shown=r["times_shown"],
    )


def get_principles_for_video(video_id: str) -> list[Principle]:
    """All principles stored for one video, in insertion order (id ASC).
    Added 2026-09-29 for the image-matching retry path
    (pipeline.retry_supplementary)."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT p.*, v.title AS video_title, v.url AS video_url
               FROM principles p LEFT JOIN videos v ON p.video_id = v.video_id
               WHERE p.video_id = ? ORDER BY p.id ASC""",
            (video_id,),
        ).fetchall()
    return [_row_to_principle(r) for r in rows]


def pick_for_resurfacing(count: int) -> list[Principle]:
    """Selection strategy, revised after a strategic review of what
    actually raises the odds of internalization (not just recall):

    1. Priority: plain times_shown ascending (least-shown first), ties
       broken by longest-since-shown. Weight/stance-based priority
       weighting was removed 2026-09-07 along with those fields
       themselves (see the schema comment above) -- every principle now
       gets equal priority by design.
    2. Domain interleaving: candidates are pulled from a wider pool
       (5x count) ranked by the above, then greedily selected to avoid
       repeating the same domain back-to-back when a same-priority
       alternative from a different domain is available. This is a
       deliberate application of the interleaving effect from cognitive
       psychology (mixed-topic exposure builds more durable, better-
       connected memory than blocked same-topic repetition) -- it does not
       override the priority order, it only breaks ties in favor of
       variety.

    Still no full SM-2 scheduler -- overkill for passive resurfacing,
    where the point is priming/exposure, not graded recall.
    """
    with _connect() as conn:
        rows = conn.execute(
            """SELECT p.*, v.title AS video_title, v.url AS video_url,
                      p.times_shown * 1.0 AS priority
               FROM principles p
               LEFT JOIN videos v ON p.video_id = v.video_id
               ORDER BY priority ASC, COALESCE(p.last_shown_at, '') ASC
               LIMIT ?""",
            (count * 5,),
        ).fetchall()

    candidates = [_row_to_principle(r) for r in rows]
    if len(candidates) <= count:
        return candidates

    selected: list[Principle] = []
    used_domains: set[str] = set()
    remaining = list(candidates)

    while remaining and len(selected) < count:
        pick = None
        for c in remaining:
            if c.domain not in used_domains:
                pick = c
                break
        if pick is None:  # every remaining domain already used this round -- reset and allow repeats
            used_domains.clear()
            pick = remaining[0]
        selected.append(pick)
        used_domains.add(pick.domain)
        remaining.remove(pick)

    return selected


def mark_shown(principle_ids: list[int]) -> None:
    with _connect() as conn:
        conn.executemany(
            """UPDATE principles SET times_shown = times_shown + 1,
               last_shown_at = ? WHERE id = ?""",
            [(datetime.now().isoformat(), pid) for pid in principle_ids],
        )


def count_all() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM principles").fetchone()[0]


# --- Video-level tracking (for backlog processing across a whole channel) ---
# The ingest pipeline itself only dedups at the principle-text level (see
# get_existing_principle_texts), which is fine for one-off bookmarklet
# clicks but not enough for a 128-video backlog run: without this, an
# interrupted batch run has no way to know which videos it already covered
# versus which ones it still owes, and would have to either re-run
# everything (wasting Gemini quota re-extracting already-done videos) or
# track progress in some ad hoc file. This table is the source of truth for
# "is this video done" -- checked before spending any Gemini quota on it.

_VIDEO_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_videos (
    video_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    principles_added INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT NOT NULL,
    ok INTEGER NOT NULL DEFAULT 1,
    model_used TEXT NOT NULL DEFAULT ''
);
"""


def _ensure_video_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_VIDEO_SCHEMA)
    # Migration for databases created before `model_used` existed -- see the
    # `weight`/`stance`/`trigger`/`confidence` migrations above for why this
    # pattern (attempt + ignore duplicate-column error) is used everywhere
    # instead of a version table.
    try:
        conn.execute("ALTER TABLE processed_videos ADD COLUMN model_used TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # column already exists


def is_video_processed(video_id: str) -> bool:
    with _connect() as conn:
        _ensure_video_table(conn)
        row = conn.execute(
            "SELECT 1 FROM processed_videos WHERE video_id = ? AND ok = 1", (video_id,)
        ).fetchone()
    return row is not None


def mark_video_processed(video_id: str, url: str, title: str, principles_added: int, ok: bool = True, model_used: str = "") -> None:
    with _connect() as conn:
        _ensure_video_table(conn)
        conn.execute(
            """INSERT INTO processed_videos (video_id, url, title, principles_added, processed_at, ok, model_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 principles_added=excluded.principles_added,
                 processed_at=excluded.processed_at,
                 ok=excluded.ok,
                 model_used=excluded.model_used""",
            (video_id, url, title, principles_added, datetime.now().isoformat(), 1 if ok else 0, model_used),
        )


def count_processed_videos() -> int:
    with _connect() as conn:
        _ensure_video_table(conn)
        return conn.execute("SELECT COUNT(*) FROM processed_videos WHERE ok = 1").fetchone()[0]


@dataclass
class PendingRetry:
    id: int
    video_id: str
    url: str
    title: str
    stage: str
    items_json: str
    model_used: str
    attempts: int
    created_at: str
    last_attempt_at: str | None
    last_error: str


def save_pending_retry(video_id: str, url: str, title: str, stage: str, items_json: str, model_used: str, error: str,
                       count_attempt: bool = True) -> None:
    """Called by pipeline.py when a StageSkippedError aborts a video --
    upserts on video_id so re-failing the SAME video (e.g. it fails again
    at a later stage on the next retry attempt) updates the existing row
    instead of piling up duplicates. `count_attempt=False` records the new state without using
    up one of the limited retry attempts (used when no model was actually called)."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO pending_retries
               (video_id, url, title, stage, items_json, model_used, attempts, created_at, last_attempt_at, last_error)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 stage=excluded.stage,
                 items_json=excluded.items_json,
                 model_used=excluded.model_used,
                 attempts=attempts + ?,
                 last_attempt_at=excluded.last_attempt_at,
                 last_error=excluded.last_error""",
            (video_id, url, title, stage, items_json, model_used, datetime.now().isoformat(), datetime.now().isoformat(), error,
             1 if count_attempt else 0),
        )


RUNNING = "running"        # last_error of the queue row that says "a run is working on this video right now" (pipeline.run_stages)


def mark_run_started(video_id: str, url: str, title: str, stage: str, pid: int) -> bool:
    """A run begins. When the video has no queue row yet, one is written that says it is running (last_error = RUNNING, the process id
    in items_json, last_attempt_at as the run's heartbeat): if the process then dies (a restart, a crash, power loss) nothing else
    knows about the video, and this row is what lets scripts/retry_pending.py finish it. A video that already has a row (a retry) is
    left as it is. Returns True when this call wrote the row."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """INSERT OR IGNORE INTO pending_retries
               (video_id, url, title, stage, items_json, model_used, attempts, created_at, last_attempt_at, last_error)
               VALUES (?, ?, ?, ?, ?, '', 0, ?, ?, ?)""",
            (video_id, url, title, stage, json.dumps({"pid": pid}), now, now, RUNNING),
        )
        return cur.rowcount == 1


def touch_run(video_id: str) -> None:
    """The heartbeat of a running run: only a row that still says RUNNING is touched (a row the run has since turned into a failure
    or a retry is not)."""
    with _connect() as conn:
        conn.execute("UPDATE pending_retries SET last_attempt_at = ? WHERE video_id = ? AND last_error = ?",
                     (datetime.now().isoformat(), video_id, RUNNING))


def get_pending_retries() -> list[PendingRetry]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM pending_retries ORDER BY created_at ASC").fetchall()
    return [
        PendingRetry(
            id=r["id"], video_id=r["video_id"], url=r["url"], title=r["title"], stage=r["stage"],
            items_json=r["items_json"], model_used=r["model_used"], attempts=r["attempts"],
            created_at=r["created_at"], last_attempt_at=r["last_attempt_at"], last_error=r["last_error"],
        )
        for r in rows
    ]


def delete_pending_retry(retry_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM pending_retries WHERE id = ?", (retry_id,))


# --- Screenshot images (see images.py for the capture/enhance/describe flow) ---

def _to_stored_image_path(p: str) -> str:
    """Absolute path inside the project folder -> project-relative path with
    forward slashes, so the DB keeps working if the folder is moved (the
    stored absolute path would break after moving the project).
    Anything outside the project, or already relative, is stored as given."""
    path = Path(p)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return p


def resolve_image_path(stored: str) -> str:
    """Inverse of _to_stored_image_path: a stored path -> an absolute path
    under the current project folder. A legacy absolute path that no longer
    exists (folder moved) falls back to the same file name inside the current
    data/enhanced_images folder, so old rows heal themselves."""
    path = Path(stored)
    if not path.is_absolute():
        return str(_PROJECT_ROOT / path)
    if path.exists():
        return stored
    alt = _PROJECT_ROOT / "data" / "enhanced_images" / path.name
    return str(alt) if alt.exists() else stored


def insert_image(video_id: str, enhanced_path: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO images (video_id, enhanced_path, created_at) VALUES (?, ?, ?)",
            (video_id, _to_stored_image_path(enhanced_path), datetime.now().isoformat()),
        )
        return cur.lastrowid


def link_principle_image(principle_id: int, image_id: int) -> None:
    """Many-to-many by design (see the principle_images schema comment) --
    one image can back several principles and vice versa, decided by
    semantic content match, not timestamp (extractor.match_principles_to_images)."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO principle_images (principle_id, image_id) VALUES (?, ?)",
            (principle_id, image_id),
        )


def get_images_for_principle(principle_id: int) -> list[Image]:
    """Used by resurface.py so a randomly-picked principle can be sent back
    out WITH its supporting image(s), if any -- this is the payoff of the
    whole screenshot feature: the Dual-Coding benefit only matters if it
    reaches the long-running random resurfacing, not just the one-time
    post-video notification."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT im.* FROM images im
               JOIN principle_images pi ON im.id = pi.image_id
               WHERE pi.principle_id = ?""",
            (principle_id,),
        ).fetchall()
    return [_row_to_image(r) for r in rows]


def _row_to_image(r: sqlite3.Row) -> Image:
    keys = r.keys()
    return Image(
        id=r["id"], video_id=r["video_id"], enhanced_path=resolve_image_path(r["enhanced_path"]),
        created_at=r["created_at"],
        standalone_caption=r["standalone_caption"] if "standalone_caption" in keys else "",
        times_shown=r["times_shown"] if "times_shown" in keys else 0,
        last_shown_at=r["last_shown_at"] if "last_shown_at" in keys else None,
    )


def get_unlinked_images_for_video(video_id: str) -> list[Image]:
    """Images processed for this video that ended up matched to NO
    principle at all (see extractor.match_principles_to_images) -- used by
    pipeline.py right after matching to find which images need a
    standalone caption (see caption_standalone_image in extractor.py)."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM images
               WHERE video_id = ? AND id NOT IN (SELECT image_id FROM principle_images)""",
            (video_id,),
        ).fetchall()
    return [_row_to_image(r) for r in rows]


def set_image_caption(image_id: int, caption: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE images SET standalone_caption = ? WHERE id = ?", (caption, image_id))


def pick_standalone_images_for_resurfacing(count: int) -> list[Image]:
    """Mirrors pick_for_resurfacing's rotation logic (least-shown,
    longest-since-shown first) but for images that carry their OWN
    standalone caption and are NOT linked to any principle -- these are
    screenshots the user took that describe a fact/context (economic,
    historical, social) the speaker explained, but that didn't cleanly
    become a life-principle itself (see caption_standalone_image). Only
    images with a non-empty caption are eligible -- an uncaptioned image
    (captioning failed or hasn't run yet) is never surfaced."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT * FROM images
               WHERE standalone_caption != '' AND id NOT IN (SELECT image_id FROM principle_images)
               ORDER BY times_shown ASC, COALESCE(last_shown_at, '') ASC
               LIMIT ?""",
            (count,),
        ).fetchall()
    return [_row_to_image(r) for r in rows]


def mark_image_shown(image_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE images SET times_shown = times_shown + 1, last_shown_at = ? WHERE id = ?",
            (datetime.now().isoformat(), image_id),
        )


# --- Generic settings (currently just the screenshot capture checkpoint) ---

def get_setting(key: str, default: str | None = None) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


# --- Raw transcripts (see transcripts schema comment) ---

def insert_transcript(video_id: str, raw_text: str) -> None:
    """Idempotent by design (ON CONFLICT DO UPDATE) -- if the bookmarklet is
    ever clicked again for a video whose transcript changed (re-uploaded
    captions, etc.) this just refreshes it rather than erroring or silently
    keeping the stale copy. Called from pipeline.py right after
    fetch_transcript succeeds, before extraction, so the raw transcript is
    preserved even if a later step (Gemini quota, Telegram, etc.) fails."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO transcripts (video_id, raw_text, fetched_at)
               VALUES (?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 raw_text = excluded.raw_text,
                 fetched_at = excluded.fetched_at""",
            (video_id, raw_text, datetime.now().isoformat()),
        )


def get_transcript(video_id: str) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT raw_text FROM transcripts WHERE video_id = ?", (video_id,)
        ).fetchone()
    return row["raw_text"] if row else None


# ---- normalized documents ---------------------------------------------------
def save_document(video_id: str, doc_json: str, doc_hash: str) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO documents (video_id, doc_json, doc_hash, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 doc_json=excluded.doc_json, doc_hash=excluded.doc_hash,
                 created_at=excluded.created_at""",
            (video_id, doc_json, doc_hash, datetime.now().isoformat()),
        )


def get_document_json(video_id: str) -> str | None:
    with _connect() as conn:
        r = conn.execute("SELECT doc_json FROM documents WHERE video_id = ?", (video_id,)).fetchone()
    return r["doc_json"] if r else None


# ---- stage fingerprints, archive and backup (see stage_fingerprints.py) ----------
def save_stage_fingerprint(video_id: str, stage: str, prompt_fp: str, effective_fp: str,
                           doc_hash: str = "", log_file: str = "", settings_fp: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO stage_fingerprints (video_id, stage, prompt_fp, effective_fp, doc_hash, produced_at, log_file,
                                               settings_fp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(video_id, stage) DO UPDATE SET
                 prompt_fp=excluded.prompt_fp, effective_fp=excluded.effective_fp, doc_hash=excluded.doc_hash,
                 produced_at=excluded.produced_at, log_file=excluded.log_file, settings_fp=excluded.settings_fp""",
            (video_id, stage, prompt_fp, effective_fp, doc_hash, datetime.now().isoformat(), log_file, settings_fp),
        )


def get_stage_fingerprints(video_id: str) -> dict:
    """{stage: {"prompt_fp", "effective_fp", "doc_hash", "produced_at", "log_file", "settings_fp"}} recorded for the video
    (the principle stages and the post stages match / caption)."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM stage_fingerprints WHERE video_id = ?", (video_id,)).fetchall()
    return {r["stage"]: {k: r[k] for k in ("prompt_fp", "effective_fp", "doc_hash", "produced_at", "log_file",
                                           "settings_fp")}
            for r in rows}


def archive_post_results(video_id: str, stages, reason: str) -> dict:
    """Moves what the stages after the principles produced for a video to the archive tables before they are made again
    (nothing is deleted). `stages` is any of: "match" = the video's principle-image links -> principle_images_archive; "caption" = the standalone captions of the
    video's images -> images_history (the image keeps its id, its caption is cleared). One transaction.
    Returns {"image_links": n, "captions": n}."""
    now = datetime.now().isoformat()
    counts = {"image_links": 0, "captions": 0}
    with _connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM principles WHERE video_id = ?", (video_id,))]
        if ids and "match" in stages:
            marks = ",".join("?" * len(ids))
            counts["image_links"] = conn.execute(
                f"""INSERT INTO principle_images_archive (principle_id, image_id, archived_at, reason)
                    SELECT principle_id, image_id, ?, ? FROM principle_images WHERE principle_id IN ({marks})""",
                [now, reason] + ids).rowcount
            conn.execute(f"DELETE FROM principle_images WHERE principle_id IN ({marks})", ids)
        if "caption" in stages:
            captioned = conn.execute("SELECT * FROM images WHERE video_id = ? AND standalone_caption != ''",
                                     (video_id,)).fetchall()
            for old in captioned:
                conn.execute(
                    """INSERT INTO images_history (image_id, video_id, enhanced_path, standalone_caption, created_at, replaced_at, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (old["id"], old["video_id"], old["enhanced_path"], old["standalone_caption"], old["created_at"], now, reason))
                conn.execute("UPDATE images SET standalone_caption = '' WHERE id = ?", (old["id"],))
                counts["captions"] += 1
    return counts


def backup_database(dest_dir) -> Path:
    """A consistent copy of the live database (SQLite online backup, safe while other processes use it) at
    dest_dir/principles.db. Returns that path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "principles.db"
    src = _connect()
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return target


def archive_video_results(video_id: str, reason: str, archive_document: bool = False) -> dict:
    """Moves a video's stored results to the archive tables (nothing is deleted): its principles with their image links, and, when archive_document is set, its normalized document together with
    the transcript text. One transaction: either everything moves or nothing does. The images themselves, the
    processed mark and the stage fingerprints are left alone. Returns the counts moved."""
    now = datetime.now().isoformat()
    counts = {"principles": 0, "image_links": 0, "documents": 0}
    with _connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM principles WHERE video_id = ?", (video_id,))]
        if ids:
            marks = ",".join("?" * len(ids))
            counts["image_links"] = conn.execute(
                f"""INSERT INTO principle_images_archive (principle_id, image_id, archived_at, reason)
                    SELECT principle_id, image_id, ?, ? FROM principle_images WHERE principle_id IN ({marks})""",
                [now, reason] + ids).rowcount
            conn.execute(f"DELETE FROM principle_images WHERE principle_id IN ({marks})", ids)
            counts["principles"] = conn.execute(
                """INSERT INTO principles_archive (id, principle, domain, video_id, doc_hash,
                                                   created_at, last_shown_at, times_shown, archived_at, reason)
                   SELECT id, principle, domain, video_id, doc_hash,
                          created_at, last_shown_at, times_shown, ?, ? FROM principles WHERE video_id = ?""",
                (now, reason, video_id)).rowcount
            conn.execute("DELETE FROM principles WHERE video_id = ?", (video_id,))
        if archive_document:
            doc = conn.execute("SELECT doc_json, doc_hash, created_at FROM documents WHERE video_id = ?",
                               (video_id,)).fetchone()
            if doc is not None:
                tr = conn.execute("SELECT raw_text FROM transcripts WHERE video_id = ?", (video_id,)).fetchone()
                conn.execute(
                    """INSERT INTO documents_history (video_id, doc_json, doc_hash, created_at, transcript_text,
                                                      archived_at, reason) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (video_id, doc["doc_json"], doc["doc_hash"], doc["created_at"],
                     tr["raw_text"] if tr else None, now, reason))
                conn.execute("DELETE FROM documents WHERE video_id = ?", (video_id,))
                counts["documents"] = 1
    return counts


# ---- replacing an image in place (before 2026-11-10 also images described with an older prompt) --------
def get_image_by_path(video_id: str, path: str) -> Image | None:
    """The image of this video stored under that path (compared in the project-relative form the DB keeps; a legacy
    absolute row is matched by file name). None when there is none."""
    wanted = _to_stored_image_path(path)
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM images WHERE video_id = ? ORDER BY id", (video_id,)).fetchall()
    for r in rows:
        if r["enhanced_path"] == wanted:
            return _row_to_image(r)
    for r in rows:
        if Path(r["enhanced_path"]).name == Path(wanted).name:
            return _row_to_image(r)
    return None


def replace_image(image_id: int, enhanced_path: str, reason: str) -> dict:
    """Replaces the picture an image row holds, keeping its id (so its shown-history stays): the old row is copied to
    images_history, its links to principles are archived and removed (the new picture has to be matched again), and the
    standalone caption is cleared. One transaction. Returns {"links": number of links archived}."""
    now = datetime.now().isoformat()
    with _connect() as conn:
        old = conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        if old is None:
            raise KeyError(f"no image with id {image_id}")
        conn.execute(
            """INSERT INTO images_history (image_id, video_id, enhanced_path, standalone_caption, created_at, replaced_at, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (image_id, old["video_id"], old["enhanced_path"], old["standalone_caption"], old["created_at"], now, reason))
        links = conn.execute(
            """INSERT INTO principle_images_archive (principle_id, image_id, archived_at, reason)
               SELECT principle_id, image_id, ?, ? FROM principle_images WHERE image_id = ?""",
            (now, reason, image_id)).rowcount
        conn.execute("DELETE FROM principle_images WHERE image_id = ?", (image_id,))
        conn.execute(
            "UPDATE images SET enhanced_path = ?, standalone_caption = '' WHERE id = ?",
            (_to_stored_image_path(enhanced_path), image_id))
    return {"links": links}


def get_processed_video_title(video_id: str) -> str:
    with _connect() as conn:
        _ensure_video_table(conn)
        row = conn.execute("SELECT title FROM processed_videos WHERE video_id = ?", (video_id,)).fetchone()
    return row["title"] if row else ""
