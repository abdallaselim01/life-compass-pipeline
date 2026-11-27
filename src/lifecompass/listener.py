#!/usr/bin/env python3
"""
Local HTTP listener for the one-click bookmarklet trigger. Same pattern as
the language pipeline's listener.py: click a bookmark after finishing a
video, it POSTs the current URL here, a background thread runs the
ingest pipeline so the click returns instantly.
"""
from __future__ import annotations

import html
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# The project's .env is read BEFORE the modules below are imported: llm.py reads GEMINI_USE_STRUCTURED_OUTPUT when it is imported, and
# override=True makes the project's own keys win over a same-named variable set elsewhere in Windows (audit 2026-11-21).
load_dotenv(override=True)

from .pipeline import process_video
from .audio import extract_video_id
from .logging_setup import get_logger

file_log = get_logger("listener")

PORT = 8935  # different port than the language pipeline's listener (8934)
             # so both can run simultaneously without conflict

# In-memory guard against duplicate concurrent processing of the same video
# -- e.g. an accidental double-click on the bookmarklet before the first
# click's background thread has finished and recorded the video as done in
# store.processed_videos. Without this, both requests pass the
# is_video_processed check (neither has finished yet) and run a full,
# separate extraction -- silently doubling Gemini quota spend for nothing.
_in_progress: set[str] = set()
_in_progress_lock = threading.Lock()


# Who may start a video (audit 2026-11-21). The listener only answers this computer, but any web page the browser has open can still
# send it a request, and a page of another site could spend the quota on videos nobody chose. The bookmarklet runs inside a YouTube page,
# so a request that names an Origin or a Referer must name YouTube. A request that names neither (a plain call, or a browser that
# strips both) is let through: it cannot be told apart from the bookmarklet, and refusing it could break the one way a video is started.
ALLOWED_SITES = ("youtube.com", "youtu.be")


def _site_allowed(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return any(host == site or host.endswith("." + site) for site in ALLOWED_SITES)


class Handler(BaseHTTPRequestHandler):
    def _foreign(self) -> str | None:
        """The Origin or Referer that is not YouTube (also the literal Origin 'null'), or None when the request is fine."""
        for name in ("Origin", "Referer"):
            value = self.headers.get(name)
            if value is not None and (value.strip().lower() == "null" or not _site_allowed(value)):
                return f"{name}={value}"
        return None

    def _cors_origin(self) -> str | None:
        """The calling page's own origin, when it is YouTube: the only origin that is told it may read the answer."""
        origin = self.headers.get("Origin")
        return origin if origin and origin.strip().lower() != "null" and _site_allowed(origin) else None

    def _send(self, status: int, body: str):
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        if (origin := self._cors_origin()):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        if (origin := self._cors_origin()):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        file_log.debug(f"GET {self.path} from {self.client_address}")
        if parsed.path != "/process":
            file_log.warning(f"404 for path={parsed.path}")
            self._send(404, "not found")
            return

        foreign = self._foreign()
        if foreign:
            file_log.warning(f"403: request not from YouTube ({foreign}) for {self.path}")
            self._send(403, "forbidden")
            return

        qs = parse_qs(parsed.query)
        url = (qs.get("url") or [""])[0]
        if not url:
            file_log.warning("400: missing ?url=")
            self._send(400, "missing ?url=")
            return

        try:
            vid = extract_video_id(url)
        except ValueError:
            vid = None

        if vid:
            with _in_progress_lock:
                if vid in _in_progress:
                    file_log.warning(f"Rejected duplicate concurrent request for video_id={vid}")
                    self._send(200, f"Already processing this video (from an earlier click) -- ignoring duplicate.\n{url}")
                    return
                _in_progress.add(vid)

        file_log.info(f"Accepted request, dispatching background thread for url={url} "
                      f"(Origin={self.headers.get('Origin')}, Referer={self.headers.get('Referer')}, "
                      f"Sec-Fetch-Site={self.headers.get('Sec-Fetch-Site')})")
        trigger = f"GET {self.path} from {self.client_address} at {datetime.now().isoformat(timespec='milliseconds')}"
        self._send(200, f"Started processing:\n{url}\n\nCheck listener.log for progress.")

        def run():
            print(f"\n{'=' * 60}")
            print(f"Triggered via bookmarklet: {url}")
            print("=" * 60)
            try:
                result = process_video(url, trigger_info=trigger)
                print(f"\n[{'OK' if result.ok else 'FAILED'}] {result.message}\n")
            except Exception as e:
                file_log.exception(f"Unhandled exception while processing url={url}")
                print("\n[FAILED] Unhandled exception -- see logs/lifecompass.log\n")
                # Final safety net: this catches bugs the pipeline itself
                # never anticipated (e.g. a code-level error, not a model
                # failure) -- added 2026-09-11 after exactly that happened
                # silently (a str.format() crash in a prompt file, before
                # any of pipeline.py's own error handling could even run)
                # and produced no notification at all, for 25+ minutes.
                # Deliberately best-effort: if Telegram itself is also
                # down, this should not raise a SECOND unhandled exception
                # inside the exception handler.
                try:
                    from .publisher import send_message
                    send_message(
                        f"⚠️ <b>Unexpected system error</b>\nLink: {html.escape(url, quote=False)}\nReason: {html.escape(type(e).__name__ + ': ' + str(e), quote=False)}",
                        log=print,
                    )
                except Exception:
                    file_log.exception("Failed to send the unhandled-exception Telegram alert itself")
            finally:
                if vid:
                    with _in_progress_lock:
                        _in_progress.discard(vid)

        threading.Thread(target=run, daemon=True).start()

    def log_message(self, format, *args):
        pass


def main():
    load_dotenv(override=True)
    file_log.info(f"listener starting on port {PORT}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Listening on http://127.0.0.1:{PORT} - click the bookmarklet on a finished video.")
    print("Leave this window open. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        file_log.info("listener stopped via KeyboardInterrupt")
    except Exception:
        file_log.exception("listener crashed")
        raise


if __name__ == "__main__":
    main()
