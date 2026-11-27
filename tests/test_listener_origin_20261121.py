"""Audit 2026-11-21, P5: only YouTube (or a caller that names no site at all) can start a video."""
import http.client
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lifecompass import listener

VIDEO = "/process?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DdQw4w9WgXcQ"


@pytest.fixture
def server(monkeypatch):
    started = []
    monkeypatch.setattr(listener, "process_video", lambda url, trigger_info=None: started.append(url) or SimpleNamespace(ok=True, message="m"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), listener.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], started
    srv.shutdown()
    listener._in_progress.clear()


def call(port, method="GET", path=VIDEO, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, headers=headers or {})
    r = conn.getresponse()
    r.read()
    conn.close()
    return r.status, {k.lower(): v for k, v in r.getheaders()}


@pytest.mark.parametrize("headers", [
    {},                                                                       # a plain call, or a browser that strips both headers
    {"Origin": "https://www.youtube.com"},                                    # the bookmarklet, as a CORS request
    {"Origin": "https://m.youtube.com", "Referer": "https://m.youtube.com/watch?v=x"},
    {"Referer": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},               # as a no-cors request or a navigation
    {"Referer": "https://youtu.be/dQw4w9WgXcQ"},
])
def test_youtube_and_unlabelled_callers_start_the_video(server, headers):
    port, started = server
    status, _ = call(port, headers=headers)
    assert status == 200
    deadline = time.time() + 3
    while not started and time.time() < deadline:
        time.sleep(0.02)
    assert started


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"},
    {"Referer": "https://evil.example/page"},
    {"Origin": "null"},
    {"Origin": "https://www.youtube.com.evil.example"},                       # a look-alike host
    {"Origin": "https://notyoutube.com"},
    {"Origin": "https://www.youtube.com", "Referer": "https://evil.example/"},
])
def test_any_other_site_is_refused_and_nothing_starts(server, headers):
    port, started = server
    status, _ = call(port, headers=headers)
    time.sleep(0.2)
    assert status == 403 and started == []


def test_only_youtube_may_read_the_answer(server):
    port, _ = server
    _, h = call(port, headers={"Origin": "https://www.youtube.com"})
    assert h.get("access-control-allow-origin") == "https://www.youtube.com"
    _, h = call(port, path="/process")                                         # no Origin: no CORS header at all (and never "*")
    assert "access-control-allow-origin" not in h
    status, h = call(port, "OPTIONS", "/", {"Origin": "https://evil.example"})
    assert status == 204 and "access-control-allow-origin" not in h
    status, h = call(port, "OPTIONS", "/", {"Origin": "https://www.youtube.com"})
    assert status == 204 and h.get("access-control-allow-origin") == "https://www.youtube.com"


def test_the_watchdog_probe_still_works(server):
    port, _ = server
    from lifecompass import liveness
    liveness.PORT = port
    try:
        assert liveness.probe() == "up"
    finally:
        liveness.PORT = 8935
