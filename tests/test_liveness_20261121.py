"""Audit 2026-11-21, P2: a listener that holds its port without answering is detected and restarted."""
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lifecompass import liveness


class _Ok(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(liveness, "STATE_PATH", tmp_path / "state.json")
    calls = {"launch": 0, "kill": [], "pids": [4242]}
    monkeypatch.setattr(liveness, "launch", lambda: calls.__setitem__("launch", calls["launch"] + 1))
    monkeypatch.setattr(liveness, "kill", lambda pid: calls["kill"].append(pid))
    monkeypatch.setattr(liveness, "listening_pids", lambda: calls["pids"])
    monkeypatch.setattr(liveness.time, "sleep", lambda s: None)
    return calls


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_down_when_nothing_listens(env, monkeypatch):
    monkeypatch.setattr(liveness, "PORT", _free_port())
    assert liveness.probe() == "down"
    assert liveness.ensure(now=1000.0) == "launched"
    assert env["launch"] == 1 and env["kill"] == []


def test_a_down_listener_is_not_launched_twice_within_the_grace_period(env, monkeypatch):
    monkeypatch.setattr(liveness, "PORT", _free_port())
    liveness.ensure(now=1000.0)
    liveness.ensure(now=1010.0)
    assert env["launch"] == 1


def test_up_when_it_answers(env, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(liveness, "PORT", server.server_address[1])
        assert liveness.probe() == "up"
        assert liveness.ensure(now=1000.0) == "up"
        assert env["launch"] == 0 and env["kill"] == []
    finally:
        server.shutdown()


def test_hung_listener_is_restarted_only_after_three_probes_in_a_row(env, monkeypatch):
    silent = socket.socket()                          # accepts connections (the backlog) and never answers: a stuck listener
    silent.bind(("127.0.0.1", 0))
    silent.listen(5)
    try:
        monkeypatch.setattr(liveness, "PORT", silent.getsockname()[1])
        monkeypatch.setattr(liveness, "probe", lambda timeout=0.3, _p=liveness.probe: _p(timeout))
        assert liveness.probe() == "hung"
        assert liveness.ensure(now=1000.0) == "hung"
        assert liveness.ensure(now=1300.0) == "hung"
        assert env["kill"] == [] and env["launch"] == 0
        # the third probe in a row: the process is stopped; once the port is free it is started again
        monkeypatch.setattr(liveness, "listening_pids", lambda: env["pids"])
        original = liveness.probe
        states = iter(["hung", "down"])
        monkeypatch.setattr(liveness, "probe", lambda timeout=5.0: next(states, "down"))
        assert liveness.ensure(now=1600.0) == "restarted"
        assert env["kill"] == [4242] and env["launch"] == 1
    finally:
        silent.close()


def test_one_answered_probe_resets_the_count(env, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(liveness, "PORT", server.server_address[1])
        liveness._write_state({"hung": 2})
        assert liveness.ensure(now=1000.0) == "up"
        assert liveness._read_state()["hung"] == 0
    finally:
        server.shutdown()
