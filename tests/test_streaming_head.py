"""The response head must be a property of the shim, not of GPU load.

Agents behind a transparent egress proxy (harbor's gost sidecar) get their
response head timed out after seconds, while reef legitimately stalls for
minutes — a weight publish pauses the engine mid-update. Measured on the
OpenClaw-RL stream, that cut one model call in four: the agent's client saw
a connection error, retried, exhausted, and the whole turn collapsed. A
streaming client asked for SSE, so the head carries nothing worth waiting
for — serve commits it immediately, and everything that can still go wrong
afterwards travels as an SSE error event the client parses instead of a
dead socket.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reef_client.serve import CaptureStore, ServeConfig, build_handler

UPSTREAM_DELAY_S = 2.0
HEAD_DEADLINE_S = 1.0  # far below the delay: the head must not wait upstream

COMPLETION = {
    "id": "c1",
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
}


class _SlowUpstream(BaseHTTPRequestHandler):
    """Answers a buffered completion, but only after a long think."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # quiet
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(UPSTREAM_DELAY_S)
        body = json.dumps(COMPLETION).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(upstream_url: str) -> tuple[ThreadingHTTPServer, CaptureStore]:
    store = CaptureStore()
    config = ServeConfig(upstream=upstream_url, listen_port=0)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(config, store))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    return proxy, store


def _stream_request(port: int) -> socket.socket:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    raw = socket.create_connection(("127.0.0.1", port))
    raw.sendall(
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body.encode()
    )
    return raw


def _read_all(raw: socket.socket, deadline_s: float = 10.0) -> bytes:
    raw.settimeout(deadline_s)
    data = b""
    try:
        while chunk := raw.recv(65536):
            data += chunk
            if data.endswith(b"0\r\n\r\n"):
                break
    except TimeoutError:
        pass
    return data


@pytest.fixture
def slow_stack():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _SlowUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy, store = _serve(f"http://127.0.0.1:{upstream.server_port}")
    yield proxy.server_port, store
    proxy.shutdown()
    proxy.server_close()
    upstream.shutdown()
    upstream.server_close()


def test_streaming_head_arrives_while_upstream_is_still_thinking(slow_stack):
    port, _store = slow_stack
    raw = _stream_request(port)

    # The head must land within the proxy-deadline budget, not the GPU's.
    raw.settimeout(HEAD_DEADLINE_S)
    started = time.monotonic()
    head = raw.recv(4096)
    elapsed = time.monotonic() - started
    assert head.startswith(b"HTTP/1.1 200"), head[:80]
    assert b"text/event-stream" in head
    assert elapsed < HEAD_DEADLINE_S

    # The buffered completion is then re-synthesized as SSE, verbatim content.
    rest = head + _read_all(raw)
    raw.close()
    assert b'"content": "hello"' in rest or b'"content":"hello"' in rest
    assert rest.endswith(b"0\r\n\r\n")


def test_an_unreachable_upstream_becomes_an_error_event_not_a_cut_socket():
    # A port nothing listens on: connection refused after the head went out.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    proxy, _store = _serve(f"http://127.0.0.1:{dead_port}")
    try:
        raw = _stream_request(proxy.server_port)
        data = _read_all(raw)
        raw.close()
        assert data.startswith(b"HTTP/1.1 200")  # the head was already committed
        assert b'"error"' in data and b"reef_client_serve" in data
        assert data.endswith(b"0\r\n\r\n")  # the stream still terminates cleanly
    finally:
        proxy.shutdown()
        proxy.server_close()


def test_a_buffered_client_still_gets_the_status_verbatim():
    # No stream requested: nothing is committed early, errors keep their status.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    proxy, _store = _serve(f"http://127.0.0.1:{dead_port}")
    try:
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        raw = socket.create_connection(("127.0.0.1", proxy.server_port))
        raw.sendall(
            f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body.encode()
        )
        raw.settimeout(5.0)
        data = raw.recv(65536)
        raw.close()
        assert data.startswith(b"HTTP/1.1 502"), data[:80]
    finally:
        proxy.shutdown()
        proxy.server_close()
