"""A client that walks away must not cost the turn that already happened.

Serve mode exists so a reef-oblivious agent needs no SDK; the price of that
is that the agent is also free to time out, be killed, or stop reading
mid-stream. When it does, the exchange upstream has already completed and
its capture is the whole reason the sidecar was in the path — so a dead
client socket may not take the capture down with it.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from reef_client.serve import CaptureStore, ServeConfig, build_handler
from reef_client.sse import SSEAccumulator

CHUNKS = [
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"he"}}]}\n\n',
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"llo"}}]}\n\n',
    b"data: [DONE]\n\n",
]


class _Upstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # quiet
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-reef-agent-data-id", "receipt-1")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in CHUNKS:
            self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def stack():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    store = CaptureStore()
    config = ServeConfig(upstream=f"http://127.0.0.1:{upstream.server_port}", listen_port=0)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(config, store))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{proxy.server_port}", store
    proxy.shutdown()
    proxy.server_close()
    upstream.shutdown()
    upstream.server_close()


def test_a_client_that_hangs_up_mid_stream_still_leaves_its_turn_captured(stack):
    url, store = stack
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True})

    # Send the request, read nothing, and drop the socket — the agent
    # equivalent of a timeout killing hermes mid-generation.
    raw = socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1])))
    raw.sendall(
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode()
        + body.encode()
    )
    raw.close()

    deadline = threading.Event()
    for _ in range(200):
        if store.snapshot():
            break
        deadline.wait(0.05)

    turns = store.snapshot()
    assert turns, "the disconnect cost the capture"
    assert turns[0]["receipt"] == "receipt-1"
    assert turns[0]["response"]["choices"][0]["message"]["content"] == "hello"


def test_a_reading_client_is_unaffected(stack):
    url, store = stack
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({"model": "m", "messages": [], "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        streamed = response.read()
    assert b"[DONE]" in streamed and b'"he"' in streamed  # relayed frame by frame
    assert store.snapshot()[0]["response"]["choices"][0]["message"]["content"] == "hello"


def test_a_malformed_frame_costs_its_own_content_not_the_stream():
    accumulator = SSEAccumulator()
    accumulator.feed(b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"he"}}]}\n\n')
    accumulator.feed(b"data: {not json\n\n")
    accumulator.feed(b'data: {"choices":[{"index":0,"delta":{"content":"llo"}}]}\n\n')
    accumulator.feed(b"data: [DONE]\n\n")
    accumulator.finish()
    assert accumulator.completion["choices"][0]["message"]["content"] == "hello"
