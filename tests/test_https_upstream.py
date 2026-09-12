"""An https upstream is reached over TLS, not over a plain connection to its host.

The API platform serves reef behind https. A plain connection to that host is
answered with a redirect to https and no body, which serve relayed as an empty
completion; the agent then reported an empty upstream response for every call.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from http.server import ThreadingHTTPServer

import pytest

from reef_client.serve import CaptureStore, ServeConfig, build_handler


def _serve(upstream_url: str) -> ThreadingHTTPServer:
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(ServeConfig(upstream=upstream_url, listen_port=0), CaptureStore()))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    return proxy


def _post(port: int) -> bytes:
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    raw = socket.create_connection(("127.0.0.1", port))
    raw.sendall(
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body.encode()
    )
    raw.settimeout(10)
    chunks = []
    while True:
        try:
            chunk = raw.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    raw.close()
    return b"".join(chunks)


def test_an_https_upstream_opens_a_tls_connection(monkeypatch):
    """The connection class follows the upstream's scheme: https opens HTTPSConnection at the upstream's host."""
    opened: list[tuple[str, tuple]] = []

    class _Recorder:
        def __init__(self, host, port=None, timeout=None, **_kwargs):
            opened.append(("https", (host, port)))

        def request(self, *args, **kwargs):
            raise OSError("no network in this test")

    monkeypatch.setattr(http.client, "HTTPSConnection", _Recorder)
    proxy = _serve("https://api.example.test")
    try:
        answer = _post(proxy.server_address[1])
    finally:
        proxy.shutdown()
    assert opened == [("https", ("api.example.test", None))]
    assert answer.startswith(b"HTTP/1.1 502")


def test_an_http_upstream_keeps_a_plain_connection(monkeypatch):
    opened: list[str] = []

    class _Recorder:
        def __init__(self, host, port=None, timeout=None, **_kwargs):
            opened.append("http")

        def request(self, *args, **kwargs):
            raise OSError("no network in this test")

    monkeypatch.setattr(http.client, "HTTPConnection", _Recorder)
    proxy = _serve("http://127.0.0.1:1")
    try:
        _post(proxy.server_address[1])
    finally:
        proxy.shutdown()
    assert opened == ["http"]


def test_a_config_refuses_an_upstream_without_a_scheme():
    with pytest.raises(ValueError, match="http or https"):
        ServeConfig(upstream="api.example.test", listen_port=0)
