"""A loopback-only HTTP fake the client tests drive (created by T3, reused by T4).

Every client test in this repository talks to this server instead of a real endpoint, so the suite
needs no model, no GPU and no network. It binds ``127.0.0.1`` on an ephemeral port and never any
other address, records each request it receives, and answers with whatever the test set: a body, an
error status, a redirect, or a slow response that outlives the client's timeout.

The answer is switchable at any time through :meth:`FakeEndpoint.answer`, or computed per request
by passing a ``responder`` to the constructor. ``requests`` and ``count`` are what prove that a
refused redirect sent no follow-up request and that a cache hit opened no socket.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

#: The only address the fake ever binds. A wildcard bind would expose the test server to the LAN.
BIND_HOST = "127.0.0.1"


@dataclass(frozen=True)
class Request:
    """One request the fake received."""

    method: str
    path: str
    headers: Dict[str, str]
    body: bytes

    def json(self) -> Any:
        """Return the request body decoded as JSON."""
        return json.loads(self.body.decode("utf-8"))

    def header(self, name: str) -> Optional[str]:
        """Return a header value by case-insensitive name, or ``None``."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if key.lower() == wanted:
                return value
        return None


@dataclass
class Response:
    """What the fake answers next.

    ``body`` may be ``None`` for an empty body, ``bytes`` or ``str`` to be sent as-is, or any other
    value to be JSON-encoded with a ``Content-Type`` header. ``delay`` holds the request open
    before answering, which is how a client timeout is exercised without a real slow service.
    """

    status: int = 200
    body: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    delay: float = 0.0


class _Server(ThreadingHTTPServer):
    """One server per fake, carrying the fake so the handler can reach it."""

    daemon_threads = True
    allow_reuse_address = True

    endpoint: "FakeEndpoint"


class _Handler(BaseHTTPRequestHandler):
    """Record the request, then write the endpoint's current answer."""

    # HTTP/1.0 closes the connection after each response, so the tests never share a keep-alive
    # socket between a refused redirect and the assertion that no second request arrived.
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._handle()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self._handle()

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        request = Request(self.command, self.path, dict(self.headers.items()), body)
        endpoint = self.server.endpoint  # type: ignore[attr-defined]
        endpoint._record(request)
        answer = endpoint._answer_for(request)
        if answer.delay:
            time.sleep(answer.delay)
        payload = answer.body
        headers = dict(answer.headers)
        if payload is None:
            raw = b""
        elif isinstance(payload, (bytes, bytearray)):
            raw = bytes(payload)
        elif isinstance(payload, str):
            raw = payload.encode("utf-8")
        else:
            raw = json.dumps(payload).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        try:
            self.send_response(answer.status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            if raw:
                self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            # The client gave up first, which is exactly what the timeout test does.
            pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
        """Silence the default stderr line; the fake records requests itself."""


class FakeEndpoint:
    """A ``http.server`` thread on loopback whose answer the test switches."""

    def __init__(self, responder: Optional[Callable[[Request], Response]] = None) -> None:
        self.responder = responder
        self.response = Response()
        self.requests: List[Request] = []
        self._lock = threading.Lock()
        self._server = _Server((BIND_HOST, 0), _Handler)
        self._server.endpoint = self
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="fake-endpoint",
            daemon=True,
        )
        self._thread.start()

    # -- address -----------------------------------------------------------------------------

    @property
    def address(self) -> Tuple[str, int]:
        """The address the server is bound to, so a test can prove it is loopback only."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    @property
    def host(self) -> str:
        return self.address[0]

    @property
    def port(self) -> int:
        return self.address[1]

    @property
    def base_url(self) -> str:
        """The endpoint root a client config points at."""
        return f"http://{self.host}:{self.port}"

    # -- behaviour ---------------------------------------------------------------------------

    def answer(
        self,
        body: Any = None,
        *,
        status: int = 200,
        headers: Optional[Mapping[str, str]] = None,
        delay: float = 0.0,
    ) -> None:
        """Answer every following request with *body* and *status*."""
        self.response = Response(status=status, body=body, headers=dict(headers or {}), delay=delay)

    def answer_with(self, responder: Optional[Callable[[Request], Response]]) -> None:
        """Answer each request through *responder* instead of the canned response."""
        self.responder = responder

    # -- observations ------------------------------------------------------------------------

    @property
    def count(self) -> int:
        """How many requests the fake has received."""
        with self._lock:
            return len(self.requests)

    def clear(self) -> None:
        """Forget the requests received so far."""
        with self._lock:
            self.requests.clear()

    def _record(self, request: Request) -> None:
        with self._lock:
            self.requests.append(request)

    def _answer_for(self, request: Request) -> Response:
        responder = self.responder
        return self.response if responder is None else responder(request)

    # -- lifecycle ---------------------------------------------------------------------------

    def close(self) -> None:
        """Stop the server and release its port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def __enter__(self) -> "FakeEndpoint":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
