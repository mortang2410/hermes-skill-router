"""The embeddings HTTP client (module M5, contract C5).

One function talks to the endpoint: ``embed_texts`` sends a batch of texts as one OpenAI-shaped
``POST`` and returns one vector per text, in input order, or ``None`` on any failure.

Three rules here exist because of what a wrong request would cost:

* Every request goes through one opener whose redirect handler refuses to build a new request.
  The default opener follows a 301, 302 or 303 on a ``POST`` to whatever host ``Location`` names,
  which would send the user's text there (D4). The handler returns ``None``, and urllib then raises
  the 3xx as an ``HTTPError``, so the call fails and no follow-up request leaves the process.
  ``opener`` is public because the rerank client (M6) reuses it rather than keeping a second
  redirect policy.
* A failure marks its endpoint, so a later call inside ``cfg.health_check_ms`` returns ``None``
  without opening a socket and a stopped service costs no per-request timeout. The first call
  after the window probes again. The flag is keyed by the full endpoint URL, so one broken
  endpoint never silences another.
* The response is used only when it carries exactly one non-empty ``embedding`` list per input, at
  one common width. A short, long, ragged or malformed body is a failure, never a partial result,
  and nothing is retried.

The client never raises: connection errors, timeouts, HTTP errors including every 3xx and unusable
bodies all return ``None``, which the rewrite turns into an unchanged request (R5).
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from time import monotonic
from typing import List, Optional, Sequence

from .config import Config

logger = logging.getLogger(__name__)

#: The only content type this client sends.
CONTENT_TYPE = "application/json"

_OPENER_LOCK = threading.Lock()
_OPENER: Optional[urllib.request.OpenerDirector] = None


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect, whatever the status or the target host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - see class
        # Returning None means "make no new request"; urllib then raises the original 3xx as an
        # HTTPError, so the caller fails instead of sending the same body to ``newurl``.
        logger.debug("endpoint answered %s for %s; refusing to follow it", code, req.full_url)
        return None


def opener() -> urllib.request.OpenerDirector:
    """Return the process-wide opener that refuses redirects.

    ``build_opener`` with this handler replaces the default redirect policy, so a 3xx raises
    instead of being followed. The rerank client (M6) reuses this opener, so both clients share
    one redirect policy rather than two copies of it.
    """
    global _OPENER
    with _OPENER_LOCK:
        if _OPENER is None:
            _OPENER = urllib.request.build_opener(_RefuseRedirect())
        return _OPENER


class HealthGate:
    """Remember the last failure per endpoint URL and stay closed for the configured window.

    ``health_check_ms`` is the minimum gap between probes of an endpoint that failed, so a stopped
    service is probed at most once per window instead of once per request. A success clears the
    entry; a window of ``0`` allows every call through, which is what the config bound means.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failed_at: dict[str, float] = {}

    def ready(self, endpoint: str, window_ms: int) -> bool:
        """Return whether *endpoint* may be called now."""
        with self._lock:
            failed_at = self._failed_at.get(endpoint)
        if failed_at is None:
            return True
        return (monotonic() - failed_at) * 1000.0 >= window_ms

    def failure(self, endpoint: str) -> None:
        """Record that *endpoint* just failed, starting its window."""
        with self._lock:
            self._failed_at[endpoint] = monotonic()

    def success(self, endpoint: str) -> None:
        """Forget a recorded failure, so the next call goes straight through."""
        with self._lock:
            self._failed_at.pop(endpoint, None)

    def reset(self) -> None:
        """Forget every recorded failure."""
        with self._lock:
            self._failed_at.clear()


#: The embeddings endpoint's health, shared by every request in this process.
HEALTH = HealthGate()


def embed_texts(cfg: Config, texts: Sequence[str]) -> Optional[List[List[float]]]:
    """Embed every text in one request, returning one vector per text in input order.

    Returns ``None`` on a connection error, any HTTP error including a 3xx, a timeout, or a body
    that does not carry exactly one usable vector per input. Never raises and never retries.
    """
    if not texts:
        return []
    endpoint = cfg.embed_base_url + cfg.embed_path
    if not HEALTH.ready(endpoint, cfg.health_check_ms):
        logger.debug("embeddings endpoint %s is inside its health window; not calling it", endpoint)
        return None
    body = json.dumps({"model": cfg.embed_model_id, "input": list(texts)}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": CONTENT_TYPE},
        method="POST",
    )
    try:
        with opener().open(request, timeout=cfg.embed_timeout_ms / 1000.0) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        HEALTH.failure(endpoint)
        logger.debug("embeddings endpoint %s answered HTTP %s", endpoint, exc.code)
        return None
    except Exception as exc:  # URLError, socket timeout, reset connection, incomplete read
        HEALTH.failure(endpoint)
        logger.debug("embeddings endpoint %s failed: %r", endpoint, exc)
        return None
    vectors = _vectors(raw, len(texts))
    if vectors is None:
        HEALTH.failure(endpoint)
        logger.debug("embeddings endpoint %s returned an unusable body", endpoint)
        return None
    HEALTH.success(endpoint)
    return vectors


def _vectors(raw: bytes, expected: int) -> Optional[List[List[float]]]:
    """Parse a response body into one vector per input, or ``None`` when it is unusable."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected:
        return None
    vectors: List[List[float]] = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        vector = entry.get("embedding")
        if not isinstance(vector, list) or not vector:
            return None
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector
        ):
            return None
        vectors.append([float(value) for value in vector])
    if len({len(vector) for vector in vectors}) != 1:
        # Mixed widths cannot be scored against one index; refuse rather than rank with them.
        return None
    return vectors
