"""The rerank HTTP client (module M6, contract C6).

One function talks to the endpoint: ``rerank`` sends the shortlist as one Cohere-shaped ``POST``
and returns one probability per document, in input order, or ``None`` on any failure.

Three rules here mirror the embeddings client, because both talk to endpoints the operator points
at:

* The request goes through :func:`hermes_skill_router.embed.opener`, the one opener whose redirect
  handler refuses to build a new request. A 301, 302, 303, 307 or 308 therefore fails the call
  instead of sending the query and the skill names to whatever host ``Location`` names (D4). This
  module reuses that opener and the ``HealthGate`` class; it never builds a second one.
* A failure marks this endpoint in this module's own gate, so a stopped reranker is probed at most
  once per ``health_check_ms`` instead of paying the full ``rerank_timeout_ms`` on every request.
  The flag is keyed by the full endpoint URL, so one broken endpoint never silences another.
* The body is used only when every input document has exactly one result: an ``index`` in range,
  used once, and a finite ``relevance_score``. A short, long, duplicated or malformed result set is
  a failure, never a padded or partly defaulted list, and nothing is retried.

Scores are used as the endpoint returned them. It already answers with a probability over the
candidates, and the selector compares that value against the cutoff directly (D10); normalising it
again would turn one confident skill into a page of suggestions. The document text is the caller's:
``cfg.document_prefix + name + " - " + description``, the DSH format the shared service parses.

The client never raises: connection errors, timeouts, HTTP errors including every 3xx and unusable
bodies all return ``None``, which the rewrite turns into an unchanged request (R5).
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

from .config import Config
from .embed import HealthGate, opener

logger = logging.getLogger(__name__)

#: The only content type this client sends.
CONTENT_TYPE = "application/json"

#: The rerank endpoint's health, shared by every request in this process.
HEALTH = HealthGate()


def rerank(cfg: Config, query: str, documents: Sequence[str]) -> Optional[List[float]]:
    """Score every document against *query*, returning one probability per document in input order.

    *documents* arrive already formatted as ``cfg.document_prefix + name + " - " + description``;
    they are sent verbatim in one request. Returns ``None`` on a connection error, any HTTP error
    including a 3xx, a timeout, or a body that does not carry exactly one usable score per document.
    Never raises and never retries.
    """
    if not documents:
        return []
    endpoint = cfg.rerank_base_url + cfg.rerank_path
    if not HEALTH.ready(endpoint, cfg.health_check_ms):
        logger.debug("rerank endpoint %s is inside its health window; not calling it", endpoint)
        return None
    body = json.dumps(
        {
            "model": cfg.rerank_model,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": CONTENT_TYPE},
        method="POST",
    )
    try:
        with opener().open(request, timeout=cfg.rerank_timeout_ms / 1000.0) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        HEALTH.failure(endpoint)
        logger.debug("rerank endpoint %s answered HTTP %s", endpoint, exc.code)
        return None
    except Exception as exc:  # URLError, socket timeout, reset connection, incomplete read
        HEALTH.failure(endpoint)
        logger.debug("rerank endpoint %s failed: %r", endpoint, exc)
        return None
    scores = _scores(raw, len(documents))
    if scores is None:
        HEALTH.failure(endpoint)
        logger.debug("rerank endpoint %s returned an unusable body", endpoint)
        return None
    HEALTH.success(endpoint)
    return scores


def _scores(raw: bytes, expected: int) -> Optional[List[float]]:
    """Parse a response body into one score per document, or ``None`` when it is unusable."""
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    results = body.get("results")
    if not isinstance(results, list) or len(results) != expected:
        return None
    scores: List[Optional[float]] = [None] * expected
    for entry in results:
        if not isinstance(entry, dict):
            return None
        index = entry.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if not 0 <= index < expected:
            return None
        if scores[index] is not None:
            # Two results for one document would leave another document without a score, and the
            # length check above cannot see that, so a duplicate index fails the whole body.
            return None
        value = entry.get("relevance_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        score = float(value)
        if not math.isfinite(score):
            return None
        scores[index] = score
    if any(score is None for score in scores):
        return None
    return [float(score) for score in scores if score is not None]
