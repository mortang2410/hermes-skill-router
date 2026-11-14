"""Whether both configured endpoints are ready (module for contract C2).

The setup check answers one question: does each configured endpoint answer the
plugin's real request shape? A custom endpoint needs only one parseable 2xx
answer to the live request at the configured path. A bundled endpoint (its base
URL equals the plugin default) must first answer ``GET <base>/health`` with
``status == "ok"`` and the configured model, then answer the same live
request. Never raises; unready is a value.

The interactive setup script and the request rewrite share this module, so the
setup-needed line the rewrite writes through the display record names the same
failing setting the script would probe, and both use the configured request
paths the live clients use rather than hardcoded ones.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from time import monotonic
from typing import Any, Mapping, Optional

from .config import DEFAULTS, Config

#: The base URLs whose servers the repository bundles. A config pointing at one
#: of these is checked with its health route and its model id; anything else
#: is a custom endpoint and only owes a correct live answer.
BUNDLED_BASES = {
    "embed": DEFAULTS["embedBaseUrl"],
    "rerank": DEFAULTS["rerankBaseUrl"],
}


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect, so a probe never posts to a host it was not asked about."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - see class
        return None


#: One opener for every probe; the bundled clients use the same redirect policy.
_OPENER = urllib.request.build_opener(_RefuseRedirect())


def _is_timeout(exc: BaseException) -> bool:
    """Whether *exc* is a socket timeout, so a capped probe can tell a slow answer from a refusal."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    return isinstance(exc, urllib.error.URLError) and isinstance(
        getattr(exc, "reason", None), (TimeoutError, socket.timeout)
    )


def _json_request(
    url: str, *, body: Optional[Mapping[str, Any]], timeout: float
) -> tuple[bool, Any, bool]:
    """Answer ``(True, decoded_json, False)`` for a parseable 2xx, else ``(False, None, timed_out)``.

    Never raises: a closed port, a timeout, any non-2xx status (including a refused redirect),
    and a body that is not JSON are all unready values. ``timed_out`` separates a probe that
    hit its timeout from one an endpoint definitely answered, so the deadline path can keep a
    slow-but-healthy endpoint indeterminate instead of blaming it.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    method = "GET" if body is None else "POST"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read()
    except Exception as exc:  # noqa: BLE001 - a probe failure is a readiness value, not an exception
        return False, None, _is_timeout(exc)
    try:
        return True, json.loads(raw), False
    except ValueError:
        return False, None, False


def _probe_timeout(timeout: float, deadline: Optional[float]) -> Optional[float]:
    """One probe's timeout capped by the routing deadline, or ``None`` when it is spent."""
    if deadline is None:
        return timeout
    left = deadline - monotonic()
    if left <= 0:
        return None
    return min(timeout, left)


def _hit_deadline(timeout: float, current: float, timed_out: bool, deadline: Optional[float]) -> bool:
    """Whether a failed probe timed out only because the routing deadline capped it.

    A timeout at the full configured timeout is a definitive refusal; a timeout at a capped
    timeout (or one that ends at the deadline) says the endpoint was too slow for the budget,
    not that it is down, so it must not name the endpoint as failing.
    """
    if not timed_out or deadline is None:
        return False
    return current < timeout or monotonic() >= deadline


def _endpoint_ready(
    base: str,
    model: str,
    path: str,
    payload: Mapping[str, Any],
    timeout: float,
    bundled_base: str,
    *,
    deadline: Optional[float] = None,
) -> Optional[bool]:
    """Whether *base* is ready: ``True`` ready, ``False`` definitively not, ``None`` indeterminate.

    ``None`` means a required probe was skipped or timed out under the routing deadline: the
    endpoint never answered negatively, so no failing setting can be named from it. Without a
    deadline this never returns ``None``.
    """
    root = base.rstrip("/")
    if root == bundled_base.rstrip("/"):
        current = _probe_timeout(timeout, deadline)
        if current is None:
            return None
        healthy, health, timed_out = _json_request(root + "/health", body=None, timeout=current)
        if not healthy or not isinstance(health, dict):
            if _hit_deadline(timeout, current, timed_out, deadline):
                return None
            return False
        if health.get("status") != "ok" or health.get("model") != model:
            return False
    current = _probe_timeout(timeout, deadline)
    if current is None:
        return None
    answered, _body, timed_out = _json_request(root + path, body=payload, timeout=current)
    if answered:
        return True
    if _hit_deadline(timeout, current, timed_out, deadline):
        return None
    return False


def check_setup_state(
    config: Config, *, deadline: Optional[float] = None
) -> tuple[bool, str]:
    """Return ``(True, "")`` when both configured endpoints answer, else the failing setting name.

    A custom endpoint needs only a correct answer to the plugin's real embeddings or rerank
    request at the configured path: any parseable 2xx body passes, with no health route and no
    bundled model id. A bundled endpoint (its base URL is one of the plugin defaults) must first
    answer ``GET <base>/health`` with ``status == "ok"`` and the configured model, then answer the
    same tiny live request. Never raises; unready is a value. The failing name is the settings
    key (``embedBaseUrl`` or ``rerankBaseUrl``) the setup-needed line names.

    *deadline* is the routing pass' absolute ``monotonic`` deadline: every probe's timeout is
    capped by the time left, so the check cannot outlive the budget a failed rank already spent.
    When a required probe is skipped or times out under that cap, the answer is indeterminate
    and this returns ``(False, "")``: no setting is named, because the endpoint never answered
    negatively. Without a deadline the check is definitive, exactly as before.
    """
    embed_base = config.embed_base_url
    embed_ready = _endpoint_ready(
        embed_base,
        config.embed_model_id,
        config.embed_path,
        {"model": config.embed_model_id, "input": ["setup check"]},
        config.embed_timeout_ms / 1000.0,
        BUNDLED_BASES["embed"],
        deadline=deadline,
    )
    if embed_ready is None:
        return False, ""
    if not embed_ready:
        return False, "embedBaseUrl"
    rerank_base = config.rerank_base_url
    rerank_ready = _endpoint_ready(
        rerank_base,
        config.rerank_model,
        config.rerank_path,
        {
            "model": config.rerank_model,
            "query": "setup check",
            "documents": ["setup check"],
            "top_n": 1,
        },
        config.rerank_timeout_ms / 1000.0,
        BUNDLED_BASES["rerank"],
        deadline=deadline,
    )
    if rerank_ready is None:
        return False, ""
    if not rerank_ready:
        return False, "rerankBaseUrl"
    return True, ""


def setup_note(failing: str, *, bundled: bool = True) -> str:
    """Return contract C2's one setup-needed line naming the failing settings key.

    *failing* is the settings key ``check_setup_state`` returned, ``embedBaseUrl`` or
    ``rerankBaseUrl``. The line rides on the display record only, with no provider token and no
    automatic setup.

    A bundled base keeps the setup-script line: the script installs those servers. A custom
    base never passes through the script (it writes no settings), so its line instead tells
    the user to repair the service at the configured setting.
    """
    if bundled:
        return f"Model setup needed: {failing} is not ready; run the model setup script and reload."
    return (
        f"Model setup needed: {failing} is not ready; "
        f"repair the service at the configured {failing} endpoint and reload."
    )
