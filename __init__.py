"""Hermes plugin entry point for hermes-skill-router.

The plugin loads with zero tools and zero hooks and registers exactly one ``llm_request``
middleware. The callback is declared ``def on_llm_request(request, **kwargs)`` on purpose: the
dispatcher calls every callback as ``cb(**kwargs)`` with no signature filtering, passing
``request`` plus ``task_id``, ``turn_id``, ``api_request_id``, ``session_id``, ``platform``,
``model``, ``provider``, ``base_url``, ``api_mode`` and ``api_call_count``. A narrower signature
raises ``TypeError``, which the dispatcher catches and skips, so routing would silently never run.
``session_id``, ``turn_id`` and ``api_mode`` from that keyword set are forwarded to the rewrite:
the ids land in the display record, and the mode gates the rewrite (D8). A missing id stays an
empty string, and a missing mode falls back to ``chat_completions``.

Settings are read with ``ctx.get_config`` from ``plugins.entries.hermes-skill-router.settings``.
The whole callback body is wrapped in ``try/except Exception`` and returns ``None`` on any
failure, so a plugin bug can never abort a model call.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

if __package__:
    from .hermes_skill_router.config import DEFAULTS, load
    from .hermes_skill_router.router import rewrite
else:  # imported as a bare module, e.g. by a test collector walking this repository
    from hermes_skill_router.config import DEFAULTS, load
    from hermes_skill_router.router import rewrite

logger = logging.getLogger(__name__)


def register(ctx: Any) -> None:
    """Register the plugin's single ``llm_request`` middleware with *ctx*."""

    def on_llm_request(request: Mapping[str, Any], **kwargs: Any) -> Any:
        try:
            return _route(
                ctx,
                request,
                session_id=kwargs.get("session_id") or "",
                turn_id=kwargs.get("turn_id") or "",
                api_mode=kwargs.get("api_mode") or "chat_completions",
            )
        except Exception:  # a plugin bug must never abort a model call
            logger.debug(
                "hermes-skill-router: llm_request middleware failed; leaving the request untouched",
                exc_info=True,
            )
            return None

    ctx.register_middleware("llm_request", on_llm_request)


def _settings(ctx: Any) -> dict[str, Any]:
    """Read every settings key through ``ctx.get_config``, defaulting when the surface lacks one."""
    raw: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        try:
            raw[key] = ctx.get_config(key, default)
        except Exception:
            raw[key] = default
    return raw


def _route(
    ctx: Any,
    request: Mapping[str, Any],
    *,
    session_id: str = "",
    turn_id: str = "",
    api_mode: str = "chat_completions",
) -> Any:
    """Validate the config, then hand the request to the rewrite. Returns ``None`` for no change.

    *session_id* and *turn_id* are the dispatcher's ids for this call, and *api_mode* is the mode
    the dispatcher declared. D8 gates the rewrite on that declared mode, not on the payload shape,
    so a request that happens to carry a system-role catalog under Codex Responses or Anthropic
    Messages stays untouched. The rewrite records the ids in the display record; a missing id stays
    an empty string rather than failing the request.
    """
    cfg = load(_settings(ctx))
    if not cfg.enabled:
        return None
    return rewrite(
        request, cfg=cfg, session_id=session_id, turn_id=turn_id, api_mode=api_mode
    )
