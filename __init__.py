"""Hermes plugin entry point for hermes-skill-router.

The plugin registers one ``llm_request`` middleware, one tool, ``skill_router_search``, and one
``post_compaction`` hook from the same ``register(ctx)`` call, which is the normal standalone load
path: the manifest declares the tool and nothing imports the tool module by itself. The tool is
registered at load and cannot be removed by a settings change, so with the switch off it answers
``disabled`` instead of disappearing (D7).

The ``post_compaction`` callback expires the session's dump record when the host commits a
compaction, which is what opens a new catalog window: the record is read by the middleware and no
message text is scanned for a summary. The host calls hook callbacks with the event's fields as
keywords plus a telemetry schema version and hands everything to a var-keyword callback, so the
callback declares the two ids it uses and ``**kwargs``. It never raises: an id that is missing,
empty or not a string is nothing to do, so a compaction can never abort a turn.

The middleware callback is declared ``def on_llm_request(request, **kwargs)`` on purpose: the
dispatcher calls every callback as ``cb(**kwargs)`` after filtering the keyword set to the names the
callback declares (``PluginDispatchMixin._hook_callback_kwargs``, which the dispatcher uses for
middleware too), passing ``request`` plus ``task_id``, ``turn_id``, ``api_request_id``, ``session_id``,
``platform``, ``model``, ``provider``, ``base_url``, ``api_mode`` and ``api_call_count``. A narrower
signature is still called, but only with the keywords it declares, so the ids below would never reach
the rewrite and the display record would stay empty for every turn.
``session_id``, ``turn_id`` and ``api_mode`` from that keyword set are forwarded to the rewrite:
the ids land in the display record, and the mode gates the rewrite (D8). A missing id stays an
empty string, and a missing mode falls back to ``chat_completions``.

Settings are read with ``ctx.get_config`` from ``plugins.entries.hermes-skill-router.settings``. The
middleware callback catches every exception and returns ``None``, so a plugin bug inside a request
can never abort a model call. Registration itself is not wrapped: a host that refuses the tool name
fails the plugin load, which is the surface that should show that conflict.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

if __package__:
    from .hermes_skill_router.config import load, read_settings
    from .hermes_skill_router.router import rewrite
    from .hermes_skill_router.search import expire_dump
    from .hermes_skill_router.tools import register_tools
else:  # imported as a bare module, e.g. by a test collector walking this repository
    from hermes_skill_router.config import load, read_settings
    from hermes_skill_router.router import rewrite
    from hermes_skill_router.search import expire_dump
    from hermes_skill_router.tools import register_tools

logger = logging.getLogger(__name__)


def on_post_compaction(session_id: str = "", old_session_id: str = "", **kwargs: Any) -> None:
    """Open a new compaction window for every session the committed compaction names.

    The host calls this after a compaction commits, with the event's fields as keywords plus a
    telemetry schema version; ``**kwargs`` accepts the rest. Both ids it names lose their dump
    record, because a fork moves the conversation to a new id and a record left on the old one
    would keep that session's catalog dumped forever. A missing, empty or non-string id is nothing
    to do, a repeated id discards an absent mark, and the callback returns ``None`` and never
    raises, so a compaction can never abort a turn.
    """
    for named in (session_id, old_session_id):
        if isinstance(named, str) and named:
            expire_dump(named)


def register(ctx: Any) -> None:
    """Register the plugin's middleware, search tool and compaction hook with *ctx*."""

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
    register_tools(ctx)
    ctx.register_hook("post_compaction", on_post_compaction)


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
    cfg = load(read_settings(ctx))
    if not cfg.enabled:
        return None
    return rewrite(
        request, cfg=cfg, session_id=session_id, turn_id=turn_id, api_mode=api_mode
    )
