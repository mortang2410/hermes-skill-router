"""Hermes plugin entry point for hermes-skill-router.

The plugin registers one ``llm_request`` middleware, one tool, ``skill_router_search``, and two
hooks, ``post_compaction`` and ``on_session_end``, from the same ``register(ctx)`` call, which is
the normal standalone load path: the manifest declares the tool and nothing imports the tool
module by itself. The tool is registered at load and cannot be removed by a settings change, so
with the switch off it answers ``disabled`` instead of disappearing (D7).

The ``post_compaction`` callback expires the session's dump record when the host commits a
compaction, which is what opens a new catalog window: the record is read by the middleware and no
message text is scanned for a summary. The ``on_session_end`` callback drops the turn's searchable
partition at turn end, keeps a named session's dump mark so its window survives, and expires an
anonymous task key's dump mark since that key ends with the turn. The host calls hook callbacks with the event's fields as keywords
plus a telemetry schema version and hands everything to a var-keyword callback, so each callback
declares the ids it uses and ``**kwargs``. Neither ever raises: an id that is missing, empty or
not a string is nothing to do, so a compaction or a turn end can never abort on cleanup.

The middleware callback is declared ``def on_llm_request(request, **kwargs)`` on purpose: the
dispatcher calls every callback as ``cb(**kwargs)`` after filtering the keyword set to the names the
callback declares (``PluginDispatchMixin._hook_callback_kwargs``, which the dispatcher uses for
middleware too), passing ``request`` plus ``task_id``, ``turn_id``, ``api_request_id``, ``session_id``,
``platform``, ``model``, ``provider``, ``base_url``, ``api_mode`` and ``api_call_count``. A narrower
signature is still called, but only with the keywords it declares, so the ids below would never reach
the rewrite and the display record would stay empty for every turn.
``session_id``, ``turn_id``, ``task_id`` and ``api_mode`` from that keyword set are forwarded to
the rewrite: the ids land in the display record (``task_id`` keys the searchable record only when
no session id names the caller), and the mode gates the rewrite (D8). A missing id stays an
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
    from .hermes_skill_router.config import (
        acknowledge_switch_deliveries,
        load,
        read_settings,
        switch_warning_messages,
    )
    from .hermes_skill_router.router import CHAT_COMPLETIONS, INJECTED_HEADER, rewrite
    from .hermes_skill_router.search import expire_dump, forget_session
    from .hermes_skill_router.tools import register_tools
else:  # imported as a bare module, e.g. by a test collector walking this repository
    from hermes_skill_router.config import (
        acknowledge_switch_deliveries,
        load,
        read_settings,
        switch_warning_messages,
    )
    from hermes_skill_router.router import CHAT_COMPLETIONS, INJECTED_HEADER, rewrite
    from hermes_skill_router.search import expire_dump, forget_session
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


def on_session_end(session_id: str = "", task_id: str = "", **kwargs: Any) -> None:
    """Drop the searchable partition for this turn's caller key.

    The host fires this at every turn finalization (``agent/turn_finalizer.py``) with the
    canonical ``session_id``/``task_id``/``turn_id`` set, so the partition the middleware
    remembered for ``_key(session_id, task_id)`` is dropped at turn end instead of living
    until the next request for that key. A named session's next turn re-parses its own
    partition, so nothing observable changes there. An anonymous caller is deliberately
    per-turn: each turn presents its own task key, so the next turn starts with no
    searchable catalog even though it presents no session id.

    A named session keeps its dump mark here: taking it would collapse every
    named session's compaction window to a per-turn window, and the first
    failed pass of every turn would dump the whole catalog again. An anonymous
    caller instead expires its task key's dump mark at turn end: each turn
    presents a fresh task key that no later event (``post_compaction`` names
    only session ids) can ever address, so keeping the mark would accumulate
    one stale key per anonymous turn. A missing, empty or non-string id is
    nothing to do, and the callback returns ``None`` and never raises, so a
    turn can never abort on cleanup.
    """
    named = session_id if isinstance(session_id, str) else ""
    task = task_id if isinstance(task_id, str) else ""
    forget_session(named, task_id=task)
    if not named and task:
        expire_dump(named, task_id=task)


def register(ctx: Any) -> None:
    """Register the plugin's middleware, search tool and compaction hook with *ctx*."""

    def on_llm_request(request: Mapping[str, Any], **kwargs: Any) -> Any:
        try:
            return _route(
                ctx,
                request,
                session_id=kwargs.get("session_id") or "",
                turn_id=kwargs.get("turn_id") or "",
                task_id=kwargs.get("task_id") or "",
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
    ctx.register_hook("on_session_end", on_session_end)


def _route(
    ctx: Any,
    request: Mapping[str, Any],
    *,
    session_id: str = "",
    turn_id: str = "",
    task_id: str = "",
    api_mode: str = "chat_completions",
) -> Any:
    """Validate the config, then hand the request to the rewrite. Returns ``None`` for no change.

    *session_id* and *turn_id* are the dispatcher's ids for this call, and *api_mode* is the mode
    the dispatcher declared. D8 gates the rewrite on that declared mode, not on the payload shape,
    so a request that happens to carry a system-role catalog under Codex Responses or Anthropic
    Messages stays untouched. The rewrite records the ids in the display record; a missing id stays
    an empty string rather than failing the request.

    Pending no-record switch warnings ride on the outgoing request as standalone system
    messages, before the disabled early return and after a rewrite alike, so the one channel
    that reaches the model carries the warning that the terminal and the host logger already
    showed. A request without a messages list, a rewrite that returns ``None`` (no catalog,
    unsupported mode, first failure, already injected), or an unsupported api mode on the
    disabled path has no supported channel to carry them: the payload stays untouched and
    the delivered fingerprint is not consumed, so a later supported request still delivers.
    Only an outgoing request that actually carries the messages acknowledges delivery.
    """
    cfg = load(read_settings(ctx))
    switch_messages = switch_warning_messages(cfg)
    if not cfg.enabled:
        result = _with_switch_messages(request, switch_messages, api_mode=api_mode)
        if result is not None:
            acknowledge_switch_deliveries(
                [message["content"] for message in switch_messages], cfg.cache_path
            )
        return result
    result = rewrite(
        request, cfg=cfg, session_id=session_id, turn_id=turn_id, task_id=task_id, api_mode=api_mode
    )
    if not isinstance(result, Mapping):
        # No catalog, unsupported mode, first failure, or already injected:
        # nothing was delivered, so the injected fingerprint stays pending.
        return result
    rewritten = result.get("request")
    if not switch_messages or not isinstance(rewritten, Mapping):
        return result
    messages = rewritten.get("messages")
    if not isinstance(messages, list):
        return result
    acknowledge_switch_deliveries(
        [message["content"] for message in switch_messages], cfg.cache_path
    )
    return {
        "request": {**dict(rewritten), "messages": _carrying_switch_messages(messages, switch_messages)}
    }


def _with_switch_messages(request: Any, switch_messages: Any, api_mode: str = "chat_completions") -> Any:
    """Return *request* with *switch_messages* appended, or ``None`` when there is no channel.

    Used on the disabled path, where the rewrite never runs: the switch warning still reaches
    the provider request even though routing and the display record are off. Unsupported
    modes (Anthropic Messages, Responses) have no system-message channel in ``messages``,
    so the payload is preserved untouched and the warning is deferred until a supported
    channel exists; delivery is not acknowledged on this path.
    """
    if not switch_messages or not isinstance(request, Mapping):
        return None
    mode = request.get("api_mode") or api_mode
    if mode != CHAT_COMPLETIONS:
        return None
    messages = request.get("messages")
    if not isinstance(messages, list):
        return None
    return {"request": {**dict(request), "messages": [*list(messages), *switch_messages]}}


def _carrying_switch_messages(messages: Any, switch_messages: Any) -> list:
    """Return *messages* with *switch_messages* carried without displacing a trailing block.

    A trailing suggestion block stays last, so the rewrite's replay guard still recognises the
    request as already injected on the next turn; otherwise the messages are appended at the end.
    """
    carried = list(messages)
    if carried and isinstance(carried[-1], Mapping):
        content = carried[-1].get("content")
        if isinstance(content, str) and content.startswith(INJECTED_HEADER):
            return [*carried[:-1], *switch_messages, carried[-1]]
    return [*carried, *switch_messages]
