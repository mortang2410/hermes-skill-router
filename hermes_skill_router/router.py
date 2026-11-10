"""Compose one provider-request rewrite (module M9, contract C9).

The rewrite is the only place that decides what one Chat Completions request becomes. It performs
the spec's order of work and nothing else:

1. Recognise the skill catalog in the first system message and partition it with
   ``cfg.eager_skills`` and ``cfg.deferred_skills`` (:mod:`hermes_skill_router.catalog`).
2. Build or load the index over the *routed* rows only (:mod:`hermes_skill_router.library`), so a
   kept row is never suggested, only listed.
3. Embed the last user message's text and take the top ``cfg.shortlist_size`` routed rows by cosine
   over the index vectors (:mod:`hermes_skill_router.embed`).
4. Rerank ``cfg.document_prefix + name + " - " + description`` for those rows
   (:mod:`hermes_skill_router.rerank`).
5. Select with ``cfg.cutoff`` and ``cfg.max_suggestions`` (:mod:`hermes_skill_router.select`).
6. Render the block with the selected rows, the kept rows and the index
   (:mod:`hermes_skill_router.inject`), and append it as one new ``system`` message at the end.

The strip is atomic with the injection: a request only loses its skills section when a non-empty
block replaces it. Any failure — an unreachable or timed-out endpoint, an unbuildable index, no
user text, an empty block — returns ``None``, and ``_apply_request_chain`` then keeps the payload
that arrived, catalog included (R5, D8). The rewrite never raises and never mutates its input.

On every request that carried the catalog on Chat Completions the same decision is written to the
display record through :mod:`hermes_skill_router.display`: ``routed`` with the selected suggestion
names, or ``inert`` with no names. The record is written after the payload is built and never
enters it, so the display costs no token. A written record is announced on the plugin event
``plugin.hermes-skill-router.suggestions.updated`` with the turn id, best effort.

The returned mapping holds exactly one key, ``request``, because the middleware replaces the whole
provider payload with what it returns: dropping ``model``, ``tools`` or any other top-level key
would break the call. The result is deterministic for one payload and one index, which matters
because Hermes replays recorded request bytes on later turns.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from .catalog import strip_catalog
from .config import Config
from .display import INERT, ROUTED, record
from .embed import embed_texts as default_embed
from .inject import HEADER, render
from .library import Index, load_or_build
from .rerank import rerank as default_rerank
from .select import select

logger = logging.getLogger(__name__)

#: The only ``api_mode`` this rewrite understands; every other mode returns the request untouched.
CHAT_COMPLETIONS = "chat_completions"

#: Both headers share their first line, so the double-injection guard recognises either variant from
#: the line alone. A last message that already starts with it was injected once already.
INJECTED_HEADER = HEADER.split("\n", 1)[0]

#: Plugin id and bare event name: the wire name is ``plugin.hermes-skill-router.suggestions.updated``.
PLUGIN_ID = "hermes-skill-router"
EVENT = "suggestions.updated"

#: One embedder: ``M5.embed_texts``, one vector per text in order or ``None`` on failure.
Embedder = Callable[[Config, Sequence[str]], Optional[Sequence[Sequence[float]]]]

#: One reranker: ``M6.rerank``, one probability per document in order or ``None`` on failure.
Reranker = Callable[[Config, str, Sequence[str]], Optional[Sequence[float]]]

#: Receives the turn id after a record was written; the default broadcasts the plugin event.
Notify = Callable[[str], None]

#: One shortlist row: the catalog name and the description the index carries for it.
ShortlistRow = Tuple[str, str]

#: Failure reasons already logged, so one broken path logs once per process instead of per request.
_LOGGED: set[str] = set()


def rewrite(
    payload: Mapping[str, Any],
    *,
    cfg: Config,
    index: Optional[Index] = None,
    embed: Optional[Embedder] = None,
    rerank: Optional[Reranker] = None,
    api_mode: str = CHAT_COMPLETIONS,
    session_id: str = "",
    turn_id: str = "",
    notify: Optional[Notify] = None,
) -> Optional[Mapping[str, Any]]:
    """Return ``{"request": <rewritten payload>}``, or ``None`` when the request must stay as it is.

    *payload* is the provider request kwargs Hermes is about to send. *index* is the routed index
    the caller already holds; when it is ``None`` the rewrite builds one from the partitioned
    catalog, which costs one embeddings request on a cold cache. *embed* and *rerank* default to
    the HTTP clients, and exist so tests and callers can supply their own transport. *api_mode*
    defaults to ``chat_completions``; a payload carrying its own ``api_mode`` wins. *session_id*
    and *turn_id* are recorded for the display, and *notify* receives the turn id after a record
    write (the default broadcasts the plugin event).

    Returns ``None`` for a non-Chat-Completions mode, a disabled plugin, a payload without the
    catalog, no user text, an empty index or block, a last message that already carries the block,
    and for any failure inside the pipeline. An exception never escapes this function.
    """
    try:
        return _compose(
            payload,
            cfg=cfg,
            index=index,
            embed=embed or default_embed,
            rerank=rerank or default_rerank,
            api_mode=api_mode,
            session_id=session_id,
            turn_id=turn_id,
            notify=notify,
        )
    except Exception:  # a plugin bug must never abort the model call
        logger.debug("skill router: rewrite failed; leaving the request unchanged", exc_info=True)
        return None


def _compose(
    payload: Mapping[str, Any],
    *,
    cfg: Config,
    index: Optional[Index],
    embed: Embedder,
    rerank: Reranker,
    api_mode: str,
    session_id: str,
    turn_id: str,
    notify: Optional[Notify],
) -> Optional[Mapping[str, Any]]:
    """Run the pipeline once and return the middleware result, or ``None`` for no change."""
    if not isinstance(payload, Mapping):
        return None
    if not cfg.enabled:
        return None
    mode = payload.get("api_mode") or api_mode
    if mode != CHAT_COMPLETIONS:
        # Codex Responses carries the prompt in ``instructions``/``input`` and Anthropic Messages
        # in ``system``, so a ``messages`` rewrite would do nothing or corrupt the call.
        _log_once("unsupported api mode")
        return None
    messages = payload.get("messages")
    if _already_injected(messages):
        # The previous turn's block is the last message; injecting again would compound it.
        return None
    stripped, partition = strip_catalog(
        payload, eager=cfg.eager_skills, deferred=cfg.deferred_skills
    )
    if partition is None:
        # No catalog on this request: there is no section to replace, so stay inert and record
        # nothing. The record describes routing decisions, not requests in general.
        return None
    inert = _recorder(cfg, INERT, (), session_id, turn_id, notify)
    query = _last_user_text(messages)
    if not query:
        _log_once("no user text")
        inert()
        return None
    if not partition.routed.names:
        # Every row was kept: there is nothing to suggest and nothing a block would replace.
        inert()
        return None
    if index is None:
        try:
            index = load_or_build(cfg, partition.routed, embed)
        except Exception:
            logger.debug("skill router: index build raised", exc_info=True)
            index = None
        if index is None:
            _log_once("index build failed")
            inert()
            return None
    if not index.names:
        inert()
        return None
    query_vector = _embed_query(cfg, query, embed, inert)
    if query_vector is None:
        return None
    shortlist = _shortlist(index, query_vector, cfg.shortlist_size)
    if not shortlist:
        inert()
        return None
    documents = [
        cfg.document_prefix + name + " - " + description for name, description in shortlist
    ]
    try:
        scores = rerank(cfg, query, documents)
    except Exception:
        logger.debug("skill router: rerank raised", exc_info=True)
        scores = None
    if scores is None:
        _log_once("rerank unreachable")
        inert()
        return None
    chosen = select(scores, [name for name, _description in shortlist], cfg)
    # The partition's variant chooses the header, so the block matches the prose this request lost.
    block = render(chosen, partition.kept, index, cfg, partition.one_shot)
    if not block:
        # No selected suggestion and no kept row: stripping without a replacement would lose the
        # catalog for nothing, so the whole request is returned untouched.
        inert()
        return None
    rewritten = {
        **dict(payload),
        "messages": [
            *list(stripped.get("messages") or ()),
            {"role": "system", "content": block},
        ],
    }
    _recorder(cfg, ROUTED, [name for name, _score in chosen], session_id, turn_id, notify)()
    return {"request": rewritten}


def _embed_query(
    cfg: Config,
    query: str,
    embed: Embedder,
    inert: Callable[[], None],
) -> Optional[List[float]]:
    """Embed the user's text, or return ``None`` after logging and recording the inert decision."""
    try:
        vectors = embed(cfg, [query])
    except TimeoutError:
        _log_once("embed timeout")
        inert()
        return None
    except Exception:
        logger.debug("skill router: embeddings call raised", exc_info=True)
        _log_once("embed unreachable")
        inert()
        return None
    if not vectors:
        _log_once("embed unreachable")
        inert()
        return None
    try:
        return [float(value) for value in vectors[0]]
    except (TypeError, ValueError):
        _log_once("embed unreachable")
        inert()
        return None


def _shortlist(index: Index, query_vector: Sequence[float], limit: int) -> List[ShortlistRow]:
    """Return the *limit* routed rows closest to *query_vector* by cosine, ties in index order."""
    scored: List[Tuple[float, int, str, str]] = []
    for position, (name, description, vector) in enumerate(
        zip(index.names, index.descriptions, index.vectors)
    ):
        scored.append((-_cosine(query_vector, vector), position, name, description))
    scored.sort()
    return [
        (name, description)
        for _score, _position, name, description in scored[: max(0, limit)]
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or ``0.0`` when either has no length."""
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for position in range(length):
        first = float(left[position])
        second = float(right[position])
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    score = dot / math.sqrt(left_norm * right_norm)
    return score if math.isfinite(score) else 0.0


def _last_user_text(messages: Any) -> str:
    """Return the text of the last ``user`` message, or ``""`` when there is none or it is empty."""
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _content_text(message.get("content"))
    return ""


def _content_text(content: Any) -> str:
    """Return a message's text: the string itself, or the text parts of a content list."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts).strip()
    return ""


def _already_injected(messages: Any) -> bool:
    """Return whether the last message already starts with the injected block header."""
    if not isinstance(messages, list) or not messages:
        return False
    last = messages[-1]
    content = last.get("content") if isinstance(last, Mapping) else None
    return isinstance(content, str) and content.startswith(INJECTED_HEADER)


def _recorder(
    cfg: Config,
    status: str,
    names: Sequence[str],
    session_id: str,
    turn_id: str,
    notify: Optional[Notify],
) -> Callable[[], None]:
    """Return a thunk that writes one display record and announces it when a write happened.

    The thunk never raises: the payload the caller is about to return must not depend on the
    display, so a record or broadcast failure is swallowed here rather than reaching the pipeline.
    """

    def _record() -> None:
        try:
            written = record(
                cfg, status=status, names=names, session_id=session_id, turn_id=turn_id
            )
        except Exception:
            logger.debug("skill router: the display record failed", exc_info=True)
            return
        if written is None:
            return
        try:
            (notify or _broadcast)(turn_id)
        except Exception:
            logger.debug("skill router: the suggestion event failed", exc_info=True)

    return _record


def _broadcast(turn_id: str) -> None:
    """Announce a written record on ``plugin.hermes-skill-router.suggestions.updated``, best effort.

    A process with no gateway, and every import or transport error, is a no-op: nobody is
    listening, or the window that was listening refreshes on its next mount.
    """
    try:
        from hermes_cli.plugin_events import broadcast_plugin_event
    except Exception:
        return
    broadcast_plugin_event(PLUGIN_ID, EVENT, {"turn_id": turn_id})


def _log_once(reason: str) -> None:
    """Log one distinct failure reason at most once per process, never including user text."""
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    logger.warning("skill router: %s; leaving the request unchanged", reason)
