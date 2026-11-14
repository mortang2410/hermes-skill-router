"""Compose one provider-request rewrite (module M9, contract C9).

The rewrite is the only place that decides what one Chat Completions request becomes. It performs
the spec's order of work and nothing else:

1. Recognise the skill catalog in the first system message and partition it with
   ``cfg.eager_skills`` and ``cfg.deferred_skills`` (:mod:`hermes_skill_router.catalog`).
2. Build or load the index over the *routed* rows only (:mod:`hermes_skill_router.library`), so a
   kept row is never suggested, only listed.
3. Embed the newest typed message with the capped context that precedes it, and take the top
   ``cfg.shortlist_size`` routed rows by cosine over the index vectors
   (:mod:`hermes_skill_router.embed`).
4. Rerank ``cfg.document_prefix + name + " - " + description`` for those rows
   (:mod:`hermes_skill_router.rerank`).
5. Resolve every reranked candidate's read path through the host lookup (:mod:`hermes_skill_router.tools`)
   and recover the names the request's own earlier blocks already suggested, plus whether this is a
   compaction window's opening block.
6. Select with the suppressed set, the path map, ``cfg.cutoff`` and ``cfg.max_suggestions``
   (:mod:`hermes_skill_router.select`), so a suppressed or pathless candidate never spends a cap slot.
7. Render the block with the selected rows, the kept rows, the path map and the opening flag
   (:mod:`hermes_skill_router.inject`), and append it as one new ``system`` message at the end.

The strip is atomic with the injection: a request only loses its skills section when a non-empty
block replaces it. Any failure, such as an unreachable or timed-out endpoint, an unbuildable index,
no user text or an empty block, returns ``None`` on the first failure in a compaction window, and
``_apply_request_chain`` then keeps the payload that arrived, catalog included (R5, D8). On a later
failure in the same window the rewrite strips the catalog and appends one marker message instead,
so the whole catalog never reaches the model twice in one window. The rewrite never raises and
never mutates its input.

Steps 3 and 4 are :func:`hermes_skill_router.search.rank`, the same pass the model's search tool
runs, so a turn and a search for the same query and catalog can never disagree. The pass hands back
every reranked candidate uncapped and applies neither the cutoff nor the cap; those are applied once,
in the selector, after every candidate's read path has been resolved. The pass runs inside one routing
budget, ``cfg.routing_timeout_ms``, which starts after the index is in hand and never covers the cold
build, which embeds the whole catalog in batches and already fails through its own path. Each
per-request timeout is capped by the milliseconds left, so the pass cannot outlive the budget, and a
pass that reaches the end of it is abandoned with one log line naming the budget.

Suppression and the opening flag both come from the payload's own messages, never from module state.
A message whose content opens with the block's shared first line is a block this plugin injected
earlier in the compaction window: its suggested names are suppressed, and its presence means this is
not the window's opening block. A message whose content opens with the dropped marker's first line
is the proof that the whole catalog already went out in this window, so a later turn strips the
catalog again and suggests nothing. A compaction that removes both leaves neither, so the names are
suggestable again and pinned rows are listed again. Routing the same payload twice therefore returns
the same result, and the second pass of an unchanged payload is left untouched by the replay guard.

The parsed partition is kept for the session's search tool, which ranks its routed half; every
request clears the previous one first, so a search never ranks rows from an earlier prompt (D10).

On every request that carried the catalog on Chat Completions the same decision is written to the
display record through :mod:`hermes_skill_router.display`: ``routed`` with the selected suggestion
names, or ``inert`` with no names. While the association store is incomplete, contract C5's one
note rides on that record, so the surfaces can say history is incomplete and healing is running, or
report the one pending heal failure the C6 job handed the turn. When a per-request call fails,
contract C7's one line rides on the inert record instead, so the surfaces can say the endpoints
were unreachable and no suggestions were made; with the live clients the setup-needed line
(:func:`hermes_skill_router.setup_check.setup_note`) refines that line to name the failing
``embedBaseUrl`` or ``rerankBaseUrl``; that line outranks both history lines, so a failed
turn never consumes them. Either note never enters the payload, so it costs no token, and a
complete store adds nothing. The record is written after the payload is built and never enters
it, so the display costs no token. A written record is announced on the
plugin event ``plugin.hermes-skill-router.suggestions.updated`` with the turn id, best effort.

A completed pass also writes one info line to the plugin log, beginning ``skill router: query=``.
The line carries the query as a JSON string, every shortlisted candidate with the score the selector
compared, the cutoff, the suggestions and the embed, rerank and total timings, so an operator can
answer "why this skill" and "why not that skill" without replaying the request. A pass that failed,
timed out or spent its budget writes no decision line; its one warning names the reason and never the
query, because that line is the only place the plugin logs the user's text.

The returned mapping holds exactly one key, ``request``, because the middleware replaces the whole
provider payload with what it returns: dropping ``model``, ``tools`` or any other top-level key
would break the call. The result is deterministic for one payload and one index, which matters
because Hermes replays recorded request bytes on later turns.
"""

from __future__ import annotations

import json
import logging
import os
from time import monotonic
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import associations
from .catalog import strip_catalog
from .config import Config
from .display import INERT, ROUTED, record
from .embed import embed_texts as default_embed
from .fusion import resolve_fusion_config
from .inject import DROPPED_HEADER, DROPPED_MESSAGE, HEADER, render, suggested_names
from .library import Index, PathResolver, load_or_build, path_resolver
from .rerank import rerank as default_rerank
from .search import (
    EMBED_FAILED,
    RERANK_FAILED,
    Ranked,
    dumped,
    forget_session,
    rank,
    remember_dump,
    remember_partition,
)
from .select import Suggestion, select
from .setup_check import check_setup_state, setup_note
from .tools import skill_path

logger = logging.getLogger(__name__)

#: The only ``api_mode`` this rewrite understands; every other mode returns the request untouched.
CHAT_COMPLETIONS = "chat_completions"

#: Contract C5's one line while the association store is incomplete; it rides on the record only.
HISTORY_NOTE = "History is incomplete and healing is running in the background."

#: Contract C7's one line when a per-request call fails; it rides on the inert record only.
#: With the live clients the setup-needed line refining this naming is setup_check.setup_note.
ENDPOINT_NOTE = "Routing endpoints are unreachable; no suggestions were made this turn."

#: Both headers share their first line, so the double-injection guard recognises either variant from
#: the line alone. A last message that already starts with it was injected once already.
INJECTED_HEADER = HEADER.split("\n", 1)[0]

#: The dropped-catalog marker's first line. It is distinct from the block header on purpose: the
#: strip guard below recognises the marker as its own proof that the catalog already went out.
DROPPED_HEADER_LINE = DROPPED_HEADER.split("\n", 1)[0]

#: The most preceding-context characters that join the newest typed message, counted over whole
#: messages from the newest backwards. The newest message itself is never part of this cap.
RECENT_CONTEXT_CHARS = 10000

#: Plugin id and bare event name: the wire name is ``plugin.hermes-skill-router.suggestions.updated``.
PLUGIN_ID = "hermes-skill-router"
EVENT = "suggestions.updated"

#: One embedder: ``M5.embed_texts``, one vector per text in order or ``None`` on failure.
Embedder = Callable[[Config, Sequence[str]], Optional[Sequence[Sequence[float]]]]

#: One reranker: ``M6.rerank``, one probability per document in order or ``None`` on failure.
Reranker = Callable[[Config, str, Sequence[str]], Optional[Sequence[float]]]

#: Receives the turn id after a record was written; the default broadcasts the plugin event.
Notify = Callable[[str], None]

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

    The call starts by dropping the session's searchable record: every request the middleware sees
    begins with no searchable catalog, and only a request that parses one sets it again, so a tool
    call can never rank rows from an earlier prompt (D10).
    """
    try:
        forget_session(session_id)
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
    previous = _prior_blocks(messages)
    dropped_before = _dropped_before(messages)
    # The dump record says whether a failed pass in this session's current window already sent the
    # whole catalog. The host's post_compaction event expires that mark when a compaction commits,
    # so no message text is inspected for a summary: a paste can neither open nor keep a window.
    dumped_before = dumped(session_id)
    if dumped_before:
        dropped_before = True
    stripped, partition = strip_catalog(
        payload, eager=cfg.eager_skills, deferred=cfg.deferred_skills
    )
    if partition is None:
        # No catalog on this request: there is no section to replace, so stay inert and record
        # nothing. The record describes routing decisions, not requests in general.
        return None
    # The tool ranks the routed half of this partition, so it remembers the rows this prompt
    # showed before any later stage can fail and leave the request as it arrived.
    remember_partition(session_id, partition)
    def _fail(reason: str, note: str = ""):
        """Return the failure result for this pass.

        The first failure in a window returns ``None`` and the request keeps its catalog, which
        is one clean dump. A later failure in the same window strips the catalog and appends the
        dropped marker, so the model never reads the whole catalog twice. Both paths record
        inert and log once, and both leave the input payload untouched. *note* is contract C7's
        one line when a per-request call failed, which outranks every other note this turn; it
        rides on the inert record only, so the surfaces can say no suggestions were made.
        """
        _log_once(reason)
        _recorder(cfg, INERT, (), session_id, turn_id, notify, note=note)()
        if not previous and not dropped_before and not dumped_before:
            remember_dump(session_id)
            return None
        return {
            "request": {
                **dict(payload),
                "messages": [
                    *list(stripped.get("messages") or ()),
                    {"role": "system", "content": DROPPED_MESSAGE},
                ],
            }
        }

    query = _query_text(messages)
    if not query:
        return _fail("no user text")
    if not partition.routed.names:
        # All-eager config: no pass can select anything. First pass is the allowed dump.
        return _fail("no routed skill to suggest")
    if index is None:
        try:
            index = load_or_build(cfg, partition.routed, embed)
        except Exception:
            logger.debug("skill router: index build raised", exc_info=True)
            index = None
        if index is None:
            return _fail(
                "index build failed",
                _failure_note(cfg, uses_live=embed is default_embed),
            )
    if not index.names:
        return _fail("the skill index carries no row")
    if dropped_before or dumped_before:
        # A dump earlier in this window already showed every skill, so nothing is suggestible
        # for the rest of the window even when ranking would succeed now.
        return _fail("catalog already dumped in this window")
    # One memo for the whole pass: the recency gate and the selection both resolve through the
    # same saved map, so a candidate is looked up at most once a pass and never again once known.
    resolve = path_resolver(cfg, index, skill_path)
    ranked = rank(cfg, index, query, embed=embed, rerank=rerank, resolve=resolve)
    if ranked.reason:
        # A failed per-request call is a dead endpoint, and its line outranks the history
        # line; a budget expiry keeps no line and never consumes the pending heal note.
        if ranked.reason in (EMBED_FAILED, RERANK_FAILED):
            note = _failure_note(
                cfg,
                uses_live=embed is default_embed and rerank is default_rerank,
            )
        else:
            note = ""
        return _fail(ranked.detail, note)
    # C4 has fired by now: the pass reached the shortlist, which opened the store and asked for one
    # heal when it was incomplete. Contract C5's note is read once here, so every record this turn
    # writes carries the same line, and a failed pass never reaches here, so the dead-endpoint line outranks it.
    note = _history_note(cfg)
    # The payload's own prior blocks decide both what this turn must not repeat and whether the
    # pinned rows belong here. Nothing is remembered between calls.
    suppressed = {name for block in previous for name in suggested_names(block)}
    eager = set(cfg.eager_skills or ())
    wanted = [name for name, _score in ranked.candidates]
    wanted.extend(name for name in partition.kept.names if name in eager)
    paths = _resolve_paths(wanted, resolve)
    selected = select(
        [score for _name, score in ranked.candidates],
        [name for name, _score in ranked.candidates],
        cfg,
        paths,
        suppressed,
    )
    _log_decision(cfg, query, ranked, selected)
    # The partition's variant chooses the header, so the block matches the prose this request lost.
    block = render(
        selected,
        partition.kept,
        index,
        cfg,
        partition.one_shot,
        opening=not previous,
        paths=paths,
    )
    if not block:
        # No selected suggestion and no kept row: stripping without a replacement would lose the
        # catalog for nothing, so the whole request is returned untouched.
        _recorder(cfg, INERT, (), session_id, turn_id, notify, note=note)()
        return None
    rewritten = {
        **dict(payload),
        "messages": [
            *list(stripped.get("messages") or ()),
            {"role": "system", "content": block},
        ],
    }
    _recorder(
        cfg,
        ROUTED,
        [name for name, _score in selected],
        session_id,
        turn_id,
        notify,
        note=note,
    )()
    return {"request": rewritten}


def _query_text(messages: Any) -> str:
    """Return the newest typed message plus the capped context that precedes it, or ``""``.

    Only ``user`` messages that carry text count, so a system message, an assistant message and an
    earlier injected block never join the query. The preceding messages are kept whole, taken from
    the newest backwards while the running total stays within :data:`RECENT_CONTEXT_CHARS`, then
    joined oldest first after the newest message with `` Context: `` between them. The scan
    stops at the first message that would cross the cap rather than cutting or skipping it. With no
    preceding typed message the query is exactly the newest message.
    """
    texts = _typed_user_texts(messages)
    if not texts:
        return ""
    context: List[str] = []
    used = 0
    for text in reversed(texts[:-1]):
        if used + len(text) > RECENT_CONTEXT_CHARS:
            break
        context.append(text)
        used += len(text)
    context.reverse()
    if not context:
        return texts[-1]
    return texts[-1] + " Context: " + " ".join(context)


def _typed_user_texts(messages: Any) -> List[str]:
    """Return the text of every typed ``user`` message that carries text, oldest first."""
    if not isinstance(messages, list):
        return []
    texts: List[str] = []
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            text = _content_text(message.get("content"))
            if text:
                texts.append(text)
    return texts


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


def _prior_blocks(messages: Any) -> List[str]:
    """Return the text of every message that opens with the block's shared first line.

    Only a message that opens with that line counts, so user text that merely mentions a skill name,
    or quotes the marker later in a line, never seeds the suppressed set. The returned blocks are the
    payload's own record of what earlier turns in this compaction window showed.
    """
    blocks: List[str] = []
    if not isinstance(messages, list):
        return blocks
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        text = _message_text(message)
        if text.startswith(INJECTED_HEADER):
            blocks.append(text)
    return blocks


def _message_text(message: Any) -> str:
    """Return one message's text from a string or a list of text parts, else ``""``."""
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _dropped_before(messages: Any) -> bool:
    """Return whether any message opens with the dropped-catalog marker's first line.

    The marker is this plugin's own proof that the whole catalog already went out in this window
    on a failed pass. A compaction that removes it starts a new window and the catalog may go
    out again, exactly as with a routed block.
    """
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        if _message_text(message).startswith(DROPPED_HEADER_LINE):
            return True
    return False


def _resolve_paths(names: Sequence[str], resolve: PathResolver) -> Dict[str, str]:
    """Resolve every candidate's and pinned name's read path before the selector cuts.

    *resolve* is this pass' resolver: the host lookup is the resolver ``skill_view`` uses, so the
    path a row carries is the file the model can open, and the pass wraps it in one memo so a name
    the recency gate already resolved is never looked up again. A name the lookup refuses, or a
    lookup that raises, costs that row its slot and never the whole block. Repeated names are
    looked up once.
    """
    paths: Dict[str, str] = {}
    for name in names:
        if name in paths:
            continue
        try:
            path = resolve(name)
        except Exception:  # the host lookup never raises, but the block must not depend on it
            logger.debug("skill router: the path lookup for %r raised", name, exc_info=True)
            path = None
        if path:
            paths[name] = str(path)
    return paths


def _store_in_play(cfg: Config) -> bool:
    """Return whether this turn's pass opens the association store, so a heal can have been asked.

    ``library.shortlist`` returns the cosine order before opening the store when the fusion mode is
    the cosine rollback or every channel is off, so that state skips nothing and heals nothing;
    C5's line would claim a heal that does not exist, and no note is written.
    """
    settings = resolve_fusion_config(
        {
            "mode": cfg.fusion_mode,
            "channels": {
                "semantic": cfg.channels_semantic,
                "bm25": cfg.channels_bm25,
                "associations": cfg.channels_associations,
            },
        }
    )
    return settings.mode != "cosine" and settings.enabled


def _configured_store_path(cfg: Config) -> Optional[str]:
    """Return the association store path the library pass opens, or ``None`` when it is off.

    This repeats the two-line path rule ``library._open_store`` applies, because M2's helper is
    private and the note must name the store the turn actually read. A configured
    ``associationDbPath`` wins; otherwise the store sits beside the index cache.
    """
    raw = (cfg.association_db_path or "").strip()
    if raw:
        return os.path.expanduser(raw)
    cache = (cfg.cache_path or "").strip()
    if not cache:
        return None
    return associations.association_db_path_for(cache)


def _history_note(cfg: Config) -> str:
    """Return the turn's one note for an incomplete history, or ``""`` when the store is complete.

    C6's completeness rule answers whether there is a line at all, over the same configured store
    path and the same Hermes session store the library pass reads. A missing, empty or unreadable
    store is incomplete by that rule, so the four store states C4 skips all carry a line.

    While the store is incomplete, C6's pending heal-failure note wins over the generic
    "healing is running" line: a failed heal is not running, so the generic line would claim work
    that stopped, and the failure is the state the user needs to see. The accessor is one-shot, so
    it is read only after the turn is known to carry a history note: a turn that shows contract
    C7's dead-endpoint note returns before this chooser runs, and a complete store never
    consumes it.
    """
    if not _store_in_play(cfg):
        return ""
    path = _configured_store_path(cfg)
    if path is None or _store_is_complete(path):
        return ""
    return associations.heal_failure_note(path) or HISTORY_NOTE


#: The last setup check per endpoint config, so a dead-endpoint turn does not pay two more
#: endpoint timeouts on every request inside the health window.
_SETUP_CACHE: Dict[tuple, tuple[float, bool, str]] = {}


def _setup_note(cfg: Config) -> str:
    """Return contract C2's setup-needed line for *cfg*, or ``""`` when both endpoints are ready.

    The check runs only on a dead-endpoint turn with the live clients. A test that injects its
    own transport proves ranking without touching a socket, so it never probes and keeps the
    generic dead-endpoint line. The result is throttled by ``healthCheckMs`` per endpoint
    config, mirroring the clients' health gates.
    """
    key = (
        cfg.embed_base_url,
        cfg.embed_path,
        cfg.embed_model_id,
        cfg.rerank_base_url,
        cfg.rerank_path,
        cfg.rerank_model,
        cfg.embed_timeout_ms,
        cfg.rerank_timeout_ms,
    )
    window_s = cfg.health_check_ms / 1000.0
    now = monotonic()
    cached = _SETUP_CACHE.get(key)
    if cached is not None and window_s > 0 and (now - cached[0]) < window_s:
        ready, failing = cached[1], cached[2]
    else:
        try:
            ready, failing = check_setup_state(cfg)
        except Exception:  # the check never raises, but the note must not depend on it
            logger.debug("skill router: the setup check raised", exc_info=True)
            return ""
        _SETUP_CACHE[key] = (now, ready, failing)
    if ready or not failing:
        return ""
    return setup_note(failing)


def _failure_note(cfg: Config, *, uses_live: bool) -> str:
    """Return the one line for a dead-endpoint turn, naming the failing setting when probed.

    With the live clients the setup-needed line names ``embedBaseUrl`` or
    ``rerankBaseUrl``; without them the generic dead-endpoint line is all a fake-transport
    failure can name. A ready check falls back to the generic line, so a budget or malformed
    body never claims setup is needed.
    """
    if uses_live:
        probed = _setup_note(cfg)
        if probed:
            return probed
    return ENDPOINT_NOTE


def _store_is_complete(path: str) -> bool:
    """Return whether the association store at *path* is complete, by contract C6's rule.

    A store that cannot be opened or read answers "incomplete", which is the same answer the
    detector gives: no evidence of coverage is not coverage. The note must never cost the turn, so
    every failure here is swallowed after one debug line.
    """
    store: Optional[associations.AssociationStore]
    try:
        store = associations.open_store(path, read_only=True)
    except Exception:  # a missing, empty or unreadable file is no evidence of coverage
        logger.debug("skill router: the store could not be opened for the C5 note", exc_info=True)
        return False
    try:
        return bool(associations.is_store_complete(store))
    except Exception:  # the detector never raises, but the note must not depend on that
        logger.debug("skill router: the store completeness check raised", exc_info=True)
        return False
    finally:
        try:
            store.close()
        except Exception:  # a close failure must not cost the turn
            logger.debug("skill router: the association store close raised", exc_info=True)


def _recorder(
    cfg: Config,
    status: str,
    names: Sequence[str],
    session_id: str,
    turn_id: str,
    notify: Optional[Notify],
    note: str = "",
) -> Callable[[], None]:
    """Return a thunk that writes one display record and announces it when a write happened.

    *note* is contract C5's one line for this turn, or ``""``. The thunk never raises: the payload
    the caller is about to return must not depend on the display, so a record or broadcast failure
    is swallowed here rather than reaching the pipeline.
    """

    def _record() -> None:
        try:
            written = record(
                cfg,
                status=status,
                names=names,
                session_id=session_id,
                turn_id=turn_id,
                note=note,
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


def _log_decision(
    cfg: Config, query: str, ranked: Ranked, selected: Sequence[Suggestion]
) -> None:
    """Write the one decision line for a completed pass.

    The line carries the query as a JSON string, every shortlisted candidate with the score the
    selector compared, the cutoff, the suggestions and the three stage timings. It is the only place
    the plugin logs the user's text; a warning never carries it. Candidates the cutoff, the suppressed
    set or an unresolved path rejected stay in the line with their scores, so a name missing from the
    block still has an answer.
    """
    candidates = ",".join(f"{name}:{score!r}" for name, score in ranked.candidates) or "-"
    suggestions = ",".join(f"{name}:{score!r}" for name, score in selected) or "-"
    logger.info(
        "skill router: query=%s candidates=%s cutoff=%r suggestions=%s "
        "embed_ms=%.1f rerank_ms=%.1f total_ms=%.1f",
        json.dumps(query),
        candidates,
        cfg.cutoff,
        suggestions,
        ranked.embed_ms,
        ranked.rerank_ms,
        ranked.total_ms,
    )


def _log_once(reason: str) -> None:
    """Log one distinct failure reason at most once per process, never including user text."""
    if reason in _LOGGED:
        return
    _LOGGED.add(reason)
    logger.warning("skill router: %s; leaving the request unchanged", reason)
