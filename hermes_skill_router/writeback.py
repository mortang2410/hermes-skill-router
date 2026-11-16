"""Conversation write-back with an origin-tagged plugin store (module M8, contract C1).

The injected suggestion block never entered conversation history: the run loop
builds ``api_messages`` from ``conversation_history`` plus the system prompt and
the middleware only replaces the outgoing payload, so the block repeated every
turn. This module gives the block a durable home without forking the session:

- a plugin-owned store (one JSON file beside the cache) carrying block bytes,
  insertion index, origin class, window generation and block hash, keyed by
  session id;
- a session DB append, only for origin ``normal`` and only when the payload
  history tail matches the session DB tail, so a detached fork on a stale
  snapshot never appends into a moved-on parent session;
- a plugin-side compression fence: before the append, M8 opens its own
  ``hermes_state.SessionDB`` handle and reads the session's compression lock
  (``get_compression_lock_holder``); while a compression is in flight the block
  lives store-only, generation-tagged;
- post-compaction cleanup: the appended row id is recorded, and on
  ``post_compaction`` it is re-resolved with ``resolve_active_row_id`` (a race
  append may have been cloned with a new id) and the result deactivated with
  ``deactivate_message``, so no plugin block row survives past the close of
  any window.

Origin classes come from the middleware ``platform`` kwarg plus prompt shape
(no middleware kwarg distinguishes the fork; turn ids are uuids): ``curator``
when the platform is curator or the user message carries the curator prompt
shape (curator-catalog-fix-study.md), ``review`` when the user message starts
with the background-review prompt shape, else ``normal``. Only ``normal``
appends to the DB; review and curator blocks live in the store only and never
write a DB row under the shared parent session id. Re-insertion replays only
same-origin, same-generation blocks, deduped by byte search, on every outgoing
request regardless of routing success; a store or append failure for a new
block sends the admitted blocks with the window's catalog state unchanged and
no fresh block, never ``None``. Writes are keyed by block hash, so a retried
middleware call appends once. Every failure fails open: this module never
raises to the model call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("hermes_skill_router.writeback")

STORE_FILENAME = "writeback-store.json"

#: Environment override for the store path; tests point it at a temporary file.
STORE_PATH_ENV = "HERMES_SKILL_ROUTER_WRITEBACK"

#: In-process fallback latch for the stripped window flag (contract C5): when the
#: store file cannot be written, the durable flag is lost, but a stripped payload
#: that already went out must still keep later failures stripped, so the flag is
#: also held in memory for the life of the process. It is cleared on expire.
#: Keyed by (store path, session) so a new store path starts clean.
_STRIPPED_FALLBACK: Set[Tuple[Optional[str], str]] = set()
_STRIPPED_LOCK = threading.Lock()

ORIGIN_NORMAL = "normal"
ORIGIN_REVIEW = "review"
ORIGIN_CURATOR = "curator"
ORIGINS = (ORIGIN_NORMAL, ORIGIN_REVIEW, ORIGIN_CURATOR)

#: The background-review prompt shape (agent/background_review.py
#: _MEMORY_REVIEW_PROMPT): a review fork's user message starts with this text.
REVIEW_PROMPT_PREFIX = "Review the conversation above"

#: The curator prompt shape (agent/curator.py): the user message carries this
#: sentence, optionally preceded by the dry-run banner, and embeds the
#: "Agent-created skills (N):" candidate list. Matching either string
#: distinguishes the curator fork when the platform kwarg is empty or renamed.
CURATOR_PROMPT_MARKERS = (
    "You are running as Hermes' background skill CURATOR",
    "Agent-created skills (",
)


def store_path(cache_path: Any) -> Optional[str]:
    """Return the plugin store path, honoring the env override and *cache_path*.

    Like the switch-warning file, the store lives beside the index cache with
    ``~`` expanded, so a relative or home-based ``cachePath`` resolves the same
    way every caller resolves it.
    """
    override = os.environ.get(STORE_PATH_ENV)
    if override:
        return override
    if not cache_path or not isinstance(cache_path, str) or not cache_path.strip():
        return None
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(cache_path.strip())))
    return os.path.join(directory, STORE_FILENAME)


def _empty_store() -> Dict[str, Any]:
    return {"windows": {}}


def _read_store(cache_path: Any) -> Dict[str, Any]:
    """Return the parsed store, or an empty store on any read/parse failure."""
    path = store_path(cache_path)
    if path is None:
        return _empty_store()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return _empty_store()
    if not isinstance(data, dict) or not isinstance(data.get("windows"), dict):
        return _empty_store()
    return data


def _write_store(cache_path: Any, store: Dict[str, Any]) -> bool:
    """Persist *store* beside the cache. Returns ``False`` on any failure."""
    path = store_path(cache_path)
    if path is None:
        return False
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(store, handle, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("hermes-skill-router writeback: store write failed", exc_info=True)
        return False


def _window(store: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    windows = store["windows"]
    window = windows.get(session_id)
    if not isinstance(window, dict):
        window = {"generation": 0, "blocks": [], "row_ids": []}
        windows[session_id] = window
    if not isinstance(window.get("blocks"), list):
        window["blocks"] = []
    if not isinstance(window.get("row_ids"), list):
        window["row_ids"] = []
    if not isinstance(window.get("generation"), int):
        window["generation"] = 0
    return window


def generation(session_id: str, cache_path: Any) -> int:
    """Return the current window generation for *session_id*."""
    return int(_window(_read_store(cache_path), session_id)["generation"])


def mark_stripped(session_id: str, cache_path: Any) -> None:
    """Record that a stripped payload went out in this window. Never raises.

    Once any stripped payload has gone out, every later failure in the
    window keeps the stripped state (contract C5, including a store-failure
    turn): the strip is no longer undone by a full-catalog resend. The flag is
    held both in the store and in an in-process latch, so a store write
    failure cannot lose it inside this process.
    """
    with _STRIPPED_LOCK:
        _STRIPPED_FALLBACK.add((store_path(cache_path), session_id))
    try:
        store = _read_store(cache_path)
        _window(store, session_id)["stripped"] = True
        _write_store(cache_path, store)
    except Exception:
        logger.debug("hermes-skill-router writeback: mark stripped failed", exc_info=True)


def stripped_sent(session_id: str, cache_path: Any) -> bool:
    """Return True when a stripped payload already went out in this window."""
    with _STRIPPED_LOCK:
        if (store_path(cache_path), session_id) in _STRIPPED_FALLBACK:
            return True
    try:
        return bool(_window(_read_store(cache_path), session_id).get("stripped"))
    except Exception:
        return False


def derive_origin(platform: Any, messages: Any) -> str:
    """Return the origin class for this request from *platform* plus prompt shape.

    The fork shares the parent ``session_id`` with persistence disabled and no
    session DB, and no middleware kwarg distinguishes it, so the prompt shape
    is the only origin signal beyond the platform kwarg. An unrecognized
    review shape is treated as ``normal`` (the documented ceiling: its block
    can become window-visible to the parent, in the over-suppression
    direction).
    """
    if isinstance(platform, str) and platform.strip().lower() == "curator":
        return ORIGIN_CURATOR
    user_text = _newest_user_text(messages)
    if user_text is None:
        return ORIGIN_NORMAL
    for marker in CURATOR_PROMPT_MARKERS:
        if marker in user_text:
            return ORIGIN_CURATOR
    if user_text.lstrip().startswith(REVIEW_PROMPT_PREFIX):
        return ORIGIN_REVIEW
    return ORIGIN_NORMAL


def _newest_user_text(messages: Any) -> Optional[str]:
    """Return the newest user message's text, or ``None`` when there is none."""
    if not isinstance(messages, (list, tuple)):
        return None
    for message in reversed(list(messages)):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "\n".join(parts) or None
        return None
    return None


def block_hash(block: str) -> str:
    """Return the sha256 hex of *block*, the idempotence key for retry writes."""
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def stored_blocks(
    session_id: str, origin: str, gen: int, cache_path: Any
) -> List[Tuple[int, str]]:
    """Return ``(insert_index, block)`` pairs for *session_id*, *origin* and *gen*."""
    window = _window(_read_store(cache_path), session_id)
    result: List[Tuple[int, str]] = []
    for entry in window["blocks"]:
        if not isinstance(entry, dict):
            continue
        if entry.get("origin") != origin or entry.get("generation") != gen:
            continue
        block = entry.get("block")
        index = entry.get("insert_index")
        if isinstance(block, str) and isinstance(index, int):
            result.append((index, block))
    return sorted(result)


def _already_stored(window: Dict[str, Any], origin: str, digest: str) -> bool:
    return any(
        isinstance(entry, dict)
        and entry.get("origin") == origin
        and entry.get("hash") == digest
        for entry in window["blocks"]
    )


def _session_db():
    """Return a fresh ``hermes_state.SessionDB`` handle, or ``None`` when absent.

    The plugin reads the DB through its own handle for the fence and the
    tail match; the host owns the live DB. Any import or construction failure
    fails open (store-only, no append).
    """
    try:
        from hermes_state import SessionDB  # type: ignore

        return SessionDB()
    except Exception:
        logger.debug("hermes-skill-router writeback: no session DB handle", exc_info=True)
        return None


def _db_tail(session_id: str, db: Any) -> Optional[List[Dict[str, Any]]]:
    """Return the session DB's recent messages, or ``None`` on any failure."""
    try:
        rows = db.get_messages(session_id, include_inactive=False, include_compacted=False)
    except Exception:
        return None
    if not isinstance(rows, list):
        return None
    tail: List[Dict[str, Any]] = []
    for row in rows[-8:]:
        if isinstance(row, dict) and isinstance(row.get("role"), str):
            tail.append({"role": row["role"], "content": row.get("content")})
    return tail


def _payload_tail(messages: Any, block: str = "") -> List[Dict[str, Any]]:
    """Return the payload's recent ``{role, content}`` messages, oldest first.

    A trailing message whose content is *block* itself is the fresh block the
    rewrite just appended; it is not part of the history tail the DB compares
    against, so it is excluded.
    """
    tail: List[Dict[str, Any]] = []
    if not isinstance(messages, (list, tuple)):
        return tail
    items = list(messages)
    if block and items and isinstance(items[-1], dict) and items[-1].get("content") == block:
        items = items[:-1]
    for message in items[-8:]:
        if isinstance(message, dict) and isinstance(message.get("role"), str):
            tail.append({"role": message["role"], "content": message.get("content")})
    return tail


def admit_block(
    payload: Any,
    session_id: str,
    block: str,
    insert_index: int,
    origin: str,
    cache_path: Any,
) -> bool:
    """Store *block* for *session_id* and DB-append it when the guards pass.

    Returns ``True`` when the block is admitted (stored, and appended when
    origin is ``normal``). The store write is keyed by block hash, so a
    retried middleware call admits once. The DB append runs only for origin
    ``normal`` and only when the payload history tail matches the session DB
    tail; while a compression is in flight (``get_compression_lock_holder``
    non-empty through M8's own handle) the append is skipped and the block
    lives store-only, generation-tagged. Any failure keeps the admitted blocks
    and returns ``False``; it never raises.
    """
    if not isinstance(block, str) or not block:
        return False
    # An empty key is the single-anonymous bucket (search._key's "" fallback):
    # a caller with neither id still gets a window, never a dropped block.
    try:
        store = _read_store(cache_path)
        window = _window(store, session_id)
        gen = int(window["generation"])
        digest = block_hash(block)
        if _already_stored(window, origin, digest):
            return True
        window["blocks"].append(
            {
                "block": block,
                "insert_index": int(insert_index),
                "origin": origin,
                "generation": gen,
                "hash": digest,
            }
        )
        if not _write_store(cache_path, store):
            window["blocks"].pop()
            return False
        if origin != ORIGIN_NORMAL:
            return True
        messages = payload.get("messages") if isinstance(payload, dict) else None
        db = _session_db()
        if db is None:
            return True
        try:
            holder = db.get_compression_lock_holder(session_id)
        except Exception:
            holder = None
        if holder:
            # A compression is in flight: the append would race the clone-back,
            # so the block stays store-only for this generation.
            return True
        tail = _db_tail(session_id, db)
        if tail is None or tail != _payload_tail(messages, block):
            logger.debug(
                "hermes-skill-router writeback: payload tail does not match the DB tail; store-only"
            )
            return True
        try:
            row_id = db.append_message(session_id, "system", content=block)
        except Exception:
            logger.debug("hermes-skill-router writeback: append failed", exc_info=True)
            return False
        window["row_ids"].append(int(row_id))
        _write_store(cache_path, store)  # best effort; the row is already durable
        return True
    except Exception:
        logger.debug("hermes-skill-router writeback: admit failed", exc_info=True)
        return False


def expire_window(session_id: str, cache_path: Any) -> None:
    """Advance the window generation and deactivate the window's block rows.

    Called from ``post_compaction``: every appended block row id is re-resolved
    with ``resolve_active_row_id`` (a race append may have been cloned with a
    new id) and the result deactivated with ``deactivate_message``, so no
    plugin block row survives past the close of any window. The store entry
    expires as the generation advances. Never raises.
    """
    with _STRIPPED_LOCK:
        _STRIPPED_FALLBACK.discard((store_path(cache_path), session_id))
    try:
        store = _read_store(cache_path)
        window = _window(store, session_id)
        row_ids = [r for r in window["row_ids"] if isinstance(r, int)]
        db = _session_db() if row_ids else None
        if db is not None:
            for row_id in row_ids:
                try:
                    active = db.resolve_active_row_id(session_id, row_id)
                    if active is not None:
                        db.deactivate_message(session_id, active)
                except Exception:
                    logger.debug(
                        "hermes-skill-router writeback: row cleanup failed", exc_info=True
                    )
        window["generation"] = int(window["generation"]) + 1
        window["blocks"] = []
        window["row_ids"] = []
        window["stripped"] = False
        _write_store(cache_path, store)
    except Exception:
        logger.debug("hermes-skill-router writeback: expire failed", exc_info=True)


def forget_session_ids(session_id: str, cache_path: Any) -> None:
    """Drop all store state for *session_id* (end of session). Never raises."""
    with _STRIPPED_LOCK:
        _STRIPPED_FALLBACK.discard((store_path(cache_path), session_id))
    try:
        store = _read_store(cache_path)
        store["windows"].pop(session_id, None)
        _write_store(cache_path, store)
    except Exception:
        logger.debug("hermes-skill-router writeback: forget failed", exc_info=True)
