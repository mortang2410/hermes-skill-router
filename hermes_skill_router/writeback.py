"""Store-only shortlist bookkeeping with an origin-tagged plugin store (M8, C1).

The shortlist is appended to the one request the rewrite is building and to no
later request. This module owns a JSON store beside the index cache that keeps
the rendered block bytes, the names the block displays, its origin class and
its window generation for bookkeeping, retry deduplication and display records
only. It is never a suppression input: decision-time suppression reads C9
structured tool records. No stored block is replayed into a later request, and
this module never opens a session database handle, appends a history row, reads
a compression lock or cleans up a row. Writes are keyed by block hash, so a
retried middleware call records once. Any store failure fails open and never
raises to the model call.

Origin classes come from the middleware ``platform`` kwarg plus prompt shape
(no middleware kwarg distinguishes the fork; turn ids are uuids): ``curator``
when the platform is curator or the user message carries the curator prompt
shape, ``review`` when the user message starts with the background-review
prompt shape, else ``normal``. Every origin is store-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

#: In-process record of the cache paths a session has written stores beside
#: (contract C5 fence): the operator's ``cachePath`` can change between
#: writes while the plugin process keeps running, and ``post_compaction``
#: must expire every store the session actually wrote to, not only the store
#: the current setting names. A path is recorded at ``admit_block`` and
#: ``mark_stripped`` time, and the record is dropped on ``expire_window`` and
#: ``forget_session_ids``. Bounded by the sessions that wrote in this
#: process; keyed by session id alone, the same granularity as the windows.
_WRITTEN_PATHS: Dict[str, List[Any]] = {}

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


def _record_path(session_id: str, cache_path: Any) -> None:
    """Remember that *session_id* wrote beside *cache_path*. Never raises."""
    if not isinstance(cache_path, str) or not cache_path.strip():
        return
    with _STRIPPED_LOCK:
        paths = _WRITTEN_PATHS.setdefault(session_id, [])
        if cache_path not in paths:
            paths.append(cache_path)


def _take_paths(session_id: str, fallback: Any) -> List[Any]:
    """Return the recorded paths for *session_id* plus *fallback*, and drop them.

    Paths are deduplicated by resolved store, so a store named by two
    recorded paths (an env override, a settings change back and forth) is
    expired once, never twice, and a session with no record falls back to
    the single configured path.
    """
    with _STRIPPED_LOCK:
        recorded = _WRITTEN_PATHS.pop(session_id, [])
    paths: List[Any] = []
    seen: Set[Optional[str]] = set()
    for candidate in [*recorded, fallback]:
        resolved = store_path(candidate)
        if resolved not in seen:
            seen.add(resolved)
            paths.append(candidate)
    return paths


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
        window = {"generation": 0, "blocks": []}
        windows[session_id] = window
    if not isinstance(window.get("blocks"), list):
        window["blocks"] = []
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
    _record_path(session_id, cache_path)
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


def admit_block(
    session_id: str,
    block: str,
    names: Sequence[str],
    insert_index: int,
    origin: str,
    cache_path: Any,
) -> bool:
    """Record *block* and its displayed names in the plugin store. Never raises.

    Returns ``True`` when the shortlist was recorded. Writes are keyed by block
    hash, so a retried middleware call records once. No session database handle
    is opened and no history row is written for any origin.
    """
    if not isinstance(block, str) or not block:
        return False
    if store_path(cache_path) is None:
        # No store is configured, which is not a write failure: the block is
        # bookkeeping-only, so the caller still injects it and nothing is recorded.
        return True
    _record_path(session_id, cache_path)
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
                "names": [name for name in names if isinstance(name, str)],
                "insert_index": int(insert_index),
                "origin": origin,
                "generation": gen,
                "hash": digest,
            }
        )
        if not _write_store(cache_path, store):
            window["blocks"].pop()
            return False
        return True
    except Exception:
        logger.debug("hermes-skill-router writeback: admit failed", exc_info=True)
        return False


def stored_names(session_id: str, origin: str, gen: int, cache_path: Any) -> List[str]:
    """Return displayed names recorded for a window, for bookkeeping only."""
    window = _window(_read_store(cache_path), session_id)
    names: List[str] = []
    for entry in window["blocks"]:
        if not isinstance(entry, dict):
            continue
        if entry.get("origin") != origin or entry.get("generation") != gen:
            continue
        for name in entry.get("names") or ():
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def expire_window(session_id: str, cache_path: Any) -> None:
    """Advance the window generation and drop its stored shortlists. Never raises.

    Called from ``post_compaction``. Every store the session wrote to is expired,
    not only the one the current setting names. No database row is touched.
    """
    for path in _take_paths(session_id, cache_path):
        _expire_at(session_id, path)


def _expire_at(session_id: str, cache_path: Any) -> None:
    """Expire the window for *session_id* in the store at *cache_path*."""
    with _STRIPPED_LOCK:
        _STRIPPED_FALLBACK.discard((store_path(cache_path), session_id))
    try:
        store = _read_store(cache_path)
        if session_id not in store.get("windows", {}):
            return
        window = _window(store, session_id)
        window["generation"] = int(window["generation"]) + 1
        window["blocks"] = []
        window["stripped"] = False
        _write_store(cache_path, store)
    except Exception:
        logger.debug("hermes-skill-router writeback: expire failed", exc_info=True)


def forget_session_ids(session_id: str, cache_path: Any) -> None:
    """Drop all store state for *session_id* (end of session). Never raises."""
    with _STRIPPED_LOCK:
        _STRIPPED_FALLBACK.discard((store_path(cache_path), session_id))
        _WRITTEN_PATHS.pop(session_id, None)
    try:
        store = _read_store(cache_path)
        store["windows"].pop(session_id, None)
        _write_store(cache_path, store)
    except Exception:
        logger.debug("hermes-skill-router writeback: forget failed", exc_info=True)
