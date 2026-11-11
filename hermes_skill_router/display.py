"""The display record: what the router last suggested, names only (module M11, contract C9).

The record is the whole display contract. It is one small JSON file at
``cfg.suggestions_path`` holding the decision for the newest routed turn or model search::

    {"schema": 2, "at": "2026-11-10T07:20:31Z", "session_id": "…", "turn_id": "…",
     "status": "routed" | "inert", "origin": "turn" | "search", "names": ["…", …]}

``names`` are the selected suggestion names only. A kept row is listed in the injected block but
never recorded, so the pane and the TUI widget can never present a pinned or deferred-excluded
skill as something the router picked. No request text is stored: the record is names plus the
identifiers of the decision that produced it. ``origin`` says who asked: ``turn`` for the automatic
pre-request pass, ``search`` for a model call to the search tool, so the surfaces can mark a lookup.

Rules that exist because a display must not cost a token or a crash:

* A write goes through a temporary file in the target directory and one ``os.replace``, so a reader
  never sees a half-written record and a crashed writer leaves nothing behind.
* A record identical to the one already on disk is not rewritten. Identity ignores ``at``, so a
  turn that routes to the same names keeps the timestamp of the decision that first produced them.
  ``origin`` is part of that identity, so a search with the same names still rewrites the marker.
* ``cfg.show_suggestions`` false removes an existing record instead of writing one, which is what
  turns both surfaces idle. ``read(cfg)`` answers ``None`` in that state before touching the file,
  so the route and the desktop pane stay idle even when a leftover record survives a refused
  removal; only the TUI widget reads the file directly.
* Every failure is swallowed and logged once per process: the rewrite return value never depends on
  the display, and an unreadable file reads as ``None`` (idle) rather than raising in a reader.

The record never appears in a provider request. :class:`~hermes_skill_router.router.rewrite` calls
:func:`record` after it has built the payload it returns, so the display cannot add a message, a
system-prompt section or a token.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import Config

logger = logging.getLogger(__name__)

#: Record file schema. Bumped when the file's shape changes; a schema 1 file still reads.
SCHEMA = 2

#: The schema a pre-origin writer produced, read as an origin of :data:`TURN`.
LEGACY_SCHEMA = 1

#: The two decisions a record can carry: a routed rewrite, or an inert request that kept the catalog.
ROUTED = "routed"
INERT = "inert"
STATUSES = (ROUTED, INERT)

#: Who asked for the decision: the automatic pre-request pass, or a model call to the search tool.
TURN = "turn"
SEARCH = "search"
ORIGINS = (TURN, SEARCH)

#: The keys a record carries, in the order it is written.
FIELDS = ("schema", "at", "session_id", "turn_id", "status", "origin", "names")

#: A reader argument: the plugin config, whose ``suggestions_path`` is used, or that path itself.
Target = Union[Config, str, "os.PathLike[str]"]

#: Reasons already logged, so one broken path logs once instead of once per turn.
_LOGGED: set[str] = set()


def path_for(target: Target) -> Optional[str]:
    """Return the absolute record path for *target*, or ``None`` when there is nowhere to write.

    A :class:`Config` contributes its ``suggestions_path``; a string is the path itself. ``~`` is
    expanded so a configured default works from any working directory, and an empty path means the
    record is disabled, which reads and writes as idle.
    """
    raw = target.suggestions_path if isinstance(target, Config) else target
    if not isinstance(raw, str):
        try:
            raw = os.fspath(raw)  # type: ignore[arg-type]
        except TypeError:
            return None
    text = raw.strip()
    if not text:
        return None
    return os.path.abspath(os.path.expanduser(text))


def record(
    cfg: Config,
    *,
    status: str,
    origin: str = TURN,
    names: Sequence[str] = (),
    session_id: str = "",
    turn_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Write the newest decision and return the record, or ``None`` when nothing was written.

    *status* is ``"routed"`` for a rewritten payload and ``"inert"`` for a request returned
    unchanged. *origin* is ``"turn"`` for the automatic pass and ``"search"`` for a model search,
    and defaults to ``"turn"`` so the existing turn call sites that pass no origin keep working.
    A search record carries no turn id, so *turn_id* is stored as ``""`` whatever a caller passes.
    *names* are the selected suggestion names; anything past ``cfg.max_display_names`` is dropped,
    duplicates keep their first place, and blanks are ignored. With ``cfg.show_suggestions`` false
    an existing record is removed and nothing is written, for either origin.

    Returns ``None`` for a skip (an identical record, a disabled or empty path, a write failure) so
    a caller can tell a real write, which is worth broadcasting, from a no-op.
    """
    if status not in STATUSES:
        _log_once(f"status:{status}", f"skill display: refusing record status {status!r}")
        return None
    if origin not in ORIGINS:
        _log_once(f"origin:{origin}", f"skill display: refusing record origin {origin!r}")
        return None
    if not cfg.show_suggestions:
        _clear(cfg)
        return None
    path = path_for(cfg)
    if path is None:
        logger.debug("skill display: suggestionsPath is empty; the record is disabled")
        return None
    entry = {
        "schema": SCHEMA,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": _text(session_id),
        # A lookup belongs to no turn, so a search record never carries a turn id.
        "turn_id": "" if origin == SEARCH else _text(turn_id),
        "status": status,
        "origin": origin,
        # An inert decision suggested nothing, so names are dropped even if a caller passes some.
        "names": [] if status == INERT else _names(names, cfg.max_display_names),
    }
    previous = read(cfg)
    if previous is not None and _same(previous, entry):
        return None
    if not _write(path, entry):
        return None
    return entry


def read(target: Target) -> Optional[Dict[str, Any]]:
    """Return the record at *target*, or ``None`` when it is absent, unreadable or not a record.

    Only the known fields are returned, and a file whose schema, status, origin or name list is
    wrong reads as idle rather than as a partial record, so both display surfaces see one shape or
    nothing. A schema 1 file still reads, with origin :data:`TURN`, so an installed older writer
    does not blank a newer pane.

    With a :class:`Config` whose ``show_suggestions`` is false this answers ``None`` before touching
    the file: the setting is authoritative for the route and the desktop pane, so a leftover record
    that survives a refused removal still cannot reach them.
    """
    if isinstance(target, Config) and not target.show_suggestions:
        return None
    path = path_for(target)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        _log_once(f"read:{path}", f"skill display: cannot read {path}: {exc}")
        return None
    if not isinstance(data, dict):
        _log_once(f"shape:{path}", f"skill display: {path} does not hold a record object")
        return None
    schema = data.get("schema")
    status = data.get("status")
    names = data.get("names")
    if schema not in (LEGACY_SCHEMA, SCHEMA) or status not in STATUSES:
        _log_once(
            f"shape:{path}",
            f"skill display: {path} does not hold a schema 1 or schema 2 record",
        )
        return None
    origin = TURN if schema == LEGACY_SCHEMA else data.get("origin")
    if origin not in ORIGINS:
        _log_once(f"shape:{path}", f"skill display: {path} does not hold a schema 2 origin")
        return None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        _log_once(f"shape:{path}", f"skill display: {path} does not hold a name list")
        return None
    return {
        "schema": SCHEMA,
        "at": _text(data.get("at")),
        "session_id": _text(data.get("session_id")),
        "turn_id": _text(data.get("turn_id")),
        "status": status,
        "origin": origin,
        "names": list(names),
    }


def remove(target: Target) -> str:
    """Delete the record at *target*; report ``removed``, ``missing`` or ``refused``.

    The three outcomes are deliberately distinct because the writer acts on them: a missing record
    means no file exists to hide, while a refused removal means the file may still be there with
    its names, which is the one case where the writer replaces it with an inert record. Never raises.
    """
    path = path_for(target)
    if path is None:
        return "missing"
    try:
        os.unlink(path)
    except FileNotFoundError:
        return "missing"
    except OSError as exc:
        _log_once(f"remove:{path}", f"skill display: cannot remove {path}: {exc}")
        return "refused"
    return "removed"


def _clear(cfg: Config) -> str:
    """Remove the record because ``showSuggestions`` is false; return the removal outcome.

    A refused removal of an existing record is the one case where the file can still carry names the
    operator hid. The writer then replaces it with an inert, empty record, so the file-only TUI
    widget shows no names; if that write also fails, both failures are logged with their paths and
    the old file survives, which the spec records as that widget's limit.
    """
    path = path_for(cfg)
    if path is None:
        return "missing"
    outcome = remove(path)
    if outcome != "refused":
        return outcome
    replacement = {
        "schema": SCHEMA,
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": "",
        "turn_id": "",
        "status": INERT,
        "origin": TURN,
        "names": [],
    }
    if not _write(path, replacement):
        _log_once(
            f"clear:{path}",
            f"skill display: the inert replacement at {path} also failed; the previous record "
            "survives and only the file-reading widget can still show its names",
        )
    return outcome


def _write(path: str, entry: Dict[str, Any]) -> bool:
    """Write *entry* beside *path* and rename it into place; log once and return False on failure."""
    directory = os.path.dirname(path) or "."
    temporary: Optional[str] = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".suggestions-",
            suffix=".tmp",
            delete=False,
        )
        temporary = handle.name
        with handle:
            json.dump(entry, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        _log_once(f"write:{path}", f"skill display: cannot write {path}: {exc}")
        return False
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return True


def _names(names: Sequence[str], limit: int) -> List[str]:
    """Return at most *limit* distinct, non-empty names, in the order they were given."""
    kept: List[str] = []
    cap = max(0, limit)
    for name in names:
        if len(kept) >= cap:
            break
        if not isinstance(name, str):
            continue
        text = name.strip()
        if text and text not in kept:
            kept.append(text)
    return kept


def _same(previous: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """Return whether *entry* says what *previous* already says; ``at`` is deliberately ignored."""
    for key in ("schema", "session_id", "turn_id", "status", "origin", "names"):
        if previous.get(key) != entry.get(key):
            return False
    return True


def _text(value: Any) -> str:
    """Return *value* as a string, and the empty string for anything else."""
    return value if isinstance(value, str) else ""


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.warning(message)
