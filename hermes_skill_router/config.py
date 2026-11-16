"""Configuration for the hermes-skill-router plugin (module M2, contract C2).

``load`` is the only place that decides what a configuration value means: every field falls back
to its default, with a note naming the key, when the value is malformed or out of bounds, and
``load`` never raises. Endpoint roots must be safe: a plain ``http`` URL is accepted only for a
loopback host, any other host must use ``https``, and a URL carrying credentials, a query or a
fragment is rejected. A note about a rejected value shows only the value's origin, never its
userinfo, path, query or fragment.

Config traps (every ``notes`` entry) also reach the display record, but only when newly seen:
``load`` reports the traps not yet in the persisted set of ``(trap_id, fingerprint)`` pairs beside
the suggestions file, and only a display-record write that actually carries a trap persists its
fingerprint (:func:`acknowledge_trap_notes`, called by the display writer), so an unchanged
offending value stays silent forever and a changed value warns once. A load that writes no record
(disabled, dashboard or pre-load read, any non-routing path) never consumes a fingerprint. The
persisted set never uses a clock.

The one no-record switch that warns (``showSuggestions: false``) warns on three
channels under the same fingerprint shape, one seen set per switch: a terminal line on stderr, an
injected context message built by :func:`switch_warning_messages`, and one host-logger event, each
naming the switch and its effect. Unlike a malformed-value trap it never reaches a display
record (the switch suppresses it), so ``load`` persists its fingerprints at emission time
instead of waiting for a record write; fixing it drops only its fingerprints.
The fingerprints live beside the index cache when a cache path is set, and in the default
``~/.cache/hermes-skill-router`` directory when ``cachePath`` is empty, so disabling the
display or the index cache never turns them back into per-load spam.

``enabled: false`` is surveyed by D2 (off stays untouched) and is different: :func:`_pending_switch_warnings`
returns at once when the master switch is off, so a load with routing off emits no switch warning
for either switch (not even a ``showSuggestions`` one) and persists no switch-warning state. The
prior three-channel warning on that path is superseded; the ``showSuggestions`` warning on the
enabled path is unchanged.

The injected channel is tracked separately from the load-time seen set: ``load`` consumes
only the terminal/logger set, while the injected fingerprint is consumed only by
:func:`acknowledge_switch_deliveries`, called once a valid outgoing request actually
carries the message on a supported channel. A tool-first load therefore defers the
injected warning instead of permanently suppressing it, and every no-rewrite path that
reaches a supported request delivers at the first request that carries messages
(no-catalog, first failure, already-injected included); only an unsupported api mode or
a malformed request without a messages list defers it. Repeats are suppressed only
after that delivery, and an ack is accepted only at the generation its note was
captured at, so a pre-fix note can never suppress a post-fix delivery.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

try:
    import fcntl
except ImportError:  # non-POSIX platforms: the thread lock below still serializes
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

#: Exact text the reference skill router puts before ``<name> - <description>`` in a rerank document.
DOCUMENT_PREFIX = (
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
    "actually need to carry out the current task correctly. Skill: "
)

#: Hosts a plain ``http`` endpoint URL may name.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Rerank request/response shapes this plugin implements.
RERANK_PROVIDERS = ("cohere",)

#: The fusion modes. ``cosine`` is the single rollback switch.
FUSION_MODES = ("live", "shadow", "cosine")

#: The one-shot skill-creation modes. The default keeps Hermes' no-create rule;
#: the opt-in value lets the one-shot header allow creating and editing skills.
ONE_SHOT_MODES = ("oneshot_no_create_skill", "oneshot_can_create_skill")

#: The curation modes. ``default_curation`` leaves the curator fork's full catalog
#: untouched; ``delta_curation`` narrows it to skills changed since the last
#: curation (contract C4).
CURATION_MODES = ("default_curation", "delta_curation")

_KIND_LABELS = {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "text": "string",
    "path": "path",
    "file_path": "path",
    "name_list": "list of skill names",
    "provider": "provider name",
    "fusion_mode": "fusion mode (live, shadow or cosine)",
    "one_shot_mode": "one-shot mode (oneshot_no_create_skill or oneshot_can_create_skill)",
    "curation_mode": "curation mode (default_curation or delta_curation)",
    "url": "endpoint URL",
}


@dataclass(frozen=True)
class _Spec:
    """One settings key: its manifest name, Config field, value kind and bounds."""

    key: str
    field: str
    kind: str
    default: Any
    lower: Optional[float] = None
    upper: Optional[float] = None


# Manifest key order follows the spec's config table.
_SPECS: Tuple[_Spec, ...] = (
    _Spec("enabled", "enabled", "bool", True),
    _Spec("eagerSkills", "eager_skills", "name_list", ()),
    _Spec("embedBaseUrl", "embed_base_url", "url", "http://127.0.0.1:8012"),
    _Spec("embedPath", "embed_path", "path", "/v1/embeddings"),
    _Spec("embedModelId", "embed_model_id", "text", "bge-m3"),
    _Spec("embedTimeoutMs", "embed_timeout_ms", "int", 6000, 100, 60000),
    _Spec("rerankBaseUrl", "rerank_base_url", "url", "http://127.0.0.1:8010"),
    _Spec("rerankPath", "rerank_path", "path", "/v1/rerank"),
    _Spec("rerankModel", "rerank_model", "text", "tasksource-decider-nano"),
    _Spec("rerankProvider", "rerank_provider", "provider", "cohere"),
    _Spec("documentPrefix", "document_prefix", "text", DOCUMENT_PREFIX),
    _Spec("deferredSkills", "deferred_skills", "name_list", ()),
    _Spec("rerankTimeoutMs", "rerank_timeout_ms", "int", 20000, 100, 60000),
    # One routing pass, from the query embed through the rerank reply. It starts after the
    # index is loaded or built, so it never covers the ~38 s cold build, and it is larger
    # than 6000 + 20000, the sum of the two per-request timeouts, so a working but slow
    # pass is never cut. Bounds are the same shape as the per-request ones.
    _Spec("routingTimeoutMs", "routing_timeout_ms", "int", 30000, 1000, 60000),
    _Spec("shortlistSize", "shortlist_size", "int", 50, 1, 383),
    _Spec("cutoff", "cutoff", "float", 0.01, 0.0, 1.0),
    # Zero means unlimited at every reader, never "select nothing".
    _Spec("maxSuggestions", "max_suggestions", "int", 10, 0, 50),
    # The fused shortlist: mode, one switch per channel and the bounded recency bump. The two
    # numeric keys are the only ones with a numeric bound; the mode and the switches are bounded
    # by their kind, and the store path by the path rule.
    _Spec("fusionMode", "fusion_mode", "fusion_mode", "live"),
    _Spec("channelsSemantic", "channels_semantic", "bool", True),
    _Spec("channelsBm25", "channels_bm25", "bool", True),
    _Spec("channelsAssociations", "channels_associations", "bool", True),
    _Spec("frecencyHalfLifeDays", "frecency_half_life_days", "int", 14, 1, 3650),
    _Spec("frecencyCapPositions", "frecency_cap_positions", "int", 5, 0, 50),
    _Spec("maxDisplayNames", "max_display_names", "int", 12, 1, 50),
    _Spec("maxDescriptionChars", "max_description_chars", "int", 400, 3, 4000),
    _Spec("cachePath", "cache_path", "file_path", "~/.cache/hermes-skill-router/index.json"),
    # Empty puts associations.db beside cachePath, and nowhere when cachePath is empty.
    _Spec("associationDbPath", "association_db_path", "file_path", ""),
    _Spec("showSuggestions", "show_suggestions", "bool", True),
    _Spec(
        "suggestionsPath",
        "suggestions_path",
        "file_path",
        "~/.cache/hermes-skill-router/suggestions.json",
    ),
    _Spec("healthCheckMs", "health_check_ms", "int", 60000, 0, 3600000),
    # One-shot skill-creation rule: the no-create default keeps Hermes' rule,
    # the opt-in value lets the one-shot header allow creating and editing skills.
    _Spec("oneShotMode", "one_shot_mode", "one_shot_mode", "oneshot_no_create_skill"),
    # Curator narrowing: default leaves the fork's full catalog untouched,
    # the opt-in value narrows both candidate surfaces to the delta set (C4).
    _Spec("curation", "curation_mode", "curation_mode", "default_curation"),
)

#: Manifest key -> default value.
DEFAULTS: dict[str, Any] = {spec.key: spec.default for spec in _SPECS}

#: Manifest key -> inclusive numeric bound, for the keys that carry one.
BOUNDS: dict[str, Tuple[float, float]] = {
    spec.key: (spec.lower, spec.upper) for spec in _SPECS if spec.lower is not None
}


#: File name of the persisted config-trap set, written beside the suggestions file.
TRAPS_FILENAME = "config-traps.json"

#: File name of the persisted switch-warning seen set, written beside the index cache.
SWITCHES_FILENAME = "config-switches.json"

#: Environment override for the traps file; tests point it at a temporary file.
TRAPS_ENV_VAR = "HERMES_SKILL_ROUTER_TRAPS_PATH"

#: Environment override for the switch-warning seen file; tests point it at a temporary file.
SWITCHES_ENV_VAR = "HERMES_SKILL_ROUTER_SWITCHES_PATH"

#: The two master switches whose off state stops a record from being written.
SWITCH_WARNING_SWITCHES = ("enabled", "showSuggestions")

#: What a switch being off costs, one effect per switch.
_SWITCH_WARNING_EFFECTS = {
    "enabled": (
        "routing is switched off, the router changes nothing and writes no record; "
        "set enabled: true to restore routing"
    ),
    "showSuggestions": (
        "suggestion names are hidden, the display record is removed and the pane stays idle; "
        "set showSuggestions: true to restore the display"
    ),
}

#: Manifest switch key -> Config field.
_SWITCH_FIELDS = {"enabled": "enabled", "showSuggestions": "show_suggestions"}


@dataclass(frozen=True)
class Config:
    """Every plugin setting with its default applied, plus the notes ``load`` produced."""

    enabled: bool
    embed_base_url: str
    embed_path: str
    embed_model_id: str
    embed_timeout_ms: int
    rerank_base_url: str
    rerank_path: str
    rerank_model: str
    rerank_provider: str
    rerank_timeout_ms: int
    routing_timeout_ms: int
    shortlist_size: int
    cutoff: float
    max_suggestions: int
    fusion_mode: str
    channels_semantic: bool
    channels_bm25: bool
    channels_associations: bool
    frecency_half_life_days: int
    frecency_cap_positions: int
    max_description_chars: int
    cache_path: str
    association_db_path: str
    health_check_ms: int
    show_suggestions: bool
    suggestions_path: str
    max_display_names: int
    document_prefix: str
    eager_skills: Tuple[str, ...]
    deferred_skills: Tuple[str, ...]
    notes: Tuple[str, ...]
    one_shot_mode: str = "oneshot_no_create_skill"
    curation_mode: str = "default_curation"
    trap_notes: Tuple[str, ...] = ()
    switch_message_notes: Tuple[str, ...] = ()
    switch_generations: Dict[str, int] = field(default_factory=dict)


def is_safe_base_url(value: Any) -> bool:
    """Return whether *value* may be used as an endpoint root.

    ``http`` is accepted only for a loopback host, any other host must use ``https``, and a URL
    carrying credentials, a query or a fragment is rejected, as is anything that is not an
    absolute http(s) URL. The request path is appended to this root, so a query or fragment would
    be swallowed and then echoed on failure.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    if not host:
        return False
    if parts.username is not None or parts.password is not None:
        return False
    if parts.query or parts.fragment:
        return False
    if parts.scheme == "http" and host.lower() not in LOOPBACK_HOSTS:
        return False
    return True


#: Prefix that names a URL scheme, e.g. ``https:`` in ``https://host``.
_SCHEME_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")

#: Characters that end a URL's authority, so an ``@`` after one is not userinfo.
_AUTHORITY_END = ("/", "?", "#")


def _carries_credentials(value: str) -> bool:
    """Return whether *value* puts an ``@`` in its authority, the shape that can carry a password.

    Handles values without ``://`` too: ``user:pw@host``, ``//user:pw@host`` and
    ``https:/user:pw@host`` all carry a credential in their authority.
    """
    text = value.strip()
    scheme = _SCHEME_PREFIX.match(text)
    if scheme:
        text = text[scheme.end():].lstrip("/")
    elif text.startswith("//"):
        text = text.lstrip("/")
    for separator in _AUTHORITY_END:
        cut = text.find(separator)
        if cut >= 0:
            text = text[:cut]
    return "@" in text


def _redact_url(value: str) -> str:
    """Return a log-safe form of *value*, never carrying userinfo, path, query or fragment.

    A rejected endpoint URL can carry credentials, and the note that explains the rejection is
    logged, so the value is reduced to its origin. A value whose authority carries an ``@`` but
    that does not parse as a URL becomes ``<redacted URL>``. Strings without either shape pass
    through unchanged.
    """
    text = value.strip()
    if "://" in text:
        try:
            parts = urlsplit(text)
            netloc = parts.netloc.rsplit("@", 1)[-1]
        except ValueError:
            return "<redacted URL>"
        if not parts.scheme or not netloc:
            return "<redacted URL>"
        return f"{parts.scheme}://{netloc}"
    if _carries_credentials(text):
        return "<redacted URL>"
    return value


def _redact_value(value: Any) -> Any:
    """Return *value* with every URL-shaped string reduced to its origin."""
    if isinstance(value, str):
        return _redact_url(value)
    if isinstance(value, Mapping):
        return {_redact_value(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _shown(value: Any) -> str:
    """Return a repr of *value* that is safe to put in a log line."""
    return repr(_redact_value(value))


def _invalid(spec: _Spec, value: Any, notes: list[str]) -> Any:
    """Record a malformed-value note and return the key's default."""
    notes.append(
        f"{spec.key}: {_shown(value)} is not a valid {_KIND_LABELS[spec.kind]}; "
        f"using default {_shown(spec.default)}"
    )
    return spec.default


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "on", "1"):
            return True
        if low in ("false", "no", "off", "0"):
            return False
    return None


def _coerce(spec: _Spec, value: Any, notes: list[str]) -> Any:
    """Return *value* coerced for *spec*, or its default with a note when it is unusable."""
    kind = spec.kind
    if kind == "bool":
        flag = _as_bool(value)
        return _invalid(spec, value, notes) if flag is None else flag
    if kind in ("int", "float"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _invalid(spec, value, notes)
        if kind == "int":
            if isinstance(value, float):
                if not value.is_integer():
                    return _invalid(spec, value, notes)
                value = int(value)
        else:
            value = float(value)
        if not (spec.lower <= value <= spec.upper):  # type: ignore[operator]
            notes.append(
                f"{spec.key}: {_shown(value)} is outside {spec.lower}..{spec.upper}; "
                f"using default {_shown(spec.default)}"
            )
            return spec.default
        return value
    if kind == "text":
        if isinstance(value, str) and value.strip():
            return value
        return _invalid(spec, value, notes)
    if kind == "path":
        if isinstance(value, str) and value.strip().startswith("/"):
            return value.strip()
        return _invalid(spec, value, notes)
    if kind == "file_path":
        if isinstance(value, str):
            return value.strip()
        return _invalid(spec, value, notes)
    if kind == "name_list":
        if not isinstance(value, (list, tuple)):
            return _invalid(spec, value, notes)
        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                return _invalid(spec, value, notes)
            name = item.strip()
            if name and name not in names:
                names.append(name)
        return tuple(names)
    if kind == "provider":
        if isinstance(value, str) and value.strip().lower() in RERANK_PROVIDERS:
            return value.strip().lower()
        return _invalid(spec, value, notes)
    if kind == "fusion_mode":
        if isinstance(value, str) and value.strip().lower() in FUSION_MODES:
            return value.strip().lower()
        return _invalid(spec, value, notes)
    if kind == "one_shot_mode":
        if isinstance(value, str) and value.strip().lower() in ONE_SHOT_MODES:
            return value.strip().lower()
        return _invalid(spec, value, notes)
    if kind == "curation_mode":
        if isinstance(value, str) and value.strip().lower() in CURATION_MODES:
            return value.strip().lower()
        return _invalid(spec, value, notes)
    if kind == "url":
        if is_safe_base_url(value):
            return value.strip().rstrip("/")
        notes.append(
            f"{spec.key}: {_shown(value)} is not a safe endpoint URL (http is allowed only for a "
            "loopback host, other hosts must use https, credentials are rejected, and a query "
            "or fragment is not allowed); "
            f"using default {_shown(spec.default)}"
        )
        return spec.default
    return _invalid(spec, value, notes)


#: Notes already reported, so a rejected or unknown setting is logged once per process.
_LOGGED_NOTES: set[str] = set()


def switches_path_for(cache_path: Any) -> Optional[str]:
    """Return the persisted switch-warning seen-set path for *cache_path*.

    The set lives beside the index cache when a cache path is set, so an
    empty ``suggestionsPath`` (display disabled) still suppresses: the
    location is independent of the optional display path. An empty cache
    path (persistence off for the index) falls back to the default
    ``~/.cache/hermes-skill-router`` directory beside the store/config
    area, so ``cachePath: ""`` still suppresses once per value instead of
    warning on every load. ``HERMES_SKILL_ROUTER_SWITCHES_PATH`` wins when set.
    """
    override = os.environ.get(SWITCHES_ENV_VAR) or ""
    if override.strip():
        return os.path.abspath(os.path.expanduser(override.strip()))
    if isinstance(cache_path, str) and cache_path.strip():
        directory = os.path.dirname(os.path.abspath(os.path.expanduser(cache_path.strip())))
        if not directory:
            directory = os.getcwd()
        return os.path.join(directory, SWITCHES_FILENAME)
    return os.path.join(
        os.path.expanduser("~/.cache/hermes-skill-router"), SWITCHES_FILENAME
    )


def traps_path_for(suggestions_path: Any) -> Optional[str]:
    """Return the persisted trap-set path for *suggestions_path*, or ``None`` when nowhere to write.

    The set lives beside the suggestions file, so moving the record moves the
    suppression with it. An empty path means the display is disabled, which
    also disables persistence. ``HERMES_SKILL_ROUTER_TRAPS_PATH`` wins when set.
    """
    override = os.environ.get(TRAPS_ENV_VAR) or ""
    if override.strip():
        return os.path.abspath(os.path.expanduser(override.strip()))
    if not isinstance(suggestions_path, str):
        return None
    text = suggestions_path.strip()
    if not text:
        return None
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(text)))
    if not directory:
        directory = os.getcwd()
    return os.path.join(directory, TRAPS_FILENAME)


def _trap_key(note: str) -> Tuple[str, str]:
    """Return the persisted ``(trap_id, fingerprint)`` identity for one config note.

    *trap_id* is the settings key named before the first colon; *fingerprint* is
    a sha256 of the full (already redacted) note, so an unchanged offending
    value hashes identically forever and any changed value hashes anew. The
    redacted note never carries credentials, so the fingerprint is log-safe.
    """
    trap_id = note.split(":", 1)[0].strip() or "config"
    fingerprint = hashlib.sha256(note.encode("utf-8")).hexdigest()
    return trap_id, fingerprint


def _read_traps(path: Optional[str]) -> Set[Tuple[str, str]]:
    """Return the persisted seen set at *path*, or an empty set when unreadable. Never raises."""
    if path is None:
        return set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return set()
    if isinstance(data, dict):
        entries = data.get("seen")
    else:
        entries = data
    if not isinstance(entries, list):
        return set()
    seen: Set[Tuple[str, str]] = set()
    for entry in entries:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], str)
        ):
            seen.add((entry[0], entry[1]))
    return seen


def _write_traps(path: Optional[str], seen: Set[Tuple[str, str]]) -> None:
    """Persist *seen* beside the record, atomically and without ever raising."""
    if path is None:
        return
    directory = os.path.dirname(path) or "."
    temporary: Optional[str] = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".config-traps-",
            suffix=".tmp",
            delete=False,
        )
        temporary = handle.name
        with handle:
            json.dump(
                {"schema": 1, "seen": sorted([list(pair) for pair in seen])},
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError:
        logger.debug("hermes-skill-router config: could not write the trap set", exc_info=True)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _pending_trap_notes(
    notes: Sequence[str], suggestions_path: Any
) -> Tuple[str, ...]:
    """Return the notes not yet in the persisted set, without persisting anything. Never raises.

    Unchanged values match a stored fingerprint and stay silent; a changed
    value hashes anew and warns once. No clock is involved: the file's value
    is the only suppression. No trap means no file: a healthy ``load`` never
    creates the set. Persistence is deferred to :func:`acknowledge_trap_notes`
    so a load that writes no display record (disabled, dashboard or pre-load
    read, any non-routing path) never consumes the fingerprint.
    """
    if not notes:
        return ()
    try:
        path = traps_path_for(suggestions_path)
        if path is None:
            # Nowhere to persist (display disabled): warn on every load. The
            # caller never writes a record on this path, so this costs nothing.
            return tuple(dict.fromkeys(notes))
        seen_before = _read_traps(path)
        return tuple(
            dict.fromkeys(note for note in notes if _trap_key(note) not in seen_before)
        )
    except Exception:  # load never raises; a persistence failure keeps the notes
        logger.debug("hermes-skill-router config: the trap set failed", exc_info=True)
        return tuple(notes)


def acknowledge_trap_notes(notes: Sequence[str], suggestions_path: Any) -> None:
    """Persist the fingerprints of *notes* that a display record actually carried. Never raises.

    Called by the display writer only after a record write succeeds and the
    written note contains the trap, so the fingerprint is consumed only once
    the warning has reached a record. No note means no file.
    """
    if not notes:
        return
    try:
        path = traps_path_for(suggestions_path)
        if path is None:
            return
        with _state_lock(path, create_dir=True):
            seen_before = _read_traps(path)
            union = seen_before | {_trap_key(note) for note in notes}
            if union != seen_before:
                _write_traps(path, union)
    except Exception:  # a persistence failure must never cost the record write
        logger.debug("hermes-skill-router config: the trap set failed", exc_info=True)


def _switch_note(key: str, raw_value: Any) -> str:
    """Return the switch warning line for *key* off with raw value *raw_value*.

    The raw value is shown redacted, so a string spelling of ``false`` and a
    boolean ``False`` read differently and hash to different fingerprints.
    """
    return f"{key}: {_shown(raw_value)} is off; {_SWITCH_WARNING_EFFECTS[key]}"


def _is_switch_note(note: str) -> bool:
    """Return whether *note* is a switch warning line (not a malformed value note)."""
    return note.startswith(("enabled: ", "showSuggestions: ")) and " is off; " in note


#: Serializes every read-modify-write of the persisted warning state in this
#: process: the switch load/fix transaction and the ack transactions read the
#: whole snapshot and write it back, so two of them interleaved between the
#: read and the write would silently restore the older snapshot (a fix lost
#: under an ack re-suppresses the re-broken warning). One lock for both files
#: (switch and trap sets) keeps the two writers of each file mutually
#: exclusive; the directory file lock below extends the same exclusion across
#: Hermes processes sharing the cache.
_SWITCH_STATE_LOCK = threading.Lock()
_SWITCH_LOCK_LOCAL = threading.local()


@contextmanager
def _state_lock(path: Optional[str], create_dir: bool = False) -> Iterator[bool]:
    """Hold the process-local lock and, when possible, the cross-process lock.

    The cross-process lock is an advisory ``fcntl.flock`` on the state
    file's directory, on the directory fd so the lock leaves no lock
    file behind (a healthy load must create no file). Advisory means
    only processes that take this lock are mutually excluded; nothing
    else enforces it. The flock is blocking with no timeout: a
    contender waits until the holder releases it. This directory has
    no other flock consumer today: the association store's migration
    lock is a file lock on ``<store>.migration.lock``, a different
    path, so the two locks never contend. When the directory cannot
    be opened or locked (read-only directory, non-POSIX platform,
    or a directory that cannot be created, see below), the
    transaction proceeds under the thread lock alone: state
    persistence is fail-open by design, and a lost lock must never
    cost the request. That fail-open is the documented condition:
    below it, only the process-local thread lock serializes, and
    concurrent processes may interleave. Re-entrant calls on the
    same thread re-use the locks already held instead of
    re-acquiring them.

    First use: when *create_dir* is set (write-producing callers:
    a load/fix that persists, an ack, a trap-note union), the
    directory is created (``mkdir``, ``exist_ok``) before it is
    opened for the lock and before the caller reads, so two
    simultaneous first-use processes both lock the same directory
    and serialize. A directory that cannot be created (permission,
    a file in the way) stays explicitly fail-open under the thread
    lock alone, never silently unprotected beyond that documented
    condition. Callers that may write nothing (a healthy no-write
    load) pass ``create_dir=False`` and create nothing: no file and
    no directory.

    Yields whether the cross-process flock is actually held
    (``True``) or the transaction runs thread-lock-only
    (``False``). A write-producing caller must not write a snapshot
    read under ``False`` when the state was discovered after the
    open attempt: see :func:`_pending_switch_warnings`.
    """
    depth = getattr(_SWITCH_LOCK_LOCAL, "depth", 0)
    if depth:
        _SWITCH_LOCK_LOCAL.depth = depth + 1
        try:
            yield getattr(_SWITCH_LOCK_LOCAL, "flock_held", False)
        finally:
            _SWITCH_LOCK_LOCAL.depth = depth
        return
    _SWITCH_STATE_LOCK.acquire()
    _SWITCH_LOCK_LOCAL.depth = 1
    _SWITCH_LOCK_LOCAL.flock_held = False
    fd = None
    try:
        if path is not None and fcntl is not None:
            directory = os.path.dirname(path) or "."
            if create_dir:
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError:
                    # Documented fail-open: the open/flock below will
                    # also fail, and the transaction runs under the
                    # thread lock alone.
                    pass
            try:
                fd = os.open(directory, os.O_RDONLY)
                fcntl.flock(fd, fcntl.LOCK_EX)
                _SWITCH_LOCK_LOCAL.flock_held = True
            except OSError:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                fd = None
        yield _SWITCH_LOCK_LOCAL.flock_held
    finally:
        _SWITCH_LOCK_LOCAL.depth = 0
        _SWITCH_LOCK_LOCAL.flock_held = False
        if fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
        _SWITCH_STATE_LOCK.release()


def _read_switch_sets(
    path: Optional[str],
) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Dict[str, int]]:
    """Return ``(seen, delivered, generations)`` from the switch file.

    Empty sets and zero generations when unreadable. A generation is the
    per-switch counter a pending note is captured at, bumped by every fix
    of that switch; a file written before generations existed reads as
    generation zero everywhere.
    """
    if path is None:
        return set(), set(), {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return set(), set(), {}
    if not isinstance(data, dict):
        return set(), set(), {}

    def _pairs(entries: Any) -> Set[Tuple[str, str]]:
        if not isinstance(entries, list):
            return set()
        pairs: Set[Tuple[str, str]] = set()
        for entry in entries:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], str)
            ):
                pairs.add((entry[0], entry[1]))
        return pairs

    raw_generations = data.get("generations")
    generations: Dict[str, int] = {}
    if isinstance(raw_generations, dict):
        for key, value in raw_generations.items():
            if (
                isinstance(key, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            ):
                generations[key] = value
    return _pairs(data.get("seen")), _pairs(data.get("delivered")), generations


def _write_switch_seen(
    path: Optional[str],
    seen: Set[Tuple[str, str]],
    delivered: Optional[Set[Tuple[str, str]]] = None,
    generations: Optional[Dict[str, int]] = None,
) -> None:
    """Persist *seen* and *delivered* directly, without a temporary file. Never raises.

    The switch set is written by ``load`` itself, so it must not depend on the
    display writer's atomic replace: a display write failure must not cost
    the switch suppression, and a direct write leaves no temporary file
    behind when the directory refuses the rename. A torn write reads as an
    empty set (a duplicate warning), never as a lost suppression.
    """
    if path is None:
        return
    try:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema": 1,
                    "seen": sorted([list(pair) for pair in seen]),
                    "delivered": sorted([list(pair) for pair in (delivered or set())]),
                    "generations": dict(generations or {}),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        logger.debug("hermes-skill-router config: could not write the switch set", exc_info=True)


def _emit_switch_warnings(lines: Sequence[str]) -> None:
    """Emit *lines* on the terminal and host-logger channels. Never raises."""
    for line in lines:
        try:
            print(line, file=sys.stderr)
        except Exception:  # a closed terminal must never cost the load
            logger.debug("hermes-skill-router config: the terminal warn failed", exc_info=True)
        try:
            logger.warning("hermes-skill-router config: %s", line)
        except Exception:  # logging never raises, but load must still hold
            logger.debug("hermes-skill-router config: the logger warn failed", exc_info=True)


def _pending_switch_warnings(
    settings: Mapping[str, Any],
    values: Mapping[str, Any],
    cache_path: Any,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict[str, int]]:
    """Return ``(pending_load, pending_injected, generations)``, emitting terminal/logger.

    Each switch keeps its own seen set inside the switch file: an unchanged
    value warns once on terminal/logger and then stays silent, a changed
    value hashes anew and warns once more, and a fixed switch drops only
    its own sets, so re-breaking the same value warns again while the other
    switch stays suppressed. ``pending_injected`` is the subset whose
    fingerprint is not yet in the delivered set; ``load`` never consumes
    the delivered set, only :func:`acknowledge_switch_deliveries` after an
    actual delivery does. The file lives beside the index cache (or in the
    default cache directory when ``cachePath`` is empty), so the display
    being disabled never turns emission into per-load spam.

    ``generations`` is the per-switch counter every pending note is
    captured at: a fix that clears its switch's sets bumps its switch's
    counter, so a late ack of a pre-fix note never matches the post-fix
    generation. A fix with nothing to clear writes nothing, so a healthy
    load stays write-free.

    D2 (off stays untouched): when the master switch is off the load returns
    at once with no pending pair, no emission and no persistence, for either
    switch. Prior state is left exactly as it was: an off load neither warns
    about ``showSuggestions`` nor clears its fingerprints.
    """
    if values.get("enabled") is False:
        return (), (), {}
    offending = {
        key: _switch_note(key, settings.get(key, False))
        for key in SWITCH_WARNING_SWITCHES
        if values.get(_SWITCH_FIELDS[key]) is False
    }
    fixed = [key for key in SWITCH_WARNING_SWITCHES if values.get(_SWITCH_FIELDS[key]) is True]
    try:
        path = switches_path_for(cache_path)
        if path is None:
            pending = tuple(offending[key] for key in SWITCH_WARNING_SWITCHES if key in offending)
            _emit_switch_warnings(pending)
            return pending, pending, {}
        # The whole read-modify-write (fix bump, seen/delivered update, write)
        # runs under the state lock, so a concurrent ack cannot restore the
        # pre-fix snapshot over this fix (or vice versa). A load with an
        # offending switch may persist the seen set, so it creates the
        # directory before locking it; a load with nothing offending
        # writes nothing (a fix that clears sets implies the directory
        # already exists), so it creates nothing.
        #
        # First-use race (round 8, F2): a no-offending load passes
        # create_dir=False, so on a first use its directory open can
        # fail ENOENT and it runs thread-lock-only. If state then
        # appears (a concurrent writer created the directory and the
        # file) and this load's fix has something to clear, writing
        # the snapshot read without the flock could discard a
        # competitor's update and recycle a generation. So when the
        # flock is not held and the transaction would write, and the
        # directory now exists, retry once under the flock and
        # re-read: the retry never creates the directory itself
        # (create_dir stays False; the directory already exists), so
        # a genuinely healthy no-write load still creates no file
        # and no directory. When the directory still cannot be
        # opened, the documented fail-open applies.
        for attempt in range(2):
            with _state_lock(path, create_dir=bool(offending)) as flock_held:
                result = _switch_transaction(offending, fixed, path)
                pending, injected, generations, changed, new_seen, new_delivered = result
                if changed and not flock_held and attempt == 0:
                    directory = os.path.dirname(path) or "."
                    if os.path.isdir(directory):
                        continue
                _emit_switch_warnings(pending)
                if changed:
                    _write_switch_seen(path, new_seen, new_delivered, generations)
                return pending, injected, generations
    except Exception:  # load never raises; a failure keeps the warnings visible
        logger.debug("hermes-skill-router config: the switch warnings failed", exc_info=True)
        pending = tuple(offending[key] for key in SWITCH_WARNING_SWITCHES if key in offending)
        return pending, pending, {}


def _switch_transaction(
    offending: Dict[str, str],
    fixed: Any,
    path: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict[str, int], bool, Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Compute the switch load/fix read-modify-write without emitting or writing.

    Return ``(pending, injected, generations, changed, new_seen,
    new_delivered)``: *changed* says whether persisting the new sets
    and generations is needed. The caller decides whether the lock
    held justifies writing, so a no-lock first-use attempt never
    writes a stale snapshot (see :func:`_pending_switch_warnings`).
    """
    seen_before, delivered_before, generations_before = _read_switch_sets(path)
    generations = dict(generations_before)
    new_seen = set(seen_before)
    new_delivered = set(delivered_before)
    for key in fixed:
        cleared = any(pair[0] == key for pair in new_seen) or any(
            pair[0] == key for pair in new_delivered
        )
        new_seen = {pair for pair in new_seen if pair[0] != key}
        new_delivered = {pair for pair in new_delivered if pair[0] != key}
        if cleared:
            # A cleared set means a pending note from before this fix
            # may still be in flight; bumping stales it. A fix with
            # nothing to clear writes nothing, so a healthy load
            # stays write-free.
            generations[key] = generations.get(key, 0) + 1
    pending_list: list[str] = []
    injected_list: list[str] = []
    for key in SWITCH_WARNING_SWITCHES:
        if key not in offending:
            continue
        line = offending[key]
        pair = _trap_key(line)
        if pair not in seen_before:
            pending_list.append(line)
        if pair not in delivered_before:
            injected_list.append(line)
        new_seen.add(pair)
    pending = tuple(pending_list)
    injected = tuple(injected_list)
    changed = (
        new_seen != seen_before
        or new_delivered != delivered_before
        or generations != generations_before
    )
    return pending, injected, generations, changed, new_seen, new_delivered


def _pending_switch_warnings_locked(
    offending: Dict[str, str],
    fixed: Any,
    path: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Dict[str, int]]:
    """Run the switch load/fix read-modify-write; caller holds the state lock."""
    pending, injected, generations, changed, new_seen, new_delivered = _switch_transaction(
        offending, fixed, path
    )
    _emit_switch_warnings(pending)
    if changed:
        _write_switch_seen(path, new_seen, new_delivered, generations)
    return pending, injected, generations


def switch_warning_messages(cfg: Config) -> Tuple[Dict[str, str], ...]:
    """Return the injected context messages for *cfg*'s undelivered switch warnings.

    One standalone ``system`` message per undelivered warning line, ready to
    be appended to the request. The delivered set is consulted at load time
    (``cfg.switch_message_notes``), never consumed by this call: only
    :func:`acknowledge_switch_deliveries` after an actual delivery on a
    supported channel consumes it. The message is the warning channel
    only: it carries no suggestion row and never uses the suggestions
    block header, so the rewrite's block recogniser never reads it as a
    suggestion.
    """
    try:
        injected = getattr(cfg, "switch_message_notes", None)
        if injected is not None:
            notes = injected
        else:
            notes = getattr(cfg, "notes", ()) or ()
    except Exception:  # a hostile notes field must never cost the call
        return ()
    messages: list[Dict[str, str]] = []
    for note in notes:
        if isinstance(note, str) and _is_switch_note(note):
            messages.append({"role": "system", "content": note})
    return tuple(messages)


def acknowledge_switch_deliveries(
    notes: Sequence[str],
    cache_path: Any,
    generations: Optional[Mapping[str, int]] = None,
) -> None:
    """Persist the injected-channel fingerprints of *notes* actually delivered. Never raises.

    Called by the middleware only after a valid outgoing request on a
    supported channel carries the messages. A load that never reached
    delivery (tool-first, unsupported mode, malformed request) never
    calls this, so its fingerprints stay pending for the next supported
    request. An acknowledgment is accepted only when its
    pending note is still in the seen set for that switch AND the note's
    captured generation (``generations``, from the load that produced
    the note) equals the switch's current generation, re-read here at
    ack time: seen-set membership alone is not identity, because in the
    fix/rebreak ABA the same pair is back in the seen set while the
    pre-fix note is stale. A rejected ack leaves the delivered set
    untouched, so a later re-break of the same value still delivers
    instead of being silently suppressed by the stale ack. An ack
    without its captured generations is rejected the same way (fail
    closed), and the delivered set stays a subset of seen.
    """
    if not notes:
        return
    try:
        path = switches_path_for(cache_path)
        if path is None:
            return
        # The whole read-modify-write runs under the state lock, so a
        # concurrent fix load cannot land between this read and write and
        # then be overwritten with the pre-fix snapshot. An ack may
        # persist the delivered set, so it creates the directory
        # before locking it.
        with _state_lock(path, create_dir=True):
            _acknowledge_switch_deliveries_locked(notes, path, generations)
    except Exception:  # a persistence failure must never cost the request
        logger.debug("hermes-skill-router config: the switch delivery failed", exc_info=True)


def _acknowledge_switch_deliveries_locked(
    notes: Sequence[str],
    path: str,
    generations: Optional[Mapping[str, int]],
) -> None:
    """Run the ack read-modify-write; caller holds the state lock."""
    seen_before, delivered_before, current = _read_switch_sets(path)
    if generations is None:
        return
    captured = dict(generations)
    accepted = set()
    for note in notes:
        if not isinstance(note, str) or not _is_switch_note(note):
            continue
        pair = _trap_key(note)
        if pair not in seen_before:
            continue
        if captured.get(pair[0], 0) != current.get(pair[0], 0):
            continue
        accepted.add(pair)
    union = delivered_before | accepted
    if union != delivered_before:
        _write_switch_seen(path, seen_before, union, current)


def report_notes(notes: Optional[Sequence[str]]) -> None:
    """Log each distinct config note once per process. A note names a setting, never request text."""
    for note in notes or ():
        if note in _LOGGED_NOTES:
            continue
        _LOGGED_NOTES.add(note)
        logger.warning("hermes-skill-router config: %s", note)


def load(raw: Optional[Mapping[str, Any]] = None) -> Config:
    """Return a :class:`Config` for *raw*, applying defaults, bounds and notes. Never raises."""
    notes: list[str] = []
    if raw is None:
        settings: Mapping[str, Any] = {}
    elif isinstance(raw, Mapping):
        settings = raw
    else:
        notes.append("config: expected a mapping of settings; using defaults")
        settings = {}
    values = {spec.field: spec.default for spec in _SPECS}
    for spec in _SPECS:
        if spec.key in settings:
            values[spec.field] = _coerce(spec, settings[spec.key], notes)
    for key in settings:
        if key not in DEFAULTS:
            name = _redact_url(key) if isinstance(key, str) else _shown(key)
            notes.append(f"{name}: unknown config key ignored")
    report_notes(notes)
    suggestions = values.get("suggestions_path", "")
    trap_notes = _pending_trap_notes(notes, suggestions)
    switch_notes, switch_message_notes, switch_generations = _pending_switch_warnings(
        settings, values, values.get("cache_path", "")
    )
    return Config(
        notes=tuple(notes) + switch_notes,
        trap_notes=trap_notes,
        switch_message_notes=switch_message_notes,
        switch_generations=switch_generations,
        **values,
    )


def read_settings(ctx: Any) -> dict[str, Any]:
    """Read every settings key through ``ctx.get_config``, defaulting when the surface lacks one.

    One reader for the request middleware and the search tool, so both see the same values from the
    same place; a context that raises for one key still yields that key's default.
    """
    raw: dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        try:
            raw[key] = ctx.get_config(key, default)
        except Exception:
            raw[key] = default
    return raw
