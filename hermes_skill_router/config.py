"""Configuration for the hermes-skill-router plugin (module M2, contract C2).

``load`` is the only place that decides what a configuration value means: every field falls back
to its default, with a note naming the key, when the value is malformed or out of bounds, and
``load`` never raises. Endpoint roots must be safe: a plain ``http`` URL is accepted only for a
loopback host, any other host must use ``https``, and a URL carrying credentials, a query or a
fragment is rejected. A note about a rejected value shows only the value's origin, never its
userinfo, path, query or fragment.

Config traps (every ``notes`` entry) also reach the display record, but only when newly seen:
``load`` persists a set of ``(trap_id, fingerprint)`` pairs beside the suggestions file, so an
unchanged offending value stays silent forever and a changed value warns once. The persisted set
never uses a clock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlsplit

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
)

#: Manifest key -> default value.
DEFAULTS: dict[str, Any] = {spec.key: spec.default for spec in _SPECS}

#: Manifest key -> inclusive numeric bound, for the keys that carry one.
BOUNDS: dict[str, Tuple[float, float]] = {
    spec.key: (spec.lower, spec.upper) for spec in _SPECS if spec.lower is not None
}


#: File name of the persisted config-trap set, written beside the suggestions file.
TRAPS_FILENAME = "config-traps.json"

#: Environment override for the traps file; tests point it at a temporary file.
TRAPS_ENV_VAR = "HERMES_SKILL_ROUTER_TRAPS_PATH"


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
    trap_notes: Tuple[str, ...] = ()


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


def _new_trap_notes(
    notes: Sequence[str], suggestions_path: Any, show_suggestions: bool = True
) -> Tuple[str, ...]:
    """Return the notes not yet in the persisted set, then persist the union. Never raises.

    Unchanged values match a stored fingerprint and stay silent; a changed
    value hashes anew and warns once. No clock is involved: the file's value
    is the only suppression. No trap means no file: a healthy ``load`` never
    creates the set.
    """
    if not notes:
        return ()
    try:
        if not show_suggestions:
            # The display is disabled: no record is writable, so the warning
            # never reaches one. Do not persist the fingerprint, or the first
            # enabled load would stay silent for a value nobody saw warned.
            return tuple(dict.fromkeys(notes))
        path = traps_path_for(suggestions_path)
        if path is None:
            # Nowhere to persist (display disabled): warn on every load. The
            # caller never writes a record on this path, so this costs nothing.
            return tuple(dict.fromkeys(notes))
        seen_before = _read_traps(path)
        current: Set[Tuple[str, str]] = {_trap_key(note) for note in notes}
        newly = tuple(
            dict.fromkeys(note for note in notes if _trap_key(note) not in seen_before)
        )
        union = seen_before | current
        if union != seen_before:
            _write_traps(path, union)
        return newly
    except Exception:  # load never raises; a persistence failure keeps the notes
        logger.debug("hermes-skill-router config: the trap set failed", exc_info=True)
        return tuple(notes)


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
    trap_notes = _new_trap_notes(
        notes, suggestions, bool(values.get("show_suggestions", True))
    )
    return Config(notes=tuple(notes), trap_notes=trap_notes, **values)


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
