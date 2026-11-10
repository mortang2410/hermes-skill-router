"""Configuration for the hermes-skill-router plugin (module M2, contract C2).

``load`` is the only place that decides what a configuration value means: every field falls back
to its default, with a note naming the key, when the value is malformed or out of bounds, and
``load`` never raises. Endpoint roots must be safe: a plain ``http`` URL is accepted only for a
loopback host, any other host must use ``https``, and a URL carrying credentials, a query or a
fragment is rejected. A note about a rejected value shows only the value's origin, never its
userinfo, path, query or fragment.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

#: Exact text the DSH skill router puts before ``<name> - <description>`` in a rerank document.
DOCUMENT_PREFIX = (
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
    "actually need to carry out the current task correctly. Skill: "
)

#: Hosts a plain ``http`` endpoint URL may name.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Rerank request/response shapes this plugin implements.
RERANK_PROVIDERS = ("cohere",)

_KIND_LABELS = {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "text": "string",
    "path": "path",
    "file_path": "path",
    "name_list": "list of skill names",
    "provider": "provider name",
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
    _Spec("shortlistSize", "shortlist_size", "int", 50, 1, 383),
    _Spec("cutoff", "cutoff", "float", 0.01, 0.0, 1.0),
    _Spec("maxSuggestions", "max_suggestions", "int", 8, 0, 50),
    _Spec("maxDisplayNames", "max_display_names", "int", 12, 1, 50),
    _Spec("maxDescriptionChars", "max_description_chars", "int", 400, 3, 4000),
    _Spec("cachePath", "cache_path", "file_path", "~/.cache/hermes-skill-router/index.json"),
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
    shortlist_size: int
    cutoff: float
    max_suggestions: int
    max_description_chars: int
    cache_path: str
    health_check_ms: int
    show_suggestions: bool
    suggestions_path: str
    max_display_names: int
    document_prefix: str
    eager_skills: Tuple[str, ...]
    deferred_skills: Tuple[str, ...]
    notes: Tuple[str, ...]


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
    return Config(notes=tuple(notes), **values)
