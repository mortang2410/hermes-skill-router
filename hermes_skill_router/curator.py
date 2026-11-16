"""Curator branch with plugin watermark and two-surface delta narrowing (module M4, contract C4).

The curator fork carries the full catalog in its system prompt and its own
upstream candidate list in the user message. ``default_curation`` (the
default) returns the payload unmodified: no strip, no route, no record, no
``suggestions.json`` write. ``delta_curation`` narrows two sets over one delta
predicate (activity newer than the plugin watermark, pinned excluded):

- the system catalog block narrows to the delta set intersected with the
  catalog partition (bundled and disabled rows may appear there; visibility
  only widens, and the prompt's hard rule refuses touching them);
- the upstream candidate list in the user message narrows to delta
  intersected with upstream eligibility (agent-created, not bundled, not
  disabled) in the upstream row format.

The eligibility intersection reads the already-rendered upstream rows of the
user message verbatim and filters them by delta name: bundled and disabled
rows never appear there because upstream already excluded them
(agent/curator.py ``_render_candidate_list``), so the plugin never rebuilds
the eligibility rule. Any failure or empty set fails open byte-identical.

The delta predicate never uses the current run's just-saved ``last_run_at``
(agent/curator.py saves state before the LLM pass) as its boundary: the
plugin keeps its own watermark file. Lifecycle: first sight fails open AND
seeds the watermark with the current ``last_run_at``; a non-empty,
non-dry delta answer advances the watermark to the current ``last_run_at``
and caches the answer so later middleware calls in the same run repeat it;
dry runs fail open and never write; prune-only runs (which bump
``last_run_at`` without a fork) leave the watermark stale, which only
widens the next candidate set. The plugin never writes ``.curator_state``
or ``.usage.json``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .catalog import CatalogRows, Partition, strip_catalog
from .writeback import ORIGIN_CURATOR, derive_origin
from .writeback import _newest_user_text

logger = logging.getLogger("hermes_skill_router.curator")

WATERMARK_FILENAME = "curator-watermark.json"

#: Environment overrides (tests point them at temporary files).
WATERMARK_PATH_ENV = "HERMES_SKILL_ROUTER_CURATOR_WATERMARK"
HERMES_HOME_ENV = "HERMES_HOME"

#: The dry-run banner (agent/curator.py CURATOR_DRY_RUN_BANNER): a dry fork
#: prompt is prefixed with it and never mutates the library, so the plugin
#: fails open and never advances or seeds.
DRY_RUN_MARKER = "DRY-RUN — REPORT ONLY"

#: The upstream candidate-list header (agent/curator.py _render_candidate_list).
CANDIDATE_HEADER_PREFIX = "Agent-created skills ("
EMPTY_CANDIDATES_TEXT = "No agent-created skills to review."

#: One upstream candidate row: "- name  provenance=...  state=...  ...".
_CANDIDATE_ROW_PATTERN = re.compile(r"^- (?P<name>\S+)\s+provenance=")

CURATOR_BLOCK_HEADER = (
    "Skills changed since the last curation. This is the candidate set "
    "for this review; run skills_list for the full catalog."
)


def hermes_home() -> str:
    """Return the Hermes home directory from the environment or the host.

    ``HERMES_HOME`` wins so isolated runs (and tests) point the plugin at a
    scratch home; otherwise the live workspace resolver answers when
    importable, else the platform default.
    """
    override = os.environ.get(HERMES_HOME_ENV) or ""
    if override.strip():
        return os.path.expanduser(override.strip())
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser("~/.hermes")


def skills_dir(home: Optional[str] = None) -> str:
    return os.path.join(home or hermes_home(), "skills")


def watermark_path(cache_path: Any) -> Optional[str]:
    """Return the plugin watermark path for *cache_path* (beside the index cache)."""
    override = os.environ.get(WATERMARK_PATH_ENV) or ""
    if override.strip():
        return os.path.expanduser(override.strip())
    if not cache_path or not isinstance(cache_path, str) or not cache_path.strip():
        return None
    directory = os.path.dirname(os.path.abspath(os.path.expanduser(cache_path.strip())))
    return os.path.join(directory, WATERMARK_FILENAME)


def read_watermark(cache_path: Any) -> Optional[Dict[str, Any]]:
    """Return the watermark record, or ``None`` when absent or unreadable."""
    path = watermark_path(cache_path)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("watermark"), str):
        return None
    return data


def store_watermark(cache_path: Any, record: Mapping[str, Any]) -> bool:
    """Persist *record* beside the cache. Returns ``False`` on any failure."""
    path = watermark_path(cache_path)
    if path is None:
        return False
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
        os.replace(tmp, path)
        return True
    except OSError:
        logger.debug("hermes-skill-router curator: watermark write failed", exc_info=True)
        return False


def parse_time(value: Any) -> Optional[datetime]:
    """Return an aware datetime for an ISO timestamp, else ``None``."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def last_run_at(skills_dir_path: str) -> Optional[str]:
    """Return the ``last_run_at`` string from ``.curator_state``, else ``None``."""
    try:
        with open(os.path.join(skills_dir_path, ".curator_state"), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    value = data.get("last_run_at") if isinstance(data, dict) else None
    return value if parse_time(value) is not None else None


def load_usage(skills_dir_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Return the ``.usage.json`` map, or ``None`` when missing or corrupt."""
    try:
        with open(os.path.join(skills_dir_path, ".usage.json"), encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _latest_activity(record: Mapping[str, Any]) -> Optional[datetime]:
    """Return the newest activity timestamp of *record* (created_at excluded)."""
    moments = [
        parse_time(record.get(key))
        for key in ("last_used_at", "last_viewed_at", "last_patched_at")
    ]
    moments = [moment for moment in moments if moment is not None]
    return max(moments) if moments else None


def delta_names(usage: Mapping[str, Mapping[str, Any]], watermark: str) -> List[str]:
    """Return names with any usage event newer than *watermark*, pinned excluded.

    A never-active skill enters only when ``created_at`` is newer than the
    watermark; a record with no timestamps at all never enters.
    """
    boundary = parse_time(watermark)
    if boundary is None:
        return []
    names: List[str] = []
    for name in sorted(usage):
        record = usage[name]
        if record.get("pinned"):
            continue
        latest = _latest_activity(record)
        if latest is not None and latest > boundary:
            names.append(name)
            continue
        created = parse_time(record.get("created_at"))
        if created is not None and created > boundary:
            names.append(name)
    return names


def is_dry_run(messages: Any) -> bool:
    """Return True when the newest user message carries the dry-run banner."""
    text = _newest_user_text(messages)
    return text is not None and DRY_RUN_MARKER in text


def filter_candidate_rows(user_text: str, allowed: set) -> Optional[str]:
    """Return *user_text* with its candidate list narrowed to *allowed* names.

    The upstream rows are kept verbatim, in upstream order; the header count
    is re-rendered. A user text without the candidate header returns ``None``
    (the surface cannot be located, so the payload fails open).
    """
    lines = user_text.splitlines()
    header_at = None
    for index, line in enumerate(lines):
        if line.startswith(CANDIDATE_HEADER_PREFIX):
            header_at = index
            break
    if header_at is None:
        return None
    rows: List[str] = []
    for line in lines[header_at + 1 :]:
        match = _CANDIDATE_ROW_PATTERN.match(line)
        if match is None:
            continue
        if match.group("name") in allowed:
            rows.append(line)
    narrowed = lines[:header_at]
    if rows:
        narrowed.append(f"Agent-created skills ({len(rows)}):")
        narrowed.extend(rows)
    else:
        narrowed.append(EMPTY_CANDIDATES_TEXT)
    return "\n".join(narrowed)


def candidate_names(
    messages: Any, cache_path: Any, home: Optional[str] = None
) -> Optional[List[str]]:
    """Return the delta names for this curator run, applying the watermark lifecycle.

    ``None`` means fail open (payload unchanged): first sight (which seeds the
    watermark), a dry run, any unreadable state, an unparseable boundary, or
    an empty delta. A non-empty, non-dry answer advances the watermark and
    caches it, so later middleware calls in the same run repeat the answer.
    """
    if is_dry_run(messages):
        return None
    directory = skills_dir(home)
    current = last_run_at(directory)
    if current is None:
        return None
    watermark = read_watermark(cache_path)
    if watermark is None:
        store_watermark(
            cache_path, {"watermark": current, "answered_run": current, "candidates": None}
        )
        return None
    cached = watermark.get("candidates")
    if watermark.get("answered_run") == current and isinstance(cached, list):
        cached_names = [name for name in cached if isinstance(name, str)]
        return cached_names or None
    usage = load_usage(directory)
    if usage is None:
        return None
    names = delta_names(usage, str(watermark["watermark"]))
    if not names:
        return None
    store_watermark(
        cache_path, {"watermark": current, "answered_run": current, "candidates": names}
    )
    return names


def render_candidates(rows: Sequence[Tuple[str, str]]) -> str:
    """Return the curator candidate block: name and description rows, catalog shape."""
    lines = [CURATOR_BLOCK_HEADER, "<available_skills>", "  general:"]
    lines.extend(f"    - {name}: {description}" for name, description in rows)
    lines.append("</available_skills>")
    return "\n".join(lines)


def curator_rewrite(payload: Mapping[str, Any], cfg: Any, messages: Any) -> Optional[Mapping[str, Any]]:
    """Run the curator branch for a delta-mode request, or return ``None``.

    ``None`` means the payload ships unmodified (default mode, first sight,
    dry run, or any failure): the fork keeps the full catalog. Otherwise the
    catalog is stripped, the candidate block is appended, and the upstream
    candidate list in the newest user message is narrowed to the delta
    intersection in the upstream row format. No embed, no rerank, no record.
    """
    if getattr(cfg, "curation_mode", "default_curation") != "delta_curation":
        return None
    try:
        names = candidate_names(messages, cfg.cache_path)
        if not names:
            return None
        user_text = _newest_user_text(messages)
        if user_text is None:
            return None
        narrowed_user = filter_candidate_rows(user_text, set(names))
        if narrowed_user is None:
            return None
        stripped, partition = strip_catalog(
            payload, eager=cfg.eager_skills, deferred=cfg.deferred_skills
        )
        if partition is None:
            return None
        descriptions = dict(zip(partition.kept.names, partition.kept.descriptions))
        descriptions.update(zip(partition.routed.names, partition.routed.descriptions))
        wanted = set(names)
        system_names = [
            name
            for name in (*partition.kept.names, *partition.routed.names)
            if name in wanted
        ]
        if not system_names:
            return None
        rows = [(name, descriptions.get(name, "")) for name in system_names]
        block = render_candidates(rows)
        final_messages: List[Any] = []
        for message in stripped.get("messages") or ():
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and CANDIDATE_HEADER_PREFIX in message["content"]
            ):
                final_messages.append({**message, "content": narrowed_user})
            else:
                final_messages.append(message)
        final_messages.append({"role": "system", "content": block})
        return {"request": {**dict(payload), "messages": final_messages}}
    except Exception:
        logger.debug("hermes-skill-router curator: branch failed open", exc_info=True)
        return None
