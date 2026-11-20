"""Decision-time parser of structured tool records into loaded skill names (C9).

The rewrite needs to know which skills the model actually opened in durable request bytes. It reads
the JSON arguments Hermes records with each tool call and the JSON (or plain-text) body of the result
matched by call id. It never scans assistant prose, never loops over the catalog for a call, and never
builds a second catalog: the known-name set and the name-to-SKILL.md path map come from the router's
cached routed index and are passed in.

Two fixed tools are recognised without configuration. ``skill_view`` carries a string ``name`` and
its matched result must say success and carry a known name. ``read_file`` carries a string ``path``
(older rows use ``file_path``); the path is evidence when it resolves to a known skill's ``SKILL.md``
or a file under that skill's ``references/`` directory, and its normal result carries neither a
success flag nor a name. A custom tool is recognised only when its exact name is in
``skill_loading_tools``; its arguments must be a JSON object with a string ``name`` that is a known
skill, or a string ``path`` that resolves as above, and path wins when both appear. Any other shape
is unreadable and never becomes a load.

Cost is one pass over the request's tool-call records, one result lookup keyed by call id, and one
path resolution index built once for the request. Work grows with the tool records, not catalog size.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

from .config import Config

SKILL_VIEW = "skill_view"
READ_FILE = "read_file"

_KIND_NAME = "name"
_KIND_PATH = "path"
_KIND_CUSTOM = "custom"

#: The configured-name set is rebuilt only when the exact configured names change.
_NAME_SET_CACHE: Dict[Tuple[str, ...], Dict[str, str]] = {}


def tool_name_set(cfg: Config) -> Dict[str, str]:
    """Return ``{tool name: argument kind}`` for the fixed tools plus configured custom tools.

    The two fixed names always win. Custom names are normalised once (blank entries and duplicates
    removed, by config coercion and here for a directly constructed Config) and the result is cached
    by exact configured names, so a later request with the same config reuses this dict.
    """
    configured: List[str] = []
    for value in getattr(cfg, "skill_loading_tools", ()) or ():
        if not isinstance(value, str):
            continue
        name = value.strip()
        if name and name not in configured:
            configured.append(name)
    key = tuple(configured)
    cached = _NAME_SET_CACHE.get(key)
    if cached is not None:
        return cached
    tools: Dict[str, str] = {SKILL_VIEW: _KIND_NAME, READ_FILE: _KIND_PATH}
    for name in configured:
        if name not in tools:
            tools[name] = _KIND_CUSTOM
    _NAME_SET_CACHE[key] = tools
    return tools


def request_records(messages: Any) -> Tuple[List[Any], Dict[str, Any]]:
    """Return one request's tool-call records and its results keyed by call id.

    Assistant messages carry the calls in ``tool_calls``; tool messages carry the matched result
    with ``tool_call_id``. A duplicate result id keeps the first message, so a call id has one
    authoritative result for the pass.
    """
    records: List[Any] = []
    results: Dict[str, Any] = {}
    if not isinstance(messages, (list, tuple)):
        return records, results
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id and call_id not in results:
                results[call_id] = message
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, (list, tuple)):
                records.extend(tool_calls)
    return records, results


def path_candidates(records: Sequence[Any], known: Mapping[str, Any]) -> List[str]:
    """Return known names that appear as a directory component of a path argument.

    The router resolves only these names through the memoised host lookup, so the host cost grows
    with the request's tool records instead of the catalog size. A skill's ``SKILL.md`` and its
    ``references/`` files live under a directory named after the skill, so a path component is the
    only evidence needed to pick the name to resolve.
    """
    candidates: List[str] = []
    seen: set = set()
    for record in records:
        batch = record.get("tool_calls") if isinstance(record, Mapping) else None
        calls = batch if isinstance(batch, (list, tuple)) else (record,)
        for item in calls:
            if not isinstance(item, Mapping):
                continue
            function = item.get("function")
            source = function if isinstance(function, Mapping) else item
            raw = source.get("arguments")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (ValueError, TypeError):
                    continue
            if not isinstance(raw, Mapping):
                continue
            path = raw.get("path") or raw.get("file_path")
            if not isinstance(path, str):
                continue
            for part in path.replace("\\", "/").split("/"):
                if part in known and part not in seen:
                    seen.add(part)
                    candidates.append(part)
    return candidates


def _result_fields(result: Any) -> Optional[Tuple[Any, Optional[str], bool]]:
    """Return ``(success, name, error)`` for a matched result, or ``None`` when unreadable.

    Hermes puts success inside the JSON result body, not on the tool message. A normal ``read_file``
    body is plain skill text with neither a success flag nor a name, which is deliberately not an
    error. An explicit JSON failure (``success`` false, ``isError`` true or a truthy ``error`` key)
    and a plain body that starts with an error prefix are errors.
    """
    if result is None:
        return None
    content = result.get("content") if isinstance(result, Mapping) else result
    if content is None:
        content = ""
    obj: Optional[Mapping] = None
    if isinstance(content, Mapping):
        obj = content
    elif isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, Mapping):
            obj = decoded
    if obj is not None:
        success = obj.get("success")
        name = obj.get("name")
        error = (
            success is False
            or obj.get("isError") is True
            or bool(obj.get("error"))
        )
        return success, name if isinstance(name, str) and name else None, error
    text = content.strip() if isinstance(content, str) else ""
    error = text.lower().startswith(("error:", "error ", "failed to", "not found"))
    return None, None, error


def _path_index(path_map: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build ``{SKILL.md path: name}`` and ``{skill root directory: name}`` once per request.

    A resolved path is either the exact ``SKILL.md`` of a known skill or a file under that skill's
    ``references/`` directory. Walking a path's parents against the root map is O(depth) dictionary
    lookups, never a loop over the catalog.
    """
    exact: Dict[str, str] = {}
    roots: Dict[str, str] = {}
    for name, path in path_map.items():
        if not isinstance(name, str) or not isinstance(path, str) or not path:
            continue
        normalised = os.path.normpath(path)
        exact[normalised] = name
        roots[os.path.dirname(normalised)] = name
    return exact, roots


def _resolve_path(path: str, exact: Mapping[str, str], roots: Mapping[str, str]) -> Optional[str]:
    """Return the known skill a resolvable path belongs to, or ``None``."""
    if not isinstance(path, str) or not path:
        return None
    normalised = os.path.normpath(path)
    name = exact.get(normalised)
    if name is not None:
        return name
    parent = os.path.dirname(normalised)
    while parent and parent != os.path.dirname(parent):
        name = roots.get(parent)
        if name is not None:
            relative = os.path.relpath(normalised, parent)
            if relative == "SKILL.md" or relative.split(os.sep)[0] == "references":
                return name
            return None
        parent = os.path.dirname(parent)
    return None


def _call_fields(record: Any) -> Optional[Tuple[str, str, Any]]:
    """Return ``(call id, tool name, parsed arguments)`` for one tool-call record."""
    if not isinstance(record, Mapping):
        return None
    function = record.get("function")
    source = function if isinstance(function, Mapping) else record
    call_id = record.get("id") or record.get("call_id")
    tool = source.get("name")
    if not isinstance(call_id, str) or not call_id:
        return None
    if not isinstance(tool, str) or not tool:
        return None
    raw = source.get("arguments")
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except (ValueError, TypeError):
            return None
    elif isinstance(raw, Mapping):
        arguments = raw
    else:
        return None
    if not isinstance(arguments, Mapping):
        return None
    return call_id, tool, arguments


def loaded_skills(
    records: Sequence[Any],
    results_by_call: Mapping[str, Any],
    known: Mapping[str, Any],
    path_map: Mapping[str, str],
    tools: Mapping[str, str],
) -> List[str]:
    """Return each skill loaded by a structured record in this request, once, in first-seen order.

    Unreadable, malformed and unmatched records yield no load and never raise. ``known`` is the
    cached routed catalog name set plus eager skills; ``path_map`` is its saved SKILL.md path map;
    ``tools`` is :func:`tool_name_set` for this config.
    """
    exact, roots = _path_index(path_map)
    loaded: List[str] = []
    seen: set = set()
    seen_calls: set = set()

    def add(name: Optional[str]) -> None:
        if name is not None and name in known and name not in seen:
            seen.add(name)
            loaded.append(name)

    for record in records:
        # A caller may hand over the payload's assistant messages instead of a flat call list.
        batch = record.get("tool_calls") if isinstance(record, Mapping) else None
        calls = batch if isinstance(batch, (list, tuple)) else (record,)
        for item in calls:
            fields = _call_fields(item)
            if fields is None:
                continue
            call_id, tool, arguments = fields
            kind = tools.get(tool)
            if kind is None or call_id in seen_calls:
                continue
            seen_calls.add(call_id)
            fields_result = _result_fields(results_by_call.get(call_id))

            if kind == _KIND_NAME:
                name = arguments.get("name")
                result_name = fields_result[1] if fields_result else None
                if (
                    isinstance(name, str)
                    and name in known
                    and fields_result is not None
                    and fields_result[0] is True
                    and not fields_result[2]
                    and result_name in known
                ):
                    add(result_name)
                continue

            if kind == _KIND_PATH:
                path = arguments.get("path")
                if not isinstance(path, str) or not path:
                    path = arguments.get("file_path")
                if fields_result is not None and not fields_result[2]:
                    add(_resolve_path(path, exact, roots))
                continue

            # Custom tool: a string path decides when present; a string name is the other shape.
            path = arguments.get("path")
            if isinstance(path, str) and path:
                if fields_result is not None and not fields_result[2]:
                    add(_resolve_path(path, exact, roots))
                continue
            name = arguments.get("name")
            if (
                isinstance(name, str)
                and name in known
                and fields_result is not None
                and fields_result[0] is not False
                and not fields_result[2]
                and fields_result[1] in known
            ):
                add(fields_result[1])

    return loaded
