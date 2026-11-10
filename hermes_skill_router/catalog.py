"""Recognise the Hermes skill catalog, partition it, and remove it (module M3, contract C3).

The skills section is one volatile part of the system prompt, rendered by
``agent/prompt_builder.py`` (``_render_skills_index``) as either the multi-line guidance that
starts with the ``## Skills`` heading or the one-shot variant that carries the same heading, and
is followed by the ``<available_skills>`` block and, optionally, the names-only note. The whole
section is removed: heading, prose before and after the block, the tags and their rows.

The block carries two row shapes: ``- name: description`` rows and ``category [names only]: a, b``
lines that a coding posture demotes. The parenthetical notes the renderer appends to the part (the
names-only note and the unloadable-copy note) are part of the section and leave with it; a name that
only appears in a copy note is never listed, because ``skill_view`` cannot load that copy. Rows are
partitioned once per request. A row whose name is
eager is kept; otherwise, when the deferred list is empty every row is routed, and a non-empty
deferred list routes the rows it names and keeps the rest. A name on a ``[names only]`` line is
always kept with an empty description and never routed, even when the deferred list names it,
because the renderer never emitted its description. Kept rows come back in the ``Partition`` so
the injected block can list them again, and ``skill_view`` still opens any name the catalog
carried.

The strip also records which prose it removed: ``one_shot`` is true when the section was the
single-query guidance (``HERMES_SINGLE_QUERY_SESSION``), whose text forbids loading general process
skills for work already known and forbids creating or editing skills. The injected block has to
carry that restriction instead of the normal invitation, so the variant travels on the partition
(:attr:`Partition.one_shot`) and the rewrite hands it to the renderer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence, Tuple

logger = logging.getLogger(__name__)

SKILLS_HEADING = "## Skills"
BLOCK_OPEN = "<available_skills>"
BLOCK_CLOSE = "</available_skills>"
TRAILING_GUIDANCE = (
    "Only proceed without loading a skill if genuinely none are relevant to the task."
)
#: The one-shot variant's guidance carries the same heading and no trailing paragraph.
ONESHOT_LEAD = "## Skills\nScan the skills below"

_ROW_PATTERN = re.compile(r"^[ \t]+-[ \t]+(?P<row>.+?)[ \t]*$")
_NAMES_ONLY_PATTERN = re.compile(
    r"^[ \t]+\S.*\[names only\][ \t]*:[ \t]*(?P<names>.+?)[ \t]*$"
)


#: Warnings already emitted, so one misconfigured name logs once per process (D8), not per request.
_LOGGED: set[str] = set()


class _Entry(NamedTuple):
    """One catalog entry in render order."""

    name: str
    description: str
    names_only: bool = False


@dataclass(frozen=True)
class CatalogRows:
    """Parallel name and description tuples in catalog order."""

    names: Tuple[str, ...] = ()
    descriptions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Partition:
    """Kept rows, routed rows, configured names the catalog does not carry, and the prose variant."""

    kept: CatalogRows
    routed: CatalogRows
    unknown_names: Tuple[str, ...] = ()
    #: True when the stripped prose was the single-query guidance, whose restriction the injected
    #: block must keep; false for the normal prose. The rewrite passes it to the renderer.
    one_shot: bool = False


def strip_catalog(
    payload: Mapping[str, Any],
    *,
    eager: Sequence[str] = (),
    deferred: Sequence[str] = (),
) -> Tuple[Mapping[str, Any], Partition | None]:
    """Remove the whole skills section from the first system message of *payload*.

    Returns ``(payload, Partition)`` when the first system message carried the section, and
    ``(payload, None)`` unchanged when the payload has no ``messages`` list, no first system
    message with string content, or no skills section. Idempotent: a second call finds nothing.
    """
    if not isinstance(payload, Mapping):
        return payload, None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, None
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return payload, None
        section = _find_section(content)
        if section is None:
            return payload, None
        start, end, open_end, close_at, one_shot = section
        entries = _parse_block(content[open_end:close_at])
        partition = _partition(entries, eager, deferred, one_shot=one_shot)
        rewritten = list(messages)
        rewritten[index] = {**message, "content": _cut(content, start, end)}
        return {**payload, "messages": rewritten}, partition
    return payload, None


def _find_section(content: str) -> Tuple[int, int, int, int, bool] | None:
    """Return ``(start, end, block_open_end, block_close_at, one_shot)`` of the section, or ``None``."""
    open_at = content.find(BLOCK_OPEN)
    if open_at == -1:
        return None
    close_at = content.find(BLOCK_CLOSE, open_at + len(BLOCK_OPEN))
    if close_at == -1:
        return None
    heading_at = content.rfind(SKILLS_HEADING, 0, open_at)
    if heading_at == -1:
        return None
    if heading_at > 0 and content[heading_at - 1] != "\n":
        return None
    one_shot = content[heading_at:open_at].startswith(ONESHOT_LEAD)
    end = close_at + len(BLOCK_CLOSE)
    if content.startswith("\n\n" + TRAILING_GUIDANCE, end):
        end += len(TRAILING_GUIDANCE) + 2
    elif not one_shot and content.startswith("\n\n", end):
        # An unknown paragraph follows the block: the section end is ambiguous, change nothing.
        return None
    end = _consume_notes(content, end)
    return heading_at, end, open_at + len(BLOCK_OPEN), close_at, one_shot


def _consume_notes(content: str, end: int) -> int:
    """Extend a section end past the parenthetical notes the renderer appends to the part.

    The renderer appends one single-line note per condition directly after the block or its
    trailing guidance: the names-only note, and a copy note for skills that share a name and a
    relative path with another skill in the same tier. Those copies cannot be loaded, so they are
    never listed; their note leaves with the section. A following prompt part is joined with a
    blank line, so only a single newline before ``(`` starts a note.
    """
    while content.startswith("\n(", end):
        next_newline = content.find("\n", end + 1)
        line_end = len(content) if next_newline == -1 else next_newline
        if not content[end + 1 : line_end].rstrip().endswith(")"):
            break
        end = line_end
    return end


def _parse_block(block: str) -> Tuple[_Entry, ...]:
    """Parse both catalog row shapes from the block body, in render order."""
    entries: list[_Entry] = []
    for line in block.splitlines():
        demoted = _NAMES_ONLY_PATTERN.match(line)
        if demoted is not None:
            for name in demoted.group("names").split(","):
                name = name.strip()
                if name:
                    entries.append(_Entry(name, "", names_only=True))
            continue
        row = _ROW_PATTERN.match(line)
        if row is None:
            continue
        name, separator, description = row.group("row").partition(": ")
        name = name.strip()
        if name:
            entries.append(_Entry(name, description.strip() if separator else ""))
    return tuple(entries)


def _partition(
    entries: Sequence[_Entry],
    eager: Sequence[str],
    deferred: Sequence[str],
    *,
    one_shot: bool = False,
) -> Partition:
    """Split catalog entries into kept and routed rows and report unknown configured names.

    *one_shot* is recorded on the partition so the rewrite can hand the variant to the renderer,
    whose header has to keep the restriction the removed prose carried.
    """
    eager_names = _configured(eager)
    deferred_names = _configured(deferred)
    eager_set = set(eager_names)
    deferred_set = set(deferred_names)
    kept_names: list[str] = []
    kept_descriptions: list[str] = []
    routed_names: list[str] = []
    routed_descriptions: list[str] = []
    both_noted: set[str] = set()
    for entry in entries:
        if entry.names_only:
            # The renderer omitted this name's description, so it is listed but never routed.
            kept_names.append(entry.name)
            kept_descriptions.append("")
        elif entry.name in eager_set:
            if entry.name in deferred_set and entry.name not in both_noted:
                both_noted.add(entry.name)
                _warn_once(
                    f"both:{entry.name}",
                    "skill %r is listed in both eagerSkills and deferredSkills; treating it as eager",
                    entry.name,
                )
            kept_names.append(entry.name)
            kept_descriptions.append(entry.description)
        elif deferred_names and entry.name not in deferred_set:
            kept_names.append(entry.name)
            kept_descriptions.append(entry.description)
        else:
            routed_names.append(entry.name)
            routed_descriptions.append(entry.description)
    catalog_names = {entry.name for entry in entries}
    unknown: list[str] = []
    for name in (*eager_names, *deferred_names):
        if name not in catalog_names and name not in unknown:
            unknown.append(name)
            _warn_once(
                f"unknown:{name}",
                "configured skill %r is not in the catalog; ignoring it",
                name,
            )
    return Partition(
        kept=CatalogRows(tuple(kept_names), tuple(kept_descriptions)),
        routed=CatalogRows(tuple(routed_names), tuple(routed_descriptions)),
        unknown_names=tuple(unknown),
        one_shot=one_shot,
    )


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log one distinct configured-name warning per process, never including request text."""
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.warning(message, *args)


def _configured(names: Sequence[str]) -> Tuple[str, ...]:
    """Strip configured names, dropping blanks and repeats while keeping their order."""
    cleaned: list[str] = []
    for name in names or ():
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return tuple(cleaned)


def _cut(content: str, start: int, end: int) -> str:
    """Remove ``content[start:end]`` plus one paragraph separator so the join survives."""
    if content.startswith("\n\n", end):
        return content[:start] + content[end + 2 :]
    if start >= 2 and content[start - 2 : start] == "\n\n":
        return content[: start - 2] + content[end:]
    if start >= 1 and content[start - 1] == "\n":
        return content[: start - 1] + content[end:]
    return content[:start] + content[end:]
