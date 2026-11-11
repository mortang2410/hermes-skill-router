"""The injected skills block (module M8, contract C8).

The block replaces the catalog section the plugin removes, so it carries the skills the router
selected for this turn, best first, and the kept rows that must stay visible: the pinned eager skills
and the rows a non-empty deferred list leaves listed. The block is also its own recogniser: the rewrite
scans a request's messages for a block by the shared first line and recovers the names it suggested
with :func:`suggested_names`, so a later turn in the same compaction window does not repeat them and a
compaction that removes the block makes them suggestable again.

Every suggestion row carries three fields: its name, its description and the absolute read path the
host lookup resolved, so the model can open the file the resolver ``skill_view`` uses:

    Suggestions:
    - name: description
      path: /absolute/path/to/SKILL.md

A suggestion whose path is missing or not absolute is not emitted: it is a non-success for that row,
and the selector only hands over rows it could resolve. Pinned rows form their own section, present
only in the opening block of a compaction window, and carry the same three fields from the kept half of
the catalog. A pinned row whose path does not resolve is omitted with one warning per process, so the
block never carries a row the model cannot open. Kept rows that are neither pinned nor suggested keep
their earlier shape under an ``Also listed:`` marker: ``- name: description``, or ``- name`` for a
demoted ``[names only]`` row, with no path, because the search tool needs no path for a row the prompt
already listed.

The description is looked up in the index by name, so a selected row can never smuggle text into the
prompt and no description is carried twice. A name that appears in two places is rendered once, as a
suggestion first, then as a pinned row, then as an also-listed row.

The header is fixed text, not a summary: it says these are the skills that may help, names the loader
tools ``skill_view`` and ``skills_list``, and carries the instructions the removed ``## Skills`` prose
gave about loading a relevant skill rather than guessing. There are two fixed headers, and the
``one_shot`` argument picks one (the rewrite passes ``Partition.one_shot``). Both share their first
line, because the rewrite recognises its own prior block by that line. A stripped request carries no
heading by the name ``## Skills``, and the block never reintroduces it.

Skill-maintenance guidance ("fix it with ``skill_manage``", "offer to save the result as a skill") is
deliberately absent from both headers. The one-shot header goes further and states the rule the removed
prose carried: do not create or edit skills.

This module applies no cap. The selector is the only place the cap is applied, so a block handed
twelve rows renders twelve rows even when ``maxSuggestions`` is 0. The function is pure. Rendering
returns ``""`` only when it has no row to show at all, which keeps the strip and the injection atomic:
a request with no block keeps its catalog byte for byte.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from .catalog import CatalogRows
from .config import Config
from .library import Index
from .select import Suggestion

logger = logging.getLogger(__name__)

#: The tools that load a skill. Both stay registered on the host, and both headers name them.
LOADER_TOOLS = ("skill_view", "skills_list")

#: Fixed header of a normal-turn block. The rewrite recognises its own block by the shared first line.
#: That first line is deliberately not the removed ``## Skills`` heading: the stripped request must
#: carry no heading by that name, and a reader of a request dump must not mistake the block for the
#: catalog it replaced.
HEADER = (
    "Skills routed for this request\n"
    "These are the skills that may help; the rest of the catalog is not listed. Scan them before "
    "replying. If one matches or is even partially relevant, load it with skill_view(name) and "
    "follow its instructions: skills carry API endpoints, tool-specific commands, conventions and "
    "pitfalls that outperform a general approach, so load one rather than guess, and load one that "
    "defines how a task should be done here even when you already know the basics. Call skills_list "
    "when none of these fits, to see every skill available. Only proceed without loading a skill if "
    "genuinely none are relevant."
)

#: Fixed header for a block that replaced the single-query catalog prose. It keeps the restriction
#: that prose carried and drops the normal invitation to load an "even partially relevant" skill:
#: after this text was stripped, inviting a load for work already known would contradict it.
ONE_SHOT_HEADER = (
    "Skills routed for this request\n"
    "These are the skills that may help; the rest of the catalog is not listed. Load one with "
    "skill_view(name) only when it carries domain knowledge you lack for this task: an API, a "
    "tool's commands, or a project's conventions. Do not load general process skills (testing, "
    "debugging, review methodology) for work you already know how to do, and do not create or "
    "edit skills: this is a one-shot run with no later session to reuse them. Call skills_list "
    "when none of these fits, to see every skill available."
)

#: Section marker that opens the selected suggestions.
SUGGESTIONS = "Suggestions:"

#: Section marker that opens the pinned rows, present only in a window's opening block.
PINNED = "Pinned skills:"

#: Marker that separates the rows that stay listed from the suggestions and pinned rows.
ALSO_LISTED = "Also listed:"

#: Prefix of the read-path line under a suggestion or pinned row.
PATH_PREFIX = "  path: "

#: The one-line marker a stripped request carries when routing failed and the catalog went out
#: instead. The first line is neither block header, so the recogniser never reads it as a
#: suggestion, and carries no skill name, so it cannot pose as exposure of any one skill.
#: The rewrite recognises it in a later request and strips the catalog again, which keeps the
#: whole catalog out of the window after the first dump.
DROPPED_HEADER = "Skill catalog dropped after a routing failure"

DROPPED_MESSAGE = (
    "Skill catalog dropped after a routing failure\n"
    "The full skill catalog was already shown earlier in this compaction window, so it is not "
    "shown again. Call skills_list to see every skill available, or load one with "
    "skill_view(name). Do not guess a skill name: use the list or the loader."
)

#: Names whose pinned row was omitted for a missing path, so one warning is logged per process.
_PINNED_WARNED: set[str] = set()


def render(
    rows: Sequence[Suggestion],
    kept: CatalogRows,
    index: Optional[Index],
    cfg: Config,
    one_shot: bool = False,
    opening: bool = False,
    paths: Optional[Dict[str, str]] = None,
) -> str:
    """Return the block for *rows*, *kept* and *cfg*, or ``""`` when it would carry no row.

    *rows* are the selected suggestions from :func:`hermes_skill_router.select.select`, already
    capped, best first. *kept* is the kept half of the catalog partition. *one_shot* is
    ``Partition.one_shot``: it chooses which of the two fixed headers opens the block. *opening*
    says the request's messages carry no earlier block, so the pinned rows are listed; a later turn
    in the same compaction window omits them. *paths* maps a name to the absolute read path the host
    lookup resolved, for both suggestions and pinned rows. *index* supplies the descriptions of
    suggested rows and may be ``None``, in which case they are listed by name only.

    A suggestion row with no absolute path is not rendered. A pinned row with no absolute path is
    omitted with one warning per process. The cap is not applied here: the selector applied it.
    """
    resolved = paths or {}
    descriptions = _descriptions(index)
    limit = cfg.max_description_chars
    eager = set(cfg.eager_skills or ())
    seen: set[str] = set()
    suggested: List[str] = []
    pinned: List[str] = []
    listed: List[str] = []
    for name, _score in rows:
        if name in seen:
            continue
        path = _absolute_path(resolved.get(name))
        if path is None:
            # The selector resolved only names with a path, so this is a caller that skipped the
            # rule; the row is a non-success and never reaches the block.
            continue
        seen.add(name)
        suggested.append(_row(name, descriptions.get(name, ""), limit, path))
    for name, description in zip(kept.names, kept.descriptions):
        if name in seen:
            # A suggestion or an earlier kept row already names this skill.
            continue
        if name in eager:
            if not opening:
                continue
            path = _absolute_path(resolved.get(name))
            if path is None:
                _warn_pinned(name)
                continue
            seen.add(name)
            pinned.append(_row(name, description, limit, path))
        else:
            seen.add(name)
            listed.append(_plain_row(name, description, limit))
    if not suggested and not pinned and not listed:
        return ""
    parts = [ONE_SHOT_HEADER if one_shot else HEADER]
    if suggested:
        parts.append(SUGGESTIONS + "\n" + "\n".join(suggested))
    if pinned:
        parts.append(PINNED + "\n" + "\n".join(pinned))
    if listed:
        parts.append(ALSO_LISTED + "\n" + "\n".join(listed))
    return "\n\n".join(parts)


def suggested_names(block_text: str) -> List[str]:
    """Return the names *block_text* lists as suggestions, in order, and never a pinned name.

    The renderer owns the answer in both directions: this is the reader the rewrite uses to recover
    the names an earlier block already suggested, so the representation of a row is written and read
    here. Only the suggestions section counts; the pinned and also-listed rows name skills the block
    did not suggest, and a name in those sections is never returned.
    """
    names: List[str] = []
    section = ""
    for line in block_text.splitlines():
        if line in (SUGGESTIONS, PINNED, ALSO_LISTED):
            section = line
            continue
        if section != SUGGESTIONS or not line.startswith("- "):
            continue
        name = line[2:].split(": ", 1)[0].strip()
        if name:
            names.append(name)
    return names


def _descriptions(index: Optional[Index]) -> Dict[str, str]:
    """Map each indexed name to its description, first occurrence winning."""
    if index is None:
        return {}
    descriptions: Dict[str, str] = {}
    for name, description in zip(index.names, index.descriptions):
        descriptions.setdefault(name, description)
    return descriptions


def _absolute_path(path: Optional[str]) -> Optional[str]:
    """Return *path* when it is an absolute string path, else ``None``: no pathless row is emitted."""
    if isinstance(path, str) and os.path.isabs(path):
        return path
    return None


def _row(name: str, description: str, limit: int, path: str) -> str:
    """Return one suggestion or pinned row: its name, description and read path."""
    return _plain_row(name, description, limit) + "\n" + PATH_PREFIX + path


def _plain_row(name: str, description: str, limit: int) -> str:
    """Return one name line: ``- name: description``, or ``- name`` without a description."""
    text = _one_line(description)[:limit].rstrip()
    if not text:
        return f"- {name}"
    return f"- {name}: {text}"


def _one_line(description: str) -> str:
    """Collapse every whitespace run, newlines included, to one space and trim the ends."""
    return " ".join(description.split())


def _warn_pinned(name: str) -> None:
    """Log one omitted pinned row once per process, never including request text."""
    if name in _PINNED_WARNED:
        return
    _PINNED_WARNED.add(name)
    logger.warning(
        "skill router: pinned skill %r has no resolvable read path; omitting it from the block",
        name,
    )
