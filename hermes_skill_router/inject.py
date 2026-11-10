"""The injected skills block (module M8, contract C8).

The block replaces the catalog section the plugin removes, so it carries two things: the skills the
router selected for this turn, best first, and the kept rows that must stay visible (eager skills
and rows a non-empty deferred list leaves listed). A selected row carries only its name and score;
the description is looked up in the index by name, so the row itself can never smuggle text into the
prompt and no description is carried twice. A name that appears in both lists is rendered once, as
a suggestion.

Each row is one line: ``- name: description`` with the description collapsed to a single line and
clipped to ``cfg.max_description_chars``, or ``- name`` when there is no description at all (a
demoted ``[names only]`` row, or a suggested name the index no longer carries). Kept rows follow
under an ``Also listed:`` marker, so stripping the catalog never drops a name the model can still
open with ``skill_view``.

The header is fixed text, not a summary: it says these are the skills that may help, tells the model
to open one with ``skill_view`` and to call ``skills_list`` when none fits, and carries the
instructions the removed ``## Skills`` prose gave about loading a relevant skill rather than
guessing. Its first line is a marker line rather than the removed ``## Skills`` heading, so a
stripped request carries no heading by that name and the block is recognisable in a request dump.
The first line is shared by both headers, so the rewrite recognises its own block by that prefix
whichever variant it injected.

There are two fixed headers, and the ``one_shot`` argument picks one (the rewrite passes
``Partition.one_shot``). The normal prompt invites the model to
load a skill that is "even partially relevant"; the single-query prompt instead forbids loading
general process skills for work already known and forbids creating or editing skills, because a
one-shot run has no later session to reuse them. Injecting the normal invitation after stripping the
one-shot prose would tell the model to do exactly what the removed text forbade, so
:data:`ONE_SHOT_HEADER` keeps that restriction and drops the invitation. Each header is one fixed
string, exported so the rewrite and the tests can pin it. Nothing else is added: no per-skill hints,
examples or file paths (D3), names and descriptions only, and ``max_suggestions`` caps suggested
rows alone, so a cap of zero still lists every kept row.

Skill-maintenance guidance ("fix it with ``skill_manage``", "offer to save the result as a skill")
is deliberately absent from both headers. The one-shot header goes further and states the rule the
removed prose carried: do not create or edit skills.

The function is pure. Rendering returns ``""`` only when it has no row to show at all, which keeps
the strip and the injection atomic: a request with no block keeps its catalog byte for byte.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .catalog import CatalogRows
from .config import Config
from .library import Index
from .select import Suggestion

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

#: Marker that separates the suggestions from the rows that stay listed.
ALSO_LISTED = "Also listed:"


def render(
    rows: Sequence[Suggestion],
    kept: CatalogRows,
    index: Optional[Index],
    cfg: Config,
    one_shot: bool = False,
) -> str:
    """Return the block for *rows*, *kept* and *cfg*, or ``""`` when it would carry no row.

    *rows* are the selected suggestions from :func:`hermes_skill_router.select.select`, already
    capped; the cap is applied again here so ``cfg.max_suggestions`` bounds suggested rows for any
    caller. *kept* is the kept half of the catalog partition. *one_shot* is
    ``Partition.one_shot``: it chooses which of the two fixed headers opens the block, so the block
    matches the prose the strip removed. *index* supplies the descriptions of suggested rows and
    may be ``None``, in which case they are listed by name only.
    """
    descriptions = _descriptions(index)
    limit = cfg.max_description_chars
    cap = max(0, cfg.max_suggestions)
    seen: set[str] = set()
    suggested: List[str] = []
    listed: List[str] = []
    for name, _score in rows:
        if len(suggested) >= cap:
            # The cap counts rendered suggested rows, so a duplicate name never spends a slot.
            break
        if name in seen:
            continue
        seen.add(name)
        suggested.append(_line(name, descriptions.get(name, ""), limit))
    for name, description in zip(kept.names, kept.descriptions):
        if name in seen:
            # A suggested name is already on a line above; the block never names a skill twice.
            continue
        seen.add(name)
        listed.append(_line(name, description, limit))
    if not suggested and not listed:
        return ""
    parts = [ONE_SHOT_HEADER if one_shot else HEADER]
    if suggested:
        parts.append("\n".join(suggested))
    if listed:
        parts.append(ALSO_LISTED + "\n" + "\n".join(listed))
    return "\n\n".join(parts)


def _descriptions(index: Optional[Index]) -> Dict[str, str]:
    """Map each indexed name to its description, first occurrence winning."""
    if index is None:
        return {}
    descriptions: Dict[str, str] = {}
    for name, description in zip(index.names, index.descriptions):
        descriptions.setdefault(name, description)
    return descriptions


def _line(name: str, description: str, limit: int) -> str:
    """Return one rendered row: ``- name: description``, or ``- name`` without a description."""
    text = _one_line(description)[:limit].rstrip()
    if not text:
        return f"- {name}"
    return f"- {name}: {text}"


def _one_line(description: str) -> str:
    """Collapse every whitespace run, newlines included, to one space and trim the ends."""
    return " ".join(description.split())
