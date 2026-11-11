"""The ``skill_router_search`` tool (module M5, contracts C5 and C6).

The tool is the model's way to search the skill catalog the prompt showed when the injected shortlist
had nothing for the task. It ranks the routed half of the session's parsed partition through
:func:`hermes_skill_router.search.select_rows`, the same pass a turn runs, so its names and order
are the ones a routed turn would have injected for the same query. The tool never scans the skill
library: a session with no parsed partition answers ``no-catalog``, and a session whose prompt
carried nothing searchable answers a healthy empty result (D4, D10).

Each returned row carries the name and the description the prompt showed, and the ``SKILL.md`` path
the host's own lookup resolves for it, which is the resolver ``skill_view`` uses. A name that lookup
refuses, or cannot resolve uniquely, keeps its row and omits the path. No score, probability, count
or body ever reaches the result (C6).

A healthy answer, an empty selection included, writes one display record with origin ``search`` and
the selected names when ``showSuggestions`` is on; with it off the writer removes the record instead,
because the setting governs both origins, and the rows are still returned. A failure or a switched
off plugin writes no record and answers the degraded envelope with its reason, so the model can tell
"nothing fits" from "the router could not answer" (D6). The query is never written anywhere, the
handler never raises, and no file beyond the existing record and cache is touched.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping, Optional

from .config import load, read_settings
from .display import ROUTED, SEARCH, record
from .embed import embed_texts
from .rerank import rerank
from .search import SearchOutcome, catalog_rows, select_rows

logger = logging.getLogger(__name__)

#: The registered name, the manifest's ``provides_tools`` entry and the tests all use this.
TOOL_NAME = "skill_router_search"

#: The plugin's own toolset: it groups the tool with the plugin that serves it.
TOOLSET = "hermes-skill-router"

#: What the model reads before calling: when the prompt's shortlist already covers the task, skip.
DESCRIPTION = (
    "Search every skill the current prompt showed, ranked for one query. The prompt already lists "
    "the skills routed for this turn: do not call this tool for one of those, and do not call it "
    "when the listed shortlist covers the task. Call it to search the rest of the catalog the "
    "prompt did not list. Returns name, description and the SKILL.md path when the host can "
    "resolve it."
)

#: One required string parameter; the free text the model wants a skill for.
SCHEMA = {
    "name": TOOL_NAME,
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What the task needs, in the model's own words.",
            }
        },
        "required": ["query"],
    },
}

#: The HTTP clients, exported so a test can supply its own transport without touching the pass.
EMBED = embed_texts
RERANK = rerank


def register_tools(ctx: Any) -> None:
    """Register the search tool through *ctx*, with the fixed schema and a handler that never raises.

    Called from the package's ``register(ctx)``, the one registration site the normal standalone load
    runs. The handler reads its settings from the same context on every call, so a settings change
    applies without a reload; a registration failure is left to the loader, which reports it.
    """

    def handler(args: Mapping[str, Any], **kwargs: Any) -> str:
        return handle(args, settings=read_settings(ctx), **kwargs)

    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=SCHEMA,
        handler=handler,
        description=DESCRIPTION,
        emoji="\U0001f50e",
    )


def handle(
    args: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    session_id: str = "",
    **kwargs: Any,
) -> str:
    """Answer one search call as JSON. Never raises: every failure becomes the degraded envelope.

    *args* carries the required ``query``; *settings* are the raw plugin settings, loaded here so the
    tool and the middleware read one place. The dispatcher passes ``session_id`` and any other
    accepted keyword, and the session id is what selects the parsed catalog record to rank.
    """
    try:
        query = args.get("query") if isinstance(args, Mapping) else None
        if not isinstance(query, str) or not query.strip():
            return _degraded(
                "invalid-query", "the query must be a non-empty string of the task's own words"
            )
        cfg = load(settings)
        if not cfg.enabled:
            return _degraded("disabled", "the hermes-skill-router switch is off")
        rows = catalog_rows(cfg, str(session_id or ""))
        if rows is None:
            return _degraded(
                "no-catalog", "no skill catalog has been parsed for this session yet"
            )
        outcome = select_rows(
            cfg, rows, query, embed=EMBED, rerank=RERANK, resolve=skill_path
        )
        if outcome.reason:
            return _degraded(outcome.reason, outcome.detail)
        _write_record(cfg, outcome, str(session_id or ""))
        return _answer(outcome)
    except Exception:  # the tool contract: a handler never raises out to the client
        logger.exception("hermes-skill-router: the search tool failed")
        return _degraded("no-index", "the search failed inside the plugin")


def skill_path(name: str) -> Optional[str]:
    """Return the ``SKILL.md`` path the host's skill lookup resolves for *name*, or ``None``.

    Uses the resolver ``skill_view`` calls, so the path a search returns is the file the model can
    open. Any host import or lookup failure is ``None``: the row stays and omits its path rather than
    dropping a skill the search selected.
    """
    try:
        from tools.skills_tool import _locate_skill, _skill_search_dirs
    except Exception:
        return None
    try:
        project_dirs, all_dirs, _active = _skill_search_dirs()
        error, _skill_dir, skill_md = _locate_skill(name, None, project_dirs, all_dirs)
    except Exception:
        logger.debug("skill search: the host lookup for %r failed", name, exc_info=True)
        return None
    if error is not None or skill_md is None:
        return None
    return str(skill_md)


def _write_record(cfg: Any, outcome: SearchOutcome, session_id: str) -> None:
    """Write the search record, or let the writer remove it when the setting is off. Never raises.

    The writer is authoritative for both values: with ``showSuggestions`` on it writes origin
    ``search`` with the selected names, and with it off it removes any record instead, so the pane
    and the widget stay idle and the search never bypasses the setting. A writer failure is logged
    and never changes the answer the model already earned.
    """
    try:
        record(
            cfg,
            status=ROUTED,
            origin=SEARCH,
            names=[row.name for row in outcome.rows],
            session_id=session_id,
        )
    except Exception:
        logger.debug("skill search: the display record failed", exc_info=True)


def _answer(outcome: SearchOutcome) -> str:
    """Return the healthy envelope: one row per skill, with the path only when one was resolved."""
    skills = []
    for row in outcome.rows:
        entry = {"name": row.name, "description": row.description}
        if row.path is not None:
            entry["path"] = row.path
        skills.append(entry)
    return json.dumps({"skills": skills}, ensure_ascii=False)


def _degraded(reason: str, detail: str) -> str:
    """Return the degraded envelope: the failure arm, its detail and no rows."""
    return json.dumps(
        {"degraded": True, "reason": reason, "detail": detail, "skills": []},
        ensure_ascii=False,
    )
