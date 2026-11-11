"""The searchable catalog record and the one selection pass the turn and the tool share.

The middleware parses the skill catalog out of one Chat Completions request and partitions it. This
module keeps the partition each session showed, so the model's search tool can rank exactly the rows
the current prompt listed, and it runs the same embed, shortlist, rerank, cutoff and suggestion-cap
pass the middleware runs for a turn. Nothing here scans the skill library: a session with no parsed
partition has no rows, and no search falls back to the filesystem.

The partition record is written at the start of a turn, which is also when the previous record is
dropped: a request that carries no catalog, a non-Chat-Completions request, and a request the
middleware returns before parsing all leave the session with no searchable catalog. A parsed
partition with no routed rows is different: the partition exists and the prompt listed nothing to
search, which the tool reports as the healthy empty answer.

The pass in :func:`rank` is the middleware's own: it takes the query vector inside the routing
budget, shortlists the routed rows by cosine, reranks those documents and applies the configured
cutoff and cap. It also hands every candidate's score and the stage timings back with the selection,
so the turn can write its decision line from the values the selector compared. The rewrite renders
its block from the same result, so a turn and a search for the same query and catalog can never
disagree. A failure names its arm (``embed-failed``, ``rerank-failed``, ``budget-expired``) instead of
a partial list; the handler maps that to the degraded envelope and the middleware to an unchanged
request.

The path of each returned row comes from the host's own skill lookup, the resolver ``skill_view``
uses, and only for the rows the search selected: that lookup walks the skill directories, so calling
it for every routed row would make one search scan the whole library for rows it never returns. A
name the lookup refuses, or cannot resolve uniquely, keeps its row and omits the path.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, replace
from time import monotonic
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .catalog import CatalogRows, Partition
from .config import Config
from .embed import embed_texts as default_embed
from .library import Index, load_or_build, shortlist
from .rerank import rerank as default_rerank
from .select import select

logger = logging.getLogger(__name__)

#: The index could not be built, or it carries no row to rank.
NO_INDEX = "no-index"
#: The query could not be embedded inside the routing budget.
EMBED_FAILED = "embed-failed"
#: The shortlist could not be reranked inside the routing budget.
RERANK_FAILED = "rerank-failed"
#: ``routingTimeoutMs`` ran out before the pass could finish.
BUDGET_EXPIRED = "budget-expired"

#: One embedder: ``M5.embed_texts``, one vector per text in order or ``None`` on failure.
Embedder = Callable[[Config, Sequence[str]], Optional[Sequence[Sequence[float]]]]

#: One reranker: ``M6.rerank``, one probability per document in order or ``None`` on failure.
Reranker = Callable[[Config, str, Sequence[str]], Optional[Sequence[float]]]

#: Resolves one selected name to its ``SKILL.md`` path, or ``None`` when the lookup refuses.
PathResolver = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class SearchRow:
    """One searchable skill: the name and description the prompt showed, and its path when known."""

    name: str
    description: str = ""
    path: Optional[str] = None


@dataclass(frozen=True)
class SearchOutcome:
    """One search answer: the selected rows, or the failure arm and its detail."""

    rows: Tuple[SearchRow, ...] = ()
    reason: str = ""
    detail: str = ""


@dataclass(frozen=True)
class Ranked:
    """The selection pass' answer: name and probability pairs best first, or a failure arm.

    ``candidates`` carries every shortlisted name with the rerank probability the selector compared,
    in shortlist order, so a caller can report a candidate the cutoff rejected. The three timings
    are milliseconds: the embed stage, the rerank stage and the whole pass.
    """

    rows: Tuple[Tuple[str, float], ...] = ()
    reason: str = ""
    detail: str = ""
    candidates: Tuple[Tuple[str, float], ...] = ()
    embed_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


#: The newest parsed partition per session. The middleware writes it; a search only reads it.
_PARTITIONS: Dict[str, Partition] = {}
_PARTITIONS_LOCK = threading.Lock()


def remember_partition(session_id: str, partition: Partition) -> None:
    """Replace the session's searchable record with *partition*: the rows this request showed."""
    with _PARTITIONS_LOCK:
        _PARTITIONS[session_id] = partition


def forget_session(session_id: str) -> None:
    """Drop the session's record, so a later search answers ``no-catalog`` until a catalog parses."""
    with _PARTITIONS_LOCK:
        _PARTITIONS.pop(session_id, None)


def catalog_rows(cfg: Config, session_id: str) -> Optional[Sequence[SearchRow]]:
    """Return the routed rows of the session's parsed partition, or ``None`` when it has none.

    One row per routed skill, in catalog order, with the name and description the prompt showed.
    Kept rows, names-only rows and anything the prompt filtered out are absent. An empty sequence is
    a parsed partition whose prompt listed nothing to search, which the caller reports as the
    healthy empty answer rather than as ``no-catalog``.

    Reads the record the middleware wrote; never touches the filesystem and writes nothing. ``cfg``
    is the loaded config, accepted so this call has the same shape as :func:`select_rows`; no setting
    changes which rows a parsed prompt showed.
    """
    with _PARTITIONS_LOCK:
        partition = _PARTITIONS.get(session_id)
    if partition is None:
        return None
    return tuple(
        SearchRow(name, description)
        for name, description in zip(partition.routed.names, partition.routed.descriptions)
    )


def select_rows(
    cfg: Config,
    rows: Sequence[SearchRow],
    query: str,
    *,
    index: Optional[Index] = None,
    embed: Optional[Embedder] = None,
    rerank: Optional[Reranker] = None,
    deadline: Optional[float] = None,
    resolve: Optional[PathResolver] = None,
) -> SearchOutcome:
    """Rank *rows* for *query* and return the selected rows, or the failure arm.

    *rows* are the routed rows of the session's parsed catalog record; *query* is non-empty. *index*
    is the caller's index when it already holds one, otherwise the shared loader builds or loads it
    from these rows. *deadline* is the routing pass' absolute ``monotonic`` deadline; when it is
    ``None`` the budget starts after the index is in hand, so a cold build is never cut, exactly as
    a turn's budget starts. *resolve* names one selected row's ``SKILL.md`` path; without it the rows
    carry no path.

    The ranking is :func:`rank`, the middleware's own pass, so the same query and catalog give the
    same names and order a turn would inject. Never raises: an index, embed or rerank failure
    returns that arm, and an empty selection is a healthy outcome with no rows.
    """
    material = [(row.name, row.description) for row in rows]
    if not material:
        return SearchOutcome()
    if index is None:
        try:
            index = load_or_build(
                cfg,
                CatalogRows(
                    tuple(name for name, _description in material),
                    tuple(description for _name, description in material),
                ),
                embed or default_embed,
            )
        except Exception:  # a plugin bug must never abort a search
            logger.debug("skill search: the index build raised", exc_info=True)
            index = None
        if index is None:
            return SearchOutcome(reason=NO_INDEX, detail="the skill index could not be built")
    if not index.names:
        return SearchOutcome(reason=NO_INDEX, detail="the skill index carries no row")
    ranked = rank(cfg, index, query, embed=embed, rerank=rerank, deadline=deadline)
    if ranked.reason:
        return SearchOutcome(reason=ranked.reason, detail=ranked.detail)
    descriptions = dict(zip(index.names, index.descriptions))
    return SearchOutcome(
        rows=tuple(
            SearchRow(name, descriptions[name], resolve(name) if resolve is not None else None)
            for name, _probability in ranked.rows
        )
    )


def rank(
    cfg: Config,
    index: Index,
    query: str,
    *,
    embed: Optional[Embedder] = None,
    rerank: Optional[Reranker] = None,
    deadline: Optional[float] = None,
) -> Ranked:
    """Run one routing pass over *index* for *query*: embed, shortlist, rerank, cutoff and cap.

    The budget starts here when *deadline* is ``None``; the caller passes an index already in hand,
    so the cold build is never inside it. That one deadline covers both stages, and each per-request
    timeout is capped by the milliseconds left, so the pass cannot outlive the budget: a stage that
    returns after the deadline, or finds it already spent, ends as ``budget-expired``. An empty
    selection is a healthy outcome with no rows; only a failed endpoint or an unbuildable shortlist
    names another arm.

    A completed pass also hands back every shortlisted candidate with the score the selector compared
    and the three stage timings, so the caller can report why a name was not selected. An arm carries
    no candidate, so a failed pass can never report a partial shortlist.
    """
    started = monotonic()
    deadline = _deadline(cfg, deadline)
    query_vector, failure = _query_vector(cfg, query, embed or default_embed, deadline)
    embed_ms = _elapsed_ms(started)
    if failure is not None:
        return failure
    if _budget_left(deadline) is None:
        return _failed(BUDGET_EXPIRED, _budget_message(cfg))
    shortlisted = shortlist(index, query_vector, cfg.shortlist_size)
    if not shortlisted:
        return Ranked(embed_ms=embed_ms, total_ms=_elapsed_ms(started))
    documents = [
        cfg.document_prefix + name + " - " + description for name, description in shortlisted
    ]
    rerank_started = monotonic()
    scores, failure = _rerank_scores(cfg, query, documents, rerank or default_rerank, deadline)
    rerank_ms = _elapsed_ms(rerank_started)
    if failure is not None:
        return failure
    if _budget_left(deadline) is None:
        return _failed(BUDGET_EXPIRED, _budget_message(cfg))
    names = [name for name, _description in shortlisted]
    chosen = select(scores, names, cfg)
    return Ranked(
        rows=tuple(chosen),
        candidates=tuple((name, float(score)) for score, name in zip(scores, names)),
        embed_ms=embed_ms,
        rerank_ms=rerank_ms,
        total_ms=_elapsed_ms(started),
    )


def _budget_message(cfg: Config) -> str:
    """The one budget-exhausted detail both passes report, so the log phrase stays stable."""
    return f"routing budget {cfg.routing_timeout_ms} ms exhausted"


def _query_vector(
    cfg: Config,
    query: str,
    embedder: Embedder,
    deadline: Optional[float],
) -> Tuple[Optional[List[float]], Optional[Ranked]]:
    """Embed the query inside the budget, or return the failure arm to report."""
    deadline = _deadline(cfg, deadline)
    budget = _budget_message(cfg)
    left = _budget_left(deadline)
    if left is None:
        return None, _failed(BUDGET_EXPIRED, budget)
    timed_out = False
    try:
        vectors = embedder(_capped(cfg, "embed_timeout_ms", left), [query])
    except TimeoutError:
        timed_out = True
        vectors = None
    except Exception:  # an endpoint client never raises, but the pass must not depend on it
        logger.debug("skill search: the embeddings call raised", exc_info=True)
        vectors = None
    if not vectors:
        if _budget_left(deadline) is None:
            return None, _failed(BUDGET_EXPIRED, budget)
        if timed_out:
            return None, _failed(EMBED_FAILED, "embed timeout")
        return None, _failed(EMBED_FAILED, "embed unreachable")
    try:
        return [float(value) for value in vectors[0]], None
    except (TypeError, ValueError):
        return None, _failed(EMBED_FAILED, "embed unreachable")


def _rerank_scores(
    cfg: Config,
    query: str,
    documents: Sequence[str],
    reranker: Reranker,
    deadline: Optional[float],
) -> Tuple[Optional[List[float]], Optional[Ranked]]:
    """Score the shortlist inside the budget, or return the failure arm to report."""
    deadline = _deadline(cfg, deadline)
    budget = _budget_message(cfg)
    left = _budget_left(deadline)
    if left is None:
        return None, _failed(BUDGET_EXPIRED, budget)
    try:
        scores = reranker(_capped(cfg, "rerank_timeout_ms", left), query, documents)
    except Exception:  # an endpoint client never raises, but the pass must not depend on it
        logger.debug("skill search: the rerank call raised", exc_info=True)
        scores = None
    if scores is None:
        if _budget_left(deadline) is None:
            return None, _failed(BUDGET_EXPIRED, budget)
        return None, _failed(RERANK_FAILED, "rerank unreachable")
    return scores, None


def _failed(reason: str, detail: str) -> Ranked:
    return Ranked(reason=reason, detail=detail)


def _elapsed_ms(started: float) -> float:
    """Return the milliseconds between *started* and now, from the clock the budget uses."""
    return (monotonic() - started) * 1000.0


def _deadline(cfg: Config, deadline: Optional[float]) -> float:
    """Return *deadline*, or a fresh one starting now: the budget begins after the index is in hand."""
    if deadline is None:
        return monotonic() + cfg.routing_timeout_ms / 1000.0
    return deadline


def _budget_left(deadline: float) -> Optional[int]:
    """Return the whole milliseconds left before *deadline*, or ``None`` once it has passed.

    The value is rounded UP, so a per-request timeout set from it can never fire before the
    deadline: the socket then times out with the budget already spent, which is what lets a
    failed call tell a budget expiry from an endpoint that failed on its own.
    """
    left = int(math.ceil((deadline - monotonic()) * 1000.0))
    return left if left > 0 else None


def _capped(cfg: Config, field: str, left: int) -> Config:
    """Return *cfg* with *field*, a per-request timeout, capped by the milliseconds left."""
    return replace(cfg, **{field: min(getattr(cfg, field), left)})
