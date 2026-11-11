"""The cutoff and cap that turn rerank probabilities into suggestions (module M7, contract C7).

The scores arrive from :func:`hermes_skill_router.rerank.rerank` already as probabilities: the shared
decider answers with its softmax share over the shortlist, and the reference selector compares that
value directly. This module does the same and nothing more. A candidate is selected when its score is
at or above ``cfg.cutoff`` (inclusive), the survivors are sorted by descending score, and the list is
truncated to ``cfg.max_suggestions``. Ties keep the order they arrived in, which is shortlist order,
so a stable descending sort is the whole ordering policy.

There is deliberately no normalisation here. A softmax over the incoming scores would turn a single
confident candidate into 0.99 and lift forty-nine near-zero candidates over a 0.01 cutoff, which is
the failure the routing study records as "two softmaxes": a page of suggestions where one skill was
meant. ``cutoff`` is a probability threshold, not a share of a recalculated distribution.

Nothing here reads a clock, a file or the network, so selection replays against recorded scores.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .config import Config

#: One selected suggestion: the catalog name and its rerank probability.
Suggestion = Tuple[str, float]


def select(scores: Sequence[float], names: Sequence[str], cfg: Config) -> List[Suggestion]:
    """Return the suggestions at or above ``cfg.cutoff``, best first, capped at ``cfg.max_suggestions``.

    *scores* and *names* are parallel, in shortlist order, and every score is a rerank probability.
    An empty input, a cap of zero, or no score at or above the cutoff returns ``[]``. The result
    never names a skill outside *names* and never holds more than ``cfg.max_suggestions`` rows. A
    length mismatch pairs as far as both sequences go, so a name without a score is never selected.
    """
    if not scores or not names:
        return []
    limit = cfg.max_suggestions
    if limit <= 0:
        return []
    cutoff = cfg.cutoff
    eligible: List[Suggestion] = []
    for score, name in zip(scores, names):
        probability = float(score)
        if probability >= cutoff:
            eligible.append((name, probability))
    # ``list.sort`` is stable, so equal probabilities keep shortlist order, as the reference
    # selector does.
    eligible.sort(key=lambda suggestion: suggestion[1], reverse=True)
    return eligible[:limit]
