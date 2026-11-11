"""The suppressed set, the cutoff and the cap that turn rerank probabilities into suggestions.

Module M7, contract C7. The scores arrive from :func:`hermes_skill_router.rerank.rerank` already as
probabilities: the shared decider answers with its softmax share over the shortlist, and the reference
selector compares that value directly. A candidate is selected when its name is not suppressed, the
caller resolved its read path, and its score is at or above ``cfg.cutoff`` (inclusive). The survivors
are sorted by descending score and truncated to ``cfg.max_suggestions``, where ``0`` means unlimited.
Ties keep the order they arrived in, which is shortlist order, so a stable descending sort is the whole
ordering policy.

The cap counts only admissible rows: a suppressed name and a name with no resolvable read path are
dropped before it, so neither consumes a slot and the next admissible candidate takes the place. Both
the cutoff and the cap are applied here and nowhere else. The ranking pass hands over every reranked
candidate uncapped, and both callers resolve every candidate's path before this call, so this is the
one place selection happens.

There is deliberately no normalisation here. A softmax over the incoming scores would turn a single
confident candidate into 0.99 and lift forty-nine near-zero candidates over a 0.01 cutoff, which is
the failure the routing study records as "two softmaxes": a page of suggestions where one skill was
meant. ``cutoff`` is a probability threshold, not a share of a recalculated distribution.

Nothing here reads a clock, a file or the network, so selection replays against recorded scores.
"""

from __future__ import annotations

from typing import Container, List, Mapping, Sequence, Tuple

from .config import Config

#: One selected suggestion: the catalog name and its rerank probability.
Suggestion = Tuple[str, float]


def select(
    scores: Sequence[float],
    names: Sequence[str],
    cfg: Config,
    paths: Mapping[str, str],
    suppressed: Container[str] = (),
) -> List[Suggestion]:
    """Return the admissible suggestions, best first, capped at ``cfg.max_suggestions`` unless it is 0.

    *scores* and *names* are parallel, in shortlist order, and every score is a rerank probability.
    *paths* maps each candidate the caller resolved to the absolute ``SKILL.md`` path the host lookup
    returned; a name the map does not carry, or carries with an empty value, is unresolved and is
    dropped. *suppressed* holds the names visible in the request's own prior blocks. An empty input,
    or no admissible score at or above the cutoff, returns ``[]``. The result never names a skill
    outside *names*, never holds a suppressed or pathless name, and never holds more than
    ``cfg.max_suggestions`` rows when that value is not 0. A length mismatch pairs as far as both
    sequences go, so a name without a score is never selected.
    """
    if not scores or not names:
        return []
    cutoff = cfg.cutoff
    eligible: List[Suggestion] = []
    for score, name in zip(scores, names):
        if name in suppressed or not paths.get(name):
            continue
        probability = float(score)
        if probability >= cutoff:
            eligible.append((name, probability))
    # ``list.sort`` is stable, so equal probabilities keep shortlist order, as the reference
    # selector does.
    eligible.sort(key=lambda suggestion: suggestion[1], reverse=True)
    limit = cfg.max_suggestions
    if limit > 0:
        return eligible[:limit]
    # Zero means unlimited, not "select nothing": the ticket's read of the setting at every reader.
    return eligible
