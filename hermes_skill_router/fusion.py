"""Reciprocal rank fusion, the recency bump and the shortlist cut (module M16, contract C16).

The fusion rule is fixed by the design:

* ``score = sum over channels of weight / (20 + rank)``, with semantic ``1.0``, BM25 ``0.5`` and
  associations ``1.0``. Only ranks are fused. A channel's own scores live on their own scale, so
  adding them would let one channel dominate the list.
* Ranks are 1-based positions in each channel's returned order. A channel that returns no rank
  contributes nothing, not a zero score.
* Ties in the fused score are broken by name, ascending, so the order does not depend on iteration
  order.
* The recency bump moves a candidate up by ``round(capPositions * min(usage, 1))`` places, once,
  before the cut. It only reorders records it is given, so it can never introduce a candidate that
  no channel returned. Rounding is half up, the same rule the reference router's ``Math.round``
  applies, so the two orders agree on every value.
* ``mode: cosine`` returns the cosine order untouched. A disabled channel and a zero weight are the
  same input: both contribute no score.

This module reads no file, no clock and no network. The caller hands it ranks, usage values and the
shortlist size, which is what lets a recorded fixture replay the fused order without a live router.
The path gate on the bump lives in the caller: a candidate whose catalog path does not resolve is
left out of the usage map, and :func:`apply_frecency` then gives it no movement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: The reciprocal rank fusion constant, fixed by the design.
RRF_CONSTANT = 20

#: The channel weights, fixed by the design.
CHANNEL_WEIGHTS: Dict[str, float] = {
    "semantic": 1.0,
    "bm25": 0.5,
    "associations": 1.0,
}

#: The three channels, in the order fusion accumulates them.
CHANNELS: Tuple[str, ...] = ("semantic", "bm25", "associations")

#: The shortlist size the fused pass cuts to when no valid size is supplied.
DEFAULT_FUSION_SHORTLIST_SIZE = 50

#: The association support floor: distinct prompts, from the fusion study.
ASSOCIATION_SUPPORT_FLOOR = 3

#: Frecency defaults.
DEFAULT_FRECENCY_HALF_LIFE_DAYS = 14
DEFAULT_FRECENCY_CAP_POSITIONS = 5

#: The three fusion modes. ``cosine`` is the single rollback switch.
FUSION_MODES: Tuple[str, ...] = ("live", "shadow", "cosine")


@dataclass(frozen=True)
class FusionConfig:
    """The resolved fusion settings, with every value inside its own rule."""

    mode: str = "live"
    semantic: bool = True
    bm25: bool = True
    associations: bool = True
    half_life_days: float = DEFAULT_FRECENCY_HALF_LIFE_DAYS
    cap_positions: int = DEFAULT_FRECENCY_CAP_POSITIONS

    @property
    def enabled(self) -> bool:
        """Return whether any channel can contribute."""
        return self.semantic or self.bm25 or self.associations


@dataclass(frozen=True)
class CandidateDiagnostic:
    """One fused candidate: its channel ranks, its bump and its final position."""

    name: str
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    association_rank: Optional[int] = None
    frecency_movement: int = 0
    final_shortlist_position: int = 0


def resolve_fusion_config(raw: Any = None) -> FusionConfig:
    """Return the settings in *raw* with every value defaulted and bounded.

    *raw* is either a :class:`FusionConfig` or a mapping shaped like the reference router's keys:
    ``mode``, ``channels`` (one boolean or ``{"enabled": ...}`` per channel) and ``frecency``
    (``halfLifeDays`` and ``capPositions``, or their snake_case spellings). A value of the wrong
    type falls back to its default instead of selecting a fourth path or producing a non-integer
    movement.
    """
    if isinstance(raw, FusionConfig):
        return raw
    record = raw if isinstance(raw, Mapping) else {}
    mode = record.get("mode")
    channels = record.get("channels")
    channels = channels if isinstance(channels, Mapping) else {}
    frecency = record.get("frecency")
    frecency = frecency if isinstance(frecency, Mapping) else {}

    def switch(name: str) -> bool:
        value = channels.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, Mapping):
            enabled = value.get("enabled")
            return enabled if isinstance(enabled, bool) else True
        return True

    half_life = frecency.get("halfLifeDays", frecency.get("half_life_days"))
    cap = frecency.get("capPositions", frecency.get("cap_positions"))
    return FusionConfig(
        mode=mode if mode in FUSION_MODES else "live",
        semantic=switch("semantic"),
        bm25=switch("bm25"),
        associations=switch("associations"),
        half_life_days=_positive(half_life, DEFAULT_FRECENCY_HALF_LIFE_DAYS),
        cap_positions=_cap(cap),
    )


def fuse_rankings(
    channels: Mapping[str, Optional[Sequence[str]]],
    config: Any = None,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> List[CandidateDiagnostic]:
    """Fuse the channel ranks into the ordered union, with one diagnostic per candidate.

    *channels* carries ``cosine`` and zero or more of ``semantic``, ``bm25`` and ``associations``,
    each a best-first list of names. ``cosine`` is the rollback order and is never fused by itself:
    the semantic channel is the cosine order when that channel is enabled. An absent channel is the
    same input as a disabled one. *weights* exists so a caller can prove a zero weight and a
    disabled channel produce the same order; the shipped weights are :data:`CHANNEL_WEIGHTS`.
    """
    settings = resolve_fusion_config(config)
    effective = _weights(settings, weights)
    if settings.mode == "cosine" or not any(value > 0 for value in effective.values()):
        return _legacy(channels.get("cosine") or [])
    rank_maps = {name: _rank_map(channels.get(name)) for name in CHANNELS}
    return [
        CandidateDiagnostic(
            name=name,
            semantic_rank=rank_maps["semantic"].get(name),
            bm25_rank=rank_maps["bm25"].get(name),
            association_rank=rank_maps["associations"].get(name),
            final_shortlist_position=position,
        )
        for position, name in enumerate(
            _ordered(_accumulate(rank_maps, effective)), start=1
        )
    ]


def apply_frecency(
    records: Sequence[CandidateDiagnostic],
    usage: Mapping[str, float],
    config: Any = None,
) -> List[CandidateDiagnostic]:
    """Move each candidate up by its capped usage, once, keeping the fused order as the tie-break.

    A candidate absent from *usage*, or carrying a non-finite or non-positive value, does not move.
    The function only reorders the records it was given, so a usage row for a name no channel
    returned cannot introduce a candidate.
    """
    settings = resolve_fusion_config(config)
    cap = settings.cap_positions
    keyed = [
        (index, _movement(usage.get(record.name), cap), record)
        for index, record in enumerate(records)
    ]
    keyed.sort(key=lambda item: (item[0] - item[1], item[0]))
    return [
        CandidateDiagnostic(
            name=record.name,
            semantic_rank=record.semantic_rank,
            bm25_rank=record.bm25_rank,
            association_rank=record.association_rank,
            frecency_movement=movement,
            final_shortlist_position=position,
        )
        for position, (_index, movement, record) in enumerate(keyed, start=1)
    ]


def cut_shortlist(
    records: Sequence[CandidateDiagnostic],
    limit: int,
) -> List[CandidateDiagnostic]:
    """Return the first *limit* records, with their final positions renumbered from one."""
    if isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0:
        size = limit
    else:
        size = DEFAULT_FUSION_SHORTLIST_SIZE
    return [
        CandidateDiagnostic(
            name=record.name,
            semantic_rank=record.semantic_rank,
            bm25_rank=record.bm25_rank,
            association_rank=record.association_rank,
            frecency_movement=record.frecency_movement,
            final_shortlist_position=position,
        )
        for position, record in enumerate(records[:size], start=1)
    ]


def movement(usage: Optional[float], cap_positions: int) -> int:
    """Return ``round(capPositions * min(usage, 1))``, half up, or zero without a usable value."""
    return _movement(usage, cap_positions)


def _movement(usage: Optional[float], cap_positions: float) -> int:
    """The effective move for one usage value: positive, finite and capped, never above the cap."""
    if usage is None or isinstance(usage, bool):
        return 0
    if not isinstance(usage, (int, float)):
        return 0
    value = float(usage)
    if not math.isfinite(value) or value <= 0:
        return 0
    if not isinstance(cap_positions, (int, float)) or not math.isfinite(float(cap_positions)):
        return 0
    cap = float(cap_positions)
    if cap <= 0:
        return 0
    return int(math.floor(cap * min(value, 1.0) + 0.5))


def _weights(settings: FusionConfig, weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """Return the effective weight of every channel: zero unless enabled."""
    configured = weights if isinstance(weights, Mapping) else {}
    effective: Dict[str, float] = {}
    for name in CHANNELS:
        enabled = getattr(settings, name)
        if not enabled:
            effective[name] = 0.0
            continue
        value = configured.get(name, CHANNEL_WEIGHTS[name])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            value = CHANNEL_WEIGHTS[name]
        effective[name] = float(value) if math.isfinite(float(value)) else 0.0
    return effective


def _rank_map(names: Optional[Sequence[str]]) -> Dict[str, int]:
    """Return a 1-based rank per name; the first occurrence wins."""
    ranks: Dict[str, int] = {}
    for position, name in enumerate(names or (), start=1):
        if name not in ranks:
            ranks[name] = position
    return ranks


def _accumulate(
    rank_maps: Mapping[str, Mapping[str, int]],
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Sum the weighted reciprocal ranks into one score per name."""
    scores: Dict[str, float] = {}
    for channel in CHANNELS:
        weight = weights.get(channel, 0.0)
        if weight == 0:
            continue
        for name, rank in rank_maps[channel].items():
            scores[name] = scores.get(name, 0.0) + weight / (RRF_CONSTANT + rank)
    return scores


def _ordered(scores: Mapping[str, float]) -> List[str]:
    """Return names by descending score, ties by name ascending."""
    return sorted(scores, key=lambda name: (-scores[name], name))


def _legacy(cosine: Sequence[str]) -> List[CandidateDiagnostic]:
    """Return the legacy order as diagnostics: no channel rank, no movement."""
    return [
        CandidateDiagnostic(name=name, final_shortlist_position=position)
        for position, name in enumerate(cosine, start=1)
    ]


def _positive(value: Any, fallback: float) -> float:
    """Return *value* when it is a positive finite number, otherwise *fallback*."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    number = float(value)
    return number if math.isfinite(number) and number > 0 else fallback


def _cap(value: Any) -> int:
    """Return an integer position cap of zero or more, otherwise the default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_FRECENCY_CAP_POSITIONS
    number = float(value)
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return DEFAULT_FRECENCY_CAP_POSITIONS
    return int(number)
