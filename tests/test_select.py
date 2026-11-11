"""Contract C7: the suppressed set, the resolved paths, the cutoff, the sort and the cap.

The scores come from the reranker as probabilities, so these tests pin the reference selector's parity
on fixed vectors: a single confident candidate plus near-zero siblings must select one row, an
inclusive cutoff keeps the exact boundary value, and the cap truncates after the sort. They also pin
the two rules the routing pass hands over: a suppressed name and a name with no resolvable read path
are dropped before the cap counts rows, and a cap of 0 means unlimited.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.select import select  # noqa: E402


def make_config(**overrides):
    return load(overrides)


def paths_for(*names):
    """The path map a caller builds after resolving every candidate through the host lookup."""
    return {name: f"/skills/{name}/SKILL.md" for name in names}


def test_the_all_but_one_vector_selects_one_row():
    """scores [1.0, 0.0, 0.0] at cutoff 0.01 select one row, not three."""
    names = ["confident", "near-zero-a", "near-zero-b"]
    selected = select(
        [1.0, 0.0, 0.0],
        names,
        make_config(cutoff=0.01, maxSuggestions=8),
        paths_for(*names),
    )

    # A second softmax over these three scores would make every row 0.99 and select all of them.
    assert selected == [("confident", 1.0)]


def test_equal_probabilities_select_both():
    selected = select(
        [0.5, 0.5], ["alpha", "beta"], make_config(cutoff=0.01, maxSuggestions=8),
        paths_for("alpha", "beta"),
    )

    assert selected == [("alpha", 0.5), ("beta", 0.5)]


def test_a_score_equal_to_the_cutoff_is_kept_and_one_epsilon_below_is_dropped():
    cutoff = 0.01
    below = math.nextafter(cutoff, 0.0)
    assert below < cutoff

    selected = select(
        [cutoff, below],
        ["at-cutoff", "below-cutoff"],
        make_config(cutoff=cutoff, maxSuggestions=8),
        paths_for("at-cutoff", "below-cutoff"),
    )

    assert selected == [("at-cutoff", cutoff)]


def test_a_zero_cutoff_selects_a_zero_score_inclusively():
    selected = select(
        [0.0, 0.0], ["alpha", "beta"], make_config(cutoff=0.0, maxSuggestions=8),
        paths_for("alpha", "beta"),
    )

    assert selected == [("alpha", 0.0), ("beta", 0.0)]


def test_a_cutoff_of_one_requires_certainty():
    selected = select(
        [1.0, 0.999], ["certain", "almost"], make_config(cutoff=1.0),
        paths_for("certain", "almost"),
    )

    assert selected == [("certain", 1.0)]


def test_the_survivors_are_sorted_by_descending_score():
    selected = select(
        [0.3, 0.9, 0.6], ["low", "high", "middle"], make_config(cutoff=0.01, maxSuggestions=8),
        paths_for("low", "high", "middle"),
    )

    assert selected == [("high", 0.9), ("middle", 0.6), ("low", 0.3)]


def test_equal_scores_keep_the_shortlist_order():
    selected = select(
        [0.4, 0.9, 0.4], ["first", "best", "second"], make_config(cutoff=0.01, maxSuggestions=8),
        paths_for("first", "best", "second"),
    )

    assert [name for name, _score in selected] == ["best", "first", "second"]


def test_max_suggestions_truncates_after_the_sort():
    selected = select(
        [0.5, 0.4, 0.3, 0.2, 0.1],
        ["a", "b", "c", "d", "e"],
        make_config(cutoff=0.01, maxSuggestions=3),
        paths_for("a", "b", "c", "d", "e"),
    )

    assert selected == [("a", 0.5), ("b", 0.4), ("c", 0.3)]
    assert len(selected) == 3


def test_max_suggestions_keeps_exactly_that_many_from_a_larger_eligible_set():
    selected = select(
        [0.9, 0.8, 0.7, 0.6],
        ["a", "b", "c", "d"],
        make_config(cutoff=0.01, maxSuggestions=4),
        paths_for("a", "b", "c", "d"),
    )

    assert len(selected) == 4


def test_a_cap_of_five_keeps_the_five_highest_of_twelve_candidates():
    scores = [0.01 + index / 100.0 for index in range(12)]
    names = [f"skill-{index}" for index in range(12)]

    selected = select(scores, names, make_config(cutoff=0.01, maxSuggestions=5), paths_for(*names))

    assert [name for name, _score in selected] == [
        "skill-11",
        "skill-10",
        "skill-9",
        "skill-8",
        "skill-7",
    ]


def test_max_suggestions_zero_selects_every_eligible_row():
    names = ["a", "b"]

    selected = select(
        [0.9, 0.8], names, make_config(cutoff=0.01, maxSuggestions=0), paths_for(*names)
    )

    assert selected == [("a", 0.9), ("b", 0.8)]


def test_the_cutoff_is_never_lifted_by_the_cap():
    names = ["a", "b", "c"]
    scores = [0.9, 0.8, 0.7]

    assert select(scores, names, make_config(cutoff=0.01, maxSuggestions=10), paths_for(*names)) == [
        ("a", 0.9),
        ("b", 0.8),
        ("c", 0.7),
    ]
    assert len(
        select(scores, names, make_config(cutoff=0.01, maxSuggestions=0), paths_for(*names))
    ) == 3


def test_a_suppressed_name_is_skipped():
    selected = select(
        [0.9, 0.8],
        ["seen", "fresh"],
        make_config(cutoff=0.01, maxSuggestions=10),
        paths_for("seen", "fresh"),
        suppressed={"seen"},
    )

    assert selected == [("fresh", 0.8)]


def test_everything_suppressed_selects_nothing():
    selected = select(
        [0.9, 0.8],
        ["seen", "also-seen"],
        make_config(cutoff=0.01, maxSuggestions=10),
        paths_for("seen", "also-seen"),
        suppressed={"seen", "also-seen"},
    )

    assert selected == []


def test_a_suppressed_name_does_not_spend_a_cap_slot():
    selected = select(
        [0.9, 0.8],
        ["seen", "fresh"],
        make_config(cutoff=0.01, maxSuggestions=1),
        paths_for("seen", "fresh"),
        suppressed={"seen"},
    )

    assert selected == [("fresh", 0.8)]


def test_a_pathless_row_is_dropped_and_the_next_admissible_row_takes_its_slot():
    selected = select(
        [0.9, 0.8, 0.7],
        ["pathless", "second", "third"],
        make_config(cutoff=0.01, maxSuggestions=2),
        paths_for("second", "third"),
    )

    assert selected == [("second", 0.8), ("third", 0.7)], (
        "a candidate whose read path did not resolve consumed a cap slot"
    )


def test_a_name_the_path_map_does_not_carry_is_never_selected():
    selected = select(
        [0.9, 0.8],
        ["unresolved", "resolved"],
        make_config(cutoff=0.0, maxSuggestions=10),
        {"resolved": "/skills/resolved/SKILL.md"},
    )

    assert selected == [("resolved", 0.8)]


def test_a_name_with_an_empty_path_is_never_selected():
    selected = select(
        [0.9],
        ["blank"],
        make_config(cutoff=0.0, maxSuggestions=10),
        {"blank": ""},
    )

    assert selected == []


def test_empty_input_returns_empty():
    cfg = make_config(cutoff=0.01, maxSuggestions=8)

    assert select([], [], cfg, {}) == []
    assert select([], ["a"], cfg, paths_for("a")) == []
    assert select([0.9], [], cfg, {}) == []


def test_names_and_scores_stay_paired():
    selected = select(
        [0.2, 0.9], ["weak", "strong"], make_config(cutoff=0.01, maxSuggestions=8),
        paths_for("weak", "strong"),
    )

    assert selected == [("strong", 0.9), ("weak", 0.2)]


def test_a_nan_score_is_never_selected():
    """NaN fails every comparison, so a corrupt score drops out instead of sorting first."""
    selected = select(
        [float("nan"), 0.5], ["corrupt", "sound"], make_config(cutoff=0.0, maxSuggestions=8),
        paths_for("corrupt", "sound"),
    )

    assert selected == [("sound", 0.5)]


def test_integer_scores_come_back_as_floats():
    selected = select(
        [1, 0], ["one", "zero"], make_config(cutoff=0.01, maxSuggestions=8),
        paths_for("one", "zero"),
    )

    assert selected == [("one", 1.0)]
    assert isinstance(selected[0][1], float)


def test_a_length_mismatch_never_invents_a_score():
    """The contract requires equal lengths; a mismatch pairs as far as both sequences go."""
    cfg = make_config(cutoff=0.01, maxSuggestions=8)

    assert select([0.9], ["a", "b"], cfg, paths_for("a", "b")) == [("a", 0.9)]
    assert select([0.9, 0.5], ["a"], cfg, paths_for("a")) == [("a", 0.9)]


@pytest.mark.parametrize("cap", [1, 3, 50])
def test_the_result_never_exceeds_the_cap(cap):
    scores = [0.9 - index * 0.01 for index in range(20)]
    names = [f"skill-{index}" for index in range(20)]

    assert (
        len(select(scores, names, make_config(cutoff=0.01, maxSuggestions=cap), paths_for(*names)))
        <= cap
    )


def test_a_cap_of_zero_never_truncates_twenty_eligible_rows():
    scores = [0.9 - index * 0.01 for index in range(20)]
    names = [f"skill-{index}" for index in range(20)]

    selected = select(
        scores, names, make_config(cutoff=0.01, maxSuggestions=0), paths_for(*names)
    )

    assert len(selected) == 20
