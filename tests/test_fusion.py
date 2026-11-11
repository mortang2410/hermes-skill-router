"""Contract C16: reciprocal rank fusion, the recency bump, the cut and the parity fixture.

The fusion module is pure, so most tests drive it directly with hand-written channel orders and a
hand-written usage map. The library tests at the end drive the real shortlist: the cosine rollback
must not open the store or score BM25, the shadow mode must log the diagnostics and return the
cosine order, the bump must be gated on a resolvable catalog path, and a missing or unreadable store
must leave the other channels routing with one warning.

The fixture test replays one recorded candidate set from the reference router's own modules, so a
change to either side that moves the fused order fails on the side that moved.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import associations  # noqa: E402
from hermes_skill_router import bm25  # noqa: E402
from hermes_skill_router import fusion  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.fusion import (  # noqa: E402
    ASSOCIATION_SUPPORT_FLOOR,
    CHANNEL_WEIGHTS,
    DEFAULT_FRECENCY_CAP_POSITIONS,
    DEFAULT_FRECENCY_HALF_LIFE_DAYS,
    RRF_CONSTANT,
    apply_frecency,
    cut_shortlist,
    fuse_rankings,
    movement,
    resolve_fusion_config,
)
from hermes_skill_router.library import Index, shortlist  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000


def make_index(names=("alpha", "beta", "gamma", "delta")):
    return Index(
        names=list(names),
        descriptions=[f"{name} skill" for name in names],
        vectors=[[1.0] for _name in names],
        dim=1,
        fingerprint="test",
    )


def seed_usage(path, now_ms, names):
    """Write one live, non-ambient load row per *name* into a fresh association store."""
    store = associations.open_store(str(path))
    try:
        prompt_ids = []
        for position in range(5):
            cursor = store.conn.execute(
                """INSERT INTO prompt_event
                   (session_id, turn, prompt_hash, terms, length, observed_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (f"session-{position}", position, f"hash-{position}", "review store", 12, now_ms - DAY),
            )
            prompt_ids.append(int(cursor.lastrowid))
        for position, name in enumerate(names):
            cursor = store.conn.execute(
                """INSERT INTO skill (harness, name, path, first_seen_ms, last_seen_ms)
                   VALUES (?, ?, ?, ?, ?)""",
                ("hermes", name, associations.catalog_path(name), now_ms - DAY, now_ms),
            )
            store.conn.execute(
                """INSERT INTO load_event
                   (prompt_event_id, skill_id, evidence, strength, origin, source, observed_ms,
                    dedup_key)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt_ids[position],
                    int(cursor.lastrowid),
                    "tool",
                    1.0,
                    "human",
                    "backfill",
                    now_ms - DAY,
                    f"{name}:one",
                ),
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# The constants and the fused order.
# ---------------------------------------------------------------------------


def test_the_rrf_constant_and_weights_are_pinned():
    assert RRF_CONSTANT == 20
    assert CHANNEL_WEIGHTS == {"semantic": 1.0, "bm25": 0.5, "associations": 1.0}
    assert (DEFAULT_FRECENCY_HALF_LIFE_DAYS, DEFAULT_FRECENCY_CAP_POSITIONS) == (14, 5)
    assert ASSOCIATION_SUPPORT_FLOOR == 3


def test_fuse_rankings_sums_the_weighted_reciprocal_ranks():
    fused = fuse_rankings(
        {
            "cosine": ["a", "b", "c"],
            "semantic": ["a", "b", "c"],
            "bm25": ["b", "a"],
            "associations": ["c"],
        }
    )

    assert [record.name for record in fused] == ["c", "a", "b"]
    first = fused[0]
    assert (first.semantic_rank, first.bm25_rank, first.association_rank) == (3, None, 1)
    assert first.final_shortlist_position == 1


def test_fused_ties_are_broken_by_name():
    fused = fuse_rankings(
        {"cosine": ["alpha", "beta"], "semantic": ["alpha"], "bm25": ["beta"]},
        weights={"semantic": 0.5, "bm25": 0.5, "associations": 0.0},
    )
    assert [record.name for record in fused] == ["alpha", "beta"]


def test_a_disabled_channel_and_a_zero_weight_produce_the_same_order():
    channels = {
        "cosine": ["a", "b", "c"],
        "semantic": ["a", "b", "c"],
        "bm25": ["c", "a"],
        "associations": ["b"],
    }
    disabled = fuse_rankings(channels, {"channels": {"bm25": {"enabled": False}}})
    zero = fuse_rankings(channels, "live", weights={"bm25": 0.0})
    assert [record.name for record in disabled] == [record.name for record in zero]


def test_an_absent_channel_contributes_no_rank():
    fused = fuse_rankings({"cosine": ["a", "b"], "semantic": ["b", "a"]})
    assert [record.name for record in fused] == ["b", "a"]


def test_cosine_mode_returns_the_cosine_order_untouched():
    fused = fuse_rankings(
        {"cosine": ["b", "a", "c"], "semantic": ["a", "b"], "bm25": ["c"]},
        {"mode": "cosine"},
    )
    assert [record.name for record in fused] == ["b", "a", "c"]
    assert all(record.semantic_rank is None for record in fused)


def test_resolve_fusion_config_defaults_and_bounds_every_value():
    settings = resolve_fusion_config(None)
    assert settings.mode == "live"
    assert settings.enabled is True
    assert (settings.half_life_days, settings.cap_positions) == (14, 5)

    bad = resolve_fusion_config(
        {
            "mode": "ranked",
            "channels": {"semantic": {"enabled": False}, "bm25": "on"},
            "frecency": {"halfLifeDays": 0, "capPositions": 2.5},
        }
    )
    assert bad.mode == "live"
    assert bad.semantic is False
    assert bad.bm25 is True
    assert (bad.half_life_days, bad.cap_positions) == (14, 5)


# ---------------------------------------------------------------------------
# The bump and the cut.
# ---------------------------------------------------------------------------


def test_the_bump_is_half_up_and_saturates_at_the_cap():
    assert movement(0.5, 5) == 3, "half up is the reference router's rounding, not banker's rounding"
    assert movement(0.99, 5) == 5
    assert movement(1.0, 5) == 5
    assert movement(4.0, 5) == 5, "a usage score above one moves no further than the cap"
    assert movement(0.0, 5) == 0
    assert movement(None, 5) == 0
    assert movement(0.9, 0) == 0


def test_the_bump_moves_one_candidate_up_by_its_movement():
    records = fuse_rankings(
        {"cosine": [f"n{index}" for index in range(8)], "semantic": [f"n{index}" for index in range(8)]}
    )
    bumped = apply_frecency(records, {"n5": 1.0}, {"frecency": {"capPositions": 5}})

    assert bumped[1].name == "n5", "n5 moved to the second slot, behind the untouched n0"
    assert bumped[1].frecency_movement == 5
    assert [record.name for record in bumped].count("n5") == 1


def test_the_bump_never_introduces_a_candidate_no_channel_returned():
    records = fuse_rankings({"cosine": ["a", "b"], "semantic": ["a", "b"]})
    bumped = apply_frecency(records, {"ghost": 1.0}, {"frecency": {"capPositions": 5}})
    assert [record.name for record in bumped] == ["a", "b"]


def test_the_cut_happens_after_the_bump():
    records = fuse_rankings(
        {"cosine": [f"n{index}" for index in range(8)], "semantic": [f"n{index}" for index in range(8)]}
    )
    without = cut_shortlist(records, 5)
    with_bump = cut_shortlist(
        apply_frecency(records, {"n5": 1.0}, {"frecency": {"capPositions": 5}}), 5
    )

    assert "n5" not in [record.name for record in without]
    assert "n5" in [record.name for record in with_bump]
    assert with_bump[1].name == "n5"
    assert [record.final_shortlist_position for record in with_bump] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# The library pass: mode rollback, shadow logging, the path gate and degrade paths.
# ---------------------------------------------------------------------------


def test_cosine_mode_reads_no_term_index_opens_no_store_and_does_not_bump(monkeypatch, tmp_path):
    cfg = load({"cachePath": str(tmp_path / "index.json"), "fusionMode": "cosine"})
    index = make_index()

    monkeypatch.setattr(
        library_module.bm25,
        "score_ranks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cosine read the term index")),
    )
    monkeypatch.setattr(
        library_module.associations,
        "open_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cosine opened the store")),
    )

    rows = shortlist(index, [1.0], "alpha review", 3, cfg=cfg)

    assert [name for name, _description in rows] == ["alpha", "beta", "gamma"]


def test_shadow_mode_logs_the_diagnostics_and_returns_the_cosine_order(tmp_path, caplog):
    cfg = load(
        {
            "cachePath": "",
            "fusionMode": "shadow",
            "channelsBm25": False,
            "channelsAssociations": False,
        }
    )
    index = make_index()

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.library"):
        rows = shortlist(index, [1.0], "alpha review", 2, cfg=cfg)

    assert [name for name, _description in rows] == ["alpha", "beta"]
    lines = [record.getMessage() for record in caplog.records if "mode=shadow" in record.getMessage()]
    assert len(lines) == 1
    assert '"name":"alpha"' in lines[0]
    assert '"position":2' in lines[0]


def test_the_bump_is_gated_on_a_resolvable_catalog_path(tmp_path):
    now_ms = T0 + DAY
    cfg = load(
        {
            "cachePath": str(tmp_path / "index.json"),
            "channelsBm25": False,
            "channelsAssociations": False,
        }
    )
    seed_usage(tmp_path / "associations.db", now_ms, ["alpha"])
    index = make_index(("one", "two", "alpha", "three", "beta", "four", "five", "six"))

    def resolver(name):
        return None if name == "alpha" else f"/skills/{name}/SKILL.md"

    records = []
    rows = shortlist(
        index,
        [1.0],
        "review store",
        8,
        cfg=cfg,
        resolver=resolver,
        now_ms=now_ms,
        on_diagnostic=records.append,
    )

    positions = {record.name: record.final_shortlist_position for record in records}
    assert positions["alpha"] == 3, "a pathless candidate must keep its fused position"
    alpha = next(record for record in records if record.name == "alpha")
    assert alpha.frecency_movement == 0, "usage for a pathless candidate produced a movement"
    assert [name for name, _description in rows] == [
        "one",
        "two",
        "alpha",
        "three",
        "beta",
        "four",
        "five",
        "six",
    ]


def test_a_missing_store_leaves_the_other_channels_routing_with_one_warning(tmp_path, caplog):
    cfg = load({"cachePath": str(tmp_path / "index.json"), "channelsBm25": False})
    index = make_index()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        rows = shortlist(index, [1.0], "alpha review", 2, cfg=cfg)

    assert [name for name, _description in rows] == ["alpha", "beta"]
    warnings = [
        record for record in caplog.records if "association database" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_an_unreadable_store_leaves_the_other_channels_routing_with_one_warning(tmp_path, caplog):
    cfg = load({"cachePath": str(tmp_path / "index.json"), "channelsBm25": False})
    (tmp_path / "associations.db").write_text("not a database", encoding="utf-8")
    index = make_index()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        rows = shortlist(index, [1.0], "alpha review", 2, cfg=cfg)

    assert [name for name, _description in rows] == ["alpha", "beta"]
    warnings = [
        record for record in caplog.records if "association database" in record.getMessage()
    ]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# The parity fixture, recorded once from the reference router.
# ---------------------------------------------------------------------------


def test_the_parity_fixture_replays_the_recorded_fused_order():
    """Replay one recorded candidate set from the reference router's own modules.

    The fixture lives in ``tests/parity-fusion.json`` and its ``comment`` field names
    the generating command and the reference commit ``c5f33fa``. The test asserts the
    recorded BM25 order, the fused order, the bumped order, the cut order and the
    cut diagnostics, so a change on either side that moves one of them fails here.
    The assertion covers identical inputs only; it is not a comparison of two live
    routers.
    """
    fixture = json.loads((ROOT / "tests" / "parity-fusion.json").read_text(encoding="utf-8"))
    assert "c5f33fa" in fixture["comment"]

    rows = [(entry["name"], entry["name"] + "\n" + entry["description"]) for entry in fixture["catalog"]]
    term_index = bm25.build_term_index(rows)
    assert bm25.score_ranks(term_index, fixture["query"], len(rows)) == fixture["bm25"]

    fused = fuse_rankings(
        {
            "cosine": fixture["cosine"],
            "semantic": fixture["semantic"],
            "bm25": fixture["bm25"],
            "associations": fixture["associations"],
        }
    )
    assert [record.name for record in fused] == fixture["fused_order"]

    bumped = apply_frecency(fused, fixture["usage"], {"frecency": {"capPositions": 5}})
    assert [record.name for record in bumped] == fixture["bumped_order"]

    cut = cut_shortlist(bumped, fixture["limit"])
    assert [record.name for record in cut] == fixture["cut_order"]
    for record, expected in zip(cut, fixture["diagnostics"]):
        assert record.name == expected["name"]
        assert record.semantic_rank == expected["semantic_rank"]
        assert record.bm25_rank == expected["bm25_rank"]
        assert record.association_rank == expected["association_rank"]
        assert record.frecency_movement == expected["frecency_movement"]
        assert record.final_shortlist_position == expected["final_shortlist_position"]


# ---------------------------------------------------------------------------
# End to end: seeded association rows contribute ranks to a fused order.
# ---------------------------------------------------------------------------


def _seed_association_rank(store, skill_name, term, now_ms, prompts=3):
    """Give *skill_name* one (term, skill) row with *prompts* distinct supports.

    The seed adds extra non-loading prompts so the skill stays below the ambient
    threshold: 3 loads in 3 prompts would be ambient and rank nothing.
    """
    skill = associations.resolve_skill_name(store, skill_name)
    assert skill is not None
    for index in range(prompts):
        cursor = store.conn.execute(
            """INSERT INTO prompt_event
               (session_id, turn, prompt_hash, terms, length, observed_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (f"assoc-session-{index}", 1, f"assoc-hash-{skill_name}-{term}-{index}", term, 20, now_ms - DAY),
        )
        store.conn.execute(
            """INSERT INTO load_event
               (prompt_event_id, skill_id, evidence, strength, origin, source, observed_ms,
                dedup_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(cursor.lastrowid),
                skill.id,
                "tool",
                1.0,
                "human",
                "backfill",
                now_ms - DAY,
                f"{skill_name}:{term}:{index}",
            ),
        )
    for index in range(10):
        store.conn.execute(
            """INSERT INTO prompt_event
               (session_id, turn, prompt_hash, terms, length, observed_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (f"filler-session-{index}", 1, f"filler-hash-{index}", term, 20, now_ms - DAY),
        )
    store.conn.execute(
        "INSERT OR REPLACE INTO association (skill_id, term, weight, support, first_ms, last_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (skill.id, term, 0.8, prompts, now_ms - DAY, now_ms - DAY),
    )


def test_seeded_association_rows_contribute_ranks_to_the_fused_order(tmp_path):
    """A skill with support>=3 for a query term ranks in the fused order."""
    now_ms = T0 + DAY
    cfg = load(
        {
            "cachePath": str(tmp_path / "index.json"),
            "associationDbPath": str(tmp_path / "associations.db"),
            "channelsBm25": False,
            "channelsAssociations": True,
        }
    )
    names = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    store = associations.open_store(str(tmp_path / "associations.db"))
    try:
        from hermes_skill_router.associations import CatalogSkill

        associations.reconcile_catalog(store, [CatalogSkill(name) for name in names], now_ms)
        # beta gets term "zed" with support 3, so the association channel ranks it.
        _seed_association_rank(store, "beta", "zed", now_ms, prompts=3)
    finally:
        store.close()

    index = make_index(names)
    records = []
    rows = shortlist(
        index,
        [1.0],
        "zed",
        6,
        cfg=cfg,
        resolver=lambda name: f"/skills/{name}/SKILL.md",
        now_ms=now_ms,
        on_diagnostic=records.append,
    )
    assert [name for name, _description in rows] == [record.name for record in records]
    by_name = {record.name: record for record in records}
    assert by_name["beta"].association_rank == 1, "the seeded row must contribute an association rank"
    assert by_name["alpha"].association_rank is None
    assert by_name["alpha"].semantic_rank is not None


def test_association_rows_below_the_floor_contribute_no_rank_end_to_end(tmp_path):
    """Support 2 on the only (term, skill) row leaves the fused order channel-free."""
    now_ms = T0 + DAY
    cfg = load(
        {
            "cachePath": str(tmp_path / "index.json"),
            "associationDbPath": str(tmp_path / "associations.db"),
            "channelsBm25": False,
            "channelsAssociations": True,
        }
    )
    names = ("alpha", "beta", "gamma", "delta")
    store = associations.open_store(str(tmp_path / "associations.db"))
    try:
        from hermes_skill_router.associations import CatalogSkill

        associations.reconcile_catalog(store, [CatalogSkill(name) for name in names], now_ms)
        _seed_association_rank(store, "beta", "zed", now_ms, prompts=2)
    finally:
        store.close()

    index = make_index(names)
    records = []
    shortlist(
        index,
        [1.0],
        "zed",
        4,
        cfg=cfg,
        resolver=lambda name: f"/skills/{name}/SKILL.md",
        now_ms=now_ms,
        on_diagnostic=records.append,
    )
    assert all(record.association_rank is None for record in records), "support below 3 must not rank"


