"""Contracts C3 and C4: the session catalog record and the one selection pass.

The search tool ranks the routed half of the partition the middleware parsed, through the same pass
a turn runs. These tests drive that pass directly with a fake embedder and reranker, so they need no
endpoint, and they compare it with the names a real turn injects for the same payload and index.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router import tools as tools_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows, Partition  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from hermes_skill_router.search import (  # noqa: E402
    BUDGET_EXPIRED,
    EMBED_FAILED,
    NO_INDEX,
    RERANK_FAILED,
    catalog_rows,
    forget_session,
    remember_partition,
    select_rows,
)

QUERY = "draft a git commit message for a bug fix"


@pytest.fixture(autouse=True)
def _host_paths(monkeypatch):
    """A host lookup that resolves every name, so a turn's block carries read paths."""
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
QUERY_VECTOR = [0.0, 1.0, 0.0]

ROWS = (("alpha", "Alpha skill"), ("beta", "Beta skill"), ("gamma", "Gamma skill"))

SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant to "
    "your task, you MUST load it with skill_view(name) and follow its instructions.\n"
    "\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "    - gamma: Gamma skill\n"
    "  creative [names only]: delta\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)

VECTORS = {
    "alpha\nAlpha skill": [1.0, 0.0, 0.0],
    "beta\nBeta skill": [0.0, 1.0, 0.0],
    "gamma\nGamma skill": [0.0, 0.0, 1.0],
}


def make_config(tmp_path, **overrides):
    settings = {"cachePath": "", "suggestionsPath": str(tmp_path / "suggestions.json")}
    settings.update(overrides)
    return load(settings)


def make_index(rows=ROWS):
    return Index(
        names=[name for name, _description in rows],
        descriptions=[description for _name, description in rows],
        vectors=[
            list(VECTORS.get(name + "\n" + description, [0.0, 0.0, 1.0]))
            for name, description in rows
        ],
        dim=3,
        fingerprint="test",
    )


def partition(routed=ROWS, kept=()):
    return Partition(
        kept=CatalogRows(tuple(name for name, _d in kept), tuple(d for _n, d in kept)),
        routed=CatalogRows(tuple(name for name, _d in routed), tuple(d for _n, d in routed)),
    )


class FakeEmbedder:
    def __init__(self, result="computed"):
        self.result = result
        self.calls = []

    def __call__(self, cfg, texts):
        self.calls.append(list(texts))
        if self.result != "computed":
            return self.result
        answer = []
        for text in texts:
            answer.append(list(VECTORS.get(text, QUERY_VECTOR)))
        return answer


class FakeReranker:
    def __init__(self, scores=(0.2, 0.9, 0.1), result="computed"):
        self.scores = list(scores)
        self.result = result
        self.calls = []

    def __call__(self, cfg, query, documents):
        self.calls.append(list(documents))
        if self.result != "computed":
            return self.result
        return self.scores[: len(documents)]


def _payload(user=QUERY):
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }


def test_a_session_with_no_parsed_catalog_has_no_rows(tmp_path):
    cfg = make_config(tmp_path)
    forget_session("session-1")

    assert catalog_rows(cfg, "session-1") is None


def test_the_record_is_the_routed_half_in_catalog_order(tmp_path):
    cfg = make_config(tmp_path)
    remember_partition("session-1", partition(kept=(("delta", ""),)))
    try:
        rows = catalog_rows(cfg, "session-1")
    finally:
        forget_session("session-1")

    assert [(row.name, row.description) for row in rows] == list(ROWS)


def test_an_empty_parsed_partition_reads_as_an_empty_sequence(tmp_path):
    cfg = make_config(tmp_path)
    remember_partition("session-1", partition(routed=()))
    try:
        assert catalog_rows(cfg, "session-1") == ()
    finally:
        forget_session("session-1")


def test_an_empty_partition_is_a_healthy_empty_answer(tmp_path):
    cfg = make_config(tmp_path)
    embedder = FakeEmbedder()

    outcome = select_rows(cfg, (), QUERY, embed=embedder)

    assert outcome.reason == ""
    assert outcome.rows == ()
    assert embedder.calls == []


def test_select_rows_repeats_the_turn_selection(tmp_path):
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()

    rewritten = rewrite(
        payload,
        cfg=cfg,
        index=index,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="session-1",
    )
    assert rewritten is not None
    turn_names = read(cfg.suggestions_path)["names"]

    outcome = select_rows(
        cfg,
        catalog_rows(cfg, "session-1"),
        QUERY,
        index=index,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        resolve=lambda name: f"/skills/{name}/SKILL.md",
    )

    assert [row.name for row in outcome.rows] == turn_names == ["alpha", "beta", "gamma"]


def test_select_rows_resolves_every_candidate_before_the_cut(tmp_path):
    cfg = make_config(tmp_path, cutoff=0.5)
    looked_up = []

    def resolve(name):
        looked_up.append(name)
        return f"/skills/{name}/SKILL.md"

    outcome = select_rows(
        cfg,
        [
            search_module.SearchRow("alpha", "Alpha skill"),
            search_module.SearchRow("delta", ""),
            search_module.SearchRow("gamma", "Gamma skill"),
        ],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"), ("delta", ""), ("gamma", "Gamma skill"))),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.5, 0.4, 0.1)),
        resolve=resolve,
    )

    assert [(row.name, row.path) for row in outcome.rows] == [
        ("alpha", "/skills/alpha/SKILL.md"),
    ]
    assert looked_up == ["alpha", "delta", "gamma"], (
        "a candidate the cutoff later rejects still has its path resolved before the cut"
    )


def test_a_refused_lookup_drops_the_row(tmp_path):
    cfg = make_config(tmp_path)

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"),)),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9,)),
        resolve=lambda name: None,
    )

    assert outcome.reason == ""
    assert outcome.rows == ()


def test_a_failed_build_names_the_no_index_arm(tmp_path):
    cfg = make_config(tmp_path)

    outcome = select_rows(cfg, [search_module.SearchRow("alpha", "Alpha skill")], QUERY, embed=FakeEmbedder(result=None))

    assert outcome.reason == NO_INDEX
    assert outcome.rows == ()


def test_an_embed_failure_names_its_arm(tmp_path):
    cfg = make_config(tmp_path)

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"),)),
        embed=FakeEmbedder(result=None),
        rerank=FakeReranker(),
    )

    assert outcome.reason == EMBED_FAILED


def test_a_rerank_failure_names_its_arm(tmp_path):
    cfg = make_config(tmp_path)

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"),)),
        embed=FakeEmbedder(),
        rerank=FakeReranker(result=None),
    )

    assert outcome.reason == RERANK_FAILED


def test_the_budget_ends_the_pass_with_its_own_arm(tmp_path):
    cfg = make_config(tmp_path)
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"),)),
        embed=embedder,
        rerank=reranker,
        deadline=0.0,
    )

    assert outcome.reason == BUDGET_EXPIRED
    assert embedder.calls == []
    assert reranker.calls == []


def test_one_deadline_covers_both_stages(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    seen = {}

    def query_vector(cfg, query, embedder, deadline):
        seen["embed"] = deadline
        return [0.0], None

    def rerank_scores(cfg, query, documents, reranker, deadline):
        seen["rerank"] = deadline
        return [0.9] * len(documents), None

    monkeypatch.setattr(search_module, "_query_vector", query_vector)
    monkeypatch.setattr(search_module, "_rerank_scores", rerank_scores)

    ranked = search_module.rank(cfg, make_index(), QUERY)

    assert ranked.reason == ""
    assert seen["embed"] is not None
    assert seen["embed"] == seen["rerank"]


def test_a_slow_rerank_cannot_outlive_the_budget(tmp_path):
    cfg = make_config(tmp_path, routingTimeoutMs=1000)

    def slow_embed(cfg, texts):
        time.sleep(0.95)
        return [[0.0]]

    def slow_rerank(cfg, query, documents):
        time.sleep(0.2)
        return [0.9] * len(documents)

    started = time.monotonic()
    ranked = search_module.rank(cfg, make_index(), QUERY, embed=slow_embed, rerank=slow_rerank)
    elapsed = time.monotonic() - started

    assert ranked.reason == BUDGET_EXPIRED
    assert elapsed < 1.6


def test_nothing_above_the_cutoff_is_a_healthy_empty_answer(tmp_path):
    cfg = make_config(tmp_path)

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"),)),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.0,)),
        resolve=lambda name: f"/skills/{name}/SKILL.md",
    )

    assert outcome.reason == ""
    assert outcome.rows == ()


def test_a_later_request_without_a_catalog_clears_the_record(tmp_path):
    cfg = make_config(tmp_path)
    rewrite(
        _payload(),
        cfg=cfg,
        index=make_index(),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="session-1",
    )
    assert catalog_rows(cfg, "session-1") is not None

    assert rewrite({"messages": [{"role": "user", "content": QUERY}]}, cfg=cfg, session_id="session-1") is None

    assert catalog_rows(cfg, "session-1") is None


def test_a_non_chat_completions_request_clears_the_record(tmp_path):
    cfg = make_config(tmp_path)
    rewrite(
        _payload(),
        cfg=cfg,
        index=make_index(),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="session-1",
    )

    rewrite(_payload(), cfg=cfg, api_mode="anthropic_messages", session_id="session-1")

    assert catalog_rows(cfg, "session-1") is None


def test_a_second_catalog_replaces_the_first_rows(tmp_path):
    cfg = make_config(tmp_path)
    rewrite(
        _payload(),
        cfg=cfg,
        index=make_index(),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="session-1",
    )

    other = SYSTEM.replace("    - alpha: Alpha skill\n    - beta: Beta skill\n    - gamma: Gamma skill\n", "    - zeta: Zeta skill\n")
    rewrite(
        {"messages": [{"role": "system", "content": other}, {"role": "user", "content": QUERY}]},
        cfg=cfg,
        index=make_index(rows=(("zeta", "Zeta skill"),)),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9,)),
        session_id="session-1",
    )

    assert [row.name for row in catalog_rows(cfg, "session-1")] == ["zeta"]


# ---------------------------------------------------------------------------
# The pass hands back the scores the selector compared, so the decision line can carry them.
# ---------------------------------------------------------------------------

# The envelope the tool answered before this change, for the partition fixture, the default
# ``FakeEmbedder`` and ``FakeReranker`` and a resolver that answers one path per name. It was
# captured from the build at 87d3e72d8b3d35da4edf7890dc39b458203e6b62.
GOLDEN_ENVELOPE = (
    '{"skills": [{"name": "alpha", "description": "Alpha skill", "path": "/skills/alpha/SKILL.md"}, '
    '{"name": "beta", "description": "Beta skill", "path": "/skills/beta/SKILL.md"}, '
    '{"name": "gamma", "description": "Gamma skill", "path": "/skills/gamma/SKILL.md"}]}'
)


def test_the_pass_hands_back_every_score_uncapped_and_without_the_cutoff(tmp_path):
    cfg = make_config(tmp_path, cutoff=0.5, maxSuggestions=1)

    ranked = search_module.rank(
        cfg,
        make_index(),
        QUERY,
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.2, 0.9, 0.1)),
    )

    assert ranked.reason == ""
    assert ranked.candidates == (("beta", 0.2), ("alpha", 0.9), ("gamma", 0.1)), (
        "the pass does not hand back every shortlisted candidate with the score the selector saw"
    )
    assert len(ranked.candidates) == 3, "the pass applies neither the cutoff nor the cap"


def test_the_pass_times_each_stage(tmp_path):
    def slow_embed(cfg, texts):
        time.sleep(0.01)
        return FakeEmbedder()(cfg, texts)

    def slow_rerank(cfg, query, documents):
        time.sleep(0.01)
        return FakeReranker()(cfg, query, documents)

    ranked = search_module.rank(
        make_config(tmp_path), make_index(), QUERY, embed=slow_embed, rerank=slow_rerank
    )

    assert ranked.reason == ""
    assert ranked.embed_ms >= 5.0
    assert ranked.rerank_ms >= 5.0
    assert ranked.total_ms >= ranked.embed_ms + ranked.rerank_ms


def test_a_failed_pass_hands_back_no_candidate_scores(tmp_path):
    ranked = search_module.rank(
        make_config(tmp_path),
        make_index(),
        QUERY,
        embed=FakeEmbedder(),
        rerank=FakeReranker(result=None),
    )

    assert ranked.reason == RERANK_FAILED
    assert ranked.candidates == ()


def test_the_search_tool_envelope_is_byte_identical_to_the_pre_change_build(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "EMBED", FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    forget_session("envelope-session")
    remember_partition("envelope-session", partition())

    answer = tools_module.handle(
        {"query": QUERY},
        settings={"cachePath": "", "suggestionsPath": str(tmp_path / "envelope.json")},
        session_id="envelope-session",
    )

    assert answer == GOLDEN_ENVELOPE


def test_a_pathless_top_row_hands_the_last_slot_to_the_next_admissible_row(tmp_path):
    cfg = make_config(tmp_path, cutoff=0.01, maxSuggestions=2)
    rows = (
        search_module.SearchRow("alpha", "Alpha skill"),
        search_module.SearchRow("beta", "Beta skill"),
        search_module.SearchRow("gamma", "Gamma skill"),
    )

    outcome = select_rows(
        cfg,
        rows,
        QUERY,
        index=make_index(),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9, 0.8, 0.7)),
        resolve=lambda name: None if name == "beta" else f"/skills/{name}/SKILL.md",
    )

    # The shortlist is beta (the top score), alpha, gamma; beta has no path, so it never consumes a
    # slot and the two admissible rows fill the cap.
    assert [(row.name, row.path) for row in outcome.rows] == [
        ("alpha", "/skills/alpha/SKILL.md"),
        ("gamma", "/skills/gamma/SKILL.md"),
    ]


def test_a_raising_lookup_costs_only_that_candidate(tmp_path):
    cfg = make_config(tmp_path, maxSuggestions=2)

    def resolve(name):
        if name == "beta":
            raise RuntimeError("the host lookup exploded")
        return f"/skills/{name}/SKILL.md"

    outcome = select_rows(
        cfg,
        (
            search_module.SearchRow("alpha", "Alpha skill"),
            search_module.SearchRow("beta", "Beta skill"),
            search_module.SearchRow("gamma", "Gamma skill"),
        ),
        QUERY,
        index=make_index(),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9, 0.8, 0.7)),
        resolve=resolve,
    )

    assert outcome.reason == ""
    assert [row.name for row in outcome.rows] == ["alpha", "gamma"]
