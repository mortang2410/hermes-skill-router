"""Contracts C3 and C4: the session catalog record and the one selection pass.

The search tool ranks the routed half of the partition the middleware parsed, through the same pass
a turn runs. These tests drive that pass directly with a fake embedder and reranker, so they need no
endpoint, and they compare it with the names a real turn injects for the same payload and index.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows, Partition  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
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
    )

    assert [row.name for row in outcome.rows] == turn_names == ["alpha", "beta", "gamma"]


def test_select_rows_resolves_the_path_of_selected_rows_only(tmp_path):
    cfg = make_config(tmp_path)
    looked_up = []

    def resolve(name):
        looked_up.append(name)
        return f"/skills/{name}/SKILL.md"

    outcome = select_rows(
        cfg,
        [search_module.SearchRow("alpha", "Alpha skill"), search_module.SearchRow("delta", "")],
        QUERY,
        index=make_index(rows=(("alpha", "Alpha skill"), ("delta", ""))),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.5, 0.4)),
        resolve=resolve,
    )

    assert [(row.name, row.path) for row in outcome.rows] == [
        ("alpha", "/skills/alpha/SKILL.md"),
        ("delta", "/skills/delta/SKILL.md"),
    ]
    assert looked_up == ["alpha", "delta"]


def test_a_refused_lookup_keeps_the_row_without_a_path(tmp_path):
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

    assert [(row.name, row.path) for row in outcome.rows] == [("alpha", None)]


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
