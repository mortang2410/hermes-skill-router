"""D12 and ticket T5: one routing budget bounds one pass, and it starts after the index build.

Two properties are pinned here, both through the real rewrite with the real HTTP clients:

* a stub endpoint that answers past the budget ends the pass at the budget: the request keeps its
  catalog and gets no block, exactly one warning names the budget, and no rerank request is sent;
* a catalog build slower than the budget does not trip it, because the budget starts once the index
  is in hand, and the same request still routes.

The budget is loaded from the plugin config, and the endpoints are loopback fakes from
``tests/fake_endpoint.py``, so the pass needs no model and no network.
"""

from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint, Response  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402

#: The budget both tests run at. The schema's lower bound is 1000, and a value no other test uses
#: keeps the router's log-once key distinct, so "exactly one line" is decided by this pass alone.
BUDGET_MS = 1100

QUERY = "draft a git commit message for a bug fix"

SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Before replying, scan the skills below.\n\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)

#: The document text ``library.document_text`` embeds for each routed row.
VECTORS = {"alpha\nAlpha skill": [1.0, 0.0], "beta\nBeta skill": [0.0, 1.0]}
#: The query points at beta, so beta is the row both stages see first.
QUERY_VECTOR = [0.0, 1.0]


def chat_payload():
    return {
        "model": "test-model",
        "tools": [{"type": "function", "function": {"name": "skill_view"}}],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUERY},
        ],
    }


def embeddings_answer(vectors):
    """Return an OpenAI-shaped embeddings body carrying *vectors*, in request order."""
    return {
        "model": "bge-m3",
        "data": [
            {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
        ],
    }


def rerank_answer(documents):
    """Return a Cohere-shaped rerank body: beta above the cutoff, alpha below it."""
    return {
        "results": [
            {"index": index, "relevance_score": 0.9 if "Beta skill" in document else 0.2}
            for index, document in enumerate(documents)
        ]
    }


@pytest.fixture()
def budget_reset():
    """Give each test a clean endpoint health state, as a fresh process would have."""
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()
    yield
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()


def settings(endpoints, suggestions_path):
    """Plugin settings that point the pass at the two fakes and the budget under test."""
    embeddings, reranking = endpoints
    return load(
        {
            "embedBaseUrl": embeddings.base_url,
            "rerankBaseUrl": reranking.base_url,
            "cachePath": "",
            "suggestionsPath": str(suggestions_path),
            "healthCheckMs": 0,
            "cutoff": 0.5,
            "routingTimeoutMs": BUDGET_MS,
            # Both per-request timeouts are far larger than the budget, so only the budget can
            # end the pass. Without the cap the pass would wait the whole 5 s stub delay.
            "embedTimeoutMs": 60000,
            "rerankTimeoutMs": 60000,
        }
    )


def built_index():
    """The routed index, supplied so the first test never builds one."""
    return Index(
        names=["alpha", "beta"],
        descriptions=["Alpha skill", "Beta skill"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        dim=2,
        fingerprint="test",
    )


def budget_warnings(caplog):
    """Every router warning line that names the routing budget."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "hermes_skill_router.router"
        and record.levelno >= logging.WARNING
        and "routing budget" in record.getMessage()
    ]


def test_a_stub_endpoint_that_answers_past_the_budget_falls_back_with_one_budget_line(
    tmp_path, caplog, budget_reset
):
    """The pass ends at the budget: bare catalog, one line naming it, and no rerank request."""
    caplog.set_level(logging.WARNING, logger="hermes_skill_router.router")
    embeddings = FakeEndpoint()
    reranking = FakeEndpoint()
    try:
        embeddings.answer(embeddings_answer([QUERY_VECTOR]), delay=5.0)
        cfg = settings((embeddings, reranking), tmp_path / "suggestions.json")
        payload = chat_payload()
        arrived = copy.deepcopy(payload)

        started = time.monotonic()
        result = rewrite(payload, cfg=cfg, index=built_index())
        elapsed = time.monotonic() - started

        assert result is None, 'a pass that cannot finish must leave the request exactly as it arrived'
        assert payload == arrived, 'the rewrite must not mutate the payload it abandons'
        assert payload["messages"][0]["content"] == SYSTEM, 'the catalog must still be on the request'

        lines = budget_warnings(caplog)
        assert len(lines) == 1, f'exactly one line must name the budget; saw {lines}'
        assert str(BUDGET_MS) in lines[0], f'the line must name the budget of {BUDGET_MS} ms: {lines[0]}'

        assert embeddings.count == 1, 'the pass ends inside the query embed, so it sends exactly one request'
        assert reranking.count == 0, 'a pass that ends at its budget must never reach the rerank stage'
        # The stub holds the answer for 5 s, so finishing well before that is the budget at work.
        assert elapsed < 4.0, f'the budget must end the pass long before the 5 s stub answer; took {elapsed:.2f} s'
    finally:
        embeddings.close()
        reranking.close()


def test_a_catalog_build_slower_than_the_budget_does_not_trip_it(tmp_path, caplog, budget_reset):
    """The build runs for longer than the budget and the same pass still routes afterwards."""
    caplog.set_level(logging.WARNING, logger="hermes_skill_router.router")

    def embed_responder(request):
        texts = request.json()["input"]
        if len(texts) > 1:
            # The index build: one batch for both routed rows, held past the whole budget.
            return Response(body=embeddings_answer([VECTORS[text] for text in texts]), delay=1.5)
        return Response(body=embeddings_answer([QUERY_VECTOR for _text in texts]))

    def rerank_responder(request):
        return Response(body=rerank_answer(request.json()["documents"]))

    embeddings = FakeEndpoint(embed_responder)
    reranking = FakeEndpoint(rerank_responder)
    try:
        cfg = settings((embeddings, reranking), tmp_path / "suggestions.json")
        payload = chat_payload()

        started = time.monotonic()
        # No index supplied, so this call builds one, and the build alone outlasts the budget.
        result = rewrite(payload, cfg=cfg)
        elapsed = time.monotonic() - started

        assert elapsed > BUDGET_MS / 1000.0, (
            f'precondition: the build must outlast the {BUDGET_MS} ms budget; it took {elapsed:.2f} s'
        )
        assert result is not None, 'the budget starts after the build, so a slow build must not end the pass'
        block = result["request"]["messages"][-1]
        assert block["role"] == "system"
        assert "Beta skill" in block["content"], 'the pass must still route its suggestion'
        assert [record.getMessage() for record in caplog.records if "routing budget" in record.getMessage()] == []
        assert reranking.count == 1, 'the pass reached the rerank stage after the slow build'
    finally:
        embeddings.close()
        reranking.close()
