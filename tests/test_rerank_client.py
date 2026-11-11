"""Contract C6: the rerank client's request shape, refusals and health window.

Every test drives the loopback fake from ``tests/fake_endpoint.py``, so no model, GPU or network is
needed and the exact bytes that would leave the process are observable. The request body is asserted
against the shared document text from ``config.DOCUMENT_PREFIX``, because the same 8010 endpoint
serves another client and this plugin and parses that prefix to recover the skill name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402
from hermes_skill_router.config import DOCUMENT_PREFIX, load  # noqa: E402
from hermes_skill_router.rerank import rerank  # noqa: E402

#: The shared skill prefix, copied here from ``judgment/jev-client.ts`` and
#: ``/home/wilder/.scripts/decider-rerank-server.py`` (``SKILL_PREFIX``) as a fixed expected string.
REFERENCE_SKILL_PREFIX = (
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
    "actually need to carry out the current task correctly. Skill: "
)


@pytest.fixture()
def fake():
    # The health flag is process state keyed by URL; start and end every test with none of it, so a
    # reused ephemeral port can never carry a window from another test.
    rerank_module.HEALTH.reset()
    endpoint = FakeEndpoint()
    try:
        yield endpoint
    finally:
        endpoint.close()
        rerank_module.HEALTH.reset()


def make_config(fake, **overrides):
    """A config pointed at *fake*, with the endpoint identity fixed for the assertions below."""
    settings = {
        "rerankBaseUrl": fake.base_url,
        "rerankPath": "/v1/rerank",
        "rerankModel": "tasksource-decider-nano",
        "rerankTimeoutMs": 20000,
        "healthCheckMs": 60000,
    }
    settings.update(overrides)
    return load(settings)


def rerank_body(scores, *, order=None, model="tasksource-decider-nano"):
    """A Cohere-shaped rerank response carrying *scores* in *order* (default input order)."""
    indices = list(range(len(scores))) if order is None else list(order)
    return {
        "results": [
            {"index": index, "relevance_score": scores[index]} for index in indices
        ],
        "model": model,
    }


def test_sends_one_cohere_shaped_request_and_returns_scores_in_input_order(fake):
    fake.answer(body=rerank_body([0.9, 0.2, 0.7], order=[2, 0, 1]))

    scores = rerank(make_config(fake), "run the tests", ["doc a", "doc b", "doc c"])

    assert scores == [0.9, 0.2, 0.7]
    assert all(isinstance(score, float) for score in scores)
    assert fake.count == 1
    request = fake.requests[0]
    assert request.method == "POST"
    assert request.path == "/v1/rerank"
    assert request.header("Content-Type") == "application/json"
    assert request.json() == {
        "model": "tasksource-decider-nano",
        "query": "run the tests",
        "documents": ["doc a", "doc b", "doc c"],
        "top_n": 3,
    }


def test_the_outgoing_documents_carry_the_reference_prefix_and_name_dash_description(fake):
    fake.answer(body=rerank_body([0.4, 0.6]))
    cfg = make_config(fake)
    documents = [
        cfg.document_prefix + "alpha" + " - " + "Alpha skill",
        cfg.document_prefix + "beta" + " - " + "Beta skill",
    ]

    assert rerank(cfg, "fix the failing test", documents) == [0.4, 0.6]

    # The default prefix is the reference sentence, so the endpoint recovers the skill names from both
    # clients the same way; the body is compared to one fixed expected string.
    assert DOCUMENT_PREFIX == REFERENCE_SKILL_PREFIX
    assert fake.requests[0].json() == {
        "model": "tasksource-decider-nano",
        "query": "fix the failing test",
        "documents": [
            REFERENCE_SKILL_PREFIX + "alpha - Alpha skill",
            REFERENCE_SKILL_PREFIX + "beta - Beta skill",
        ],
        "top_n": 2,
    }


def test_the_top_n_is_the_document_count(fake):
    fake.answer(body=rerank_body([0.1] * 4))

    rerank(make_config(fake), "q", ["a", "b", "c", "d"])

    assert fake.requests[0].json()["top_n"] == 4


def test_the_endpoint_path_is_appended_to_the_base_url(fake):
    fake.answer(body=rerank_body([0.5]))

    assert rerank(make_config(fake, rerankPath="/other/rerank"), "q", ["a"]) == [0.5]

    assert fake.requests[0].path == "/other/rerank"


def test_empty_documents_open_no_socket(fake):
    assert rerank(make_config(fake), "q", []) == []
    assert fake.count == 0


def test_scores_are_returned_unchanged_without_a_second_normalisation(fake):
    fake.answer(body=rerank_body([0.2, 0.6]))

    assert rerank(make_config(fake), "q", ["a", "b"]) == [0.2, 0.6]


def test_an_http_error_returns_none(fake):
    fake.answer(status=500, body={"error": "boom"})

    assert rerank(make_config(fake), "q", ["a"]) is None
    assert fake.count == 1


def test_a_timeout_returns_none(fake):
    fake.answer(body=rerank_body([0.5]), delay=0.5)

    assert rerank(make_config(fake, rerankTimeoutMs=100), "q", ["a"]) is None
    assert fake.count == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_to_the_same_endpoint_is_refused(fake, status):
    fake.answer(status=status, headers={"Location": fake.base_url + "/v1/rerank"})

    assert rerank(make_config(fake), "semantic query", ["skill document"]) is None
    assert fake.count == 1, "the client followed the redirect instead of refusing it"


@pytest.mark.parametrize("status", [302, 307])
def test_a_redirect_to_another_host_sends_nothing_there(fake, status):
    other = FakeEndpoint()
    try:
        other.answer(body=rerank_body([0.5]))
        fake.answer(status=status, headers={"Location": other.base_url + "/v1/rerank"})

        assert rerank(make_config(fake), "semantic query", ["skill document"]) is None

        assert fake.count == 1
        assert other.count == 0, "the client sent the query and the skill names to the target"
    finally:
        other.close()


def test_the_client_reuses_the_embeddings_opener():
    """One redirect policy for both clients: no second opener is built here (D4)."""
    assert rerank_module.opener is embed_module.opener
    assert rerank_module.opener() is embed_module.opener()


@pytest.mark.parametrize(
    "body",
    [
        {"model": "tasksource-decider-nano"},
        {"results": []},
        {"results": 5},
        {"results": [{"index": 0, "relevance_score": 0.5}]},
        {
            "results": [
                {"index": 0, "relevance_score": 0.5},
                {"index": 1, "relevance_score": 0.4},
                {"index": 2, "relevance_score": 0.3},
            ]
        },
        {"results": [{"index": 2, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.4}]},
        {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]},
        {"results": [{"index": "0", "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.4}]},
        {"results": [{"index": True, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.4}]},
        {"results": [{"index": 0, "relevance_score": "0.5"}, {"index": 1, "relevance_score": 0.4}]},
        {"results": [{"index": 0, "relevance_score": None}, {"index": 1, "relevance_score": 0.4}]},
        {"results": [{"index": 0}, {"index": 1, "relevance_score": 0.4}]},
        {
            "results": [
                {"index": 0, "relevance_score": float("nan")},
                {"index": 1, "relevance_score": 0.4},
            ]
        },
        {
            "results": [
                {"index": 0, "relevance_score": float("inf")},
                {"index": 1, "relevance_score": 0.4},
            ]
        },
        {"results": ["not an object", {"index": 1, "relevance_score": 0.4}]},
        [1, 2],
    ],
)
def test_an_unusable_body_returns_none(fake, body):
    """A short, long, duplicated or malformed result set fails; scores are never padded."""
    fake.answer(body=body)

    assert rerank(make_config(fake), "q", ["a", "b"]) is None


def test_a_body_that_is_not_json_returns_none(fake):
    fake.answer(body="<html>bad gateway</html>")

    assert rerank(make_config(fake), "q", ["a"]) is None


def test_health_window_skips_the_endpoint_after_a_failure(monkeypatch, fake):
    clock = [1000.0]
    monkeypatch.setattr(embed_module, "monotonic", lambda: clock[0])
    cfg = make_config(fake)
    fake.answer(status=500, body={"error": "boom"})
    assert rerank(cfg, "q", ["a"]) is None
    assert fake.count == 1

    fake.answer(body=rerank_body([0.5]))
    assert rerank(cfg, "q", ["a"]) is None, "a call inside the window must fail without a socket"
    assert fake.count == 1

    clock[0] += 60.0
    assert rerank(cfg, "q", ["a"]) == [0.5], "the first call after the window must probe again"
    assert fake.count == 2


def test_a_success_clears_the_health_flag(monkeypatch, fake):
    clock = [1000.0]
    monkeypatch.setattr(embed_module, "monotonic", lambda: clock[0])
    cfg = make_config(fake)
    fake.answer(status=500, body={"error": "boom"})
    assert rerank(cfg, "q", ["a"]) is None

    fake.answer(body=rerank_body([0.5]))
    clock[0] += 60.0
    assert rerank(cfg, "q", ["a"]) == [0.5]
    # No window applies any more, so the next call goes out immediately.
    assert rerank(cfg, "q", ["a"]) == [0.5]
    assert fake.count == 3


def test_a_health_window_of_zero_probes_every_call(fake):
    cfg = make_config(fake, healthCheckMs=0)
    fake.answer(status=500, body={"error": "boom"})

    assert rerank(cfg, "q", ["a"]) is None
    assert rerank(cfg, "q", ["a"]) is None
    assert fake.count == 2


def test_health_is_kept_per_endpoint(fake):
    other = FakeEndpoint()
    try:
        other.answer(body=rerank_body([0.5]))
        fake.answer(status=500, body={"error": "boom"})
        assert rerank(make_config(fake), "q", ["a"]) is None
        assert fake.count == 1

        assert rerank(make_config(other), "q", ["a"]) == [0.5]
        assert other.count == 1

        # The path is part of the endpoint identity too: a different path is a different target.
        fake.answer(body=rerank_body([0.5]))
        assert rerank(make_config(fake), "q", ["a"]) is None
        assert rerank(make_config(fake, rerankPath="/other/rerank"), "q", ["a"]) == [0.5]
        assert fake.count == 2
    finally:
        other.close()
