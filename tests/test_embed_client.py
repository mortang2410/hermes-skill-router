"""Contract C5: the embeddings client's request shape, refusals and health window.

Every test drives the loopback fake from ``tests/fake_endpoint.py``, so no model, GPU or network is
needed and the exact bytes that would leave the process are observable.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import BIND_HOST, FakeEndpoint  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.embed import embed_texts  # noqa: E402


@pytest.fixture()
def fake():
    # Health state is process state keyed by URL; start and end every test with none of it, so a
    # reused ephemeral port can never carry a window from another test.
    embed_module.HEALTH.reset()
    endpoint = FakeEndpoint()
    try:
        yield endpoint
    finally:
        endpoint.close()
        embed_module.HEALTH.reset()


def make_config(fake, **overrides):
    """A config pointed at *fake*, with the endpoint identity fixed for the assertions below."""
    settings = {
        "embedBaseUrl": fake.base_url,
        "embedPath": "/v1/embeddings",
        "embedModelId": "bge-m3",
        "embedTimeoutMs": 2000,
        "healthCheckMs": 60000,
    }
    settings.update(overrides)
    return load(settings)


def embeddings_body(vectors, model="bge-m3"):
    """An OpenAI-shaped embeddings response carrying *vectors* in order."""
    return {
        "model": model,
        "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
    }


def test_sends_one_openai_shaped_request_and_returns_its_vectors(fake):
    fake.answer(body=embeddings_body([[1.0, 0.0], [0.0, 1.0]]))

    vectors = embed_texts(make_config(fake), ["alpha\nAlpha skill", "beta\nBeta skill"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert fake.count == 1
    request = fake.requests[0]
    assert request.method == "POST"
    assert request.path == "/v1/embeddings"
    assert request.header("Content-Type") == "application/json"
    assert request.json() == {
        "model": "bge-m3",
        "input": ["alpha\nAlpha skill", "beta\nBeta skill"],
    }


def test_the_endpoint_path_is_appended_to_the_base_url(fake):
    fake.answer(body=embeddings_body([[1.0]]))

    assert embed_texts(make_config(fake, embedPath="/other/path"), ["a"]) == [[1.0]]

    assert fake.requests[0].path == "/other/path"


def test_empty_texts_open_no_socket(fake):
    assert embed_texts(make_config(fake), []) == []
    assert fake.count == 0


def test_an_http_error_returns_none(fake):
    fake.answer(status=500, body={"error": "boom"})

    assert embed_texts(make_config(fake), ["a"]) is None
    assert fake.count == 1


def test_a_timeout_returns_none(fake):
    fake.answer(body=embeddings_body([[1.0]]), delay=0.5)

    assert embed_texts(make_config(fake, embedTimeoutMs=100), ["a"]) is None
    assert fake.count == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_to_the_same_endpoint_is_refused(fake, status):
    fake.answer(status=status, headers={"Location": fake.base_url + "/v1/embeddings"})

    assert embed_texts(make_config(fake), ["alpha\nAlpha skill"]) is None
    assert fake.count == 1, "the client followed the redirect instead of refusing it"


def test_a_redirect_to_another_host_is_refused_without_reaching_it(fake):
    other = FakeEndpoint()
    try:
        other.answer(body=embeddings_body([[1.0]]))
        fake.answer(status=307, headers={"Location": other.base_url + "/v1/embeddings"})

        assert embed_texts(make_config(fake), ["alpha\nAlpha skill"]) is None

        assert fake.count == 1
        assert other.count == 0, "the client sent the text to the redirect target"
    finally:
        other.close()


def test_the_shared_opener_refuses_redirects_by_itself(fake):
    """The rerank client (T4) reuses this opener, so its policy must hold without embed_texts."""
    fake.answer(status=302, headers={"Location": fake.base_url + "/next"})
    request = urllib.request.Request(
        fake.base_url + "/v1/embeddings",
        data=json.dumps({"input": ["a"]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError):
        embed_module.opener().open(request, timeout=5)

    assert fake.count == 1
    assert embed_module.opener() is embed_module.opener()


@pytest.mark.parametrize(
    "body",
    [
        {"model": "bge-m3"},
        {"data": []},
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 1}]},
        {"data": [{"index": 0, "embedding": []}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0, 2.0]}, {"index": 1, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [1.0, 2.0]}, "not an object"]},
        [1, 2],
    ],
)
def test_an_unusable_body_returns_none(fake, body):
    fake.answer(body=body)

    assert embed_texts(make_config(fake), ["a", "b"]) is None


def test_a_body_that_is_not_json_returns_none(fake):
    fake.answer(body="<html>bad gateway</html>")

    assert embed_texts(make_config(fake), ["a"]) is None


def test_health_window_skips_the_endpoint_after_a_failure(monkeypatch, fake):
    clock = [1000.0]
    monkeypatch.setattr(embed_module, "monotonic", lambda: clock[0])
    cfg = make_config(fake)
    fake.answer(status=500, body={"error": "boom"})
    assert embed_texts(cfg, ["a"]) is None
    assert fake.count == 1

    fake.answer(body=embeddings_body([[1.0]]))
    assert embed_texts(cfg, ["a"]) is None, "a call inside the window must fail without a socket"
    assert fake.count == 1

    clock[0] += 60.0
    assert embed_texts(cfg, ["a"]) == [[1.0]], "the first call after the window must probe again"
    assert fake.count == 2


def test_a_success_clears_the_health_flag(monkeypatch, fake):
    clock = [1000.0]
    monkeypatch.setattr(embed_module, "monotonic", lambda: clock[0])
    cfg = make_config(fake)
    fake.answer(status=500, body={"error": "boom"})
    assert embed_texts(cfg, ["a"]) is None

    fake.answer(body=embeddings_body([[1.0]]))
    clock[0] += 60.0
    assert embed_texts(cfg, ["a"]) == [[1.0]]
    # No window applies any more, so the next call goes out immediately.
    assert embed_texts(cfg, ["a"]) == [[1.0]]
    assert fake.count == 3


def test_a_health_window_of_zero_probes_every_call(fake):
    cfg = make_config(fake, healthCheckMs=0)
    fake.answer(status=500, body={"error": "boom"})

    assert embed_texts(cfg, ["a"]) is None
    assert embed_texts(cfg, ["a"]) is None
    assert fake.count == 2


def test_health_is_kept_per_endpoint(fake):
    other = FakeEndpoint()
    try:
        other.answer(body=embeddings_body([[1.0]]))
        fake.answer(status=500, body={"error": "boom"})
        assert embed_texts(make_config(fake), ["a"]) is None
        assert fake.count == 1

        assert embed_texts(make_config(other), ["a"]) == [[1.0]]
        assert other.count == 1

        # The path is part of the endpoint identity too: a different path is a different target.
        fake.answer(body=embeddings_body([[1.0]]))
        assert embed_texts(make_config(fake, embedPath="/other/path"), ["a"]) == [[1.0]]
        assert fake.count == 2
    finally:
        other.close()


def test_a_malformed_body_also_opens_the_health_window(monkeypatch, fake):
    clock = [1000.0]
    monkeypatch.setattr(embed_module, "monotonic", lambda: clock[0])
    cfg = make_config(fake)
    fake.answer(body={"model": "bge-m3"})
    assert embed_texts(cfg, ["a"]) is None
    assert fake.count == 1

    fake.answer(body=embeddings_body([[1.0]]))
    assert embed_texts(cfg, ["a"]) is None
    assert fake.count == 1


def test_the_fake_binds_loopback_only(fake):
    assert fake.host == BIND_HOST == "127.0.0.1"
    assert fake.port > 0
    assert fake.base_url.startswith("http://127.0.0.1:")
