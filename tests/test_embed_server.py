"""Tests for the BGE-M3 embeddings service (contract C1).

Nothing is read from disk: the state is installed with a fake session and a
fake tokenizer, so the suite passes with no model directory and no network.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from service import embed_server

DIM = 1024
MODEL_ID = "bge-m3"


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Maps a text to fixed token ids and records every call."""

    def __init__(self, ids_by_text: dict[str, list[int]]) -> None:
        self.ids_by_text = ids_by_text
        self.calls: list[str] = []

    def encode(self, text: str) -> _FakeEncoding:
        self.calls.append(text)
        return _FakeEncoding(self.ids_by_text.get(text, [0, 99, 2]))


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Answers ``token_embeddings`` for the encoded ids and records each run."""

    def __init__(self, rows_by_ids: dict[tuple[int, ...], np.ndarray]) -> None:
        self.rows_by_ids = rows_by_ids
        self.calls: list[dict[str, np.ndarray]] = []

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput("token_embeddings"), _FakeOutput("sentence_embedding")]

    def run(self, output_names, feeds):
        self.calls.append(feeds)
        ids = tuple(int(token) for token in feeds["input_ids"][0])
        rows = np.asarray(self.rows_by_ids[ids], dtype=np.float32)
        # Deliberately different from CLS, to prove this output is never read.
        sentence_embedding = np.full((1, rows.shape[-1]), 99.0, dtype=np.float32)
        return [rows[None, :, :], sentence_embedding]


def _rows(*rows: tuple[int, float]) -> np.ndarray:
    """Build a ``(len(rows), DIM)`` array, one (column, value) per row."""
    array = np.zeros((len(rows), DIM), dtype=np.float32)
    for row, (column, value) in enumerate(rows):
        array[row, column] = value
    return array


def _ready_state(
    ids_by_text: dict[str, list[int]],
    rows_by_ids: dict[tuple[int, ...], np.ndarray],
) -> embed_server.EmbeddingState:
    state = embed_server.EmbeddingState(model_id=MODEL_ID)
    state.install(_FakeSession(rows_by_ids), _FakeTokenizer(ids_by_text), DIM)
    return state


def _client(state: embed_server.EmbeddingState) -> TestClient:
    return TestClient(embed_server.create_app(state))


def test_health_reports_ok_model_and_dim() -> None:
    state = _ready_state({"hello": [0, 11, 2]}, {(0, 11, 2): _rows((0, 1.0))})
    with _client(state) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": MODEL_ID, "dim": DIM}


def test_health_reports_loading_before_the_model_is_ready() -> None:
    state = embed_server.EmbeddingState(model_id=MODEL_ID)
    with _client(state) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "loading"}


def test_embeddings_returns_503_before_the_model_is_ready() -> None:
    state = embed_server.EmbeddingState(model_id=MODEL_ID)
    with _client(state) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL_ID, "input": ["hello"]}
        )
    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"error"}
    assert isinstance(body["error"], str)


def test_one_vector_per_input_in_input_order() -> None:
    ids_by_text = {"first": [0, 11, 2], "second": [0, 22, 2], "third": [0, 33, 2]}
    rows_by_ids = {
        (0, 11, 2): _rows((0, 2.0), (1, 9.0), (2, 9.0)),
        (0, 22, 2): _rows((1, 3.0), (0, 9.0), (0, 9.0)),
        (0, 33, 2): _rows((2, 4.0), (0, 9.0), (0, 9.0)),
    }
    state = _ready_state(ids_by_text, rows_by_ids)
    with _client(state) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": MODEL_ID, "input": ["first", "second", "third"]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == MODEL_ID
    assert [entry["index"] for entry in body["data"]] == [0, 1, 2]
    vectors = [entry["embedding"] for entry in body["data"]]
    assert all(len(vector) == DIM for vector in vectors)
    for position, vector in enumerate(vectors):
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
        assert vector[position] == pytest.approx(1.0)
        assert max(vector) == pytest.approx(1.0)


def test_cls_pooling_takes_position_zero_of_token_embeddings() -> None:
    state = _ready_state(
        {"hello": [0, 11, 2]},
        {(0, 11, 2): _rows((0, 3.0), (1, 5.0), (2, 7.0))},
    )
    with _client(state) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL_ID, "input": ["hello"]}
        )
    assert response.status_code == 200
    vector = response.json()["data"][0]["embedding"]
    assert vector[0] == pytest.approx(1.0)
    assert all(abs(value) < 1e-9 for value in vector[1:])
    # ``sentence_embedding`` (all 99.0) would come out as 1/sqrt(DIM) per entry.
    assert vector[0] != pytest.approx(1.0 / np.sqrt(DIM))


def test_embeds_one_unpadded_text_at_a_time() -> None:
    ids_by_text = {"first": [0, 11, 2], "second": [0, 22, 2]}
    rows_by_ids = {
        (0, 11, 2): _rows((0, 1.0), (1, 1.0), (2, 1.0)),
        (0, 22, 2): _rows((0, 1.0), (1, 1.0), (2, 1.0)),
    }
    state = _ready_state(ids_by_text, rows_by_ids)
    with _client(state) as client:
        response = client.post(
            "/v1/embeddings", json={"model": MODEL_ID, "input": ["first", "second"]}
        )
    assert response.status_code == 200
    session = state.session
    assert [call["input_ids"].tolist() for call in session.calls] == [
        [[0, 11, 2]],
        [[0, 22, 2]],
    ]
    for call in session.calls:
        assert call["input_ids"].shape == (1, 3)
        assert call["attention_mask"].shape == (1, 3)
        assert call["input_ids"].dtype == np.int64
        assert call["attention_mask"].dtype == np.int64
        assert bool((call["attention_mask"] == 1).all())


def test_the_vector_for_one_text_is_identical_across_calls() -> None:
    state = _ready_state({"hello": [0, 11, 2]}, {(0, 11, 2): _rows((0, 3.0), (1, 4.0))})
    with _client(state) as client:
        first = client.post(
            "/v1/embeddings", json={"model": MODEL_ID, "input": ["hello"]}
        ).json()["data"][0]["embedding"]
        second = client.post(
            "/v1/embeddings", json={"model": MODEL_ID, "input": ["hello"]}
        ).json()["data"][0]["embedding"]
    assert first == second


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"model": MODEL_ID},
        {"model": MODEL_ID, "input": []},
        {"model": MODEL_ID, "input": "hello"},
        {"model": MODEL_ID, "input": [1]},
        {"model": MODEL_ID, "input": ["hello", None]},
    ],
)
def test_bad_input_is_400(payload: dict) -> None:
    state = _ready_state({"hello": [0, 11, 2]}, {(0, 11, 2): _rows((0, 1.0))})
    with _client(state) as client:
        response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}
    assert isinstance(body["error"], str)
    assert state.session.calls == []


def test_malformed_json_is_400() -> None:
    state = _ready_state({"hello": [0, 11, 2]}, {(0, 11, 2): _rows((0, 1.0))})
    with _client(state) as client:
        response = client.post(
            "/v1/embeddings",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 400
    assert set(response.json()) == {"error"}


def test_response_echoes_the_requested_model() -> None:
    state = _ready_state({"hello": [0, 11, 2]}, {(0, 11, 2): _rows((0, 1.0))})
    with _client(state) as client:
        echoed = client.post(
            "/v1/embeddings", json={"model": "other-model", "input": ["hello"]}
        ).json()
        fallback = client.post(
            "/v1/embeddings", json={"input": ["hello"]}
        ).json()
    assert echoed["model"] == "other-model"
    assert fallback["model"] == MODEL_ID


def test_settings_default_to_the_cache_layout(monkeypatch) -> None:
    for key in ("HSR_MODEL_DIR", "HSR_PORT", "HSR_MODEL_ID", "HSR_MAX_LENGTH"):
        monkeypatch.delenv(key, raising=False)
    settings = embed_server.Settings.from_env()
    expected = os.path.join(
        os.path.expanduser("~"), ".cache", "hermes-skill-router", "models", "bge-m3"
    )
    assert settings.model_dir == expected
    assert embed_server.DEFAULT_MODEL_DIR == expected
    assert embed_server.DEFAULT_MODEL_DIR.endswith(
        "/.cache/hermes-skill-router/models/bge-m3"
    )
    assert settings.port == 8012
    assert settings.model_id == "bge-m3"
    assert settings.max_length == 512


def test_settings_read_every_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("HSR_MODEL_DIR", "/models/bge-m3")
    monkeypatch.setenv("HSR_PORT", "9111")
    monkeypatch.setenv("HSR_MODEL_ID", "custom-id")
    monkeypatch.setenv("HSR_MAX_LENGTH", "64")
    settings = embed_server.Settings.from_env()
    assert settings.model_dir == "/models/bge-m3"
    assert settings.port == 9111
    assert settings.model_id == "custom-id"
    assert settings.max_length == 64
