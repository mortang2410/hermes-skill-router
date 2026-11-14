"""Contract C3: the bundled decider rerank server's wire and fresh-checkout behaviour.

The decider model is never loaded here. Every test installs a fake decider into the state,
starts the real ``http.server`` on an ephemeral loopback port and drives it over HTTP, so the
wire the router uses is exercised without torch, the 298 MB weights or the network.

The fresh-checkout tests prove the script ships and stays importable on a host that has not
installed the model stack, because the setup script starts it only after the download.
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service import decider_rerank_server  # noqa: E402

MODEL_ID = "tasksource-decider-nano"


class FakeDecider:
    """Stand-in for ``decider.Decider``: records each call and answers fixed probabilities."""

    def __init__(
        self,
        probabilities: Optional[Dict[str, float]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self.probabilities = dict(probabilities or {})
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def answer(self, *, state: Dict[str, Any], questions: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        if self.error is not None:
            raise self.error
        return {"answers": {"skills": {"probabilities": dict(self.probabilities)}}}


def _request(
    address: Tuple[str, int],
    method: str,
    path: str,
    payload: Optional[Any] = None,
    raw: Optional[bytes] = None,
) -> Tuple[int, Optional[Any]]:
    """Send one request over a real socket and return ``(status, decoded JSON body)``."""
    if raw is not None:
        body = raw
    elif payload is not None:
        body = json.dumps(payload).encode()
    else:
        body = None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection = http.client.HTTPConnection(address[0], address[1], timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        return response.status, (json.loads(data) if data else None)
    finally:
        connection.close()


@contextmanager
def _serve(state: decider_rerank_server.DeciderState) -> Iterator[Tuple[str, int]]:
    """Run the real server on an ephemeral loopback port for the duration of the block."""
    settings = decider_rerank_server.Settings(
        model_dir="/models/tasksource-decider-nano", threads=8, host="127.0.0.1", port=0
    )
    server = decider_rerank_server.create_server(settings, state)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        name="decider-rerank-test",
        daemon=True,
    )
    thread.start()
    try:
        yield str(server.server_address[0]), int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _ready_state(probabilities: Optional[Dict[str, float]] = None) -> decider_rerank_server.DeciderState:
    state = decider_rerank_server.DeciderState(threads=8)
    state.install(FakeDecider(probabilities))
    return state


def _skill(name: str, description: str) -> str:
    return f"{decider_rerank_server.SKILL_PREFIX}{name} - {description}"


def _tool(name: str, description: str) -> str:
    return f"{decider_rerank_server.TOOL_PREFIX}{name} - {description}"


# -- the wire the router uses ---------------------------------------------------------------


def test_health_reports_ok_with_the_model_id_and_thread_count() -> None:
    with _serve(_ready_state()) as address:
        status, body = _request(address, "GET", "/health")
    assert status == 200
    assert body == {"status": "ok", "model": MODEL_ID, "device": "cpu", "threads": 8}


def test_the_root_path_answers_health_too() -> None:
    with _serve(_ready_state()) as address:
        status, body = _request(address, "GET", "/")
    assert status == 200
    assert body == {"status": "ok", "model": MODEL_ID, "device": "cpu", "threads": 8}


def test_health_reports_loading_before_the_model_is_installed() -> None:
    with _serve(decider_rerank_server.DeciderState(threads=8)) as address:
        status, body = _request(address, "GET", "/health")
    assert status == 200
    assert body == {"status": "loading"}


def test_an_unknown_get_path_is_404() -> None:
    with _serve(_ready_state()) as address:
        status, body = _request(address, "GET", "/nope")
    assert status == 404
    assert body == {"error": "not found"}


def test_an_unknown_post_path_is_404() -> None:
    with _serve(_ready_state()) as address:
        status, body = _request(address, "POST", "/v1/embeddings", {"input": ["text"]})
    assert status == 404
    assert body == {"error": "not found"}


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank", "/v2/rerank"])
def test_every_rerank_path_alias_answers_the_cohere_shape(path: str) -> None:
    probabilities = {"alpha": 0.75, "beta": 0.25}
    with _serve(_ready_state(probabilities)) as address:
        status, body = _request(
            address,
            "POST",
            path,
            {
                "model": MODEL_ID,
                "query": "which skill",
                "documents": [_skill("alpha", "a"), _skill("beta", "b")],
                "top_n": 1,
            },
        )
    assert status == 200
    assert body == {
        "results": [
            {"index": 0, "relevance_score": 0.75},
            {"index": 1, "relevance_score": 0.25},
        ],
        "model": MODEL_ID,
    }


def test_every_document_gets_a_score_in_input_index_order() -> None:
    probabilities = {"alpha": 0.1, "beta": 0.6, "gamma": 0.3}
    documents = [_skill("alpha", "a"), _skill("beta", "b"), _skill("gamma", "c")]
    with _serve(_ready_state(probabilities)) as address:
        status, body = _request(address, "POST", "/v1/rerank", {"query": "q", "documents": documents})
    assert status == 200
    assert [entry["index"] for entry in body["results"]] == [0, 1, 2]
    assert [entry["relevance_score"] for entry in body["results"]] == [0.1, 0.6, 0.3]
    assert set(body) == {"results", "model"}
    assert all(set(entry) == {"index", "relevance_score"} for entry in body["results"])


def test_a_score_the_decider_does_not_return_is_zero() -> None:
    with _serve(_ready_state({"alpha": 0.5})) as address:
        status, body = _request(
            address, "POST", "/v1/rerank", {"query": "q", "documents": [_skill("alpha", "a"), _skill("beta", "b")]}
        )
    assert status == 200
    assert [entry["relevance_score"] for entry in body["results"]] == [0.5, 0.0]


# -- the request handed to the decider ------------------------------------------------------


def test_skill_documents_use_the_skill_instruction_and_the_query() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"alpha": 0.8, "beta": 0.2})
    state.install(fake)
    documents = [_skill("alpha", "alpha text"), _skill("beta", "beta text")]
    with _serve(state) as address:
        status, _body = _request(address, "POST", "/v1/rerank", {"query": "which skill", "documents": documents})
    assert status == 200
    assert fake.calls == [
        {
            "state": {"current_request": "which skill", "recent_requests": []},
            "questions": {
                "skills": {
                    "type": "choice",
                    "instructions": decider_rerank_server.SKILL_INSTRUCTION,
                    "criteria": {"alpha": "alpha text", "beta": "beta text"},
                }
            },
        }
    ]


def test_all_tool_documents_use_the_tool_instruction() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"gamma": 1.0})
    state.install(fake)
    documents = [_tool("gamma", "a tool"), _tool("delta", "another tool")]
    with _serve(state) as address:
        status, _body = _request(address, "POST", "/v1/rerank", {"query": "q", "documents": documents})
    assert status == 200
    questions = fake.calls[0]["questions"]["skills"]
    assert questions["instructions"] == decider_rerank_server.TOOL_INSTRUCTION
    assert questions["criteria"] == {"gamma": "a tool", "delta": "another tool"}


def test_mixed_documents_fall_back_to_the_skill_instruction() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"alpha": 0.4, "gamma": 0.6})
    state.install(fake)
    with _serve(state) as address:
        status, _body = _request(
            address, "POST", "/v1/rerank", {"query": "q", "documents": [_skill("alpha", "a"), _tool("gamma", "g")]}
        )
    assert status == 200
    assert fake.calls[0]["questions"]["skills"]["instructions"] == decider_rerank_server.SKILL_INSTRUCTION


def test_duplicate_skill_names_keep_separate_scores() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"same": 0.7, "same#1": 0.2})
    state.install(fake)
    with _serve(state) as address:
        status, body = _request(
            address, "POST", "/v1/rerank", {"query": "q", "documents": [_skill("same", "one"), _skill("same", "two")]}
        )
    assert status == 200
    assert fake.calls[0]["questions"]["skills"]["criteria"] == {"same": "one", "same#1": "two"}
    assert [entry["relevance_score"] for entry in body["results"]] == [0.7, 0.2]


def test_a_name_without_a_separator_uses_its_whole_body() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"lonely": 0.9})
    state.install(fake)
    with _serve(state) as address:
        status, _body = _request(
            address, "POST", "/v1/rerank", {"query": "q", "documents": [f"{decider_rerank_server.SKILL_PREFIX}lonely"]}
        )
    assert status == 200
    assert fake.calls[0]["questions"]["skills"]["criteria"] == {"lonely": "lonely"}


def test_an_unrecognised_document_is_scored_as_a_skill_by_position() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"candidate_0": 0.3})
    state.install(fake)
    with _serve(state) as address:
        status, _body = _request(address, "POST", "/v1/rerank", {"query": "q", "documents": ["plain text"]})
    assert status == 200
    questions = fake.calls[0]["questions"]["skills"]
    assert questions["instructions"] == decider_rerank_server.SKILL_INSTRUCTION
    assert questions["criteria"] == {"candidate_0": "plain text"}


# -- refusals and failures ------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "q"},
        {"query": "q", "documents": []},
        {"query": 5, "documents": ["doc"]},
        {"query": "q", "documents": "doc"},
        {"query": "q", "documents": {"a": "b"}},
    ],
    ids=["empty", "no-documents", "empty-documents", "query-not-text", "documents-not-list", "documents-object"],
)
def test_bad_input_is_400_without_calling_the_decider(payload: Dict[str, Any]) -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"alpha": 1.0})
    state.install(fake)
    with _serve(state) as address:
        status, body = _request(address, "POST", "/v1/rerank", payload)
    assert status == 400
    assert set(body) == {"error"}
    assert isinstance(body["error"], str)
    assert fake.calls == []


def test_an_absent_query_is_the_empty_string() -> None:
    with _serve(_ready_state({"alpha": 1.0})) as address:
        status, _body = _request(address, "POST", "/v1/rerank", {"documents": [_skill("alpha", "a")]})
    assert status == 200


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"documents": "not a list"}'])
def test_a_bad_or_malformed_body_is_400(raw: bytes) -> None:
    with _serve(_ready_state({"alpha": 1.0})) as address:
        status, body = _request(address, "POST", "/v1/rerank", raw=raw)
    assert status == 400
    assert set(body) == {"error"}


def test_a_decider_error_is_500_and_the_service_keeps_serving() -> None:
    state = decider_rerank_server.DeciderState(threads=8)
    fake = FakeDecider({"alpha": 1.0})
    state.install(fake)
    fake.error = RuntimeError("model exploded")
    with _serve(state) as address:
        status, body = _request(address, "POST", "/v1/rerank", {"query": "q", "documents": [_skill("alpha", "a")]})
        assert status == 500
        assert body == {"error": "RuntimeError: model exploded"}
        fake.error = None
        health_status, _health = _request(address, "GET", "/health")
        assert health_status == 200
        retry_status, retry_body = _request(address, "POST", "/v1/rerank", {"query": "q", "documents": [_skill("alpha", "a")]})
    assert retry_status == 200
    assert retry_body["results"] == [{"index": 0, "relevance_score": 1.0}]


# -- settings and the default model directory ------------------------------------------------


def test_settings_default_to_the_documented_values() -> None:
    settings = decider_rerank_server.Settings.from_env({})
    assert settings.model_dir == decider_rerank_server.DEFAULT_MODEL_DIR
    assert settings.threads == 8
    assert settings.host == "127.0.0.1"
    assert settings.port == 8010


def test_settings_read_every_environment_variable() -> None:
    settings = decider_rerank_server.Settings.from_env(
        {
            "DECIDER_DIR": "/models/confirmed",
            "DECIDER_THREADS": "3",
            "DECIDER_HOST": "127.0.0.2",
            "DECIDER_PORT": "9111",
        }
    )
    assert settings.model_dir == "/models/confirmed"
    assert settings.threads == 3
    assert settings.host == "127.0.0.2"
    assert settings.port == 9111


@pytest.mark.parametrize("name", ["DECIDER_THREADS", "DECIDER_PORT"])
@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_a_bad_count_is_refused(name: str, value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        decider_rerank_server.Settings.from_env({name: value})


@pytest.mark.parametrize("name", ["DECIDER_THREADS", "DECIDER_PORT"])
def test_a_blank_count_override_uses_the_default(name: str) -> None:
    settings = decider_rerank_server.Settings.from_env({name: " "})
    assert settings.threads == 8
    assert settings.port == 8010


def test_the_default_model_directory_is_the_skill_router_cache() -> None:
    expected = Path.home() / ".cache" / "hermes-skill-router" / "models" / MODEL_ID
    assert Path(decider_rerank_server.DEFAULT_MODEL_DIR) == expected


def test_load_model_names_a_missing_model_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="model directory not found"):
        decider_rerank_server.load_model(
            decider_rerank_server.Settings(model_dir=str(tmp_path / "absent"), threads=8)
        )


# -- a fresh checkout ------------------------------------------------------------------------


def test_the_vendored_script_carries_no_host_specific_path() -> None:
    source = (ROOT / "service" / "decider_rerank_server.py").read_text(encoding="utf-8")
    assert "/home/" not in source


def test_the_script_ships_beside_the_embed_server_and_is_tracked() -> None:
    script = ROOT / "service" / "decider_rerank_server.py"
    assert script.is_file()
    assert script.parent == (ROOT / "service")
    assert (script.parent / "embed_server.py").is_file()
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout, so the install scanner's file list is unknown")
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "service/decider_rerank_server.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_the_script_imports_without_torch_or_the_decider_module(tmp_path: Path) -> None:
    blocker = "raise ImportError('the model stack is not installed')\n"
    (tmp_path / "torch.py").write_text(blocker, encoding="utf-8")
    (tmp_path / "decider.py").write_text(blocker, encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(ROOT)])
    result = subprocess.run(
        [sys.executable, "-c", "import service.decider_rerank_server as m; print(m.MODEL_ID)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == MODEL_ID
