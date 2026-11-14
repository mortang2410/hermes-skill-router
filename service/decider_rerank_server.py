"""Cohere-compatible /v1/rerank server backed by tasksource-decider-nano on CPU.

This is the bundled copy of the decider rerank server, shipped beside
``service/embed_server.py`` so a fresh checkout holds both servers the default
config points at. The setup script starts it with ``DECIDER_DIR`` set to the
download directory the user confirmed.

Wire contract (same shape the skill router's cohere-compatible client sends):

  POST /rerank | /v1/rerank | /v2/rerank
    {"model": "...", "query": "...", "documents": ["...", ...],
     "top_n": 32, "return_documents": false}
  ->
    {"results": [{"index": 0, "relevance_score": 0.98}, ...]}
  GET /health | /
  ->
    {"status": "ok", "model": "tasksource-decider-nano", "device": "cpu", "threads": 8}
    or {"status": "loading"} before the model is installed

Results are returned for EVERY input document, in input index order. The skill
router pairs scores to candidates by the ``index`` field, so order and truncation
do not matter, but the full index-ordered set is what the other consumers of this
wire shape expect.

Each document arrives as

    "This skill supplies ... Skill: <name> - <description>"

(the skill corpus) or the same with "This tool supplies ... Tool: <name> - <description>"
(the tool corpus). The name becomes the decider option key, the description the
option text, and the corpus picks the instruction. The decider's softmax share
over the options is returned as relevance_score, which the router clamps to [0,1]
and compares against its cutoff directly.

Environment overrides:
  DECIDER_DIR     snapshot directory (default: the skill-router cache model directory)
  DECIDER_THREADS torch CPU threads (default 8)
  DECIDER_HOST / DECIDER_PORT   listen address (default 127.0.0.1:8010)

The model stack (``torch`` and the snapshot's ``decider`` module) is imported inside
``load_model``, not at module import, so the script can be inspected and tested on a
host that has not downloaded the weights yet.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Sequence

MODEL_ID = "tasksource-decider-nano"
DEFAULT_THREADS = 8
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010

#: The cache layout the setup script defaults to; ``DECIDER_DIR`` overrides it with
#: the directory the user confirmed before the download.
DEFAULT_MODEL_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "hermes-skill-router", "models", MODEL_ID
)

SKILL_PREFIX = (
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent "
    "would actually need to carry out the current task correctly. Skill: "
)
TOOL_PREFIX = (
    "This tool supplies a capability that the agent would actually need to carry out the "
    "current task correctly. Tool: "
)
SKILL_INSTRUCTION = "Which skills would the agent need to load to carry out the current task correctly?"
TOOL_INSTRUCTION = "Which tools would the agent need to load to carry out the current task correctly?"


@dataclass(frozen=True)
class Settings:
    """Service settings, read from ``DECIDER_*`` environment variables."""

    model_dir: str = DEFAULT_MODEL_DIR
    threads: int = DEFAULT_THREADS
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        return cls(
            model_dir=values.get("DECIDER_DIR") or DEFAULT_MODEL_DIR,
            threads=_env_int(values, "DECIDER_THREADS", DEFAULT_THREADS),
            host=values.get("DECIDER_HOST") or DEFAULT_HOST,
            port=_env_int(values, "DECIDER_PORT", DEFAULT_PORT),
        )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


class DeciderState:
    """The loaded decider and the lock that serialises one answer at a time.

    ``main`` installs the model before the server binds, so a served request never
    meets an empty state. Tests install a fake decider to exercise the wire without
    the weights.
    """

    def __init__(self, model_id: str = MODEL_ID, threads: int = DEFAULT_THREADS) -> None:
        self.model_id = model_id
        self.threads = threads
        self.model: Any | None = None
        # The decider's forward path and its tensors are shared process state, so
        # one answer runs at a time.
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def install(self, model: Any) -> None:
        self.model = model


def load_model(settings: Settings) -> Any:
    """Load the snapshot's ``Decider``, warm it, and return it.

    The snapshot directory must already hold ``decider.py`` and
    ``model.safetensors``; the setup script downloads it and passes the path.
    """
    if not os.path.isdir(settings.model_dir):
        raise SystemExit(
            f"decider model directory not found: {settings.model_dir}; "
            "download the models with the setup script or set DECIDER_DIR"
        )

    sys.path.insert(0, settings.model_dir)
    from decider import Decider  # noqa: PLC0415 - the model stack is optional at import time

    import torch  # noqa: PLC0415 - see the module docstring

    torch.set_num_threads(settings.threads)
    print(f"loading decider from {settings.model_dir}", flush=True)
    model = Decider.from_pretrained(settings.model_dir, device="cpu")
    print("decider loaded", flush=True)

    # Warm the forward path so the first real turn is not the one that pays for it.
    model.answer(
        state={"current_request": "warmup", "recent_requests": []},
        questions={
            "skills": {
                "type": "choice",
                "instructions": SKILL_INSTRUCTION,
                "criteria": {"a": "warmup one", "b": "warmup two"},
            }
        },
    )
    print("decider warmed", flush=True)
    return model


def parse_document(doc: str) -> tuple[str, str, str]:
    """(corpus, name, description) for one document string."""
    if doc.startswith(SKILL_PREFIX):
        body = doc[len(SKILL_PREFIX) :]
        name, sep, desc = body.partition(" - ")
        return "skill", name.strip(), (desc.strip() if sep else body.strip())
    if doc.startswith(TOOL_PREFIX):
        body = doc[len(TOOL_PREFIX) :]
        name, sep, desc = body.partition(" - ")
        return "tool", name.strip(), (desc.strip() if sep else body.strip())
    return "skill", "", doc.strip()


def score_documents(state: DeciderState, query: str, documents: Sequence[str]) -> list[float]:
    """One relevance score per input document, in input index order."""
    parsed = [parse_document(document) for document in documents]
    corpus = "tool" if parsed and all(item[0] == "tool" for item in parsed) else "skill"
    instruction = TOOL_INSTRUCTION if corpus == "tool" else SKILL_INSTRUCTION

    criteria: dict[str, str] = {}
    keys: list[str] = []
    for index, (_corpus, name, description) in enumerate(parsed):
        key = name if name else f"candidate_{index}"
        if key in criteria:
            key = f"{key}#{index}"
        criteria[key] = description if description else (name or f"candidate {index}")
        keys.append(key)

    with state.lock:
        out = state.model.answer(
            state={"current_request": query, "recent_requests": []},
            questions={
                "skills": {
                    "type": "choice",
                    "instructions": instruction,
                    "criteria": criteria,
                }
            },
        )
    probabilities = out["answers"]["skills"]["probabilities"]
    return [float(probabilities.get(key, 0.0)) for key in keys]


class _Server(ThreadingHTTPServer):
    """One server per launch, carrying the state its handler reads."""

    daemon_threads = True
    allow_reuse_address = True

    state: DeciderState


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> DeciderState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # keep the journal readable
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        if self.path not in ("/health", "/"):
            self._send(404, {"error": "not found"})
            return
        if not self.state.ready:
            self._send(200, {"status": "loading"})
            return
        self._send(
            200,
            {
                "status": "ok",
                "model": self.state.model_id,
                "device": "cpu",
                "threads": self.state.threads,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        if self.path not in ("/rerank", "/v1/rerank", "/v2/rerank"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "body must be a JSON object"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "body must be a JSON object"})
            return
        query = payload.get("query", "")
        documents = payload.get("documents", [])
        if not isinstance(query, str) or not isinstance(documents, list) or not documents:
            self._send(400, {"error": "query and a non-empty documents array are required"})
            return
        try:
            started = time.time()
            scores = score_documents(self.state, query, [str(document) for document in documents])
            elapsed = time.time() - started
            top = max(scores) if scores else 0.0
            corpus = "tool" if any(str(d).startswith(TOOL_PREFIX) for d in documents[:1]) else "skill"
            print(
                f"rerank docs={len(documents)} corpus={corpus} max={top:.4f} {elapsed:.2f}s",
                flush=True,
            )
            results = [{"index": index, "relevance_score": score} for index, score in enumerate(scores)]
            self._send(200, {"results": results, "model": self.state.model_id})
        except Exception as err:  # noqa: BLE001 - report, never crash the service
            print(f"rerank failed: {type(err).__name__}: {err}", flush=True)
            self._send(500, {"error": f"{type(err).__name__}: {err}"})


def create_server(settings: Settings, state: DeciderState) -> ThreadingHTTPServer:
    """Bind the server on ``settings.host`` and ``settings.port`` (0 picks a free port)."""
    server = _Server((settings.host, settings.port), Handler)
    server.state = state
    return server


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no command-line flags: the environment carries every setting
    settings = Settings.from_env()
    state = DeciderState(threads=settings.threads)
    state.install(load_model(settings))
    server = create_server(settings, state)
    print(f"listening on http://{settings.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
