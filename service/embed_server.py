"""OpenAI-compatible embeddings server for BGE-M3 on CPU.

Contract C1 (``docs/work/router-sync/skeleton.md``): ``GET /health`` and
``POST /v1/embeddings`` on loopback. The weights are the published BGE-M3 ONNX
export under ``$HSR_MODEL_DIR``; the tokenizer is read from ``tokenizer.json``
through ``tokenizers``. ``transformers`` is never imported.

Two measured facts from the DSH embedder
(``packages/dsh-skill-router/src/index/bge-embedder.ts``) shape the embedding
path, and both fail silently when ignored:

* the published export ignores the attention mask, so every row after the
  first in a padded batch is wrong; text is therefore embedded one at a time
  and the tokenizer never pads;
* the export exposes ``token_embeddings`` and ``sentence_embedding``; BGE-M3 is
  CLS-pooled, so position 0 of ``token_embeddings`` is taken and
  ``sentence_embedding`` is never read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from tokenizers import Tokenizer

DEFAULT_MODEL_DIR = (
    "/home/wilder/git-repos/skill-router-and-jev/.scratch/hf_local/BAAI/bge-m3"
)
DEFAULT_PORT = 8012
DEFAULT_MODEL_ID = "bge-m3"
DEFAULT_MAX_LENGTH = 512

LOG = logging.getLogger("hermes-skill-router.embed")
LOG_DIR = os.path.join(os.path.expanduser("~"), ".cache", "hermes-skill-router")
LOG_FILE = os.path.join(LOG_DIR, "embed.log")


@dataclass(frozen=True)
class ModelPaths:
    """Files the published BGE-M3 export needs, below ``model_dir``."""

    model_dir: str

    @property
    def onnx_model(self) -> str:
        return os.path.join(self.model_dir, "onnx", "model.onnx")

    @property
    def tokenizer(self) -> str:
        return os.path.join(self.model_dir, "tokenizer.json")


@dataclass(frozen=True)
class Settings:
    """Service settings, read from ``HSR_*`` environment variables."""

    model_dir: str = DEFAULT_MODEL_DIR
    port: int = DEFAULT_PORT
    model_id: str = DEFAULT_MODEL_ID
    max_length: int = DEFAULT_MAX_LENGTH

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        values = os.environ if env is None else env
        return cls(
            model_dir=values.get("HSR_MODEL_DIR") or DEFAULT_MODEL_DIR,
            port=_env_int(values, "HSR_PORT", DEFAULT_PORT),
            model_id=values.get("HSR_MODEL_ID") or DEFAULT_MODEL_ID,
            max_length=_env_int(values, "HSR_MAX_LENGTH", DEFAULT_MAX_LENGTH),
        )


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}")
    return value


class EmbeddingState:
    """Model slots shared by the request handlers.

    The loader installs the session, tokenizer and dimension once they are
    ready; until then ``ready`` is False, ``GET /health`` reports ``loading``
    and embeddings requests get 503. Tests install a fake session and tokenizer
    so nothing is read from disk.
    """

    def __init__(self, model_id: str, dim: int = 0) -> None:
        self.model_id = model_id
        self.dim = dim
        self.session: Any | None = None
        self.tokenizer: Any | None = None
        self._ready = False
        # One text at a time: the ONNX session and the tokenizer are shared
        # process state, and a serialised embed keeps the vector for a text
        # identical across concurrent calls.
        self.embed_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._ready

    def install(self, session: Any, tokenizer: Any, dim: int) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.dim = int(dim)
        self._ready = True


def _token_embeddings(session: Any, outputs: Sequence[Any]) -> Any:
    """Select the ``token_embeddings`` output, never ``sentence_embedding``."""
    names = [output.name for output in session.get_outputs()]
    try:
        index = names.index("token_embeddings")
    except ValueError:
        index = 0
    return outputs[index]


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """L2 normalise a float vector; an all-zero vector stays zero."""
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector, dtype=np.float32) / norm


def embed_one(state: EmbeddingState, text: str) -> list[float]:
    """Embed one text: one unpadded sequence, CLS pool, L2 normalise."""
    if state.session is None or state.tokenizer is None:
        raise RuntimeError("the embedding model is not loaded")
    with state.embed_lock:
        encoding = state.tokenizer.encode(text)
        input_ids = np.asarray([encoding.ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)
        outputs = state.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    token_embeddings = _token_embeddings(state.session, outputs)
    cls = np.asarray(token_embeddings, dtype=np.float32)[0, 0, :]
    return _unit_vector(cls).tolist()


def _output_dim(session: Any) -> int:
    for output in session.get_outputs():
        if output.name == "token_embeddings":
            shape = output.shape
            if shape and isinstance(shape[-1], int):
                return int(shape[-1])
    raise RuntimeError("the model exposes no token_embeddings output with a fixed width")


def load_models(state: EmbeddingState, settings: Settings) -> None:
    """Load the tokenizer and ONNX session, then mark the state ready."""
    paths = ModelPaths(settings.model_dir)
    tokenizer = Tokenizer.from_file(paths.tokenizer)
    # Truncation only: padding is deliberately never enabled, because the
    # published export ignores the attention mask.
    tokenizer.enable_truncation(max_length=settings.max_length)
    session = onnxruntime.InferenceSession(
        paths.onnx_model, providers=["CPUExecutionProvider"]
    )
    dim = _output_dim(session)
    state.install(session, tokenizer, dim)
    LOG.info(
        "embeddings model ready: model_id=%s dim=%d max_length=%d model=%s",
        settings.model_id,
        dim,
        settings.max_length,
        paths.onnx_model,
    )


def _load_in_background(loader: Callable[[], None]) -> None:
    try:
        loader()
    except Exception:  # noqa: BLE001 - the process must keep answering /health
        LOG.exception("could not load the embedding model")


async def _read_json_object(request: Request) -> dict[str, Any] | None:
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def create_app(
    state: EmbeddingState, loader: Callable[[], None] | None = None
) -> FastAPI:
    """Build the ASGI app; ``loader`` runs in a thread once serving starts."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if loader is not None:
            thread = threading.Thread(
                target=_load_in_background,
                args=(loader,),
                name="embedding-model-loader",
                daemon=True,
            )
            thread.start()
        yield

    app = FastAPI(
        title="hermes-skill-router embeddings",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        if not state.ready:
            return {"status": "loading"}
        return {"status": "ok", "model": state.model_id, "dim": state.dim}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        if not state.ready:
            return JSONResponse(
                status_code=503, content={"error": "the embedding model is still loading"}
            )
        payload = await _read_json_object(request)
        if payload is None:
            return JSONResponse(
                status_code=400, content={"error": "body must be a JSON object"}
            )
        texts = payload.get("input")
        if (
            not isinstance(texts, list)
            or not texts
            or any(not isinstance(text, str) for text in texts)
        ):
            return JSONResponse(
                status_code=400,
                content={"error": "input must be a non-empty list of strings"},
            )
        model = payload.get("model")
        if not isinstance(model, str) or model == "":
            model = state.model_id
        data = []
        for index, text in enumerate(texts):
            vector = await run_in_threadpool(embed_one, state, text)
            data.append({"index": index, "embedding": vector})
        return {"model": model, "data": data}

    return app


def _configure_logging() -> None:
    """Log to stderr and to ``~/.cache/hermes-skill-router/embed.log``."""
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger()
    if not any(
        getattr(handler, "baseFilename", None) == LOG_FILE for handler in root.handlers
    ):
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible BGE-M3 embeddings server"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=settings.port)
    args = parser.parse_args(argv)

    _configure_logging()
    state = EmbeddingState(model_id=settings.model_id)
    app = create_app(state, loader=lambda: load_models(state, settings))
    LOG.info(
        "starting embeddings service on http://%s:%d (model_dir=%s)",
        args.host,
        args.port,
        settings.model_dir,
    )
    # log_config=None keeps uvicorn's records on the root logger, so the file
    # handler above receives them too.
    uvicorn.run(app, host=args.host, port=args.port, log_config=None, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
