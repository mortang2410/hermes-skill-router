# T1 Embeddings service on 8012

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

A standalone OpenAI-compatible embeddings server on `127.0.0.1:8012`, backed by the BGE-M3 ONNX
already on disk, its unit tests, and a systemd user unit beside the two rerank units.

## Covers

Stories S6. Requirements R6. Module M1. Contract C1.

## Exclusive files

- `service/__init__.py`
- `service/embed_server.py`
- `tests/test_embed_server.py`
- `service/hermes-skill-router-embed.service`

## Blockers

None. This ticket starts the run.

## What it must do

1. `GET /health` returns `{"status": "ok", "model": <model id>, "dim": 1024}` once the session is
   ready and `{"status": "loading"}` before that.
2. `POST /v1/embeddings` takes `{"model": str, "input": [str]}` and returns
   `{"model": str, "data": [{"index": int, "embedding": [float]}]}` with one entry per input, in
   input order. 400 `{"error": str}` when `input` is missing, empty, or holds a non-string. 503
   `{"error": str}` while the model loads.
3. Read its settings from the environment: `HSR_MODEL_DIR` (default
   `/home/wilder/git-repos/skill-router-and-jev/.scratch/hf_local/BAAI/bge-m3`), `HSR_PORT`
   (default `8012`), `HSR_MODEL_ID` (default `bge-m3`), `HSR_MAX_LENGTH` (default `512`).
4. Embed one text at a time, pool position 0 of `token_embeddings` (BGE-M3 is CLS-pooled), and L2
   normalise. Never pad a batch: the published export ignores the attention mask, so every row
   after the first in a padded batch is wrong.
5. Load with `onnxruntime.InferenceSession(paths.onnx_model, providers=["CPUExecutionProvider"])`
   and the tokenizer from `tokenizer.json` through `tokenizers`. No `transformers` import.
6. Serve with fastapi and uvicorn, run by the miniforge interpreter
   (`/home/wilder/miniforge3/bin/python`), which already has fastapi 0.159.1, uvicorn 0.54.0,
   onnxruntime 1.31.0 and tokenizers 0.23.3.
7. Install the unit from the checkout: symlink the tracked unit file into `~/.config/systemd/user/`,
   run `systemctl --user daemon-reload`, then `systemctl --user enable --now hermes-skill-router-embed`.
   The unit file stays in the repository; the link is the install step.

## Verification

```bash
python3 -m pytest tests/test_embed_server.py -q
```

The tests inject a fake session and tokenizer, so they pass with no model directory and no network.
They cover: shape and order, 400 on bad input, 503 before ready, the health body, and CLS pooling
against a hand-built 3 token `token_embeddings` array.

Live proof, run after the unit tests:

```bash
/home/wilder/miniforge3/bin/python -m service.embed_server --port 8012 &
curl -s 127.0.0.1:8012/health
curl -s -X POST 127.0.0.1:8012/v1/embeddings -H 'content-type: application/json' \
  -d '{"model":"bge-m3","input":["draft a changelog"]}'
```

Accept the live run only when `/health` reports `dim 1024`, the POST returns one vector, its
L2 norm is 1.0 within 1e-3, and its cosine against a vector computed directly from the ONNX file
for the same text is at least 0.9999.

## Acceptance criteria

- [ ] `/health` and `/v1/embeddings` answer in the documented shapes.
- [ ] One vector per input, in input order, L2 normalised, length 1024.
- [ ] 400 and 503 paths covered by tests.
- [ ] No `transformers` import; `pip list` in miniforge is unchanged by the service.
- [ ] The link, `daemon-reload`, `enable --now` sequence works from a clean checkout, and
      `systemctl --user is-enabled hermes-skill-router-embed` prints `enabled`.
- [ ] Port 8000 and the Hindsight reranker are untouched.

## Notes

- The model directory is 2.27 GB of weights; do not copy it into the repository.
- The unit writes to `~/.cache/hermes-skill-router/` for logs, never inside the checkout.
