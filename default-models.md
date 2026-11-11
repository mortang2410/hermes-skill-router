# Default models

The plugin defaults name two models: `bge-m3` for embeddings and `tasksource-decider-nano` for rerank.
The plugin talks to both over HTTP, so you can serve them however you prefer as long as the request shapes match.

## Embeddings: BGE-M3

BGE-M3 is a multilingual embedding model with 1024 dimensions and an 8192 token sequence length.
It runs on CPU with the ONNX export used by the bundled server.

Model page: https://huggingface.co/BAAI/bge-m3

### Download

Install the Hugging Face CLI, then download the model:

```bash
pip install -U "huggingface_hub[cli]"
hf download BAAI/bge-m3 --local-dir /path/to/bge-m3
```

The CLI guide is at https://huggingface.co/docs/huggingface_hub/en/guides/cli and the download guide is at https://huggingface.co/docs/huggingface_hub/en/guides/download.
The bundled server reads `onnx/model.onnx` and `tokenizer.json` below `HSR_MODEL_DIR`, so the local directory must keep that layout. The ONNX graph keeps its weights in `onnx/model.onnx_data` (about 2.3 GB), so keep that file beside `model.onnx`; the runtime loads it from the same directory, and a copy of `model.onnx` alone fails to open. The full repository is about 4.5 GB.
The alternative is the git route: `git lfs install` then `git clone https://huggingface.co/BAAI/bge-m3 /path/to/bge-m3`, which the download guide describes as the classic workflow the CLI replaces.

### Option 1: bundled service in this repo

From the checkout root, with `onnxruntime`, `tokenizers`, `fastapi`, `uvicorn` and `numpy` installed:

```bash
HSR_MODEL_DIR=/path/to/bge-m3 HSR_PORT=8012 python3 -m service.embed_server
```

The code in `service/embed_server.py` reads `HSR_MODEL_DIR`, `HSR_PORT` (default 8012), `HSR_MODEL_ID` (default `bge-m3`) and `HSR_MAX_LENGTH` (default 512) from the environment.
It listens on `127.0.0.1` and serves `POST /v1/embeddings` plus `GET /health`.

Health check:

```bash
curl -s http://127.0.0.1:8012/health
```

Expected: `{"status": "ok", "model": "bge-m3", "dim": 1024}` once loaded, and `{"status": "loading"}` while the model loads.

### Option 2: Text Embeddings Inference

TEI is a server for embedding and reranker models. Its repo is https://github.com/huggingface/text-embeddings-inference and its quick tour is https://huggingface.co/docs/text-embeddings-inference/en/quick_tour.

```bash
docker run -p 8012:80 -v $PWD/data:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-1.9 --model-id BAAI/bge-m3
```

The quick tour shows this pattern with a `--model-id` flag and a volume mounted at `/data` so weights are kept between runs, and a `/v1/embeddings` route that accepts OpenAI-shaped requests.
CPU installs without Docker are documented at https://huggingface.co/docs/text-embeddings-inference/en/local_cpu, which launches `text-embeddings-router --model-id <model>`.

Health check: request `POST http://127.0.0.1:8012/v1/embeddings` with `{"model": "bge-m3", "input": ["test"]}` and expect a response with a `data` list whose entries carry `embedding` vectors.
TEI lists `/health` in its CLI reference; the embeddings request above is the portable check.

### Option 3: Ollama

Ollama hosts a BGE-M3 model at https://ollama.com/library/bge-m3. It lists a 1.2 GB model with an 8K context window and more than 100 languages.

```bash
ollama pull bge-m3
```

Ollama serves embeddings at `POST /api/embed` with `{"model": "bge-m3", "input": "..."}`. Its response uses `embeddings`, not the OpenAI `data` shape, so you need a small proxy that translates between the two.
Ollama also exposes an OpenAI-compatible embeddings route in recent versions; that claim is unverified because the Ollama docs page was not fetched for this document.
The model name must match what Ollama knows (`bge-m3`), which is why this option needs the proxy plus a matching `embedModelId`.

Health check: `curl http://localhost:11434/api/tags` returns a list that contains `bge-m3`.

## Rerank: tasksource-decider-nano

Model page: https://huggingface.co/tasksource/tasksource-decider-nano

The model exists on Hugging Face under the full id `tasksource/tasksource-decider-nano`. Its card tags include `cross-encoder`, `text-embeddings-inference` and `endpoints_compatible`, and its base is `cross-encoder/ettin-reranker-150m-v1`.
The plugin default `rerankModel` is the short name `tasksource-decider-nano`, exactly as set in `README.md` and `technical-documentation.md`.
A 149M parameter model with that short name exists, but the full repo id above is the address you download and serve.

Serve a Cohere-compatible rerank. TEI documents a `/rerank` route for cross-encoders at https://huggingface.co/docs/text-embeddings-inference/en/quick_tour, with `{"query": ..., "texts": [...], "raw_scores": false}`.
TEI does not document a Cohere-shaped `/v1/rerank` route with `documents` and `top_n`, so TEI alone does not answer the plugin shape.
The Cohere v1 reference at https://docs.cohere.com/v1/reference/rerank documents `POST /v1/rerank` with `model`, `query`, `documents` and `top_n`, and a `results` list carrying `index` and `relevance_score` entries.
That is the shape `technical-documentation.md` specifies for this plugin.

Example service (not run here). This Python service uses sentence-transformers `CrossEncoder` plus FastAPI to answer that shape.
The CrossEncoder usage page is https://github.com/huggingface/sentence-transformers/blob/main/docs/cross_encoder/usage/usage.rst and shows `CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")` with `model.predict` scoring query and passage pairs.

```python
# rerank_server.py  -- example only, not run for this document
from fastapi import FastAPI
from sentence_transformers import CrossEncoder
import math

app = FastAPI()
model = CrossEncoder("tasksource/tasksource-decider-nano")

@app.post("/v1/rerank")
def rerank(body: dict):
    query = body["query"]
    docs = body["documents"]
    scores = model.predict([(query, d) for d in docs])
    results = [
        {"index": i, "relevance_score": float(1 / (1 + math.exp(-s)))}
        for i, s in enumerate(scores)
    ]
    return {"results": results}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it with `uvicorn rerank_server:app --host 127.0.0.1 --port 8010`. Send `rerankModel: tasksource-decider-nano` in the plugin config so the model field matches the default.
This snippet is an example I did not run. The decider model is a typed-decision model, so its scores may not behave like a standard reranker; that behavior is unverified.

## Checking it works

Embeddings:

```bash
curl -s http://127.0.0.1:8012/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3", "input": ["test"]}'
```

Expect a response with `model` and `data`, where each `data` entry carries `index` and a numeric `embedding` list. The bundled service returns 1024 wide vectors.

Rerank:

```bash
curl -s http://127.0.0.1:8010/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model": "tasksource-decider-nano", "query": "test query", "documents": ["doc one", "doc two"], "top_n": 2}'
```

Expect a response with `results`, where each entry carries `index` and `relevance_score` between 0 and 1.

Plugin config keys that point at these endpoints:

| Key | Default | Meaning |
| --- | --- | --- |
| `embedBaseUrl` | `http://127.0.0.1:8012` | Embeddings root. |
| `embedModelId` | `bge-m3` | Model field sent to embeddings. |
| `rerankBaseUrl` | `http://127.0.0.1:8010` | Rerank root. |
| `rerankModel` | `tasksource-decider-nano` | Model field sent to rerank. |
| `rerankProvider` | `cohere` | Request shape. Cohere is the only implemented provider. |

## If the endpoint is down

The plugin fails open. When an endpoint is unreachable or times out, the plugin leaves the full catalog in the prompt and adds no suggestions, so routing degrades instead of failing.
`README.md` states this as the fallback, and `technical-documentation.md` lists the failure reasons and the `healthCheckMs` window that skips re-probing a failed endpoint.
