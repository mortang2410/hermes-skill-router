# The rerank stage's measured cost

The README publishes the rerank stage's cost. This file carries the numbers, the scripts that produced
them and the machine they ran on, so a reader can check the claim instead of trusting it.

Measured 2026-11-13 on the machine that serves the router, against the live endpoint.

## Setup

- Endpoint: `POST http://127.0.0.1:8010/v1/rerank`, provider shape `cohere`, model
  `tasksource/tasksource-decider-nano`.
- The server's own health report: `device cpu`, `threads 8`.
- Model: a joint cross-encoder on `cross-encoder/ettin-reranker-150m-v1` (ModernBERT family). Its
  `model.safetensors` is 298,044,578 bytes and its header declares 149,015,041 parameters in BF16.
- Documents are built exactly as `hermes_skill_router/search.py` builds them:
  `documentPrefix + name + " - " + description`. Fifty rows were taken at stride 8 from the live
  index at `~/.cache/hermes-skill-router/index.json` (420 rows).
- Query: `draft a git commit message for a bug fix`.

## Commands

```sh
cd /home/wilder/Trash/compaction-window
python3 measure_rerank.py
python3 measure_50.py
```

Both scripts are stored beside this file (`measure_rerank.py`, `measure_50.py`).

## Raw timings, milliseconds, wall clock per HTTP call

50 documents, first batch (other host jobs were competing for the CPUs, load average 11.18 on 16):

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| ms | 2694.1 | 5174.9 | 4482.9 | 978.4 | 715.0 |

50 documents, second batch, same documents and query:

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| ms | 713.0 | 630.2 | 654.1 | 632.4 | 672.3 |

Pooled ten calls: median 714.0 ms, minimum 630.2 ms, maximum 5174.9 ms.

10 documents:

| run | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| ms | 167.7 | 153.4 | 147.2 | 144.1 | 149.0 |

Median 149.0 ms.

Discarded warmup calls, the first request after an idle period: 5958.6 ms at 50 documents and
6396.0 ms at 10.

## What the README says, and why

The README says the default 50-row shortlist reranked in a median 0.71 seconds over ten calls, with the
two slowest calls at 4.5 and 5.2 seconds while other jobs competed for the CPU, and that a 10-row
shortlist took 0.15 seconds.

The quiet-run median alone (654 ms) would understate what a loaded machine pays; the loaded-run median
alone (2694 ms) would overstate what the stage costs when nothing else is running. The ten-call median
with the two outliers named is the measured range, and the outlier sentence says what caused them.
