"""Measure the live reranker on the exact document format the plugin sends.

Documents come from the live index at cachePath, the same rows search.py reads, and each document is
built as search.py:330-334 builds it: cfg.document_prefix + name + " - " + description.

Usage: python3 measure_rerank.py
Prints one JSON object with the per-call timings for each shortlist size.
"""

import json
import os
import statistics
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:8010/v1/rerank"
MODEL = "tasksource/tasksource-decider-nano"
PREFIX = (
    "This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
    "actually need to carry out the current task correctly. Skill: "
)
QUERY = "draft a git commit message for a bug fix"
INDEX = os.path.expanduser("~/.cache/hermes-skill-router/index.json")
SIZES = (50, 10)
RUNS = 5


def documents(count):
    with open(INDEX, encoding="utf-8") as handle:
        data = json.load(handle)
    names, descriptions = data["names"], data["descriptions"]
    step = max(1, len(names) // count)
    rows = [(names[i], descriptions[i]) for i in range(0, len(names), step)][:count]
    return [PREFIX + name + " - " + description for name, description in rows]


def call(docs):
    body = json.dumps({"model": MODEL, "query": QUERY, "documents": docs}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    return (time.perf_counter() - started) * 1000.0, len(payload["results"])


def main():
    report = {"endpoint": ENDPOINT, "model": MODEL, "query": QUERY, "runs": {}, "warmup": []}
    # One warmup call per size, discarded, so the first measured call is not a cold model.
    for size in SIZES:
        docs = documents(size)
        warm_ms, count = call(docs)
        report["warmup"].append({"size": size, "ms": round(warm_ms, 1), "results": count})
    for size in SIZES:
        docs = documents(size)
        timings = []
        for _ in range(RUNS):
            ms, count = call(docs)
            assert count == size, f"expected {size} results, got {count}"
            timings.append(round(ms, 1))
        report["runs"][str(size)] = {
            "documents": size,
            "timings_ms": timings,
            "median_ms": round(statistics.median(timings), 1),
            "min_ms": min(timings),
            "max_ms": max(timings),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
