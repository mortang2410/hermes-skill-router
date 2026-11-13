import json, os, statistics, time, urllib.request
ENDPOINT = "http://127.0.0.1:8010/v1/rerank"
MODEL = "tasksource/tasksource-decider-nano"
PREFIX = ("This skill supplies a procedure, tool, or domain-specific instructions that the agent would "
          "actually need to carry out the current task correctly. Skill: ")
QUERY = "draft a git commit message for a bug fix"
with open(os.path.expanduser("~/.cache/hermes-skill-router/index.json"), encoding="utf-8") as f:
    data = json.load(f)
names, descriptions = data["names"], data["descriptions"]
rows = [(names[i], descriptions[i]) for i in range(0, len(names), len(names)//50)][:50]
docs = [PREFIX + n + " - " + d for n, d in rows]
timings = []
for _ in range(5):
    body = json.dumps({"model": MODEL, "query": QUERY, "documents": docs}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        p = json.loads(r.read())
    assert len(p["results"]) == 50
    timings.append(round((time.perf_counter() - t) * 1000.0, 1))
print(json.dumps({"documents": 50, "timings_ms": timings, "median_ms": statistics.median(timings), "min_ms": min(timings), "max_ms": max(timings)}))
