# Review round 1: fused shortlist, dump store and docs

Scope: `e5be2bd^..4f4ebbf` on `main` of hermes-skill-router. Frozen tree: `4f4ebbf`.

## Threat model

In scope: ordinary accidental errors in a single-writer plugin. A routing pass that loses or mangles
the user's request, a catalog dump that repeats or vanishes inside a compaction window, a channel that
silently stops contributing, a config bound that accepts a nonsense value, an endpoint failure that
takes the turn down instead of failing open, and documentation that describes behavior the code no
longer has.

Out of scope: hostile concurrent writers to the same cache files, malicious skill names, a
compromised host, and fantastical future edits. The association store is written by one plugin path
inside one Hermes process.

## Round scan (static)

`scripts/static-scan.sh e5be2bd^` over 25 changed paths, 20 code files. Full output:
`/home/wilder/Trash/hermes-review/round1-scan.txt`. cccc exit 0, lizard exit 0, duplicate rate 4.12%
with every duplicate group inside `tests/`. Crossings to judge: `bm25._parse` cognitive 43, cyclomatic
32; `config._coerce` cognitive 53; `associations.py` cognitive total 425.

## Reviewers

Four routes were dispatched on 2026-11-12, all `bifrost` / `opencode-go-custom/deepseek-go` /
reasoning `max`:

- `4ce656cc-44fb-4bbb-bed5-901e1ffdeacd` product code and tests.
- `335ddbd7-7781-4950-b855-851d6274a12d` the three documents and the catalog entry text.
- `995e771a-cf02-4d5a-89de-fc6f376e9015` live behavior, mutation probes, the upstream catalog gate.

## Verification run by the orchestrator

- Exact upstream gate, run locally against the submitted entry:
  `CHANGED_FILES=plugin-catalog/hermes-skill-router.yaml python3 -I scripts/ci/catalog_validate_pinned_entries.py`
  printed `PASS: plugin-catalog/hermes-skill-router.yaml` and
  `PASS: all changed catalog entries validated at their pinned shas`, exit 0. It cloned the pin
  `edffe7bbefb624cd4a755c3d25ddcbbc2b8b8745` and ran `hermes plugins validate --install-deps` there.
  The same run prints one warning the PR body does not carry yet: private core imports at
  `hermes_skill_router/tools.py:168` (`tools.skills_tool._locate_skill`, `_skill_search_dirs`).
- GitHub never ran the workflow itself: run 43109289776 finished `action_required` with zero jobs,
  which is the fork-PR approval hold, not a failure.

## Findings

All three verdicts were REQUEST CHANGES against `4f4ebbf`. Reports on disk:
`/home/wilder/Trash/hermes-review/code/findings.md`, `/home/wilder/Trash/hermes-review/docs/findings.md`,
`/home/wilder/Trash/hermes-review/behavior/evidence.md`.

Product code, from the code review and confirmed by live measurement in the behaviour review:

- C1 (P1) The per-session dump record never expires. `clear_dump` sits in the branch reached only when
  no dump is recorded, so one failed pass keeps stripping the catalog and suggesting nothing for the
  life of the process. Probe `probe_window_expiry.py`. The covering test uses three session ids, so it
  cannot see it.
- C2 (P1) The recency gate resolves a host skill path once per usage-bearing skill on every turn, then
  the selection resolves up to 50 more after the budget has ended. Live turns measured 19.5 to 22
  seconds with under 1 s in the endpoints, and one pass crossed `routingTimeoutMs` and dumped the
  catalog. The suite cannot see it because `skill_path` is stubbed.
- C3 (P2) Shadow mode logs the pre-bump order with movement 0 while the documentation promised the
  bumped order.
- C4 (P2) A warm vector cache with no `bm25.json` never rebuilds the term index, so the default-on BM25
  channel runs off until the catalog fingerprint changes.
- C5 (P3) `search.py` claims the search tool honours the dump record; nothing outside `router.py` reads
  it.
- C6 (P3) The dump paragraph described a payload marker the code no longer uses.

Documents, from the documentation review: seventeen findings, all confirmed and fixed in the four
documents and the manifest, including the false live-learning claim, the association half-life, the
eager-skill window claim, the failure-reason list, the dated verification section and four citation
errors in the new models guide.

Behaviour review: three findings, all duplicating C2, C3 and the README live-learning claim, plus
seven asked checks that passed, including the three mutation probes and the catalog validator.

## Dispositions

Product code: fixed in `8e59c2b`, reviewed in round 2, and fixed again in `61a9e1c` after round
2's findings; each accepted fix carries a test that fails when the fix is reverted. Round 2 and
round 3 are recorded in `15-round2-deepseek-review.md`.

Documents: seventeen of seventeen CONFIRMED and fixed, none refuted. Report
`/home/wilder/Trash/hermes-review/docs/fix-report.md`. `python3 -m pytest -q` printed
`693 passed in 13.67s` and `hermes plugins validate .` printed `Validation passed.`

Open question carried in this round: whether the router's earlier block ever reaches a later request's
payload. Hoisting answered that it does not (`agent/turn_api_request.py` assigns the middleware payload
to `api_kwargs` only), which makes the window-opening test always true in production. No request dump
on this host carries the routed text, so the claim stays unverified; the documents now state the input
condition instead of asserting the suppression happens. Probe that would settle it: a live multi-turn
session with `HERMES_DUMP_REQUESTS=1`, then check whether a later request carries a system message
opening with `Skills routed for this request`.
