# Review: the router half of the compaction window, tickets 01 to 03

- Date: 2026-11-13
- Mode: standard, two finders (A: behavior, data, errors, resources, baseline security; B: requirements
  coverage, contracts, caller compatibility, whether the tests could detect a wrong implementation);
  the request includes a commit, so the orchestrator commits after closure
- Repository: `/home/wilder/git-repos/hermes-skill-router`
- Feature: `compaction-window`
- Base/head: `2d963ac..12c5d55` (ticket 01 at `0e36963`, tickets 02 and 03 at `6b0e64e`, the review fixes
  and the measurements at `12c5d55`)
- Scope: the event-driven window in `hermes_skill_router/`, the deleted summary scan and its tests, the
  manifest floor and release, the host-backed test fixture, the README and term index documentation, the
  end-to-end probe under `docs/work/compaction-window/probes/`, and the measurement evidence added by the
  review round
- Threat model: ordinary accidental errors in a single-writer plugin whose host calls it through the
  documented plugin API; out of scope are hostile concurrent writers, malicious skill libraries, and
  future edits outside the changed surface
- Reviewer route: provider `codex`, model `codex/luna-bifrost`, effort `max`, both finders
- Rounds: 2 (round 1 a full review of `2d963ac..6b0e64e`; round 2 a delta over `a3f8de0..12c5d55`)
- Publication: report only; no comments published

## Findings

### [P3] Reject the removed summary-boundary claim - tests/test_readme.py:167

The boundary test asserted only the new event wording. Restoring the deleted summary-digest paragraph
alongside those assertions still passed, although ticket 03 and skeleton C25 require the obsolete scan
claim to stay gone, so a maintainer could receive contradictory boundary guidance from a green suite.

Fix: assert that the removed paragraph's fragments are absent from the prose in
`test_the_documentation_names_the_compaction_event_as_the_window_boundary`, and prove the assertion red by
restoring the paragraph in a scratch copy.

### [P3] Assert the measured rerank times - tests/test_readme.py:221

The cost test checked the model, the weight size, the thread count and the shortlist labels but no
duration. Deleting the 0.71 second median or the 0.15 second 10-row result left the test green, so the
user-facing cost the README promises could drift or vanish silently.

Fix: assert the 50-row median, the 10-row time and the slow-call figures in
`test_the_readme_names_the_rerank_cost_it_measured`, and prove each assertion red by deleting the times in a
scratch copy.

## Possible concerns

- Skeleton C22 promised a mapping payload the host never sends; a positional mapping is ignored by the
  callback's string guard. at hermes_skill_router/__init__.py (the `on_post_compaction` callback)
  Settle with: a positional-mapping call against the real callback plus a read of the dispatch helper.
  Result: confirmed as a wording defect only, since the host always calls hook callbacks with keywords
  (`hermes_cli/plugins_dispatch.py:634`, `:755`, helper at `:250-263`). The orchestrator revised C22 to say
  the callback receives the event's fields as keyword arguments and recorded the change in the skeleton's
  revision log as tier 1. Both finders confirmed the new wording matches the host in round 2.

## Serious unresolved risks

None. Both finders closed every finding and every concern in round 2.

## Queued for the owner

- deepcanary: none, the concern list closed empty
- board: none published, the concern list closed empty

## Verdict

VERDICT: clean

Finder A: "VERDICT: approved", with no prior findings to disposition and no new findings.
Finder B: "VERDICT: approved", both prior P3s FIXED.

The review closes on the current tree at `12c5d55`.

## Dispositions

| Finding | Disposition | Evidence |
|---|---|---|
| [P3] Reject the removed summary-boundary claim | RESOLVED | `tests/test_readme.py:182-192` asserts five stale fragments absent; finder B restored the old paragraph in a scratch copy and the boundary test failed at `the record holds the summary's digest` (1 failed, 16 passed); finder A reproduced the same red at the new stale-claim assertion |
| [P3] Assert the measured rerank times | RESOLVED | `tests/test_readme.py:231-249` asserts the median, the 10-row time and the slow calls; finder B deleted the times in a scratch copy and the cost test failed at the median assertion (1 failed, 16 passed); finder A reproduced the same red at `the README does not carry the measured 50-row median it states` |

## Verification

```text
cd /home/wilder/git-repos/hermes-skill-router && python3 -m pytest -q
719 passed in 14.97s
```

```text
cd /home/wilder/git-repos/hermes-skill-router && python3 -m pytest tests/test_readme.py -q
17 passed in 0.07s
```

```text
# finder B, scratch copy with the old summary-digest paragraph restored
cd ~/Trash/hermes-skill-router-review-12c5d55/summary && python3 -m pytest tests/test_readme.py -q
1 failed, 16 passed
```

```text
# finder B, scratch copy with every measured time deleted
cd ~/Trash/hermes-skill-router-review-12c5d55/timing && python3 -m pytest tests/test_readme.py -q
1 failed, 16 passed
```

```text
# finder A, scratch copy with the digest fragment inserted
git archive 12c5d55 | tar -x -C ~/Trash/hermes-skill-router-delta-12c5d55/stale-summary
python3 -m pytest tests/test_readme.py::test_the_documentation_names_the_compaction_event_as_the_window_boundary -q
exit 1: "the documentation restores the removed summary-scan claim"
```

```text
# static scan, round 2 (3 changed Python files)
cccc ... : exit 0, 24 functions, cognitive sum 33 / max 12, cyclomatic sum 50 / max 9
lizard ... : exit 0, no thresholds exceeded, duplication 0.00%
```

```text
python3 /home/wilder/.dsh/skills/post-spec-skeleton/scripts/skeleton_lint.py docs/work/compaction-window/skeleton.md
traceability_rows: 7
verification_hooks: 7
status: OK
```

The probe was run before the review and again in Phase 5: the host loads the plugin through its own
discovery, `fire_post_compaction` reaches the registered callback, the record expires, the next failed
pass dumps the catalog once, and the second failed pass strips it. Exit 0, `PASS`.

## Dispatch notes

The round 1 brief named `/home/wilder/git-repos/dotfiles/.agents/skills/coding-preferences/SKILL.md` as
the preferences path; no such file exists. The correct path is
`/home/wilder/.dsh/skills/coding-preferences/SKILL.md`, and finder A read it after the orchestrator sent
the correction. Both round 2 briefs carry the correct path. The same brief told both finders to run the
static scan; both ran it on the frozen tree.

## Unverified risks

- No real compaction has emitted the event on this host. The evidence is the synthetic probe firing the
  host's own dispatch path with the host's real plugin callback. A real compaction in a live Hermes session
  is the remaining observation, and no code change can produce it on demand.
- The engine half of the feature, ticket 04 in `context-handoff-engine`, lands after this review and gets
  its own code review in that repository.
- The drift check still reports the 13 pre-existing errors recorded in the skeleton's tier-2 row. This
  change closed the two errors it owed and added none; the baseline is unchanged, so the drift output is
  not a measure of this change.
