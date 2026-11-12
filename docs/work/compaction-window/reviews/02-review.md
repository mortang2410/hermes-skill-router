# Review: spec, skeleton and tickets for the compaction window, round 2

- Date: 2026-11-13
- Shape: one reviewer, read-only, no commit in the request; the orchestrator writes this report
- Repository: `/home/wilder/git-repos/hermes-skill-router`
- Feature: `compaction-window`
- Base/head: `main` at `5345ce9`, documents untracked under `docs/work/compaction-window/`
- Scope: the spec, skeleton and four tickets as republished after round 1, the four implementer briefs,
  and the kata records `gw6w`, `085f`, `1516`, `rzfc`, `b3qh`
- Reviewer route: provider `codex`, model `codex/luna-bifrost`, effort `max`
- Static scan: `scan skipped`, reason: no code changed in this round, only documents and tracker bodies
- Rounds: 2
- Publication: report only; no comments published

## Findings

### [P1] The probe does not exercise the handoff-to-router path - tickets/04-engine-announces.md:24

Round 1's finding was answered with a self-check over stubbed host modules plus a probe that sends a
handoff-shaped event through the dispatcher. Neither establishes that the engine's own commit invokes
the live host event and reaches the router's callback, so a break in that wiring could leave the dump
record uncleared after a committed handoff.

Fix: the probe calls the engine's real announcement function for a scratch session, so the event
travels engine code, the host's fire path, the dispatcher and the router's registered callback, and it
asserts the record expires and the next failed pass dumps again.

### [P2] Cleanup after a timed-out hook call is promised, not guaranteed - skeleton.md:181-186

The skeleton said that if the worker were abandoned mid-call, the next compaction would expire the
entry. The dispatcher skips a later call while the earlier call identity is still running, and keeps
skipping for the suppression window after a timeout, so a mark can outlive the event that should have
cleared it.

Fix: describe recovery as conditional and name the skip conditions with their anchors.

### [P2] The authoring rule describes the wrong text inputs - spec.md:181-184, briefs/03-brief.md

The spec, the ticket and the brief said the description is the only text the router scores, and that
text past `maxDescriptionChars` never reaches the model. The embedder and the term index read the name
plus the clipped description; the reranker reads the name plus the full description. The guidance
therefore misleads an author about the name's weight and about where the cap bites.

Fix: state both scored fields, say which stage reads the clipped text and which reads the full text,
and align the ticket criterion, the brief and the README pin with that.

### [P3] R6 cites a section title that no longer exists - skeleton.md:22

R6 anchored "The host floor is a declaration", which the spec now titles "The host enforces the floor
before it imports the plugin", and it omitted user stories 21 to 24, which contract C25 and ticket 03
now carry.

Fix: point R6 at the current section title and include stories 21 to 24 in its anchors.

## Dispositions

| Finding | Disposition | Evidence |
|---|---|---|
| 1 below-floor behaviour stated backwards | RESOLVED | spec.md:153-159, skeleton.md C23, tickets/03:30-31 |
| 2 "opens a compaction window" for block presence | RESOLVED | briefs/03:31-45 vocabulary rules |
| 3 no ticket owned the engine-to-router check | PARTIAL | tickets/04:24-26, carried into the local report's P1 |
| 4 ticket 04 blocked only by 01 | RESOLVED | tickets/04:13, kata `b3qh` blocked by 085f, 1516, rzfc |
| 5 no test through the registration | RESOLVED | tickets/01:32-34, briefs/01:84-87 |
| 6 R5 had no contract, skip untested | RESOLVED | skeleton.md C26 and the R5 trace, tickets/02:30-33, briefs/02:61-63 |
| 7 incomplete module ownership | RESOLVED | skeleton.md M27 and M33 rows, briefs/04:34-41 |
| 8 state section named M26 | RESOLVED | skeleton.md:38, :177 |
| 9 bounded-worker claim overstated | PARTIAL | skeleton.md:181-186, carried into the local report's P2 |
| 10 dated revision-log row that was not a revision | RESOLVED | skeleton.md revision log |
| 11 stale worktree claim in the directive table | RESOLVED | spec.md:199-211 |
| 12 brief 02 rationale contradicted its dispatch | RESOLVED | briefs/02:5-7, :17-20 |

## Verdict

VERDICT: request changes

Nine of twelve findings are closed. The three that remain are one structural gap carried forward (the
probe still does not exercise the engine's own announcement) and two local wording defects. One new
P3 records a stale anchor. None of them reopens a settled decision.

## Verification

```text
$ python3 skeleton_lint.py docs/work/compaction-window/skeleton.md
requirements: 7
modules: 14
contracts: 7
stable: 6
provisional: 1
uncertainties: 2
traceability_rows: 7
verification_hooks: 7
status: OK
```

## Unverified risks

- No live compaction has delivered `post_compaction` to a router callback on this host.
- The engine's commit path has never called the event, and the engine repository has no tests.
- The README cost figures do not exist yet, so no reader has checked them.
