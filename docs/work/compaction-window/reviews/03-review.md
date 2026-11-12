# Review: spec, skeleton and tickets for the compaction window, round 3

- Date: 2026-11-13
- Shape: one reviewer, read-only, no commit in the request; the orchestrator writes this report
- Repository: `/home/wilder/git-repos/hermes-skill-router`
- Feature: `compaction-window`
- Base/head: `main` at `5345ce9`, documents under `docs/work/compaction-window/`
- Scope: the spec, skeleton and four tickets as republished after round 2, the four implementer briefs,
  and the kata records `gw6w`, `085f`, `1516`, `rzfc`, `b3qh`
- Reviewer route: provider `codex`, model `codex/luna-bifrost`, effort `max`
- Static scan: `scan skipped`, reason: no code changed in this round, only documents and tracker bodies
- Rounds: 3
- Publication: report only; no comments published

## Findings

No findings.

## Dispositions

| Finding | Disposition | Evidence |
|---|---|---|
| R2 P1 the probe did not exercise the handoff-to-router path | RESOLVED | tickets/04:24-28 and :47-49, briefs/04:104-108 |
| R2 P2 cleanup after a timed-out call was promised | RESOLVED | skeleton.md:181-188 |
| R2 P2 the authoring rule named the wrong text inputs | RESOLVED | spec.md:181-186, skeleton C25:144-146, tickets/03:39-43, briefs/03:91-106 |
| R2 P3 R6 anchored a section title that no longer exists | RESOLVED | skeleton.md:22 |

## Verdict

VERDICT: clean

The set is approved for implementation: seven requirements, fourteen modules, seven contracts (six
stable, one provisional), two open uncertainties, and every requirement traced to a contract and a
ticket.

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

```text
$ grep -rnP "\x{2014}" docs/work/compaction-window/
$ echo $?
1
```

## Unverified risks

- U1: no live compaction has delivered `post_compaction` to a router callback on this host. The probe in
  ticket 04 is the settling step, and until it runs the event path is read from source, not observed.
- U3: the engine's commit path has never called the event, and the engine repository has no tests.
- The README cost figures this feature requires do not exist yet, so no reader has checked them.
