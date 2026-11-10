# router-sync run state

Phase: 3, implementation. Review round 7 returned `VERDICT: APPROVE` over the spec, skeleton, tickets
T1 to T8 and STATE, after closing three findings: the required `provides_middleware` declaration, the
demoted `[names only]` rows that must stay kept, and the required non-empty `description`. Round 6's
approval was voided by those edits. Record in `reviews/07-approval.md`. A later T6 scope correction
(no `src/index/index.ts` exists; the four HTTP keys need documentation) was approved as round 8 in
`reviews/08-t6-delta-approval.md`.

## Done

- Spec: `docs/work/router-sync/spec.md`, published as kata `93w7`.
- Skeleton: `docs/work/router-sync/skeleton.md`, `skeleton_lint.py` status OK.
- Tickets T1 to T8 under `docs/work/router-sync/tickets/`, published as kata `wfbv` (T1),
  `xyad` (T2), `kb0a` (T3), `10h3` (T4), `tp1d` (T5), `k9f9` (T6), `cnmg` (T7), `d3f6` (T8), all
  `ready-for-agent`, all children of `93w7`, edges: T3 blocked by T2, T4 by T3, T5 by T1 and T4, T6 by
  T1, T7 by T5 and T6, T8 by T5.
- Review round 1 in `reviews/01-review.md`: `VERDICT: CHANGES`, ten findings, all fixed.
- Display amendment: S10, D12, `showSuggestions` / `suggestionsPath` / `maxDisplayNames`, R12, modules
  M11 to M14 inside C9, ticket T8, README section in T7.
- Display delta review in `reviews/02-display-review.md`: nine findings fixed, then `VERDICT: APPROVE`
  on the re-read.
- Eager and deferred amendment: S9, D11, R11, the C3 partition, T2/T3/T5/T7. Source study in
  `research/dsh-router-study.md`.
- Review round 6 in `reviews/06-approval.md`: `VERDICT: APPROVE` over the whole set. The approval
  covers the spec, skeleton and tickets as they stand; any later edit voids it.
- Review round 5 in `reviews/05-review.md`: `VERDICT: CHANGES`, four findings, all fixed in one pass:
  the routed-only index reached D9 and T3, the no-block condition uses kept rows, the record holds
  selected names only in D12 and the live proof, and the R11 contracts match the trace row.
- Review round 4 in `reviews/04-review.md`: `VERDICT: CHANGES`, six findings, all fixed in one pass:
  the index is routed-only, `maxSuggestions` caps suggestions only, the record carries selected names
  only, M8 and M9 allow the dependencies their contracts need, and R11 is traced through the block
  with an end-to-end criterion.
- Review round 3 in `reviews/03-review.md`: `VERDICT: CHANGES`, ten findings, all reproduced and fixed
  in one pass. The fixes are listed there; the headline ones are the unfiltered middleware kwargs, the
  record writer moved to T5, whole-section strip with eager rows in the block, redirect refusal in both
  HTTP clients, and the T6 file and profile owners.
- Live state checked: 8010 answers `{"model": "tasksource-decider-nano"}`; 8012 has no listener; port
  8000 stays Hindsight's; `~/.hermes/plugins` exists; pytest 9.1.1 in miniforge; the service
  dependencies are present; Hermes reports `v0.22.2+1671.gc479f48.dirty`.

## Open

- Source trees differ: the live Hermes runtime is a workspace install at
  `/home/wilder/.hermes/installs/5323f763494eb399/environments/*/workspace`, which emits the
  shadowed-copy note and the names-only note; `/home/wilder/.hermes/hermes-agent` is the tree the
  registration test imports `hermes_cli` from and lacks that note. Ticket line references to
  `agent/prompt_builder.py` anchor the older tree; the strip is verified against captured live bytes.

- Phase 3 implementation is complete: T1 to T8 done and verified, 441 tests passing, validator clean.
  Round 18 approval covers the caption wording, D4's redirect description and the query/fragment rule
  (`reviews/13-round18-approval.md`). Phase 4 closed four review rounds; every finding fixed and
  verified, including credential redaction (`reviews/code-01..04.md`).
- Approval voids on any later edit to the spec, a ticket, or the skeleton.

## Next

1. Phase 3: implementers on `bifrost` / `opencode-go-custom/deepseek-go` / `max`, one child per ready
   ticket, disjoint exclusive files, no commits.
2. Phase 4: code review on luna max. Phase 5: verify with the four catalog checks, commit, push, close
   kata, delete the reminder `schedule-859030b2-634a-4a68-8a04-adf162e77e6f`.

## Do not repeat

- Do not quote plugin sample config as a live route.
- Do not use `skills.disabled` to hide the catalog; it also disables `skill_view`.
- Do not touch port 8000 (Hindsight) or 8010's owner unit.
- Do not pass `run_in_background: false`.
- Do not implement before the document verdict.
- Do not patch Hermes UI code for the display, and do not promise a per-message transcript row:
  Hermes offers a desktop pane and a docked TUI widget, nothing inside the transcript.
- Do not call middleware `inspect.signature`-filtered: the dispatcher calls `cb(**kwargs)` and a narrow
  signature is silently skipped.
- Do not expect a disabled desktop plugin to render anything, including an explanation.
