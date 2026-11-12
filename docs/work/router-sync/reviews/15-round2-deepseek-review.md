# Review rounds 2 to 5: the fix round's own review

Scope: round 2 reviewed `4f4ebbf..5a86e39` (the round-1 fixes), round 3 reviewed `5a86e39..61a9e1c`
(the round-2 fixes), round 4 reviewed `61a9e1c..e616666` (the round-3 fixes), and round 5 reviewed
`e616666..804540d` in deltas. Round 1 is recorded in `14-round1-deepseek-review.md`.

## Round 2

Dispatcher: luna on `codex/luna-bifrost`, effort `max`, subagent `af3b7360-e644-4866-8efd-ae3bbc76b4aa`.
Report on disk: `/home/wilder/Trash/hermes-review/round2/findings.md`. Verdict: REQUEST CHANGES.

Findings, all reproduced by the orchestrator before fixing:

- P1 window rule: the dump record expired on any shorter request list, but Hermes can replace the
  request-local messages for one call without compacting durable history, so the catalog could be
  dumped twice in one window. Probe: `probe_window.py` printed `4 -> 6 -> 5: [(4, 'catalog_kept'),
  (6, 'catalog_stripped'), (5, 'catalog_kept')]`.
- P1 path-map scope: the saved read-path map was keyed by cache file and fingerprint only, so two
  profiles sharing one cache directory could hand profile B profile A's resolved path. Probe:
  `probe_path_cache.py` returned the `profile-a` path while calls stayed `['A']`.
- P2 hit revalidation: a same-process cache hit was returned without checking the file, so a moved
  or replaced skill kept its stale path. Probe: `probe_path_stale.py` printed
  `host_lookup_calls: ['initial']`.
- P2 miss memo: a failed host lookup was retried in the same pass. Probe: `probe_path_miss.py`
  printed `pathless_alpha_calls: 2`.
- P2 docs: the shadow paragraph said the log carried the pre-bump order with movement 0, while the
  code logs the bumped, cut order.

The reviewer also proved the bounded gate leaves the shortlist unchanged and that malformed
messages, empty session ids and process restarts do not introduce new window faults.

## Round 3

The fix commit `61a9e1c` changes the window rule to the compaction summary's digest, scopes the
read-path map by the active skill roots, validates every hit against the file's declared name,
memoises failed lookups for the pass, and restores the shadow paragraph.

Dispatcher: luna on `codex/luna-bifrost`, effort `max`, subagent `3e404ba2-788c-459e-9567-13f3114df86c`.
Report on disk: `/home/wilder/Trash/hermes-review/round3/findings.md`. Verdict: REQUEST CHANGES.

Findings so far, both P1 and both confirmed by its probes:

- P1 provenance: `router._summary_token` hashed any message whose content opened with the compaction
  prefix, so a user or tool message quoting the marker expired the dump and re-dumped the catalog.
  The host's own summaries can be user or assistant role, so a role check alone does not settle it.
- P1 roots: the skill-roots digest was memoised for the process, but the host reads project roots
  when it is called, so a project switch inside one process kept the previous profile's map.

Both are fixed in `834760d fix(router): require host provenance for the window summary and read
roots per pass`, and the provenance rule was tightened twice more after follow-up probes: `c073ede`
accepts only a message carrying the host's private `_compressed_summary` metadata, or a text that
closes with the host suffix and either opens with the full reference-only prefix or carries the
merge delimiter, and only in the leading messages where the host puts its summary; `17cab85` pins
the leading pasted-marker case as a test. `b00a075` mirrors the host's name resolution in
`_declared_name` (last duplicate `name:` wins, directory-name fallback, BOM stripped). The roots
digest is computed for every pass. Each covering test fails when its fix is reverted in a scratch
copy: the quoted-marker test, the metadata-tagged summary test, the project-root-switch test, the
moved-skill test and the failed-lookup test were all mutation-checked.

## Round 4

Round 4 reviewed `61a9e1c..e616666` (the round-3 fixes) in a clean scratch worktree at `e616666`;
the named checkout was dirty when it started, so the exact-range tests and the static scan ran in
the scratch copy and the full suite ran in both. Dispatcher: luna on `codex/luna-bifrost`, effort
`max`, subagent `7f6e6436-4104-4f90-9248-f4df1a20850f`. Report on disk:
`/home/wilder/Trash/hermes-review/round4/findings.md`. Verdict: REQUEST CHANGES.

Findings, all reproduced by the orchestrator before fixing:

- P1 merged summary: `_summary_token` accepted the merge delimiter anywhere after the closing
  suffix, so a leading message carrying only `[END OF PRIOR CONTEXT ... COMPACTION SUMMARY BELOW]`
  and `not the summary above ---` expired the dump record. The probe printed
  `leading_merge_delimiter: token=True first_catalog_kept=True pasted_message_catalog_kept=True`.
  Fixed in `401475f`: the merged form must carry the summary prefix within 160 characters after the
  delimiter, or carry the host metadata tag.
- P2 documentation: the paragraph claimed the host writes different summary text at every
  compaction, which nothing on the host checks, and said a tool message could qualify while the
  code accepts only user and assistant roles. Fixed in `401475f`.
- P2 BOM test: the fixture sat at `roots/beta/SKILL.md` and declared `name: beta`, so removing the
  BOM strip still passed through the directory-name fallback. `401475f` moves it to
  `roots/other/SKILL.md` declaring `beta`; removing BOM stripping now fails the test.
- P2 tool-role test: the spoof message was appended outside the first three non-system candidates,
  so removing the role filter left the test green. `401475f` inserts it at index 1 of
  `_window_history(2)`; removing the role filter now fails.

Its mutation battery at `e616666` failed six covering tests and passed two (BOM and tool role),
which produced the two test findings above. Its round-3 probes confirmed the marker, roots and
duplicate-name dispositions, and it accepted the byte-identical-summary tradeoff as pre-existing.
The full suite printed `711 passed` in both the named checkout and the scratch worktree.

The reviewer was re-dispatched at 11:00 to re-read `e616666..9e3e0a7`. That delta returned
REQUEST CHANGES with three new findings: a retained tail that quotes the merge delimiter hides the
host's later delimiter (P2); the four Default_Ignorable Hangul fillers sit in a letter category and
padded a short marker past the 1,000-character floor (P3); and the pasted-summary paragraph
overstated the shape the predicate accepts (P3). `804540d` had already replaced that paragraph, and
`fd8dff2` fixed the first two: `_is_compaction_summary` scans every delimiter occurrence, and
`_substantive_length` excludes U+115F, U+1160, U+3164 and U+FFA0. Both fixes carry
leading-history regressions that fail when reverted.

The second delta re-read on `9e3e0a7..fd8dff2` returned REQUEST CHANGES with one P3: the filler
regression covered U+115F and U+3164 but not U+1160 or U+FFA0, so removing either of those two
from the exclusion set left the test green. `e285d3c` runs the padded-marker loop with all four
fillers, and dropping any single member now fails the focused test.

## Round 5

Round 5 reviewed `e616666..67ed81c` and then every delta through `804540d`. Dispatcher: luna on
`codex/luna-bifrost`, effort `max`, subagent `99d3aff0-8d7a-43cf-a877-2d17f111158a`. The report on
disk is `/home/wilder/Trash/hermes-review/round5/findings.md`, which carries the reports through
`0ec523e`; its last three verdicts were returned to the orchestrator and are recorded here.

Findings, all reproduced by the orchestrator before fixing:

- P2 whitespace padding (`e616666..67ed81c`): the 105-character reference-only marker plus 1,895
  spaces reached the 2,000-character floor, so a failed pass expired the record. The probe printed
  `whitespace-padding: chars=2000 outcome=EXPIRED`. Fixed in `6e0be1f` by counting non-whitespace
  characters; `f087da0` pins the leading space-padded marker. Verdict on `67ed81c..f087da0`:
  APPROVE.
- P2 invisible padding (`f087da0..ea344ca`): U+034F and U+00AD counted toward the floor, and the
  new documentation overclaimed the filter. `ea344ca` filtered those two characters; `0ec523e`
  replaced the list with visible-character categories (`_substantive_length`, which excludes
  whitespace, control, format, combining-mark, surrogate and unassigned code points, and the four
  Hangul fillers). Removing the filter now fails the padded-marker test. Verdict on
  `f087da0..ea344ca`: REQUEST CHANGES; the category fix closed it.
- P3 candidate wording (`f087da0..0ec523e`): the documentation said the first three user or
  assistant messages are checked, while the code counts all non-system messages before filtering
  roles. `3e6b140` changes the wording to the code's rule. Verdict on `0ec523e..3e6b140`: APPROVE.
- P2 short marker below the floor (`3e6b140..9e3e0a7`): the new 1,000-character branch accepted a
  message carrying only the short reference-only marker, 1,100 filler characters and the closing
  suffix, which the documentation said could not qualify. `804540d` requires the host header
  signature (`Earlier turns were compacted into the summary below. This is a handoff from a
  previous`) below the 2,000-character floor. The signed input and the host's real header (1,632
  visible characters) pass; the abbreviated marker stays held. Verdict on `3e6b140..804540d`:
  APPROVE.

Round 5 also measured the host header at 1,954 characters (1,632 visible) and confirmed that the
metadata tag never reaches the middleware on the Chat Completions path, because the transport strips
underscore-prefixed keys before the middleware runs; that is why the text floor has to admit host
summaries without the tag.

## Verification run by the orchestrator

- `python3 -m pytest -q` on `61a9e1c`: `705 passed in 14.08s`.
- `hermes plugins validate .`: `Validation passed.`
- Static scan `scripts/static-scan.sh 4f4ebbf`: cccc exit 0, lizard exit 0, duplicate rate 2.72%.
- Mutation checks in a clone at `~/Trash/hermes-review/round3/check`, one fix reverted at a time:
  disabling hit revalidation failed the moved-skill test, dropping the miss memo failed the
  failed-lookup test, forcing an empty roots digest failed the profile-scope test, and restoring the
  two files from `5a86e39` failed both window tests.
- Live one-shot turn on the deployed tree: decision lines with `total_ms` 615.0, 559.0, 545.8 and
  519.2, against 19.0 to 23.75 s before the fix. Nine request dumps from that session each carry one
  system message starting `Skills routed for this request` at the last index and no `## Skills`
  catalog, so the block reaches the wire for its own request and never returns in the next
  request's payload.

At `e285d3c`, after the round-4 and round-5 fixes: `python3 -m pytest -q` printed
`711 passed in 14.18s`; `hermes plugins validate .` printed `Validation passed.`;
`scripts/static-scan.sh fd8dff2` reported cccc exit 0, lizard exit 0 and a duplicate rate of
3.08%, with every duplicate group in unchanged test setup. The round-4 and round-5 mutation checks
all fail with their fix reverted in a scratch copy: the 2,000-character floor, the non-whitespace
count, the invisible categories, the header signature, the merged shape, the delimiter iteration,
the tool role, the BOM strip, the 160-character merged anchor, and each of the four
default-ignorable fillers on its own.

## Dispositions

All five findings are fixed in `61a9e1c`, each with a regression test that fails when the fix is
reverted. Round 1's open question about the block reaching later requests is settled by the live
request dumps: it does not.

Round 3 also filed two P2s. The first: two compactions whose summary text is byte-identical cannot
be told apart by a digest, so the second window could keep the old record. The reviewer did not
observe the host emitting identical summary text, and the host accepts the summarizer's returned
text without comparing it with the previous summary, so the text may repeat and the record then
keeps the catalog dumped; the limitation is stated in the documentation as an input condition
rather than fixed. The second: `_declared_name` read the first duplicate `name` key while the
host's parser keeps the last; `b00a075` mirrors the host (last key, directory-name fallback, BOM
stripped) and `e616666` adds a warm-cache replacement test.

The reviewer's residual note that a pasted complete summary at the head of the history is
indistinguishable from a host summary is also documented as an input condition; `bd49354` narrows it
further by requiring a summary without the host metadata tag to be at least 2000 characters, so a
short pasted marker cannot expire the record, and `67ed81c` pins the leading short-paste case as a
test. A message that merely quotes the opening marker or the closing suffix does not expire the
record.

The round-3 reviewer closed its report on the frozen commit and, by source reading of the later
checkout, asked whether a full reference-only prefix plus suffix with no host metadata still
qualifies. `bd49354` answers that: without the metadata tag a text summary must also be at least
2000 characters, so a short pasted marker is rejected, and `67ed81c` pins the leading short-paste
case. A complete long paste at the head of the history remains the documented input condition.

Round 4's first four findings are fixed in `401475f`, and its delta re-read on
`e616666..9e3e0a7` filed three more, fixed in `fd8dff2` after `804540d` had already replaced the
paragraph the third one quoted. Its last delta, `fd8dff2..e285d3c`, APPROVED the filler-coverage
fix with no new findings. Round 5's findings are fixed in `6e0be1f`..`804540d` and approved at
`3e6b140..804540d`. Every finding from both reviewers is closed with a regression that fails when
the fix is reverted, and no code change is unreviewed.
