# Review round 5

Reviewer: same luna max child, read-only, whole current set. Verdict: `VERDICT: CHANGES`, four
findings, all reproduced and fixed in one pass.

## Dispositions

1. The routed-only index fix missed D9 and T3's summary, which still described the index as built from
   the rows being removed, a set that includes kept rows once the whole section goes. Fixed: D9 now
   names `Partition.routed` and says a kept-row change leaves the fingerprint alone, T3's summary says
   the same, and T3's acceptance checks a kept-only change against a routed-row change.
2. The no-block condition still used "no eager rows" while C8 renders on "no kept rows". Fixed:
   D3, Solution step 4, C9 and T5 now use the kept set, so a deferred list that leaves a non-eager row
   listed causes a strip and lists that row. T5's zero-cap criterion now states that the routed index is
   non-empty, which is the only case where an empty suggestion list still strips.
3. The live proof asked the record to hold "the same names" as the block, while only selected names
   belong there. Fixed: D12 says the record holds the selected suggestion names and never a kept row,
   and T5's live criterion now requires a kept row in the block to be absent from the record.
4. The R11 trace named C2, C4, C8 and C9, but their `satisfies` fields omitted R11. Fixed: all four now
   list R11, matching the traceability row.

## Not verified

Read-only document pass: no runtime tests, no endpoint probes, no live desktop or TUI interaction.
