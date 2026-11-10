# Review round 4

Reviewer: the same luna max child that reviewed round 3, read-only, whole current set. Verdict:
`VERDICT: CHANGES`, six findings. All six were reproduced against the files and fixed in one pass.

## Dispositions

1. The overview said the index covers exactly the skills Hermes listed, while the detailed rule limits
   it to `Partition.routed`. Accepted: the Solution now says the index is built from the routed rows
   only, and eager rows and rows outside a non-empty deferred list are never suggested from it. T3
   already tested that an eager row is absent from the index; the new end-to-end criteria in T5 cover
   the deferred-excluded row.
2. `maxSuggestions` was described as a cap on all injected rows, which contradicts rendering the kept
   rows. Accepted: the config row now says it caps suggested rows, `0` selects none while kept rows are
   still listed, and C8/T4 state the same. T4 and T5 gained checks for the zero case.
3. The display record carried every injected name under `status: routed`, so a pinned or
   deferred-excluded skill could appear as a suggestion. Accepted: the record now carries the selected
   suggestion names only, kept rows are never recorded, and `routed` with an empty list means the turn
   suggested nothing. T8 paints only those names and says so.
4. M8 could not legally import the `Config` and `CatalogRows` its own contract needed. Accepted: M8's
   allowed dependencies are now M2, M3 and M7, and T4's signature is `render(rows, kept, cfg)`.
5. M9's allowlist omitted M11, although C9 writes the record through it. Accepted: M11 is now in M9's
   allowed dependencies; M11 still forbids depending on M9, so the edge stays one way.
6. R11 was traced only through the partition, with no proof that a skill excluded by a non-empty
   deferred list survives a real strip. Accepted: the trace row now names C8 and C9 plus T4 and T5, and
   T5 has an end-to-end criterion with one eager skill, one deferred skill and one skill outside the
   deferred list, checking that the third appears under `Also listed:` and none of the two kept skills
   lands in the record.

## Confirmed

- The no-routed-rows path stays inert and leaves the original section.
- The T2 to T5 handoff on `router.py` is acyclic, and T5 does not take over `__init__.py`.
- T5 owns the record writer and T8 only reads and paints.

## Not verified

Read-only document pass: no runtime tests, no live endpoint behavior, no desktop or TUI interaction.
