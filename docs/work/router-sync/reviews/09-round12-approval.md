# Review round 12: approval after the T3 cache-identity delta

Reviewer: the same independent luna max child. Scope: the changes that followed round 10, namely
clip parity, the cache identity that includes full descriptions, the renderer's description source,
and the dependency and requirement rows those touched.
Verdict: APPROVE, no open finding.

## Findings closed in this round

- R12-1: exact DSH text parity was stated unconditionally while both sides have their own
  `maxDescriptionChars` (3..4000). The spec, the config table and T3 now say the texts are identical
  only while both values agree, with 400 on both sides by default.
- R12-2: a suffix-only description edit left the fingerprint unchanged, but cached descriptions feed
  the rerank documents and the injected block. The cache identity now hashes each routed row's
  clipped document text and its full description, so that edit rebuilds. D9, R8 and C4 say so, and
  T3's implementer changed the code and its tests.
- R12-3: C8's renderer had no description source for selected `(name, score)` rows. `render` now takes
  `index` and looks descriptions up by name; M8 may depend on M4; T4 states the lookup and T5 passes
  the index.

## Consequences

The document set is approved again for phase 3. T4 is the next ticket and depends only on T2 and T3,
both done. T5, T7 and T8 follow in order. Any later edit to the spec, the skeleton or a ticket voids
this approval and needs a fresh round.
