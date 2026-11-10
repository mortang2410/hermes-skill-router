# Review round 8: T6 delta approval

Reviewer: the same independent luna max child, with no part in writing the ticket or the code.
Scope: only `tickets/T6-dsh-http-embedder.md` after the implementer's two reports, plus the package
layout it references. Verdict: APPROVE, no open finding.

## Findings closed in this round

- R8-1: the ticket listed `src/index/index.ts` as a possible re-export site. That file does not exist;
  the package barrel is `src/index.ts`, it builds the public entry there, and it does not re-export
  `createBgeEmbedder`, so no re-export edit is expected. The exclusive list is corrected and the barrel
  is excluded with the reason.
- R8-2: the four new endpoint keys were not documented anywhere in the package, although its README and
  config sample document `modelPath`. Accepted: the ticket now requires the keys to be documented in
  both places, and requires the non-empty `modelPath` rule to stay stated because suppression is gated
  on it (`src/index.ts:833-842`).

## Evidence the reviewer read

The package's real layout (`src/index.ts`, `src/index/embedding-index.ts`, `src/injection/router.ts`,
`test/http-embedder.test.mjs`), the package README and `cordis.patch.yml`, and the current ticket.
No repository file was written and no runtime test was run by the reviewer.

## Consequences

The document set is approved again after the T6 edit. The implementer may finish the documentation
clause; the live profile switch and its one restart remain the orchestrator's step.
