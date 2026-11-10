# Review round 18: approval after the caption and endpoint-safety wording

Reviewer: the same independent luna max child. Scope: the display caption and names-only wording, D4's
redirect description, and the D4 query/fragment rule that the code review forced.
Verdict: APPROVE, no open finding.

## Findings closed in this round

- R18-1: D4 described the redirect refusal as "their opener's redirect handler raises". It now states
  what the code does: one shared opener whose `redirect_request` returns `None`, so urllib turns a 3xx
  into an `HTTPError` and no follow-up request leaves the process.
- R18-2: D12 named only the routed-only caption, while T8 listed the loading, missing-route and
  read-failure suppressions. D12 now names all four.
- R18-3: several documents still said the display is "names only", which the caption makes imprecise.
  S10, R12, T8's deliverable and TUI step, and T7 now say skill names plus the record's turn id and
  time in a caption, with no descriptions or request text. The remaining names-only phrases describe
  the persisted record, not the surfaces.
- R18-4: the endpoint base URL now rejects a query or fragment, because the request path is appended
  and a query would be swallowed and echoed on failure. The spec states it and the code enforces it;
  the README explains it in the places an operator reads.

## Consequences

The document set is approved again. The code review's four rounds closed every finding, with the
credential redaction verified independently. Phase 5 verification, commit, push, the DSH restart and
the completion gate follow.
