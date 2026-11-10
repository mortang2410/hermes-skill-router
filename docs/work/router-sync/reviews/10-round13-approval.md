# Review round 13: approval after the live findings

Reviewer: the same independent luna max child. Scope: the fixes forced by T5's live run, namely the
shadowed-copy note the live catalog appends after the block, the chunked index build, and the dump path
this Hermes build actually writes.
Verdict: APPROVE, no open finding.

## Findings closed in this round

- R13-1: the live one-shot catalog follows `</available_skills>` with a parenthetical note naming
  shadowed copies `skill_view` cannot load. The strip left that line in the system message. T2 now
  removes every single-line parenthetical note after the block and drops those names entirely, with
  tests built on the live bytes (em dash and hyphen forms).
- R13-2: a cold build of the live 424 routed rows took 19.3 to 42.2 seconds in one request, so the
  2000 ms default made the first live turn inert. T3 now embeds in chunks of at most 16 texts, making
  the timeout a per-request bound; a failed chunk fails the build with no partial index and no cache.
  Measured on the live service afterwards: 27 calls, 37.8 seconds total, 424 vectors of width 1024.
- R13-3: `HERMES_DUMP_REQUEST_STDOUT=1` prints nothing in this build; dumps land in
  `~/.hermes/sessions/request_dump_*.json` under `request.body`. T5's verification says so.

## Consequences

The document set is approved again. T7 and T8 are the remaining tickets; both depend only on T5, which
is done. The live profile currently carries `embedTimeoutMs=60000` from T5's probe; with chunking the
default works, so that override is redundant but harmless.
