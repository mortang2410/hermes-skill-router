# Review round 15: approval after the T8 entry and T3 timeout wording

Reviewer: the same independent luna max child. Scope: the 6000 ms default, T8's hidden tab plus the
no-op dashboard entry, and the two wording corrections that followed.
Verdict: APPROVE, no open finding.

## Findings closed in this round

- R15-1: hiding a dashboard tab does not stop the client from requesting the manifest's default entry.
  `web/src/plugins/usePlugins.ts` injects every manifest's script and `hermes_cli/web_server_dashboard.py`
  defaults `entry` to `dist/index.js`, so a plugin without that file produced a console error. Hermes is
  not patched, because the design forbids it: the plugin ships `dashboard/dist/index.js`, a comment-only
  module that registers nothing, and a test proves it is valid JavaScript with no global side effects.
- R15-2: T3 and the library docstring now say the 6000 ms timeout bounds each 16-row chunk, with the
  measured 1.2 to 2.4 s per chunk, while the whole 27-request build takes about 38 seconds.
- R15-3: the live profile's 60000 ms override from T5's probe was reverted to the new 6000 ms default,
  so a dead endpoint cannot stall a turn for a minute.

## Consequences

The document set is approved again. All eight tickets are implemented; phase 4 code review and the
phase 5 verification, commit, push and completion gate remain.
