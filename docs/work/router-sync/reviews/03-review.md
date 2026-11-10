# Review round 3

Reviewer: luna max child, read-only, whole document set (spec, skeleton, T1 to T8, both earlier review
reports, STATE). Verdict: `VERDICT: CHANGES`, ten findings. All ten were reproduced against the files
and the installed trees, accepted, and fixed in one pass.

## Dispositions

1. P1, middleware signature. T2 claimed the dispatcher filters callback arguments through
   `inspect.signature`. Reproduced: `hermes_cli/plugins_dispatch.py:591-602` calls `cb(**kwargs)` with
   no filtering, and `agent/turn_api_request.py:143-148` passes `request` plus ten context keys; a
   narrow signature raises and the dispatcher skips it silently. T2 now requires
   `def on_llm_request(request, **kwargs)`, and its registration test drives the real
   `apply_llm_request_middleware` with the full keyword set plus a negative case for a narrow callback.
2. P1, record writer dependency gap. C9 required the rewrite to write through M11, while only T8 owned
   `display.py` and T8 is blocked by T5. Fixed by moving the writer, `hermes_skill_router/display.py`
   and `tests/test_display.py` into T5; T8 now owns only the dashboard, desktop and TUI surfaces and
   `tests/test_display_surfaces.py`.
3. P1, `## Skills` prose removal. Reproduced: `agent/prompt_builder.py:1395-1420` emits the heading and
   the guidance outside the tags. C3 now removes the whole section (heading, prose, tags and rows, or
   the single-query variant), the block lists the eager rows so a pinned skill never disappears, and a
   request with an empty block keeps the section byte for byte. The spec, skeleton C3/C8/C9, T2, T4 and
   T5 carry the same rule.
4. P1, redirects. Reproduced with a live probe: the default urllib opener followed a POST to a 302 and
   fetched the target. T3 and T4 now require a redirect-refusing opener, HTTP 3xx counted as a failure,
   and a fake-endpoint test that asserts no follow-up request, including a cross-host `Location`.
5. P1, T6 ownership. `src/index/embedding-index.ts` is now in T6's exclusive files, and the live profile
   switch names the exact row and key: `id: skill-router` in
   `/home/wilder/.dsh/profiles/web/cordis.patch.yml`, with `embedBaseUrl` added beside `modelPath`,
   edited by the orchestrator after the code lands.
6. P2, T2 and T5 overlap. Resolved as a declared sequential handoff: T2 keeps `router.py` with the
   identity body and keeps `hermes_skill_router/__init__.py`, which already imports `rewrite`; T5
   replaces only the identity body and no longer lists `__init__.py`.
7. P2, C4 duplicate names. C4 now returns an index whose length is the number of distinct routed names,
   first occurrence wins, and C3 keeps duplicates in `routed` for C4 to collapse, matching T3.
8. P2, traceability and README check. R5 now names T2, T3, T4 and T5; R6 names T1, T3, T4 and T6; R7
   names T2 and T7; R12 names T5 and T8. T2, T3 and T4 gained the S9 and R11 coverage their work
   carries. T7's test now fails on a README row with an empty bound or effect cell, and its acceptance
   line says so.
9. P1, desktop toggle message. Reproduced: a disabled desktop plugin never calls `register()`
   (`apps/desktop/src/contrib/runtime-loader.ts:389-395`), so the pane cannot report its own toggle.
   T8 now says the pane is absent when that toggle is off, reports only the missing backend route, and
   leaves the second gate to the README.
10. P2, STATE contradictions. STATE now has one current round and one next action: round 4 approval,
    then phase 3.

## Notes

- The reviewer confirmed the ten-contract cap is met, the display delta is carried, the DSH one-text
  contract is real, and the desktop pane and TUI widget seams exist.
- Not verified in this document-only pass: live 8010/8012 reachability, unit activation, actual desktop
  or TUI interaction, the trailing second system message (U1), and the cache and catalog probes U2/U3.
