# code-01: independent review of hermes-skill-router

Scope: every changed code file and its tests, the plugin surface files, and the intended behaviour in
`docs/work/router-sync/` (spec D1-D12, skeleton C1-C9, tickets T1-T8). I did not write this code and did
not modify it. This is a reading of the listed files plus the reproductions recorded below; it is not
proof of correctness, and it does not replace a live restart or a provider-side traffic test.

Probes live in `/home/wilder/Trash/router-review/` (final/ holds the probes and logs named here).

## Reviewed revision

- Branch `main`, HEAD `e1eff17 chore: initialise hermes-skill-router`, working tree dirty: `README.md`
  modified and `__init__.py`, `dashboard/`, `desktop/`, `docs/`, `hermes_skill_router/`, `plugin.yaml`,
  `pytest.ini`, `service/`, `tests/`, `tui/` untracked. The review therefore covers the working tree, not
  the commit.
- The tree was being edited by another agent while I read it. I re-checked before writing this: the
  newest code mtime is `tests/test_display_surfaces.py` at 08:30:42, the newest doc is
  `reviews/11-round15-approval.md` at 08:36, and no reviewed source changed after my read of it. Hashes
  below are the revision these verdicts apply to; re-check them before acting on this review.

| file | sha256 (first 16) | mtime |
| --- | --- | --- |
| hermes_skill_router/catalog.py | 420a1b548a37e16d | 07:46:34 |
| hermes_skill_router/config.py | 4aebd56da4a3fc16 | 08:08:48 |
| hermes_skill_router/display.py | 66b76c1d87901ab9 | 07:15:34 |
| hermes_skill_router/embed.py | aa84de308b8083ee | 06:20:20 |
| hermes_skill_router/inject.py | 75c5182ff6a882d9 | 07:02:02 |
| hermes_skill_router/library.py | eb3d16d785270ed2 | 08:29:54 |
| hermes_skill_router/rerank.py | 8cf2143e43ea4067 | 06:56:59 |
| hermes_skill_router/router.py | 3e85d9d927ac6e33 | 07:11:48 |
| hermes_skill_router/select.py | e4af5e36220c71c7 | 07:02:09 |
| hermes_skill_router/__init__.py | 23d6662495a10851 | 05:48:35 |
| __init__.py | 5ef88065f69c1713 | 08:04:17 |
| service/embed_server.py | 47e6e33640e826a6 | 05:46:59 |
| service/__init__.py | 923702eb1e7f4389 | 05:46:59 |
| dashboard/plugin_api.py | 3d904d522f434866 | 08:10:34 |
| dashboard/dist/index.js | 664eee9697934326 | 08:29:40 |
| desktop/plugin.js | 154c9b5577b9aa6a | 08:09:50 |
| tui/skill-router.mjs | d7bfaa7d805e2804 | 08:12:28 |
| plugin.yaml | c739f6518358dada | 08:08:48 |

Tests: 15 files (`tests/fake_endpoint.py` plus 14 `test_*.py`), newest `tests/test_display_surfaces.py`
`95154e6517ec9f48`. Config and surface files: `pytest.ini` `94fdc2d545b2d407`, `.gitignore` `a2c5d1b1e8fc2d00`,
`service/hermes-skill-router-embed.service` `4c40812ffd577915`, `dashboard/manifest.json` `c6be265bac6c14aa`.

## Repo checks

Run in `/home/wilder/git-repos/hermes-skill-router`, exit codes as shown.

1. `python3 -m pytest -q`, exit 0:
   ```
   ........................................................................ [ 91%]
   ..................................                                       [100%]
   394 passed in 3.85s
   ```
   (The earlier read of 391 tests predates three tests the concurrent editor added; the tree as
   reviewed has 394 and all pass.)
2. `hermes plugins validate .`, exit 0, tail:
   ```
   ✓ built-in tool collisions — no tools to check
   ✓ security scan — safe
   ✓ no core override — no runtime rebinds of Hermes core
   ✓ desktop surface — stays inside the plugin SDK surface
   ◆ Isolation — runs in the plugin host (plugins.isolation: host)
     · dashboard backend API is served through the plugin host (buffered responses)

   Validation passed.
   ```

## Static scans

Same 29 files: 12 source files, `__init__.py`, `dashboard/plugin_api.py`, `tests/fake_endpoint.py`, and the
14 `tests/test_*.py`.

- `~/.local/bin/cccc <29 files>`, exit 0: 29 files, 532 functions, 0 parse errors. Cognitive sum 725,
  max 49, median 0, p90 4, p95 7. Cyclomatic sum 1114, max 29, median 1, p90 4, p95 6. Hot functions
  (cognitive/cyclomatic): `config.py:_coerce` 49/29 at 169, `library.py:_read_cache` 27/25 at 207,
  `rerank.py:_scores` 24/18 at 99, `router.py:_compose` 23/22 at 121, `embed.py:_vectors` 19/15 at 151,
  `library.py:_embed_rows` 19/12 at 146, `dashboard/plugin_api.py:_display_module` 18/15 at 88,
  `catalog.py:_parse_block` 17/8 at 149, `catalog.py:_partition` 16/12 at 170. The 29-file aggregate is
  dominated by the test files; no single test function is hot.
- `~/.local/bin/lizard <29 files>`, exit 1 by design for this tree (warnings only): 4 warnings for
  cyclomatic complexity over 15. `_coerce` CCN 29, NLOC 59, `config.py:169-228`; `_read_cache` CCN 25,
  NLOC 51, `library.py:207-258`; `_scores` CCN 18, NLOC 31, `rerank.py:99-132`; `_compose` CCN 22,
  NLOC 83, `router.py:121-212`. Totals 5803 NLOC, average CCN 2.2, 502 functions, warning count 4.

Both scanners agree on the four hot spots. They are the parse and validation layers; the complexity is
in branches over input shapes, and the code has tests for those shapes, so I record this as a
maintainability cost rather than a defect. `_coerce` at 49 cognitive is the one I would split first.

## Per-file verdicts

PASS means "no defect found in this reading plus the reproductions listed"; it is not a proof.

- `hermes_skill_router/catalog.py` - FINDINGS 1, 3 (variant not surfaced; warnings repeat per request)
- `hermes_skill_router/config.py` - FINDINGS 3 (notes are produced but no caller logs them)
- `hermes_skill_router/inject.py` - FINDINGS 1 (one fixed header for both catalog variants)
- `hermes_skill_router/router.py` - FINDINGS 1 (renders without the stripped variant; call site at 198)
- `hermes_skill_router/library.py` - PASS
- `hermes_skill_router/embed.py` - PASS
- `hermes_skill_router/rerank.py` - PASS
- `hermes_skill_router/select.py` - PASS
- `hermes_skill_router/display.py` - PASS
- `hermes_skill_router/__init__.py` - PASS
- `__init__.py` - FINDINGS 3 (`_route` drops `cfg.notes`)
- `service/embed_server.py` - PASS
- `service/__init__.py` - PASS
- `service/hermes-skill-router-embed.service` - PASS
- `dashboard/plugin_api.py` - FINDINGS 3 (also drops `cfg.notes`; the six-field projection itself is correct)
- `dashboard/manifest.json` - PASS
- `dashboard/dist/index.js` - PASS (comment-only, as the test now asserts)
- `desktop/plugin.js` - FINDINGS 2 (records `turn_id`/`at` never reach the pane)
- `tui/skill-router.mjs` - FINDINGS 2 (same, and `face()` paints no caption)
- `tests/fake_endpoint.py` - PASS
- `tests/test_catalog.py` - PASS (gap: no one-shot variant case, see finding 1)
- `tests/test_config.py` - PASS
- `tests/test_display.py` - PASS
- `tests/test_display_surfaces.py` - PASS (gap: asserts the record carries ids, not that a surface shows them, see finding 2)
- `tests/test_embed_client.py` - PASS
- `tests/test_embed_server.py` - PASS
- `tests/test_inject.py` - PASS (gap: asserts the header text, not its fit to the one-shot prose, see finding 1)
- `tests/test_library.py` - PASS
- `tests/test_middleware_contract.py` - PASS
- `tests/test_plugin_registration.py` - PASS (the session/turn id path through the real dispatcher is proven)
- `tests/test_readme.py` - PASS
- `tests/test_rerank_client.py` - PASS
- `tests/test_rewrite.py` - PASS
- `tests/test_select.py` - PASS
- `README.md` - FINDINGS 2 (line 486), 3 (lines 235-236); otherwise consistent with the code
- `docs/work/router-sync/spec.md` - FINDINGS 1 (D3 "carries the guidance the removed section provided"), 2 (D12 "looks like on screen")
- `docs/work/router-sync/tickets/T4-closed-loop.md` - FINDINGS 1 (item 4)
- `docs/work/router-sync/tickets/T8-display-names.md` - FINDINGS 2 ("the display says which turn it came from")

## Findings

### 1. A one-shot session gets guidance opposite to the prose the plugin removed

Where: `hermes_skill_router/inject.py:48-59` (one fixed `HEADER`), `hermes_skill_router/catalog.py:38`
and `:106-125` (`_find_section` recognises `ONESHOT_LEAD` but the result carries no variant),
`hermes_skill_router/catalog.py:63-69` (`Partition` has no variant field), `hermes_skill_router/router.py:198`
(`render(chosen, partition.kept, index, cfg)`).

What breaks: with `HERMES_SINGLE_QUERY_SESSION=1` the real renderer emits a different `## Skills` prose
that forbids loading general process skills and forbids creating or editing skills. The plugin strips
that prose and injects the same header it uses for a normal turn, which tells the model to load a skill
when one is "even partially relevant" and to "load one that defines how a task should be done here even
when you already know the basics". In the one-shot mode the model is now pushed toward exactly what the
removed prose forbade, and nothing replaces the "do not create or edit skills" rule that the header's
own docstring says it deliberately omits.

Reproduced on this revision (`/home/wilder/Trash/router-review/final/repro_header.py`):

```
oneshot prompt 74249 routed/kept 715 0 block 661
oneshot restriction in stripped: False | in block: False
oneshot block says load even when you know basics: True
normal  prompt 75008 ... same block text
```

The removed sentence, from the real one-shot prompt: "Do not load general process skills (testing,
debugging, review methodology) for work you already know how to do, and do not create or edit skills:
this is a one-shot run with no later session to reuse them."

Docs broken: spec D3 ("behind a fixed header that ... carries the guidance the removed section
provided") and tickets/T4 item 4 ("it carries the guidance the removed `## Skills` prose provided").
Both variants exist in the live renderer, so one fixed header cannot satisfy that sentence for both.

Smallest fix: return the variant from the strip (`Partition.variant` or the one-shot restriction
sentence itself) and append that sentence to the header in `inject.render` when the removed section was
the one-shot variant. That is one field plus one string; the block then carries the guidance of the
prose it replaced. Add a test that strips the real one-shot prompt and asserts the restriction sentence
is in the rendered block.

### 2. Neither display surface shows the session or turn identity it stores

Where: `desktop/plugin.js:88-112` (`summary`) and `:128-167` (paint passes heading, names, note only);
`tui/skill-router.mjs:95-114` (`summary`) and `:143-156` (`face`). `tui/skill-router.mjs:62-85`
(`readRecord`) keeps the fields, so only the paint drops them.

What breaks: a record written for one turn paints identically to a record a second session or a fork
wrote, on both surfaces. The docs say the reader can tell which turn a record came from; the code that
writes the record forwards `session_id`, `turn_id` and `at`, and both surfaces then throw them away.
The data exists at the paint call site, so this is a display gap, not a data gap.

Reproduced on this revision with the real modules loaded under node (SDK stubs match the real
signatures, `ui-tui/src/sdk/host.tsx:84` and `registry.ts:7`; `/home/wilder/Trash/router-review/node/probe.mjs`),
record `{at: "2026-11-10T07:20:31Z", session_id: "session-abc", turn_id: "turn-42", status: "routed", names: ["alpha"]}`:

```
desktop summary -> {"count":1,"heading":"Skills routed (1)","names":["alpha"],"note":"","status":"routed"}
tui     summary -> {"count":1,"names":["alpha"],"note":"","status":"routed","title":"Suggestions"}
session-abc: desktop false, tui false
turn-42:     desktop false, tui false
07:20:31:    desktop false, tui false
```

Docs broken: README.md:486 ("carries the session and turn ids ... visible on screen when the ids are
present"), tickets/T8 (Notes: "the display says which turn it came from"), spec D12 ("what a second
session or a fork replacing it looks like on screen").

Smallest fix: return `turn_id` (and `at`, if the caption should carry the time) from both `summary`
functions and paint one muted caption line, for example `via turn-42`, in the desktop pane body and in
the TUI dialog title or note. Extend the surface test to assert the identity appears in the painted
summary, which is the property the test misses today.

### 3. A bad configured name warns on every request, and config notes are never shown

Where: `hermes_skill_router/catalog.py:190-194` (name in both lists) and `:208` (unknown name) call
`logger.warning` on every partition; `hermes_skill_router/config.py:121` (`notes`) is filled by `load`
but dropped by `__init__.py:82-86` and by `dashboard/plugin_api.py`'s route.

What breaks: a typo in `eagerSkills` or `deferredSkills` writes the same warning for every request until
someone fixes the config, which floods `agent.log` on a busy profile and buries the first occurrence.
Separately, a setting that `load` rejected or an unknown key is collected in `Config.notes` and then
thrown away, so the operator's only signal is outside the plugin's log; the README tells them the
plugin reports these names in one warning and there is no path by which it does.

Reproduced on this revision with the real rendered prompt
(`/home/wilder/Trash/router-review/final/repro_warn2.py`; three requests, `eager=["not-a-skill"]`,
`deferred=["alpha"]`):

```
WARNING configured skill 'not-a-skill' is not in the catalog; ignoring it
WARNING configured skill 'alpha' is not in the catalog; ignoring it
(repeated for the second and third request: 6 lines total)
cfg.notes count 3
```

and `load({"shortlistSize": 5000, "notAKey": 1, "embedBaseUrl": "http://example.com"})` yields three
notes (`shortlistSize` out of range, unknown key, unsafe endpoint) that no caller logs.

Docs broken: README.md:235-236 ("the plugin logs one warning" / "reported in one warning"). The
while-process convention the same README states at line 438 for failure reasons is the behaviour the
rest of the package already implements (`_log_once` in `library.py` and `router.py`).

Smallest fix: route both warnings through a module-level once-per-name `_log_once` keyed like
`library._log_once(f"skill-name:{name}", ...)`, so each distinct configured name logs once per process,
and log `cfg.notes` once in `__init__._route` (and surface them on the dashboard route) at the same
level the plugin uses for its own config problems. Add a test that partitions twice and asserts one
warning record per name.

## Cleared after reproduction

These were suspected and are not findings; each is listed with what settled it.

- Session and turn ids never reaching the record: the live record at 07:38 with empty ids predates the
  current `__init__.py` (mtime 08:04). The current path forwards both, and
  `tests/test_plugin_registration.py` exercises it through the real dispatcher.
- Strip correctness: the real 75,008-character live prompt becomes `STABLE + "\n\n" + MEMORY` exactly,
  with 715 routed rows and no residue; the one-shot and demoted-category variants also strip correctly,
  including the hidden note and 4-part placement fuzz (normal/one-shot x note x position), 0 failures.
- Payload preservation and provider fit: `model`, `tools` and unrelated keys survive; the appended
  system message after the user message is the documented design (C9, T5) and the live provider
  accepted it; redirects 301/302/303/307/308 are refused rather than followed.
- Cache trust and safety: fingerprint, dimension shape, count, per-row types and vector widths are
  re-validated on read; the write is atomic with `fsync` and `os.replace`; a cache miss rebuilds; no
  request text appears in `index.json`, the record or `embed.log` (grepped for the live query, 0 hits).
- Dashboard route: projects exactly `schema, at, session_id, turn_id, status, names` and never request
  text.
- Desktop and TUI SDK fit: `PANES_AREA`, `DisclosureCaret`, nanostores `atom`, `ctx.rest`, `ctx.onEvent`,
  and the broadcast name `plugin.hermes-skill-router.suggestions.updated` all exist in the installed
  Hermes; `updateWidget<S>(app, fn)` takes the app object the widget passes.
- `dashboard/dist/index.js`: exists, is comment-only, would be rejected as a classic script if it used
  module syntax, and a test now guards it.
- Live end-to-end run: 8012 + 8010 with the real catalog produced the block (1233 characters), the
  record (8 names, `status: routed`) and a 70,743 to 22,657-character system message with the catalog
  gone; request dumps at 07:27 confirm the on-wire shape.
- Cold build and failure modes: 33.7 s cold build in 45 chunks is within the documented ~38 s; endpoint
  error, timeout, 503 health and malformed bodies fail open (block absent, catalog kept) and log once.

## Limits of this review

- Read-only: no source file was changed, so the three fixes are described, not applied.
- The display surfaces were exercised by loading the repo's real modules under node with SDK stubs that
  match the installed signatures, not by looking at a running desktop pane or TUI.
- No Hermes restart and no provider-side traffic test were performed; the live checks above used the
  running services on 8010/8012 and the request dumps from the user's own session.
- `README.md` claims were read at the lines cited; the one-shot prose in finding 1 is the live
  renderer's output at this revision of the installed Hermes (`v0.22.2+1671`), not a hand-written copy.
