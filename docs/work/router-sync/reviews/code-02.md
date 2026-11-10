# code-02: verification of the three code-01 fixes

Second round. The three code-01 findings are fixed in the working tree; this verdict checks each fix
against that tree instead of the fix reports. Probes are in `/home/wilder/Trash/router-review/round2/`
(`probe_oneshot.py`, `probe_log.py`, and `../node/probe2.mjs`, which loads the current
`desktop/plugin.js` and `tui/skill-router.mjs` under node with SDK stubs matching the installed
Hermes). Read-only: no source file was changed by this review.

## Reviewed revision

HEAD `e1eff17`, dirty worktree, as in code-01. Changed since code-01 (all other files keep their
code-01 hashes):

| file | sha256 (first 16) | mtime |
| --- | --- | --- |
| hermes_skill_router/catalog.py | c141e880523b6c4a | 08:54:00 |
| hermes_skill_router/config.py | 52f43f3022806962 | 08:46:21 |
| hermes_skill_router/inject.py | 622ca6bf7c96c459 | 08:54:00 |
| hermes_skill_router/router.py | 15b9f737e2d458ff | 08:54:00 |
| desktop/plugin.js | dabf723c8c101df7 | 08:44:38 |
| tui/skill-router.mjs | 18840a743caacca3 | 08:44:38 |
| README.md | 869a51b8adba0050 | 08:55:48 |
| tests/test_catalog.py | 10ef045823adc41e | 08:54:13 |
| tests/test_config.py | 743db50da1edf1fe | 08:44:37 |
| tests/test_inject.py | 593d8973bfb175d7 | 08:54:13 |
| tests/test_rewrite.py | eb85fbd60ec2ffc9 | 08:54:13 |
| tests/test_display_surfaces.py | d55d65610df3c623 | 08:46:21 |
| tests/test_readme.py | 06df8cba927fabb8 | 08:54:49 |
| tests/test_plugin_registration.py | a9e5743ece18bf81 | 08:46:10 |

The mtimes above were stable through the verification window (re-checked at 09:01 local). No other
file differs from code-01.

## Repo checks

1. `python3 -m pytest -q`, exit 0:
   ```
   ........................................................................ [ 88%]
   ................................................                         [100%]
   408 passed in 3.99s
   ```
   The count grew from 394 to 408 with the new tests; all pass.
2. `hermes plugins validate .`, exit 0, tail:
   ```
   ◆ Isolation — runs in the plugin host (plugins.isolation: host)
     · dashboard backend API is served through the plugin host (buffered responses)

   Validation passed.
   ```

## Static scans

Changed Python and tests (the brief's scope), `cccc` exit 0: 11 files, 263 functions, 0 parse errors.
Cognitive sum 401, max 49, median 0, p90 4, p95 7. Cyclomatic sum 570, max 29, median 1, p90 5,
p95 6. Hot functions (cognitive/cyclomatic): `config.py:_coerce` 49/29 at 169, `router.py:_compose`
23/22 at 122, `catalog.py:_parse_block` 17/8 at 161, `catalog.py:_partition` 16/12 at 182,
`inject.py:render` 13/11 at 84, `config.py:load` 10/8 at 244. `inject.render` added one branch (the
header choice) and stayed at 13.

`lizard` over the same 11 files, exit 1 by design (warnings only): 2 warnings.
`_coerce` CCN 29, NLOC 59, `config.py:169-228`; `_compose` CCN 22, NLOC 83, `router.py:122-214`.
Totals 3360 NLOC, average CCN 2.2, 253 functions, warning count 2.

Full-tree context (all 29 files, same set as code-01): 548 functions, 0 parse errors, cognitive sum
755 (was 725), max 49, p95 6; cyclomatic sum 1154 (was 1114), max 29. `lizard` reports the same four
warnings as code-01: `_coerce` 29, `_read_cache` 25, `_compose` 22, `_scores` 18. The two new
branches did not create a new hot spot.

## Verification of the three fixes

### Fix 1, one-shot header: closed

Re-read `catalog.py` (`Partition.one_shot`, `_find_section` returns the variant from
`content[heading_at:open_at].startswith(ONESHOT_LEAD)`), `inject.py` (`ONE_SHOT_HEADER`, `render(...,
one_shot=False)`), `router.py` (`INJECTED_HEADER = HEADER.split("\n", 1)[0]`, render call at 200
passing `partition.one_shot`).

Reproduced against the real renderer, both modes, with the current code
(`python3 /home/wilder/Trash/router-review/round2/probe_oneshot.py`):

```
one_shot one_shot flag: True | header used: ONE_SHOT
  restriction in block: True | create/edit ban: True
  invitation in block: False | first line shared: True
normal one_shot flag: False | header used: NORMAL
  restriction in block: False | create/edit ban: False
  invitation in block: True | first line shared: True
```

The one-shot block now carries "Do not load general process skills ... do not create or edit skills"
and does not carry the normal invitation; the normal prompt is unchanged. The live one-shot lead
("## Skills\nScan the skills below") and the live normal lead ("## Skills\nBefore replying, scan the
skills below") are distinct, so the variant test cannot misfire on this renderer.
`tests/test_rewrite.py` covers both directions through `rewrite()` and pins the double-injection
guard for either header from the shared first line (`test_the_injection_guard_recognises_either_header_from_its_first_line`,
`test_an_already_injected_one_shot_block_is_left_alone`). README now documents both headers
(lines 40-63). The code-01 finding is closed.

### Fix 2, display caption: closed in code, documentation incomplete (finding 2)

Re-read `desktop/plugin.js` (`shortTime`, `captionFor`, `summary` carrying `caption`, paint at
`caption ? jsx('p', ...)` before the list) and `tui/skill-router.mjs` (same three functions, `face`
passing the caption as the first `Dialog` child).

Reproduced with the current modules under node (`node /home/wilder/Trash/router-review/node/probe2.mjs`):

```
routed  desktop caption= "turn turn-42 · 14:20" tui caption= "turn turn-42 · 14:20"
        has turn-42: desktop true, tui true | has session-abc: false
inert   desktop caption= "" tui caption= "" | has turn-42: false, false
no-ids  desktop caption= "" tui caption= ""  (record without turn_id/at)
captionFor(inert)= "" ""   captionFor(null)= "" ""
```

So the caption appears for routed records only, carries the turn id and the reader's clock time, and
an inert or id-less record paints no stale identity. The caption is painted in both surfaces, not
just returned: the desktop pane renders it as a `<p>` before the name list, and the TUI passes it as
a child of `Dialog`, whose `Box` is `flexDirection="column"` (`ui-tui/src/components/overlay.tsx:76-108`),
so it stacks above the list. The route still projects only the six record fields, and the caption is
built from `turn_id` and `at` alone. The behavioural finding is closed.

What is not closed is the prose: README, spec D12 and T8 still describe the surfaces without the
caption, and README still does not say the config notes are logged. See finding 2.

### Fix 3, warnings and config notes: closed, with one new disclosure (finding 1)

`catalog.py` now routes both warnings through `_warn_once(key, ...)` with a module-level `_LOGGED`
set; `config.py` gained `report_notes`, called at the end of `load`, which logs each distinct note
once per process.

Reproduced with the real rendered prompt and three requests
(`python3 /home/wilder/Trash/router-review/round2/probe_log.py`):

```
catalog warnings after 3 requests: 2
   configured skill 'not-a-skill' is not in the catalog; ignoring it
   configured skill 'alpha' is not in the catalog; ignoring it
config notes after 3 loads: 3
   hermes-skill-router config: embedBaseUrl: 'http://example.com' is not a safe endpoint URL ...
   hermes-skill-router config: shortlistSize: 5000 is outside 1..383; using default 50
   hermes-skill-router config: notAKey: unknown config key ignored
```

One warning per configured name regardless of request count, and each note reported once on first
`load`. `tests/test_catalog.py` (`test_a_repeated_request_warns_once_per_unknown_name`,
`test_a_repeated_request_warns_once_for_a_name_in_both_lists`) and `tests/test_config.py`
(`test_a_rejected_bound_logs_its_note_once`) pin both behaviours. The code-01 finding is closed.
The new report path does disclose a rejected URL's userinfo; that is finding 1.

## Findings

### 1. A rejected endpoint URL is logged with its credentials intact

Where: `hermes_skill_router/config.py:222-226` builds the note with `f"{spec.key}: {value!r} is not a
safe endpoint URL ..."` and `hermes_skill_router/config.py:235-241` (`report_notes`) now writes that
note to the log.

What breaks: a configured `embedBaseUrl` or `rerankBaseUrl` that carries credentials is rejected by
`is_safe_base_url` precisely because such a URL is unsafe, and the rejection note then echoes the
whole value, userinfo included, into `agent.log` and any log shipping from it. Reproduced:

```
hermes-skill-router config: embedBaseUrl: 'https://alice:hunter2@example.com/v1' is not a safe
endpoint URL (http is allowed only for a loopback host, other hosts must use https, and credentials
are rejected); using default 'http://127.0.0.1:8012'
```

Before this fix the note was collected and dropped, so nothing leaked; the fix makes the note
reachable, and this is the input class most likely to contain a live token (a pasted URL with basic
auth or a query token). The docstring at line 236 promises "A note names a setting, never request
text", which is true but not sufficient: the note can carry the setting's secret.

Docs affected: README:239-240 promises only "appends a note naming the key", so no doc states this
value is logged; the operator has no warning that a bad paste is preserved in the log.

Smallest fix: do not echo the value for `url` notes. Print the sanitized form only, for example
`urlsplit(value).scheme` and `.hostname` with userinfo dropped, or drop the value entirely and say
"embedBaseUrl: not a safe endpoint URL (http only for loopback, https for other hosts, no
credentials); using default ...". Keep the note text (hence the dedupe key) deterministic so it still
logs once per process. Add a test that loads a URL with `user:pass@` and asserts neither appears in
the captured log record.

### 2. The docs do not describe the caption the surfaces now paint, or that notes are logged

Where: `README.md:510` ("lists the names and the count"), `README.md:528-531` ("shows the count in
its title, lists the names one per line, and shows one idle line"), `README.md:239-240` (a rejected
value "appends a note naming the key", with no statement that the note reaches the log),
`docs/work/router-sync/spec.md:142-155` (D12) and `docs/work/router-sync/tickets/T8-display-names.md:132-134`.

What breaks: the code now paints `turn <id> · <local HH:MM>` on both surfaces for routed records, and
logs config notes once, but the README's two surface paragraphs, the spec of record and the ticket
note still read as though the surfaces show only names, a count and an idle line. A reader cannot
learn from the docs what the extra line is, that it is local time rather than the record's UTC
stamp, or that it is suppressed for `inert` and for a record without a turn id. T8:132-134 says "the
display says which turn it came from" without saying how, which is the sentence the caption
implements.

Reproduced by reading: `grep -n "caption" README.md docs/work/router-sync/spec.md
docs/work/router-sync/tickets/T8-display-names.md` returns nothing, while both surfaces paint one.
`tests/test_readme.py` checks the config table and both header strings, not the display prose, so
this cannot regress visibly.

Smallest fix: one sentence in each surface paragraph of README ("and one muted caption line `turn
<id> · <local HH:MM>` when the record is routed"), one clause at README:239-240 ("and logs that note
once per process"), one clause in D12 ("each surface paints the record's turn id and the reader's
local clock time for routed records only"), and one clause in the T8 note replacing "says which turn
it came from" with the caption format.

## Observations, not findings

- The turn id a Hermes dispatcher supplies is `f"{session_id}:{task_id}:{uuid4().hex[:8]}"`
  (`agent/turn_context.py:547-549`), so the caption is long, around 30 to 40 characters. That wraps
  in a 260 px pane; it is a cosmetic choice, not a defect, but if the pane should stay one line the
  caption could show a short suffix of the turn id.
- `session_id` is carried in the record and not painted; the turn id embeds it, so the README claim
  that a second session is visible still holds in practice.
- `_warn_once` and `report_notes` check then add without a lock, matching the existing
  `library._log_once` and `router._log_once` pattern. A race can duplicate one line, never lose one.
- The new tests clear the private `catalog._LOGGED` set. It works, but the module-level state means a
  test that forgets to clear it depends on ordering; a fixture that clears it would be sturdier.

## Per-file verdicts

- `hermes_skill_router/catalog.py` - PASS (variant travels on `Partition.one_shot`; both warnings once per process)
- `hermes_skill_router/config.py` - FINDINGS 1 (note echoes a rejected URL's userinfo)
- `hermes_skill_router/inject.py` - PASS (`ONE_SHOT_HEADER` keeps the restriction, shared first line)
- `hermes_skill_router/router.py` - PASS (passes the variant; guard shared by both headers)
- `desktop/plugin.js` - PASS (caption for routed records only, painted before the list)
- `tui/skill-router.mjs` - PASS (same caption, painted as the first `Dialog` child)
- `README.md` - FINDINGS 2 (surface prose and note-reporting prose stale)
- `tests/test_catalog.py` - PASS
- `tests/test_config.py` - PASS (does not pin the URL redaction, see finding 1)
- `tests/test_inject.py` - PASS
- `tests/test_rewrite.py` - PASS (both variants and the shared-line guard through `rewrite()`)
- `tests/test_display_surfaces.py` - PASS (asserts the caption node and the drop for inert/idle)
- `tests/test_readme.py` - PASS (does not cover the display prose, see finding 2)
- `tests/test_plugin_registration.py` - PASS
- `docs/work/router-sync/spec.md` - FINDINGS 2 (D12 does not describe the caption)
- `docs/work/router-sync/tickets/T8-display-names.md` - FINDINGS 2 (note does not describe the caption)

All other files keep their code-01 verdicts: none changed.

## Cleared after reproduction

- Fix 1 and fix 3 are closed as described above; no further defect in either.
- The one-shot variant detection cannot misread the normal prose at this Hermes revision: the two
  live leads differ on the line after the heading ("Scan the skills below and load one" against "Before replying,
  scan the skills below").
- `render` still returns `""` when there is nothing to show, so strip and injection stay atomic and a
  one-shot request with no selection keeps its catalog.
- The TUI `Dialog` accepts two children and stacks them (column box), so the caption cannot replace
  or hide the accordion; `updateWidget(app, fn)` remains the installed signature.
- Scans and validation show no regression from the two new branches: no new parse errors, no new
  complexity warning, same four hot spots as code-01.

## Limits

- Read-only and static: no live Hermes restart and no provider traffic. The fixes were exercised at
  module level with the real prompt renderer and the real display modules, plus the rewrite tests.
- The credential-echo finding requires a config value that carries userinfo; I did not write such a
  value into the live profile, so the evidence is the captured log record above, not `agent.log`.
- The docs finding is a reading of the current prose against the current code, not a claim that the
  user asked for that prose.
