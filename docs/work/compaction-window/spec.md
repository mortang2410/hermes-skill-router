# The compaction window opens on the host's event

- Repository: `hermes-skill-router` (branch `main`, HEAD `5345ce9`), plus one ticket in `context-handoff-engine` (branch `master`, HEAD `dda0705`)
- Date: 2026-11-12
- Status: ready for tickets
- Skeleton: `docs/work/compaction-window/skeleton.md`
- Kata spec: `personal` issue `gw6w`, published from this file
- Kata tickets: `personal` issues 01 to 04, children of `gw6w`, in order `085f`, `1516`, `rzfc`, `b3qh`; each has an implementer brief under `docs/work/compaction-window/briefs/`

This spec removes the guess. The router decides when to resend the whole skill catalog by scanning
the conversation for the summary text a host writes when it compacts, and that scan is wrong on the
host it runs on. The host publishes an event when a compaction commits, the router will consume that
event instead, and the text scan goes.

## Problem Statement

A Hermes user loses the router's shortlist for the rest of the session, and nothing tells them.

The router sends the whole skill catalog on the first turn of a compaction window and a short
suggestion block on every turn after it. To know when a window ends it scans the message list for the
summary text the host writes when it compacts a conversation, and it keys a per-session record on
that text's digest. On the live host the scan finds nothing, so the record never expires.

Four defects in that scan are verified against the running install:

- The scan looks at the first three non-system messages, and the host puts its compaction summary
  after the three protected messages, so the first summary of a session sits in the fourth position
  and is never seen.
- The scan's metadata branch is dead: the host's chat transport drops every underscore-prefixed key
  before the middleware runs, so the tag it tests can never arrive.
- The closing-suffix test runs before every other test, so a real summary that ends differently is
  rejected before its prefix is read.
- The digest is a guess at a window identity, and two byte-identical summaries are one identity.

The text scan is not the only option, which is the point. The host fires a `post_compaction` event on
every path that commits a compaction, with the session id, the kind, the reason and the token counts.
The router registers one hook and reads that event. No text is parsed, no marker is guessed, and a
change to the host's summary format cannot break the router again.

Two smaller faults belong to the same window:

- The plugin never declares the hook it needs, so the manifest says `provides_hooks: []` and a host
  older than the event cannot be told from one that has it.
- Both host-importing tests default to `/home/wilder/.hermes/hermes-agent`, a stale source copy with
  41 hooks and no `post_compaction`, so the test suite cannot see the event the design depends on.

## Solution

The plugin registers `post_compaction` and, when a compaction commits, expires that session's dump
record. The record keeps its one job: after a failed pass has already sent the whole catalog in this
window, later failed passes strip the catalog and leave the dropped-catalog marker instead of sending
it twice. A compaction opens a new window, so the next failure may dump once more.

The manifest declares the hook and the host floor: `provides_hooks: [post_compaction]`,
`requires_hermes: ">=0.22.3"`, `version: "0.2.0"`.

The marker constants, the padding-category filter, the visible-length counter, the summary test and
the digest function are deleted, together with their tests. What replaces them is one registration,
one callback, and one dict pop.

The technical documentation describes the event as the window boundary and drops the paragraph that
describes the scan, and its block and pinned-row sentences stop calling block presence a window
boundary, because after this change the window belongs to the event. The README gains what a user
decides with: the measured CPU cost of the rerank stage, and how to write a skill description so the
router picks it. The tests resolve the running install instead of the stale copy, so a hook test runs
against a host that can deliver the hook.

## User Stories

1. As a Hermes user, I want the router to open a new compaction window when the host commits a
   compaction, so that the catalog can be sent again after the host replaced the history.
2. As a Hermes user, I want the window boundary to come from the host's own event, so that a change to
   the host's summary text cannot silently stop the catalog from ever being sent again.
3. As a Hermes user, I want a pasted or quoted compaction summary in my own message to have no effect
   on routing, so that my text cannot widen or narrow what the router does.
4. As a Hermes user, I want two compactions in one session to open two windows even when their summary
   text is identical, so that a repeated summary does not merge two windows into one.
5. As a Hermes user, I want a fork to open a window for both the old and the new session, so that a
   forked session does not inherit a stale record.
6. As a Hermes user, I want a session that compacts and then fails a routing pass to send the catalog
   again once, so that a failure after a compaction still leaves me a full catalog to search by hand.
7. As a Hermes user, I want a second failure in the same window to strip the catalog and leave the
   dropped marker, so that I never read the same catalog twice in one window.
8. As a Hermes user, I want the router to keep routing when the event carries an unexpected payload,
   so that a host change costs me nothing in the turn I am in.
9. As a Hermes user, I want a hook callback that never raises, so that a compaction can never abort a
   turn.
10. As a Hermes user on a host older than the event, I want the plugin's manifest to state the host
    floor, so that an install tool can refuse a combination the plugin cannot serve.
11. As an operator, I want the manifest to declare the hook the plugin registers, so that a host or a
    catalog can tell what the plugin consumes without reading its code.
12. As an operator, I want the plugin version to move to 0.2.0, so that a released floor and a removed
    detector are visible in the version I run.
13. As an operator, I want the two host-importing tests to resolve the install the running `hermes`
    launcher reports, so that the suite tests the host I actually run.
14. As an operator, I want a skipped host contract test to say why it skipped, so that a missing host
    is visible rather than green.
15. As a maintainer, I want the summary-scan tests deleted with the scan, so that no test pins a
    mechanism the code no longer has.
16. As a maintainer, I want one test that drives the record through the event, so that the window
    boundary is pinned by behaviour rather than by a helper call.
17. As a maintainer, I want the technical documentation to name the event as the boundary and to drop
    the scan paragraph, so that the document describes the code I run.
18. As a maintainer, I want a documentation test that pins the new boundary sentence, so that a later
    edit cannot quietly restore the text-scan claim.
19. As a maintainer, I want the engine that rewrites the conversation to fire the same event, so that
    the two halves of this design agree and the window opens on the engine I run.
20. As a Hermes user, I want the request payload to stay byte-identical on every failure path, so that
    a broken endpoint costs me suggestions rather than a corrupted call.
21. As a Hermes user deciding whether to install this, I want the README to say what the rerank stage
    costs on a CPU-only machine, so that I can tell whether my hardware can carry it before I download
    2.2 GB of embedding weights.
22. As a Hermes user, I want that cost stated as a measured number with the model named, so that I can
    check it against my own machine instead of trusting an adjective.
23. As a Hermes user who writes skills, I want the README to say how a description should read for the
    router to pick it, so that a skill I wrote is found by the words I would actually type.
24. As a Hermes user, I want the README to say that tidying a library helps the model pick a skill even
    without this plugin, so that I keep doing it when I turn the plugin off.

## Implementation Decisions

### The window record loses its token

The dump record stays a per-session fact with one lock, and it stops carrying a summary token. It
becomes a membership set: a session is in the set while the whole catalog has already gone out in its
current window. Writing it, reading it and expiring it are three small operations on that set.

The callback that expires it is registered from the plugin's own `register(ctx)` call, next to the
middleware and the tool. The host calls hook callbacks with the event's fields as keywords plus a
telemetry schema version, and supports a `**kwargs` callback, so the callback takes the two session
ids it uses and `**kwargs` for the rest. It pops one or two entries and returns nothing. It is wrapped
so that no input can make it raise: a payload with no session id, a non-string id and a repeated id
all do nothing.

`old_session_id` is expired too when it is non-empty. A fork changes the session id, and a record left
behind under the old one would keep that session's catalog dumped forever.

### Every kind opens a window

The host's `kind` set is closed and the event reaches us after a compaction committed, so every kind
expires the record. Building a kind list would put a second guess in place of the one being deleted,
and would need an edit every time the host adds a kind. The cost of expiring too eagerly is bounded: a
compaction that leaves the earlier router block visible can cost one more full catalog dump on a
further failed pass, which is a token cost, not a wrong answer.

### The parser is deleted, not kept behind a switch

The marker constants, the padding-category set, the default-ignorable filler set, the visible-length
counter, the summary test and the summary token function are removed. There is no fallback path, no
setting to re-enable the scan, and no partial scan. A host that cannot deliver the event cannot make
the window open, and the manifest floor is what the host enforces instead.

### The host enforces the floor before it imports the plugin

The host checks the declared floor before it imports a plugin: it compares `requires_hermes` with the
running version and skips the plugin with one warning that names both
(`hermes_cli/plugins_loader.py:251`, `:576-582`, `hermes_cli/plugins_manifest.py:502-510`). A host below
the floor therefore registers nothing at all, no middleware, no tool and no hook, and one log line says
why. The documentation states that consequence beside the floor.

### The manifest declares what the plugin consumes

`version` moves to `0.2.0` because a released floor and a removed mechanism are a breaking release for
a host below the floor. `requires_hermes` moves to `>=0.22.3`, the release whose tag contains the
commit that introduced the event. `provides_hooks` becomes `["post_compaction"]`.

### The tests resolve the running host

The two test modules that import `hermes_cli` stop defaulting to a hard-coded stale directory and
resolve the install the running `hermes` launcher reports, keeping a skip with a reason when they
cannot. The stale copy has 41 hooks and no `post_compaction`, so a hook contract test against it would
pass only by never running.

### The README carries the cost and the authoring rule

Two things move into the README because a user decides with them. The rerank stage runs on a small
model that needs no GPU, and the README says so with the numbers from the live setup rather than an
adjective: the model, its weight size, the CPU thread count and the measured time for a shortlist at
the default size and at a short one. The numbers are re-measured when the ticket is built, and the
command that produced each one is in the implementer's report. The skill-library-maintenance paragraph
keeps its point and gains the authoring rule the router makes worth following: the router scores two
fields, the name and the description. The embedding and the term index read the name plus the
description clipped at `maxDescriptionChars` (default 400), the reranker reads the name plus the full
description, so the words a user would type for the task belong in the description and the distinctive
ones belong first. It also says that tidying a library helps the model pick a skill without this plugin
and helps more with it.

### What does not change

The injection block, the suggestion selection, the routing pipeline, the bounded failure policy, the
decision log line, the display record, the search tool, the read-path resolution, and the pinned-row
rules are untouched. The window record keeps its meaning and its single writer. A turn in which no
compaction happened behaves exactly as today.

## Directive Table

Each row quotes the user verbatim where his own words exist, marks a board selection as a selection,
names the artifact the words resolve against today at `file:line`, and names where this design
implements the row.

| Directive | Artifact today | Where the design implements it |
|---|---|---|
| "Doing manual parsing which is error-prone, is completely retarded when there are already better and more fool-proof mechanisms." (his typed answer, the compaction board's `process-rule` question) | `hermes_skill_router/router.py:446-544` (the whole scan: four marker constants, two length floors, the padding filter, `_is_compaction_summary`, `_summary_token`) | Implementation Decisions, "The parser is deleted, not kept behind a switch"; contracts C20 and C21; ticket 01 |
| Board selection `parser-floor` option a: "Delete the parser; floor >=0.22.3; version 0.2.0" | `plugin.yaml:4` (`version: "0.1.0"`), `:8` (`requires_hermes: ">=0.22.2"`), `:10` (`provides_hooks: []`); the parser at `hermes_skill_router/router.py:446-544`; the load gate at `hermes_cli/plugins_loader.py:576-582` | Implementation Decisions, "The parser is deleted", "The host enforces the floor before it imports the plugin", "The manifest declares what the plugin consumes"; contracts C20, C21 and C23; tickets 01, 02 |
| "let a deepseek-eco worker update README to contain information that the reranker model is small and has good performance on CPU (use our live setup as reference). Also in the skill library maintenance part in the intro, add how skill descriptions should be written clearer to maximize the chance of it being picked. Such maintenance is a good idea even in normal use of hermes without this skill router, but they become even more useful with the skill router." (his instruction, 2026-11-13) | `README.md:7` (the warm-turn claim), `README.md:20-24` (the maintenance paragraph), `plugin.yaml:134-139` (`maxDescriptionChars`, default 400); the live reranker at `127.0.0.1:8010` serving `tasksource/tasksource-decider-nano`, 298 MB of weights on CPU | Implementation Decisions, "The README carries the cost and the authoring rule"; contract C25; ticket 03 |
| "Fix handoff plugin to fire post_compaction event too." (his typed answer, the earlier compaction board's `detection` question) | `/home/wilder/git-repos/context-handoff-engine/_commit.py:222` `commit_handoff` rewrites the live message list and fires nothing; `grep -rn post_compaction` over that directory returns no match | Contract C24 and ticket 04, the engine half of one spec |
| "the context handoff plugin should be fixed to fire the same hook / event as built-in compressor." (his typed answer, the same board's `pipeline-rule` question) | the built-in compressor's own commit path is the one that fires `agent/compaction_events.py:239` `fire_post_compaction`; the engine's `compress()` raises `RuntimeError` instead (`__init__.py:121-145`) | Contract C24; ticket 04 |
| Board selection `event-kind` option a: "Summarize, with reason handoff_commit" | `agent/compaction_events.py:33` `_KINDS`; `:239-250` passes `kind` and `reason` through unvalidated; the probe in the earlier verification delivered `kind=summarize`, `reason=handoff_commit` verbatim | Contract C24; ticket 04 |
| Board selection `repo-flow` option a: "One spec in the router repo, engine commit as its second half" | this file; the router at `hermes-skill-router` `main` `5345ce9`, the engine at `context-handoff-engine` `master` `dda0705` | Tickets 01 to 04 under this one spec; commits and pushes per repository |
| "focus on just my hermes skill router repo first" (this session) | `hermes-skill-router` branch `main`, HEAD `5345ce9` | Ticket order: 01, 02 and 03 land and are verified before ticket 04 is dispatched |
| "ignore the coding / review pipeline work (another session did that)" (this session) | the board's `process-rule` answer asked for the coding and review skills to change; the skills board at `http://127.0.0.1:3080/grill/abmrdu4ky4cgisgd` is owned elsewhere | Out of Scope: no skill, no `coding-preferences` section, no reviewer-process change in this spec |
| "ignore the upstream PR for now" (this session) | [NousResearch/hermes-agent#154098](https://github.com/NousResearch/hermes-agent/pull/154098) adds a catalog entry pinned to `5345ce9`, `version: "0.1.0"`, `requires_hermes: ">=0.22.2"`, `provides_hooks: []` | Out of Scope: the catalog PR is not touched, and its pin goes stale when this lands |
| "we don't need two copies. Remove stale one and use just one copy with all our fixes and work." (his typed answer, the earlier board's `host-tests` question) | `tests/test_plugin_registration.py:139-149` and `tests/test_middleware_contract.py:177-187` default to `~/.hermes/hermes-agent`, which holds 41 hooks and no `post_compaction` | Implementation Decisions, "The tests resolve the running host"; contract C26; ticket 02 |

## Compliance Table

Status words: holds (the behaviour exists today), partial (the mechanism exists but the rule differs),
absent (no equivalent).

| Requirement | Router today | What changes |
|---|---|---|
| Window boundary from the host's event | absent. The record is keyed on a summary-text digest (`hermes_skill_router/search.py:110-159`, `hermes_skill_router/router.py:519-545`), and `__init__.py:40-60` registers no hook. | The parser goes, the record becomes a set, the plugin registers `post_compaction` and expires the entry. |
| A compaction opens exactly one new window | partial. The digest expires the record when a different summary appears, which is right, and two identical summaries keep one window. | The event expires it every time, so two identical summaries open two windows. |
| A fork opens a window on both ids | absent. `_DUMPED` is keyed by one session id. | The callback expires `session_id` and `old_session_id`. |
| The manifest declares the hook and the floor | absent. `provides_hooks: []` (`plugin.yaml:10`), floor `>=0.22.2` (`:8`). | `["post_compaction"]`, `>=0.22.3`, `version: "0.2.0"`. |
| The hook contract is tested against the host that has it | partial. The host-importing tests default to a stale copy with no `post_compaction` (`tests/test_plugin_registration.py:139-149`). | Both fixtures resolve the running install, with a reasoned skip. |
| The failure policy is unchanged | holds. `_fail` at `hermes_skill_router/router.py:221-240` decides the dump, the strip and the marker. | Unchanged, minus the token argument. |

## Testing Decisions

Tests assert observable behaviour at the plugin's outer seams. For this change those seams are:
`register(ctx)` producing a hook whose callback the test can invoke, `rewrite()` returning a payload
or `None` across a compaction, and the manifest parse. Nothing new is mocked beyond the transports
that are already injectable.

Prior art to follow: `tests/test_rewrite.py:1633-1717` pins the dump record's boundary behaviour today
and keeps its shape, with the record driven by the event instead of by summary text;
`tests/test_plugin_registration.py:206` pins the registered surface; `tests/test_middleware_contract.py`
pins the callback signature the dispatcher calls.

The tests that exist to pin the scan are deleted with it: the marker-padding cases, the
default-ignorable filler cases, the merged-delimiter cases, the header-signature floor cases, the
metadata-tag case and the quoted-marker cases. Their replacement is small: one test per preserved rule
of the record, one test that a complete pasted summary changes nothing, one test per callback
tolerance, and one test that the manifest declares what the code registers.

A test that would otherwise need the host names the skip reason, and the reason names the path it
looked for, so a missing host is visible instead of silent.

## Out of Scope

- The engine half in `context-handoff-engine` is one ticket under this same spec (his `repo-flow`
  selection: one spec, engine commit as its second half), and it is dispatched after the router half
  is verified, per his instruction to focus on the router repository first.
- The coding and review skills, the `coding-preferences` section, and the reviewer-process rule. His
  answer to the board's `process-rule` question asked for those, and he has assigned them to another
  session.
- The upstream catalog PR. It is not amended, not closed, and not merged here.
- The DSH skill router's window rule. That half is `packages/dsh-skill-router` in another checkout and
  already derives its window from the visible conversation.
- The parked in-chat display for routed skills, which needs a host change.
- Removing the stale host source directory from the machine. The repo stops depending on it; deleting
  it is a machine action with 22 documented references across the dotfiles tree and is not this spec.
- Any change to the injection block, the selection rules, the routing pipeline, the decision line, the
  display record, or the search tool.

## Deliverables

- This spec, the skeleton, the tickets, the implementer briefs and the review reports under
  `docs/work/compaction-window/`.
- The router change on `hermes-skill-router` `main`: the event-driven window, the manifest declaration
  and the documentation, each with its tests.
- The engine change on `context-handoff-engine` `master`: one announcement after the commit lands, with
  its self-check.
- The probe that settles U1 in the skeleton, once a live compaction has run with the router loaded.

## Further Notes

### What is measured and what is claimed

- Measured: `post_compaction` is one of the running install's 48 valid hooks
  (`hermes_cli/plugins.py:207`), the running host is `v0.23.1`, and the event's fire site is
  `agent/compaction_events.py:239-250`. A probe against that install registered a real hook through a
  real plugin context and received a handoff-shaped payload verbatim, including
  `kind=summarize`, `reason=handoff_commit` and `telemetry_schema_version=hermes.observer.v1`.
- Measured: the engine directory contains no `post_compaction` call, so the live window never opens
  today for the live engine.
- Measured: the host's default plugin isolation is `in_process` (`hermes_cli/plugin_isolation.py:3`,
  `ISOLATION_IN_PROCESS`), and `~/.hermes/config.yaml` carries no `plugins.isolation` key, so this
  plugin's middleware and its hook callback load into one interpreter and one record.
- Unverified: that a live compaction delivers the event to the router's callback on a real turn. No
  compaction has run since the hook was written. The probe that settles it is a live compaction with
  the router loaded, then one line from the router's own log.

### The defect this spec exists to stop

The scan was not a small error. Four separate defects were found in it after it was written, each
patched with its own test, and the live host still could not open a window with it. The host had
published the same fact as an event the whole time. The design rule that follows is the user's own,
quoted in the directive table: use the mechanism the owner already publishes.

### Open questions

1. What the callback does with a repeated event. Taken: the pop is idempotent and no count is kept, so
   a repeated event is a second no-op and cannot split a window.
2. Whether the record should also expire when the visible conversation loses every router block.
   Taken: no, that stays as it is. The conversation already decides suggestion suppression, and the
   record's only job is the failed-pass dump.
3. Whether the stale host directory is deleted. Taken: not in this spec. The test fixtures stop naming
   it, which is the repository-side half of his answer.
