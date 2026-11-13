# Skeleton: compaction window

- status: approved
- mode: design
- spec: docs/work/compaction-window/spec.md
- scope: hermes_skill_router/, __init__.py, plugin.yaml, tests/, README.md and technical-documentation.md in hermes-skill-router at 5345ce9, plus _commit.py and plugin.yaml in context-handoff-engine at dda0705. Only the window boundary changes; every other module is out of scope.
- grill: none. The design is the user's board answers, quoted in the spec directive table.

Module ids continue the numbering in docs/work/hermes-parity-2026-11-11/skeleton.md, so M2, M4, M7,
M8, M22 and M23 keep their meaning and every row here takes a fresh id. Contract ids likewise start
at C20.

## Requirements map

| Id | Requirement | Spec anchor | Kind |
|----|-------------|-------------|------|
| R1 | A committed compaction opens a new window: the plugin registers the host's post_compaction event and expires that session's dump record when it arrives, for every kind and for both session ids | Solution; Implementation Decisions, "The window record loses its token", "Every kind opens a window" | behaviour |
| R2 | The text scan is gone: no summary marker constant, no visible-length floor, no summary test, no digest, no fallback and no switch that re-enables them | Implementation Decisions, "The parser is deleted, not kept behind a switch" | behaviour |
| R3 | The failure policy is unchanged: one clean dump per window, the dropped marker on a later failure, byte-identical payloads, and the hook callback never raises | Implementation Decisions, "The window record loses its token"; User Stories 6, 7, 8, 9, 20 | error |
| R4 | The manifest declares the hook, the floor and the release: provides_hooks [post_compaction], requires_hermes >=0.22.3, version 0.2.0 | Implementation Decisions, "The manifest declares what the plugin consumes" | non-functional |
| R5 | The host-importing tests resolve the install the running hermes launcher reports, and a missing host is a reasoned skip rather than a silent pass | Implementation Decisions, "The tests resolve the running host" | behaviour |
| R6 | The technical documentation names the event as the window boundary, drops the scan paragraph, states what happens below the floor, and the README carries the rerank cost and the description-authoring rule | Implementation Decisions, "The host enforces the floor before it imports the plugin", "The README carries the cost and the authoring rule"; User Stories 17, 18, 21 to 24 | non-functional |
| R7 | The engine that rewrites the conversation fires the same event with the seven documented fields after its commit lands | Implementation Decisions; directive rows 3, 4, 5 | behaviour |

## Modules

Six rows below (M2, M4, M7, M8, M22, M23) are the neighbours this feature reads and does not change.
They are listed so the edges in this table name declared ids. Their own dep columns say `none`
because this skeleton does not govern them: their edges today are the ones recorded in
docs/work/hermes-parity-2026-11-11/skeleton.md and this feature moves none of them.

| Id | Path | Responsibility | Allowed deps | Forbidden deps | Exposed surface |
|----|------|----------------|--------------|----------------|-----------------|
| M2 | hermes_skill_router/config.py | Unchanged. Decides what a configuration value means | none | none | `Config`, `load`, `DEFAULTS` |
| M4 | hermes_skill_router/library.py | Unchanged. Owns the index, its cache identity and the shortlist order | none | none | `load_or_build`, `Index` |
| M7 | hermes_skill_router/select.py | Unchanged. Decides which candidates are suggested | none | none | `select` |
| M8 | hermes_skill_router/inject.py | Unchanged. Decides what the injected block looks like | none | none | `render` |
| M22 | hermes_skill_router/search.py | Is the only place that runs one Hermes ranking pass for the turn and the search tool, and the only place that holds per-session routing state: the parsed partition and the dump record, with the record's write, read and expire operations under one lock | M2 | M25, M27, M28, M29, M30, M31, M32, M33 | `rank`, `select_rows`, `remember_dump`, `dumped`, `expire_dump`, `remember_partition`, `forget_session`, `clear_dumps` |
| M23 | hermes_skill_router/tools.py | Unchanged. Owns the host path resolver and the search tool | none | none | `skill_path`, `search_tool` |
| M25 | hermes_skill_router/router.py | Is the only place that composes one request rewrite, owns its failure policy, reads the window record and writes the decision log line | M2, M4, M7, M8, M22, M23 | M27, M28, M29, M30, M31, M32, M33 | `rewrite` |
| M27 | __init__.py | Is the only place that registers the plugin's surface with the host: one middleware, one tool and one hook | M25, M22, M23 | M2, M4, M7, M8, M28, M29, M30, M31, M32, M33 | `register`, `on_post_compaction` |
| M28 | plugin.yaml | Is the only place the manifest declares what the plugin consumes and the host release it needs | none | none | the manifest fields |
| M29 | tests/test_rewrite.py and tests/test_tools.py | Pin the rewrite seam and the record's rules: one dump per window, the marker on a later failure, an unchanged payload, a pasted summary that changes nothing | M25, M22 | M27, M28, M30, M31, M32, M33 | none |
| M30 | tests/test_plugin_registration.py | Pins the registered surface and the manifest, and resolves the running install for the hook contract | M27, M28 | M25, M22, M2, M4, M7, M8, M23, M29, M31, M32, M33 | none |
| M31 | tests/test_middleware_contract.py | Pins the middleware callback against the running host's own dispatcher | M25, M27 | M22, M28, M29, M30, M32, M33 | none |
| M32 | technical-documentation.md, README.md and tests/test_readme.py | Own the prose that states the window boundary and the test that pins it | M25, M22, M27, M28 | M29, M30, M31, M33 | none |
| M33 | _commit.py, _announce.py and plugin.yaml in context-handoff-engine | Is the only place the handoff engine announces a committed rewrite to the host, through the host module `agent.compaction_events` which is outside this skeleton | none | M2, M4, M7, M8, M22, M23, M25, M27, M28, M29, M30, M31, M32 | nothing to the router; the host event is the seam |

## Contracts

### C20: the dump record

- kind: function
- owner: M22
- stability: stable
- satisfies: R1, R3

```python
def remember_dump(session_id: str) -> None: ...
def dumped(session_id: str) -> bool: ...
def expire_dump(session_id: str) -> None: ...
def clear_dumps() -> None: ...
```

- pre: `session_id` is the id the middleware received, and for the hook the id the event carried.
- post: `remember_dump` marks the session as having sent the whole catalog in its current window. `dumped` reports that mark and never mutates it. `expire_dump` removes it, and removing an absent mark is a no-op. `clear_dumps` empties the record and exists for tests only.
- errors: none. Every operation takes the module's one lock and none raises on any string, including the empty one.
- invariants: the record is a set of session ids with no token, no timestamp and no count. `dumped` is a pure read: a probe or a second turn cannot open a window by reading.
- state: two states per session, absent and marked. `remember_dump` moves absent to marked, `expire_dump` moves either state to absent, and no other call changes it.

### C21: the rewrite and its window read

- kind: function
- owner: M25
- stability: stable
- satisfies: R1, R2, R3

```python
def rewrite(request: Any, *, api_mode: str = CHAT_COMPLETIONS, session_id: str = "", turn_id: str = "", ...) -> Optional[Mapping[str, Any]]: ...
```

- pre: the caller is the host's llm_request middleware dispatcher and passes the request payload.
- post: the signature, the failure policy and the returned payload are unchanged from 0.1.0. The only difference is where the window comes from: `dumped(session_id)` replaces the summary-token comparison, and no message text is inspected for a compaction summary.
- errors: no exception escapes. A plugin bug is logged at debug and returns `None`, which leaves the request untouched.
- invariants: the returned payload is either `None` or the input payload with the catalog section replaced. No path returns a partially stripped request. A module-level constant list of removed behaviour does not exist: no `_SUMMARY_*` name, no `_is_compaction_summary`, no `_summary_token`, no `_substantive_length`.
- state: reads M22's record; writes it only through `remember_dump` on the first failed pass of a window.

### C22: the registration surface

- kind: function
- owner: M27
- stability: stable
- satisfies: R1, R3

```python
def register(ctx: Any) -> None: ...
def on_post_compaction(session_id: str = "", old_session_id: str = "", **kwargs: Any) -> None: ...
```

- pre: `ctx` is the host's plugin context; the hook name `post_compaction` is one the host dispatches.
- post: exactly one `llm_request` middleware, one `skill_router_search` tool, and one `post_compaction` hook are registered, in that order after the existing tool registration. The hook callback expires M22's record for `session_id` and, when non-empty, for `old_session_id`.
- errors: the callback catches nothing and raises nothing: it receives the event's fields as keyword arguments, treats a missing, empty or non-string id as nothing to do, and returns `None`. A registration failure is the host's to raise; the plugin does not swallow it.
- invariants: the callback takes `**kwargs`, because the host calls hook callbacks with the event's fields as keywords plus a telemetry schema version and supports a var-keyword callback. The callback is idempotent: a repeated event pops an absent mark and changes nothing.
- state: writes M22's record only through `expire_dump`.

### C23: the manifest declaration

- kind: schema
- owner: M28
- stability: stable
- satisfies: R4

- pre: the host's plugin loader reads this file before it imports the plugin's code.
- post: `version` is `"0.2.0"` as a quoted string, `requires_hermes` is `">=0.22.3"`, `provides_hooks` is `["post_compaction"]`, and `provides_middleware` and `provides_tools` are unchanged.
- errors: a host below the floor imports nothing. The loader compares the floor with the running version and skips the plugin with one warning naming both (`hermes_cli/plugins_loader.py:251`, `:576-582`, `hermes_cli/plugins_manifest.py:502-510`).
- invariants: every capability the code registers is declared here and nothing else is declared. The file parses under a strict loader that refuses duplicate keys.
- state: none.

### C24: the engine announces its commit

- kind: function
- owner: M33
- stability: stable
- satisfies: R7

```python
def commit_handoff(...) -> ...: ...  # unchanged signature; announces after the rewrite lands
```

- pre: a handoff commit has rewritten the live message list and knows its session id and its row counts before and after.
- post: after the rewrite, one call to the host's `agent.compaction_events.fire_post_compaction` with `session_id`, `old_session_id=""`, `kind="summarize"`, `reason="handoff_commit"`, `before_tokens=None`, `after_tokens=None` and `messages_removed` equal to the rows removed.
- errors: the announcement is best effort. An import failure or an event failure is logged and never fails the commit, because the rewrite has already landed and the user's turn must not break.
- invariants: one announcement per committed rewrite, never on a dry run and never on a lock miss. The engine registers no hook of its own.
- state: none the engine owns; the event is the seam.

### C25: the documentation statement

- kind: text
- owner: M32
- stability: stable
- satisfies: R6

- pre: the documentation is read by a maintainer and pinned by a test.
- post: the technical documentation states that a committed compaction event opens a new window and that no message text is inspected for a summary; the paragraph describing the digest scan is gone; the block and pinned-row sentences say "opening block" where they mean block presence and reserve "compaction window" for the event; the documentation states that a host below the declared floor skips the plugin and logs why; the README names the rerank model, its weight size, the CPU thread count and the measured shortlist timings, and its maintenance paragraph carries the description-authoring rule: the name and the description are the two fields the router scores, the embedding and the term index read the name plus the description clipped at `maxDescriptionChars` (default 400), and the reranker reads the name plus the full description.
- errors: none.
- invariants: no sentence claims the summary text or its digest decides a window. No sentence claims a below-floor host loads the plugin. No sentence claims the description is the only scored field, or that the character cap applies to the reranker's input. Every README number names what was measured.
- state: none.

### C26: the host tree resolution

- kind: function
- owner: M30
- stability: provisional
- satisfies: R5

```python
def resolve_host_tree() -> Path: ...   # skipped test when nothing resolves
```

- pre: a test module needs `hermes_cli` from the host to assert a contract against it. M30 owns this contract; M31 consumes it and adds no resolver of its own.
- post: the resolver returns the install directory the running `hermes` launcher reports, or a caller-named override. When neither resolves, the test skips with a reason that names the path or command it tried. The result is cached for the test session and the launcher call is bounded by a timeout.
- errors: never raises into a test body. A missing launcher, a non-zero exit and unparsable output are all the same answer: skip with the reason.
- invariants: no test module names a fixed host directory. The host's own hook catalogue contains `post_compaction` before any hook assertion runs against it.
- state: one resolution per test session.

## Data flow

A turn arrives at M27's middleware, which calls M25's `rewrite`. M25 reads M22's record for the
session id and, on the first failed pass of a window, marks it. When the host commits a compaction it
fires `post_compaction`, M27's callback runs in the same process, and M22's record loses that
session's mark. The next failed pass therefore dumps the catalog again instead of leaving the dropped
marker. Independently, the handoff engine commits a rewrite and calls the host's fire function, which
is the same producer the built-in compressor uses.

## State and concurrency

One module-level set in M22, guarded by the lock that already guards the partitions, keyed by session
id. Three writers: the middleware marks, the hook callback expires, and tests clear. Every operation is
a single dict or set mutation under the lock, so two turns cannot interleave a read with a write.

The host runs `post_compaction` through its bounded worker path: the hook is in
`_HOOK_TIMEOUT_BOUNDED_HOOKS` (`hermes_cli/plugins_dispatch.py:42-52`), the dispatcher starts a daemon
worker with a timeout (`:295-307`, `:350-433`), and a hook call identity keyed on `session_id` means a
still-running callback skips that session's next event. The callback does two lock-protected removals
and no I/O, so it cannot approach the timeout. Recovery after an abandoned call is conditional, not
guaranteed: while that call identity is still running, and for the suppression window after a timeout,
the dispatcher skips later events for that session (`:350-370`, `:421-432`), so the mark can outlive
the event that should have cleared it until an event is actually delivered.

The plugin loads in one process with its middleware, because the host's default isolation is
`in_process` and this profile sets no override (`hermes_cli/plugin_isolation.py:3`). A live compaction
delivering the event remains unproven; see U1.

## Brownfield delta

| Path | Existing fact | Proposed change | Evidence |
|------|---------------|-----------------|----------|
| hermes_skill_router/router.py | Scans the first three non-system messages for a host summary and digests its text as the window identity | Delete the scan, the constants, the length floors and the digest; call the record without a token | router.py:446-544, :205-208 |
| hermes_skill_router/search.py | The record maps a session id to a summary token, and a differing token expires it | The record becomes a set of session ids with a write, a read and an expire operation | search.py:110-159 |
| __init__.py | Registers one middleware and one tool, no hook | Also register the post_compaction hook and its callback | __init__.py:40-60 |
| plugin.yaml | version 0.1.0, floor >=0.22.2, provides_hooks [] | version 0.2.0, floor >=0.22.3, provides_hooks [post_compaction] | plugin.yaml:1-13 |
| tests/test_rewrite.py | Pins the scan: marker padding, fillers, merged delimiters, header floors, the metadata tag and the quoted marker, plus three window tests | Delete the scan tests; drive the three window rules through the event; add one test that a complete pasted summary changes nothing | tests/test_rewrite.py:1626-2024 |
| tests/test_tools.py | Calls `remember_dump(session)` and reads the record | Follow the new signatures | tests/test_tools.py:40, :232 |
| tests/test_plugin_registration.py | Asserts no hook, `provides_hooks == []`, version 0.1.0, floor >=0.22.2, and defaults the host tree to ~/.hermes/hermes-agent | Assert the hook and the new manifest fields; resolve the running install | tests/test_plugin_registration.py:139-149, :206, :238 |
| tests/test_middleware_contract.py | Defaults the host tree to ~/.hermes/hermes-agent | Resolve the running install | tests/test_middleware_contract.py:177-187 |
| technical-documentation.md | Describes the summary digest as the window boundary in one long paragraph | Describe the event, drop the scan paragraph, state what happens below the floor | technical-documentation.md:665-695, :782 |
| tests/test_readme.py | Pins the block and pinned-row rules, nothing about the window record | Pin the new boundary sentence | tests/test_readme.py:140-163 |
| _commit.py in context-handoff-engine | commit_handoff rewrites the live list and fires nothing | Call the host's fire function after the rewrite | _commit.py:222, :316-318, :425 |

## Verification hooks

| Contract | Check | Command |
|----------|-------|---------|
| C20 | the record's three operations, idempotent expire, no-token identity, and a read that cannot open a window | `python3 -m pytest tests/test_tools.py tests/test_rewrite.py -q` |
| C21 | the three window rules through the event, a complete pasted summary that changes nothing, the unchanged payload on every failure, and no removed name left in the module | `python3 -m pytest tests/test_rewrite.py -q` |
| C22 | one middleware, one tool and one hook registered; the callback is var-keyword and tolerant; two ids expire | `python3 -m pytest tests/test_plugin_registration.py tests/test_rewrite.py -q` |
| C23 | strict manifest parse, the four declared fields, and the quoted version literal | `python3 -m pytest tests/test_plugin_registration.py -q` |
| C25 | the documentation states the boundary and keeps its existing pins | `python3 -m pytest tests/test_readme.py -q` |
| C24 | the announcement payload is asserted by the engine's self-check, the commit path is shown to call it once after the list is replaced and never on a refusal, and the end-to-end probe confirms the router's record expires | `cd /home/wilder/git-repos/context-handoff-engine && python3 -m py_compile _commit.py __init__.py _announce.py && python3 _announce.py --self-check && python3 _announce.py --self-check-commit-path`, then the named probe under `docs/work/compaction-window/probes/` against the live host |
| C26 | the host fixtures resolve the running install, and a forced missing host skips with a reason naming the attempted path | `python3 -m pytest tests/test_plugin_registration.py tests/test_middleware_contract.py -q` and that module's missing-host test |

## Traceability

| Requirement | Contracts | Tickets |
|-------------|-----------|---------|
| R1 | C20, C21, C22 | 01 |
| R2 | C21 | 01 |
| R3 | C20, C21, C22 | 01 |
| R4 | C23 | 02 |
| R5 | C26 | 02 |
| R6 | C25 | 03 |
| R7 | C24 | 04 |

## Uncertainties and tradeoffs

| Id | Uncertainty | Why unresolved | Settle by | Owner |
|----|-------------|----------------|-----------|-------|
| U1 | Whether a live compaction delivers the event to the router's callback on a real turn, which the shared record's expiry depends on | The isolation setting is known (`in_process`, no override), but no compaction has run since the hook was written | a live compaction with the router loaded, then one line from the router's own log | the named end-to-end probe in ticket 04 |
| U3 | Whether one announcement per commit is enough when the engine's commit retries | The engine has no tests today and its retry path was never exercised | the engine's own probe in ticket 04 | ticket 04 |

## Revision log

| Date | Change | Trigger | Approved by |
|------|--------|---------|-------------|
| 2026-11-13 | The below-floor uncertainty is closed: the loader compares `requires_hermes` and skips the plugin with one warning, so a host below the floor imports nothing (C23, ticket 03) | the round-1 review found the earlier claim wrong; the loader source settles it | tier 1, recorded here |
| 2026-11-13 | The drift baseline reports 13 errors that predate this feature and that this feature neither adds nor needs: `plugin.yaml` has no adapter, the Python adapter does not list M2's `DEFAULTS` dict or M23's `search_tool`, the six neighbour rows say `no deps` while their code imports M2, M4 and M7, and M27's forbidden list holds M2 while `__init__.py` imports `config.load`. The feature closes `M22:expire_dump` and `M27:on_post_compaction` and adds no error | ticket 01's drift run at `0e36963` against the baseline at `2d963ac` | tier 2, recorded; the code review round confirms this reading before the drift check can count as clean |
| 2026-11-13 | C22's error line now says the callback receives the event's fields as keyword arguments, instead of promising a mapping payload the host never sends. The code is unchanged: the host calls every hook callback as `cb(**_hook_callback_kwargs(...))` (`hermes_cli/plugins_dispatch.py:634`, `:755`) | the code review's mapping probe: a positional mapping is ignored by the string guard, and no host path passes one | tier 1, wording only; both reviewers read the same probe |
