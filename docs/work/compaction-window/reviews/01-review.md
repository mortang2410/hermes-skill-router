# Review: spec, skeleton and tickets for the compaction window

- Date: 2026-11-13
- Shape: one reviewer, read-only, no commit in the request; the orchestrator writes this report
- Repository: `/home/wilder/git-repos/hermes-skill-router`
- Feature: `compaction-window`
- Base/head: `main` at `5345ce9`, documents untracked under `docs/work/compaction-window/`
- Scope: `docs/work/compaction-window/spec.md`, `skeleton.md`, `tickets/01-window-event.md`,
  `tickets/02-manifest-and-host-fixture.md`, `tickets/03-documentation.md`,
  `tickets/04-engine-announces.md`, the four implementer briefs, and the kata records `gw6w`,
  `085f`, `1516`, `rzfc`, `b3qh`
- Reviewer route: provider `codex`, model `codex/luna-bifrost`, effort `max`
- Rounds: 1
- Publication: report only; no comments published

## Findings

### [P2] The below-floor behaviour is stated backwards - spec.md, "The host floor is a declaration"

The spec and the ticket-02/-03 text say that a host below the floor loads the plugin and logs a warning
because the hook name is unknown there. The loader does the opposite: it compares `requires_hermes`
before importing the plugin, skips the plugin, and logs one warning that names both versions.

Fix: state in the spec, the skeleton and the documentation ticket that the host imports nothing below
the floor and logs the comparison, with `hermes_cli/plugins_loader.py:251`, `:576-582` and
`hermes_cli/plugins_manifest.py:502-510` as the evidence.

### [P2] "Opens a compaction window" is used for block presence - technical-documentation.md:204-207, :232-238, README.md:167-173

Three passages say a request with no earlier router block opens a compaction window. That phrase now
names the event-driven record, so the two meanings collide and a reader cannot tell which rule they
are reading.

Fix: reserve "compaction window" for the record the event expires, and say "opening block" where the
passage means block presence. Assign the wording to ticket 03, which owns both files.

### [P1] No ticket owns the integration check between the engine's commit and the router's record - tickets/04

Ticket 04 verified only the payload of the engine's announcement. Nothing in the ticket set proved that
an announcement reaches the router's registered callback and changes what the next turn dumps.

Fix: add a named end-to-end probe to ticket 04's Verify, alongside a self-check of the payload and a
self-check of the commit path.

### [P2] Ticket 04 is blocked only by ticket 01 - tickets/04-engine-announces.md, kata `b3qh`

The engine's announcement is only correct against the manifest and the documentation that describe it,
but the ticket's blocker line named ticket 01 alone, so it could be dispatched before the contract it
depends on existed.

Fix: block ticket 04 by 01, 02 and 03, in the ticket file and in kata.

### [P2] Ticket 01 has no test that goes through the registration - tickets/01-window-event.md

Every window test the ticket named called the callback directly. A registration that stored the
callback under the wrong name, or never stored it, would pass all of them.

Fix: one test registers the plugin, invokes the callback the registration stored, and then observes
`rewrite()` dump the catalog again.

### [P2] R5 has no contract and its missing-host skip is untested - skeleton.md, requirements map

Requirement R5 (the host tests resolve the running install, and a missing host is a skip with a reason)
traced to no contract, so no ticket could own it, and the skip branch that R5 describes was never
exercised.

Fix: add contract C26 for the resolution, trace R5 to it, and give ticket 02 a test that forces the
missing-host branch.

### [P2] Module ownership is incomplete - skeleton.md modules table, briefs/04-brief.md

M27's exposed surface named neither the hook callback nor its registration, and M33 named `_commit.py`
and `plugin.yaml` but not the new `_announce.py`. Brief 04 also told the implementer to add a
`.gitignore`, which no module owns.

Fix: name `on_post_compaction` in M27's surface, name `_announce.py` in M33's path, and drop the
`.gitignore` instruction.

### [P2] The state section names a module that does not exist - skeleton.md, "State and concurrency"

The section attributed the dump record to M26. The record is M22.

Fix: correct the module id.

### [P2] The bounded-worker claim overstated the callback - skeleton.md, "State and concurrency"

The text said the callback cannot be abandoned. The host runs `post_compaction` through the bounded
worker path, which can abandon a slow callback, so the record can keep an entry for one session.

Fix: state the bounded path and its consequence, and keep the recovery claim to what a later
compaction does.

### [P3] The revision log carries a dated status row - skeleton.md, revision log

The row recorded a status observation rather than a revision, and its date did not match anything this
feature had recorded.

Fix: remove it. A revision row appears only when a revision is made under the policy.

### [P3] The directive table carries a stale worktree claim - spec.md, directive table

One row asserted a worktree state that no longer held.

Fix: remove the claim from the row and keep the user's words and the design pointer.

### [P3] Brief 02's commit rationale contradicts its own dispatch state - briefs/02-brief.md

The brief told the implementer not to commit because ticket 01 was being committed concurrently, and
then, three paragraphs later, stated that ticket 01 had already landed. Both cannot be true of one
dispatch.

Fix: give the reason that holds at dispatch time, which is that ticket 03 runs beside ticket 02 on one
index.

## Possible concerns

- Whether the engine's commit path mutates the live list before or after the persist-index re-anchoring
  at `_commit.py:425` - Settle with: reading `_commit_locked` in the engine checkout at `dda0705`;
  result: confirmed as a real ordering question, carried into ticket 04's commit-path self-check.

- Whether a live compaction ever delivers `post_compaction` to this host's plugin - Settle with: the
  named end-to-end probe plus one live compaction; result: unreproduced, it is uncertainty U1.

## Verdict

VERDICT: request changes

Twelve findings, two of them structural: the below-floor paragraph was factually backwards, and no
ticket proved the integration between the engine's announcement and the router's record. Both are the
kind that a passing test suite would not catch, so the document set goes back for another round before
any code is written.

## Verification

```text
$ python3 .dsh/skills/post-spec-skeleton/scripts/skeleton_lint.py docs/work/compaction-window/skeleton.md
requirements: 7
modules: 14
contracts: 7
stable: 6
provisional: 1
uncertainties: 2
traceability_rows: 7
verification_hooks: 7
status: OK
```

```text
$ grep -rnP "\x{2014}" docs/work/compaction-window/
$ echo $?
1
```

Baseline suite at the reviewed commit: `python3 -m pytest -q` in the router repository, 711 passed,
exit 0.

## Unverified risks

- No live compaction has delivered `post_compaction` to a router callback on this host. The named probe
  in ticket 04 is the settling step.
- The engine's commit path has never called the event, and the engine repository has no tests, so the
  announcement's position in the commit is read from source, not observed.
- The README cost figures this feature requires do not exist yet, so no reader has checked them.
