# Review: the handoff engine fires the host's compaction event, ticket 04

- Date: 2026-11-13
- Mode: standard, two finders (A: behavior, data, errors, resources, baseline security; B: requirements
  coverage, contracts, caller compatibility, whether the checks could detect a wrong implementation);
  the request includes a commit, so the orchestrator commits after closure
- Repository: `/home/wilder/git-repos/context-handoff-engine` (the feature's design of record lives in
  `hermes-skill-router`, so this report sits in that feature's review trail)
- Feature: `compaction-window`
- Base/head: `dda0705..ade6f30` (ticket 04 at `2a84b4f`, the review fixes at `3ffe35d` and `ade6f30`)
- Scope: `_announce.py` (new), the `announce_commit` call site in `_commit.py`, and `plugin.yaml` at
  `0.4.0`; the end-to-end probe in the router repository that travels this module's announce path
- Threat model: ordinary accidental errors in a single-writer plugin loaded by one host process: a wrong
  call order, a double fire, a missed refusal, a swallowed failure that hides a real one, a check that
  would stay green through a regression, a manifest the host's own validator rejects. Out of scope:
  hostile concurrent writers, malformed host state, and future edits outside the changed surface
- Reviewer route: provider `codex`, model `codex/luna-bifrost`, effort `max`, both finders
- Rounds: 3 (round 1 the full commit, round 2 the longer-replacement fix, round 3 the validator fix)
- Publication: report only; no comments published

## Findings

### [P2] Exercise a longer committed replacement in the commit-path check - _announce.py:335-346

Both finders raised this one. The commit-path self-check derived its expected removed count from the
same rows the production path counts, and only a shrinking history ran (`_history()`: 5 rows to 3). The
contract maps a negative removed count to `None`, and a regression that sent the negative value through
would still pass: the check's own expectation was computed with the same negative-to-`None` policy, so
the bad value matched. The direct `--self-check` covered `None` but only for a hand-built call, not for
the production row-count mapping.

Fix: add a real commit case whose live history is one row and whose replacement is two, run it through
`_check_commit_lands`, and assert the host event carries `None`; prove it by removing the production
guard and watching the check fail. Landed at `3ffe35d` (`_history_longer`, the extra `_check_commit_lands`
call, and the printed `longer replacement:` line).

### [P2] The manifest fails the host's own validator - _announce.py:127-128

Found by the orchestrator after round 2, from the repository's own verification item `hermes plugins
validate .`. The self-check stub bound its fake host modules with two subscript assignments
(`sys.modules["agent"] = ...`), and the validator reads a subscript assignment as a runtime rebind of
Hermes core, so the plugin failed with `no core override` even though the stubs only run under
`python3 _announce.py --self-check`. The pre-change tree at `dda0705` validated, so the failure was
introduced here, and it would have shipped a manifest the host rejects.

Fix: bind the same two stubs through one `sys.modules.update({...})` call, which the validator accepts,
and keep both self-checks passing. Landed at `ade6f30`.

## Possible concerns

None raised. Both finders ran the self-checks, the static scan, and their own scratch mutations.

## Serious unresolved risks

None. Every finding closed, and the remaining warnings are pre-existing.

## Queued for the owner

- deepcanary: none, the concern list closed empty
- board: none published, the concern list closed empty

## Verdict

VERDICT: clean

Finder A: "VERDICT: approved" after the validator fix.
Finder B: "Approved; no other findings", after re-running the validator against `ade6f30` once the checkout's HEAD was confirmed at that commit.

The review closes on the current tree at `ade6f30`.

## Dispositions

| Finding | Disposition | Evidence |
|---|---|---|
| [P2] Exercise a longer committed replacement in the commit-path check | RESOLVED | `_announce.py` `_history_longer` runs a 1 row to 2 row commit through the real body; the check prints `longer replacement: 1 host event, messages_removed=None, expected None, rows 1 -> 2`; removing the production negative guard fails with `expected None, observed -1`, and sending `None` for a positive count fails on the 5 to 3 case. Both finders reproduced both mutations in scratch copies |
| [P2] The manifest fails the host's own validator | RESOLVED | `hermes plugins validate .` prints `✓ no core override - no runtime rebinds of Hermes core` and `Validation passed.`; the pre-change tree at `dda0705` also validates, so the regression is gone; finder A removed the stub binding in a scratch copy and both self-checks failed with `call count: expected 2, observed 0` |

## Verification

```text
cd /home/wilder/git-repos/context-handoff-engine && python3 -m py_compile _commit.py __init__.py _announce.py
exit 0, no output
```

```text
python3 _announce.py --self-check
PASS: announce_commit sent 2 call(s) with fields ['session_id', 'old_session_id', 'kind', 'reason',
'before_tokens', 'after_tokens', 'messages_removed'], messages_removed [7, None], no host module importable
```

```text
python3 _announce.py --self-check-commit-path
commit path: 1 host event, fields [...], messages_removed=2, expected 2, rows 5 -> 3, after the rewrite=True, lock held=True
longer replacement: 1 host event, messages_removed=None, expected None, rows 1 -> 2
refusal (lock miss: try_acquire refused): stage=lock, host events=0, rewritten=False, unchanged=True
refusal (watermark refusal: no active rows): stage=watermark, host events=0, rewritten=False, unchanged=True
refusal (watermark refusal: held row inactive): stage=watermark, host events=0, rewritten=False, unchanged=True
refusal (commit refusal: archive_and_compact raised): stage=commit, host events=0, rewritten=False, unchanged=True
refusal (messages refusal: no commit row): stage=messages, host events=0, rewritten=False, unchanged=True
dry run: absent; signatures (messages: 'list', args: 'dict') -> 'str' / (agent, session_db, messages, args, document, session_id, lock_holder) -> 'str'; marker hits {} []
PASS: one host event after the rewrite with the rows the rewrite removed, None for a longer replacement; none on the lock miss, the watermark refusals or the commit and messages refusals; no dry-run path
```

```text
hermes plugins validate .
✓ no core override - no runtime rebinds of Hermes core
⚠ private core imports (catalog rule 9): _announce.py:60 agent.compaction_events.fire_post_compaction;
  _commit.py:210, :332, :416. These predate the change, apart from the one host seam this ticket exists to call
Validation passed.
```

```text
# the router's end-to-end probe, delivering through this module's announce path
cd docs/work/compaction-window/probes
HERMES_DISABLE_LAZY_INSTALLS=1 python3 end_to_end_event.py --engine /home/wilder/git-repos/context-handoff-engine
delivering through /home/wilder/git-repos/context-handoff-engine/_announce.announce_commit
record after the event: dumped=False
PASS: the event reached the router's registered callback, and the next passes dumped then stripped
exit 0
```

```text
# static scan, round 3 delta
cccc exit 0, lizard exit 0, duplicate rate 0.00%
```

## Unverified risks

- No real compaction has fired the event on this host. The probe exercises the real host dispatch path
  with the real router callback and this module's real `announce_commit` body, but the process is driven
  by the probe rather than by a live session that crosses its context threshold.
- The commit-path self-check executes the real `announce_commit` body but cannot intercept the
  package-qualified import the host's own loader resolves; the probe covers that seam, and the commit
  path itself imports by plain import.
- The engine has no test suite of its own. The two self-checks and the probe are the whole evidence, by
  design of the ticket.
