# Implementer brief 04: the handoff engine fires the host's compaction event

## Your seat and route

- You are the implementer for ticket 04 only. You build, you check, and you commit in the engine's own
  repository. You do not touch the router repository, and you do not review your own work.
- Resolved route: provider `bifrost`, model `opendesign/deepseek-eco`, effort `max`.
- Read `/home/wilder/git-repos/dotfiles/.agents/skills/implement/SKILL.md` in full before your first
  edit and follow it.
- Bounds: no command runs longer than five minutes; the full suite takes about 15 seconds and you run it
  once. Start no servers, install nothing, do not touch another session's files, and keep scratch files
  under `/home/wilder/Trash/compaction-window/`.

## Where you work

- Repository: `/home/wilder/git-repos/context-handoff-engine`, branch `master`, HEAD `dda0705`. Work in
  the working tree; do not create a branch and do not rebase.
- Live path: `~/.hermes/plugins/context-handoff-engine` is a symlink to this checkout, so your edit is
  live after the next Hermes start. Do not restart anything.
- Design of record: `/home/wilder/git-repos/hermes-skill-router/docs/work/compaction-window/spec.md`
  and `skeleton.md`. Ticket: kata `personal` issue `b3qh`.
- The router half is done: its plugin registers a `post_compaction` hook, so a host that delivers the
  event will expire the router's record. Your half is the producer.
- The host's function you call, in the running install
  `/home/wilder/.hermes/installs/5323f763494eb399/environments/6e4b37e4dab14600ac4818bd4b5ccaec/workspace`:

```python
# agent/compaction_events.py:239
def fire_post_compaction(*, session_id: str, old_session_id: str = "", kind: str | None,
                         reason: str | None, before_tokens: int | None, after_tokens: int | None,
                         messages_removed: int | None) -> None:
```

  It is keyword-only, it is already gated on `has_hook("post_compaction")` internally, and it returns
  nothing. You do not need to check for listeners yourself.

## Files you own, exclusively

- `_commit.py`
- `_announce.py` (new)
- `plugin.yaml`

Do not add a `.gitignore` or touch the untracked `__pycache__/`: those are not this ticket's, and the
deliverable list is these three files.

## Skeleton rows you must satisfy

Module M33; contract C24 (the engine announces its commit); requirement R7.

## What changes, with today's anchors

`_announce.py` (new)

- One public function `announce_commit(session_id, messages_removed) -> None` that imports
  `agent.compaction_events.fire_post_compaction` inside the function, calls it with
  `old_session_id=""`, `kind="summarize"`, `reason="handoff_commit"`, `before_tokens=None`,
  `after_tokens=None`, `messages_removed=messages_removed`, and swallows every exception from the
  import and the call with one `logger.warning` that names the failure and the session id. It never
  raises and it returns nothing.
- A `--self-check` path under `if __name__ == "__main__":` that installs a stub `agent` and
  `agent.compaction_events` into `sys.modules`, calls `announce_commit` once with a known session id
  and row count, and asserts the stubbed function received exactly the seven keyword fields with the
  values above. Exit non-zero with a printed diff on any mismatch, zero with one line on success. It
  must run with no host importable.
- A `--self-check-commit-path` path in the same file that drives the commit body itself. Stub the host
  modules it touches, build a fake agent and a fake session database, call the locked commit body with
  a small message list, and assert the announcement fires exactly once, after the live list was
  replaced, with the row count the rewrite removed. Then assert it does not fire on each refusal
  branch the body has: the lock miss, the watermark refusal and a dry run. Exit non-zero with the
  observed call log on failure.
  - If a host call inside that body cannot be stubbed honestly, do not weaken the check: report which
    call it was and what it needs, keep the rest of the check, and say so in your report.

`_commit.py`

- `_commit_locked` replaces the live list at `messages[:] = new_list` (around `:425`). Call
  `announce_commit` once after that slice assignment has happened and after the persist-index
  re-anchoring directly below it, so the event describes a rewrite that is already live.
- `rows_before` is already in scope. Pass `rows_before - len(new_list)` as `messages_removed`. If that
  figure is negative, pass `None` rather than a negative count.
- Never call it on a refusal path: every `_fail` return, the lock miss at the `try_acquire` branch and
  the watermark refusal keep the conversation unchanged and must announce nothing.
- Update the module docstring's numbered commit steps to name the announcement where it happens.

`plugin.yaml`

- `version: "0.3.0"` becomes `version: "0.4.0"`. `provides_hooks` stays
  `[pre_llm_call, transform_tool_result]`: the engine consumes no hook and provides none for this
  change. Add nothing else.

## How to verify

```sh
cd /home/wilder/git-repos/context-handoff-engine
python3 -m py_compile _commit.py __init__.py _announce.py
python3 _announce.py --self-check
python3 _announce.py --self-check-commit-path
grep -n "announce_commit" _commit.py
```

All four must exit 0 and print what they assert. Then prove the negative: `grep` every `_fail` return
and every refusal branch in `_commit_locked` and show no `announce_commit` call sits before the rewrite
lands. Say in your report how you checked it.

## The end-to-end probe is not yours

A named probe under `docs/work/compaction-window/probes/` in the router repository calls your own
`announce_commit` with a scratch session id, so the event travels your module, the host's fire path, the
host's dispatcher and the router's registered callback. It asserts the router's record expires and that
the next failed pass dumps the catalog again. That is why `_announce.py` must stay importable with no
side effects: the probe imports it the way the engine does.

## Commit

Stage only your three files, check `git diff --cached --name-only` lists exactly those, and commit with
a message beginning `feat(commit): fire the host's post_compaction event`. Do not push; the orchestrator
pushes after the review.

## Your report

Reply with: the commit sha; each acceptance criterion with the command or assertion that proves it; the
four command outputs verbatim; a statement of where in `_commit_locked` the call sits relative to the
slice assignment and to every refusal; and any host call you could not stub for the commit-path check.
Do not claim a criterion you did not prove.
