# 04: The handoff engine fires the host's compaction event

**What to build:** the handoff engine, which is the component that actually rewrites this session's
conversation, announces that rewrite to the host with the same event the built-in compressor fires.
The announcement happens once, after the live list has been replaced, and it carries the session id,
an empty old session id, the kind and reason the board settled, no token counts, and the number of rows
the rewrite removed. The announcement is best effort: a failed import or a failed call is logged and
never fails the commit, because the rewrite has already landed. The engine's own manifest moves to a
release that carries the announcement, and the engine ships two self-checks that run without the host:
one for the payload and one for the commit path. A named end-to-end probe, run after this lands beside
the router's hook, is what proves the pair works on a live turn.

**Blocked by:** 01, 02, 03

**Verify:**

```sh
cd /home/wilder/git-repos/context-handoff-engine
python3 -m py_compile _commit.py __init__.py _announce.py
python3 _announce.py --self-check
python3 _announce.py --self-check-commit-path
```

Then the named probe under `docs/work/compaction-window/probes/`. It calls the engine's own
announcement function for a scratch session id, so the event travels the engine's code, the host's fire
path, the host's dispatcher and the router's registered callback. It asserts the router's record
expires and that the next failed pass dumps the catalog again. A probe that builds the event itself
proves nothing about the link between the two halves.

**Status:** ready-for-agent

- [ ] The commit path calls the host's fire function exactly once per committed rewrite, after the live
      list is replaced, and never on a refusal, a dry run, or a lock miss.
- [ ] The call carries `session_id` from the agent, `old_session_id=""`, `kind="summarize"`,
      `reason="handoff_commit"`, `before_tokens=None`, `after_tokens=None`, and `messages_removed` equal
      to the rows the rewrite dropped.
- [ ] A missing host module, an import error and an exception inside the call are all caught, logged at
      warning, and do not change the commit's returned text or its retry behaviour.
- [ ] The announcement module registers no hook and reads no configuration.
- [ ] The engine's manifest version names the release that carries the announcement.
- [ ] `--self-check` runs with no host importable, stubs the host module, and asserts the seven fields
      of the delivered call, exiting non-zero on any mismatch.
- [ ] `--self-check-commit-path` drives the locked commit body with stubbed host modules, a fake agent
      and a fake session database, and asserts one announcement after the list replacement and none on
      each refusal branch. If a host call in that body cannot be stubbed, it reports the call instead of
      weakening the check.
- [ ] The named end-to-end probe runs on this host after both halves are installed, calls the engine's
      own announcement function rather than a synthetic payload, prints the router's record state before
      and after, and asserts that the next failed pass dumps the catalog again.

**Deliverables**

- `_commit.py`, `_announce.py`, `plugin.yaml` in `context-handoff-engine`, by the ticket 04 implementer
- `docs/work/compaction-window/probes/` in the router repository, by the orchestrator, because it needs
  a live host and the implementer does not touch that repository
