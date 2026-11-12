# 01: The compaction window opens on the host's event

**What to build:** the router stops reading the conversation for compaction summary text and instead
registers the host's `post_compaction` event. When a compaction commits, the session's dump record is
expired, so the next failed pass in that session may send the whole skill catalog once more. Every kind
opens a window, and a fork expires both the old and the new session id. The marker constants, the
visible-length floors, the padding filter, the summary test and the digest function are deleted, with
their tests, and nothing replaces them: no fallback scan and no setting that re-enables one. The
failure policy, the returned payload and the log line are untouched.

**Blocked by:** None (can start immediately)

**Verify:**

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_rewrite.py tests/test_tools.py -q
```

**Status:** ready-for-agent

- [ ] The dump record is a set of session ids with a write, a read and an expire operation. It carries
      no summary token, no digest and no timestamp, and a read never mutates it.
- [ ] The plugin's registration adds exactly one hook, named `post_compaction`, beside the one
      middleware and the one tool it already registers.
- [ ] The hook callback takes the session id, the old session id and any further keyword fields, and it
      expires the record for the session id and, when non-empty, for the old session id.
- [ ] The callback never raises: a payload with no session id, an empty id, a non-string id and a
      repeated id all leave the process running and change nothing beyond the intended expire.
- [ ] A committed compaction makes the next failed pass dump the catalog again once, and a second
      failure in that new window strips the catalog and leaves the dropped-catalog marker.
- [ ] One test goes through the registration, not around it: it registers the plugin, invokes the
      callback that the registration stored, and then observes `rewrite()` dump the catalog again, so
      the event is proven to reach the window read by the route the host uses.
- [ ] A request that carries a complete pasted compaction summary, header, suffix and all, no longer
      changes the window: the record decides, and the paste changes nothing.
- [ ] Two compactions in one session open two windows even when their summary text would have been
      byte-identical.
- [ ] No removed name survives in the module: no `_SUMMARY_` constant, no `_is_compaction_summary`, no
      `_summary_token`, no `_substantive_length`, no padding-category or default-ignorable set.
- [ ] The tests that pinned the scan are deleted with it, and the request payload is byte-identical to
      0.1.0's on every failure path.

**Deliverables**

- `hermes_skill_router/router.py`, `hermes_skill_router/search.py`, `__init__.py`
- `tests/test_rewrite.py`, `tests/test_tools.py`
