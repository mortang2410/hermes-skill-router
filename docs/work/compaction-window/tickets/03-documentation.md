# 03: The documentation names the event as the window boundary, with the README cost and authoring rule

**What to build:** the documentation describes the compaction window as the router reads it after this
change: a committed compaction event opens a new window, every kind opens one, both session ids of a
fork are expired, and no message text is inspected for a summary. The paragraph that described the
summary digest and its floors is deleted rather than amended. Where the documents say a request with
no earlier router block "opens a compaction window", they say instead that it forms the opening router
block, and "compaction window" is reserved for the event-driven dump record. The documentation states
that a host below the declared floor skips the plugin and logs why, because the host compares the floor
before it imports anything. The README gains what a user decides with: the measured CPU cost of the
rerank stage, and how to write a skill description so the router picks it. A documentation test pins
the new boundary statement so a later edit cannot quietly restore the old claim.

**Blocked by:** 01

**Verify:**

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_readme.py -q
```

**Status:** ready-for-agent

- [ ] The technical documentation states that a committed compaction event opens a new window, and that
      every kind and both session ids do so.
- [ ] No sentence in either document claims that summary text or its digest decides a window.
- [ ] Both documents say "opening block" where they mean block presence, and no sentence calls block
      presence a window boundary.
- [ ] The documentation states that the host compares `requires_hermes` before importing the plugin, so
      a host below the floor registers nothing and logs one warning naming both versions.
- [ ] The documentation names the event the plugin registers, and no warning text is described as
      carrying user text.
- [ ] The README names the rerank model, its weight size, the CPU thread count, and the measured time for
      a shortlist at the default size and at a short one. Every number is re-measured by the implementer
      before it is written, and the command that produced it is in the report.
- [ ] The README's skill-library-maintenance paragraph says that tidying a library helps the model pick a
      skill without this plugin and helps more with it.
- [ ] The same paragraph says how to write a description the router can pick, in the router's real terms:
      the two scored fields are the name and the description; the embedding and the term index read the
      name plus the description clipped at `maxDescriptionChars` (default 400); the reranker reads the
      name plus the full description; the words a user would type for the task belong in the description
      and the distinctive ones go first.
- [ ] The documentation test pins the boundary statement, the missing-host statement and the
      description-authoring rule, and passes with the rewritten text.
- [ ] The existing documentation pins still pass, including the block, pinned-row and decision-line
      statements, and both documents still carry no em dash.

**Deliverables**

- `technical-documentation.md`, `README.md`
- `tests/test_readme.py`
