# Implementer brief 01: the compaction window opens on the host's event

## Your seat and route

- You are the implementer for ticket 01 only. You build, you test, and you commit. You do not review
  your own work and you do not edit any file outside your list.
- Resolved route for this run: provider `bifrost`, model `opendesign/deepseek-eco`, effort `max`. Any
  worker you dispatch inherits this route; pass all three fields.
- Read `/home/wilder/git-repos/dotfiles/.agents/skills/implement/SKILL.md` in full before your first
  edit and follow it. It owns the procedure; this brief owns the content.
- Bounds: no command runs longer than five minutes; the full suite takes about 15 seconds and you run it
  once. Start no servers, install nothing, do not touch another session's files, and keep scratch files
  under `/home/wilder/Trash/compaction-window/`.

## Where you work

- Repository: `/home/wilder/git-repos/hermes-skill-router`, branch `main`, HEAD `5345ce9`. Work in the
  working tree, do not create a branch, do not rebase, do not touch another session's files.
- Design of record: `docs/work/compaction-window/spec.md` (approved by the document review) and
  `docs/work/compaction-window/skeleton.md`. The skeleton is the approved structure.
- Ticket: kata `personal` issue `085f` (this feature's ticket 01). Acceptance criteria are its
  checklist.

## Files you own, exclusively

- `hermes_skill_router/router.py`
- `hermes_skill_router/search.py`
- `__init__.py`
- `tests/test_rewrite.py`
- `tests/test_tools.py`

Do not edit `plugin.yaml`, the two other test modules, or the two documents: tickets 02 and 03 own
them.

## Skeleton rows you must satisfy

- Module M25 (`hermes_skill_router/router.py`), M22 (`hermes_skill_router/search.py`), M27
  (`__init__.py`), M29 (`tests/test_rewrite.py`, `tests/test_tools.py`).
- Contracts C20 (the dump record), C21 (the rewrite and its window read), C22 (the registration
  surface). Requirements R1, R2, R3.
- Read `docs/work/compaction-window/skeleton.md` for the exact contract text before you start. The
  skeleton's `post` and `invariants` bullets in C20, C21 and C22 are your specification.

## What changes, with today's anchors

`hermes_skill_router/search.py`

- `_DUMPED` at `:119` currently maps a session id to a summary token. It becomes the membership record
  C20 describes.
- `remember_dump` (`:123`) loses its token argument. `dumped` (`:141`) loses its token argument and
  becomes a pure read that never mutates. Add `expire_dump(session_id)`. `clear_dumps` (`:132`) stays
  for tests.
- Keep the one existing lock (`_PARTITIONS_LOCK`) and the partitions half untouched.
- Update the long comment above `_DUMPED` so it describes the event, not the summary token.

`hermes_skill_router/router.py`

- Delete `_SUMMARY_PREFIX`, `_MERGED_SUMMARY_DELIMITER`, `_SUMMARY_SUFFIX`, `_SUMMARY_METADATA_KEY`,
  `_SUMMARY_MIN_CHARS`, `_SUMMARY_FULL_CHARS`, `_SUMMARY_HEADER_SIGNATURE`, `_PADDING_CATEGORIES`,
  `_DEFAULT_IGNORABLE_FILLERS`, `_substantive_length`, `_is_compaction_summary` and `_summary_token`
  (roughly `:446-545`).
- In `_compose` (`:205-208`), replace the `_summary_token` call and the token comparison with
  `dumped(session_id)`. The `summary_token or ""` argument at `:232` becomes nothing.
- Remove the module `import unicodedata` and `import hashlib` if they have no other user; check each
  with `grep -n`.
- Leave `_fail`, `_prior_blocks`, `_dropped_before`, `_already_injected`, the ranking path, the log
  line and every failure message exactly as they are apart from the two call sites above.

`__init__.py`

- Register one `post_compaction` hook in `register()` (`:40-60`), after the tool registration.
- The callback is a module-level function matching C22: `def on_post_compaction(session_id: str = "",
  old_session_id: str = "", **kwargs: Any) -> None`. It calls `expire_dump` for `session_id` and, when
  non-empty, for `old_session_id`. It must accept anything the host passes, including nothing.
- Export it (no `__all__` gymnastics) so the registration test can import it.

Tests

- Delete every test that pins the scan: the padding, filler, merged-delimiter, header-signature-floor,
  metadata-tag and quoted-marker cases in `tests/test_rewrite.py` (roughly `:1626-1900`, `:1943-2024`).
  Keep and adapt `test_a_dump_record_ignores_a_shrinking_request_list` (`:1633`).
- Rewrite the two window tests (`:1663`, `:1717`) so the window is driven by the record's expire
  operation, exactly as the hook calls it, and the request text carries whatever the old tests pasted
  with no effect. Add the "a complete pasted summary changes nothing" case.
- Add a test that the callback expires both ids, tolerates a missing, empty, non-string and repeated id,
  and returns `None`.
- Add the registration-path test: call `register()` with a recording fake `ctx`, take the callback the
  ctx stored for `post_compaction`, invoke it with a session id that has already dumped, then call
  `rewrite()` and assert it dumps the catalog again. This is the only test that proves the event
  reaches the window read by the route the host uses, rather than by a direct call to the callback.
- `tests/test_tools.py` imports `remember_dump` (`:40`) and calls it (`:232`): follow the new signature.

## How to verify

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_rewrite.py tests/test_tools.py -q
python3 -m pytest -q          # the whole suite once at the end
grep -n "_SUMMARY_\|_is_compaction_summary\|_summary_token\|_substantive_length\|_PADDING_CATEGORIES" \
  hermes_skill_router/router.py   # must print nothing
```

Then run the drift check against the skeleton once:

```sh
python3 /home/wilder/.dsh/skills/post-spec-skeleton/scripts/drift_check.py \
  docs/work/compaction-window/skeleton.md --root . \
  --baseline /home/wilder/Trash/compaction-window/outline-before.json
```

Report its output verbatim. A warning is examined and either explained by a recorded revision or
reported as a proposal.

## Commit

Stage only your five files, check `git diff --cached --name-only` lists exactly those five, and commit
on `main` with a message beginning `feat(router): open the compaction window on the host's event`. Do
not push. Do not commit another session's files, and do not use `git add -A`.

## Your report

Reply with: the commit sha; each acceptance criterion from the ticket with the test name that pins it;
the two command outputs above; the drift check output; every skeleton divergence you had to make
(tier 1 recorded, tier 2 or 3 proposed with evidence, never edited silently); and any test you deleted,
named. Do not claim a criterion you did not pin with a test.
