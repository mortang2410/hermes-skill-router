# Implementer brief 03: the documentation names the event as the window boundary, and the README carries its cost

## Your seat and route

- You are the implementer for ticket 03 only. You build, you test. Do not commit: ticket 02 runs
  beside you on the same branch, and two concurrent commits on one index can absorb each other's
  files. The orchestrator commits your files after the review.
- Resolved route: provider `bifrost`, model `opendesign/deepseek-eco`, effort `max`.
- Read `/home/wilder/git-repos/dotfiles/.agents/skills/implement/SKILL.md` in full before your first
  edit and follow it, except for its commit step, replaced by the paragraph above.
- Bounds: no command runs longer than five minutes; the full suite takes about 15 seconds and you run it
  once. Start no servers, install nothing, do not touch another session's files, and keep scratch files
  under `/home/wilder/Trash/compaction-window/`.

## Where you work

- Repository: `/home/wilder/git-repos/hermes-skill-router`, branch `main`.
- Design of record: `docs/work/compaction-window/spec.md`, `docs/work/compaction-window/skeleton.md`.
- Ticket: kata `personal` issue `rzfc` (this feature's ticket 03).
- Tickets 01 and 02 have landed. Read the changed `hermes_skill_router/router.py`,
  `hermes_skill_router/search.py`, `__init__.py` and `plugin.yaml` before you write a word, because
  every sentence you write must describe that code.

## Files you own, exclusively

- `technical-documentation.md`
- `README.md`
- `tests/test_readme.py`

## Skeleton rows you must satisfy

Module M32; contract C25 (the documentation statement); requirement R6.

## Three vocabulary rules, applied everywhere you touch prose

1. **"Compaction window" means the record the event opens.** It is per session, it is expired by the
   host's `post_compaction` event, and it decides whether the next failed pass may dump the catalog
   again. Nothing about message text decides it.
2. **Block presence is not a window boundary.** Where the documents say that a request whose earlier
   router block a compaction removed lists the pinned rows again, that is the **opening block**.
   Rewrite any sentence that calls block presence "opening a compaction window" to say it forms the
   opening block. The two known sites are `technical-documentation.md:204-207` and `:232-238`, and
   `README.md:167-173`; sweep the rest of both files for the same misuse.
3. **A host below the floor imports nothing.** The loader compares `requires_hermes` before it imports
   the plugin, then skips the plugin and logs one warning that names both versions
   (`hermes_cli/plugins_loader.py:251`, `:576-582`, `hermes_cli/plugins_manifest.py:502-510`). Never
   write that a below-floor host loads the plugin, sees an unknown hook name, or half-works.

## What changes, with today's anchors

`technical-documentation.md`

- The paragraph at `:665-695` is the one to replace. Its current claims: the record holds the summary's
  digest, a later request carrying a different summary is a new window, the first three non-system
  messages are checked, the closing suffix and the header signature gate the match, and the length
  floors are 1,000 and 2,000 visible characters. Every one of those sentences is now false.
- Write the replacement in the same register and at the same level of detail: the record is a
  per-session mark, the host's `post_compaction` event expires it, every kind opens a window, a fork
  expires both the old and the new session id, the callback is idempotent, and no message text is
  inspected. Say that the plugin registers the event and that the event is the only boundary.
- Keep the parts of that section that are still true and are not about the scan: the dump on the first
  failed pass, the dropped marker on a later failure, the system-role-only payload check for the
  marker, the "suggests nothing after a dump" rule, and the healthy-empty-selection rule.
- Add a short statement of what a host below the declared floor does, in rule 3's terms: it imports
  nothing, registers nothing, and logs one warning naming both versions, so the window never opens.
- `:782` cites a live check as `7c38fb1` with Hermes `v0.22.2+1671`. The running host is `v0.23.1` and
  the code under test has moved since. State the version you verify with, or drop the version claim.
  Do not leave a stale one.

`README.md`

- The block section at `:166-174` describes the pinned rows and the opening block. Check every sentence
  against the new code and apply vocabulary rules 1 and 2. Do not move the window-boundary paragraph
  into the README: the technical documentation owns it.

### The two additions the user asked for on 2026-11-13

His words: "let a deepseek-eco worker update README to contain information that the reranker model is
small and has good performance on CPU (use our live setup as reference). Also in the skill library
maintenance part in the intro, add how skill descriptions should be written clearer to maximize the
chance of it being picked. Such maintenance is a good idea even in normal use of hermes without this
skill router, but they become even more useful with the skill router."

1. **The rerank stage and its cost.** Add a short README passage saying the rerank stage runs on a
   small model that does not need a GPU, and give the reference numbers: model
   `tasksource/tasksource-decider-nano`, 298 MB of weights, CPU with 8 threads, and the measured time
   for a shortlist at the default size and at a short one.
   - Measure it yourself against the live server before you write a number: `POST
     http://127.0.0.1:8010/v1/rerank` with your own documents, several runs, and report the numbers
     you measured with the command that produced them. Do not copy numbers from this brief.
   - Keep the claim honest. A 50-row shortlist on 8 CPU threads is a few seconds, which is the cost
     per routed turn; say that plainly and say what it buys, rather than calling it fast. If your
     measurement differs from the reference, the measurement wins and the sentence changes.
2. **How to write a description the router can pick.** In the README intro's skill-library-maintenance
   paragraph, say first that tidying a library helps the model pick a skill in normal Hermes use, and
   helps more with this plugin. Then say how to write the description, in the router's real terms:
   - The router scores two fields, the name and the description (`hermes_skill_router/library.py:96-104`
     builds the embedded document text as the name, a newline, then the description clipped to
     `max_description_chars`; `:326` gives the term index the same text).
   - The rerank stage reads the name plus the **full** description (`hermes_skill_router/search.py:330-334`
     builds each rerank document from the prefix, the name, a dash and the description).
   - So the cap (`maxDescriptionChars`, default 400, `plugin.yaml`) limits what the embedding and the term
     index read, and does not shorten what the reranker reads. Say that plainly, and do not claim text
     past the cap is dropped before ranking.
   - The words a user would type for the task belong in the description, and the distinctive ones go
     first, because a description that opens with generic words ranks behind one that opens with the
     user's own words.
   Do not write that the description is the only scored text: the name is scored in both stages, and a
   vague name costs recall whatever the description says.

`tests/test_readme.py`

- `:140-163` pins the block, cap, path and opening rules. Add one pin for the boundary statement: the
  documentation must state that a committed compaction event opens a new window. Follow the existing
  style, which asserts a literal sentence from the prose.
- Add one pin for the below-floor statement, in rule 3's terms, and one for the description-authoring
  rule in the README, so a later edit cannot quietly soften either.
- `:286-296` asserts both documents carry no em dash. Keep that passing: no em dash in anything you
  write.

## How to verify

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_readme.py -q
python3 -m pytest -q
grep -n "opens a compaction window\|digest\|first three non-system" technical-documentation.md README.md
```

The last command must print nothing that uses block presence as a window boundary or describes the
deleted scan.

## Your report

Reply with: each acceptance criterion from the ticket with the test name or the quoted sentence that
satisfies it; the three command outputs verbatim; your rerank measurement with the command that
produced it; and every sentence you deleted with its old line number. Do not claim a criterion you did
not pin with a test.
