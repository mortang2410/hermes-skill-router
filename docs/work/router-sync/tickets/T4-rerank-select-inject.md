# T4 Rerank client, selection, injected block

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

The rerank HTTP client that speaks the same document format as DSH, the cutoff and cap selection
that reads the returned probabilities directly, and the rendered block that replaces the catalog.

## Covers

Stories S2, S3, S9. Requirements R3, R4, R5, R6, R11. Modules M6, M7, M8. Contracts C6, C7, C8.

## Exclusive files

- `hermes_skill_router/rerank.py`
- `hermes_skill_router/select.py`
- `hermes_skill_router/inject.py`
- `tests/test_rerank_client.py`
- `tests/test_select.py`
- `tests/test_inject.py`

## Blockers

T3, for the fake endpoint helper and the index shape.

## What it must do

1. `rerank(cfg, query, documents)`: `POST` JSON `{"model": cfg.rerank_model, "query": query,
   "documents": [documents], "top_n": len(documents)}` to `cfg.rerank_base_url + cfg.rerank_path`,
   timeout from `cfg.rerank_timeout_ms`. Build the request through the opener T3 exposes from
   `embed.py`, not a second copy of the policy, so a 301, 302, 303, 307 or 308 fails the call instead of
   sending the query and
   the skill names to another host (D4). Parse `{"results": [{"index": int, "relevance_score": float}]}`
   and return one score per input document, in input order. Return `None` on connection error, HTTP
   error including any 3xx, timeout, a missing `results` list, or a result whose length differs from
   the document count. No retry.
2. Document text is fixed by config: `cfg.document_prefix + name + " - " + description`. The default
   prefix is the DSH sentence ending in `Skill: `, so the same endpoint sees the same shape from both
   clients. The service parses that prefix to recover the skill name
   (`/home/wilder/.scripts/decider-rerank-server.py`, `parse_document`), and an unprefixed document
   arrives as an unnamed candidate.
3. `select(scores, names, cfg)`: the scores are already probabilities from the reranker. Compare each
   one directly and inclusively against `cfg.cutoff`, sort descending, truncate to
   `cfg.max_suggestions`. Do not apply a softmax: a second normalization turns a single confident
   candidate into 0.99 and pushes 49 near-zero candidates over a 0.01 cutoff. Empty input or
   `max_suggestions == 0` returns `[]`.
4. `render(rows, kept, index, cfg, one_shot)`: `rows` are the selected suggestions, at most `max_suggestions`, and
   `kept` is the kept half of the partition: eager rows plus every row a non-empty deferred list
   leaves listed. Return one of two fixed headers chosen by `one_shot`, sharing their first line so the
   router's double-injection guard recognises either: the one-shot header keeps
   the restriction its prose carried (general process skills are not loaded for work already known, and
   skills are not created or edited), the normal header carries the loading guidance. Then one
   `- <name>: <description>` line per suggested row,
   with the description looked up in `index` by name (never stored on the row), then one line per kept
   row under an `Also listed:` marker, with every description cut to `cfg.max_description_chars` and
   no name repeated (a name in both lists appears once, as a suggestion). A kept row with an empty description is listed as `- <name>` with no trailing colon.
   Return `""` only when both lists are empty. The header says these are the skills that
   may help, to open one with `skill_view`, and to call `skills_list` when none fits, and it carries
   the guidance the removed `## Skills` prose provided.
5. Every function is pure apart from the HTTP client, so selection and rendering are testable without
   a server.

## Verification

```bash
python3 -m pytest tests/test_rerank_client.py tests/test_select.py tests/test_inject.py -q
```

## Acceptance criteria

- [ ] Scores are restored to input order when the endpoint returns `results` shuffled.
- [ ] A results list shorter than the documents returns `None`, never a padded score.
- [ ] The outgoing documents start with the configured prefix and carry `name - description` exactly;
      a captured request body is asserted against a fixed expected string.
- [ ] Parity with DSH on fixed vectors: scores `[1.0, 0.0, 0.0]` with `cutoff 0.01` select one row,
      not three; scores `[0.5, 0.5]` select both; a score exactly equal to the cutoff is kept and one
      epsilon below it is dropped.
- [ ] `max_suggestions` boundary: exactly that many rows survive a larger eligible set.
- [ ] `render` clips at `max_description_chars`, never emits two rows for one name, and its header
      mentions both `skill_view` and `skills_list`.
- [ ] `render` lists every kept name whenever the block is non-empty, including a name with an empty
      description listed as `- <name>`, so a pinned skill and a skill a
      non-empty deferred list leaves out both survive the strip, and returns `""` when suggestions and
      kept rows are empty.
- [ ] `max_suggestions` caps suggested rows only: with `0` and a non-empty kept set the block carries
      the kept rows and no suggestion.
- [ ] The rerank client makes no follow-up request when the fake endpoint answers 302 or 307, including
      a `Location` pointing at another host, and returns `None` for that call.
- [ ] A description containing a newline still renders as one line.

## Notes

- The header text is part of the injected contract; T7 documents it so a reader can recognise it in a
  request dump.
- Do not add per-skill hints, examples or file paths. Names and descriptions only; that is D3.
- `cutoff` is a probability threshold, not a share of a recalculated distribution. DSH's selector
  (`packages/dsh-skill-router/src/selection/suggestion-selector.ts`) is the reference.
