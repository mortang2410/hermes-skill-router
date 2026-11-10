# T5 Compose the rewrite and prove it live

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

The full request rewrite behind the middleware: query extraction, catalog strip with row capture,
shortlist, rerank, selection, injection, one log line per failure reason, and the live end-to-end
proof on this machine.

## Covers

Stories S1, S2, S3, S4, S5, S9, S10. Requirements R1, R2, R3, R5, R11, R12. Modules M9, M11.
Contract C9.

## Exclusive files

- `hermes_skill_router/router.py` (replaces the identity body T2 wrote; a declared sequential
  handoff, not parallel ownership of the same revision)
- `hermes_skill_router/display.py` (the record writer M11, so the rewrite can complete without
  waiting for the later display UI ticket)
- `tests/test_rewrite.py`
- `tests/test_middleware_contract.py`
- `tests/test_display.py` (the record writer half; the UI halves are T8's)

`hermes_skill_router/__init__.py` stays T2's file: it already imports `rewrite` from `router.py`, so
this ticket changes no registration code.

## Blockers

T1 and T4, for a live embeddings endpoint and the finished selection path.

## What it must do

1. `rewrite(request, cfg, ...)` runs only when `api_mode == "chat_completions"`. Any other mode
   returns `None` unchanged: Codex Responses carries the prompt in `instructions` and `input`, and
   Anthropic Messages carries it in `system`, so a `messages` rewrite would either do nothing or
   corrupt the call.
2. Find the last message with `role == "user"` and take its text. No user message, empty text,
   `enabled` false, no catalog block, or an empty index all return `None` (no change).
3. Order of work: partition the catalog with `cfg.eagerSkills` and `cfg.deferredSkills` and remove the
   whole skills section, embed the query, take the top `cfg.shortlist_size` rows by cosine over the
   index of routed rows, rerank `cfg.document_prefix + name + " - " + description` for those rows,
   select with `cfg.cutoff` and `cfg.max_suggestions`, then render with the selected rows,
   `Partition.kept`, the index, which is the description source for suggested rows, and
   `Partition.one_shot`, which picks the header variant, so the block re-lists the kept rows. Kept rows never enter the index, so a kept skill is never reranked as a
   suggestion, and its listing comes from the block.
4. Return `{"request": {**request, "messages": [...stripped, {"role": "system", "content": block}]}}`.
   The middleware replaces the whole payload with what the callback returns, so dropping `model`,
   `tools` or any other top-level key breaks the call. The block is a message appended at the end,
   never an edit to an existing historical message: user rows replay their recorded `api_content`
   sidecar, so an edited user message would be replayed and compound on the next turn.
5. Strip and inject are atomic: the whole skills section is removed only on a request that gets a
   non-empty block, and a request that produces no block (no selected suggestion and no kept row) is
   returned untouched with its section, tags and guidance byte-identical. The block lists the kept
   rows, eager ones and rows a non-empty deferred list leaves listed, so none is lost by the strip.
6. Guard against double injection: when the last message already starts with the first line both headers share, return
   `None`.
7. Log each distinct failure reason (`embed unreachable`, `embed timeout`, `index build failed`,
   `rerank unreachable`, `no user text`, `unsupported api mode`) at most once per process, through the
   module logger. Never log the user's text.
8. Never raise. Every exception inside `rewrite` returns `None`, so the catalog stays in place.
9. Write the display record through M11 on the same decision: `routed` with the selected suggestion
   names only when it returns a rewritten payload, `inert` with an empty list when it returns `None` for
   a request that carried the skills section on `chat_completions`. Kept rows are listed in the block
   but never recorded, so the display cannot present a pinned or deferred-excluded skill as a
   suggestion. The record is names only, capped at `cfg.max_display_names`, written atomically,
   skipped when identical, removed when `cfg.show_suggestions` is false, and it never appears in the
   payload. Broadcast `plugin.hermes-skill-router.suggestions.updated` with the turn id after a write,
   best effort.

## Verification

Unit level:

```bash
python3 -m pytest tests/test_rewrite.py tests/test_middleware_contract.py -q
python3 -m pytest -q
```

`tests/test_middleware_contract.py` drives the plugin callback through the real
`hermes_cli.middleware.apply_llm_request_middleware` when a Hermes install is importable, and
through a stand-in with the same shape otherwise. It asserts the returned dict has only the `request`
key, that `model`, `tools` and an unrelated top-level key survive the rewrite, and that an unchanged
result is `None`. A `codex_responses` payload and an `anthropic_messages` payload must come out
identical.

Live, with `127.0.0.1:8012` and `127.0.0.1:8010` up:

```bash
ln -sfn /home/wilder/git-repos/hermes-skill-router ~/.hermes/plugins/hermes-skill-router
curl -s 127.0.0.1:8012/health && curl -s 127.0.0.1:8010/health
HERMES_DUMP_REQUESTS=1 HERMES_DUMP_REQUEST_STDOUT=1 hermes -z "draft a git commit message for a bug fix"
```

In this build `HERMES_DUMP_REQUEST_STDOUT=1` prints nothing; the dump lands under
`~/.hermes/sessions/request_dump_*.json`, newest first, with the request under `request.body`. Read it
from there.

The dumped preflight payload must contain no `<available_skills>`, no `## Skills` heading and none of
the removed prose, must contain the injected header plus at least one `- <skill>: ` line, must list any
kept row, and must still carry `model` and `tools`. The record file must hold the selected suggestion
names only with `status: "routed"`, so a kept row that appears in the block must be absent from the
record. Then stop the embeddings service and repeat: the payload must contain the original
`## Skills` section byte for byte, the record must read `inert`, and the turn must still answer.

## Acceptance criteria

- [ ] With both endpoints up, the dumped request has the whole skills section gone, the kept rows
      listed in the block, and `model` and `tools` intact.
- [ ] The display record holds the selected suggestion names only, capped at `maxDisplayNames`, with
      `status: "routed"`; a kept row never appears in it; with the service stopped it holds no names and
      `status: "inert"`.
- [ ] A name on a demoted `[names only]` line appears in the block, is absent from the index and from
      the record, and stays listed even when it is named in `deferredSkills`.
- [ ] With `eagerSkills` set to one skill and `deferredSkills` limited to another, a third skill
      outside the deferred list appears in the block under `Also listed:` after the strip, and none of
      those two appears among the suggestions in the record.
- [ ] With a non-empty routed index, `maxSuggestions` 0 and one kept row, the rewritten payload still
      has the section gone and the block carries only the kept row; the record holds no names with
      `status: "routed"`.
- [ ] With `8012` stopped, the dumped request is byte-identical to the pre-plugin payload.
- [ ] With `8010` stopped, the same inert result.
- [ ] A `codex_responses` request is returned untouched (unit test).
- [ ] A turn whose router is down still completes (S4).
- [ ] The model can open a suggested skill with `skill_view` in that same session (S3).
- [ ] `/btw` and the after-turn background review take the same path: `application` of the middleware
      is unconditional, so assert it by reading the call site, not by a spawn list (S5, D2).
- [ ] A trailing second `system` message is accepted by the configured route (U1). If the provider
      rejects it, append a user-role message instead and record a tier 1 skeleton revision.

## Notes

- Keep the rewrite deterministic for one payload and one index; Hermes replays the exact bytes on
  later turns and a moving block in a replayed turn would be a cache and audit hazard.
- Report the measured latency of one cold and one warm rewrite; the operator pays it per API call.
