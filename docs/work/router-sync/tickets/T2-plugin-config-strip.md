# T2 Plugin shell, config, catalog strip, inert fallback

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

An installable Hermes plugin that loads with zero tools, registers one `llm_request` middleware,
validates its config with defaults and bounds, removes the skill catalog from a request when the
strip is enabled, and changes nothing else yet.

## Covers

Stories S1, S4, S7, S9. Requirements R1, R5, R7, R11. Modules M2, M3. Contracts C2, C3.

## Exclusive files

- `plugin.yaml`
- `__init__.py`
- `hermes_skill_router/__init__.py`
- `hermes_skill_router/config.py`
- `hermes_skill_router/catalog.py`
- `hermes_skill_router/router.py`
- `tests/test_plugin_registration.py`
- `tests/test_config.py`
- `tests/test_catalog.py`

## Blockers

None. Runs beside T1.

## What it must do

1. `plugin.yaml`: `manifest_version: 2`, `api_version: 1`, `name: hermes-skill-router`, quoted
   `version: "0.1.0"`, a non-empty `description`, `requires_hermes: ">=0.22.2"` (the installed
   Hermes reports `v0.22.2+1671.gc479f48.dirty`), `provides_tools: []`, `provides_hooks: []`,
   `license: MIT`, and a `config_schema` holding every key of the spec's config table. The installed
   validator requires `name`, `version` and a non-empty `description` (`hermes_cli/plugin_validate.py:96-107`).
2. Declare `provides_middleware: ["llm_request"]` in `plugin.yaml`. The installed `hermes plugins
   validate` fails a plugin whose `register()` registers middleware it does not declare
   (`hermes_cli/plugin_validate.py:403-416`), so this field is required for the pinned check to pass.
   This build's manifest parser does not know the key and logs one `unknown manifest field(s) ignored`
   warning per load (`hermes_cli/plugins_manifest.py:32-39,174-179`); that warning is expected and
   harmless, and the field must not be removed to silence it. The registration test asserts both: the
   field is present and the validator passes.
3. `register(ctx)` registers `ctx.register_middleware("llm_request", callback)`. The dispatcher calls
   it as `cb(**kwargs)` with no signature filtering (`hermes_cli/plugins_dispatch.py:591-602`), passing
   `request` plus `task_id`, `turn_id`, `api_request_id`, `session_id`, `platform`, `model`, `provider`,
   `base_url`, `api_mode` and `api_call_count` (`agent/turn_api_request.py:143-148`). The callback must
   therefore be declared `def on_llm_request(request, **kwargs)`; a narrower signature raises
   `TypeError`, which the dispatcher catches and skips, so routing would silently never run. A
   `TypeError` inside `register()` loads the plugin with zero registrations.
4. Read settings with `ctx.get_config(<key>, <default>)`, which resolves
   `plugins.entries.hermes-skill-router.settings.<key>` in the profile config. `load()` applies the
   spec's bounds: an out-of-bound value falls back to its default and appends a note naming the key;
   an unknown key appends a note and is ignored. `load()` never raises.
5. `strip_catalog(payload, eager, deferred)` parses the `<available_skills>` rows, then removes the
   whole skills section from the first system message: the `## Skills` heading, the prose before and
   after the block, the tags and their rows, or the single-query variant guidance with its block
   (`agent/prompt_builder.py:1395-1420`). Both row shapes are parsed: `- name: description` rows, and
   `category [names only]: a, b` lines that a coding posture demotes (`agent/prompt_builder.py:1378`).
   A name from a `[names only]` line enters `kept` with an empty description and is never routed, even
   when `deferredSkills` names it, because its description was never in the prompt. The `kept` rows are
   returned in the `Partition` so the block can list them again; nothing of the section stays behind.
   The shadowed-copy note that can follow the block, `(A copy of <names> is not listed: ...)`, is
   removed with the section and its names are dropped entirely: `skill_view` cannot load a shadowed
   copy, so listing one would offer a dead skill. It returns `(payload, Partition)` with `kept`,
   `routed`, `unknown_names` and `one_shot`, true when the stripped prose was the single-query guidance
   and false for the normal prose, or `(payload, None)` when the payload has no `messages` list or no
   skills section. It is idempotent.
   - A row whose name is in `eager` is kept.
   - Otherwise, when `deferred` is empty, the row is routed.
   - Otherwise, a row whose name is in `deferred` is routed and every other row is kept.
   - A name in both lists is kept, and one note records it.
   - A configured name the catalog does not carry lands in `unknown_names` and one note names it.
   This mirrors DSH's eager/deferred split for tools: the eager set stays visible, everything not
   eager is deferred unless the operator narrows the set.
6. `rewrite()` in this ticket is the identity: it returns `None` so the middleware reports no change.
   T5 fills it in. Returning `None` (not a dict) is the inert contract: `_apply_request_chain` skips
   any middleware result that is not a dict.
7. `config.load()` rejects an endpoint URL that is not safe: a plain `http` URL is accepted only for
   `127.0.0.1`, `localhost` or `::1`; any other host must use `https`; a URL carrying credentials is
   rejected. A rejected value falls back to its default and appends a note naming the key.
8. The callback wraps its whole body in `try/except Exception` and returns `None` on any failure, so
   a plugin bug can never abort a model call.

## Verification

```bash
python3 -m pytest tests/test_config.py tests/test_catalog.py tests/test_plugin_registration.py -q
hermes plugins validate /home/wilder/git-repos/hermes-skill-router
```

The registration test builds a fake ctx that records registrar calls and asserts that `register()`
recorded exactly one `llm_request` middleware and no tools or hooks, that `provides_tools` is empty,
and that `plugin.yaml` parses with a strict loader that refuses duplicate keys. The manifest key
`provides_middleware` is what the validator compares against the middleware `register()` declares, which is
why the manifest carries it.
The same test drives the callback through `hermes_cli.middleware.apply_llm_request_middleware` when a
Hermes install is importable, with the full keyword set that `agent/turn_api_request.py:143-148` passes,
and asserts the callback actually runs; a second case registers a callback declared with only `request`
and asserts the dispatcher skips it, which is the trap this ticket exists to avoid.
The strip test builds a system message in the shape `agent/prompt_builder.py:1395-1420` emits, heading,
prose and tags included, plus the single-query variant and a demoted `[names only]` line, and asserts
the whole section is gone, the `kept` rows come back in the `Partition`, the demoted names are kept
with empty descriptions and never routed even when named in `deferredSkills`, and a failure path
leaves the message byte-identical.

## Acceptance criteria

- [ ] `hermes plugins validate` exits 0 on the repository path.
- [ ] The plugin loads with zero tools and one `llm_request` middleware.
- [ ] Every config key from the spec table exists in `config_schema` with its default and bound,
      including `eagerSkills` and `deferredSkills` as string lists.
- [ ] `plugin.yaml` declares `provides_middleware: ["llm_request"]`, and `hermes plugins validate`
      exits 0; the expected unknown-field warning appears once at load and is not a failure.
- [ ] A demoted `[names only]` line is parsed: its names are in `Partition.kept` with empty
      descriptions, never in `routed`, and the rerank documents never contain them.
- [ ] A shadowed-copy note `(A copy of ... is not listed: ...)` is removed with the section, and none
      of its names appears in `kept`, `routed` or the injected block.
- [ ] A mutation check: with `strip_catalog` disabled, the test that asserts the catalog is gone
      fails, and it is restored afterwards.
- [ ] `strip_catalog` is idempotent and returns the same object content when nothing matches.
- [ ] Partition boundaries: both lists empty routes everything; an eager name is kept and routed never;
      a deferred name is routed; a name in both is kept with one note; an unknown name is reported and
      changes no row; a non-empty deferred list keeps every other skill listed.
- [ ] The middleware callback returns `None` when `enabled` is false and when it raises internally.
- [ ] `load()` rejects `http://example.com:8012` and `http://user:pw@127.0.0.1:8012`, and accepts
      `http://127.0.0.1:8012` and `https://embed.example.com`.

## Notes

- `tests/` uses `pytest` from whichever interpreter has it; the plugin itself is stdlib-only.
- Do not import `hermes_cli` internals at module import time; import only inside functions so the
  unit tests run without a Hermes install on `sys.path`.
