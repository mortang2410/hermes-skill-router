# Implementer brief 02: the manifest declares the hook and the floor, and the host tests read the running install

## Your seat and route

- You are the implementer for ticket 02 only. You build, you test. Do not commit: ticket 03 runs
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
- Ticket: kata `personal` issue `1516` (this feature's ticket 02).
- Ticket 01 has landed and is committed: `__init__.py` registers a `post_compaction` hook through a
  module-level callback named `on_post_compaction`, and `hermes_skill_router/search.py` exposes
  `remember_dump`, `dumped`, `expire_dump` and `clear_dumps`. Read both before you write a test
  against them.

## Files you own, exclusively

- `plugin.yaml`
- `tests/test_plugin_registration.py`
- `tests/test_middleware_contract.py`

## Skeleton rows you must satisfy

Modules M28 (`plugin.yaml`), M30 (`tests/test_plugin_registration.py`, which owns the resolver) and
M31 (`tests/test_middleware_contract.py`, which consumes the resolver and adds none of its own).
Contracts C22 (the registration surface, as it is asserted), C23 (the manifest declaration) and C26
(the host tree resolution, provisional: its shape is yours to refine). Requirements R4 and R5.

## What changes, with today's anchors

`plugin.yaml`

- `version: "0.1.0"` (`:4`) becomes `version: "0.2.0"`, still a quoted string.
- `requires_hermes: ">=0.22.2"` (`:8`) becomes `">=0.22.3"`.
- `provides_hooks: []` (`:10`) becomes `provides_hooks: [post_compaction]`.
- Nothing else changes. `provides_tools`, `provides_middleware` and every config key stay as they are.

`tests/test_plugin_registration.py`

- `:238` asserts `provides_hooks == []` and `:206` asserts `ctx.hooks == []`. Both must now assert the
  hook the code actually registers, by name.
- The version literal assertion at `:236-237` must follow the new value.
- `:139-149` defaults its host source tree to `Path.home() / ".hermes" / "hermes-agent"`. That
  directory is a stale copy with 41 hooks and no `post_compaction`, so a hook assertion there would be
  a test of a host the code cannot serve.
- Replace that default with the resolver this module owns (C26): ask the `hermes` executable on
  `PATH` for its own report. `hermes --version` prints a line `Install directory: <path>`; parse that
  line, keep an explicit override for a caller that names a tree, and `pytest.skip` with a reason when
  neither resolves. The reason names the path or the command that was looked for.
- Cache one resolution per test session and bound the launcher call with a short timeout, so a broken
  launcher cannot hang the suite. The existing precedent for locating the launcher is `:344-349`.
- Add one assertion that the resolved host's own hook catalogue contains `post_compaction`, so the
  registration test proves the host can deliver what the plugin registers. The catalogue is
  `VALID_HOOKS` in `hermes_cli/plugins.py` (line 207 of the running install).
- Add one test that forces the missing-host branch: make the resolver see no launcher or an empty
  report, then assert the skip fires and its reason names what was tried. Use `pytest.raises` on the
  skip exception or `pytest.skip`'s own exception class, not a bare `try` around it.

`tests/test_middleware_contract.py`

- `:177-187` defaults the same stale directory. Replace it with the resolver imported from the
  registration module, so exactly one module asks the launcher. Do not write a second resolver.
- The contract it pins against the host stays as it is: the var-keyword callback requirement and the
  dispatcher call shape.

## How to verify

```sh
cd /home/wilder/git-repos/hermes-skill-router
python3 -m pytest tests/test_plugin_registration.py tests/test_middleware_contract.py -q
python3 -m pytest -q
python3 -c "import yaml;d=yaml.safe_load(open('plugin.yaml'));print(d['version'],d['requires_hermes'],d['provides_hooks'])"
```

The last command must print `0.2.0 >=0.22.3 ['post_compaction']`. The middle one must show the two new
host tests passing rather than skipping, and the missing-host test passing.

## Your report

Reply with: each acceptance criterion from the ticket with the test name that pins it; the three
command outputs verbatim; the exact host path your resolution found and the command that produced it;
what your forced-missing-host test did to force it; and a one-line statement that no file outside your
three changed. Do not claim a criterion you did not pin with a test.
