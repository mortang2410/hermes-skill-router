# T8 Display the suggested skill names

Status: ready for agent. Repo: `hermes-skill-router`. Work root: `docs/work/router-sync/`.

## Delivers

A names-only display of the last routed decision in the Hermes desktop app and in the Ink TUI: skill
names plus the record's turn id and time as a caption, with no descriptions or request text. The record
is written by T5; this ticket reads it and paints it, and adds no text to any provider request.

## Covers

Story S10. Requirement R12. Modules M12, M13, M14. Contract C9, which carries the record shape
beside the rewrite decision.

## Exclusive files

- `dashboard/manifest.json`, `dashboard/plugin_api.py`, `dashboard/dist/index.js`
- `desktop/plugin.js`
- `tui/skill-router.mjs`
- `tests/test_display_surfaces.py`

## Blockers

T5, for the record writer and one live routed decision to read.

## What it must do

1. Read the record T5 writes: `{"schema": 1, "at": <ISO 8601 UTC>, "session_id", "turn_id",
   "status": "routed" | "inert", "names": [str, ...]}` at `cfg.suggestions_path`. `names` holds the
   selected suggestions only, never a kept row, so the pane and the widget show what the router picked,
   not what the block always lists. They render what is there and never write to the record. An absent
   or unreadable record renders idle, and `status: "routed"` with an empty list means the turn
   suggested nothing.
2. Desktop half: `desktop/plugin.js` default-exports a Hermes desktop plugin whose `register(ctx)`
   contributes one `panes` area entry, right placement, 260 px wide, title `Skills routed`, rendering
   a collapsed-by-default list of the names from `ctx.rest('/suggestions')`, refreshed on mount and on
   the plugin event `plugin.hermes-skill-router.suggestions.updated`. It must stay inside the
   `@hermes/plugin-sdk` surface the admission lint allows: no prototype patching, no dynamic code, no
   remote imports. Hermes copies this half into `$HERMES_HOME/desktop-plugins/hermes-skill-router/`
   when the plugin folder lands in a plugins root.
3. Backend half: `dashboard/manifest.json` is `{"name": "hermes-skill-router", "api": "plugin_api.py",
   "tab": {"hidden": true}}` and `plugin_api.py` exposes `GET /suggestions`, mounted at
   `/api/plugins/hermes-skill-router/suggestions`, returning the record or
   `{"status": "idle", "names": []}` when the file is absent or unreadable. Read-only; it never
   returns the user's text. The route imports only when the plugin is enabled in the profile's
   `plugins.enabled` list, so the pane shows a plain "enable the plugin" line rather than an error when
   the route answers 404. The hidden tab is required: the plugin ships no web dashboard UI, so without
   the flag the web dashboard shows a dead nav entry. Hiding the tab does not stop the client from
   requesting the manifest's default entry (`web/src/plugins/usePlugins.ts` injects every manifest's
   script, and `hermes_cli/web_server_dashboard.py` defaults the entry to `dist/index.js`), so the plugin
   also ships `dashboard/dist/index.js`, a no-op ES module that registers nothing, so that request
   succeeds and the console stays clean. Do not patch Hermes for this.
4. The desktop half is enabled separately from the Python backend, by the plugin's own toggle in the
   desktop app's Capabilities panel; both gates are documented by T7. When that toggle is off Hermes
   never calls `register()` (`apps/desktop/src/contrib/runtime-loader.ts:389-395`), so the pane does
   not exist and no code in this plugin can report the toggle. The pane reports only the missing
   backend route, and the README explains the other gate.
5. The rewrite broadcasts `plugin.hermes-skill-router.suggestions.updated` with the turn id after a
   record write, best effort, so a connected desktop window refreshes immediately.
6. TUI half: Hermes gives plugins no display hook in the TUI, so this half is a user widget the
   plugin ships rather than a plugin contribution. `tui/skill-router.mjs` default-exports `register(sdk)`
   and defines an ambient widget `id: "skills-routed"`, `mode: "ambient"`. Hermes passes user widgets
   only the widget SDK, so the widget reads the record file itself with Node filesystem APIs: user
   modules are imported in the TUI process and run with its privileges. It takes the record path as its
   launch argument, `/skills-routed <path>`, defaults to
   `~/.cache/hermes-skill-router/suggestions.json`, and puts that in its help line. It reads the
   record's names and caption fields only, never plugin config, and re-reads the file every 500 ms. It renders a `Dialog` with an
   `sdk.Accordion` whose title carries the count and whose body lists the record's names one per line,
   landing each update through `sdk.updateWidget`, which no-ops after the widget closes. It uses theme
   tones only, keeps its width fixed, and shows one line when the record is missing, corrupt or `inert`.
   Its install step is a symlink from the repository into `$HERMES_HOME/tui-widgets/`, documented by T7.
7. Failure policy: an unreadable or corrupt record renders idle in both clients; neither client raises
   out of a read or a render, and neither ever changes a request.

## Verification

Unit level:

```bash
python3 -m pytest tests/test_display_surfaces.py -q
node --check desktop/plugin.js
node --check tui/skill-router.mjs
hermes plugins validate .
```

`tests/test_display_surfaces.py` covers the route answer for a record and for a missing record, the
exact schema it passes through, and the widget reader as a pure function over a record file, including
a corrupt file and an `inert` record. The record writer, its cap, its removal on `showSuggestions`
false and the zero-token comparison against the no-display build are T5's `tests/test_display.py`.

Live, with `127.0.0.1:8012` and `127.0.0.1:8010` up:

```bash
ln -sfn /home/wilder/git-repos/hermes-skill-router ~/.hermes/plugins/hermes-skill-router
ln -sfn /home/wilder/git-repos/hermes-skill-router/tui/skill-router.mjs ~/.hermes/tui-widgets/skill-router.mjs
hermes -z "draft a git commit message for a bug fix"
```

Then enable the plugin twice: the profile's `plugins.enabled` entry for the Python backend, and the
plugin's own toggle in the desktop app's Capabilities panel for the copied desktop half. Open the
`Skills routed` pane and read the names. Run `hermes --tui` and `/skills-routed`, then relaunch it to
collapse. Stop `8012`, run the same turn again, and both surfaces must show the inert state with no
names. Turn `showSuggestions` off with a record present, and both surfaces must go idle. Turn the
desktop toggle off and the pane must be absent, not stale.

## Acceptance criteria

- [ ] The route returns the record, or `{"status": "idle", "names": []}` when the file is missing or
      corrupt, and passes the record schema through unchanged.
- [ ] The desktop pane lists the names after a routed turn and shows the inert state after a turn with
      the embeddings endpoint stopped.
- [ ] With the desktop toggle off the pane is absent; with the Python backend disabled the pane shows
      the enable line for the missing route.
- [ ] `/skills-routed` docks the widget, it lists the same names, collapses on a second launch, and
      follows the next routed turn within a second.
- [ ] `/skills-routed <path>` reads a custom record file.
- [ ] The dashboard manifest hides its tab, a test asserts the three keys, and
      `dashboard/dist/index.js` exists as a valid no-op module so the dashboard's default entry request
      resolves without a console error.
- [ ] `hermes plugins validate .` accepts the tree and lints `desktop/plugin.js` with the static
      admission lint, which is a tripwire for selected constructs and not an SDK allowlist, so the file
      is also read against the SDK docs.
- [ ] README (T7) documents the pane, both enable steps, the widget install step and its path argument,
      and `showSuggestions`, `suggestionsPath`, `maxDisplayNames`.

## Notes

- Hermes has no per-message transcript row a plugin can contribute: desktop contributions target
  panes and bars, and the TUI offers docked widget apps, so the display is a pane plus a docked card
  rather than a chip under each answer. The classic prompt_toolkit TUI has no plugin display surface
  at all and is out of scope.
- The newest record wins and carries its session and turn id; a second session, a `/btw` turn or a
  background review can replace it, and the display says which turn it came from through a muted
  caption line under the heading, `turn <turn_id> · <local HH:MM>`, painted only for a routed record
  and suppressed while loading, for a missing route or for a read failure. That scope is decided and
  tested in T5.
- The surfaces are called `Skills routed` because they show routed suggestions; kept rows are visible
  to the model in the block but deliberately absent from the display, so the pane cannot present a
  pinned or deferred-excluded skill as a suggestion.
- The widget reads the file with `node:fs` because user widget modules are imported in the TUI process
  and run with its privileges (`ui-tui/src/sdk/userWidgets.ts` dynamic import); it never reads plugin
  config, which is why the disabled state is the missing record and the path is an argument.
