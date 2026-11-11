"""Ticket T8: the two names-only display surfaces and the record they share.

Five things are pinned here, because each one is a way the display could quietly stop being a
display:

* the backend route answers the record exactly as the writer stored it, and answers
  ``{"status": "idle", "names": []}`` for a missing, corrupt, wrong-schema or ``inert`` file;
* the desktop pane's contribution and its collapsed/expanded labels, exercised through the real
  ``desktop/plugin.js`` with the SDK replaced by recording stubs;
* the TUI widget's reader, labels and render tree, exercised through the real
  ``tui/skill-router.mjs`` the same way;
* the dashboard entry module the loader injects is comment-only, parses, and imports without
  registering anything or touching a global;
* neither surface reads or shows anything but the record's names and identifiers — a record that
  carries a request message or a ``text`` key renders nothing from it, and no route answer holds a
  key outside the seven-field schema.

Both surfaces are runtimes this test cannot host (a React renderer, an Ink app), so the JS halves
are loaded in Node with stub modules on ``node_modules`` and a stub SDK/context. That exercises the
shipped files rather than a Python re-implementation of their logic.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router.config import DEFAULTS  # noqa: E402
from hermes_skill_router.display import FIELDS, read  # noqa: E402

DASHBOARD_DIR = ROOT / "dashboard"
MANIFEST_PATH = DASHBOARD_DIR / "manifest.json"
PLUGIN_API_PATH = DASHBOARD_DIR / "plugin_api.py"
INDEX_ENTRY_PATH = DASHBOARD_DIR / "dist" / "index.js"
DESKTOP_PLUGIN_PATH = ROOT / "desktop" / "plugin.js"
TUI_WIDGET_PATH = ROOT / "tui" / "skill-router.mjs"

NODE = shutil.which("node")

DEFAULT_RECORD_PATH = os.path.expanduser(DEFAULTS["suggestionsPath"])
NAMESPACE = "/api/plugins/hermes-skill-router"

RECORD = {
    "schema": 2,
    "at": "2026-11-10T07:20:31Z",
    "session_id": "session-1",
    "turn_id": "turn-1",
    "status": "routed",
    "origin": "turn",
    "names": ["alpha", "beta", "gamma"],
}

#: Text a request could carry; a display must never read it, return it or paint it.
REQUEST_TEXT = "draft a git commit message for a bug fix, verbatim user text"

#: The record's UTC timestamp, the local clock time both surfaces must derive from it, and the one
#: caption line that pairs it with the turn id.
CAPTION_STAMP = "2026-11-10T07:20:31Z"
CAPTION_TIME = (
    datetime.fromisoformat(CAPTION_STAMP.replace("Z", "+00:00")).astimezone().strftime("%H:%M")
)
CAPTION = f"turn turn-1 · {CAPTION_TIME}"


def write_record(path: Path, **overrides) -> Path:
    """Write one record file at *path* with *overrides* applied."""
    entry = dict(RECORD)
    entry.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def plugin_api():
    """Import ``dashboard/plugin_api.py`` the way Hermes' dashboard loader does: by path."""
    pytest.importorskip("fastapi", reason="the dashboard backend is a FastAPI router")
    name = "hermes_dashboard_plugin_hermes_skill_router_t8"
    spec = importlib.util.spec_from_file_location(name, PLUGIN_API_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_the_manifest_names_the_plugin_hides_the_tab_and_declares_the_backend_file():
    # The plugin ships no web dashboard UI: its only dashboard half is the /suggestions route, so
    # the manifest hides the tab discovery would otherwise invent (a nav entry whose dist/index.js
    # would not exist). The Desktop pane is a separate runtime plugin and does not use this file.
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == {
        "name": "hermes-skill-router",
        "api": "plugin_api.py",
        "tab": {"hidden": True},
    }


def test_the_dashboard_entry_module_is_a_comment_only_no_op():
    # The dashboard injects the manifest's default entry as a classic script, so the file must stay
    # comment-only: no import/export syntax (a classic script rejects it) and no statement at all.
    source = INDEX_ENTRY_PATH.read_text(encoding="utf-8")

    assert INDEX_ENTRY_PATH.is_file()
    assert "no web dashboard UI" in source
    assert re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL).strip() == ""


def test_the_dashboard_entry_module_has_no_side_effects_in_node(tmp_path):
    if NODE is None:
        pytest.skip("node is required to load the dashboard entry module")
    driver = tmp_path / "entry-driver.mjs"
    driver.write_text(ENTRY_DRIVER, encoding="utf-8")

    result = subprocess.run(
        [NODE, str(driver), str(INDEX_ENTRY_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["added"] == []
    assert observed["removed"] == []
    assert observed["register"] is False
    assert set(observed["exported"]) <= {"default", "module.exports"}


def test_the_router_exposes_one_read_only_route(plugin_api):
    routes = [(route.path, sorted(route.methods)) for route in plugin_api.router.routes]

    assert routes == [("/suggestions", ["GET"])]


def test_the_route_answers_the_record_it_reads(tmp_path, plugin_api):
    path = write_record(tmp_path / "suggestions.json")

    answer = plugin_api.answer(path)

    assert list(answer) == list(FIELDS)
    assert answer == read(path)
    assert answer == {
        "schema": 2,
        "at": "2026-11-10T07:20:31Z",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "status": "routed",
        "origin": "turn",
        "names": ["alpha", "beta", "gamma"],
    }


def test_the_route_normalizes_a_schema_1_record(tmp_path, plugin_api):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "at": "2026-11-09T06:00:00Z",
                "session_id": "session-0",
                "turn_id": "turn-0",
                "status": "routed",
                "names": ["legacy"],
            }
        ),
        encoding="utf-8",
    )

    answer = plugin_api.answer(path)

    assert list(answer) == list(FIELDS)
    assert answer["schema"] == 2
    assert answer["origin"] == "turn"
    assert answer["names"] == ["legacy"]


def test_the_route_answers_idle_when_the_record_is_missing(tmp_path, plugin_api):
    assert plugin_api.answer(tmp_path / "nothing-here.json") == {"status": "idle", "names": []}


@pytest.mark.parametrize(
    "body",
    [
        "{not json at all",
        "[]",
        json.dumps({"schema": 2, "status": "routed", "names": ["alpha"]}),
        json.dumps({"schema": 1, "status": "paused", "names": ["alpha"]}),
        json.dumps({"schema": 1, "status": "routed", "names": "alpha"}),
        json.dumps({"schema": 1, "status": "routed", "names": [1, 2]}),
    ],
)
def test_the_route_answers_idle_for_a_corrupt_record(tmp_path, plugin_api, body):
    path = tmp_path / "suggestions.json"
    path.write_text(body, encoding="utf-8")

    assert plugin_api.answer(path) == {"status": "idle", "names": []}


def test_the_route_passes_the_inert_status_through(tmp_path, plugin_api):
    path = write_record(tmp_path / "suggestions.json", status="inert", names=[])

    assert plugin_api.answer(path) == {
        "schema": 2,
        "at": "2026-11-10T07:20:31Z",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "status": "inert",
        "origin": "turn",
        "names": [],
    }


def test_the_route_projects_origin_for_a_search_record(tmp_path, plugin_api):
    path = write_record(tmp_path / "suggestions.json", origin="search", turn_id="")

    answer = plugin_api.answer(path)

    assert answer["origin"] == "search"
    assert answer["turn_id"] == ""
    assert list(answer) == list(FIELDS), "origin is projected with no added key"


def test_the_route_returns_no_request_content_even_from_a_tampered_record(tmp_path, plugin_api):
    path = tmp_path / "suggestions.json"
    path.write_text(
        json.dumps(
            {
                **RECORD,
                "text": REQUEST_TEXT,
                "messages": [{"role": "user", "content": REQUEST_TEXT}],
                "query": REQUEST_TEXT,
                "system": REQUEST_TEXT,
            }
        ),
        encoding="utf-8",
    )

    answer = plugin_api.answer(path)
    body = json.dumps(answer)

    assert list(answer) == list(FIELDS)
    assert "text" not in answer and "messages" not in answer and "system" not in answer
    assert REQUEST_TEXT not in body


def test_the_route_never_writes_the_record(tmp_path, plugin_api):
    path = write_record(tmp_path / "suggestions.json")
    before = path.read_bytes()

    plugin_api.answer(path)

    assert path.read_bytes() == before
    assert path.read_text(encoding="utf-8") == json.dumps(RECORD)


def test_settings_from_reads_the_documented_entry(plugin_api):
    config = {"plugins": {"entries": {"hermes-skill-router": {"settings": {"suggestionsPath": "/tmp/x.json"}}}}}

    assert plugin_api.settings_from(config) == {"suggestionsPath": "/tmp/x.json"}


def test_settings_from_falls_back_to_the_legacy_subtree_and_tolerates_junk(plugin_api):
    legacy = {"plugins": {"entries": {"hermes-skill-router": {"config": {"showSuggestions": False}}}}}

    assert plugin_api.settings_from(legacy) == {"showSuggestions": False}
    assert plugin_api.settings_from({}) == {}
    assert plugin_api.settings_from(None) == {}
    assert plugin_api.settings_from({"plugins": {"entries": {"other-plugin": {"settings": {}}}}}) == {}


def test_the_route_reads_the_path_the_profile_configures(tmp_path, plugin_api, monkeypatch):
    path = write_record(tmp_path / "custom.json", names=["delta"])
    monkeypatch.setattr(plugin_api, "settings", lambda: {"suggestionsPath": str(path)})

    assert plugin_api.answer() == read(path)
    assert plugin_api.answer()["names"] == ["delta"]


def test_an_empty_config_targets_the_documented_default_path(plugin_api, monkeypatch):
    monkeypatch.setattr(plugin_api, "settings", dict)

    assert plugin_api.configured().suggestions_path == DEFAULTS["suggestionsPath"]


def test_the_route_is_mounted_under_the_plugin_namespace(tmp_path, plugin_api, monkeypatch):
    pytest.importorskip("httpx", reason="fastapi's test client speaks over httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    path = write_record(tmp_path / "suggestions.json")
    monkeypatch.setattr(plugin_api, "settings", lambda: {"suggestionsPath": str(path)})
    app = FastAPI()
    app.include_router(plugin_api.router, prefix=NAMESPACE)
    client = TestClient(app)

    response = client.get(f"{NAMESPACE}/suggestions")

    assert response.status_code == 200
    assert response.json() == read(path)

    monkeypatch.setattr(plugin_api, "settings", lambda: {"suggestionsPath": str(tmp_path / "gone.json")})

    idle = client.get(f"{NAMESPACE}/suggestions")

    assert idle.status_code == 200
    assert idle.json() == {"status": "idle", "names": []}


def test_the_route_answers_idle_when_the_reader_is_unavailable(plugin_api, monkeypatch):
    def broken(target=None):
        raise RuntimeError("no record module")

    monkeypatch.setattr(plugin_api, "answer", broken)

    assert plugin_api.suggestions() == {"status": "idle", "names": []}


# ---------------------------------------------------------------------------
# The JavaScript halves. Node loads the shipped files with stubbed SDK modules.
# ---------------------------------------------------------------------------

DESKTOP_SDK_STUB = """
export const atom = initial => {
  let value = initial

  return {
    get: () => value,
    set: next => {
      value = typeof next === 'function' ? next(value) : next
    }
  }
}
export const useValue = store => store.get()
export const PANES_AREA = 'panes'
export const DisclosureCaret = 'DisclosureCaret'
export const useQuery = options => {
  globalThis.__t8Query = options

  return globalThis.__t8Result ?? {}
}
export const queryClient = {
  invalidateQueries: async options => {
    globalThis.__t8Invalidated = options
  }
}
"""

JSX_STUB = """
export const Fragment = 'Fragment'
export const jsx = (type, props) => ({ props: props ?? {}, type })
export const jsxs = jsx
"""

SDK_PACKAGE_STUB = json.dumps({"name": "@hermes/plugin-sdk", "type": "module", "main": "./index.mjs"})

REACT_PACKAGE_STUB = json.dumps(
    {
        "name": "react",
        "type": "module",
        "exports": {".": "./index.mjs", "./jsx-runtime": "./jsx-runtime.mjs"},
    }
)

REACT_INDEX_STUB = """
export const useState = initial => [typeof initial === 'function' ? initial() : initial, () => {}]
export const useEffect = () => {}
"""

#: Shared tree helpers: our jsx/h stubs produce plain `{type, props}` objects.
TREE_HELPERS = """
const childrenOf = node => (node && typeof node === 'object' ? node.props?.children : undefined)
const text = node => {
  if (node === null || node === undefined || node === false) return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(text).join(' ')
  if (typeof node === 'object') return text(childrenOf(node))

  return ''
}
const find = (node, predicate) => {
  if (!node || typeof node !== 'object') return null
  if (predicate(node)) return node
  const children = Array.isArray(childrenOf(node)) ? childrenOf(node) : [childrenOf(node)]
  for (const child of children) {
    const hit = find(child, predicate)
    if (hit) return hit
  }

  return null
}
const types = node => {
  const seen = []
  const walk = current => {
    if (!current || typeof current !== 'object') return
    if (typeof current.type === 'string') seen.push(current.type)
    const children = Array.isArray(childrenOf(current)) ? childrenOf(current) : [childrenOf(current)]
    children.forEach(walk)
  }
  walk(node)

  return seen
}
"""

DESKTOP_DRIVER = (
    """
import plugin, {
  ENABLE_LINE,
  PANE_TITLE,
  READ_ERROR_LINE,
  SUGGESTIONS_QUERY_KEY,
  UPDATED_EVENT,
  captionFor,
  isMissingRoute,
  shortTime,
  summary
} from './plugin.mjs'

"""
    + TREE_HELPERS
    + """
const NAMES = ['alpha', 'beta', 'gamma']
const STAMP = '2026-11-10T07:20:31Z'
const out = {}
const registered = []
const events = []
const rest = []
const routeBody = { at: STAMP, names: NAMES, session_id: 'session-1', status: 'routed', turn_id: 'turn-1' }
const ctx = {
  register: contribution => {
    registered.push(contribution)

    return () => {}
  },
  onEvent: (type, listener) => {
    events.push({ listener, type })

    return () => {}
  },
  rest: async path => {
    rest.push(path)

    return routeBody
  }
}

plugin.register(ctx)

const contribution = registered[0]
out.plugin = { id: plugin.id, name: plugin.name }
out.contribution = {
  area: contribution.area,
  data: contribution.data,
  id: contribution.id,
  title: contribution.title
}
out.registered = registered.length
out.events = events.map(event => event.type)
out.queryKey = SUGGESTIONS_QUERY_KEY
out.lines = { enable: ENABLE_LINE, readError: READ_ERROR_LINE, title: PANE_TITLE, updated: UPDATED_EVENT }

out.summary = {
  empty: summary({ at: STAMP, names: [], status: 'routed', turn_id: 'turn-1' }),
  idle: summary(null),
  inert: summary({ at: STAMP, names: ['ignored'], status: 'inert', turn_id: 'turn-1' }),
  routed: summary({ at: STAMP, names: NAMES, session_id: 'session-1', status: 'routed', turn_id: 'turn-1' }),
  search: summary({ at: STAMP, names: NAMES, origin: 'search', status: 'routed', turn_id: '' }),
  tampered: summary({
    at: STAMP,
    messages: [{ content: 'SECRET' }],
    names: ['alpha'],
    status: 'routed',
    text: 'SECRET',
    turn_id: 'turn-1'
  })
}

// The caption is built from the record's identifiers only, and drops a half it cannot read.
out.captions = {
  badAt: captionFor({ at: 'not-a-date', names: NAMES, status: 'routed', turn_id: 'turn-1' }),
  idle: captionFor(null),
  inert: captionFor({ at: STAMP, names: [], status: 'inert', turn_id: 'turn-1' }),
  noAt: captionFor({ names: NAMES, status: 'routed', turn_id: 'turn-1' }),
  noTurn: captionFor({ at: STAMP, names: NAMES, status: 'routed' }),
  routed: captionFor({ at: STAMP, names: NAMES, status: 'routed', turn_id: 'turn-1' }),
  search: captionFor({ at: STAMP, names: NAMES, origin: 'search', status: 'routed', turn_id: '' })
}
out.times = {
  bad: shortTime('not-a-date'),
  blank: shortTime(''),
  good: shortTime(STAMP),
  missing: shortTime(undefined)
}


out.failure = {
  missing: isMissingRoute(new Error('404: Not Found')),
  none: isMissingRoute(null),
  server: isMissingRoute(new Error('500: boom')),
  statusCode: isMissingRoute({ statusCode: 404 })
}

// The real render path: run the contribution's component with stubbed hooks.
globalThis.__t8Result = { data: routeBody }
const element = contribution.render()
const render = () => element.type(element.props)
const first = render()
const captionNode = find(
  first,
  node => typeof node.props?.children === 'string' && node.props.children.startsWith('turn turn-1')
)
out.captionStyle = { className: captionNode?.props?.className ?? '' }
out.collapsed = {
  leaked: NAMES.filter(name => text(first).includes(name)),
  text: text(first)
}
find(first, node => typeof node.props?.onClick === 'function').props.onClick()
const second = render()
out.expanded = {
  shown: NAMES.filter(name => text(second).includes(name)),
  text: text(second)
}

// The pane's data really comes from the plugin's own route.
const options = globalThis.__t8Query
out.query = { body: await options.queryFn(), calls: rest.slice(), key: options.queryKey }

// A missing backend route (the plugin is disabled) paints the enable line and no names.
globalThis.__t8Result = { error: new Error('404: Not Found') }
find(render(), node => typeof node.props?.onClick === 'function').props.onClick()
const missing = render()
out.missingRoute = { leaked: NAMES.filter(name => text(missing).includes(name)), text: text(missing) }

// Any other failure paints the short read line.
globalThis.__t8Result = { error: new Error('500: boom') }
out.readError = { text: text(render()) }

// A record carrying request text renders none of it, even expanded.
globalThis.__t8Result = {
  data: {
    at: STAMP,
    messages: [{ content: 'SECRET' }],
    names: NAMES,
    status: 'routed',
    text: 'SECRET',
    turn_id: 'turn-1'
  }
}
find(render(), node => typeof node.props?.onClick === 'function').props.onClick()
const tampered = render()
out.tampered = {
  leaked: text(tampered).includes('SECRET'),
  shown: NAMES.filter(name => text(tampered).includes(name))
}

// The first fetch paints no idle line or caption before the route answers.
globalThis.__t8Result = { isPending: true }
out.loading = { text: text(render()) }

// A search record paints the model-search marker where a turn paints its turn id.
globalThis.__t8Result = { data: { at: STAMP, names: NAMES, origin: 'search', status: 'routed', turn_id: '' } }
out.searchPane = { text: text(render()) }

// An inert record paints its idle line with no stale turn, and an idle answer has no caption.
globalThis.__t8Result = { data: { at: STAMP, names: [], status: 'inert', turn_id: 'turn-1' } }
out.inert = { text: text(render()) }
globalThis.__t8Result = { data: { status: 'idle', names: [] } }
out.idle = { text: text(render()) }

console.log(JSON.stringify(out))
"""
)

TUI_DRIVER = (
    """
import register, {
  ACCORDION_TITLE,
  DEFAULT_RECORD_PATH,
  POLL_MS,
  TITLE,
  WIDGET_ID,
  WIDTH,
  captionFor,
  expandPath,
  helpPath,
  readRecord,
  shortTime,
  summary
} from './skill-router.mjs'
import { writeFileSync } from 'node:fs'
import { homedir } from 'node:os'

"""
    + TREE_HELPERS
    + """
const NAMES = ['alpha', 'beta', 'gamma']
const theme = { color: { accent: 'accent', error: 'error', label: 'label', muted: 'muted', primary: 'primary', text: 'text' } }
const out = {}
const apps = []
const updates = []
const timers = []

globalThis.setInterval = (fn, ms) => {
  const timer = { cleared: false, fn, ms, unref() {} }
  timers.push(timer)

  return timer
}
globalThis.clearInterval = timer => {
  if (timer) timer.cleared = true
}

const sdk = {
  Accordion: 'Accordion',
  Box: 'Box',
  Dialog: 'Dialog',
  Text: 'Text',
  defineWidgetApp: app => {
    apps.push(app)

    return app
  },
  h: (type, props, ...children) => ({
    props: { ...(props ?? {}), children: children.length === 1 ? children[0] : children },
    type
  }),
  updateWidget: (app, fn) => {
    updates.push({ app: app.id, fn })
  }
}

register(sdk)

const app = apps[0]
out.registered = apps.length
out.widget = { help: app.help, id: app.id, mode: app.mode, usage: app.usage, width: app.width }
out.constants = {
  accordion: ACCORDION_TITLE,
  defaultPath: DEFAULT_RECORD_PATH,
  id: WIDGET_ID,
  poll: POLL_MS,
  title: TITLE,
  width: WIDTH
}

const record = extra => ({
  at: '2026-11-10T07:20:31Z',
  names: NAMES,
  schema: 1,
  session_id: 'session-1',
  status: 'routed',
  turn_id: 'turn-1',
  ...extra
})
const STAMP = '2026-11-10T07:20:31Z'
const dump = (file, value) => writeFileSync(file, JSON.stringify(value))
const render = path => app.render({ cols: 120, rows: 30, state: { path, record: readRecord(path) }, t: theme })
const dialogChildren = tree => (Array.isArray(tree.props.children) ? tree.props.children : [tree.props.children])
const captionNodeOf = tree => dialogChildren(tree)[0]
const captionOf = tree => text(captionNodeOf(tree))
const bodyOf = tree => {
  const children = dialogChildren(tree)

  return children[children.length - 1]
}

// Routed: the real init reads the file, starts the poll, and renders the card.
dump('routed.json', record({}))
const state = app.init('routed.json')
out.state = state
out.poll = timers.map(timer => timer.ms)
const routed = render('routed.json')
const routedCaption = captionNodeOf(routed)
out.routed = {
  accordion: {
    count: bodyOf(routed).props.count,
    defaultOpen: bodyOf(routed).props.defaultOpen,
    title: bodyOf(routed).props.title
  },
  body: text(bodyOf(routed)),
  bodyType: bodyOf(routed).type,
  caption: captionOf(routed),
  captionNode: { color: routedCaption?.props?.color, type: routedCaption?.type },
  dialog: { title: routed.props.title, width: routed.props.width },
  dialogType: routed.type
}
out.summary = summary(readRecord('routed.json'))

// The caption is built from the record's identifiers only, and drops a half it cannot read.
out.captions = {
  badAt: captionFor({ at: 'not-a-date', names: NAMES, status: 'routed', turn_id: 'turn-1' }),
  idle: captionFor(null),
  inert: captionFor({ at: STAMP, names: [], status: 'inert', turn_id: 'turn-1' }),
  noAt: captionFor({ names: NAMES, status: 'routed', turn_id: 'turn-1' }),
  noTurn: captionFor({ at: STAMP, names: NAMES, status: 'routed' }),
  routed: captionFor({ at: STAMP, names: NAMES, status: 'routed', turn_id: 'turn-1' })
}
out.times = {
  bad: shortTime('not-a-date'),
  blank: shortTime(''),
  good: shortTime(STAMP),
  missing: shortTime(undefined)
}

// A search record reads with origin search and paints the model-search marker.
out.searchCaption = captionFor({ at: STAMP, names: NAMES, origin: 'search', status: 'routed', turn_id: '' })
dump('search.json', record({ origin: 'search', schema: 2, turn_id: '' }))
const searchCard = render('search.json')
out.search = {
  caption: captionOf(searchCard),
  record: readRecord('search.json')
}
// A schema 2 turn record reads with origin turn.
dump('turn2.json', { ...record({}), schema: 2, origin: 'turn' })
out.schemaTwoTurn = readRecord('turn2.json')

// The poll re-reads the file and lands the result through updateWidget.
dump('routed.json', record({ names: ['delta', 'epsilon'] }))
timers[timers.length - 1].fn()
out.update = {
  app: updates[0]?.app,
  next: updates[0]?.fn(state)
}

// A relaunch replaces the timer instead of leaving the old poller behind.
const relaunched = app.init('')
out.relaunch = { cleared: timers[timers.length - 2]?.cleared, path: relaunched.path }
out.pollsAfterRelaunch = timers.map(timer => timer.ms)

// The states without names paint one line each, and no list.
dump('inert.json', record({ names: ['ignored'], status: 'inert' }))
const inert = render('inert.json')
out.inert = {
  body: text(bodyOf(inert)),
  bodyType: bodyOf(inert).type,
  caption: captionOf(inert),
  record: readRecord('inert.json')
}
dump('empty.json', record({ names: [] }))
const empty = render('empty.json')
out.empty = { body: text(bodyOf(empty)), bodyType: bodyOf(empty).type, caption: captionOf(empty) }
writeFileSync('corrupt.json', '{not json at all')
const corrupt = render('corrupt.json')
out.corrupt = {
  body: text(bodyOf(corrupt)),
  bodyType: bodyOf(corrupt).type,
  caption: captionOf(corrupt),
  record: readRecord('corrupt.json')
}
const missing = render('missing.json')
out.missing = {
  body: text(bodyOf(missing)),
  bodyType: bodyOf(missing).type,
  caption: captionOf(missing),
  record: readRecord('missing.json')
}

// Names only: request text in the file is never projected and never painted.
dump('tampered.json', record({ messages: [{ content: 'SECRET' }], text: 'SECRET' }))
const tampered = render('tampered.json')
out.tampered = {
  caption: captionOf(tampered),
  projection: readRecord('tampered.json'),
  rendered: text(dialogChildren(tampered)).includes('SECRET')
}

out.paths = {
  absolute: expandPath('/tmp/custom.json'),
  blank: expandPath('  '),
  home: expandPath('~/custom.json'),
  plain: expandPath('custom.json'),
  tildeDefault: expandPath('~/.cache/hermes-skill-router/suggestions.json')
}
out.helpPath = { default: helpPath(), other: helpPath('/tmp/custom.json') }
out.homedir = homedir()

console.log(JSON.stringify(out))
"""
)

ENTRY_DRIVER = """
import { pathToFileURL } from 'node:url'

const before = new Set(Object.keys(globalThis))
const namespace = await import(pathToFileURL(process.argv[2]).href)
const added = Object.keys(globalThis).filter(key => !before.has(key))

console.log(
  JSON.stringify({
    added,
    exported: Object.keys(namespace),
    register: 'registerPlugin' in globalThis,
    removed: [...before].filter(key => !(key in globalThis))
  })
)
"""


def run_node_driver(tmp_path: Path, driver: str, files: dict) -> dict:
    """Run one Node driver inside *tmp_path* and return the JSON object it printed."""
    assert NODE is not None, "node is required to load the shipped display halves"
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    driver_path = tmp_path / "driver.mjs"
    driver_path.write_text(driver, encoding="utf-8")

    result = subprocess.run(
        [NODE, str(driver_path)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines, result.stdout
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def desktop(tmp_path_factory):
    """The desktop half, registered and rendered through stubs of the SDK it is allowed to import."""
    if NODE is None:
        pytest.skip("node is required to load desktop/plugin.js")
    workspace = tmp_path_factory.mktemp("desktop-display")
    files = {
        "plugin.mjs": DESKTOP_PLUGIN_PATH.read_text(encoding="utf-8"),
        "node_modules/@hermes/plugin-sdk/index.mjs": DESKTOP_SDK_STUB,
        "node_modules/@hermes/plugin-sdk/package.json": SDK_PACKAGE_STUB,
        "node_modules/react/index.mjs": REACT_INDEX_STUB,
        "node_modules/react/jsx-runtime.mjs": JSX_STUB,
        "node_modules/react/package.json": REACT_PACKAGE_STUB,
    }
    return run_node_driver(workspace, DESKTOP_DRIVER, files)


@pytest.fixture(scope="module")
def tui(tmp_path_factory):
    """The TUI widget, registered against a recording SDK and rendered over real record files."""
    if NODE is None:
        pytest.skip("node is required to load tui/skill-router.mjs")
    workspace = tmp_path_factory.mktemp("tui-display")
    files = {"skill-router.mjs": TUI_WIDGET_PATH.read_text(encoding="utf-8")}
    return run_node_driver(workspace, TUI_DRIVER, files)


def test_the_desktop_half_registers_one_right_hand_pane(desktop):
    assert desktop["plugin"]["id"] == "hermes-skill-router"
    assert desktop["registered"] == 1
    assert desktop["contribution"] == {
        "area": "panes",
        "data": {"collapsible": True, "placement": "right", "width": "260px"},
        "id": "skills-routed",
        "title": "Skills routed",
    }
    assert desktop["events"] == ["plugin.hermes-skill-router.suggestions.updated"]
    assert desktop["queryKey"] == ["hermes-skill-router", "suggestions"]


def test_the_desktop_pane_is_collapsed_then_lists_the_names(desktop):
    assert "Skills routed (3)" in desktop["collapsed"]["text"]
    assert desktop["collapsed"]["leaked"] == []
    assert CAPTION in desktop["collapsed"]["text"]
    assert desktop["expanded"]["shown"] == ["alpha", "beta", "gamma"]
    assert "Skills routed (3)" in desktop["expanded"]["text"]


def test_the_desktop_pane_labels_each_record_state(desktop):
    summary = desktop["summary"]

    assert summary["routed"] == {
        "caption": CAPTION,
        "count": 3,
        "heading": "Skills routed (3)",
        "names": ["alpha", "beta", "gamma"],
        "note": "",
        "status": "routed",
    }
    assert summary["empty"]["heading"] == "Skills routed (0)"
    assert summary["empty"]["note"] == "The last routed turn suggested nothing."
    assert summary["empty"]["caption"] == CAPTION
    assert summary["inert"]["names"] == []
    assert summary["inert"]["note"] == "Router inactive: no suggestions this turn."
    assert summary["inert"]["caption"] == ""
    assert summary["idle"]["status"] == "idle"
    assert summary["idle"]["note"] == "No suggestions recorded yet."
    assert summary["idle"]["caption"] == ""
    assert summary["tampered"]["names"] == ["alpha"]


def test_the_desktop_pane_paints_one_muted_turn_caption(desktop):
    captions = desktop["captions"]
    times = desktop["times"]

    assert captions["routed"] == CAPTION
    assert captions["noTurn"] == CAPTION_TIME
    assert captions["noAt"] == "turn turn-1"
    assert captions["badAt"] == "turn turn-1"
    assert captions["inert"] == ""
    assert captions["idle"] == ""
    assert times == {"bad": "", "blank": "", "good": CAPTION_TIME, "missing": ""}
    assert "text-muted-foreground" in desktop["captionStyle"]["className"]
    # The caption is painted while collapsed, so the turn is readable without expanding.
    assert CAPTION in desktop["collapsed"]["text"]
    assert "turn-1" not in desktop["missingRoute"]["text"]
    assert "turn-1" not in desktop["readError"]["text"]
    assert "turn-1" not in desktop["loading"]["text"]
    assert "turn-1" not in desktop["inert"]["text"]
    assert CAPTION_TIME not in desktop["inert"]["text"]
    assert "turn-1" not in desktop["idle"]["text"]


def test_the_desktop_pane_reads_the_plugin_route(desktop):
    assert desktop["query"]["calls"] == ["/suggestions"]
    assert desktop["query"]["body"] == {
        "at": CAPTION_STAMP,
        "names": ["alpha", "beta", "gamma"],
        "session_id": "session-1",
        "status": "routed",
        "turn_id": "turn-1",
    }


def test_the_desktop_pane_reports_a_missing_route_and_a_failed_read(desktop):
    assert desktop["failure"] == {"missing": True, "none": False, "server": False, "statusCode": True}
    assert desktop["lines"]["enable"] in desktop["missingRoute"]["text"]
    assert desktop["missingRoute"]["leaked"] == []
    assert desktop["lines"]["readError"] in desktop["readError"]["text"]
    assert desktop["lines"]["enable"] not in desktop["readError"]["text"]


def test_the_desktop_pane_shows_no_request_text(desktop):
    assert desktop["tampered"]["leaked"] is False
    assert desktop["tampered"]["shown"] == ["alpha", "beta", "gamma"]
    assert "SECRET" not in desktop["summary"]["tampered"]["heading"]


def test_the_desktop_pane_paints_no_idle_line_before_the_route_answers(desktop):
    assert "Skills routed" in desktop["loading"]["text"]
    assert "No suggestions recorded yet." not in desktop["loading"]["text"]


def test_the_desktop_pane_marks_a_model_search(desktop):
    marker = f"model search · {CAPTION_TIME}"

    assert desktop["captions"]["search"] == marker
    assert desktop["summary"]["search"]["caption"] == marker
    assert desktop["summary"]["search"]["names"] == ["alpha", "beta", "gamma"]
    assert "model search" in desktop["searchPane"]["text"]
    assert "turn turn-1" not in desktop["searchPane"]["text"]


def test_the_tui_widget_declares_an_ambient_fixed_width_card(tui):
    assert tui["registered"] == 1
    assert tui["widget"]["id"] == "skills-routed"
    assert tui["widget"]["mode"] == "ambient"
    assert tui["widget"]["width"] == 44
    assert tui["widget"]["width"] == tui["constants"]["width"]
    assert "record.json" in tui["widget"]["usage"]
    assert tui["constants"]["poll"] == 500


def test_the_tui_widget_defaults_to_the_plugins_record_path(tui):
    assert tui["constants"]["defaultPath"] == DEFAULT_RECORD_PATH
    assert tui["relaunch"]["path"] == DEFAULT_RECORD_PATH
    assert tui["paths"]["blank"] == DEFAULT_RECORD_PATH
    assert tui["paths"]["tildeDefault"] == DEFAULT_RECORD_PATH
    assert tui["paths"]["home"] == os.path.join(tui["homedir"], "custom.json")
    assert tui["paths"]["absolute"] == "/tmp/custom.json"
    assert tui["paths"]["plain"] == "custom.json"
    assert tui["helpPath"] == {
        "default": "~/.cache/hermes-skill-router/suggestions.json",
        "other": "/tmp/custom.json",
    }
    assert "~/.cache/hermes-skill-router/suggestions.json" in tui["widget"]["help"]


def test_the_tui_widget_renders_a_dialog_with_a_collapsed_accordion(tui):
    assert tui["routed"]["dialogType"] == "Dialog"
    assert tui["routed"]["dialog"] == {"title": "Skills routed", "width": 44}
    assert tui["routed"]["bodyType"] == "Accordion"
    assert tui["routed"]["accordion"] == {"count": 3, "defaultOpen": False, "title": "Suggestions"}
    assert tui["summary"] == {
        "caption": CAPTION,
        "count": 3,
        "names": ["alpha", "beta", "gamma"],
        "note": "",
        "status": "routed",
        "title": "Suggestions",
    }
    for name in ("alpha", "beta", "gamma"):
        assert name in tui["routed"]["body"]


def test_the_tui_widget_paints_one_muted_turn_caption(tui):
    captions = tui["captions"]

    assert tui["routed"]["caption"] == CAPTION
    assert tui["routed"]["captionNode"] == {"color": "muted", "type": "Text"}
    assert captions["routed"] == CAPTION
    assert captions["noTurn"] == CAPTION_TIME
    assert captions["noAt"] == "turn turn-1"
    assert captions["badAt"] == "turn turn-1"
    assert captions["inert"] == ""
    assert captions["idle"] == ""
    assert tui["times"] == {"bad": "", "blank": "", "good": CAPTION_TIME, "missing": ""}
    assert tui["empty"]["caption"] == CAPTION


def test_the_tui_widget_marks_a_model_search(tui):
    marker = f"model search · {CAPTION_TIME}"

    assert tui["searchCaption"] == marker
    assert tui["search"]["caption"] == marker
    assert tui["search"]["record"]["origin"] == "search"
    assert tui["search"]["record"]["names"] == ["alpha", "beta", "gamma"]
    assert tui["schemaTwoTurn"]["origin"] == "turn"


def test_the_tui_widget_rereads_the_record_on_its_poll(tui):
    assert tui["poll"] == [500]
    assert tui["pollsAfterRelaunch"] == [500, 500]
    assert tui["relaunch"]["cleared"] is True
    assert tui["update"]["app"] == "skills-routed"
    assert tui["update"]["next"]["record"]["names"] == ["delta", "epsilon"]


def test_the_tui_widget_shows_one_line_for_the_states_without_names(tui):
    assert tui["inert"]["bodyType"] == "Text"
    assert "router inactive" in tui["inert"]["body"]
    assert tui["inert"]["record"]["names"] == []
    assert "ignored" not in tui["inert"]["body"]
    assert tui["inert"]["caption"] == ""
    assert tui["empty"]["bodyType"] == "Text"
    assert "suggested nothing" in tui["empty"]["body"]
    assert tui["missing"]["bodyType"] == "Text"
    assert "no suggestions recorded yet" in tui["missing"]["body"]
    assert tui["missing"]["caption"] == ""
    assert tui["corrupt"]["bodyType"] == "Text"
    assert "no suggestions recorded yet" in tui["corrupt"]["body"]
    assert tui["corrupt"]["caption"] == ""
    assert tui["missing"]["record"] == {"at": "", "names": [], "origin": "", "status": "idle", "turn_id": ""}
    assert tui["corrupt"]["record"] == {"at": "", "names": [], "origin": "", "status": "idle", "turn_id": ""}


def test_the_tui_widget_shows_no_request_text(tui):
    assert tui["tampered"]["projection"] == {
        "at": "2026-11-10T07:20:31Z",
        "names": ["alpha", "beta", "gamma"],
        "origin": "turn",
        "status": "routed",
        "turn_id": "turn-1",
    }
    assert tui["tampered"]["caption"] == CAPTION
    assert tui["tampered"]["rendered"] is False


@pytest.mark.parametrize(
    "path",
    [DESKTOP_PLUGIN_PATH, TUI_WIDGET_PATH, INDEX_ENTRY_PATH],
    ids=["desktop", "tui", "dashboard-entry"],
)
def test_the_javascript_halves_are_valid_modules(path):
    if NODE is None:
        pytest.skip("node is required to syntax-check the display halves")
    result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr
