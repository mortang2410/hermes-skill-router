"""Ticket 04 (trm3): the degraded line on every surface.

Five things are pinned here:

* desktop pane and TUI widget render a record's multi-part note in the narrow strip
  without silent truncation (full text in the summary and the rendered tree, no clamp);
* the injected context block carries the same stored note;
* the dashboard API route passes the degraded record through unchanged;
* a healthy routed turn after a degraded one writes a note-free record and every
  surface clears the line;
* a successful search keeps carrying the previous record's note forward.

Desktop and TUI are runtimes pytest cannot host, so the JS halves are loaded in Node
with stub SDK modules, the same way test_display_surfaces does.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import display as display_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router import tools as tools_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows, Partition  # noqa: E402
from hermes_skill_router.display import ROUTED, read, record  # noqa: E402
from hermes_skill_router.inject import suggested_names  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from hermes_skill_router.tools import handle  # noqa: E402
from tests.test_display import PAYLOAD, make_config  # noqa: E402

DASHBOARD_DIR = ROOT / "dashboard"
PLUGIN_API_PATH = DASHBOARD_DIR / "plugin_api.py"
DESKTOP_PLUGIN_PATH = ROOT / "desktop" / "plugin.js"
TUI_WIDGET_PATH = ROOT / "tui" / "skill-router.mjs"

NODE = shutil.which("node")

# A long multi-part line: three degraded parts joined, long enough to wrap in 260px / 44 cells.
MULTI_PART_NOTE = (
    "History incomplete (missing store) and healing is running in the background; "
    "routing continues without learned history. "
    "Cache persistence is off (cachePath empty); the index is rebuilt each turn. "
    "Fusion is off (cosine or all channels off); routing uses the cosine order only."
)


# ---------------------------------------------------------------------------
# Shared fixtures and turn helpers (same shapes as test_history_note.py).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process logger sets, a fresh index memo, dumps, and the host lookup answered."""
    monkeypatch.setattr(display_module, "_LOGGED", set())
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    search_module.clear_dumps()


def _turn(cfg, turn_id="turn-1"):
    """Drive one real rewrite turn."""
    return rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [
            [1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts
        ],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="session-1",
        turn_id=turn_id,
        notify=lambda _turn_id: None,
    )


def _last_block(result) -> str:
    assert result is not None
    return result["request"]["messages"][-1]["content"]


def _session_store(path: Path, newest_ms: int) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at REAL NOT NULL,
            last_activity_at REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, last_activity_at) VALUES (?, ?, ?, ?)",
        ("s-newest", "cli", newest_ms / 1000.0, newest_ms / 1000.0),
    )
    conn.commit()
    conn.close()


def _complete_store(store_path: Path, newest_ms: int) -> None:
    store = associations_module.open_store(str(store_path))
    try:
        associations_module.write_meta(
            store, associations_module.META_LAST_SCAN_MS, str(newest_ms)
        )
    finally:
        store.close()


@pytest.fixture(scope="module")
def plugin_api():
    """Import dashboard/plugin_api.py the way Hermes' dashboard loader does."""
    pytest.importorskip("fastapi", reason="the dashboard backend is a FastAPI router")
    name = "hermes_dashboard_plugin_hermes_skill_router_t04"
    spec = importlib.util.spec_from_file_location(name, PLUGIN_API_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


# ---------------------------------------------------------------------------
# Injected context block carries the same stored note (C4).
# ---------------------------------------------------------------------------


def test_the_injected_block_carries_the_stored_note(tmp_path):
    """The provider request's injected block carries the turn's stored degraded line verbatim."""
    store_path = tmp_path / "associations.db"
    assert not store_path.exists()
    cfg = make_config(tmp_path, associationDbPath=str(store_path))

    result = _turn(cfg)
    assert result is not None
    stored = read(cfg)
    assert stored["status"] == ROUTED
    note = stored["note"]
    assert note, "a missing store is incomplete: the turn must write a degraded line"
    assert "History incomplete" in note

    block = _last_block(result)
    assert note in block, "the injected block must carry the same stored note"
    assert note in json.dumps(result["request"])
    # The note is one final paragraph and does not pose as a suggested name.
    assert note not in suggested_names(block)
    assert suggested_names(block), "the note must not disturb the suggestion list"


def test_a_healthy_turn_block_carries_no_degraded_line(tmp_path, monkeypatch):
    """The cleared line stays cleared: a healthy block carries no degraded text at all."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda home=None: str(state_path)
    )
    store_path = tmp_path / "associations.db"
    _complete_store(store_path, newest)
    cfg = make_config(
        tmp_path,
        cachePath=str(tmp_path / "index.json"),
        associationDbPath=str(store_path),
    )

    result = _turn(cfg)
    assert result is not None
    assert read(cfg)["note"] == ""
    block = _last_block(result)
    assert "History incomplete" not in block
    assert "Cache persistence is off" not in block


# ---------------------------------------------------------------------------
# Dashboard API route passes the degraded record through unchanged (C4).
# ---------------------------------------------------------------------------


def test_the_dashboard_route_passes_the_degraded_record_unchanged(tmp_path, plugin_api):
    """The read-only route is a projection: the degraded body equals display.read verbatim."""
    record_path = tmp_path / "suggestions.json"
    record_path.write_text(
        json.dumps(
            {
                "schema": 2,
                "at": "2026-11-10T07:20:31Z",
                "session_id": "session-1",
                "turn_id": "turn-1",
                "status": "routed",
                "origin": "turn",
                "names": ["alpha", "beta"],
                "note": MULTI_PART_NOTE,
            }
        ),
        encoding="utf-8",
    )

    answer = plugin_api.answer(record_path)

    assert answer == read(record_path)
    assert answer["note"] == MULTI_PART_NOTE
    assert answer["names"] == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Healthy turn clears the line on every surface (R7).
# ---------------------------------------------------------------------------


def test_a_healthy_turn_after_a_degraded_one_writes_a_note_free_record(tmp_path, monkeypatch):
    """The stale-line rule: a healthy turn overwrites the degraded record with a note-free one."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda home=None: str(state_path)
    )
    store_path = tmp_path / "associations.db"
    cfg = make_config(
        tmp_path,
        cachePath=str(tmp_path / "index.json"),
        associationDbPath=str(store_path),
    )

    first = _turn(cfg, turn_id="turn-1")
    assert first is not None
    degraded_note = read(cfg)["note"]
    assert "History incomplete" in degraded_note
    assert degraded_note in _last_block(first)

    _complete_store(store_path, newest)
    second = _turn(cfg, turn_id="turn-2")
    assert second is not None

    stored = read(cfg)
    assert stored["note"] == "", "a healthy turn must write a note-free record"
    # The stale-line rule is record-scoped: the healthy turn writes its own fresh
    # block with no degraded note and replays no bytes from turn 1 (D8/D9).
    # Round-2 F24: inequality alone would pass on any byte drift. Pin the
    # absence of turn 1's exact block bytes in the fresh block.
    assert _last_block(first) not in _last_block(second), "turn 1's block is never re-inserted"
    assert "History incomplete" not in _last_block(second), "no stale note in the fresh block"


# ---------------------------------------------------------------------------
# Search carries the previous record's note forward (C6).
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    def __call__(self, cfg, texts):
        vectors = {
            "alpha\nAlpha skill": [1.0, 0.0, 0.0],
            "beta\nBeta skill": [0.0, 1.0, 0.0],
            "gamma\nGamma skill": [0.0, 0.0, 1.0],
        }
        return [list(vectors.get(text, [0.0, 1.0, 0.0])) for text in texts]


class _FakeReranker:
    def __call__(self, cfg, query, documents):
        return [0.9, 0.5, 0.1][: len(documents)]


def test_a_successful_search_carries_the_previous_note_forward(tmp_path, monkeypatch):
    """A lookup during a degraded state must not erase the loss from view, new producers included."""
    cfg = make_config(tmp_path, associationDbPath=str(tmp_path / "associations.db"))
    # A synthetic multi-part line naming the follow-up producers: carry-forward is producer-agnostic.
    degraded = (
        "History incomplete (missing store) and healing is running in the background; "
        "routing continues without learned history. "
        "Catalog already dumped in this window; call skills_list to see every skill available. "
        "Routing fell back to the saved index; rebuild the index for fresh results."
    )
    written = record(
        cfg,
        status=ROUTED,
        names=["alpha"],
        session_id="session-1",
        turn_id="turn-1",
        note=degraded,
    )
    assert written is not None
    assert read(cfg)["note"] == degraded

    rows = (("alpha", "Alpha skill"), ("beta", "Beta skill"), ("gamma", "Gamma skill"))
    partition = Partition(
        kept=CatalogRows(),
        routed=CatalogRows(
            tuple(name for name, _d in rows), tuple(d for _n, d in rows)
        ),
    )
    search_module.remember_partition("session-1", partition)
    settings = {
        "cachePath": "",
        "suggestionsPath": str(tmp_path / "suggestions.json"),
        "associationDbPath": str(tmp_path / "associations.db"),
    }
    monkeypatch.setattr(tools_module, "EMBED", _FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", _FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")

    answer = json.loads(handle({"query": "query text"}, settings=settings, session_id="session-1"))

    assert "degraded" not in answer, answer
    stored = read(cfg)
    assert stored["origin"] == "search"
    assert stored["note"] == degraded, "the search record must carry the degraded line verbatim"
    search_module.forget_session("session-1")


# ---------------------------------------------------------------------------
# Desktop and TUI render the multi-part note without silent truncation (R6).
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
"""

DESKTOP_DRIVER = (
    """
import plugin, { summary } from './plugin.mjs'

"""
    + TREE_HELPERS
    + """
const NOTE = __NOTE__
const STAMP = '2026-11-10T07:20:31Z'
const record = { at: STAMP, names: ['alpha', 'beta'], note: NOTE, origin: 'turn', status: 'routed', turn_id: 'turn-1' }
const out = {}
out.summary = summary(record)
out.cleared = summary({ ...record, note: '' })
// Round-2 F19: actually mount and render the pane; summary alone never paints it.
const panes = []
plugin.register({
  register: descriptor => panes.push(descriptor),
  rest: async () => record,
  onEvent: () => {}
})
globalThis.__t8Result = { data: record, isPending: false, error: null }
const pane = panes[0].render()
// The pane descriptor returns a lazy element; invoke its component so the
// hooks run and the note paragraph is in the tree the way a mount would paint it.
out.rendered = text(typeof pane.type === 'function' ? pane.type(pane.props) : pane)
console.log(JSON.stringify(out))
"""
)

TUI_DRIVER = (
    """
import register, { readRecord, summary } from './skill-router.mjs'
import { writeFileSync } from 'node:fs'

"""
    + TREE_HELPERS
    + """
const NOTE = __NOTE__
const out = {}
const theme = { color: { label: 'label', muted: 'muted' } }
const apps = []
const sdk = {
  Accordion: 'Accordion',
  Dialog: 'Dialog',
  Text: 'Text',
  defineWidgetApp: app => { apps.push(app); return app },
  h: (type, props, ...children) => ({ props: { ...(props ?? {}), children: children.length === 1 ? children[0] : children }, type }),
  updateWidget: () => {}
}
register(sdk)
const app = apps[0]
const record = extra => ({ at: '2026-11-10T07:20:31Z', names: ['alpha', 'beta'], schema: 2, origin: 'turn', session_id: 'session-1', status: 'routed', turn_id: 'turn-1', ...extra })
writeFileSync('note.json', JSON.stringify(record({ note: NOTE })))
const render = path => app.render({ cols: 120, rows: 30, state: { path, record: readRecord(path) }, t: theme })
const dialogChildren = tree => (Array.isArray(tree.props.children) ? tree.props.children : [tree.props.children])
out.record = readRecord('note.json')
out.summary = summary(readRecord('note.json'))
const tree = render('note.json')
out.rendered = text(tree)
writeFileSync('clear.json', JSON.stringify(record({ note: '' })))
out.cleared = summary(readRecord('clear.json'))
console.log(JSON.stringify(out))
"""
)


def _run_node_driver(tmp_path: Path, driver: str, files: dict) -> dict:
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


def test_the_desktop_pane_renders_the_multi_part_note_without_truncation(tmp_path):
    if NODE is None:
        pytest.skip("node is required to load desktop/plugin.js")
    files = {
        "plugin.mjs": DESKTOP_PLUGIN_PATH.read_text(encoding="utf-8"),
        "node_modules/@hermes/plugin-sdk/index.mjs": DESKTOP_SDK_STUB,
        "node_modules/@hermes/plugin-sdk/package.json": SDK_PACKAGE_STUB,
        "node_modules/react/index.mjs": REACT_INDEX_STUB,
        "node_modules/react/jsx-runtime.mjs": JSX_STUB,
        "node_modules/react/package.json": REACT_PACKAGE_STUB,
    }
    driver = DESKTOP_DRIVER.replace("__NOTE__", json.dumps(MULTI_PART_NOTE))
    out = _run_node_driver(tmp_path / "desktop", driver, files)

    assert out["summary"]["note"] == MULTI_PART_NOTE, "the summary must carry the full note"
    assert MULTI_PART_NOTE in out["rendered"], "the mounted pane must paint the full multi-part line"
    assert out["cleared"]["note"] == "", "a note-free record must not resurrect the degraded line"

    source = DESKTOP_PLUGIN_PATH.read_text(encoding="utf-8")
    note_line = next(line for line in source.splitlines() if "note ? jsx('p'" in line)
    for clamp in ("truncate", "line-clamp", "ellipsis"):
        assert clamp not in note_line, f"the note paragraph must wrap, not clamp with {clamp!r}"


def test_the_tui_widget_renders_the_multi_part_note_without_truncation(tmp_path):
    if NODE is None:
        pytest.skip("node is required to load tui/skill-router.mjs")
    files = {"skill-router.mjs": TUI_WIDGET_PATH.read_text(encoding="utf-8")}
    driver = TUI_DRIVER.replace("__NOTE__", json.dumps(MULTI_PART_NOTE))
    out = _run_node_driver(tmp_path / "tui", driver, files)

    assert out["record"]["note"] == MULTI_PART_NOTE
    assert out["summary"]["note"] == MULTI_PART_NOTE
    assert MULTI_PART_NOTE in out["rendered"], "the card must paint the full multi-part line"
    assert out["cleared"]["note"] == "", "a note-free record must not resurrect the degraded line"

    source = TUI_WIDGET_PATH.read_text(encoding="utf-8")
    for clamp in ('wrap="truncate"', "maxLines", "ellipsize", "numberOfLines"):
        assert clamp not in source, f"the note Text must wrap, not truncate with {clamp!r}"


def test_the_record_note_round_trips_through_display(tmp_path):
    """The writer stores the multi-part line verbatim, so every surface reads the same text."""
    cfg = make_config(tmp_path)
    assert record(cfg, status=ROUTED, names=["alpha"], note=MULTI_PART_NOTE) is not None
    assert read(cfg)["note"] == MULTI_PART_NOTE
    assert os.path.exists(cfg.suggestions_path)
