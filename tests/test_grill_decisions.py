"""Grill Q1-Q4: red-first behavior pins (real rewrite / load / surfaces).

These tests were written before the settled implementation and must fail on the
old code for the expected reason (no persisted traps, time replay, silent
degraded producers, search overwrites note, heading without origin).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import display as display_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_display import PAYLOAD, make_config  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(display_module, "_LOGGED", set())
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    search_module.clear_dumps()
    try:
        if hasattr(router_module, "_SETUP_CACHE"):
            router_module._SETUP_CACHE.clear()
    except Exception:
        pass


def _turn(cfg, **kwargs):
    return rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [
            [1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts
        ],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="session-1",
        turn_id="turn-1",
        notify=lambda _turn_id: None,
        **kwargs,
    )


# Q1: value-scoped not-ready (probe every dead-endpoint turn, no health replay).


def test_setup_probe_repeats_on_every_turn_regardless_of_health_window(tmp_path, monkeypatch):
    from hermes_skill_router.config import load as cfg_load

    cfg = cfg_load(
        {
            "suggestionsPath": str(tmp_path / "suggestions.json"),
            "cachePath": str(tmp_path / "index.json"),
            "healthCheckMs": 60000,
        }
    )
    calls = []

    def fake_check(config, deadline=None):
        calls.append(1)
        return False, "embedBaseUrl"

    monkeypatch.setattr(router_module, "check_setup_state", fake_check)
    first = router_module._setup_note(cfg)
    second = router_module._setup_note(cfg)
    assert first, "a not-ready probe must name the failing setting"
    assert second, "the line must stay while the servers do not both answer"
    assert len(calls) == 2, (
        "every dead-endpoint turn must re-probe (value-scoped); the health window "
        f"must not replay a cached answer, saw {len(calls)} probe(s)"
    )


def test_setup_note_follows_the_current_answer_not_a_cached_one(tmp_path, monkeypatch):
    from hermes_skill_router.config import load as cfg_load

    cfg = cfg_load(
        {
            "suggestionsPath": str(tmp_path / "suggestions.json"),
            "cachePath": str(tmp_path / "index.json"),
            "healthCheckMs": 60000,
        }
    )
    state = {"answer": (False, "embedBaseUrl")}
    monkeypatch.setattr(
        router_module, "check_setup_state", lambda config, deadline=None: state["answer"]
    )
    assert router_module._setup_note(cfg), "a not-ready value must show the line"
    state["answer"] = (True, "")
    assert router_module._setup_note(cfg) == "", (
        "once both servers answer, the line must disappear even inside the health window"
    )


# Q4: persisted (trap_id, fingerprint) in config.load.


def test_first_load_exposes_the_trap_and_unchanged_value_stays_silent(tmp_path):
    path = str(tmp_path / "suggestions.json")
    first = load({"shortlistSize": 0, "suggestionsPath": path})
    assert first.notes, "a rejected bound must produce a note"
    assert getattr(first, "trap_notes", None), (
        "load must expose the newly-seen trap for the display record"
    )
    assert any("shortlistSize" in note for note in first.trap_notes)
    second = load({"shortlistSize": 0, "suggestionsPath": path})
    assert second.notes == first.notes, "the underlying notes are still produced"
    assert second.trap_notes == (), (
        "an unchanged offending value must stay silent on the display record"
    )
    third = load({"shortlistSize": 0, "suggestionsPath": path})
    assert third.trap_notes == (), "unchanged stays silent forever, not just once"


def test_changed_value_warns_once_on_the_display_record(tmp_path):
    path = str(tmp_path / "suggestions.json")
    first = load({"shortlistSize": 0, "suggestionsPath": path})
    assert first.trap_notes, "the first offending value warns"
    changed = load({"shortlistSize": 999, "suggestionsPath": path})
    assert changed.trap_notes, (
        "a changed offending value is a new fingerprint and must warn once"
    )
    assert all("shortlistSize" in note for note in changed.trap_notes)
    again = load({"shortlistSize": 999, "suggestionsPath": path})
    assert again.trap_notes == (), "the changed value then stays silent"


def test_trap_persists_across_processes_via_the_file(tmp_path):
    path = str(tmp_path / "suggestions.json")
    first = load({"eagerSkills": "alpha", "suggestionsPath": path})
    assert first.trap_notes, "a malformed list warns once"
    # Simulate a fresh process: _LOGGED_NOTES is process-local, so a per-process
    # log-once set would warn again. The persisted file must suppress the
    # display warning without any in-process memory.
    from hermes_skill_router import config as config_module

    config_module._LOGGED_NOTES.clear()
    second = load({"eagerSkills": "alpha", "suggestionsPath": path})
    assert second.trap_notes == (), (
        "after a restart the unchanged trap must stay silent on the display record"
    )


# Q2: one always-visible degraded line enumerating each loss.


def test_cache_persistence_off_is_named_on_the_line(tmp_path):
    cfg = make_config(tmp_path)  # cachePath "" (persistence off), store off

    assert _turn(cfg) is not None
    note = read(cfg)["note"]
    assert "cache" in note.lower() and "persistence" in note.lower(), (
        f"cachePath off must be enumerated, got {note!r}"
    )


def test_fusion_off_is_named_on_the_line(tmp_path):
    cfg = make_config(tmp_path, fusionMode="cosine")

    assert _turn(cfg) is not None
    note = read(cfg)["note"]
    assert "fusion" in note.lower(), f"cosine must be enumerated, got {note!r}"


def test_missing_bm25_is_named_on_the_line(tmp_path):
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    index = Index(
        names=["alpha", "beta"],
        descriptions=["Alpha skill", "Beta skill"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        dim=2,
        fingerprint="test-fingerprint",
        term_index=None,
    )

    assert _turn(cfg, index=index) is not None
    note = read(cfg)["note"]
    assert "bm25" in note.lower(), f"missing BM25 must be enumerated, got {note!r}"


def test_history_line_names_the_skip_state(tmp_path):
    missing_cfg = make_config(
        tmp_path, associationDbPath=str(tmp_path / "missing.db")
    )
    assert _turn(missing_cfg) is not None
    missing_note = read(missing_cfg)["note"]
    assert "missing" in missing_note.lower(), (
        f"a missing store must name the skip state, got {missing_note!r}"
    )

    empty_path = tmp_path / "empty.db"
    associations_module.open_store(str(empty_path)).close()
    empty_cfg = make_config(tmp_path, associationDbPath=str(empty_path))
    assert _turn(empty_cfg) is not None
    empty_note = read(empty_cfg)["note"]
    assert "empty" in empty_note.lower(), (
        f"an empty store must name the skip state, got {empty_note!r}"
    )
    assert empty_note != missing_note, "different skip states must read differently"


def test_budget_expiry_carries_a_degraded_line(tmp_path, monkeypatch):
    from hermes_skill_router.search import Ranked

    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    monkeypatch.setattr(
        router_module,
        "rank",
        lambda *a, **k: Ranked(reason="budget-expired", detail="routing budget 1000 ms exhausted"),
    )
    failed = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="budget-session",
        notify=lambda _t: None,
    )
    assert failed is None, "a budget expiry leaves the request unchanged"
    note = read(cfg)["note"]
    assert "budget" in note.lower(), (
        f"a budget expiry must carry a degraded line, got {note!r}"
    )
    assert "routingtimeoutms" in note.lower() or "server" in note.lower(), (
        f"the budget line must name the fix, got {note!r}"
    )
    assert note != router_module.ENDPOINT_NOTE, "budget is not a dead endpoint"


def test_dropped_window_carries_a_degraded_line(tmp_path):
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    session = "dropped-session"
    # First failure in the window: keeps the catalog, remembers the dump.
    first = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: None,
        session_id=session,
        notify=lambda _t: None,
    )
    assert first is None, "the first failure keeps the catalog"
    # Second failure in the same window: strips the catalog and drops it.
    second = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: None,
        session_id=session,
        notify=lambda _t: None,
    )
    assert second is not None and "messages" in second["request"]
    note = read(cfg)["note"]
    assert "already" in note.lower(), (
        f"a dropped window must name the loss and its fix, got {note!r}"
    )
    assert "skills_list" in note, f"the dropped line must name the fix, got {note!r}"


def test_partial_view_refusal_is_named_on_the_line(tmp_path):
    from tests.test_rewrite import FakeEmbedder, FakeReranker, make_index, _payload
    import hermes_skill_router.library as library_mod

    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    library_mod._write_cache(cfg, make_index())
    result = rewrite(
        _payload(rows=(("alpha", "Alpha skill"),)),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="partial-session",
        notify=lambda _t: None,
    )
    assert result is not None, "a refused catalog routes from the saved index"
    note = read(cfg)["note"]
    assert "partial" in note.lower(), (
        f"a refused partial view must be enumerated, got {note!r}"
    )


def test_heal_finished_incomplete_is_not_reported_as_running(tmp_path, monkeypatch):
    from hermes_skill_router.associations import BackfillReport

    def finishing_heal(store_path, state_db_path=None, *, catalog=(), now_ms=None):
        return BackfillReport(sessions_found=0, loads=0, ms=1)

    store_path = tmp_path / "associations.db"
    associations_module.open_store(str(store_path)).close()
    cfg = make_config(
        tmp_path,
        cachePath=str(tmp_path / "index.json"),
        associationDbPath=str(store_path),
    )
    monkeypatch.setattr(associations_module, "heal_store", finishing_heal)
    assert associations_module.request_heal(str(store_path)) is True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            settled = associations_module._HEAL_STATE.get(str(store_path)) != "running"
        if settled:
            break
        time.sleep(0.01)
    assert settled, "the finishing heal never settled"
    # Use a fresh turn: the store is still incomplete (no watermark), the heal
    # finished without a failure note, so the line must say finished.
    assert _turn(cfg) is not None
    note = read(cfg)["note"]
    assert "heal" in note.lower() and ("finish" in note.lower() or "complete" in note.lower()), (
        f"a finished-but-incomplete heal must name that state, got {note!r}"
    )
    assert "running" not in note.lower(), (
        f"a finished heal must not claim healing is running, got {note!r}"
    )


def test_desktop_heading_names_the_origin(tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    assert node is not None, "node is required to load the shipped desktop half"
    desktop_dir = tmp_path / "desktop-surface"
    for name, text in {
        "plugin.mjs": (ROOT / "desktop" / "plugin.js").read_text(encoding="utf-8"),
        "node_modules/@hermes/plugin-sdk/index.mjs": (
            "export const atom = initial => { let value = initial; "
            "return { get: () => value, set: next => { value = next } } }\n"
            "export const useValue = store => store.get()\n"
            "export const PANES_AREA = 'panes'\n"
            "export const DisclosureCaret = 'DisclosureCaret'\n"
            "export const useQuery = () => ({})\n"
            "export const queryClient = { invalidateQueries: async () => {} }\n"
        ),
        "node_modules/@hermes/plugin-sdk/package.json": json.dumps(
            {"name": "@hermes/plugin-sdk", "type": "module", "main": "./index.mjs"}
        ),
        "node_modules/react/index.mjs": (
            "export const useState = initial => [initial, () => {}]\n"
            "export const useEffect = () => {}\n"
        ),
        "node_modules/react/jsx-runtime.mjs": (
            "export const Fragment = 'Fragment'\n"
            "export const jsx = () => {}\n"
            "export const jsxs = jsx\n"
        ),
        "node_modules/react/package.json": json.dumps(
            {
                "name": "react",
                "type": "module",
                "exports": {".": "./index.mjs", "./jsx-runtime": "./jsx-runtime.mjs"},
            }
        ),
    }.items():
        target = desktop_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    driver = desktop_dir / "driver.mjs"
    driver.write_text(
        "import { summary } from './plugin.mjs'\n"
        "const turn = summary({ at: '2026-11-10T07:20:31Z', names: ['a', 'b'], origin: 'turn', status: 'routed', turn_id: 't-1' })\n"
        "const search = summary({ at: '2026-11-10T07:20:31Z', names: ['a', 'b'], origin: 'search', status: 'routed', turn_id: '' })\n"
        "console.log(JSON.stringify({ turn: turn.heading, search: search.heading, turnNote: turn.note, inertNote: summary({ status: 'inert', origin: 'turn', note: 'DEGRADED' }).note }))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(driver)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(desktop_dir),
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert "latest pass" in observed["turn"].lower(), (
        f"a turn heading must name the latest pass, got {observed['turn']!r}"
    )
    assert "search" in observed["search"].lower(), (
        f"a search heading must name the search origin, got {observed['search']!r}"
    )
    assert observed["turn"] != observed["search"], "turn and search headings must differ"
    assert observed["inertNote"] == "DEGRADED", "the pane must render the note in every state"


# Q3: search preserves the turn note; desktop heading names the origin.


def test_search_record_preserves_the_turn_note(tmp_path, monkeypatch):
    from hermes_skill_router.catalog import CatalogRows, Partition
    from hermes_skill_router import tools as tools_module
    from hermes_skill_router.tools import handle

    store_path = tmp_path / "associations.db"
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    assert _turn(cfg) is not None
    turn_note = read(cfg)["note"]
    assert turn_note, "the turn must have written a degraded note"

    session = "search-session"
    rows = (("alpha", "Alpha skill"), ("beta", "Beta skill"))
    search_module.remember_partition(
        session,
        Partition(
            kept=CatalogRows(),
            routed=CatalogRows(
                tuple(name for name, _d in rows),
                tuple(desc for _n, desc in rows),
            ),
        ),
    )
    monkeypatch.setattr(
        tools_module,
        "EMBED",
        lambda config, texts: [[0.0, 1.0] for _t in texts],
    )
    monkeypatch.setattr(
        tools_module,
        "RERANK",
        lambda config, query, documents: [0.9, 0.2][: len(documents)],
    )
    monkeypatch.setattr(
        tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md"
    )
    settings = {
        "cachePath": cfg.cache_path,
        "suggestionsPath": cfg.suggestions_path,
        "associationDbPath": str(store_path),
    }
    answer = json.loads(
        handle({"query": "alpha task words here"}, settings=settings, session_id=session)
    )
    assert "degraded" not in answer, "a healthy search must answer with rows"
    stored = read(cfg)
    assert stored["origin"] == "search", "search overwrites the turn record (Q3)"
    assert stored["note"] == turn_note, (
        f"the search record must preserve the degraded line, got {stored['note']!r} "
        f"want {turn_note!r}"
    )
