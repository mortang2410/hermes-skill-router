"""Ticket 07 (C8): continuous incremental learning independent of the completeness window.

A routed turn ingests new sessions and advances the learning watermark
regardless of the completeness-window check; the window only picks the
degraded-line wording, never gates the ingest. Provenance and attribution
are unchanged. The scan runs on the existing background thread under the
one-attempt lock, so scans serialize; a session that arrives mid-scan keeps
``behind`` set after the completion, so the next routed turn ingests it; a
failed scan and a restart keep ``behind`` and the watermark, so the next
routed turn retries (a successful retry clears ``behind`` unless another
session arrives during that retry).

Model: ``model/Learning.tla`` v2.1.0 (transitions ``new_session`` /
``turn_starts`` / ``scan_completes`` / ``scan_fails`` / ``restart`` /
``observe_complete`` / ``observe_incomplete``). The code maps ``behind``
to the derived fact ``newest_session_stamp > watermark`` or an unseen
session id, ``inflight`` to the one-attempt running latch, and ``late``
to the derived behind being set again after a completion, so a mid-scan
arrival is never missed. The finished-heal flag (C2) gates
``request_heal`` only; it never gates the learning scan, though a scan
settle writes the flag as any uncapped settle does.

Every association database and session store is a temporary file pytest
owns, and every clock is a fixed epoch millisecond.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_associations import (  # noqa: E402
    CATALOG,
    DAY,
    SUBSTANTIVE,
    T0,
    HermesStoreFixture,
    rows,
)
from tests.test_display import PAYLOAD, make_config  # noqa: E402

INDEX = Index(
    names=["alpha", "beta"],
    descriptions=["Alpha skill", "Beta skill"],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    dim=2,
    fingerprint="followup-learning",
)


@pytest.fixture
def hermes(tmp_path: Path) -> Iterable[HermesStoreFixture]:
    """A temporary Hermes session store this suite owns."""
    fixture = HermesStoreFixture(tmp_path / "state.db")
    yield fixture
    fixture.close()


@pytest.fixture(autouse=True)
def no_live_heal():
    """Shadow the conftest guard: this module drives the real scan entry points.

    The one-attempt latch is global to the process, so every case starts and
    ends with no attempt recorded. Clearing it simulates a restart: the
    in-memory state is gone, only the persisted watermark remains.
    """
    associations_module.reset_heal_state()
    yield
    associations_module.reset_heal_state()


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process state, so one test never masks another."""
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    search_module.clear_dumps()
    yield


@pytest.fixture(autouse=True)
def _default_state_path(hermes, monkeypatch):
    """Route the default session store at this test's fixture session store."""
    monkeypatch.setattr(associations_module, "default_state_db_path", lambda: str(hermes.path))
    yield


def _add_session(hermes: HermesStoreFixture, session_id: str, base_ms: int, skill: str = "code-review") -> None:
    """Append one human session with one typed prompt and one skill_view load."""
    hermes.session(session_id, started_ms=base_ms)
    hermes.user(session_id, f"{SUBSTANTIVE} {session_id}", base_ms + 100)
    hermes.skill_view(session_id, skill, base_ms + 200)
    hermes.commit()


def _newest_stamp(hermes: HermesStoreFixture) -> int:
    """Return the newest session stamp in the fixture store."""
    return associations_module.newest_session_stamp_ms(str(hermes.path))


def _watermark(store_path: Path) -> str:
    """Return the persisted learning watermark, or None when unset."""
    if not os.path.exists(str(store_path)):
        return "0"
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        return associations_module.read_meta(store, associations_module.META_LAST_SCAN_MS)
    finally:
        store.close()


def _settled(store_path: str, timeout: float = 5.0) -> bool:
    """Wait until no scan is in flight: neither a heal nor a learning scan."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
            learning = associations_module._LEARNING_RUNNING
        if state != associations_module._HEAL_RUNNING and not learning:
            return True
        time.sleep(0.01)
    return False


def _scan(store_path: Path, hermes: HermesStoreFixture, **kwargs):
    """Run one direct backfill (test setup only; no flag, no background thread)."""
    kwargs.setdefault("catalog", CATALOG)
    kwargs.setdefault("now_ms", T0 + 10 * DAY)
    store = associations_module.open_store(str(store_path))
    try:
        return associations_module.backfill(store, str(hermes.path), **kwargs)
    finally:
        store.close()


def _complete_store_with_new_session(hermes, tmp_path, label="learning.db") -> Path:
    """A store complete per the window, with (or without) one session newer than the watermark.

    Runs a first learning scan over ``s-first`` at T0, then appends ``s-new``
    at T0 + 5000, so the store is inside the 24 hour window (complete) yet
    behind the watermark by one session.
    """
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / label
    report = _scan(store_path, hermes)
    assert report.sessions_scanned == 1
    assert _watermark(store_path) == str(_newest_stamp(hermes))
    _add_session(hermes, "s-new", T0 + 5000)
    assert _newest_stamp(hermes) > int(_watermark(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert associations_module.is_store_complete(store, str(hermes.path)) is True, (
            "the fixture store must read complete so the window cannot explain the scan"
        )
    finally:
        store.close()
    return store_path


def _routed_turn(store_path: Path, tmp_path, **overrides):
    """Drive one successful routed turn (``rewrite``) over the store at *store_path*."""
    cfg = make_config(tmp_path, associationDbPath=str(store_path), **overrides)
    kwargs = {
        "index": INDEX,
        "embed": lambda config, texts: [[1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts],
        "rerank": lambda config, query, documents: [0.9, 0.2][: len(documents)],
        "session_id": "session-1",
        "turn_id": "turn-1",
        "notify": lambda _turn_id: None,
    }
    return rewrite(PAYLOAD, cfg=cfg, **kwargs)


# --------------------------------------------------------------------------------------------------
# A routed turn ingests new sessions on a complete store (window never gates)
# --------------------------------------------------------------------------------------------------


def test_a_routed_turn_on_a_complete_store_advances_the_watermark(hermes, tmp_path):
    """Complete store + new session: the routed turn's scan ingests it and moves the watermark."""
    store_path = _complete_store_with_new_session(hermes, tmp_path)
    before = _watermark(store_path)

    result = _routed_turn(store_path, tmp_path)

    assert result is not None, "the routed turn did not route"
    assert _settled(str(store_path)), "the routed turn's scan never settled"
    assert _watermark(store_path) == str(_newest_stamp(hermes)), (
        "the scan did not advance the watermark to the newest session"
    )
    assert _watermark(store_path) != before
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s-new"}, "the routed turn's scan did not ingest the new session"
        assert rows(store, "SELECT count(*) FROM load_event")[0][0] == 2
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False, (
        "behind is still set after the scan ingested every session"
    )
    assert associations_module.request_learning_scan(str(store_path)) is False, (
        "a follow-up turn starts another scan when nothing is behind"
    )


def test_a_routed_turn_with_no_new_sessions_is_a_noop(hermes, tmp_path, monkeypatch):
    """A routed turn when newest == watermark starts no scan (idempotent)."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "noop.db"
    _scan(store_path, hermes)
    assert _watermark(store_path) == str(_newest_stamp(hermes))

    calls: List[str] = []
    real = associations_module.heal_store

    def recorder(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(associations_module, "heal_store", recorder)
    result = _routed_turn(store_path, tmp_path)

    assert result is not None
    assert calls == [], "a turn with nothing behind started a scan"
    assert associations_module.request_learning_scan(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# One-attempt serialization
# --------------------------------------------------------------------------------------------------


def test_a_second_routed_turn_while_a_scan_is_in_flight_starts_no_parallel_scan(
    hermes, tmp_path, monkeypatch
):
    """While the background scan blocks, a second routed turn starts no second scan."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "serial.db"
    # First, a caught-up complete store, then one session behind it, so the
    # incomplete-store heal has nothing to claim and the learning scan is
    # the one the turn under test starts.
    _scan(store_path, hermes)
    _add_session(hermes, "s-new", T0 + 5000)
    entered = threading.Event()
    release = threading.Event()
    calls: List[str] = []

    def blocking_entry(store_path_arg, state_db_path=None, *, catalog=(), now_ms=None):
        calls.append(store_path_arg)
        entered.set()
        release.wait(5.0)
        return None

    monkeypatch.setattr(associations_module, "heal_store", blocking_entry)
    first = _routed_turn(store_path, tmp_path)
    assert first is not None, "the first routed turn did not route"
    assert entered.wait(2.0), "the first routed turn started no scan"

    second = _routed_turn(store_path, tmp_path)
    assert second is not None, "the second routed turn did not route"
    assert associations_module.request_learning_scan(str(store_path)) is False, (
        "a scan started while another scan was in flight"
    )
    assert len(calls) == 1, "the second turn started a parallel scan"

    release.set()
    assert _settled(str(store_path))
    assert len(calls) == 1


# --------------------------------------------------------------------------------------------------
# Mid-scan arrival: behind survives the completion, next turn ingests it
# --------------------------------------------------------------------------------------------------


def test_a_session_arriving_mid_scan_is_ingested_by_the_next_routed_turn(hermes, tmp_path, monkeypatch):
    """A session that arrives while the scan runs keeps behind set after completion."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "midsan.db"
    _scan(store_path, hermes)
    assert _watermark(store_path) == str(T0 + 900_000)
    _add_session(hermes, "s-mid", T0 + 4000)

    listed = threading.Event()
    release = threading.Event()
    real_list = associations_module._list_sessions
    blocked_once = {"done": False}

    def holding_list(source, max_sessions):
        captured = real_list(source, max_sessions)
        if not blocked_once["done"]:
            blocked_once["done"] = True
            listed.set()
            release.wait(5.0)
            # Hand the backfill the list captured before the arrival, so the
            # scan ingests only the sessions that existed at its start.
            return captured[:2] if len(captured) > 2 else captured
        return captured

    monkeypatch.setattr(associations_module, "_list_sessions", holding_list)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert listed.wait(2.0), "the scan never listed sessions"

    _add_session(hermes, "s-late", T0 + 8000)
    release.set()
    assert _settled(str(store_path)), "the first scan never settled"

    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s-mid"}, "the first scan ingested a session that arrived mid-scan"
    finally:
        store.close()
    assert _watermark(store_path) == str(T0 + 904_000), "the mid-scan arrival advanced the watermark"
    assert associations_module.learning_behind(str(store_path)) is True, (
        "the completion cleared behind over the mid-scan arrival"
    )

    # The next routed turn ingests it: start a fresh scan and let it finish.
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s-mid", "s-late"}, "the next turn did not ingest the mid-scan arrival"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# Retry after a failed scan or a restart: both retry outcomes (Learning v2.1.0)
# --------------------------------------------------------------------------------------------------


def _exploding_learning(store_path_arg, state_db_path=None, *, catalog=(), now_ms=None):
    """Stand in for a learning scan whose pass failed."""
    raise sqlite3.OperationalError("database is locked")


def _run_retry_trigger(store_path: Path, hermes, trigger: str) -> None:
    """Simulate one trigger with behind set: *trigger* is ``fail`` or ``restart``."""
    _add_session(hermes, "s-first", T0)
    _scan(store_path, hermes)
    before = _watermark(store_path)
    _add_session(hermes, "s-behind", T0 + 5000)
    # First, a failed scan that keeps behind and the watermark.
    saved = associations_module.heal_store
    associations_module.heal_store = _exploding_learning
    try:
        assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    finally:
        associations_module.heal_store = saved
    assert _settled(str(store_path)), "the failed scan never settled"
    assert associations_module.learning_behind(str(store_path)) is True, (
        "a failed scan cleared behind"
    )
    assert _watermark(store_path) == before, "a failed scan advanced the watermark"
    if trigger == "restart":
        # The process dies: the in-memory latch is gone, the watermark stays.
        associations_module.reset_heal_state()
        assert associations_module.learning_behind(str(store_path)) is True, (
            "a restart cleared behind"
        )
        assert _watermark(store_path) == before, "a restart touched the watermark"
    elif trigger != "fail":
        raise ValueError(trigger)


@pytest.mark.parametrize("trigger", ["fail", "restart"])
def test_retry_after_failure_or_restart_clears_behind_without_a_new_arrival(
    hermes, tmp_path, monkeypatch, trigger
):
    """Clean retry: next routed turn scans, clears behind, and starts no further scan."""
    store_path = tmp_path / "retry-clean.db"
    _run_retry_trigger(store_path, hermes, trigger)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    assert _watermark(store_path) == str(_newest_stamp(hermes)), (
        "the retry did not advance the watermark"
    )
    assert associations_module.learning_behind(str(store_path)) is False, (
        "a clean retry did not clear behind"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a follow-up turn started another scan after a clean retry"
    )


@pytest.mark.parametrize("trigger", ["fail", "restart"])
def test_retry_after_failure_or_restart_keeps_behind_with_a_new_arrival(
    hermes, tmp_path, monkeypatch, trigger
):
    """Late retry: a session arrives during the retry, so behind stays for the next turn."""
    store_path = tmp_path / "retry-late.db"
    _run_retry_trigger(store_path, hermes, trigger)

    # The retry is the real scan, but its session listing is held until the
    # late session lands, so the retry cannot ingest it.
    listed = threading.Event()
    release = threading.Event()
    real_list = associations_module._list_sessions
    blocked_once = {"done": False}

    def holding_list(source, max_sessions):
        captured = real_list(source, max_sessions)
        if not blocked_once["done"]:
            blocked_once["done"] = True
            listed.set()
            release.wait(5.0)
            return captured[:1] if len(captured) > 1 else captured
        return captured

    monkeypatch.setattr(associations_module, "_list_sessions", holding_list)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert listed.wait(2.0), "the retry never listed sessions"
    _add_session(hermes, "s-late", T0 + 5000)
    release.set()
    assert _settled(str(store_path)), "the retry never settled"

    assert associations_module.learning_behind(str(store_path)) is True, (
        "the retry cleared behind over the session that arrived during it"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True, (
        "the next turn started no scan for the session that arrived during the retry"
    )
    assert _settled(str(store_path))
    assert associations_module.learning_behind(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# The finished flag (C2) never gates the learning scan
# --------------------------------------------------------------------------------------------------


def test_a_learning_turn_still_starts_when_the_finished_flag_is_set(hermes, tmp_path):
    """A settled scan writes the flag (C2), and the next new session still gets a scan."""
    store_path = _complete_store_with_new_session(hermes, tmp_path)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert (
            associations_module.read_meta(store, associations_module.META_HEAL_FINISHED)
            == "finished"
        ), "an uncapped backfill settle wrote no finished flag"
    finally:
        store.close()
    _add_session(hermes, "s-next", T0 + 10_000)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True, (
        "the finished flag blocked the next continuous-learning turn"
    )
    assert _settled(str(store_path))
    assert associations_module.learning_behind(str(store_path)) is False


def test_a_finished_heal_never_blocks_the_learning_scan(hermes, tmp_path):
    """Restart with the flag finished and sessions behind: learning retries, heal stays gated."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "finished.db"
    # A settled heal writes the finished flag (C2); the restart clears the latch.
    report = associations_module.heal_store(
        str(store_path), str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
    )
    assert report.sessions_scanned == 1
    _add_session(hermes, "s-new", T0 + 5 * DAY)
    associations_module.reset_heal_state()  # the restart
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the persisted flag did not gate the heal after the restart"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True, (
        "the finished flag also gated the continuous-learning scan"
    )
    assert _settled(str(store_path))
    assert associations_module.learning_behind(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# Window observation never gates: the degraded-line wording is unchanged
# --------------------------------------------------------------------------------------------------


def test_the_completeness_window_only_picks_the_degraded_line_wording(hermes, tmp_path):
    """Complete store: no history line. Stale store: the exact pre-ticket wording."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "wording.db"
    _scan(store_path, hermes)
    cfg = make_config(tmp_path, associationDbPath=str(store_path))

    complete_note = router_module._degraded_note(cfg, index=INDEX)
    assert "History incomplete" not in complete_note, (
        "a complete store gained a history line"
    )

    store = associations_module.open_store(str(store_path))
    try:
        associations_module.write_meta(
            store, associations_module.META_LAST_SCAN_MS, str(T0 - DAY)
        )
    finally:
        store.close()
    stale_note = router_module._degraded_note(cfg, index=INDEX)
    assert (
        "History incomplete (stale store); routing continues without learned history."
        in stale_note
    ), f"the stale-store degraded line changed: {stale_note!r}"

    # The window observation flips the line only: behind is unaffected by it.
    assert associations_module.learning_behind(str(store_path)) is True, (
        "making the store stale/changing the window line changed behind"
    )
