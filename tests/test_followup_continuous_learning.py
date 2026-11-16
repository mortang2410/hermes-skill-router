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
to the derived fact that some session's current source revision (its
greatest message row id) differs from its durable ``scan_ledger`` row, so
failed, skipped and merely-enumerated sessions stay behind even when a
newer session advanced the global watermark; ``inflight`` maps to the
one-attempt running latch, and ``late`` to the derived behind being set
again after a completion, so a mid-scan arrival is never missed. The
finished-heal flag (C2) gates ``request_heal`` only; it never gates the
learning scan, though a scan settle writes the flag as any uncapped
settle does. The per-session revision comparison is the implementation
derivation inside these transitions, not a new transition: the model's
state and invariants are unchanged (round 0607).

Every association database and session store is a temporary file pytest
owns, and every clock is a fixed epoch millisecond.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
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
    # (Round 0607 retired the old form of this check, ``learning_behind is
    # True`` after watermark surgery: behind now derives from per-session
    # ledger progress, not the global watermark, so lowering the watermark
    # by itself manufactures no outstanding work. s-first is fully scanned,
    # hence behind is False by derivation, and the line flip still holds.)
    assert associations_module.learning_behind(str(store_path)) is False, (
        "lowering the watermark manufactured behind over fully scanned progress"
    )


# --------------------------------------------------------------------------------------------------
# F1: a settled zero-watermark store bootstraps learning (round 0607)
# --------------------------------------------------------------------------------------------------


def _finished_flag(store_path: Path) -> Optional[str]:
    """Return the persisted heal finished flag, or None when unset."""
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        return associations_module.read_meta(store, associations_module.META_HEAL_FINISHED)
    finally:
        store.close()


def _ledger_row(store_path: Path, session_id: str):
    """Return the scan_ledger row (scanned_max_row_id, ...) for one session, or None."""
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        return store.conn.execute(
            "SELECT scanned_max_row_id FROM scan_ledger WHERE session_id = ?", (session_id,)
        ).fetchone()
    finally:
        store.close()


def test_an_empty_heal_then_a_new_session_bootstraps_learning(hermes, tmp_path):
    """F1: a heal settled against an empty source (watermark 0) must not trap learning."""
    store_path = tmp_path / "bootstrap-empty.db"
    report = associations_module.heal_store(
        str(store_path), str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
    )
    assert report.sessions_found == 0
    assert _watermark(store_path) == "0"
    assert _finished_flag(store_path) == "finished"

    assert associations_module.learning_behind(str(store_path)) is False, (
        "a settled empty store with no source data is behind"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False

    _add_session(hermes, "s-new", T0)
    assert associations_module.learning_behind(str(store_path)) is True, (
        "a settled zero-watermark store ignored a new session"
    )
    associations_module.reset_heal_state()  # a restart keeps the bootstrap condition
    assert associations_module.learning_behind(str(store_path)) is True, (
        "a restart cleared the zero-watermark bootstrap"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-new"}, "the bootstrapped scan did not ingest the new session"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a follow-up turn started another scan after the bootstrap ingest"
    )


def test_an_in_memory_finished_heal_after_a_failed_flag_write_bootstraps_learning(
    hermes, tmp_path, monkeypatch
):
    """M1: first heal settles empty but its finished-flag write fails transiently.

    ``_run_heal`` still records the attempt as finished in memory, so
    ``request_heal`` refuses the latch; once writes recover and source data
    arrives in the same process, ``learning_behind`` must still bootstrap.
    The failure is fail-once (same-process recovery), not a restart.
    """
    store_path = tmp_path / "bootstrap-memory-flag.db"
    real_write = associations_module._write_heal_finished
    attempts = {"n": 0}

    def flaky_write(store):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise sqlite3.Error("finished flag write failed (transient)")
        return real_write(store)

    monkeypatch.setattr(associations_module, "_write_heal_finished", flaky_write)

    # First heal settles against an empty source via _run_heal, so the
    # in-memory latch is finished even though no flag was persisted.
    associations_module._run_heal(
        associations_module.heal_store,
        str(store_path),
        str(hermes.path),
        catalog=CATALOG,
        now_ms=T0 + 10 * DAY,
    )
    assert attempts["n"] == 1
    assert _watermark(store_path) == "0"
    assert _finished_flag(store_path) is None, "the failed flag write was persisted"
    assert associations_module.heal_state(str(store_path)) == "finished"
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the in-memory finished latch no longer gates the heal"
    )
    assert associations_module.learning_behind(str(store_path)) is False, (
        "a settled empty store with no source data is behind"
    )

    _add_session(hermes, "s-new", T0)
    assert _finished_flag(store_path) is None
    assert associations_module.learning_behind(str(store_path)) is True, (
        "an in-memory-finished zero-watermark store ignored a new session"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-new"}, "the bootstrapped scan did not ingest the new session"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a follow-up turn started another scan after the bootstrap ingest"
    )


def test_a_thread_start_failure_latch_bootstraps_learning(hermes, tmp_path, monkeypatch):
    """M1: a thread that cannot start settles the heal in memory only,
    and later data must still bootstrap.

    The same store with data is asserted unsettled (heal-owned) before the
    failure and settled (learning-owned) after it: the contrast isolates the
    in-memory latch as the bootstrap signal.
    """
    store_path = tmp_path / "bootstrap-memory-thread.db"
    store = associations_module.open_store(str(store_path))
    store.close()
    _add_session(hermes, "s-new", T0)
    assert _watermark(store_path) == "0"
    assert _finished_flag(store_path) is None
    assert associations_module.learning_behind(str(store_path)) is False, (
        "an unsettled zero-watermark store is not the heal's bootstrap"
    )

    real_start = threading.Thread.start
    attempts = {"n": 0}

    def flaky_start(self):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("cannot start heal thread (transient)")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the thread-start failure must refuse the heal launch"
    )
    assert associations_module.heal_state(str(store_path)) == "finished"
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the in-memory finished latch no longer gates the heal"
    )
    assert _finished_flag(store_path) is None
    assert associations_module.learning_behind(str(store_path)) is True, (
        "a thread-start-failure latch ignored later data"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-new"}, "the bootstrapped scan did not ingest the new session"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


def test_a_thread_start_failure_with_no_store_file_bootstraps_learning(
    hermes, tmp_path, monkeypatch
):
    """Finder B micro delta: thread-start failure with no association file.

    First use with no file: an unsettled absent store stays heal-owned
    (``learning_behind`` is False and no scan starts, and neither creates
    the file). A transient ``Thread.start`` failure records an in-memory
    finished heal without creating the file; once thread creation
    recovers in the same process, the absent store with source data must
    bootstrap learning — ``learning_behind`` treats the genuinely absent
    path as bootstrap-eligible on the in-memory finished attempt, and
    the learning scan creates the store and ingests. An existing but
    unreadable or foreign store never reaches this bootstrap (doctored
    safeguard), pinned separately below with the same latch trigger.
    """
    store_path = tmp_path / "bootstrap-no-file.db"
    _add_session(hermes, "s-new", T0)
    assert not os.path.exists(str(store_path)), "the fixture precreated the store"
    assert associations_module.learning_behind(str(store_path)) is False, (
        "an unsettled absent store is not the heal's bootstrap"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a scan started over an unsettled absent store"
    )
    assert not os.path.exists(str(store_path)), "the unsettled check created the store"

    real_start = threading.Thread.start
    attempts = {"n": 0}

    def flaky_start(self):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("cannot start heal thread (transient)")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the thread-start failure must refuse the heal launch"
    )
    assert attempts["n"] == 1
    assert associations_module.heal_state(str(store_path)) == "finished"
    assert not os.path.exists(str(store_path)), "the thread-start failure created the store"
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the in-memory finished latch no longer gates the heal"
    )
    assert associations_module.learning_behind(str(store_path)) is True, (
        "an absent store with an in-memory finished attempt ignored later data"
    )
    assert not os.path.exists(str(store_path)), "learning_behind created the store"
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    assert os.path.exists(str(store_path)), "the bootstrapped scan did not create the store"
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-new"}, "the bootstrapped scan did not ingest the new session"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a follow-up turn started another scan after the bootstrap ingest"
    )


def test_an_existing_corrupt_store_with_finished_latch_stays_heal_owned(
    hermes, tmp_path, monkeypatch
):
    """Doctored safeguard: the same finished latch over an existing file.

    With a corrupt existing association file, a thread-start failure
    still records an in-memory finished heal, but ``learning_behind``
    stays False and no learning scan starts: an existing but
    unreadable or foreign file is heal-owned, never safely creatable
    by the bootstrap. The file bytes are unchanged.
    """
    store_path = tmp_path / "corrupt-latch.db"
    store_path.write_text("this is not a database\n", encoding="utf-8")
    before = store_path.read_bytes()
    _add_session(hermes, "s-new", T0)

    real_start = threading.Thread.start
    attempts = {"n": 0}

    def flaky_start(self):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("cannot start heal thread (transient)")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False
    assert attempts["n"] == 1
    assert associations_module.heal_state(str(store_path)) == "finished"
    assert associations_module.learning_behind(str(store_path)) is False, (
        "an existing corrupt store with a finished latch claimed the bootstrap"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a scan started over an existing corrupt store"
    )
    assert store_path.read_bytes() == before, "the corrupt store was modified by the check"


def test_a_failed_first_heal_then_recovery_bootstraps_learning(hermes, tmp_path):
    """F1: a first heal whose pass failed (watermark 0) must not trap learning."""
    store_path = tmp_path / "bootstrap-failed.db"
    with pytest.raises(sqlite3.Error):
        associations_module.heal_store(
            str(store_path), str(tmp_path / "no-state.db"), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
    assert _watermark(store_path) == "0"
    assert _finished_flag(store_path) == "finished"

    _add_session(hermes, "s-recovered", T0)
    associations_module.reset_heal_state()  # a restart keeps the bootstrap condition
    assert associations_module.learning_behind(str(store_path)) is True, (
        "a failed first heal with the flag settled ignored later data"
    )
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "the finished flag no longer gates the heal (C2)"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-recovered"}, "the recovery scan did not ingest the new session"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# F2: outstanding work derives from per-session scan progress (round 0607)
# --------------------------------------------------------------------------------------------------


def test_a_transiently_unreadable_session_is_retried_after_a_newer_session_succeeds(
    hermes, tmp_path, monkeypatch
):
    """F2: s2 fails to read once while newer s3 succeeds; the next turn must learn s2."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "retry-skipped.db"
    _scan(store_path, hermes)
    _add_session(hermes, "s2", T0 + 1000)
    _add_session(hermes, "s3", T0 + 2000)

    real_rows = associations_module._session_rows
    failed_once = {"done": False}

    def flaky_rows(source, session_id):
        if session_id == "s2" and not failed_once["done"]:
            failed_once["done"] = True
            return None
        return real_rows(source, session_id)

    monkeypatch.setattr(associations_module, "_session_rows", flaky_rows)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))

    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s3"}, "the scan with the skipped session ingested the wrong set"
    finally:
        store.close()
    assert _ledger_row(store_path, "s2") is None, "the skipped session gained a ledger row"
    assert _watermark(store_path) == str(T0 + 2000 + 900_000), "the newer session did not advance the watermark"
    assert associations_module.learning_behind(str(store_path)) is True, (
        "the skipped session is hidden behind the newer session's watermark"
    )

    # The flake is over (the patched reader delegates after one failure): the
    # next routed turn must learn s2.
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s2", "s3"}, "the retry did not ingest the skipped session"
    finally:
        store.close()
    assert _ledger_row(store_path, "s2") is not None
    assert associations_module.learning_behind(str(store_path)) is False


def test_messages_arriving_in_an_empty_tied_session_are_ingested_next_turn(hermes, tmp_path):
    """F2: messages landing in an already-enumerated empty session at a tied stamp are ingested."""
    _add_session(hermes, "s-anchor", T0 + 8000)
    store_path = tmp_path / "tied-empty.db"
    _scan(store_path, hermes)
    assert _watermark(store_path) == str(T0 + 8000 + 900_000)

    hermes.session("s-empty", started_ms=T0 + 8000)  # stamp ties the anchor's
    hermes.commit()
    _scan(store_path, hermes)
    row = _ledger_row(store_path, "s-empty")
    assert row is not None and row[0] is None, "the empty session did not commit a NULL ledger row"
    assert _newest_stamp(hermes) == int(_watermark(store_path)), "the fixture must tie the watermark"
    assert associations_module.learning_behind(str(store_path)) is False, (
        "an empty tied session is behind before its messages arrive"
    )

    hermes.user("s-empty", f"{SUBSTANTIVE} s-empty-late", T0 + 8000 + 100)
    hermes.skill_view("s-empty", "code-review", T0 + 8000 + 200)
    hermes.commit()
    assert _newest_stamp(hermes) == int(_watermark(store_path)), "the late messages must not move the newest stamp"
    assert associations_module.learning_behind(str(store_path)) is True, (
        "messages in an already-enumerated session at a tied stamp are missed"
    )

    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-anchor", "s-empty"}, "the next turn did not ingest the late messages"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


def test_new_data_in_an_already_scanned_session_below_the_watermark_is_ingested(hermes, tmp_path):
    """F2: a new message in a scanned session whose stamp is below the watermark is ingested."""
    _add_session(hermes, "s-first", T0)
    _add_session(hermes, "s-later", T0 + 9000)
    store_path = tmp_path / "below-watermark.db"
    _scan(store_path, hermes)
    assert _watermark(store_path) == str(T0 + 9000 + 900_000)

    hermes.user("s-first", f"{SUBSTANTIVE} s-first-again", T0 + 500)
    hermes.skill_view("s-first", "code-review", T0 + 600)
    hermes.commit()
    assert associations_module.learning_behind(str(store_path)) is True, (
        "new data in an already-scanned session below the watermark is missed"
    )

    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        prompts = rows(
            store,
            "SELECT count(*) FROM prompt_event WHERE session_id = 's-first'",
        )[0][0]
        assert prompts == 2, "the next turn did not ingest the second prompt of s-first"
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


# --------------------------------------------------------------------------------------------------
# F3: restart kills a scan genuinely in flight (round 0607)
# --------------------------------------------------------------------------------------------------

_CHILD_DRIVER = '''\
import sys
import time
from pathlib import Path

root, store, state, ready, go = sys.argv[1:6]
sys.path.insert(0, root)
sys.path.insert(0, root + "/tests")
from hermes_skill_router import associations as m
from test_associations import CATALOG

m.default_state_db_path = lambda: state
real_write = m._write_session
paused = {"done": False}


def pausing_write(store_, session, scan, report):
    out = real_write(store_, session, scan, report)
    if not paused["done"]:
        paused["done"] = True
        Path(ready).write_text("ready")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not Path(go).exists():
            time.sleep(0.02)
    return out


m._write_session = pausing_write
m.request_learning_scan(store, catalog=CATALOG)
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    time.sleep(0.05)
'''


_NEVER_READY_DRIVER = '''\
import sys
import time

deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    time.sleep(0.05)
'''


def _kill_scan_midflight(
    tmp_path: Path,
    hermes,
    store_path: Path,
    *,
    readiness_timeout: float = 20.0,
    driver_source: str = _CHILD_DRIVER,
    _proc_holder: Optional[dict] = None,
) -> None:
    """Start a real scan in a subprocess, pause it after its first session
    commits, and SIGKILL it. Asserts the persisted watermark was not advanced.
    Bounded: a 20 s readiness wait, a 10 s kill wait (repeated as a
    kill-if-running plus bounded wait inside ``finally``, so a readiness
    failure cannot orphan the child), and a file-count cap.
    """
    ready = tmp_path / "child-ready"
    go = tmp_path / "child-go"
    driver = tmp_path / "kill_child.py"
    driver.write_text(driver_source)
    driver_argv = [
        sys.executable,
        "-B",
        str(driver),
        str(ROOT),
        str(store_path),
        str(hermes.path),
        str(ready),
        str(go),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    log = open(tmp_path / "child.log", "w")
    try:
        proc = subprocess.Popen(
            driver_argv, stdout=log, stderr=subprocess.STDOUT, cwd=str(tmp_path), env=env
        )
    except Exception:
        log.close()
        raise
    if _proc_holder is not None:
        _proc_holder["proc"] = proc
    try:
        deadline = time.monotonic() + readiness_timeout
        while time.monotonic() < deadline and not ready.exists():
            if proc.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), "the child scan never paused before the kill"
        proc.kill()
        rc = proc.wait(timeout=10.0)
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
        finally:
            log.close()
    assert rc == -9, f"the child was not killed (rc={rc})"
    assert len(list(tmp_path.rglob("*"))) < 200, "the kill probe created too many files"


def test_restart_killing_an_inflight_scan_retries_cleanly(hermes, tmp_path):
    """F3 clean retry: the process dies mid-scan; the next turn retries and ingests once."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "kill-clean.db"
    _scan(store_path, hermes)
    before = _watermark(store_path)
    # Two sessions are behind; the child commits the newest first and then
    # pauses, so one session stays genuinely unprocessed after the kill.
    _add_session(hermes, "s-behind", T0 + 4000)
    _add_session(hermes, "s-behind2", T0 + 5000)

    _kill_scan_midflight(tmp_path, hermes, store_path)
    assert _watermark(store_path) == before, "the killed scan advanced the watermark"
    assert associations_module.learning_behind(str(store_path)) is True, (
        "the killed scan cleared behind"
    )

    # The ledger for s-behind2 committed before the kill, so the retry
    # rescans it but must store nothing twice.
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    assert _watermark(store_path) == str(_newest_stamp(hermes))
    assert associations_module.learning_behind(str(store_path)) is False
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        loads = rows(store, "SELECT count(*) FROM load_event")[0][0]
        assert loads == 3, "the retry after the kill duplicated or lost a load"
    finally:
        store.close()
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is False, (
        "a follow-up turn started another scan after the clean retry"
    )


def test_restart_killing_an_inflight_scan_retries_with_a_late_arrival(hermes, tmp_path, monkeypatch):
    """F3 late retry: after the kill, session s-late arrives during the retry."""
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "kill-late.db"
    _scan(store_path, hermes)
    before = _watermark(store_path)
    _add_session(hermes, "s-behind", T0 + 4000)
    _add_session(hermes, "s-behind2", T0 + 5000)

    _kill_scan_midflight(tmp_path, hermes, store_path)
    assert _watermark(store_path) == before, "the killed scan advanced the watermark"

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
            return captured
        return captured

    monkeypatch.setattr(associations_module, "_list_sessions", holding_list)
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert listed.wait(2.0), "the retry never listed sessions"
    _add_session(hermes, "s-late", T0 + 8000)
    release.set()
    assert _settled(str(store_path))

    assert associations_module.learning_behind(str(store_path)) is True, (
        "the retry cleared behind over the session that arrived during it"
    )
    assert associations_module.request_learning_scan(str(store_path), catalog=CATALOG) is True
    assert _settled(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        stored = {row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")}
        assert stored == {"s-first", "s-behind", "s-behind2", "s-late"}, (
            "the next turn did not ingest the session that arrived during the retry"
        )
    finally:
        store.close()
    assert associations_module.learning_behind(str(store_path)) is False


def test_readiness_failure_reaps_the_child(hermes, tmp_path):
    """M2: a child that never signals ready is still killed and waited for.

    Uses the same tmp_path hygiene as the kill tests: the never-ready driver
    mirrors the normal driver's 300 s loop, the readiness wait is short
    (2 s), and the probe asserts the subprocess handle is reaped (poll is
    not None) after the helper raises, so no orphan can race tmp cleanup.
    """
    _add_session(hermes, "s-first", T0)
    store_path = tmp_path / "kill-never.db"
    _scan(store_path, hermes)
    holder = {}
    with pytest.raises(AssertionError, match="never paused"):
        _kill_scan_midflight(
            tmp_path,
            hermes,
            store_path,
            readiness_timeout=2.0,
            driver_source=_NEVER_READY_DRIVER,
            _proc_holder=holder,
        )
    proc = holder.get("proc")
    assert proc is not None, "the helper exposed no child handle for the reaping check"
    assert proc.poll() is not None, "the never-ready child was left alive after the failure"
    assert len(list(tmp_path.rglob("*"))) < 200, "the readiness probe created too many files"
