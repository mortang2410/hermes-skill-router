"""Contract C6: the background backfill never blocks a turn, is uncapped, idempotent, and resumeable.

Ticket 04 landed the C4 entry points the heal runs through (``heal_store``, ``request_heal`` and the
one-attempt latch) and the completeness rule this job satisfies. This file pins the C6 behaviour of
the job itself:

- the first run has no session cap: the pass walks the whole session store and leaves a scan
  watermark only after it completes;
- a second run inserts nothing, because ``prompt_event`` and ``load_event`` uniqueness, not a
  pre-check, is what makes it idempotent;
- an interrupted pass is safe to rerun at run granularity: the rows already committed are skipped
  by that uniqueness, no mid-pass cursor was written, and the rerun finishes the store;
- the launch returns without waiting for the job, so a turn never blocks on it;
- a failure warns once in the host logger and hands exactly one note to contract C5's turn-side
  chooser; a completed pass and an observed complete store drop that note, and completion logs only
  in the host logger;
- ``state_db_path`` is ``default_state_db_path()`` when the caller passes none.

The real job runs here, so this module shadows ``tests/conftest.py``'s ``no_live_heal`` guard and
points every job at its own fixture session store. Every association database and session store is a
temporary file pytest owns, and every clock is a fixed epoch millisecond.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router import associations as associations_module  # noqa: E402
from tests.test_associations import (  # noqa: E402
    CATALOG,
    DAY,
    SUBSTANTIVE,
    T0,
    HermesStoreFixture,
    rows,
)

#: Enough sessions that a pass with any leftover session cap cannot cover the whole store.
SESSIONS = 30


@pytest.fixture
def hermes(tmp_path: Path) -> Iterable[HermesStoreFixture]:
    """A temporary Hermes session store this suite owns."""
    fixture = HermesStoreFixture(tmp_path / "state.db")
    yield fixture
    fixture.close()


@pytest.fixture(autouse=True)
def no_live_heal():
    """Shadow the conftest guard: this module drives the real C6 job over its own stores.

    The one-attempt latch is global to the process, so every case starts and ends with no attempt
    recorded.
    """
    associations_module.reset_heal_state()
    yield
    associations_module.reset_heal_state()


def _populate(hermes: HermesStoreFixture, count: int = SESSIONS) -> None:
    """Write *count* human sessions, each with one typed prompt and one skill_view load."""
    for index in range(count):
        base = T0 + index * 2000
        session_id = f"s-{index:03d}"
        hermes.session(session_id, started_ms=base)
        hermes.user(session_id, f"{SUBSTANTIVE} case {index}", base + 100)
        hermes.skill_view(session_id, CATALOG[index % len(CATALOG)].name, base + 200)
    hermes.commit()


def _newest_stamp(count: int = SESSIONS) -> int:
    """Return the newest session stamp ``_populate`` writes: its last activity, 900 s after start."""
    return T0 + (count - 1) * 2000 + 900_000


def _counts(store: associations_module.AssociationStore) -> list:
    """Return the row and weight totals a rerun must leave untouched."""
    return [
        rows(store, "SELECT count(*) FROM skill"),
        rows(store, "SELECT count(*) FROM skill_alias"),
        rows(store, "SELECT count(*) FROM prompt_event"),
        rows(store, "SELECT count(*) FROM load_event"),
        rows(store, "SELECT count(*), COALESCE(round(sum(weight), 6), 0), "
                    "COALESCE(sum(support), 0) FROM association"),
    ]


def _settled(store_path: str, timeout: float = 5.0) -> bool:
    """Wait until the recorded attempt for *store_path* is no longer running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
        if state != associations_module._HEAL_RUNNING:
            return True
        time.sleep(0.01)
    return False


def _pending_failure(store_path: str) -> bool:
    """Return whether a failure note waits for contract C5, without consuming it."""
    with associations_module._HEAL_LOCK:
        return store_path in associations_module._HEAL_FAILURES


def _open(store_path: Path, *, read_only: bool = True) -> associations_module.AssociationStore:
    """Open the association store this module created for inspection."""
    return associations_module.open_store(str(store_path), read_only=read_only)


def _heal(store_path: Path, hermes: HermesStoreFixture, **kwargs):
    """Run one real C6 job against the fixture session store."""
    return associations_module.heal_store(
        str(store_path), str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY, **kwargs
    )


class _Interrupted(Exception):
    """A simulated process death in the middle of a pass."""


# --------------------------------------------------------------------------------------------------
# Uncapped first run, idempotent rerun
# --------------------------------------------------------------------------------------------------


def test_the_first_heal_run_is_uncapped_and_learns_from_the_whole_store(
    hermes, tmp_path, monkeypatch
):
    """The job passes no session cap and the store ends up covering every session."""
    _populate(hermes)
    store_path = tmp_path / "uncapped.db"
    seen = {}
    real_backfill = associations_module.backfill

    def spy(store, state_db_path, **kwargs):
        seen.update(kwargs)
        return real_backfill(store, state_db_path, **kwargs)

    monkeypatch.setattr(associations_module, "backfill", spy)
    report = _heal(store_path, hermes)

    assert "max_sessions" not in seen, "the first heal run passed a session cap"
    assert report.sessions_found == SESSIONS
    assert report.sessions_scanned == SESSIONS, "the pass skipped part of the store it found"
    assert report.loads == SESSIONS

    store = _open(store_path)
    try:
        stored = {
            row[0] for row in rows(store, "SELECT DISTINCT session_id FROM prompt_event")
        }
        assert stored == {f"s-{index:03d}" for index in range(SESSIONS)}, (
            "the pass did not learn from every session in the store"
        )
        assert rows(store, "SELECT count(*) FROM load_event")[0][0] == SESSIONS
        watermark = associations_module.read_meta(store, associations_module.META_LAST_SCAN_MS)
        assert watermark == str(_newest_stamp()), (
            "an uncapped pass writes the watermark after it completes"
        )
    finally:
        store.close()


def test_a_second_heal_run_inserts_nothing_and_leaves_the_store_bit_for_bit(hermes, tmp_path):
    """A rerun walks the same sessions and adds no row, weight or watermark change."""
    _populate(hermes)
    store_path = tmp_path / "idempotent.db"
    first = _heal(store_path, hermes)
    store = _open(store_path)
    try:
        before = _counts(store)
        first_watermark = associations_module.read_meta(
            store, associations_module.META_LAST_SCAN_MS
        )
        second = _heal(store_path, hermes)
        assert second.sessions_scanned == SESSIONS
        assert second.loads == 0, "the rerun inserted a load the store already held"
        assert second.prompts == 0, "the rerun inserted a prompt the store already held"
        assert second.duplicates == first.loads, "the rerun did not report the rows as duplicates"
        assert second.ambient_loads == 0, "the report describes this run's inserts only"
        assert _counts(store) == before, "a rerun changed a stored row or an association weight"
        assert associations_module.read_meta(
            store, associations_module.META_LAST_SCAN_MS
        ) == first_watermark, "a rerun without new sessions moved the scan watermark"
    finally:
        store.close()


# --------------------------------------------------------------------------------------------------
# Interruption safety at run granularity
# --------------------------------------------------------------------------------------------------


def test_an_interrupted_heal_reruns_to_completion_without_duplicates(
    hermes, tmp_path, monkeypatch
):
    """A pass that dies mid-store leaves no cursor and no duplicate; the rerun finishes it."""
    _populate(hermes)
    store_path = tmp_path / "interrupted.db"
    real_write = associations_module._write_session
    calls = {"count": 0}

    def die_after_ten_sessions(store, session, scan, report):
        calls["count"] += 1
        if calls["count"] == 11:
            raise _Interrupted("simulated process death")
        return real_write(store, session, scan, report)

    monkeypatch.setattr(associations_module, "_write_session", die_after_ten_sessions)
    with pytest.raises(_Interrupted):
        _heal(store_path, hermes)

    store = _open(store_path)
    try:
        assert rows(store, "SELECT count(*) FROM prompt_event")[0][0] == 10, (
            "the sessions committed before the interruption were not kept"
        )
        assert rows(store, "SELECT count(*) FROM load_event")[0][0] == 10
        assert associations_module.read_meta(store, associations_module.META_LAST_SCAN_MS) == "0", (
            "an interrupted pass wrote the scan watermark, so a rerun could skip the rest"
        )
    finally:
        store.close()

    monkeypatch.setattr(associations_module, "_write_session", real_write)
    resumed = _heal(store_path, hermes)
    assert resumed.sessions_scanned == SESSIONS
    assert resumed.loads == SESSIONS - 10, "the rerun re-stored rows the interrupted pass wrote"
    assert resumed.duplicates == 10, "the rerun did not recognise the committed rows"

    store = _open(store_path)
    try:
        assert rows(store, "SELECT count(*) FROM prompt_event")[0][0] == SESSIONS
        assert rows(store, "SELECT count(*) FROM load_event")[0][0] == SESSIONS
        assert rows(store, "SELECT count(DISTINCT dedup_key) FROM load_event")[0][0] == SESSIONS
        assert associations_module.read_meta(
            store, associations_module.META_LAST_SCAN_MS
        ) == str(_newest_stamp())
        assert associations_module.is_store_complete(store, str(hermes.path)) is True, (
            "the rerun did not make the store complete"
        )
    finally:
        store.close()


# --------------------------------------------------------------------------------------------------
# Never blocks the turn, default state path
# --------------------------------------------------------------------------------------------------


def test_the_heal_launch_returns_while_the_job_is_still_running(hermes, tmp_path, monkeypatch):
    """request_heal starts the job and returns; a second request is refused without waiting."""
    store_path = tmp_path / "blocking.db"
    associations_module.open_store(str(store_path)).close()
    entered = threading.Event()
    release = threading.Event()

    def blocking_entry(store_path_arg, state_db_path=None, *, catalog=(), now_ms=None):
        entered.set()
        release.wait(5.0)
        return None

    monkeypatch.setattr(associations_module, "heal_store", blocking_entry)
    started = time.monotonic()
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is True
    elapsed = time.monotonic() - started
    assert entered.wait(2.0), "the background job never started"
    assert elapsed < 2.5, "the launch waited for the job instead of returning to the turn"
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "a second request started or waited for another job"
    )
    release.set()
    assert _settled(str(store_path))


def test_the_heal_job_reads_the_default_session_store_when_none_is_named(
    hermes, tmp_path, monkeypatch
):
    """C6 takes state_db_path from default_state_db_path() unless a _Spec row lands first."""
    _populate(hermes)
    store_path = tmp_path / "default.db"
    monkeypatch.setattr(associations_module, "default_state_db_path", lambda: str(hermes.path))
    default = inspect.signature(associations_module.heal_store).parameters["state_db_path"].default
    assert default is None
    report = associations_module.heal_store(
        str(store_path), catalog=CATALOG, now_ms=T0 + 10 * DAY
    )
    assert report.sessions_found == SESSIONS
    store = _open(store_path)
    try:
        assert rows(store, "SELECT count(*) FROM prompt_event")[0][0] == SESSIONS, (
            "the job did not read the session store default_state_db_path() names"
        )
    finally:
        store.close()


# --------------------------------------------------------------------------------------------------
# Failure and completion surfaces
# --------------------------------------------------------------------------------------------------


def _exploding_entry(store_path_arg, state_db_path=None, *, catalog=(), now_ms=None):
    """Stand in for a job whose pass failed before it could finish."""
    raise sqlite3.OperationalError("database is locked")


def test_a_failed_heal_warns_in_the_host_log_and_hands_one_note_to_c5(
    hermes, tmp_path, caplog, capsys, monkeypatch
):
    """Failure: one host-logger warning plus exactly one C5 note, and nothing on stderr."""
    store_path = str(tmp_path / "failing.db")
    monkeypatch.setattr(associations_module, "heal_store", _exploding_entry)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(store_path, str(hermes.path), catalog=CATALOG)
        assert _settled(store_path), "the failed job never settled its attempt"

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "background heal" in record.getMessage()
    ]
    assert len(warnings) == 1, "a failed heal did not warn exactly once in the host logger"
    assert store_path in warnings[0].getMessage(), "the warning did not name the store it healed"

    note = associations_module.heal_failure_note(store_path)
    assert note == associations_module.HEAL_FAILURE_NOTE
    assert note and "\n" not in note, "the injected message must be one line"
    assert associations_module.heal_failure_note(store_path) is None, (
        "a single failure handed C5 a second note"
    )
    assert capsys.readouterr().err == "", "the background job wrote to stderr"


def test_a_completed_heal_logs_only_completion_and_injects_no_note(
    hermes, tmp_path, caplog, capsys, monkeypatch
):
    """A completed heal logs one completion line and injects no failure note."""
    _populate(hermes)
    store_path = str(tmp_path / "recovering.db")
    real_heal = associations_module.heal_store

    monkeypatch.setattr(associations_module, "heal_store", _exploding_entry)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(store_path, str(hermes.path), catalog=CATALOG)
        assert _settled(store_path)
    assert _pending_failure(store_path), "the failed attempt recorded no note for C5"

    # A host restart is what clears the one-attempt latch in production; here the latch is reset.
    associations_module.reset_heal_state()
    caplog.clear()
    monkeypatch.setattr(associations_module, "heal_store", real_heal)
    with caplog.at_level(logging.INFO, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(store_path, str(hermes.path), catalog=CATALOG)
        assert _settled(store_path), "the successful job never settled its attempt"

    completions = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO and "background heal" in record.getMessage()
    ]
    assert len(completions) == 1, "a completed heal did not log exactly one host-logger line"
    assert "finished" in completions[0].getMessage()
    assert "sessions" in completions[0].getMessage(), "the completion line did not report the pass"
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING], (
        "a successful heal warned"
    )
    assert associations_module.heal_failure_note(store_path) is None, (
        "a completed heal injected a failure note"
    )
    assert capsys.readouterr().err == "", "the background job wrote to stderr"


def test_an_observed_complete_store_drops_a_pending_failure_note(
    hermes, tmp_path, caplog, monkeypatch
):
    """A turn that observes the store complete drops a failure C5 has not read yet."""
    store_path = str(tmp_path / "recovered-elsewhere.db")
    monkeypatch.setattr(associations_module, "heal_store", _exploding_entry)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(store_path, str(hermes.path), catalog=CATALOG)
        assert _settled(store_path)
    assert _pending_failure(store_path), "the failed attempt recorded no note for C5"
    associations_module.note_store_complete(store_path)
    assert associations_module.heal_failure_note(store_path) is None, (
        "a complete store kept a stale failure note pending"
    )


def test_a_failure_on_a_store_another_writer_completed_surfaces_no_note(
    hermes, tmp_path, caplog, monkeypatch
):
    """A failed pass whose store is complete by the time it settles drops its note instead."""
    _populate(hermes)
    store_path = tmp_path / "completed-elsewhere.db"
    _heal(store_path, hermes)
    associations_module.reset_heal_state()
    caplog.clear()

    monkeypatch.setattr(associations_module, "heal_store", _exploding_entry)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(
            str(store_path), str(hermes.path), catalog=CATALOG
        )
        assert _settled(str(store_path))
    assert associations_module.heal_failure_note(str(store_path)) is None, (
        "a failure on a complete store was surfaced as a stale note"
    )
    assert [record for record in caplog.records if record.levelno == logging.WARNING], (
        "the failure was not warned in the host logger"
    )


def test_a_launch_that_cannot_start_a_thread_is_a_failed_heal(tmp_path, caplog, monkeypatch):
    """A thread that cannot start warns once and hands C5 the same one-line note."""
    store_path = str(tmp_path / "unstartable.db")

    class UnstartableThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        associations_module, "threading", SimpleNamespace(Thread=UnstartableThread)
    )
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.associations"):
        assert associations_module.request_heal(store_path) is False
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "cannot start the background heal" in record.getMessage()
    ]
    assert len(warnings) == 1, "a launch failure did not warn exactly once"
    assert (
        associations_module.heal_failure_note(store_path)
        == associations_module.HEAL_FAILURE_NOTE
    )
    with associations_module._HEAL_LOCK:
        state = associations_module._HEAL_STATE.get(store_path)
    assert state == associations_module._HEAL_FINISHED, "a launch failure left the attempt running"
