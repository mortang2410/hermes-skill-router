"""Ticket 02 (C2): heal finished flag, loss marker, v1 to v2 migration and finality ledger.

The finished-scan flag persists in the store ``meta`` table, so a restart does
not re-heal what already finished; ``last_history_loss_ms`` records every
destructive path; the v1 to v2 migration seeds that marker for migrated stores
in its stamp phase; every successful session scan commits its ``scan_ledger``
finality row in the same transaction as its prompts and loads; and the parser
records unreadable calls and results instead of dropping them silently.

Every association database and session store is a temporary file pytest owns,
and every clock is a fixed epoch millisecond.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path
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


@pytest.fixture
def hermes(tmp_path: Path) -> Iterable[HermesStoreFixture]:
    """A temporary Hermes session store this suite owns."""
    fixture = HermesStoreFixture(tmp_path / "state.db")
    yield fixture
    fixture.close()


@pytest.fixture(autouse=True)
def no_live_heal():
    """Shadow the conftest guard: this module drives the real heal entry point.

    The one-attempt latch is global to the process, so every case starts and
    ends with no attempt recorded. Clearing it simulates a restart: the
    in-memory state is gone, only the persisted ``meta`` flag remains.
    """
    associations_module.reset_heal_state()
    yield
    associations_module.reset_heal_state()


def _populate(hermes: HermesStoreFixture, count: int = 3) -> None:
    """Write *count* human sessions, each with one prompt and one skill_view load."""
    for index in range(count):
        base = T0 + index * 2000
        session_id = f"s-{index:03d}"
        hermes.session(session_id, started_ms=base)
        hermes.user(session_id, f"{SUBSTANTIVE} case {index}", base + 100)
        hermes.skill_view(session_id, CATALOG[index % len(CATALOG)].name, base + 200)
    hermes.commit()


def _heal(store_path: Path, hermes: HermesStoreFixture, **kwargs):
    """Run one real uncapped heal against the fixture session store."""
    return associations_module.heal_store(
        str(store_path), str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY, **kwargs
    )


def _ledger(store, session_id: str):
    """Return one session's ``scan_ledger`` row as plain tuples, or an empty list."""
    return rows(
        store,
        "SELECT session_id, scanned_max_row_id, pending_calls, unreadable_results,"
        " unreadable_calls FROM scan_ledger WHERE session_id = ?",
        (session_id,),
    )


def _flag(store) -> str:
    """Return the persisted heal finished flag value, or ``None`` when unset."""
    return associations_module.read_meta(store, associations_module.META_HEAL_FINISHED)


def _wait_settled(store_path: str, timeout: float = 5.0) -> bool:
    """Wait until the recorded attempt for *store_path* is no longer running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
        if state != associations_module._HEAL_RUNNING:
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------------------------------
# The finished flag: settle, restart gate, observe-complete, reset
# --------------------------------------------------------------------------------------------------


def test_a_settled_heal_writes_the_finished_flag(hermes, tmp_path):
    """An uncapped backfill that settles writes the flag into ``meta``."""
    _populate(hermes)
    store_path = tmp_path / "settled.db"
    report = _heal(store_path, hermes)
    assert report.sessions_scanned == 3
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _flag(store) == "finished", "the settle did not persist the finished flag"
    finally:
        store.close()


def test_a_failed_settle_also_writes_the_finished_flag(hermes, tmp_path, monkeypatch):
    """A backfill that fails against a writable store still settles finished, with the flag."""
    _populate(hermes)
    store_path = tmp_path / "failed.db"
    monkeypatch.setattr(
        associations_module,
        "_write_session",
        lambda store, session, scan, report: (_ for _ in ()).throw(
            sqlite3.OperationalError("database is locked")
        ),
    )
    report = _heal(store_path, hermes)
    assert report.skip_reasons["write-failed"] == 3, "every session write failed"
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _flag(store) == "finished", "a failed settle wrote no finished flag"
    finally:
        store.close()


def test_a_restarted_process_starts_no_second_heal_until_observed_complete(
    hermes, tmp_path, monkeypatch
):
    """Restart restores the finished state from ``meta``; observing complete clears it once."""
    _populate(hermes, count=3)
    # Make the store incomplete after the heal by ageing the newest session far
    # past the watermark window (append a newer session the heal never saw).
    started = T0 + 5_000_000
    hermes.session("s-new", started_ms=started)
    hermes.user("s-new", f"{SUBSTANTIVE} new", started + 100)
    hermes.skill_view("s-new", "code-review", started + 200)
    hermes.commit()
    store_path = tmp_path / "restart.db"
    _heal(store_path, hermes)  # settles and writes the flag, then sees s-new missing

    calls = []
    monkeypatch.setattr(
        associations_module,
        "heal_store",
        lambda *args, **kwargs: calls.append(args[0]),
    )
    associations_module.reset_heal_state()  # the restart
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is False, (
        "a second heal started while the persisted flag reads finished"
    )
    assert calls == [], "the restart launched a heal thread"
    associations_module.note_store_complete(str(store_path))
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _flag(store) is None, "observing complete did not clear the persisted flag"
    finally:
        store.close()
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is True, (
        "after one complete observation the next staleness never got its own heal"
    )
    assert _wait_settled(str(store_path))
    assert len(calls) == 1


def test_observe_complete_while_running_clears_only_the_pending_note(tmp_path, monkeypatch):
    """A running attempt survives observe-complete; the pending note does not."""
    store_path = str(tmp_path / "running.db")
    associations_module.open_store(store_path).close()
    entered = threading.Event()
    release = threading.Event()

    def blocking_entry(store_path_arg, state_db_path=None, *, catalog=(), now_ms=None):
        entered.set()
        release.wait(5.0)
        return None

    monkeypatch.setattr(associations_module, "heal_store", blocking_entry)
    assert associations_module.request_heal(store_path) is True
    assert entered.wait(2.0), "the background job never started"
    with associations_module._HEAL_LOCK:
        associations_module._HEAL_FAILURES[store_path] = "pending"
    associations_module.note_store_complete(store_path)
    with associations_module._HEAL_LOCK:
        assert associations_module._HEAL_STATE.get(store_path) == "running", (
            "observe-complete cleared the running attempt"
        )
        assert store_path not in associations_module._HEAL_FAILURES, (
            "observe-complete kept the pending failure note while running"
        )
    release.set()
    assert _wait_settled(store_path)


def test_a_launch_failure_writes_no_flag_and_a_restart_may_retry(tmp_path, monkeypatch):
    """A thread that cannot start leaves a failure note, no flag, and retry is allowed."""
    store_path = str(tmp_path / "nothread.db")
    associations_module.open_store(store_path).close()

    class UnstartableThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        associations_module, "threading", type("T", (), {"Thread": UnstartableThread})()
    )
    assert associations_module.request_heal(store_path) is False
    assert (
        associations_module.heal_failure_note(store_path)
        == associations_module.HEAL_FAILURE_NOTE
    )
    store = associations_module.open_store(store_path, read_only=True)
    try:
        assert _flag(store) is None, "a launch failure wrote a finished flag"
    finally:
        store.close()
    associations_module.reset_heal_state()  # the restart
    monkeypatch.setattr(associations_module, "threading", sys.modules["threading"])
    monkeypatch.setattr(
        associations_module,
        "heal_store",
        lambda *args, **kwargs: None,
    )
    assert associations_module.request_heal(store_path) is True, (
        "a restart did not retry the launch after a launch failure"
    )
    assert _wait_settled(store_path)


def test_a_settle_that_cannot_open_a_writable_store_writes_no_flag(tmp_path):
    """A store path that cannot be opened writable settles with no flag and no suppression."""
    store_path = tmp_path / "directory.db"
    store_path.mkdir()
    with pytest.raises(associations_module.AssociationStoreError):
        associations_module.heal_store(str(store_path), str(tmp_path / "state.db"))
    assert not store_path.is_file(), "the refused directory was replaced by a file"


def test_a_settle_whose_flag_write_fails_leaves_meta_unchanged(hermes, tmp_path, monkeypatch):
    """Store opened, flag write raised: in-memory finished, meta unchanged, restart retries."""
    _populate(hermes)
    store_path = tmp_path / "writefail.db"

    def boom(store):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(associations_module, "_write_heal_finished", boom)
    report = _heal(store_path, hermes)  # an ok backfill still reports
    assert report.sessions_scanned == 3
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _flag(store) is None, "a failed flag write changed meta"
    finally:
        store.close()
    monkeypatch.undo()
    associations_module.reset_heal_state()  # the restart
    monkeypatch.setattr(
        associations_module,
        "heal_store",
        lambda *args, **kwargs: None,
    )
    assert associations_module.request_heal(str(store_path), str(hermes.path)) is True, (
        "a restart did not retry after a failed flag write"
    )
    assert _wait_settled(str(store_path))


def test_a_bounded_backfill_never_writes_the_flag(hermes, tmp_path):
    """Only an uncapped heal settles the flag; bounded and direct passes never do."""
    _populate(hermes)
    store_path = tmp_path / "bounded.db"
    store = associations_module.open_store(str(store_path))
    try:
        backfilled = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY, max_sessions=1
        )
        assert backfilled.sessions_scanned == 1
        associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert _flag(store) is None, "a direct backfill pass wrote the finished flag"
    finally:
        store.close()


def test_a_reset_clears_the_flag_and_records_the_loss_marker(hermes, tmp_path):
    """Named and full resets clear the flag and record last_history_loss_ms in-transaction."""
    _populate(hermes)
    store_path = tmp_path / "reset.db"
    _heal(store_path, hermes)
    store = associations_module.open_store(str(store_path))
    try:
        assert _flag(store) == "finished"
        associations_module.reset_associations(store, skills=["code-review"])
        assert _flag(store) is None, "a named reset kept the finished flag"
        marker = associations_module.read_meta(
            store, associations_module.META_LAST_HISTORY_LOSS_MS
        )
        assert marker and int(marker) > 0, "the named reset recorded no loss time"
        assert (
            associations_module.read_meta(store, associations_module.META_CREATION_ORIGIN)
            == "fresh"
        ), "a reset cleared the creation provenance"
        associations_module.reset_associations(store)
        assert associations_module.read_meta(
            store, associations_module.META_LAST_HISTORY_LOSS_MS
        ), "the full reset recorded no loss time"
    finally:
        store.close()


def test_maintenance_records_the_loss_marker_only_when_it_deletes_rows(tmp_path):
    """A purge that deletes rows records the loss; a pass with nothing to purge does not."""
    store_path = tmp_path / "maintenance.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.maintenance(store, now_ms=T0 + 20 * DAY, vacuum=False)
        assert (
            report.deleted_skills_removed == 0 and report.stale_prompts_removed == 0
        )
        assert (
            associations_module.read_meta(store, associations_module.META_LAST_HISTORY_LOSS_MS)
            is None
        ), "an empty maintenance pass recorded a loss"
        associations_module.upsert_prompts(
            store,
            [
                associations_module.PromptEventInput(
                    session_id="old", turn=1, text=SUBSTANTIVE, observed_ms=T0
                )
            ],
        )
        assert (
            associations_module.maintenance(store, now_ms=T0 + 400 * DAY, vacuum=False)
            .stale_prompts_removed
            == 1
        )
        assert associations_module.read_meta(
            store, associations_module.META_LAST_HISTORY_LOSS_MS
        ), "a deleting maintenance pass recorded no loss time"
    finally:
        store.close()


# --------------------------------------------------------------------------------------------------
# Migration: v1 to v2, crash recovery, lock, provenance, self-check
# --------------------------------------------------------------------------------------------------


def _v1_file(path: Path) -> None:
    """Write a steady v1 store file by hand, with one skill row of evidence."""
    conn = sqlite3.connect(str(path))
    conn.executescript(associations_module.SCHEMA_V1)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        (associations_module.META_HARNESS, "hermes"),
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        (associations_module.META_SCHEMA_VERSION, "1"),
    )
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        (associations_module.META_LAST_SCAN_MS, "12345"),
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def test_the_v1_to_v2_migration_seeds_the_marker_in_the_stamp_phase(tmp_path):
    """Migration bumps to 2, seeds the loss marker, and keeps the v1 evidence."""
    path = tmp_path / "v1.db"
    _v1_file(path)
    store = associations_module.open_store(str(path))
    try:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            associations_module.read_meta(store, associations_module.META_SCHEMA_VERSION)
            == "2"
        )
        marker = associations_module.read_meta(
            store, associations_module.META_LAST_HISTORY_LOSS_MS
        )
        assert marker and int(marker) > 0, "the migrated store carries no loss marker"
        assert (
            associations_module.read_meta(store, associations_module.META_LAST_SCAN_MS)
            == "12345"
        ), "the migration lost the v1 watermark"
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"scan_ledger", "seen_sessions"} <= tables, (
            "phase one did not create the v2 tables"
        )
    finally:
        store.close()


def test_a_fresh_v2_store_carries_no_marker_and_keeps_its_provenance(tmp_path):
    """Fresh creation stamps v2 with creation_origin fresh and no loss marker."""
    path = tmp_path / "fresh.db"
    store = associations_module.open_store(str(path))
    try:
        assert associations_module.SCHEMA_VERSION == 2
        assert (
            associations_module.read_meta(store, associations_module.META_CREATION_ORIGIN)
            == "fresh"
        )
        assert (
            associations_module.read_meta(store, associations_module.META_LAST_HISTORY_LOSS_MS)
            is None
        ), "a fresh v2 store carries a loss marker"
    finally:
        store.close()


def test_a_migration_kill_before_the_ddl_leaves_a_clean_v1_file(tmp_path, monkeypatch):
    """Fault in phase one: the v1 file is untouched and the retry migrates."""
    path = tmp_path / "kill-early.db"
    _v1_file(path)
    before = path.read_bytes()
    monkeypatch.setattr(
        associations_module,
        "_phase_one_v2_ddl",
        lambda conn: (_ for _ in ()).throw(
            associations_module.AssociationStoreError("disk full")
        ),
    )
    with pytest.raises(associations_module.AssociationStoreError, match="disk full"):
        associations_module.open_store(str(path))
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1, (
            "a pre-DDL kill stamped the file"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name IN"
                " ('scan_ledger', 'seen_sessions')"
            ).fetchone()[0]
            == 0
        ), "a pre-DDL kill created v2 tables"
    finally:
        conn.close()
    monkeypatch.undo()
    store = associations_module.open_store(str(path))
    try:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        store.close()


def test_a_migration_kill_between_phases_resumes_idempotently(tmp_path, monkeypatch):
    """Half-migrated (v2 tables present, user_version 1) resumes, never refuses."""
    path = tmp_path / "kill-late.db"
    _v1_file(path)

    def once(conn, harness, *, fresh, path):
        raise associations_module.AssociationStoreError("disk full")

    monkeypatch.setattr(associations_module, "_phase_two_stamp", once)
    with pytest.raises(associations_module.AssociationStoreError, match="disk full"):
        associations_module.open_store(str(path))
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1, (
            "the stamp committed before the fault"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name = 'scan_ledger'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM meta WHERE key = 'last_history_loss_ms'"
            ).fetchone()[0]
            == 0
        ), "the marker was seeded before the stamp"
    finally:
        conn.close()
    monkeypatch.undo()
    store = associations_module.open_store(str(path))
    try:
        marker = associations_module.read_meta(
            store, associations_module.META_LAST_HISTORY_LOSS_MS
        )
        assert marker and int(marker) > 0
    finally:
        store.close()
    # A second open must not reseed the marker.
    store = associations_module.open_store(str(path))
    try:
        assert (
            associations_module.read_meta(store, associations_module.META_LAST_HISTORY_LOSS_MS)
            == marker
        ), "the resumed store was reseeded"
    finally:
        store.close()


def test_a_disk_full_after_each_ddl_statement_leaves_one_of_three_states(
    tmp_path, monkeypatch
):
    """Fault after statement one of two: never mistaken for v2; retry completes."""
    path = tmp_path / "diskfull.db"
    _v1_file(path)

    def partial(conn):
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scan_ledger (session_id TEXT PRIMARY KEY,"
            " scanned_max_row_id INTEGER, pending_calls INTEGER NOT NULL,"
            " unreadable_results INTEGER NOT NULL, unreadable_calls INTEGER NOT NULL)"
        )
        raise associations_module.AssociationStoreError("disk full")

    monkeypatch.setattr(associations_module, "_phase_one_v2_ddl", partial)
    with pytest.raises(associations_module.AssociationStoreError, match="disk full"):
        associations_module.open_store(str(path))
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1, (
            "a partial-DDL file was mistaken for v2"
        )
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "scan_ledger" in names and "seen_sessions" not in names
    finally:
        conn.close()
    monkeypatch.undo()
    store = associations_module.open_store(str(path))
    try:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        store.close()


def test_two_overlapping_writable_opens_migrate_once(tmp_path, monkeypatch):
    """One winner migrates; the waiter rechecks and opens without reseeding."""
    path = tmp_path / "contended.db"
    _v1_file(path)
    gate = threading.Event()
    real_ddl = associations_module._phase_one_v2_ddl

    def slow_ddl(conn):
        real_ddl(conn)
        gate.wait(5.0)

    monkeypatch.setattr(associations_module, "_phase_one_v2_ddl", slow_ddl)
    errors: list = []
    winner_marker: list = []

    def open_writable():
        try:
            store = associations_module.open_store(str(path))
            winner_marker.append(
                associations_module.read_meta(
                    store, associations_module.META_LAST_HISTORY_LOSS_MS
                )
            )
            store.close()
        except Exception as exc:  # pragma: no cover - recorded for the assert below
            errors.append(exc)

    thread = threading.Thread(target=open_writable)
    thread.start()
    time.sleep(0.2)  # let the winner take the lock and block inside the DDL
    waiter = associations_module.open_store(str(path))  # waits, then rechecks
    try:
        waiter_marker = associations_module.read_meta(
            waiter, associations_module.META_LAST_HISTORY_LOSS_MS
        )
    finally:
        waiter.close()
        gate.set()
    thread.join(5.0)
    assert errors == [], f"the winner failed: {errors!r}"
    assert waiter_marker and waiter_marker == winner_marker[0], (
        "the waiter reseeded the marker or opened before the winner stamped"
    )


def test_a_contender_that_cannot_get_the_lock_is_refused_by_name(tmp_path, monkeypatch):
    """With the lock held and no timeout budget, a second opener fails locked."""
    path = tmp_path / "locked.db"
    _v1_file(path)
    monkeypatch.setattr(associations_module, "MIGRATION_LOCK_TIMEOUT_MS", 0)
    with open(associations_module._migration_lock_path(str(path)), "a+") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(associations_module.AssociationStoreError, match="locked"):
            associations_module.open_store(str(path))
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_two_first_opens_of_an_absent_file_serialize(tmp_path, monkeypatch):
    """Exactly one opener runs phase zero through stamp; the other opens v2."""
    path = tmp_path / "both-fresh.db"
    gate = threading.Event()
    real_fresh = associations_module._phase_one_fresh_ddl

    def slow_fresh(conn):
        real_fresh(conn)
        gate.wait(5.0)

    monkeypatch.setattr(associations_module, "_phase_one_fresh_ddl", slow_fresh)
    created: list = []

    def open_writable():
        store = associations_module.open_store(str(path))
        created.append(
            associations_module.read_meta(store, associations_module.META_CREATION_ORIGIN)
        )
        store.close()

    thread = threading.Thread(target=open_writable)
    thread.start()
    time.sleep(0.2)
    second = associations_module.open_store(str(path))
    try:
        assert (
            associations_module.read_meta(second, associations_module.META_CREATION_ORIGIN)
            == "fresh"
        )
        assert (
            associations_module.read_meta(second, associations_module.META_LAST_HISTORY_LOSS_MS)
            is None
        ), "the second first-open seeded a marker"
    finally:
        second.close()
        gate.set()
    thread.join(5.0)
    assert created == ["fresh"]


def test_a_zero_byte_file_migrates_as_fresh(tmp_path):
    """A zero-byte file is a fresh creation, not a corrupt refusal."""
    path = tmp_path / "zero.db"
    path.write_bytes(b"")
    store = associations_module.open_store(str(path))
    try:
        assert (
            associations_module.read_meta(store, associations_module.META_CREATION_ORIGIN)
            == "fresh"
        )
    finally:
        store.close()


def test_a_corrupt_file_refuses_by_name_and_is_not_replaced(tmp_path):
    """A non-database file refuses as corrupt and the heal does not replace it."""
    path = tmp_path / "garbage.db"
    path.write_text("this is not a database\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(associations_module.AssociationStoreError, match="corrupt"):
        associations_module.open_store(str(path))
    assert path.read_bytes() == before, "the corrupt file was modified"


def test_a_permission_denied_store_refuses_as_not_writable_not_corrupt_or_locked(tmp_path):
    """A store that cannot be written names not-writable, distinct from corrupt/locked."""
    path = tmp_path / "noperm.db"
    _v1_file(path)
    path.chmod(0o000)
    try:
        with pytest.raises(associations_module.AssociationStoreError) as caught:
            associations_module.open_store(str(path))
        message = str(caught.value)
        assert "corrupt" not in message and "locked" not in message, message
        if "not writable" not in message:
            pytest.skip(f"platform opened a 000 file anyway: {message}")
    finally:
        path.chmod(0o644)


def test_a_fresh_crash_after_phase_zero_resumes_on_retry(tmp_path, monkeypatch):
    """Phase zero persisted provenance; the next writable open completes creation."""
    path = tmp_path / "fresh-crash.db"
    monkeypatch.setattr(
        associations_module,
        "_phase_one_fresh_ddl",
        lambda conn: (_ for _ in ()).throw(
            associations_module.AssociationStoreError("disk full")
        ),
    )
    with pytest.raises(associations_module.AssociationStoreError, match="disk full"):
        associations_module.open_store(str(path))
    conn = sqlite3.connect(str(path))
    try:
        assert (
            conn.execute(
                "SELECT value FROM meta WHERE key = 'creation_origin'"
            ).fetchone()[0]
            == "fresh"
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name = 'skill'"
            ).fetchone()[0]
            == 0
        ), "the crash left more than the provenance key"
    finally:
        conn.close()
    monkeypatch.undo()
    store = associations_module.open_store(str(path))
    try:
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            associations_module.read_meta(store, associations_module.META_CREATION_ORIGIN)
            == "fresh"
        ), "the resumed store lost its provenance"
    finally:
        store.close()


def test_a_partial_file_without_provenance_refuses_unmodified(tmp_path):
    """A hand-malformed partial file (tables, no stamp, no provenance) is refused as-is."""
    path = tmp_path / "malformed.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(associations_module.SCHEMA_V1)
    conn.commit()
    conn.close()
    before = path.read_bytes()
    with pytest.raises(associations_module.AssociationStoreError, match="malformed"):
        associations_module.open_store(str(path))
    assert path.read_bytes() == before, "the refused partial file was modified"


@pytest.mark.parametrize("read_only", [True, False])
def test_a_v2_store_missing_its_ledger_or_seen_table_refuses_self_check(
    tmp_path, read_only
):
    """A v2 store missing scan_ledger or seen_sessions (or a column) fails self-check."""
    path = tmp_path / "noself.db"
    store = associations_module.open_store(str(path))
    store.close()
    conn = sqlite3.connect(str(path))
    conn.execute("DROP TABLE scan_ledger")
    conn.commit()
    conn.close()
    with pytest.raises(associations_module.AssociationStoreError, match="self-check"):
        associations_module.open_store(str(path), read_only=read_only)


def test_a_read_only_open_that_needs_migration_refuses_by_name(tmp_path):
    """Read-only opens never migrate; the heal thread's writable open is the only path."""
    path = tmp_path / "readonly-v1.db"
    _v1_file(path)
    before = path.read_bytes()
    with pytest.raises(associations_module.AssociationStoreError, match="migration"):
        associations_module.open_store(str(path), read_only=True)
    assert path.read_bytes() == before, "the read-only refusal modified the file"
    # Exactly one heal is requested per turn-path refusal.
    calls = []
    associations_module.reset_heal_state()
    import hermes_skill_router.associations as live

    original = live.heal_store
    live.heal_store = lambda *args, **kwargs: calls.append(1)
    try:
        assert live.request_heal(str(path)) is True
        assert live.request_heal(str(path)) is False
        assert _wait_settled(str(path))
    finally:
        live.heal_store = original
    assert len(calls) == 1, "the turn path launched more than one heal"


# --------------------------------------------------------------------------------------------------
# seen_sessions and scan_ledger finality
# --------------------------------------------------------------------------------------------------


def test_every_listing_upserts_seen_sessions_with_started_at(hermes, tmp_path):
    """A backfill records each listed session's start, even unprovenanced ones."""
    hermes.session("s-a", started_ms=T0)
    hermes.user("s-a", SUBSTANTIVE, T0 + 100)
    hermes.skill_view("s-a", "code-review", T0 + 200)
    hermes.session("s-b", source="subagent", started_ms=T0 + 5000)
    hermes.commit()
    store_path = tmp_path / "seen.db"
    _heal(store_path, hermes)
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        seen = dict(rows(store, "SELECT session_id, started_at_ms FROM seen_sessions"))
        assert seen == {"s-a": T0, "s-b": T0 + 5000}, (
            "the listing did not upsert every session's start"
        )
    finally:
        store.close()


def test_a_scanned_then_deleted_session_keeps_its_ledger_row(hermes, tmp_path):
    """Deleting a scanned session from the source leaves its ledger row readable."""
    _populate(hermes, count=1)
    store_path = tmp_path / "deleted-after.db"
    _heal(store_path, hermes)
    hermes.conn.execute("DELETE FROM sessions WHERE id = 's-000'")
    hermes.conn.commit()
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        found = _ledger(store, "s-000")
        assert len(found) == 1 and found[0][1] is not None and found[0][2:] == (0, 0, 0), (
            "the scanned-then-deleted session lost its finality row"
        )
    finally:
        store.close()


def test_a_message_less_session_commits_an_empty_scan_row(hermes, tmp_path):
    """Message-less while current: zero counters and a NULL scanned id."""
    hermes.session("s-empty", started_ms=T0)
    hermes.commit()
    store_path = tmp_path / "empty.db"
    _heal(store_path, hermes)
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _ledger(store, "s-empty") == [("s-empty", None, 0, 0, 0)], (
            "the empty scan did not commit its NULL-id revision"
        )
    finally:
        store.close()


def test_an_empty_scan_still_passes_when_the_session_is_later_deleted(hermes, tmp_path):
    """Empty row, then source deletion with no messages: the row still passes."""
    hermes.session("s-empty", started_ms=T0)
    hermes.commit()
    store_path = tmp_path / "empty-deleted.db"
    _heal(store_path, hermes)
    empty_before = _ledger(
        associations_module.open_store(str(store_path), read_only=True), "s-empty"
    )
    hermes.conn.execute("DELETE FROM sessions WHERE id = 's-empty'")
    hermes.conn.commit()
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        assert _ledger(store, "s-empty") == empty_before, (
            "the empty-scan row changed after the session left the source"
        )
    finally:
        store.close()


def test_an_empty_scan_is_invalidated_by_its_first_message_whatever_its_id(
    hermes, tmp_path
):
    """First message after the empty scan (id 0 below) fails the empty revision."""
    hermes.session("s-empty", started_ms=T0)
    hermes.commit()
    store_path = tmp_path / "empty-gains.db"
    _heal(store_path, hermes)
    hermes.conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp)"
        " VALUES (0, 's-empty', 'user', ?, ?)",
        (SUBSTANTIVE, (T0 + 1000) / 1000.0),
    )
    hermes.conn.commit()
    store = associations_module.open_store(str(store_path), read_only=True)
    try:
        empty = _ledger(store, "s-empty")[0]
        source_ids = rows(
            store,
            "SELECT scanned_max_row_id FROM scan_ledger WHERE session_id = 's-empty'",
        )
        source_max = hermes.conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id = 's-empty'"
        ).fetchone()[0]
        assert empty[1] is None, "the empty revision is no longer empty-marked"
        assert source_max == 0, (
            "the named coverage check must fail on a message after an empty scan"
        )
    finally:
        store.close()


def test_a_session_write_error_rolls_back_prompts_loads_and_ledger(hermes, tmp_path, monkeypatch):
    """Failure after prompt insertion, before loads: nothing of the session lands."""
    hermes.session("s-one", started_ms=T0)
    hermes.user("s-one", SUBSTANTIVE, T0 + 100)
    hermes.skill_view("s-one", "code-review", T0 + 200)
    hermes.session("s-two", started_ms=T0 + 5000)
    hermes.user("s-two", SUBSTANTIVE, T0 + 5100)
    hermes.skill_view("s-two", "writing", T0 + 5200)
    hermes.commit()
    real_insert = associations_module._insert_events

    def flaky(store, events, result):
        if any(event.session_id == "s-two" for event in events):
            raise sqlite3.OperationalError("disk full")
        return real_insert(store, events, result)

    monkeypatch.setattr(associations_module, "_insert_events", flaky)
    store_path = tmp_path / "rollback.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.skip_reasons["write-failed"] == 1
        assert rows(store, "SELECT count(*) FROM prompt_event WHERE session_id = 's-two'") == [
            (0,)
        ], "the failed session's prompts committed"
        assert _ledger(store, "s-two") == [], (
            "the failed session left a current finality ledger"
        )
        assert len(_ledger(store, "s-one")) == 1
    finally:
        store.close()
    monkeypatch.undo()
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.loads == 1, "the retry did not commit the failed session's loads"
        assert len(_ledger(store, "s-two")) == 1, "the retry did not commit the ledger"
    finally:
        store.close()


def test_an_in_flight_call_leaves_pending_calls_positive(hermes, tmp_path):
    """A call with no result at scan end records pending_calls > 0."""
    hermes.session("s-flight", started_ms=T0)
    hermes.user("s-flight", SUBSTANTIVE, T0 + 100)
    hermes.calls("s-flight", [("skill_view", {"name": "code-review"})], T0 + 200)
    hermes.commit()
    store_path = tmp_path / "flight.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.sessions_scanned == 1
        pending = _ledger(store, "s-flight")[0][2]
        assert pending > 0, "an in-flight call left pending_calls at zero"
    finally:
        store.close()


def test_parser_counts_unreadable_calls_and_orphan_results(hermes, tmp_path):
    """Truncated call + its valid skill result: unreadable_calls and unreadable_results > 0."""
    hermes.session("s-trunc", started_ms=T0)
    hermes.user("s-trunc", SUBSTANTIVE, T0 + 100)
    hermes.raw_messages("s-trunc", '[{"id": "c9", "function": {"name": "skill_view",'
                                    ' "arguments": "{bad json', T0 + 200)
    hermes.result(
        "s-trunc",
        "c9",
        "skill_view",
        json.dumps({"success": True, "name": "code-review"}),
        T0 + 300,
    )
    hermes.commit()
    store_path = tmp_path / "trunc.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.unreadable_calls > 0, "the truncated call was dropped silently"
        assert report.unreadable_results > 0, "the orphan result was dropped silently"
        row = _ledger(store, "s-trunc")[0]
        assert row[4] > 0 and row[3] > 0, "the ledger recorded no unreadable counts"
        assert row[2] == 0, "a truncated call left pending_calls nonzero"
    finally:
        store.close()


def test_a_truncated_payload_with_no_result_counts_only_the_call(hermes, tmp_path):
    """Truncated tool_calls with no following result: unreadable_calls > 0, results 0."""
    hermes.session("s-trunc-only", started_ms=T0)
    hermes.user("s-trunc-only", SUBSTANTIVE, T0 + 100)
    hermes.raw_messages("s-trunc-only", "{not json at all", T0 + 200)
    hermes.commit()
    store_path = tmp_path / "trunc-only.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.unreadable_calls > 0
        assert report.unreadable_results == 0, (
            "no result row follows a truncated payload, so none is unreadable"
        )
        row = _ledger(store, "s-trunc-only")[0]
        assert (row[2], row[3], row[4]) == (0, 0, report.unreadable_calls)
    finally:
        store.close()


def test_a_name_only_skill_view_and_an_unrelated_match_record_no_unreadable(
    hermes, tmp_path
):
    """Valid name-only skill_view loads; a matched non-skill result counts nothing."""
    hermes.session("s-nameonly", started_ms=T0)
    hermes.user("s-nameonly", SUBSTANTIVE, T0 + 100)
    hermes.skill_view("s-nameonly", "code-review", T0 + 200)  # name only, no file
    hermes.calls("s-nameonly", [("write_file", {"path": "/tmp/x"})], T0 + 400)
    hermes.result("s-nameonly", "call-2", "write_file", "ok", T0 + 500)
    hermes.commit()
    store_path = tmp_path / "nameonly.db"
    store = associations_module.open_store(str(store_path))
    try:
        report = associations_module.backfill(
            store, str(hermes.path), catalog=CATALOG, now_ms=T0 + 10 * DAY
        )
        assert report.loads == 1, "the name-only skill_view recorded no load"
        assert report.unreadable_calls == 0, "a classifiable call counted unreadable"
        assert report.unreadable_results == 0, "a matched unrelated result counted unreadable"
        row = _ledger(store, "s-nameonly")[0]
        assert row[2:] == (0, 0, 0), "the finality row is not clean"
    finally:
        store.close()
