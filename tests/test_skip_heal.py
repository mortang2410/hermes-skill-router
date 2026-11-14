"""Contract C4: an incomplete association store skips its channel, heals once, and routes.

Ticket 03 measured what the four bad store inputs do to the current code. This file pins the
settled behaviour for the same four inputs, and every named case proves the same three things in
one turn:

- the middleware does not raise, and the request mapping it received is byte-identical after the
  call (the rewrite replaces the payload rather than editing it in place);
- the semantic and BM25 channels still return ranks, and no candidate carries an association rank,
  so the turn routes from those two channels;
- the turn asks the C6 background job entry point for exactly one heal of the configured store, and
  a second pass over the same incomplete store does not launch a second job.

The C6 entry point is replaced by a recorder, so the test observes the launch without walking the
live Hermes session store. ``tests/conftest.py`` already stubs the entry point for every test that
does not prove the launch.

Completeness itself is checked at the boundary by
``test_completeness_is_the_newest_session_stamp_behind_the_scan_watermark``: newest session stamp
minus ``META_LAST_SCAN_MS`` at most 86400000 ms, no watermark meaning incomplete.
"""

from __future__ import annotations

import copy
import functools
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from tests.test_crash_probe import (  # noqa: E402
    _assert_routed_without_crashing,
    _canonical,
    _query_payload,
    _shortlist_probe,
)
from tests.test_middleware_contract import (  # noqa: E402
    DISPATCH_KWARGS,
    _callback_for,
    _load_plugin_module,
    endpoints,
    settings_for,
)

#: The top-level router copy ``_shortlist_probe`` drives.
TOP_LEVEL_ASSOCIATIONS = "hermes_skill_router.associations"


# --------------------------------------------------------------------------------------------------
# The launch recorder
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HealCall:
    """One recorded C6 entry-point call: the router copy, the store, and the catalog."""

    module: str
    store_path: str
    catalog: Tuple[str, ...]


class HealSpy:
    """Record every C6 entry-point call and let a test wait for it without polling the clock."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._calls: List[HealCall] = []

    def record(
        self,
        module_name: str,
        store_path: str,
        state_db_path: Optional[str] = None,
        *,
        catalog: Sequence[associations_module.CatalogSkill] = (),
        now_ms: Optional[int] = None,
    ) -> None:
        """Stand in for ``associations.heal_store``: record the launch and start no work."""
        call = HealCall(module_name, store_path, tuple(skill.name for skill in catalog))
        with self._condition:
            self._calls.append(call)
            self._condition.notify_all()

    @property
    def calls(self) -> List[HealCall]:
        with self._condition:
            return list(self._calls)

    def count(self, module_name: str) -> int:
        """Return how many launches *module_name* requested."""
        with self._condition:
            return sum(1 for call in self._calls if call.module == module_name)

    def wait_for(self, module_name: str, count: int = 1, timeout: float = 5.0) -> bool:
        """Wait until *module_name* requested *count* launches, or the timeout expires."""
        with self._condition:
            return self._condition.wait_for(
                lambda: sum(1 for call in self._calls if call.module == module_name) >= count,
                timeout,
            )


@pytest.fixture(scope="module")
def plugin_module():
    """A private plugin copy, so this file's recorder patches never reach another test module."""
    return _load_plugin_module("hermes_skill_router_skip_heal_under_test")


@pytest.fixture(autouse=True)
def _host_seams(monkeypatch):
    """The host lookup resolves every name, and no dump record leaks between turns."""
    search_module.clear_dumps()
    resolver = lambda name: f"/skills/{name}/SKILL.md"
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("router") and hasattr(module, "skill_path"):
            monkeypatch.setattr(module, "skill_path", resolver)
            monkeypatch.setattr(module, "_broadcast", lambda turn_id: None)
        elif module_name.endswith("search") and hasattr(module, "clear_dumps"):
            module.clear_dumps()


@pytest.fixture(autouse=True)
def heal_spy(no_live_heal, plugin_module, monkeypatch):
    """Replace the C6 entry point in every copy with one recorder, and forget old attempts."""
    spy = HealSpy()
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith("associations"):
            continue
        if hasattr(module, "heal_store"):
            monkeypatch.setattr(module, "heal_store", functools.partial(spy.record, module_name))
        if hasattr(module, "reset_heal_state"):
            module.reset_heal_state()
    return spy


# --------------------------------------------------------------------------------------------------
# One turn
# --------------------------------------------------------------------------------------------------


def _settings(store_path, tmp_path, endpoints):
    """Settings that point both endpoints at their fakes and the channel at *store_path*."""
    settings = dict(settings_for(endpoints, tmp_path / "suggestions.json"))
    settings["associationDbPath"] = str(store_path)
    return settings


def _middleware_turn(callback, payload):
    """Drive one real middleware turn and keep the bytes the caller can compare afterwards."""
    snapshot = copy.deepcopy(payload)
    before = _canonical(payload)
    raised = None
    result = None
    try:
        result = callback(request=payload, **DISPATCH_KWARGS)
    except Exception as exc:
        raised = exc
    return SimpleNamespace(
        middleware_raised=raised,
        result=result,
        payload=payload,
        snapshot=snapshot,
        before=before,
    )


def _plugin_associations(plugin_module) -> str:
    """Return the module name the plugin copy's library reads the store through."""
    return f"{plugin_module.__name__}.hermes_skill_router.associations"


def _assert_incomplete_turn(store_path, tmp_path, plugin_module, endpoints, heal_spy):
    """Assert one turn over an incomplete store: channels route, bytes hold, one heal launches.

    The channel probe and the middleware each drive their own loaded copy of the router, and each
    copy is one process' view of the store, so the launch is asserted per copy: exactly one, and
    a second pass over the same incomplete store adds none.
    """
    settings = _settings(store_path, tmp_path, endpoints)
    cfg = load(settings)
    # The middleware reads the same mapping through read_settings, so this keeps a skipped store
    # from passing the test because it was never configured at all.
    assert cfg.association_db_path == str(store_path)

    shortlist_raised, rows, records = _shortlist_probe(cfg)
    callback = _callback_for(plugin_module, settings)
    turn = _middleware_turn(callback, _query_payload())
    probe = SimpleNamespace(
        middleware_raised=turn.middleware_raised,
        result=turn.result,
        payload=turn.payload,
        snapshot=turn.snapshot,
        before=turn.before,
        shortlist_raised=shortlist_raised,
        rows=rows,
        records=records,
    )
    _assert_routed_without_crashing(probe)

    plugin_associations = _plugin_associations(plugin_module)
    assert heal_spy.wait_for(TOP_LEVEL_ASSOCIATIONS), (
        "the library pass never asked for the background heal"
    )
    assert heal_spy.wait_for(plugin_associations), (
        "the middleware turn never asked for the background heal"
    )
    assert heal_spy.count(TOP_LEVEL_ASSOCIATIONS) == 1, "the library pass launched a second heal"
    assert heal_spy.count(plugin_associations) == 1, "the middleware turn launched a second heal"
    assert {call.store_path for call in heal_spy.calls} == {str(store_path)}, (
        "the launch named a store other than the configured one"
    )

    # The condition has not changed: another pass over the same store still skips it and must not
    # start a second job. This is the loop C4 forbids.
    shortlist_raised, rows, records = _shortlist_probe(cfg)
    assert shortlist_raised is None, (
        f"the library open path raised on the second pass: {shortlist_raised!r}"
    )
    assert rows, "the second pass shortlisted no row at all"
    again = _middleware_turn(callback, turn.payload)
    assert again.middleware_raised is None, (
        f"the middleware raised on the second pass: {again.middleware_raised!r}"
    )
    assert again.payload == again.snapshot, "the second turn mutated the request mapping in place"
    assert _canonical(again.payload) == again.before, (
        "the second request mapping is not byte-identical after the middleware call"
    )
    assert heal_spy.count(TOP_LEVEL_ASSOCIATIONS) == 1, "the second pass launched a second heal"
    assert heal_spy.count(plugin_associations) == 1, "the second turn launched a second heal"


# --------------------------------------------------------------------------------------------------
# The four named store inputs
# --------------------------------------------------------------------------------------------------


def test_incomplete_store_missing_file_skips_association_and_heals_once(
    tmp_path, plugin_module, endpoints, heal_spy
):
    """Missing file: the turn routes from semantic and BM25 and asks for one heal."""
    store_path = tmp_path / "associations.db"
    assert not store_path.exists()
    _assert_incomplete_turn(store_path, tmp_path, plugin_module, endpoints, heal_spy)


def test_incomplete_store_zero_byte_skips_association_and_heals_once(
    tmp_path, plugin_module, endpoints, heal_spy
):
    """Zero-byte file: the turn routes from semantic and BM25 and asks for one heal."""
    store_path = tmp_path / "associations.db"
    store_path.write_bytes(b"")
    _assert_incomplete_turn(store_path, tmp_path, plugin_module, endpoints, heal_spy)


def test_incomplete_store_tables_but_no_rows_skips_association_and_heals_once(
    tmp_path, plugin_module, endpoints, heal_spy
):
    """A valid empty store: it opens, but contributes nothing, and the turn asks for one heal."""
    store_path = tmp_path / "associations.db"
    store = associations_module.open_store(str(store_path))
    store.close()
    _assert_incomplete_turn(store_path, tmp_path, plugin_module, endpoints, heal_spy)


def test_incomplete_store_directory_in_place_skips_association_and_heals_once(
    tmp_path, plugin_module, endpoints, heal_spy
):
    """A directory in the store's place: the turn routes from semantic and BM25, one heal asked."""
    store_path = tmp_path / "associations.db"
    store_path.mkdir()
    _assert_incomplete_turn(store_path, tmp_path, plugin_module, endpoints, heal_spy)


# --------------------------------------------------------------------------------------------------
# The completeness rule and the one-attempt bookkeeping
# --------------------------------------------------------------------------------------------------


def _session_store(path: Path, newest_ms: int) -> None:
    """Write the two-column Hermes session fixture whose newest stamp is *newest_ms*."""
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


def test_completeness_is_the_newest_session_stamp_behind_the_scan_watermark(tmp_path):
    """At most 86400000 ms behind the newest session stamp is complete; one ms more is not."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    store = associations_module.open_store(str(tmp_path / "complete.db"))
    try:
        assert associations_module.newest_session_stamp_ms(str(state_path)) == newest
        assert associations_module.is_store_complete(store, str(state_path)) is False, (
            "a store with no scan watermark must report incomplete"
        )
        associations_module.write_meta(
            store,
            associations_module.META_LAST_SCAN_MS,
            str(newest - associations_module.COMPLETENESS_WINDOW_MS),
        )
        assert associations_module.is_store_complete(store, str(state_path)) is True
        associations_module.write_meta(
            store,
            associations_module.META_LAST_SCAN_MS,
            str(newest - associations_module.COMPLETENESS_WINDOW_MS - 1),
        )
        assert associations_module.is_store_complete(store, str(state_path)) is False
        missing = str(tmp_path / "missing.db")
        assert associations_module.is_store_complete(store, missing) is False, (
            "an unreadable session store is no evidence of coverage"
        )
    finally:
        store.close()


def _wait_attempt_finished(store_path: str, timeout: float = 5.0) -> bool:
    """Wait until the recorded heal thread settled this store's attempt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
        if state != associations_module._HEAL_RUNNING:
            return True
        time.sleep(0.01)
    return False


def test_the_heal_launch_is_once_per_incomplete_condition(tmp_path, heal_spy):
    """One job per incomplete condition; one that finished without progress never relaunches."""
    store_path = tmp_path / "associations.db"
    associations_module.open_store(str(store_path)).close()
    assert associations_module.request_heal(str(store_path)) is True
    assert heal_spy.wait_for(TOP_LEVEL_ASSOCIATIONS), "the first launch never reached the recorder"
    assert heal_spy.count(TOP_LEVEL_ASSOCIATIONS) == 1
    assert associations_module.request_heal(str(store_path)) is False, (
        "an unchanged incomplete condition launched a second job"
    )
    assert _wait_attempt_finished(str(store_path)), "the recorded heal thread never settled"
    assert heal_spy.count(TOP_LEVEL_ASSOCIATIONS) == 1, "a finished attempt relaunched itself"
    associations_module.note_store_complete(str(store_path))
    assert associations_module.request_heal(str(store_path)) is True, (
        "a freshly observed staleness never got its own job"
    )
    assert heal_spy.wait_for(TOP_LEVEL_ASSOCIATIONS, 2)
    assert heal_spy.count(TOP_LEVEL_ASSOCIATIONS) == 2
