"""Contract C5: one display note while the association store is incomplete.

The note is out of band: it rides on the display record and never on the provider request, so the
turn's bytes stay identical. These tests drive the real rewrite over the default fusion path and
pin both acceptance halves, an incomplete store puts one line on the record and a complete store
puts none.

Completeness is contract C6's rule (newest session stamp minus ``META_LAST_SCAN_MS`` at most
86400000 ms, ``hermes_skill_router.associations.is_store_complete``). The live session store is
not the fixture, so a test that needs a complete store points ``default_state_db_path`` at its own
temp database the way ticket 04's boundary test passes its own state path.
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
from hermes_skill_router.display import ROUTED, read  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_display import PAYLOAD, make_config  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process logger sets, a fresh index memo, dumps, and the host lookup answered."""
    monkeypatch.setattr(display_module, "_LOGGED", set())
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    search_module.clear_dumps()


def _turn(cfg):
    """Drive one real rewrite turn, equivalent to the one the middleware runs."""
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
    )


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


def _complete_store(store_path: Path, newest_ms: int) -> None:
    """Create a store whose scan watermark covers the newest fixture session."""
    store = associations_module.open_store(str(store_path))
    try:
        associations_module.write_meta(
            store, associations_module.META_LAST_SCAN_MS, str(newest_ms)
        )
    finally:
        store.close()


def test_an_incomplete_store_puts_one_note_on_the_turn_record(tmp_path):
    """A missing store is incomplete, so one line rides on the record and on no request byte."""
    store_path = tmp_path / "associations.db"
    assert not store_path.exists()
    cfg = make_config(tmp_path, associationDbPath=str(store_path))

    result = _turn(cfg)

    assert result is not None
    stored = read(cfg)
    assert stored["status"] == ROUTED
    assert stored["note"] == router_module.HISTORY_NOTE
    assert router_module.HISTORY_NOTE not in json.dumps(result["request"]), (
        "the note must not reach the provider request"
    )


def test_a_valid_but_empty_store_also_carries_the_note(tmp_path):
    """A store that opens but has no scan watermark is incomplete: no more silent empty store."""
    store_path = tmp_path / "associations.db"
    associations_module.open_store(str(store_path)).close()
    cfg = make_config(tmp_path, associationDbPath=str(store_path))

    assert _turn(cfg) is not None

    assert read(cfg)["note"] == router_module.HISTORY_NOTE


def test_a_complete_store_writes_no_note(tmp_path, monkeypatch):
    """The acceptance half that matters for silence: a covered store adds no line."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda home=None: str(state_path)
    )
    store_path = tmp_path / "associations.db"
    _complete_store(store_path, newest)
    cfg = make_config(tmp_path, associationDbPath=str(store_path))

    result = _turn(cfg)

    assert result is not None
    assert read(cfg)["note"] == "", "a complete store must show nothing"


def test_a_healed_store_turns_the_note_off_on_the_next_turn(tmp_path, monkeypatch):
    """Both halves in one sequence: the line follows the store's completeness, not a sticky flag."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda home=None: str(state_path)
    )
    store_path = tmp_path / "associations.db"
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    assert not store_path.exists()

    _turn(cfg)
    assert read(cfg)["note"] == router_module.HISTORY_NOTE

    _complete_store(store_path, newest)
    _turn(cfg)
    assert read(cfg)["note"] == ""


def test_the_cosine_rollback_writes_no_note(tmp_path):
    """Cosine mode returns before the store opens, so no heal exists and no line is true."""
    cfg = make_config(
        tmp_path,
        associationDbPath=str(tmp_path / "associations.db"),
        fusionMode="cosine",
    )

    result = _turn(cfg)

    assert result is not None
    assert read(cfg)["note"] == "", "the cosine rollback never opens the store and starts no heal"


# ---------------------------------------------------------------------------
# Contract C6's failure note: the turn-side chooser contract C5 owns.
# ---------------------------------------------------------------------------


def _exploding_heal(store_path, state_db_path=None, *, catalog=(), now_ms=None):
    """Stand in for a C6 pass that failed before it could finish."""
    raise sqlite3.OperationalError("database is locked")


def _wait_settled(store_path: str, timeout: float = 5.0) -> bool:
    """Wait until the recorded heal thread settled this store's attempt."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
        if state != associations_module._HEAL_RUNNING:
            return True
        time.sleep(0.01)
    return False


def _pending_failure(store_path: str) -> bool:
    """Return whether a failure note waits for the turn, without consuming it."""
    with associations_module._HEAL_LOCK:
        return store_path in associations_module._HEAL_FAILURES


def test_a_failed_heal_replaces_the_healing_line_once(tmp_path, monkeypatch):
    """C6 hands C5 one note: the turn shows it once, then resumes the incomplete-store line."""
    store_path = tmp_path / "associations.db"
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    monkeypatch.setattr(associations_module, "heal_store", _exploding_heal)
    assert associations_module.request_heal(str(store_path)) is True
    assert _wait_settled(str(store_path)), "the failed heal never settled its attempt"
    assert _pending_failure(str(store_path)), "the failed heal left no note for the turn"

    assert _turn(cfg) is not None
    assert read(cfg)["note"] == associations_module.HEAL_FAILURE_NOTE

    assert _turn(cfg) is not None
    assert read(cfg)["note"] == router_module.HISTORY_NOTE, (
        "one failure must reach the user once, not on every turn"
    )


def test_a_failed_turn_leaves_the_failure_note_pending(tmp_path, monkeypatch):
    """A turn that shows no history note must not consume it: C5 reads it only when it wins."""
    store_path = tmp_path / "associations.db"
    associations_module.open_store(str(store_path)).close()
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    monkeypatch.setattr(associations_module, "heal_store", _exploding_heal)
    assert associations_module.request_heal(str(store_path)) is True
    assert _wait_settled(str(store_path)), "the failed heal never settled its attempt"
    assert _pending_failure(str(store_path)), "the failed heal left no note for the turn"

    failed = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _text in texts],
        rerank=lambda config, query, documents: None,
        session_id="failure-window",
        notify=lambda _turn_id: None,
    )
    assert failed is None, "a failed rerank leaves the request unchanged"
    assert read(cfg)["note"] == "", "the failed turn showed no history note"
    assert _pending_failure(str(store_path)), "a turn that showed no note consumed it"

    assert _turn(cfg) is not None
    assert read(cfg)["note"] == associations_module.HEAL_FAILURE_NOTE
