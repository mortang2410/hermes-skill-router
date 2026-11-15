"""Ticket 01 (kata t2c1): degraded-line producers on the note chooser.

Contract C1 (M1, model NoteChooser v2.5.0): the stored note carries the
dropped-catalog and saved-index states, a cold build announces itself while
it runs, and the history wording matches the heal attempt state. Tier order
is failure > budget > degraded; the heal note is consumed at most once, by
the degraded tier, on the first eligible successful incomplete-store turn.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint, Response  # noqa: E402

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import display as display_module  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router import tools as tools_module  # noqa: E402
from hermes_skill_router.display import INERT, ROUTED, read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_display import PAYLOAD, make_config  # noqa: E402
from tests.test_rewrite import FakeEmbedder, FakeReranker, make_index, _payload  # noqa: E402

INDEX = Index(
    names=["alpha", "beta"],
    descriptions=["Alpha skill", "Beta skill"],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    dim=2,
    fingerprint="followup-test",
)


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process state, so one test never masks another."""
    monkeypatch.setattr(display_module, "_LOGGED", set())
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    search_module.clear_dumps()
    associations_module.reset_heal_state()
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()
    yield
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()


def _turn(cfg, **kwargs):
    kwargs.setdefault("session_id", "session-1")
    kwargs.setdefault("turn_id", "turn-1")
    kwargs.setdefault("notify", lambda _turn_id: None)
    kwargs.setdefault(
        "embed",
        lambda config, texts: [
            [1.0, 0.0] if "Alpha" in text else [0.0, 1.0] for text in texts
        ],
    )
    kwargs.setdefault("rerank", lambda config, query, documents: [0.9, 0.2][: len(documents)])
    return rewrite(PAYLOAD, cfg=cfg, **kwargs)


def _exploding(*args, **kwargs):
    raise AssertionError("no endpoint call may happen before the dumped-window check")


def _pending_failure(store_path: str) -> bool:
    with associations_module._HEAL_LOCK:
        return store_path in associations_module._HEAL_FAILURES


def _seed_heal_state(store_path: str, state: str) -> None:
    with associations_module._HEAL_LOCK:
        associations_module._HEAL_STATE[store_path] = state


def _seed_pending_failure(store_path: str) -> None:
    with associations_module._HEAL_LOCK:
        associations_module._HEAL_FAILURES[store_path] = associations_module.HEAL_FAILURE_NOTE


# ---------------------------------------------------------------------------
# Dropped-window producer (failure tier).
# ---------------------------------------------------------------------------


def test_a_second_failure_in_one_window_writes_the_dropped_window_note(tmp_path):
    """A forced second failure in one compaction window names the loss and the fix."""
    cfg = make_config(tmp_path)
    session = "dropped-healthy"
    first = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: None,
        session_id=session,
        notify=lambda _t: None,
    )
    assert first is None, "the first failure keeps the catalog"
    assert search_module.dumped(session)

    rerank_calls = []

    def rerank(config, query, documents):
        rerank_calls.append((query, list(documents)))
        return [0.9, 0.2][: len(documents)]

    second = _turn(cfg, session_id=session, index=INDEX, rerank=rerank)
    assert second is not None and "messages" in second["request"]
    note = read(cfg)["note"]
    assert "already" in note.lower() and "skills_list" in note, (
        f"the dropped-window note must name the loss and the fix, got {note!r}"
    )
    assert note != router_module.ENDPOINT_NOTE
    assert rerank_calls == [], "the dumped-window check must run before rank"


def test_a_supplied_index_with_a_dead_endpoint_still_writes_the_dropped_note(tmp_path):
    """No endpoint call happens before the dumped-window check, so death coexists."""
    cfg = make_config(tmp_path)
    search_module.remember_dump("supplied-session")
    result = _turn(
        cfg,
        session_id="supplied-session",
        index=INDEX,
        embed=_exploding,
        rerank=_exploding,
    )
    assert result is not None
    note = read(cfg)["note"]
    assert "already" in note.lower() and "skills_list" in note, (
        f"a dead endpoint must not hide the dropped-window note, got {note!r}"
    )


def test_a_warm_memo_with_a_dead_endpoint_still_writes_the_dropped_note(tmp_path):
    """The library returns the memo without an endpoint call, so the check still runs."""
    cfg = make_config(tmp_path)
    assert _turn(cfg, session_id="memo-session") is not None
    assert library_module._MEMO, "the first turn must have filled the memo"
    search_module.remember_dump("memo-session")
    result = _turn(
        cfg,
        session_id="memo-session",
        embed=_exploding,
        rerank=_exploding,
    )
    assert result is not None
    note = read(cfg)["note"]
    assert "already" in note.lower() and "skills_list" in note, (
        f"a warm memo must not need the endpoint for the dropped note, got {note!r}"
    )


def test_a_cold_endpoint_build_failure_writes_no_dropped_note(tmp_path):
    """A build failure before the dumped-window check pre-empts it: endpoint note only."""
    embed = FakeEndpoint()
    rerank = FakeEndpoint()

    def embed_responder(request):
        return Response(status=500, body={"error": "broken"})

    def rerank_responder(request):
        body = request.json()
        return Response(
            body={
                "results": [
                    {"index": i, "relevance_score": 0.9 if i == 0 else 0.2}
                    for i in range(len(body.get("documents", [])))
                ]
            }
        )

    embed.answer_with(embed_responder)
    rerank.answer_with(rerank_responder)
    try:
        cfg = make_config(
            tmp_path,
            cachePath=str(tmp_path / "index.json"),
            embedBaseUrl=embed.base_url,
            embedPath="/embeddings",
            embedModelId="my-embed",
            rerankBaseUrl=rerank.base_url,
            rerankPath="/rerank",
            rerankModel="my-rerank",
            healthCheckMs=0,
        )
        search_module.remember_dump("cold-fail-session")
        result = rewrite(
            PAYLOAD,
            cfg=cfg,
            session_id="cold-fail-session",
            turn_id="turn-1",
            notify=lambda _t: None,
        )
        assert result is not None, "a dumped window strips and appends the marker"
        note = read(cfg)["note"]
        assert "embedBaseUrl" in note or note == router_module.ENDPOINT_NOTE, (
            f"an endpoint build failure must select the endpoint diagnostic, got {note!r}"
        )
        assert "skills_list" not in note, (
            f"a pre-empted dumped check must write no dropped part, got {note!r}"
        )
    finally:
        embed.close()
        rerank.close()


def test_a_cold_local_build_failure_pre_empts_the_dropped_check(tmp_path):
    """A local build failure (the build raises) writes no note parts and no dropped part."""
    cfg = make_config(tmp_path)
    search_module.remember_dump("cold-local-session")

    def raising_embed(config, texts):
        raise ValueError("local build fault")

    result = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=raising_embed,
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="cold-local-session",
        notify=lambda _t: None,
    )
    assert result is not None
    stored = read(cfg)
    assert stored["status"] == INERT
    assert stored["note"] == "", (
        f"a local build failure pre-empts the dumped check with no parts, got {stored['note']!r}"
    )


# ---------------------------------------------------------------------------
# Saved-index (partial-view) producer (degraded tier).
# ---------------------------------------------------------------------------


def test_a_truncated_catalog_read_writes_the_saved_index_note(tmp_path):
    """Below half the saved rows: the note names the saved-index fallback and the rebuild fix."""
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    library_module._write_cache(cfg, make_index())
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
    assert "saved index" in note.lower(), (
        f"the note must name the saved-index fallback, got {note!r}"
    )
    assert "rebuild" in note.lower(), (
        f"the note must name the rebuild fix, got {note!r}"
    )
    assert "History incomplete" in note, (
        f"the degraded tier must carry every active producer, got {note!r}"
    )


def test_a_full_catalog_read_writes_no_saved_index_note(tmp_path):
    """The clearing half: a catalog matching the saved index names no fallback."""
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    library_module._write_cache(cfg, make_index())
    result = rewrite(
        _payload(),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="full-session",
        notify=lambda _t: None,
    )
    assert result is not None
    assert "saved index" not in read(cfg)["note"].lower()


# ---------------------------------------------------------------------------
# Cold-build announcement.
# ---------------------------------------------------------------------------


def test_a_cold_build_announces_itself_while_it_runs(tmp_path):
    """While a blocking responder holds the build, the live record shows the note."""
    released = threading.Event()

    def embed_responder(request):
        released.wait(timeout=4)
        body = request.json()
        count = len(body.get("input", [])) or 1
        return Response(
            body={
                "model": body.get("model", "my-embed"),
                "data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(count)],
            }
        )

    def rerank_responder(request):
        body = request.json()
        return Response(
            body={
                "results": [
                    {"index": i, "relevance_score": 0.9 if i == 0 else 0.2}
                    for i in range(len(body.get("documents", [])))
                ]
            }
        )

    embed = FakeEndpoint(embed_responder)
    rerank = FakeEndpoint(rerank_responder)
    try:
        cfg = make_config(
            tmp_path,
            cachePath=str(tmp_path / "index.json"),
            embedBaseUrl=embed.base_url,
            embedPath="/embeddings",
            embedModelId="my-embed",
            rerankBaseUrl=rerank.base_url,
            rerankPath="/rerank",
            rerankModel="my-rerank",
            healthCheckMs=0,
        )
        broadcast = []
        outcome = {}

        def run():
            outcome["result"] = rewrite(
                PAYLOAD,
                cfg=cfg,
                session_id="cold-session",
                turn_id="cold-turn",
                notify=broadcast.append,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 4.0
            live_note = ""
            while time.monotonic() < deadline:
                stored = read(cfg)
                if stored is not None:
                    live_note = stored["note"]
                if "Building" in live_note:
                    break
                time.sleep(0.01)
            assert "Building" in live_note, (
                f"the live record must announce the build while it runs, got {live_note!r}"
            )
            assert broadcast == ["cold-turn"], (
                f"the build announcement must broadcast before the build blocks, got {broadcast!r}"
            )
        finally:
            released.set()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the rewrite never returned after the build was released"
        assert outcome["result"] is not None
        final_note = read(cfg)["note"]
        assert "Building" not in final_note, (
            f"the settled record must not claim the build is still running, got {final_note!r}"
        )
    finally:
        released.set()
        embed.close()
        rerank.close()


# ---------------------------------------------------------------------------
# History wording and the heal-note Consumes rule.
# ---------------------------------------------------------------------------


def test_history_wording_matches_the_heal_attempt_state(tmp_path):
    """Running says running, finished says finished, absent claims neither."""
    store_path = str(tmp_path / "associations.db")
    cfg = make_config(tmp_path, associationDbPath=store_path)

    _seed_heal_state(store_path, "running")
    note = router_module._degraded_note(cfg)
    assert "healing is running" in note.lower(), f"running must say running, got {note!r}"

    _seed_heal_state(store_path, "finished")
    note = router_module._degraded_note(cfg)
    assert "heal finished, still incomplete" in note.lower(), (
        f"finished must say finished, got {note!r}"
    )
    assert "running" not in note.lower(), f"finished must not claim running, got {note!r}"

    associations_module.reset_heal_state()
    note = router_module._degraded_note(cfg)
    assert "running" not in note.lower() and "finished" not in note.lower(), (
        f"an absent attempt must claim neither, got {note!r}"
    )
    assert "History incomplete" in note, f"the loss is still named, got {note!r}"


def _exploding_heal(store_path, state_db_path=None, *, catalog=(), now_ms=None):
    raise sqlite3.OperationalError("database is locked")


def _wait_settled(store_path: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with associations_module._HEAL_LOCK:
            state = associations_module._HEAL_STATE.get(store_path)
        if state != associations_module._HEAL_RUNNING:
            return True
        time.sleep(0.01)
    return False


def test_a_heal_failure_shows_once_on_the_first_eligible_turn(tmp_path, monkeypatch):
    """Two successful incomplete-store turns: the failure line only on the first."""
    store_path = tmp_path / "associations.db"
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    monkeypatch.setattr(associations_module, "heal_store", _exploding_heal)
    assert associations_module.request_heal(str(store_path)) is True
    assert _wait_settled(str(store_path)), "the failed heal never settled"
    assert _pending_failure(str(store_path))

    assert _turn(cfg) is not None
    first = read(cfg)["note"]
    assert "heal failed" in first.lower(), f"the first turn shows the failure, got {first!r}"

    assert _turn(cfg) is not None
    second = read(cfg)["note"]
    assert "heal failed" not in second.lower(), (
        f"the heal note is consumed at most once, got {second!r}"
    )
    assert "heal finished, still incomplete" in second.lower(), (
        f"after the settle the line says finished, got {second!r}"
    )


def test_a_failed_and_budget_turns_never_consume_the_heal_note(tmp_path, monkeypatch):
    """Consumes requires the degraded tier: failure and budget tiers leave it pending."""
    from hermes_skill_router.search import Ranked

    store_path = tmp_path / "associations.db"
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    _seed_pending_failure(str(store_path))

    failed = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: None,
        session_id="consume-session",
        notify=lambda _t: None,
    )
    assert failed is None
    assert read(cfg)["note"] == router_module.ENDPOINT_NOTE
    assert _pending_failure(str(store_path)), "a failed turn consumed the heal note"

    real_rank = router_module.rank
    monkeypatch.setattr(
        router_module,
        "rank",
        lambda *a, **k: Ranked(reason="budget-expired", detail="routing budget 1000 ms exhausted"),
    )
    assert rewrite(
        PAYLOAD,
        cfg=cfg,
        index=INDEX,
        embed=lambda config, texts: [[1.0, 0.0] for _t in texts],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="consume-budget-session",
        notify=lambda _t: None,
    ) is None
    assert read(cfg)["note"] == router_module.BUDGET_NOTE
    assert _pending_failure(str(store_path)), "a budget turn consumed the heal note"
    monkeypatch.setattr(router_module, "rank", real_rank)

    assert _turn(cfg, session_id="consume-ok-session") is not None
    assert "heal failed" in read(cfg)["note"].lower(), (
        "the first eligible successful turn still shows the failure"
    )
    assert not _pending_failure(str(store_path)), "the eligible turn consumed it"


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


def test_a_complete_store_never_consumes_and_its_observation_clears_unread(tmp_path, monkeypatch):
    """Consumes requires an incomplete store; the observation may clear before display."""
    newest = 1_800_000_000_000
    state_path = tmp_path / "state.db"
    _session_store(state_path, newest)
    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda home=None: str(state_path)
    )
    store_path = tmp_path / "associations.db"
    _complete_store(store_path, newest)
    cfg = make_config(tmp_path, associationDbPath=str(store_path))
    _seed_pending_failure(str(store_path))

    note = router_module._degraded_note(cfg)
    assert "heal failed" not in note.lower(), f"a complete store shows no failure, got {note!r}"
    assert _pending_failure(str(store_path)), "a complete-store turn consumed the heal note"

    associations_module.note_store_complete(str(store_path))
    assert not _pending_failure(str(store_path)), "the observation did not clear it"
    note = router_module._degraded_note(cfg)
    assert "heal failed" not in note.lower(), (
        f"a later turn must show no failure line, got {note!r}"
    )


# ---------------------------------------------------------------------------
# Tier order.
# ---------------------------------------------------------------------------


def test_failure_beats_budget_and_budget_beats_degraded(tmp_path, monkeypatch):
    """The dropped-window failure pre-empts rank; the budget note hides degraded parts."""
    from hermes_skill_router.search import Ranked

    cfg = make_config(tmp_path, associationDbPath=str(tmp_path / "associations.db"))
    search_module.remember_dump("order-session")
    monkeypatch.setattr(router_module, "rank", _exploding)
    result = _turn(cfg, session_id="order-session", index=INDEX)
    assert result is not None
    assert "skills_list" in read(cfg)["note"], "failure must beat budget and degraded"

    monkeypatch.setattr(
        router_module,
        "rank",
        lambda *a, **k: Ranked(reason="budget-expired", detail="routing budget 1000 ms exhausted"),
    )
    assert _turn(cfg, session_id="order-budget-session", index=INDEX) is None
    note = read(cfg)["note"]
    assert note == router_module.BUDGET_NOTE, f"budget must beat degraded, got {note!r}"


def test_a_build_failure_beats_the_degraded_tier(tmp_path):
    """A local build failure with an incomplete store writes no degraded parts."""
    cfg = make_config(tmp_path, associationDbPath=str(tmp_path / "associations.db"))

    def raising_embed(config, texts):
        raise ValueError("local build fault")

    assert rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=raising_embed,
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="order-fail-session",
        notify=lambda _t: None,
    ) is None
    assert read(cfg)["note"] == "", "a build failure must leave no degraded parts"


def test_every_active_degraded_producer_appears(tmp_path):
    """History, skip state, BM25, cache and traps all appear in their eligible note."""
    store_path = tmp_path / "associations.db"
    associations_module.open_store(str(store_path)).close()
    cfg = make_config(
        tmp_path,
        associationDbPath=str(store_path),
        badkey=1,
    )
    assert getattr(cfg, "trap_notes", ()), "the unknown key must raise a trap"
    note = router_module._degraded_note(cfg, index=INDEX)
    assert "History incomplete" in note, f"history must appear, got {note!r}"
    assert "(empty store)" in note, f"the skip state must appear, got {note!r}"
    assert "BM25" in note, f"a missing term index must appear, got {note!r}"
    assert "Cache persistence is off" in note, f"persistence off must appear, got {note!r}"
    for trap in cfg.trap_notes:
        assert trap in note, f"trap {trap!r} must appear, got {note!r}"


def test_fusion_off_replaces_the_history_producer(tmp_path):
    """The cosine rollback never opens the store, so it names fusion off, not history."""
    cfg = make_config(
        tmp_path,
        associationDbPath=str(tmp_path / "associations.db"),
        fusionMode="cosine",
    )
    note = router_module._degraded_note(cfg, index=INDEX)
    assert "Fusion is off" in note, f"fusion must appear, got {note!r}"
    assert "History" not in note, f"cosine must not claim history, got {note!r}"


# ---------------------------------------------------------------------------
# Search preserves the note (C6).
# ---------------------------------------------------------------------------


def test_a_search_during_a_degraded_state_carries_the_note_forward(tmp_path, monkeypatch):
    """A manual lookup never erases the loss: the search record keeps the turn's note."""
    cfg = make_config(tmp_path, associationDbPath=str(tmp_path / "associations.db"))
    assert _turn(cfg) is not None
    degraded = read(cfg)["note"]
    assert "History incomplete" in degraded

    monkeypatch.setattr(
        tools_module, "EMBED", lambda config, texts: [[1.0, 0.0] if "Alpha" in t else [0.0, 1.0] for t in texts]
    )
    monkeypatch.setattr(tools_module, "RERANK", lambda config, query, documents: [0.9, 0.2][: len(documents)])
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    answer = tools_module.handle(
        {"query": "alpha skill"},
        settings={
            "cachePath": "",
            "suggestionsPath": str(tmp_path / "suggestions.json"),
            "associationDbPath": str(tmp_path / "associations.db"),
        },
        session_id="session-1",
    )
    envelope = json.loads(answer)
    assert "skills" in envelope, f"the search must answer, got {answer!r}"
    stored = read(cfg)
    assert stored["origin"] == "search"
    assert stored["note"] == degraded, (
        f"the search record must carry the degraded line forward, got {stored['note']!r}"
    )
