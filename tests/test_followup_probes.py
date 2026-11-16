"""Ticket 06 (kata hc1p): README, kept behaviors, and the evidence probes.

Two evidence probes run against an isolated profile (temporary files only,
never the real data root): Probe A is the not-ready endpoint regression (a
dead custom endpoint names the failing setting and the repair-service
remedy; a dead bundled endpoint names the failing setting and the model
setup script remedy), and Probe B is the continuous-update probe (on a
routed turn after new session data, the watermark advances and the
incomplete window never blocks the update). The integrated new-producer
search check (a successful search during a degraded state carries the
previous note, including ticket 01's new producers; a failed search writes
no record) lives here because it is gated by tickets 01 and 04.

The probes are built from the verified temporary scripts in the ticket
body, adapted to pytest fixtures: per-test temporary directories, one
bounded poll for the async ingest, and payload helpers that are
self-contained (never ``from tests.test_...`` and never
``test_middleware_contract``). No production code is changed by this
module: every behavior pinned here is already live from tickets 01, 04,
and 07.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
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
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import SEARCH, read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402

SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Before replying, scan the skills below.\n\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)
QUERY = "draft a git commit message for a bug fix"
PAYLOAD = {
    "model": "test-model",
    "tools": [{"type": "function", "function": {"name": "skill_view"}}],
    "unrelated_key": {"kept": True},
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUERY},
    ],
}
INDEX = Index(
    names=["alpha", "beta"],
    descriptions=["Alpha skill", "Beta skill"],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    dim=2,
    fingerprint="followup-probes",
)
T0 = 1_700_000_000_000
SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    started_at REAL NOT NULL,
    last_activity_at REAL
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    display_kind TEXT,
    compacted INTEGER DEFAULT 0,
    active INTEGER DEFAULT 1
);
"""


@pytest.fixture(autouse=True)
def no_live_heal():
    """Shadow the conftest guard: Probe B drives the real scan entry points.

    Every store path in this module is a per-test temporary file, so the
    real heal and learning scans never touch the operator's profile.
    """
    associations_module.reset_heal_state()
    yield
    associations_module.reset_heal_state()


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process state, so one probe never masks another."""
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


def _make_config(tmp_path, **overrides):
    settings = {
        "cachePath": "",
        "suggestionsPath": str(tmp_path / "suggestions.json"),
        "associationDbPath": str(tmp_path / "associations.db"),
    }
    settings.update(overrides)
    return load(settings)


# ---------------------------------------------------------------------------
# Probe A (not-ready endpoint regression): custom endpoint.
# ---------------------------------------------------------------------------


def test_probe_a_custom_endpoint_names_failing_setting_and_repair_remedy(tmp_path):
    """A dead custom base URL names embedBaseUrl and the repair-service remedy."""
    cfg = _make_config(
        tmp_path,
        embedBaseUrl="http://127.0.0.1:9",
        rerankBaseUrl="http://127.0.0.1:9",
        healthCheckMs=0,
        cutoff=0.5,
    )
    result = rewrite(
        PAYLOAD,
        cfg=cfg,
        session_id="s1",
        turn_id="t1",
        notify=lambda _turn_id: None,
    )
    assert result is None, "a dead endpoint leaves the request unchanged"
    note = json.loads((tmp_path / "suggestions.json").read_text(encoding="utf-8"))["note"]
    assert note == (
        "Model setup needed: embedBaseUrl is not ready; "
        "repair the service at the configured embedBaseUrl endpoint and reload."
    ), f"the custom-endpoint note must be the repair-service line, got {note!r}"


# ---------------------------------------------------------------------------
# Probe A (not-ready endpoint regression): bundled endpoint.
# ---------------------------------------------------------------------------


def _broken_responder(request):
    return Response(status=500, body={"error": "broken"})


def _healthy_responder(request):
    if request.path == "/health":
        return Response(body={"status": "ok", "model": "my-rerank"})
    body = request.json()
    documents = body.get("documents", [])
    return Response(
        body={
            "results": [
                {"index": i, "relevance_score": 0.9 if i == 0 else 0.2}
                for i in range(len(documents))
            ]
        }
    )


def test_probe_a_bundled_endpoint_names_failing_setting_and_setup_script_remedy(
    tmp_path, monkeypatch
):
    """A dead bundled base names embedBaseUrl and the setup-script remedy.

    BUNDLED_BASES is pointed at loopback fakes (mutated in place, so the
    rewrite's own import sees the same rows), so the answer never depends
    on whether a real service happens to occupy a default port.
    """
    from hermes_skill_router import setup_check

    embed = FakeEndpoint()
    rerank = FakeEndpoint()
    embed.answer_with(_broken_responder)
    rerank.answer_with(_healthy_responder)
    try:
        monkeypatch.setitem(setup_check.BUNDLED_BASES, "embed", embed.base_url)
        monkeypatch.setitem(setup_check.BUNDLED_BASES, "rerank", rerank.base_url)
        cfg = _make_config(
            tmp_path,
            embedBaseUrl=embed.base_url,
            embedModelId="my-embed",
            rerankBaseUrl=rerank.base_url,
            rerankModel="my-rerank",
            healthCheckMs=0,
        )
        result = rewrite(
            PAYLOAD,
            cfg=cfg,
            index=INDEX,
            session_id="bundled-session",
            turn_id="bundled-turn",
            notify=lambda _turn_id: None,
        )
        assert result is None, "a dead bundled endpoint leaves the request unchanged"
        note = read(cfg)["note"]
        assert note == (
            "Model setup needed: embedBaseUrl is not ready; "
            "run the model setup script and reload."
        ), f"the bundled-endpoint note must be the setup-script line, got {note!r}"
    finally:
        embed.close()
        rerank.close()


# ---------------------------------------------------------------------------
# Probe B (continuous update): the watermark advances on a routed turn.
# ---------------------------------------------------------------------------


def _embed_responder(request):
    body = request.json()
    return Response(
        body={
            "model": "bge-m3",
            "data": [
                {"index": i, "embedding": [1.0, 0.0] if "alpha" in text else [0.0, 1.0]}
                for i, text in enumerate(body["input"])
            ],
        }
    )


def _rerank_responder(request):
    return Response(
        body={
            "results": [
                {"index": i, "relevance_score": 0.9}
                for i in range(len(request.json()["documents"]))
            ]
        }
    )


def _watermark(path) -> int:
    row = sqlite3.connect(
        f"file:{path}?mode=ro", uri=True, timeout=5
    ).execute("SELECT value FROM meta WHERE key='last_scan_ms'").fetchone()
    return int(row[0])


def test_probe_b_routed_turn_after_new_session_data_advances_the_watermark(
    tmp_path, monkeypatch
):
    """Complete store + new session: the routed turn's scan advances the watermark."""
    state_db = tmp_path / "state.db"
    assoc_db = tmp_path / "associations.db"

    conn = sqlite3.connect(str(state_db))
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, last_activity_at)"
        " VALUES (?, 'cli', ?, ?)",
        ("s1", T0 / 1000.0, (T0 + 900_000) / 1000.0),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp,"
        " display_kind, compacted, active) VALUES (?, 'user', ?, ?, NULL, 0, 1)",
        ("s1", "open the alpha skill", T0 / 1000.0),
    )
    conn.commit()
    conn.close()

    subprocess.run(
        [sys.executable, "-m", "hermes_skill_router.associations",
         "--db", str(assoc_db), "--state-db", str(state_db)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        timeout=60,
    )

    monkeypatch.setattr(
        associations_module, "default_state_db_path", lambda: str(state_db)
    )

    store = associations_module.open_store(str(assoc_db), read_only=True)
    try:
        complete = associations_module.is_store_complete(store)
    finally:
        store.close()
    assert complete, "precondition: the store must be complete inside the window"

    w0 = _watermark(str(assoc_db))

    conn = sqlite3.connect(str(state_db))
    conn.execute(
        "INSERT INTO sessions (id, source, started_at, last_activity_at)"
        " VALUES (?, 'cli', ?, ?)",
        ("s2", (T0 + 86_400_000) / 1000.0, (T0 + 86_400_000 + 60_000) / 1000.0),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp,"
        " display_kind, compacted, active) VALUES (?, 'user', ?, ?, NULL, 0, 1)",
        ("s2", "route beta again", (T0 + 86_400_000) / 1000.0),
    )
    conn.commit()
    conn.close()

    embeddings = FakeEndpoint(_embed_responder)
    reranking = FakeEndpoint(_rerank_responder)
    try:
        cfg = _make_config(
            tmp_path,
            embedBaseUrl=embeddings.base_url,
            rerankBaseUrl=reranking.base_url,
            healthCheckMs=0,
            cutoff=0.5,
        )
        assert cfg.association_db_path == str(assoc_db)
        result = rewrite(
            PAYLOAD,
            cfg=cfg,
            session_id="s2",
            turn_id="t2",
            notify=lambda _turn_id: None,
        )
        assert result is not None, "the routed turn did not route"
    finally:
        embeddings.close()
        reranking.close()

    deadline = time.monotonic() + 10
    w1 = _watermark(str(assoc_db))
    while w1 <= w0 and time.monotonic() < deadline:
        time.sleep(0.1)
        w1 = _watermark(str(assoc_db))
    assert w1 > w0, (
        f"the window must never block ingestion: W0={w0} must advance to W1, got W1={w1}"
    )


# ---------------------------------------------------------------------------
# Integrated search check: degraded note carry-forward with new producers.
# ---------------------------------------------------------------------------


def _search_settings(tmp_path):
    return {
        "cachePath": "",
        "suggestionsPath": str(tmp_path / "suggestions.json"),
        "associationDbPath": str(tmp_path / "associations.db"),
    }


def test_search_during_a_degraded_state_carries_the_note_with_new_producers(
    tmp_path, monkeypatch
):
    """A manual lookup keeps the saved-index producer on the search record."""
    cache_path = str(tmp_path / "index.json")
    cfg = _make_config(tmp_path, cachePath=cache_path)
    saved = Index(
        names=["alpha", "beta", "gamma", "delta"],
        descriptions=["Alpha skill", "Beta skill", "Gamma skill", "Delta skill"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dim=2,
        fingerprint="saved-index-producer",
    )
    library_module._write_cache(cfg, saved)
    rows = (("alpha", "Alpha skill"),)
    lines = ["  general:"] + [f"    - {n}: {d}" for n, d in rows]
    system = (
        "You are Hermes Agent.\n\n"
        "## Skills\n"
        "Before replying, scan the skills below.\n\n"
        "<available_skills>\n"
        + "\n".join(lines)
        + "\n</available_skills>\n\n"
        "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
        "## Memory\n\nNothing yet."
    )
    payload = {
        "model": "test-model",
        "tools": [{"type": "function", "function": {"name": "skill_view"}}],
        "unrelated_key": {"kept": True},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": QUERY},
        ],
    }
    result = rewrite(
        payload,
        cfg=cfg,
        embed=lambda config, texts: [[1.0, 0.0] if "Alpha" in t else [0.0, 1.0] for t in texts],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="saved-index-session",
        turn_id="turn-1",
        notify=lambda _turn_id: None,
    )
    assert result is not None, "a refused catalog routes from the saved index"
    degraded = read(cfg)["note"]
    assert "saved index" in degraded.lower(), f"the new producer must be present, got {degraded!r}"
    assert "History incomplete" in degraded, f"history must be present, got {degraded!r}"

    monkeypatch.setattr(
        tools_module, "EMBED",
        lambda config, texts: [[1.0, 0.0] if "Alpha" in t else [0.0, 1.0] for t in texts],
    )
    monkeypatch.setattr(
        tools_module, "RERANK", lambda config, query, documents: [0.9, 0.2][: len(documents)]
    )
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    answer = tools_module.handle(
        {"query": "alpha skill"},
        settings=_search_settings(tmp_path),
        session_id="saved-index-session",
    )
    envelope = json.loads(answer)
    assert "skills" in envelope, f"the search must answer, got {answer!r}"
    stored = read(cfg)
    assert stored["origin"] == SEARCH
    assert stored["note"] == degraded, (
        f"the search record must carry the new producers forward, got {stored['note']!r}"
    )


def test_a_failed_search_writes_no_record(tmp_path, monkeypatch):
    """A degraded search answer leaves the previous record untouched."""
    cfg = _make_config(tmp_path)
    result = rewrite(
        PAYLOAD,
        cfg=cfg,
        index=INDEX,
        embed=lambda config, texts: [[1.0, 0.0] if "Alpha" in t else [0.0, 1.0] for t in texts],
        rerank=lambda config, query, documents: [0.9, 0.2][: len(documents)],
        session_id="search-session",
        turn_id="turn-1",
        notify=lambda _turn_id: None,
    )
    assert result is not None
    before = (tmp_path / "suggestions.json").read_bytes()
    before_note = read(cfg)["note"]

    monkeypatch.setattr(tools_module, "EMBED", lambda config, texts: None)
    monkeypatch.setattr(
        tools_module, "RERANK", lambda config, query, documents: [0.9, 0.2][: len(documents)]
    )
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    answer = tools_module.handle(
        {"query": "alpha skill"},
        settings=_search_settings(tmp_path),
        session_id="search-session",
    )
    envelope = json.loads(answer)
    assert envelope.get("degraded") is True, f"a failed search must answer degraded, got {answer!r}"
    assert envelope.get("skills") == []
    assert (tmp_path / "suggestions.json").read_bytes() == before, (
        "a failed search must write no record"
    )
    assert read(cfg)["note"] == before_note
