"""Ticket 05: the analytics script for score-vs-association drift (contract C5).

Every case runs the script as a subprocess against a synthetic v2 association
copy, a synthetic Hermes session store and a decision log, with a stubbed
embed endpoint. Ticket 02 owns the real v2 ingestion this script gates on;
the stores here are built to the same contract (``scan_ledger``,
``seen_sessions``, ``creation_origin`` provenance, loss marker) so the
fixtures stand in for its output until it lands.

Coverage of the ticket checklist: planted anomalies flag exactly the planted
skills with the planted top-5; a shortlist-cut skill still scores; catalog
identity is by name through ``resolve_skill_name`` (path never keys the
join); a repeated catalog name is an identity error; the anomaly threshold
is one named constant; BM25 parity (tokenizer, pinned ``k1``/``b``,
document text, cap sensitivity); historical replay (future events ignored,
zero when verified with no matching events); every coverage-provenance
fixture (marker, pre-v2, enrolment signals, finality counters, empty scans,
non-positive ids, malformed/unclassifiable calls, valid controls); the copy
protocol (SQLite backup includes uncheckpointed rows, pre-v2 rejected);
source-deletion fixtures; source-finality flow; the same-millisecond append
regression; and the missing-data skips. Integration against a real
ticket-02 store is blocked by 02 and not run here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint, Response  # noqa: E402
from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router import bm25  # noqa: E402
from hermes_skill_router import library  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402

SCRIPT = ROOT / "tools" / "analytics_scores.py"

#: v2 tables per ticket 02's contract; ticket 05 asserts their content.
V2_DDL = """
CREATE TABLE IF NOT EXISTS scan_ledger (
  session_id TEXT PRIMARY KEY,
  scanned_max_row_id INTEGER,
  pending_calls INTEGER,
  unreadable_results INTEGER,
  unreadable_calls INTEGER
);
CREATE TABLE IF NOT EXISTS seen_sessions (
  session_id TEXT PRIMARY KEY,
  started_at_ms INTEGER
);
"""

T0 = 1_700_000_000_000
T_LOG = T0 + 5_000_000

#: One day in milliseconds, the unit the decay windows use.
DAY = 86_400_000

#: Vocabulary the stub embedder counts. Query and document tokens resolve
#: against it, so cosine is controllable per fixture.
VOCAB = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "quokka", "mammoth", "zebra",
] + [f"w{index:02d}" for index in range(7)]


def iso(ms: int) -> str:
    """Return the ISO-8601 UTC form of one epoch millisecond."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def embed_vector(text: str) -> List[float]:
    """Embed by counting vocabulary tokens; the stub's scoring rule."""
    tokens = bm25.tokenise(text)
    return [float(tokens.count(word)) for word in VOCAB]


def embed_response(request) -> Response:
    """Answer one OpenAI-shaped embeddings request with the stub rule."""
    payload = request.json()
    vectors = [embed_vector(text) for text in payload["input"]]
    return Response(body={"data": [{"embedding": vector} for vector in vectors]})


@pytest.fixture
def endpoint():
    """A loopback embed endpoint the script talks to as a subprocess."""
    server = FakeEndpoint(responder=embed_response)
    try:
        yield server
    finally:
        server.close()


def make_store(
    path: Path,
    *,
    user_version: int = 2,
    schema_version: str = "2",
    creation_origin: Optional[str] = "fresh",
    marker: Optional[str] = None,
    meta_extra: Optional[Dict[str, str]] = None,
    skills: Sequence[Tuple[str, str]] = (
        ("alpha-skill", "hermes:alpha-skill"),
        ("beta-skill", "hermes:beta-skill"),
    ),
    prompt_events: Sequence[Tuple[int, str, str, int]] = (),
    load_events: Sequence[Tuple[int, int, float, int, str]] = (),
    ledger: Sequence[Tuple[str, Optional[int], int, int, int]] = (),
    seen: Sequence[Tuple[str, Optional[int]]] = (),
    wal: bool = False,
) -> sqlite3.Connection:
    """Build one v2-shaped association copy and return its open writer.

    ``prompt_events`` rows are ``(id, session_id, terms, observed_ms)`` and
    ``load_events`` rows are ``(prompt_event_id, skill_id, strength,
    observed_ms, terms_or_empty)``. The store is left open so the
    WAL/backup cases can hold it; callers close it.
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.executescript(associations_module.SCHEMA_V1)
    conn.executescript(V2_DDL)
    meta: Dict[str, str] = {
        "harness": "hermes",
        "schema_version": schema_version,
        "last_scan_ms": "0",
        "catalog_fingerprint": "",
    }
    if creation_origin is not None:
        meta["creation_origin"] = creation_origin
    if marker is not None:
        meta["last_history_loss_ms"] = marker
    if meta_extra:
        meta.update(meta_extra)
    conn.execute("BEGIN IMMEDIATE")
    for key, value in meta.items():
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
        )
    conn.execute(f"PRAGMA user_version = {int(user_version)}")
    for name, skill_path in skills:
        cursor = conn.execute(
            """INSERT INTO skill (harness, name, path, first_seen_ms, last_seen_ms,
                              deleted_at_ms)
               VALUES ('hermes', ?, ?, ?, ?, NULL)""",
            (name, skill_path, T0, T0),
        )
        conn.execute(
            """INSERT INTO skill_alias (harness, name, skill_id, valid_from_ms, valid_to_ms)
               VALUES ('hermes', ?, ?, ?, NULL)""",
            (name, int(cursor.lastrowid), T0),
        )
    for prompt_id, session_id, terms, observed_ms in prompt_events:
        conn.execute(
            """INSERT INTO prompt_event (id, session_id, turn, prompt_hash, terms, length,
                                      observed_ms)
               VALUES (?, ?, 1, ?, ?, ?, ?)""",
            (prompt_id, session_id, f"hash-{prompt_id}", terms, len(terms), observed_ms),
        )
    for index, (prompt_id, skill_id, strength, observed_ms, _terms) in enumerate(
        load_events
    ):
        conn.execute(
            """INSERT INTO load_event (prompt_event_id, skill_id, evidence, strength, origin,
                                   source, observed_ms, dedup_key)
               VALUES (?, ?, 'tool', ?, 'human', 'backfill', ?, ?)""",
            (prompt_id, skill_id, strength, observed_ms, f"dedup-{index}|{observed_ms}"),
        )
    for session_id, scanned_max, pending, unreadable_results, unreadable_calls in ledger:
        conn.execute(
            """INSERT INTO scan_ledger (session_id, scanned_max_row_id, pending_calls,
                                     unreadable_results, unreadable_calls)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, scanned_max, pending, unreadable_results, unreadable_calls),
        )
    for session_id, started_ms in seen:
        conn.execute(
            "INSERT INTO seen_sessions (session_id, started_at_ms) VALUES (?, ?)",
            (session_id, started_ms),
        )
    conn.execute("COMMIT")
    return conn


def make_state_db(
    path: Path,
    *,
    sessions: Sequence[Tuple[str, Optional[int], Optional[int]]] = (),
    messages: Sequence[Tuple[int, str, str, int, Optional[str], Optional[str], Optional[str]]] = (),
) -> None:
    """Build one Hermes session store: ``(id, started_ms, last_activity_ms)``
    sessions and ``(id, session_id, role, ts_ms, tool_calls, tool_call_id,
    tool_name)`` messages."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            started_at REAL,
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
    )
    for session_id, started_ms, last_activity_ms in sessions:
        conn.execute(
            "INSERT INTO sessions (id, source, started_at, last_activity_at)"
            " VALUES (?, 'cli', ?, ?)",
            (
                session_id,
                None if started_ms is None else started_ms / 1000.0,
                None if last_activity_ms is None else last_activity_ms / 1000.0,
            ),
        )
    for message_id, session_id, role, ts_ms, tool_calls, tool_call_id, tool_name in messages:
        conn.execute(
            """INSERT INTO messages (id, session_id, role, content, tool_calls,
                                 tool_call_id, tool_name, timestamp)
               VALUES (?, ?, ?, '', ?, ?, ?, ?)""",
            (message_id, session_id, role, tool_calls, tool_call_id, tool_name, ts_ms / 1000.0),
        )
    conn.commit()
    conn.close()


def decision_line(stamp_ms: int, query: str, candidates: str = "alpha-skill:0.9,beta-skill:0.1") -> str:
    """One decision log line in the router's format."""
    return (
        f"{iso(stamp_ms)} INFO hermes_skill_router.router "
        f"skill router: query={json.dumps(query)} candidates={candidates} cutoff=0.5 "
        f"suggestions=alpha-skill:0.9 embed_ms=1.0 rerank_ms=2.0 total_ms=3.0\n"
    )


#: One history event for alpha-skill on the happy query, before the log line.
HAPPY_PROMPTS = [(1, "s-1", "quokka mammoth zebra", T_LOG - 4_000_000)]
HAPPY_EVENTS = [(1, 1, 1.0, T_LOG - 3_000_000, "")]
HAPPY_MESSAGES = [
    (1, "s-1", "user", T0, None, None, None),
    (2, "s-1", "user", T0 + 100, None, None, None),
]


def make_case(
    tmp_path: Path,
    *,
    catalog_names: Sequence[str] = ("alpha-skill", "beta-skill"),
    descriptions: Sequence[str] = ("beta gamma delta epsilon", "eta theta iota kappa"),
    queries: Sequence[Tuple[int, str]] = ((T_LOG, "quokka mammoth zebra"),),
    store_kwargs: Optional[dict] = None,
    sessions: Optional[Sequence[Tuple[str, Optional[int], Optional[int]]]] = None,
    messages: Optional[Sequence[Tuple[int, str, str, int, Optional[str], Optional[str], Optional[str]]]] = None,
    catalog_model: str = "bge-m3",
) -> Dict[str, Path]:
    """Build one complete case: store, session store, log and catalog.

    ``catalog_model`` is the build model the catalog copy records: it must
    match the model the replay resolves to, or the run is a parity error.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "store": tmp_path / "associations.db",
        "state": tmp_path / "state.db",
        "log": tmp_path / "plugin.log",
        "catalog": tmp_path / "index.json",
    }
    store_kwargs = dict(store_kwargs or {})
    store_kwargs.setdefault("prompt_events", HAPPY_PROMPTS)
    store_kwargs.setdefault("load_events", HAPPY_EVENTS)
    store_kwargs.setdefault("ledger", [("s-1", 2, 0, 0, 0)])
    store_kwargs.setdefault("seen", [("s-1", T0)])
    make_store(paths["store"], **store_kwargs).close()
    make_state_db(
        paths["state"],
        sessions=[("s-1", T0, T0 + 900_000)] if sessions is None else list(sessions),
        messages=HAPPY_MESSAGES if messages is None else list(messages),
    )
    paths["log"].write_text(
        "".join(decision_line(stamp, query) for stamp, query in queries),
        encoding="utf-8",
    )
    payload = library._cache_payload(
        Index(
            names=list(catalog_names),
            descriptions=list(descriptions),
            vectors=[[0.0] * 4 for _ in catalog_names],
            dim=4,
            fingerprint="fixture",
        ),
        catalog_model,
    )
    paths["catalog"].write_text(json.dumps(payload), encoding="utf-8")
    return paths


def run_script(
    endpoint,
    paths: Dict[str, Path],
    *,
    cap: int = 400,
    assert_history: bool = True,
    store: Optional[Path] = None,
    log: Optional[Path] = None,
    state: Optional[Path] = None,
    embed_url: Optional[str] = None,
    embed_model: str = "bge-m3",
    embed_timeout_ms: int = 6000,
):
    """Run the script as a subprocess and return the completed process."""
    command = [
        sys.executable,
        str(SCRIPT),
        "--store", str(store or paths["store"]),
        "--log", str(log or paths["log"]),
        "--catalog", str(paths["catalog"]),
        "--embed-url", embed_url or f"{endpoint.base_url}/v1/embeddings",
        "--embed-model", embed_model,
        "--embed-timeout-ms", str(embed_timeout_ms),
        "--max-description-chars", str(cap),
        "--state-db", str(state or paths["state"]),
    ]
    if assert_history:
        command.append("--assert-complete-source-history")
    return subprocess.run(command, capture_output=True, text=True, timeout=120)


def anomalies_for(stdout: str) -> Dict[str, List[Tuple[str, float]]]:
    """Parse the report into ``{skill: [(query, divergence), ...]}``."""
    sections: Dict[str, List[Tuple[str, float]]] = {}
    current: Optional[str] = None
    decoder = json.JSONDecoder()
    for line in stdout.splitlines():
        if line.startswith("ANOMALY "):
            current = line.split(" ", 1)[1]
            sections[current] = []
        elif line.startswith("  ") and current is not None:
            stripped = line.strip()
            query, end = decoder.raw_decode(stripped)
            sections[current].append((query, float(stripped[end:].strip())))
    return sections


def assert_coverage_error(result, needle: str = "") -> None:
    """Assert a named coverage error on stderr and a nonzero exit."""
    assert result.returncode != 0, result.stdout
    assert "analytics coverage error" in result.stderr, result.stderr
    if needle:
        assert needle in result.stderr, result.stderr


#--------------------------------------------------------------------------------------------------
# Happy path: planted anomalies, top-5 order, clean verdict
#--------------------------------------------------------------------------------------------------


def test_planted_anomaly_flags_exactly_the_planted_skill(endpoint, tmp_path):
    paths = make_case(tmp_path)
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    sections = anomalies_for(result.stdout)
    assert set(sections) == {"alpha-skill"}
    assert sections["alpha-skill"] == [("quokka mammoth zebra", pytest.approx(1.0))]


def test_documented_invocation_verbatim_without_embed_flags(endpoint, tmp_path):
    """Round 3 P2: ticket 05 / skeleton C5's exact command works unchanged.

    The documented invocation omits --embed-model/--embed-timeout-ms; the
    run config's defaults (embed_model_id/embed_timeout_ms) resolve them,
    so it replays in the same vector space as the explicit invocation.
    """
    paths = make_case(tmp_path)
    command = [
        sys.executable,
        str(SCRIPT),
        "--store", str(paths["store"]),
        "--log", str(paths["log"]),
        "--catalog", str(paths["catalog"]),
        "--embed-url", f"{endpoint.base_url}/v1/embeddings",
        "--max-description-chars", "400",
        "--state-db", str(paths["state"]),
        "--assert-complete-source-history",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}
    explicit = run_script(endpoint, paths)
    assert explicit.returncode == 0, explicit.stderr
    assert result.stdout == explicit.stdout


CUSTOM_MODEL = "custom-embed-model"
CUSTOM_TIMEOUT_MS = 12345


def _hermes_home(tmp_path, *, model=CUSTOM_MODEL, timeout_ms=CUSTOM_TIMEOUT_MS):
    """Write a workspace config naming a nondefault embed model and timeout."""
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    hermes-skill-router:\n      settings:\n"
        f"        embedModelId: {model}\n        embedTimeoutMs: {timeout_ms}\n",
        encoding="utf-8",
    )
    return home


def _multi_model_endpoint(model=CUSTOM_MODEL):
    """An endpoint that only accepts *model* and records what it receives."""
    received: List[str] = []

    def responder(request):
        payload = request.json()
        received.append(str(payload.get("model")))
        if str(payload.get("model")) != model:
            return Response(body={"error": "wrong model space"}, status=400)
        vectors = [embed_vector(text) for text in payload["input"]]
        return Response(body={"data": [{"embedding": vector} for vector in vectors]})

    server = FakeEndpoint(responder)
    return server, received


def _documented_command(endpoint, paths, *, model_flag=None, timeout_flag=None):
    command = [
        sys.executable,
        str(SCRIPT),
        "--store", str(paths["store"]),
        "--log", str(paths["log"]),
        "--catalog", str(paths["catalog"]),
        "--embed-url", f"{endpoint.base_url}/v1/embeddings",
    ]
    if model_flag is not None:
        command += ["--embed-model", model_flag]
    if timeout_flag is not None:
        command += ["--embed-timeout-ms", str(timeout_flag)]
    command += [
        "--max-description-chars", "400",
        "--state-db", str(paths["state"]),
        "--assert-complete-source-history",
    ]
    return command


def test_custom_workspace_model_replays_in_its_own_space(tmp_path, monkeypatch):
    """Round 4 P2: the documented invocation replays in the run's own space.

    The workspace config names a nondefault embed model and timeout; with no
    embed flags, the script resolves them from the workspace config and an
    endpoint that only knows the custom model accepts the run. The builtin
    default model would be rejected by that endpoint (see the override test).
    """
    monkeypatch.setenv("HERMES_HOME", str(_hermes_home(tmp_path)))
    endpoint, received = _multi_model_endpoint()
    try:
        paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
        result = subprocess.run(
            _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
        assert set(anomalies_for(result.stdout)) == {"alpha-skill"}
        assert received and all(model == CUSTOM_MODEL for model in received), received
    finally:
        endpoint.close()


def test_explicit_flags_override_the_workspace_config(tmp_path, monkeypatch):
    """Round 4 P2: the explicit flags stay the override.

    With the same custom workspace config, an explicit builtin model flag
    replays in the builtin space, which the custom-only endpoint rejects.
    """
    monkeypatch.setenv("HERMES_HOME", str(_hermes_home(tmp_path)))
    endpoint, _received = _multi_model_endpoint()
    try:
        paths = make_case(tmp_path)
        result = subprocess.run(
            _documented_command(endpoint, paths, model_flag="bge-m3", timeout_flag=6000),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode != 0
        assert "analytics embed error" in result.stderr, result.stderr
    finally:
        endpoint.close()


def test_single_explicit_model_flag_ignores_invalid_workspace_model(tmp_path, monkeypatch):
    """Round 7 coverage: one explicit flag overrides its workspace field.

    Only ``--embed-model`` is explicit; the workspace ``embedModelId``
    is invalid (blank) but the explicit flag overrides it, so it is
    not validated (C5: explicit flags always override the workspace
    config). The workspace timeout is still used for the omitted
    timeout flag, and the run must succeed in the explicit space.
    """
    monkeypatch.setenv(
        "HERMES_HOME", str(_hermes_home(tmp_path, model="''", timeout_ms=6000))
    )
    endpoint, received = _multi_model_endpoint()
    try:
        paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
        result = subprocess.run(
            _documented_command(endpoint, paths, model_flag=CUSTOM_MODEL),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert set(anomalies_for(result.stdout)) == {"alpha-skill"}
        assert received and all(model == CUSTOM_MODEL for model in received), received
    finally:
        endpoint.close()


def test_single_explicit_timeout_flag_ignores_invalid_workspace_timeout(tmp_path, monkeypatch):
    """Round 7 coverage: one explicit flag overrides its workspace field.

    Only ``--embed-timeout-ms`` is explicit; the workspace
    ``embedTimeoutMs`` is out of range but the explicit flag
    overrides it, so it is not validated (C5). The workspace model
    is still used for the omitted model flag, and the run must
    succeed in the workspace space.
    """
    monkeypatch.setenv(
        "HERMES_HOME", str(_hermes_home(tmp_path, timeout_ms=999999))
    )
    endpoint, received = _multi_model_endpoint()
    try:
        paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
        result = subprocess.run(
            _documented_command(endpoint, paths, timeout_flag=6000),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert set(anomalies_for(result.stdout)) == {"alpha-skill"}
        assert received and all(model == CUSTOM_MODEL for model in received), received
    finally:
        endpoint.close()


def test_default_only_invocation_without_custom_config(endpoint, tmp_path, monkeypatch):
    """Round 4 P2: with no custom config, the documented invocation uses the builtins."""
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - hermes-skill-router\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    paths = make_case(tmp_path)
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_unreadable_or_unusable_workspace_config_fails_closed(endpoint, tmp_path, monkeypatch):
    """Round 4 P2: a bad workspace config is a named nonzero exit, never a silent default."""
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    paths = make_case(tmp_path)

    (home / "config.yaml").write_text("plugins: [unclosed\n", encoding="utf-8")
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "workspace config" in result.stderr, result.stderr

    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    hermes-skill-router:\n      settings:\n"
        "        embedModelId: ''\n        embedTimeoutMs: 6000\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "embedModelId" in result.stderr, result.stderr

    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    hermes-skill-router:\n      settings:\n"
        "        embedModelId: bge-m3\n        embedTimeoutMs: 99\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "embedTimeoutMs" in result.stderr, result.stderr


def test_invalid_utf8_workspace_config_fails_closed(endpoint, tmp_path, monkeypatch):
    """Round 5 P2: invalid UTF-8 in the workspace config is a named nonzero
    exit, never a UnicodeDecodeError traceback."""
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    paths = make_case(tmp_path)
    (home / "config.yaml").write_bytes(
        b"plugins:\n  entries:\n    hermes-skill-router:\n      settings:\n"
        b"        embedModelId: \xff\xfe\n"
    )
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "analytics error" in result.stderr, result.stderr
    assert "workspace config" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr


def _multi_model_accepting_endpoint(models):
    """An endpoint that accepts every model in *models* and records them.

    A wrong-space replay would be accepted here, so a passing run proves
    nothing about parity unless the script itself refuses.
    """
    received: List[str] = []

    def responder(request):
        payload = request.json()
        received.append(str(payload.get("model")))
        vectors = [embed_vector(text) for text in payload["input"]]
        return Response(body={"data": [{"embedding": vector} for vector in vectors]})

    server = FakeEndpoint(responder)
    return server, received


def test_wrong_model_replay_is_a_named_parity_error_before_embedding(tmp_path):
    """Round 5 P2: an explicit model other than the catalog's recorded build
    model is a parity error, before any embedding call is made."""
    endpoint, received = _multi_model_accepting_endpoint(["bge-m3", CUSTOM_MODEL])
    try:
        paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
        result = run_script(endpoint, paths, embed_model="bge-m3")
        assert result.returncode != 0
        assert "analytics parity error" in result.stderr, result.stderr
        assert CUSTOM_MODEL in result.stderr and "bge-m3" in result.stderr, result.stderr
        assert received == [], f"parity must refuse before embedding, got {received}"
        matched = run_script(endpoint, paths, embed_model=CUSTOM_MODEL)
        assert matched.returncode == 0, matched.stderr
        assert received and all(model == CUSTOM_MODEL for model in received), received
    finally:
        endpoint.close()


def test_stale_workspace_config_is_a_named_parity_error(tmp_path, monkeypatch):
    """Round 5 P2: the workspace config changed since the logged run.

    The current workspace names a different model than the catalog's
    recorded build model; the documented invocation (no flags) must exit
    with a parity error instead of silently replaying in the new space.
    """
    home = tmp_path / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    hermes-skill-router:\n      settings:\n"
        "        embedModelId: stale-model\n        embedTimeoutMs: 6000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    endpoint, received = _multi_model_accepting_endpoint(["stale-model", CUSTOM_MODEL])
    try:
        paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
        result = subprocess.run(
            _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
        )
        assert result.returncode != 0
        assert "analytics parity error" in result.stderr, result.stderr
        assert "stale-model" in result.stderr and CUSTOM_MODEL in result.stderr, result.stderr
        assert received == [], f"parity must refuse before embedding, got {received}"
    finally:
        endpoint.close()


def test_legacy_catalog_without_recorded_model_needs_an_explicit_flag(endpoint, tmp_path):
    """Round 5 P2: a catalog written before the build model was recorded.

    The documented invocation without ``--embed-model`` is a parity error
    naming the flag; with the explicit flag the run replays normally.
    """
    paths = make_case(tmp_path)
    payload = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    payload.pop("embed_model", None)
    paths["catalog"].write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "analytics parity error" in result.stderr, result.stderr
    assert "--embed-model" in result.stderr, result.stderr

    explicit = run_script(endpoint, paths, embed_model="bge-m3")
    assert explicit.returncode == 0, explicit.stderr
    assert set(anomalies_for(explicit.stdout)) == {"alpha-skill"}


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_legacy_catalog_blank_explicit_model_is_a_named_parity_error(tmp_path, blank):
    """Round 6 P2: a blank explicit model names no vector space.

    On a legacy catalog (no recorded build model) with a permissive
    endpoint that accepts any model, ``--embed-model ''`` (or whitespace)
    must be a named parity refusal with zero embed requests, not a silent
    pass-through to the endpoint's own default.
    """
    endpoint, received = _multi_model_accepting_endpoint(["", "   ", "\t", "bge-m3"])
    try:
        paths = make_case(tmp_path)
        payload = json.loads(paths["catalog"].read_text(encoding="utf-8"))
        payload.pop("embed_model", None)
        paths["catalog"].write_text(json.dumps(payload), encoding="utf-8")

        result = run_script(endpoint, paths, embed_model=blank)
        assert result.returncode != 0, result.stdout
        assert "analytics parity error" in result.stderr, result.stderr
        assert "--embed-model" in result.stderr, result.stderr
        assert received == [], f"a blank model must refuse before embedding, got {received}"
    finally:
        endpoint.close()


def test_analytics_run_does_not_touch_warning_state(endpoint, tmp_path, monkeypatch):
    """Round 4 concern: the analytics process writes no persisted warning state.

    Initialize and acknowledge a disabled-switch warning under an isolated
    HOME, run the documented analytics command (no embed flags), then reload
    disabled: the warning stays suppressed instead of recurring.
    """
    import logging

    from hermes_skill_router.config import (
        acknowledge_switch_deliveries,
        load,
        switch_warning_messages,
    )

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_HOME", str(_hermes_home(tmp_path)))
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    settings = {"enabled": False, "cachePath": "", "suggestionsPath": ""}
    cfg = load(settings)
    notes = [
        note
        for note in cfg.notes
        if note.startswith(("enabled: ", "showSuggestions: ")) and " is off; " in note
    ]
    assert len(notes) == 1
    acknowledge_switch_deliveries(notes, "", cfg.switch_generations)
    assert switch_warning_messages(load(settings)) == ()

    env = {**__import__("os").environ, "HOME": str(tmp_path / "home"),
           "HERMES_HOME": str(tmp_path / "hermes-home")}
    paths = make_case(tmp_path, catalog_model=CUSTOM_MODEL)
    result = subprocess.run(
        _documented_command(endpoint, paths), capture_output=True, text=True, timeout=120, env=env
    )
    assert result.returncode == 0, result.stderr

    reloaded = load(settings)
    assert [n for n in reloaded.notes if " is off; " in n] == []
    assert switch_warning_messages(reloaded) == ()


def test_top5_prompts_per_skill_in_divergence_order(endpoint, tmp_path):
    """Top-5 is the five highest divergences, ties by query JSON ascending.

    Each query carries two unique terms shared with no other prompt, so one
    prompt's events cannot leak into another query's replay.
    """
    queries = [(T_LOG + index * 1000, f"n00{index + 1} m00{index + 1}") for index in range(7)]
    # Every event precedes every log line (its own included).
    prompts = [
        (index + 1, "s-1", f"n00{index + 1} m00{index + 1}", T_LOG - 10_000 - index * 1000)
        for index in range(7)
    ]
    beta_strength = {0: 1.25, 1: 2.0, 2: 2.0, 3: 1.0, 4: 1.5, 5: 1.1, 6: 0.0}
    events = []
    for index in range(7):
        events.append((index + 1, 1, 1.0, T_LOG - 10_000 - index * 1000 + 10, ""))
        if beta_strength[index] > 0:
            events.append(
                (index + 1, 2, beta_strength[index], T_LOG - 10_000 - index * 1000 + 10, "")
            )
    paths = make_case(
        tmp_path,
        queries=queries,
        store_kwargs={"prompt_events": prompts, "load_events": events},
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    alpha = anomalies_for(result.stdout)["alpha-skill"]
    assert [query for query, _value in alpha] == [
        "n004 m004", "n007 m007", "n006 m006", "n001 m001", "n005 m005",
    ]
    assert [value for _query, value in alpha] == pytest.approx(
        [1.0, 1.0, 1.0 / 1.1, 1.0 / 1.25, 1.0 / 1.5]
    )


def test_clean_fixture_prints_the_no_anomalies_line(endpoint, tmp_path):
    paths = make_case(tmp_path, store_kwargs={"load_events": (), "prompt_events": ()})
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no anomalies"


def test_threshold_is_one_named_constant_and_below_threshold_is_quiet(endpoint, tmp_path):
    script_text = SCRIPT.read_text(encoding="utf-8")
    assert script_text.count("DIVERGENCE_THRESHOLD = 0.4") == 1
    events = [
        (1, 1, 1.0, T_LOG - 3_000_000, ""),
        (1, 2, 0.3, T_LOG - 3_000_000, ""),
    ]
    paths = make_case(tmp_path, store_kwargs={"load_events": events})
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    # alpha diverges 1.0 (flagged); beta diverges 0.3, below the constant.
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_a_skill_the_shortlist_cut_is_still_scored(endpoint, tmp_path):
    """The log line's candidates omit alpha-skill; the script scores it anyway."""
    paths = make_case(tmp_path)
    paths["log"].write_text(
        decision_line(T_LOG, "quokka mammoth zebra", candidates="beta-skill:0.9"),
        encoding="utf-8",
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_output_is_deterministic(endpoint, tmp_path):
    paths = make_case(tmp_path)
    first = run_script(endpoint, paths)
    second = run_script(endpoint, paths)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


#--------------------------------------------------------------------------------------------------
# Catalog identity: by name, never by path
#--------------------------------------------------------------------------------------------------


def test_a_changed_stored_path_still_matches_by_name(endpoint, tmp_path):
    """A row whose path changed from ``hermes:<name>`` to a real path keeps its history."""
    paths = make_case(
        tmp_path,
        store_kwargs={
            "skills": (
                ("alpha-skill", "/real/path/alpha-skill/SKILL.md"),
                ("beta-skill", "hermes:beta-skill"),
            )
        },
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_a_name_with_no_store_row_scores_zero(endpoint, tmp_path):
    paths = make_case(
        tmp_path,
        catalog_names=("alpha-skill", "beta-skill", "never-loaded"),
        descriptions=("beta gamma delta epsilon", "eta theta iota kappa", "gamma delta"),
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_a_repeated_catalog_name_is_an_identity_error(endpoint, tmp_path):
    paths = make_case(
        tmp_path,
        catalog_names=("alpha-skill", "alpha-skill"),
        descriptions=("beta gamma", "delta epsilon"),
    )
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics identity error" in result.stderr, result.stderr
    assert "alpha-skill" in result.stderr


#--------------------------------------------------------------------------------------------------
# BM25 parity and the description cap
#--------------------------------------------------------------------------------------------------


def _script_module():
    spec = importlib.util.spec_from_file_location("analytics_scores", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bm25_constants_and_document_text_match_the_live_router():
    assert bm25.BM25_K1 == 1.2
    assert bm25.BM25_B == 0.75
    module = _script_module()
    cfg = types.SimpleNamespace(max_description_chars=64)
    assert module.document_text("n", "d" * 200, 64) == library.document_text(cfg, "n", "d" * 200)
    assert module.cosine([1.0, 0.0], [0.0, 2.0, 0.0]) == library._cosine(
        [1.0, 0.0], [0.0, 2.0, 0.0]
    )


def test_script_bm25_raw_scores_rank_like_score_ranks():
    module = _script_module()
    rows = [
        ("one", "one\nquokka mammoth zebra"),
        ("two", "two\nquokka"),
        ("three", "three\nmammoth"),
    ]
    index = bm25.build_term_index(rows)
    raw = module.bm25_raw_scores(index, "quokka mammoth zebra")
    order = sorted(
        (position for position, score in enumerate(raw) if score > 0),
        key=lambda position: -raw[position],
    )
    ranked = bm25.score_ranks(index, "quokka mammoth zebra", 10)
    assert [rows[position][0] for position in order] == ranked


def test_a_different_description_cap_changes_the_result(endpoint, tmp_path):
    """A query term living past char 400 flips the verdict with the cap."""
    description = ("filler " * 60) + "quokka"
    paths = make_case(
        tmp_path,
        descriptions=(description, "eta theta iota kappa"),
    )
    narrow = run_script(endpoint, paths, cap=400)
    wide = run_script(endpoint, paths, cap=4000)
    assert narrow.returncode == wide.returncode == 0
    assert set(anomalies_for(narrow.stdout)) == {"alpha-skill"}
    assert wide.stdout.strip() == "no anomalies"


#--------------------------------------------------------------------------------------------------
# Historical replay
#--------------------------------------------------------------------------------------------------


def test_an_event_after_the_log_timestamp_does_not_count(endpoint, tmp_path):
    """A future load is not history; the past-only replay scores it zero."""
    past_query, future_query = "quokka mammoth zebra", "guess who nothing"
    paths = make_case(
        tmp_path,
        queries=((T_LOG, past_query), (T_LOG + 1_000_000, future_query)),
        store_kwargs={
            # The second event is observed after BOTH log lines, so neither
            # replay may count it; the second query scores exactly zero.
            "prompt_events": [
                (1, "s-1", "quokka mammoth zebra", T_LOG - 1_000_000),
                (2, "s-1", future_query, T_LOG + 1_500_000),
            ],
            "load_events": [
                (1, 1, 1.0, T_LOG - 500_000, ""),
                (2, 1, 1.0, T_LOG + 1_600_000, ""),
            ],
        },
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    alpha = anomalies_for(result.stdout)["alpha-skill"]
    # The future prompt's score is exactly zero; only the past one diverges.
    assert alpha == [
        (past_query, pytest.approx(1.0)),
        (future_query, 0.0),
    ]


def test_skipped_query_rows_and_zero_maximum(endpoint, tmp_path):
    """Non-JSON-string query rows are skipped; a zero maximum maps all to 0."""
    paths = make_case(
        tmp_path,
        queries=((T_LOG, "quokka mammoth zebra"),),
        store_kwargs={"load_events": (), "prompt_events": ()},
    )
    raw = paths["log"].read_text(encoding="utf-8")
    raw += f"{iso(T_LOG)} INFO plugin skill router: query=notjson candidates=a:1\n"
    raw += f"{iso(T_LOG)} INFO plugin skill router: query=123 candidates=a:1\n"
    raw += "noise line without a decision\n"
    paths["log"].write_text(raw, encoding="utf-8")
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no anomalies"


def test_python_asctime_timestamps_parse(endpoint, tmp_path):
    """The asctime form (``YYYY-MM-DD HH:MM:SS,mmm``) is a valid timestamp."""
    paths = make_case(
        tmp_path,
        queries=((T_LOG, "quokka mammoth zebra"),),
    )
    stamp = datetime.fromtimestamp(T_LOG / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S,000"
    )
    paths["log"].write_text(
        f"{stamp} INFO hermes_skill_router.router skill router: "
        'query="quokka mammoth zebra" candidates=a:1\n',
        encoding="utf-8",
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


#--------------------------------------------------------------------------------------------------
# Coverage provenance: marker, pre-v2, enrolment signals
#--------------------------------------------------------------------------------------------------


def test_reset_marker_fails_even_after_a_reload(endpoint, tmp_path):
    """(a) A per-skill reset followed by a reload leaves the marker; fail."""
    paths = make_case(tmp_path, store_kwargs={"marker": str(T0 + 1000)})
    assert_coverage_error(run_script(endpoint, paths), "last_history_loss_ms")


def test_partial_reset_marker_fails(endpoint, tmp_path):
    """(b) One skill's events removed, another's remain, marker recorded; fail."""
    events = [(1, 2, 1.0, T_LOG - 3_000_000, "")]
    paths = make_case(
        tmp_path,
        store_kwargs={"load_events": events, "marker": str(T0 + 1000)},
    )
    assert_coverage_error(run_script(endpoint, paths), "last_history_loss_ms")


def test_maintenance_marker_fails(endpoint, tmp_path):
    """(c) Maintenance purged a prompt after a logged prompt; old marker; fail."""
    paths = make_case(tmp_path, store_kwargs={"marker": str(T0 - DAY)})
    assert_coverage_error(run_script(endpoint, paths), "last_history_loss_ms")


def test_marker_at_the_log_timestamp_still_fails(endpoint, tmp_path):
    """(d) Equality cannot order a reset after a same-millisecond prompt."""
    paths = make_case(tmp_path, store_kwargs={"marker": str(T_LOG)})
    assert_coverage_error(run_script(endpoint, paths), "last_history_loss_ms")


@pytest.mark.parametrize("skewed", ["915148800000", "32509900800000"])
def test_a_skewed_marker_value_still_fails(endpoint, tmp_path, skewed):
    paths = make_case(tmp_path, store_kwargs={"marker": skewed})
    assert_coverage_error(run_script(endpoint, paths), "last_history_loss_ms")


def test_pre_v2_store_fails_with_the_pre_enrollment_error(endpoint, tmp_path):
    """(f) schema version 1, no marker, no enrollment record."""
    paths = make_case(
        tmp_path,
        store_kwargs={
            "user_version": 1,
            "schema_version": "1",
            "creation_origin": None,
        },
    )
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "pre-v2" in result.stderr


def test_version_mismatch_fails_and_never_passes(endpoint, tmp_path):
    paths = make_case(
        tmp_path,
        store_kwargs={"user_version": 2, "schema_version": "1"},
    )
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "mismatch" in result.stderr


def test_a_marker_on_a_v1_store_names_the_marker_not_pre_v2(endpoint, tmp_path):
    paths = make_case(
        tmp_path,
        store_kwargs={
            "user_version": 1,
            "schema_version": "1",
            "creation_origin": None,
            "marker": str(T0),
        },
    )
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "last_history_loss_ms" in result.stderr
    assert "pre-v2" not in result.stderr


def test_a_migrated_store_with_a_deleted_marker_still_fails(endpoint, tmp_path):
    """No fresh creation record: provenance is untrustworthy even sans marker."""
    paths = make_case(tmp_path, store_kwargs={"creation_origin": None})
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "provenance untrustworthy" in result.stderr


def test_never_loaded_skill_with_no_marker_scores_zero(endpoint, tmp_path):
    """(e) Verified coverage and no matching events: zero, run succeeds."""
    paths = make_case(
        tmp_path,
        catalog_names=("alpha-skill", "beta-skill", "never-loaded"),
        descriptions=("beta gamma delta epsilon", "eta theta iota kappa", "gamma delta"),
        store_kwargs={"load_events": (), "prompt_events": ()},
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no anomalies"


#--------------------------------------------------------------------------------------------------
# Source finality: ledger counters, empty scans, ids, later activity
#--------------------------------------------------------------------------------------------------


def test_candidate_without_a_ledger_row_fails_even_with_a_newer_watermark(endpoint, tmp_path):
    """(g) ``meta.last_scan_ms`` never gates replay."""
    paths = make_case(
        tmp_path,
        store_kwargs={
            "ledger": [],
            "meta_extra": {"last_scan_ms": str(T_LOG + 10_000_000)},
        },
    )
    assert_coverage_error(run_script(endpoint, paths), "scan_ledger")


def test_pending_calls_fail(endpoint, tmp_path):
    """(h) A scan that ended with an in-flight tool call is not final."""
    paths = make_case(tmp_path, store_kwargs={"ledger": [("s-1", 2, 1, 0, 0)]})
    assert_coverage_error(run_script(endpoint, paths), "pending_calls")


def test_unreadable_calls_fail(endpoint, tmp_path):
    """(i) A scan that could not classify a relevant call is not final."""
    paths = make_case(tmp_path, store_kwargs={"ledger": [("s-1", 2, 0, 0, 3)]})
    assert_coverage_error(run_script(endpoint, paths), "unreadable_calls")


def test_unreadable_results_fail(endpoint, tmp_path):
    """(i-bis) A scan that could not parse a relevant result is not final."""
    paths = make_case(tmp_path, store_kwargs={"ledger": [("s-1", 2, 0, 2, 0)]})
    assert_coverage_error(run_script(endpoint, paths), "unreadable_results")


def test_later_activity_with_a_pending_row_fails(endpoint, tmp_path):
    """(j) A call predates the log, later activity follows it, row pending."""
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T_LOG + 1_000_000)],
        store_kwargs={"ledger": [("s-1", 2, 1, 0, 0)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "pending_calls")


def test_zero_or_negative_message_ids_fail(endpoint, tmp_path):
    """(j-bis) Non-positive ids are unsupported, scanned max notwithstanding."""
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[
            (0, "s-1", "user", T0, None, None, None),
            (5, "s-1", "user", T0 + 100, None, None, None),
        ],
        store_kwargs={"ledger": [("s-1", 5, 0, 0, 0)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "non-positive")


def test_empty_scan_with_a_zero_id_message_fails(endpoint, tmp_path):
    """(j-bis) An empty scan is not a sentinel: any later row invalidates it."""
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[(0, "s-1", "user", T0, None, None, None)],
        store_kwargs={"ledger": [("s-1", None, 0, 0, 0)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "empty scan")


def test_message_less_empty_scan_passes_while_current_and_after_deletion(endpoint, tmp_path):
    """The passing empty-scan cases: current, then deleted with no messages."""
    current = make_case(
        tmp_path / "current",
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[],
        store_kwargs={"ledger": [("s-1", None, 0, 0, 0)]},
    )
    assert run_script(endpoint, current).returncode == 0
    deleted = make_case(
        tmp_path / "deleted",
        sessions=[],
        messages=[],
        store_kwargs={"ledger": [("s-1", None, 0, 0, 0)]},
    )
    assert run_script(endpoint, deleted).returncode == 0


def test_truncated_call_plus_valid_result_fails(endpoint, tmp_path):
    """Malformed ``tool_calls`` then its skill result: both unreadable counts > 0."""
    truncated = json.dumps([{"id": "call-1", "type": "function", "function": {"name": "sk"}}])
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[
            (1, "s-1", "assistant", T0, truncated, None, None),
            (2, "s-1", "tool", T0 + 100, None, "call-1", "skill_view"),
        ],
        store_kwargs={"ledger": [("s-1", 2, 0, 1, 1)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "unreadable")


def test_truncated_payload_only_fails(endpoint, tmp_path):
    """A truncated ``tool_calls`` payload with no following result still fails."""
    truncated = json.dumps([{"id": "call-1", "function": {"name": "sk"}}])
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[(1, "s-1", "assistant", T0, truncated, None, None)],
        store_kwargs={"ledger": [("s-1", 1, 0, 0, 1)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "unreadable_calls")


def test_missing_call_id_fails(endpoint, tmp_path):
    """A well-formed ``skill_view`` call with no call id records unreadable_calls."""
    payload = json.dumps(
        [{"call_id": None, "type": "function", "function": {"name": "skill_view", "arguments": json.dumps({"name": "alpha-skill"})}}]
    )
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[(1, "s-1", "assistant", T0, payload, None, None)],
        store_kwargs={"ledger": [("s-1", 1, 0, 0, 1)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "unreadable_calls")


def test_orphan_result_fails(endpoint, tmp_path):
    """A result matching no parsed call records unreadable_results."""
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[(1, "s-1", "tool", T0, None, "call-ghost", "skill_view")],
        store_kwargs={"ledger": [("s-1", 1, 0, 1, 0)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "unreadable_results")


def test_bad_skill_view_arguments_fail(endpoint, tmp_path):
    """A ``skill_view`` call with unparseable arguments records unreadable_calls."""
    payload = json.dumps(
        [{"id": "call-1", "type": "function", "function": {"name": "skill_view", "arguments": "{bad json"}}]
    )
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[(1, "s-1", "assistant", T0, payload, None, None)],
        store_kwargs={"ledger": [("s-1", 1, 0, 0, 1)]},
    )
    assert_coverage_error(run_script(endpoint, paths), "unreadable_calls")


def test_valid_calls_pass_coverage(endpoint, tmp_path):
    """Controls: a matched name-only skill_view and a matched unrelated call."""
    skill_call = json.dumps(
        [{"id": "call-1", "type": "function", "function": {"name": "skill_view", "arguments": json.dumps({"name": "alpha-skill"})}}]
    )
    read_call = json.dumps(
        [{"id": "call-2", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "/x"})}}]
    )
    paths = make_case(
        tmp_path,
        sessions=[("s-valid", T0, T0 + 900_000), ("s-other", T0, T0 + 900_000)],
        messages=[
            (1, "s-valid", "assistant", T0, skill_call, None, None),
            (2, "s-valid", "tool", T0 + 100, None, "call-1", "skill_view"),
            (3, "s-other", "assistant", T0, read_call, None, None),
            (4, "s-other", "tool", T0 + 100, None, "call-2", "read_file"),
        ],
        store_kwargs={
            "ledger": [("s-1", 2, 0, 0, 0), ("s-valid", 2, 0, 0, 0), ("s-other", 4, 0, 0, 0)],
            "seen": [("s-1", T0), ("s-valid", T0), ("s-other", T0)],
        },
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_pending_then_incorporated_scan_scores_normally(endpoint, tmp_path):
    """The finality flow: pending scan fails; the later scan covers the result."""
    paths = make_case(tmp_path, store_kwargs={"ledger": [("s-1", 2, 1, 0, 0)]})
    stale = run_script(endpoint, paths)
    assert_coverage_error(stale, "pending_calls")
    # The result lands in the source; a later scan incorporates it.
    conn = sqlite3.connect(str(paths["state"]))
    conn.execute(
        "INSERT INTO messages (id, session_id, role, timestamp) VALUES (3, 's-1', 'tool', ?)",
        ((T_LOG - 1_000_000) / 1000.0,),
    )
    conn.commit()
    conn.close()
    store = sqlite3.connect(str(paths["store"]))
    store.execute(
        "UPDATE scan_ledger SET pending_calls = 0, scanned_max_row_id = 3 WHERE session_id = 's-1'"
    )
    store.commit()
    store.close()
    fresh = run_script(endpoint, paths)
    assert fresh.returncode == 0, fresh.stderr
    assert set(anomalies_for(fresh.stdout)) == {"alpha-skill"}


def test_a_too_old_copy_before_later_activity_fails(endpoint, tmp_path):
    paths = make_case(tmp_path)
    conn = sqlite3.connect(str(paths["state"]))
    conn.execute(
        "INSERT INTO messages (id, session_id, role, timestamp) VALUES (3, 's-1', 'tool', ?)",
        ((T_LOG + 1_000_000) / 1000.0,),
    )
    conn.commit()
    conn.close()
    assert_coverage_error(run_script(endpoint, paths), "after the scan")


def test_same_millisecond_append_fails_on_the_id_check(endpoint, tmp_path):
    """A larger id with an equal timestamp still exceeds the scanned max."""
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[
            (1, "s-1", "user", T0, None, None, None),
            (2, "s-1", "user", T0, None, None, None),
        ],
        store_kwargs={"ledger": [("s-1", 1, 0, 0, 0)]},
    )
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "after the scan" in result.stderr


#--------------------------------------------------------------------------------------------------
# Source deletion and the copy protocol
#--------------------------------------------------------------------------------------------------


def test_seen_session_deleted_before_any_scan_fails(endpoint, tmp_path):
    """(k) Listed, then deleted before any scan: ``seen_sessions`` without a ledger row."""
    paths = make_case(
        tmp_path,
        sessions=[],
        messages=[],
        store_kwargs={"ledger": []},
    )
    assert_coverage_error(run_script(endpoint, paths), "scan_ledger")


def test_missing_assertion_fails_even_when_every_candidate_passes(endpoint, tmp_path):
    """(l) The operator assertion is mandatory on an otherwise clean copy."""
    paths = make_case(tmp_path)
    result = run_script(endpoint, paths, assert_history=False)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "--assert-complete-source-history" in result.stderr


def test_assertion_text_covers_pre_enrollment_deletions(endpoint):
    """(n) The assertion flag's help names deletions before v2 enrollment."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "including deletions before v2 enrollment" in result.stdout


def test_scanned_then_deleted_session_passes(endpoint, tmp_path):
    """(m) A scanned session deleted from the source keeps its ledger row."""
    paths = make_case(
        tmp_path,
        sessions=[],
        messages=[],
        store_kwargs={"ledger": [("s-1", 2, 0, 0, 0)]},
    )
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_non_maximum_message_deletion_is_outside_the_check(endpoint, tmp_path):
    """(o) Id and max are unchanged, so the check cannot see the deletion.

    The run's coverage rests on the operator assertion; this fixture
    documents that trust boundary.
    """
    paths = make_case(
        tmp_path,
        sessions=[("s-1", T0, T0 + 900_000)],
        messages=[
            (1, "s-1", "user", T0, None, None, None),
            (2, "s-1", "user", T0 + 100, None, None, None),
            (3, "s-1", "user", T0 + 200, None, None, None),
        ],
        store_kwargs={"ledger": [("s-1", 3, 0, 0, 0)]},
    )
    conn = sqlite3.connect(str(paths["state"]))
    conn.execute("DELETE FROM messages WHERE id = 2")
    conn.commit()
    conn.close()
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr


def test_read_only_run_never_writes_the_store(endpoint, tmp_path):
    """The store file is byte-identical after a successful run."""
    paths = make_case(tmp_path)
    before = hashlib.sha256(paths["store"].read_bytes()).hexdigest()
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    after = hashlib.sha256(paths["store"].read_bytes()).hexdigest()
    assert before == after
    assert not list(paths["store"].parent.glob("associations.db-*"))


def test_backup_copy_includes_uncheckpointed_rows(endpoint, tmp_path):
    """A copy through the SQLite backup API sees WAL rows; a file copy would not."""
    paths = make_case(tmp_path)
    # The live store is built in WAL mode with its writer still open, so the
    # committed rows live only in -wal (no checkpoint has run).
    conn = make_store(
        tmp_path / "live.db",
        wal=True,
        prompt_events=HAPPY_PROMPTS,
        load_events=HAPPY_EVENTS,
        ledger=[("s-1", 2, 0, 0, 0)],
        seen=[("s-1", T0)],
    )
    destination = tmp_path / "backup-copy.db"
    dest = sqlite3.connect(str(destination))
    conn.backup(dest)
    dest.close()
    assert (tmp_path / "live.db-wal").stat().st_size > 0, (
        "the fixture needs uncheckpointed WAL rows"
    )
    result = run_script(endpoint, {**paths, "store": destination})
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}
    conn.close()


def test_pre_v2_backup_copy_is_rejected(endpoint, tmp_path):
    paths = make_case(
        tmp_path,
        store_kwargs={"user_version": 1, "schema_version": "1", "creation_origin": None},
    )
    source = sqlite3.connect(str(paths["store"]))
    destination = tmp_path / "v1-backup.db"
    dest = sqlite3.connect(str(destination))
    source.backup(dest)
    dest.close()
    source.close()
    result = run_script(endpoint, {**paths, "store": destination})
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "pre-v2" in result.stderr


def test_unreachable_embed_endpoint_is_a_named_error(endpoint, tmp_path):
    paths = make_case(tmp_path)
    result = run_script(endpoint, paths, embed_url="http://127.0.0.1:1/v1/embeddings")
    assert result.returncode != 0
    assert "analytics embed error" in result.stderr
    assert "127.0.0.1:1" in result.stderr


def test_foreign_harness_v2_store_fails_provenance(endpoint, tmp_path):
    """A fresh v2 store from another harness fails before any scoring."""
    paths = make_case(tmp_path, store_kwargs={"meta_extra": {"harness": "other-harness"}})
    result = run_script(endpoint, paths)
    assert result.returncode != 0
    assert "analytics coverage error" in result.stderr
    assert "foreign harness" in result.stderr


def test_hermes_harness_store_passes_provenance_control(endpoint, tmp_path):
    """Control: the same store with this harness's stamp scores normally."""
    paths = make_case(tmp_path, store_kwargs={"meta_extra": {"harness": "hermes"}})
    result = run_script(endpoint, paths)
    assert result.returncode == 0, result.stderr
    assert set(anomalies_for(result.stdout)) == {"alpha-skill"}


def test_an_endpoint_rejects_a_mismatched_embed_model(tmp_path):
    """A replay model other than the catalog's recorded build model is a parity
    error before the endpoint is ever called; the recorded model replays."""

    def picky(request):
        payload = request.json()
        if payload.get("model") != "run-model":
            return Response(status=400, body={"error": "unknown model"})
        vectors = [embed_vector(text) for text in payload["input"]]
        return Response(body={"data": [{"embedding": vector} for vector in vectors]})

    server = FakeEndpoint(responder=picky)
    try:
        paths = make_case(tmp_path, catalog_model="run-model")
        matched = run_script(server, paths, embed_model="run-model")
        assert matched.returncode == 0, matched.stderr
        assert set(anomalies_for(matched.stdout)) == {"alpha-skill"}
        mismatched = run_script(server, paths, embed_model="other-model")
        assert mismatched.returncode != 0
        assert "analytics parity error" in mismatched.stderr, mismatched.stderr
    finally:
        server.close()


def test_a_slow_but_valid_endpoint_needs_the_run_timeout(tmp_path):
    """A 2s response succeeds within the run's 15s budget and fails within 500 ms."""

    def slow(request):
        payload = request.json()
        vectors = [embed_vector(text) for text in payload["input"]]
        return Response(body={"data": [{"embedding": vector} for vector in vectors]}, delay=2.0)

    server = FakeEndpoint(responder=slow)
    try:
        paths = make_case(tmp_path)
        patient = run_script(server, paths, embed_timeout_ms=15000)
        assert patient.returncode == 0, patient.stderr
        assert set(anomalies_for(patient.stdout)) == {"alpha-skill"}
        impatient = run_script(server, paths, embed_timeout_ms=500)
        assert impatient.returncode != 0
        assert "analytics embed error" in impatient.stderr, impatient.stderr
    finally:
        server.close()
