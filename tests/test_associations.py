"""T4: the Hermes association store, the gate and the backfill.

The DSH implementation (``src/associations/{store,gate,backfill}.ts``) defines the behaviour; this
suite restates its cases against Hermes and adds the two Hermes-only rules the ticket names: the
``skill_view`` load path with its ``file=`` follow-on prerequisite, and the automation provenance
flag that keeps 84% of the store while a subagent session contributes nothing.

Every database here is a temporary one under the directory pytest owns, and every clock is a fixed
epoch-millisecond constant, so a result never depends on when the suite runs. The Hermes session
store is a fixture SQLite file with the two tables the backfill reads; the one test that reads the
live store states its own bound.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import associations as associations_module  # noqa: E402
from hermes_skill_router.associations import (  # noqa: E402
    ASSOCIATION_DB_NAME,
    ASSOCIATION_HALF_LIFE_MS,
    AssociationEvent,
    AssociationStoreError,
    CatalogSkill,
    EXPLICIT_STRENGTH,
    FOLLOW_ON_STRENGTH,
    GATE_CONDITIONS,
    GateContext,
    LoadEvent,
    META_CATALOG_FINGERPRINT,
    META_HARNESS,
    META_LAST_SCAN_MS,
    META_SCHEMA_VERSION,
    PromptEventInput,
    PromptEvidence,
    SUBSTANTIVE_MIN_CHARS,
    UBIQUITY_THRESHOLD,
    ambient_skill_ids,
    association_db_path_for,
    association_stats,
    backfill,
    catalog_path,
    decay_factor,
    decayed_weights,
    default_association_db_path,
    default_state_db_path,
    delete_association_database,
    gate_rejection,
    is_ambient,
    is_continuation,
    is_eligible_load,
    is_eligible_prompt,
    maintenance,
    normalise_terms,
    open_store,
    read_meta,
    recompute_associations,
    reconcile_catalog,
    reset_associations,
    resolve_skill_name,
    resolve_skill_path,
    upsert_events,
    upsert_prompts,
    write_meta,
)

T0 = 1_700_000_000_000
DAY = 86_400_000
SUBSTANTIVE = "Review the association store and its migration path"

CATALOG: Tuple[CatalogSkill, ...] = (
    CatalogSkill("code-review"),
    CatalogSkill("handoff-doc"),
    CatalogSkill("writing"),
)


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------


def eligible_event(**overrides):
    """Return an event that passes every condition, overridden per test."""
    base = LoadEvent(
        evidence="tool",
        strength=EXPLICIT_STRENGTH,
        skill_id=7,
        prompt=PromptEvidence(text=SUBSTANTIVE, source_kind="user", index=3),
        index=9,
        nearest_prompt_index=3,
        skill_path=catalog_path("code-review"),
        origin="human",
    )
    return replace(base, **overrides)


def temp_store(tmp_path: Path, label: str = "associations"):
    """Open one temporary association database, never reused across tests."""
    return open_store(str(tmp_path / f"{label}.db"))


def prompt_of(i: int, **overrides) -> PromptEventInput:
    """Return one typed prompt fixture: session id, turn, text and clock all derived from *i*."""
    values = dict(
        session_id=f"session-{i}",
        turn=1,
        text=f"Review the association store implementation case {i}",
        observed_ms=T0 + i * DAY,
    )
    values.update(overrides)
    return PromptEventInput(**values)


def load_of(i: int, **overrides) -> AssociationEvent:
    """Return one load fixture for a prompt, defaulting to the code-review skill."""
    prompt = prompt_of(i)
    values = dict(
        session_id=prompt.session_id,
        turn=prompt.turn,
        prompt_text=prompt.text,
        prompt_observed_ms=prompt.observed_ms,
        skill_name="code-review",
        skill_path=catalog_path("code-review"),
        evidence="tool",
        strength=EXPLICIT_STRENGTH,
        origin="human",
        source="backfill",
        observed_ms=prompt.observed_ms + 1000,
    )
    values.update(overrides)
    return AssociationEvent(**values)


def seed_prompts(store, count: int) -> int:
    """Store *count* eligible prompts, none of which loads anything."""
    return upsert_prompts(store, [prompt_of(i) for i in range(count)])


def rows(store, sql: str, params: Sequence = ()):
    """Return plain tuples, so a comparison never trips on sqlite3.Row."""
    return [tuple(row) for row in store.conn.execute(sql, tuple(params)).fetchall()]


def zero_rejections():
    """Return the per-condition rejection counters the backfill reports."""
    return {condition: 0 for condition in GATE_CONDITIONS}


class HermesStoreFixture:
    """A temporary Hermes session store with the two tables the backfill reads.

    The column set is the live store's, narrowed to what the backfill selects, so the fixture
    exercises the real query rather than a convenient shape.
    """

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

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(self.SCHEMA)
        self._calls = 0

    def session(self, session_id: str, source: str = "cli", started_ms: int = T0) -> None:
        self.conn.execute(
            "INSERT INTO sessions (id, source, started_at, last_activity_at) VALUES (?, ?, ?, ?)",
            (session_id, source, started_ms / 1000.0, (started_ms + 900_000) / 1000.0),
        )

    def user(
        self,
        session_id: str,
        text: str,
        ms: int,
        *,
        display_kind: Optional[str] = None,
        compacted: int = 0,
        active: int = 1,
    ) -> None:
        self.conn.execute(
            """INSERT INTO messages
               (session_id, role, content, timestamp, display_kind, compacted, active)
               VALUES (?, 'user', ?, ?, ?, ?, ?)""",
            (session_id, text, ms / 1000.0, display_kind, compacted, active),
        )

    def calls(self, session_id: str, calls: Sequence[Tuple[str, dict]], ms: int) -> List[str]:
        """Store one assistant row with *calls* and return their call ids."""
        ids = []
        payload = []
        for name, arguments in calls:
            self._calls += 1
            call_id = f"call-{self._calls}"
            ids.append(call_id)
            payload.append(
                {
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, 'assistant', '', ?, ?)",
            (session_id, json.dumps(payload), ms / 1000.0),
        )
        return ids

    def result(
        self, session_id: str, call_id: str, tool_name: str, content: str, ms: int
    ) -> None:
        self.conn.execute(
            """INSERT INTO messages
               (session_id, role, content, tool_call_id, tool_name, timestamp)
               VALUES (?, 'tool', ?, ?, ?, ?)""",
            (session_id, content, call_id, tool_name, ms / 1000.0),
        )

    def skill_view(self, session_id: str, name: str, ms: int, *, file_path=None) -> None:
        """Store one ``skill_view`` call and its JSON result."""
        arguments = {"name": name}
        if file_path is not None:
            arguments["file_path"] = file_path
        call_id = self.calls(session_id, [("skill_view", arguments)], ms)[0]
        payload = {
            "success": True,
            "name": name,
            "_source_path": f"/home/wilder/.hermes/skills/{name}/SKILL.md",
            "path": f"{name}/SKILL.md",
        }
        if file_path is not None:
            payload["file"] = file_path
        self.result(session_id, call_id, "skill_view", json.dumps(payload), ms + 100)

    def read_file(self, session_id: str, file_path: str, ms: int) -> None:
        """Store one ``read_file`` call and a text result, with the live store's ``path`` key."""
        call_id = self.calls(session_id, [("read_file", {"path": file_path})], ms)[0]
        self.result(session_id, call_id, "read_file", "# loaded\n", ms + 100)

    def read_reference(self, session_id: str, file_path: str, ms: int) -> None:
        """Store one ``read_file`` of a file under a skill's ``references/`` directory."""
        self.read_file(session_id, file_path, ms)

    def raw_messages(self, session_id: str, tool_calls: str, ms: int) -> None:
        """Store one assistant row with a raw ``tool_calls`` string, for a malformed-record test."""
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) VALUES (?, 'assistant', '', ?, ?)",
            (session_id, tool_calls, ms / 1000.0),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


@pytest.fixture
def hermes(tmp_path: Path) -> Iterable[HermesStoreFixture]:
    """A temporary Hermes session store this suite owns."""
    fixture = HermesStoreFixture(tmp_path / "state.db")
    yield fixture
    fixture.close()


def backfilled(store, hermes_path: Path, **overrides):
    """Run one backfill against the fixture store with the standard catalog."""
    values = dict(catalog=CATALOG, now_ms=T0 + 10 * DAY)
    values.update(overrides)
    return backfill(store, str(hermes_path), **values)


def load_rows(store):
    """Return every load with its skill, prompt and provenance, oldest first."""
    return [
        dict(row)
        for row in store.conn.execute(
            """SELECT le.evidence, le.strength, le.origin, le.source, s.name AS skill_name,
                      pe.session_id, pe.turn, pe.terms
               FROM load_event le
               JOIN skill s ON s.id = le.skill_id
               JOIN prompt_event pe ON pe.id = le.prompt_event_id
               ORDER BY le.id"""
        ).fetchall()
    ]


def _read_file_skill_md_paths(raw: str) -> List[str]:
    """Return the ``SKILL.md`` paths of one raw ``tool_calls`` column, or none when it is malformed."""
    try:
        calls = json.loads(raw) if raw else []
    except ValueError:
        return []
    paths: List[str] = []
    for call in calls or ():
        function = (call or {}).get("function") or {}
        if function.get("name") != "read_file":
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            continue
        path = arguments.get("path") if isinstance(arguments, dict) else None
        if isinstance(path, str) and path.endswith("/SKILL.md"):
            paths.append(path)
    return paths


def live_skill_md_read_paths(state_path: Path) -> List[str]:
    """Return the paths of the live store's ``read_file`` calls for a ``SKILL.md``.

    Hermes writes the argument as ``path``; this reads the live file in place, read-only, and never
    copies it. The live test uses it to check that the backfill recognises the shape Hermes really
    produces rather than the one a fixture finds convenient.
    """
    conn = sqlite3.connect(pathlib.Path(state_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        paths: List[str] = []
        for (raw,) in conn.execute(
            "SELECT tool_calls FROM messages WHERE tool_calls LIKE '%read_file%'"
        ):
            paths.extend(_read_file_skill_md_paths(raw))
        return paths
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------------
# The gate: one accepted and one rejected fixture per condition
# --------------------------------------------------------------------------------------------------


def test_the_gate_has_six_conditions_in_the_design_order():
    assert GATE_CONDITIONS == (
        "explicit-load",
        "typed-prompt",
        "substantive-prompt",
        "same-boundary",
        "session-provenance",
        "ubiquity",
    )


def test_g1_explicit_load_accepts_the_three_evidence_kinds_and_refuses_an_invented_one():
    for evidence in ("tool", "slash", "skill_md_read"):
        assert is_eligible_load(eligible_event(evidence=evidence)) is True
    assert gate_rejection(eligible_event(evidence="assistant-inference")) == "explicit-load"
    assert gate_rejection(eligible_event(skill_id=None)) == "explicit-load"
    assert gate_rejection(eligible_event(skill_id=0)) == "explicit-load"


def test_g1_a_skill_view_with_file_counts_only_after_the_same_skill_md_in_that_session():
    orphan = eligible_event(evidence="tool", strength=FOLLOW_ON_STRENGTH, after_skill_md=False)
    assert gate_rejection(orphan) == "explicit-load", "a reference read with no earlier SKILL.md is not a load"
    followed = eligible_event(evidence="tool", strength=FOLLOW_ON_STRENGTH, after_skill_md=True)
    assert is_eligible_load(followed) is True, "the follow-on read is eligible at the lower strength"
    slash = eligible_event(evidence="slash", strength=FOLLOW_ON_STRENGTH, after_skill_md=True)
    assert gate_rejection(slash) == "explicit-load", "a typed /name is never a follow-on read"
    skill_md = eligible_event(evidence="skill_md_read", strength=FOLLOW_ON_STRENGTH, after_skill_md=True)
    assert gate_rejection(skill_md) == "explicit-load", "a SKILL.md read is never a follow-on read"


def test_g2_typed_prompt_accepts_a_user_row_and_refuses_an_injected_one():
    assert is_eligible_load(eligible_event()) is True
    injected = eligible_event(
        prompt=PromptEvidence(text=SUBSTANTIVE, source_kind="skill-injection", index=3)
    )
    assert gate_rejection(injected) == "typed-prompt", "a skill body the harness injected is not the user"
    for source_kind in ("compacted", "steer", "process_complete", "async_delegation_complete"):
        event = eligible_event(prompt=PromptEvidence(text=SUBSTANTIVE, source_kind=source_kind, index=3))
        assert gate_rejection(event) == "typed-prompt", f"{source_kind} is not the user's typing"
    assert gate_rejection(eligible_event(prompt=None)) == "typed-prompt", "a load with no prompt is unattributable"


def test_g2_a_system_marker_is_not_a_prompt_even_when_its_kind_says_user():
    reminder = eligible_event(
        prompt=PromptEvidence(
            text="<system-reminder>Injected instruction the harness wrote</system-reminder>",
            source_kind="user",
            index=3,
        )
    )
    assert gate_rejection(reminder) == "typed-prompt"
    oob = eligible_event(
        prompt=PromptEvidence(
            text="[OUT-OF-BAND USER MESSAGE - a direct message from the user]\nfix the parser\n[/OUT-OF-BAND USER MESSAGE]",
            source_kind="user",
            index=3,
        )
    )
    assert gate_rejection(oob) == "typed-prompt", "the steer wrapper is harness text, not the prompt"


def test_g3_substantive_prompt_needs_twenty_characters_and_three_terms():
    assert len(SUBSTANTIVE) >= SUBSTANTIVE_MIN_CHARS
    short = eligible_event(prompt=PromptEvidence(text="fix it now", source_kind="user", index=3))
    assert gate_rejection(short) == "substantive-prompt"
    two_terms = eligible_event(
        prompt=PromptEvidence(text="Review associationstore", source_kind="user", index=3)
    )
    assert len(normalise_terms(two_terms.prompt.text)) == 2
    assert gate_rejection(two_terms) == "substantive-prompt"
    good = eligible_event(
        prompt=PromptEvidence(text="Please review the thing carefully", source_kind="user", index=3)
    )
    assert gate_rejection(good) is None


def test_g3_every_continuation_phrase_is_refused_and_a_task_that_starts_with_one_is_not():
    for text in ("continue", "yes", "go on", "ok", "do it", "proceed", "next", "please continue"):
        assert is_continuation(text) is True, f"{text} is a continuation"
        event = eligible_event(prompt=PromptEvidence(text=text, source_kind="user", index=3))
        assert gate_rejection(event) == "substantive-prompt", f"{text} must not become evidence"
    task = "Continue the migration and add the association tests"
    assert is_continuation(task) is False
    assert is_eligible_load(eligible_event(prompt=PromptEvidence(text=task, source_kind="user", index=3)))


def test_g4_the_hermes_boundary_walks_back_to_the_nearest_preceding_user_message():
    assert is_eligible_load(eligible_event()) is True
    skipped_closer = eligible_event(
        prompt=PromptEvidence(text=SUBSTANTIVE, source_kind="user", index=2),
        index=9,
        nearest_prompt_index=4,
    )
    assert gate_rejection(skipped_closer) == "same-boundary", "the nearest preceding prompt is the only boundary"
    later_prompt = eligible_event(
        prompt=PromptEvidence(text=SUBSTANTIVE, source_kind="user", index=12),
        index=9,
        nearest_prompt_index=12,
    )
    assert gate_rejection(later_prompt) == "same-boundary", "a load is never paired with a prompt that follows it"
    assert gate_rejection(eligible_event(nearest_prompt_index=None)) == "same-boundary"
    assert gate_rejection(eligible_event(index=None)) == "same-boundary"


def test_g5_provenance_keeps_automation_and_refuses_a_subagent_session():
    assert is_eligible_load(eligible_event(origin="human")) is True
    assert is_eligible_load(eligible_event(origin="automation")) is True, "automation is kept and flagged"
    assert is_eligible_load(eligible_event(origin=None)) is True, "unattributed counts as human"
    assert gate_rejection(eligible_event(origin="subagent")) == "session-provenance"
    assert gate_rejection(eligible_event(origin="imported")) == "session-provenance", "unknown provenance fails closed"


def test_g6_ubiquity_excludes_a_skill_loaded_by_more_than_a_quarter_of_prompts():
    assert UBIQUITY_THRESHOLD == 0.25
    assert is_eligible_load(eligible_event(), GateContext(skill_prompts=2, eligible_prompts=10)) is True
    assert (
        gate_rejection(eligible_event(), GateContext(skill_prompts=3, eligible_prompts=10)) == "ubiquity"
    )
    assert is_ambient(1, 4) is False, "exactly 25% does not exceed the threshold"
    assert is_ambient(2, 8) is False
    assert is_ambient(1, 0) is False, "an empty window carries no evidence either way"
    assert is_ambient(4, 4) is True


def test_g6_a_malformed_ubiquity_count_fails_closed():
    assert gate_rejection(eligible_event(), GateContext(skill_prompts=float("nan"), eligible_prompts=10)) == "ubiquity"
    assert gate_rejection(eligible_event(), GateContext(skill_prompts=3, eligible_prompts=None)) == "ubiquity"
    assert gate_rejection(eligible_event(), GateContext(skill_prompts=3)) == "ubiquity", "half a context is not measurable"
    assert gate_rejection(eligible_event(), GateContext()) is None, "no context at all is condition 6 deferred"


def test_the_first_failing_condition_is_the_one_reported():
    everything_wrong = LoadEvent(
        evidence="not-evidence",
        strength=0.0,
        skill_id=None,
        prompt=None,
        index=None,
        nearest_prompt_index=None,
        origin="subagent",
    )
    assert gate_rejection(everything_wrong) == "explicit-load"
    assert gate_rejection(eligible_event(prompt=None, origin="subagent")) == "typed-prompt"
    assert gate_rejection(eligible_event(origin="subagent")) == "session-provenance"


def test_a_malformed_event_is_rejected_rather_than_thrown():
    for event in (None, 42, "load", [], {}, {"prompt": 5}):
        assert gate_rejection(event) is not None
        assert is_eligible_load(event) is False
    bad_prompt = eligible_event(prompt=PromptEvidence(text=42, source_kind="user", index=3))
    assert gate_rejection(bad_prompt) == "typed-prompt"
    assert is_eligible_prompt(None) is False
    assert is_eligible_prompt(PromptEvidence(text=SUBSTANTIVE, source_kind="skill-injection")) is False


def test_normalise_terms_is_the_privacy_boundary():
    terms = normalise_terms("Review the Association-Store,  and its MIGRATION path!! (T4)")
    assert terms == ["review", "association", "store", "migration", "path", "t4"]
    assert all(term.isalnum() for term in terms)
    assert normalise_terms("a b c d") == []
    assert normalise_terms("Người dùng tải ảnh lên") == ["người", "dùng", "tải", "ảnh", "lên"]
    assert normalise_terms("Build build BUILD") == ["build", "build", "build"], "no stemming"


# --------------------------------------------------------------------------------------------------
# The store: schema, harness isolation, the write path, recompute and reset
# --------------------------------------------------------------------------------------------------


def test_the_store_opens_with_schema_one_wal_and_meta_harness_hermes(tmp_path: Path):
    store = temp_store(tmp_path, "open")
    journal = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy = store.conn.execute("PRAGMA busy_timeout").fetchone()[0]
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert store.harness == "hermes"
    assert store.read_only is False
    assert journal == "wal", "write-ahead logging is on"
    assert int(busy) == 5000, "one writer waits five seconds for the other"
    assert int(version) == 1, "schema version 1 is stamped in the file"
    assert read_meta(store, META_HARNESS) == "hermes"
    assert read_meta(store, META_SCHEMA_VERSION) == "1"
    assert read_meta(store, META_LAST_SCAN_MS) == "0"
    tables = [row[0] for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    assert [name for name in tables if not name.startswith("sqlite_")] == [
        "association",
        "load_event",
        "meta",
        "prompt_event",
        "skill",
        "skill_alias",
    ]
    assert default_association_db_path("/home/someone") == "/home/someone/.cache/hermes-skill-router/associations.db"
    assert association_db_path_for("/cache/hermes-skill-router/index.json") == "/cache/hermes-skill-router/associations.db"
    assert default_state_db_path("/home/someone") == "/home/someone/.hermes/state.db"
    store.close()


def test_the_store_refuses_a_harness_it_does_not_own_and_a_file_stamped_by_another(tmp_path: Path):
    path = str(tmp_path / "owned.db")
    store = open_store(path)
    write_meta(store, META_HARNESS, "dsh")
    store.close()
    with pytest.raises(AssociationStoreError, match="dsh harness"):
        open_store(path)
    assert Path(path).exists(), "the refused file is left alone rather than replaced"
    with pytest.raises(AssociationStoreError, match="only the hermes harness store"):
        open_store(str(tmp_path / "other.db"), "dsh")


def test_the_store_refuses_a_foreign_schema_a_newer_version_and_garbage(tmp_path: Path):
    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(str(foreign))
    conn.execute("CREATE TABLE something_else (x INTEGER)")
    conn.close()
    with pytest.raises(AssociationStoreError, match="not a version-1 database"):
        open_store(str(foreign))

    newer = tmp_path / "newer.db"
    store = open_store(str(newer))
    store.conn.execute("PRAGMA user_version = 2")
    write_meta(store, META_SCHEMA_VERSION, "2")
    store.close()
    with pytest.raises(AssociationStoreError, match="newer than this build"):
        open_store(str(newer))

    garbage = tmp_path / "garbage.db"
    garbage.write_text("this is not a database\n", encoding="utf-8")
    with pytest.raises(AssociationStoreError):
        open_store(str(garbage))


def test_an_incremental_write_is_idempotent_and_records_its_provenance(tmp_path: Path):
    store = temp_store(tmp_path, "upsert")
    reconcile_catalog(store, CATALOG, T0)
    events = [
        load_of(0, origin="automation", evidence="slash", strength=EXPLICIT_STRENGTH),
        load_of(1),
    ]
    first = upsert_events(store, events)
    assert (first.prompts, first.loads, first.duplicates) == (2, 2, 0)
    second = upsert_events(store, events)
    assert (second.prompts, second.loads, second.duplicates) == (0, 0, 2)
    row = load_rows(store)[0]
    assert row["evidence"] == "slash"
    assert row["strength"] == 1.0
    assert row["origin"] == "automation", "automation provenance is kept, not dropped"
    assert row["source"] == "backfill"
    assert row["session_id"] == "session-0"
    assert row["turn"] == 1
    assert row["terms"] == "review association store implementation case"
    store.close()


def test_no_raw_prompt_text_reaches_the_database(tmp_path: Path):
    path = tmp_path / "privacy.db"
    store = open_store(str(path))
    reconcile_catalog(store, CATALOG, T0)
    upsert_events(store, [load_of(0, prompt_text="Review ZebraCorn-42 storefront immediately")])
    terms = store.conn.execute("SELECT terms FROM prompt_event").fetchone()[0]
    assert terms == "review zebracorn 42 storefront immediately"
    for term in terms.split(" "):
        assert term.isalnum(), "terms are lowercase tokens, never text"
    assert store.conn.execute("SELECT count(*) FROM prompt_event WHERE instr(terms, 'ZebraCorn') > 0").fetchone()[0] == 0
    store.close()
    payload = path.read_bytes()
    assert b"ZebraCorn" not in payload, "the mixed-case original is nowhere in the file"
    assert b"ZebraCorn-42 storefront" not in payload, "no sentence of the prompt survives"


def test_equal_evidence_gives_an_equal_weight_and_only_a_follow_on_read_is_weaker(tmp_path: Path):
    store = temp_store(tmp_path, "equal")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    observed = T0 + 5000
    upsert_events(
        store,
        [
            load_of(0, prompt_text=SUBSTANTIVE, prompt_observed_ms=observed, observed_ms=observed, evidence="tool"),
            load_of(
                1,
                prompt_text=SUBSTANTIVE,
                prompt_observed_ms=observed,
                observed_ms=observed,
                skill_name="handoff-doc",
                skill_path=catalog_path("handoff-doc"),
                evidence="skill_md_read",
            ),
            load_of(
                2,
                prompt_text=SUBSTANTIVE,
                prompt_observed_ms=observed,
                observed_ms=observed,
                skill_name="writing",
                skill_path=catalog_path("writing"),
                evidence="tool",
                strength=FOLLOW_ON_STRENGTH,
            ),
        ],
    )
    weights = {row.skill_id: row.weight for row in decayed_weights(store, ["review"], observed)}
    code_review = resolve_skill_name(store, "code-review").id
    handoff = resolve_skill_name(store, "handoff-doc").id
    writing = resolve_skill_name(store, "writing").id
    assert weights[code_review] == weights[handoff], "a skill_view and a SKILL.md read weigh the same"
    assert weights[writing] == pytest.approx(weights[code_review] / 2), "a follow-on read weighs half"
    store.close()


def test_association_weight_decays_with_a_thirty_day_half_life(tmp_path: Path):
    store = temp_store(tmp_path, "decay")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(store, [load_of(0, observed_ms=T0 + DAY)])
    assert decay_factor(0) == 1.0
    assert decay_factor(ASSOCIATION_HALF_LIFE_MS) == pytest.approx(0.5)
    for elapsed, expected in ((0, 1.0), (30 * DAY, 0.5), (60 * DAY, 0.25)):
        weight = decayed_weights(store, ["review"], T0 + DAY + elapsed)[0].weight
        assert weight == pytest.approx(expected), f"after {elapsed} ms the weight is {expected}"
    store.close()


def test_support_counts_distinct_prompts_not_loads(tmp_path: Path):
    store = temp_store(tmp_path, "support")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(
        store,
        [
            load_of(0),
            load_of(0, evidence="skill_md_read", strength=EXPLICIT_STRENGTH, observed_ms=T0 + 2000),
        ],
    )
    assert decayed_weights(store, ["review"])[0].support == 1, "two loads inside one prompt are one supporting prompt"
    upsert_events(store, [load_of(1)])
    assert decayed_weights(store, ["review"])[0].support == 2
    store.close()


def test_recompute_matches_the_incremental_update_bit_for_bit_including_out_of_order_events(tmp_path: Path):
    store = temp_store(tmp_path, "recompute")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 12)
    upsert_events(
        store,
        [
            load_of(1, observed_ms=T0 + 3 * DAY),
            load_of(0, observed_ms=T0 + 2 * DAY, evidence="skill_md_read"),
            load_of(2, skill_name="handoff-doc", skill_path=catalog_path("handoff-doc"), evidence="slash"),
            load_of(0, evidence="tool", strength=FOLLOW_ON_STRENGTH, observed_ms=T0 + 4 * DAY),
            load_of(3, observed_ms=T0 + DAY, skill_name="writing", skill_path=catalog_path("writing")),
        ],
    )
    incremental = rows(store, "SELECT skill_id, term, weight, support, first_ms, last_ms FROM association ORDER BY skill_id, term")
    assert incremental, "the fixtures produced association rows"
    store.conn.execute("DELETE FROM association")
    assert recompute_associations(store) == len(incremental)
    rebuilt = rows(store, "SELECT skill_id, term, weight, support, first_ms, last_ms FROM association ORDER BY skill_id, term")
    assert rebuilt == incremental, "a rebuild is bit-identical to the incremental update, not merely close"
    store.close()


def test_an_ambient_skill_is_excluded_from_ranks_without_losing_its_rows(tmp_path: Path):
    store = temp_store(tmp_path, "ambient")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(
        store,
        [
            load_of(0),
            load_of(1),
            load_of(2),
            load_of(3, skill_name="handoff-doc", skill_path=catalog_path("handoff-doc")),
            load_of(4, skill_name="handoff-doc", skill_path=catalog_path("handoff-doc")),
        ],
    )
    code_review = resolve_skill_name(store, "code-review").id
    handoff = resolve_skill_name(store, "handoff-doc").id
    ambient = ambient_skill_ids(store, T0 + 5 * DAY)
    assert code_review in ambient, "3 of 10 prompts is above 25%"
    assert handoff not in ambient, "2 of 10 prompts is not"
    ranked = [row.skill_id for row in decayed_weights(store, ["review"], T0 + 5 * DAY)]
    assert ranked == [handoff], "only the non-ambient skill produces a rank"
    assert rows(store, "SELECT count(*) FROM association WHERE skill_id = ?", (code_review,))[0][0] > 0
    assert association_stats(store, T0 + 5 * DAY).ambient_skills == 1
    store.close()


def test_catalog_reconciliation_soft_deletes_a_dropped_name_and_only_maintenance_purges_it(tmp_path: Path):
    store = temp_store(tmp_path, "deleted")
    first = reconcile_catalog(store, CATALOG, T0)
    assert (first.changed, first.inserted) == (True, 3)
    assert reconcile_catalog(store, CATALOG, T0 + DAY).changed is False, "an unchanged fingerprint skips the pass"
    seed_prompts(store, 10)
    upsert_events(store, [load_of(0, skill_name="writing", skill_path=catalog_path("writing"))])
    writing = resolve_skill_name(store, "writing").id
    assert decayed_weights(store, ["review"], T0 + DAY), "a live skill ranks"

    dropped = reconcile_catalog(store, CATALOG[:2], T0 + DAY)
    assert dropped.soft_deleted == 1
    assert resolve_skill_name(store, "writing").deleted_at_ms is not None
    assert decayed_weights(store, ["review"], T0 + DAY) == [], "a soft-deleted skill produces no rank"
    assert rows(store, "SELECT count(*) FROM load_event WHERE skill_id = ?", (writing,))[0][0] > 0

    early = maintenance(store, catalog=CATALOG[:2], now_ms=T0 + 29 * DAY)
    assert early.deleted_skills_removed == 0, "nothing is purged before 30 days"
    late = maintenance(store, catalog=CATALOG[:2], now_ms=T0 + 32 * DAY)
    assert late.deleted_skills_removed == 1
    assert resolve_skill_name(store, "writing") is None
    assert rows(store, "SELECT count(*) FROM load_event WHERE skill_id = ?", (writing,))[0][0] == 0
    assert rows(store, "SELECT count(*) FROM association WHERE skill_id = ?", (writing,))[0][0] == 0
    store.close()


def test_reset_removes_selected_associations_and_leaves_the_rest_intact(tmp_path: Path):
    store = temp_store(tmp_path, "reset-selected")
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(
        store,
        [
            load_of(0, skill_name="writing", skill_path=catalog_path("writing")),
            load_of(1),
        ],
    )
    writing = resolve_skill_name(store, "writing").id
    report = reset_associations(store, skills=["writing"])
    assert report.skills == 1
    assert report.load_events_removed == 1
    assert report.association_rows_removed > 0
    assert rows(store, "SELECT count(*) FROM load_event WHERE skill_id = ?", (writing,))[0][0] == 0
    assert rows(store, "SELECT count(*) FROM association WHERE skill_id = ?", (writing,))[0][0] == 0
    assert rows(store, "SELECT count(*) FROM load_event")[0][0] == 1, "the other skill keeps its evidence"
    assert resolve_skill_path(store, catalog_path("writing")).deleted_at_ms is None, "the catalog row is not learned data"
    assert len(decayed_weights(store, ["review"], T0 + 2 * DAY)) == 1

    cleared = reset_associations(store)
    assert cleared.skills == 3
    stats = association_stats(store)
    assert (stats.loads, stats.prompts, stats.associations) == (0, 0, 0)
    assert decayed_weights(store, ["review"]) == []
    assert read_meta(store, META_LAST_SCAN_MS) == "0"
    assert read_meta(store, META_CATALOG_FINGERPRINT) == ""
    rebuilt = reconcile_catalog(store, CATALOG, T0 + 3 * DAY)
    assert rebuilt.changed is True, "the cleared fingerprint forces a reconcile"
    assert rebuilt.inserted == 3
    store.close()


def test_deleting_the_association_database_leaves_the_vector_index_byte_for_byte_identical(tmp_path: Path):
    index_path = tmp_path / "index.json"
    index_text = json.dumps({"version": 1, "names": ["code-review"], "vectors": [[0.1, 0.9, 0.3]]})
    index_path.write_text(index_text, encoding="utf-8")
    index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()

    db_path = tmp_path / ASSOCIATION_DB_NAME
    store = open_store(str(db_path))
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(store, [load_of(0)])
    assert association_stats(store).loads > 0

    delete_association_database(store)
    assert db_path.exists() is False, "the database file is gone"
    assert Path(str(db_path) + "-wal").exists() is False, "and its WAL"
    assert Path(str(db_path) + "-shm").exists() is False, "and its shared-memory file"
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == index_hash, "the vector index is untouched"
    assert index_path.read_text(encoding="utf-8") == index_text
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["index.json"], "the reset left nothing else"

    reopened = open_store(str(db_path))
    stats = association_stats(reopened)
    assert (stats.skills, stats.prompts, stats.loads, stats.associations) == (0, 0, 0, 0)
    reopened.close()


# --------------------------------------------------------------------------------------------------
# The backfill: the Hermes store, attribution, provenance and the load paths
# --------------------------------------------------------------------------------------------------


def test_backfill_attributes_a_skill_view_to_the_nearest_preceding_user_message(hermes, tmp_path: Path):
    hermes.session("s-auto", source="kanban")
    hermes.user("s-auto", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-auto", "code-review", T0 + 2000)
    hermes.user("s-auto", "Write the handoff document for this task now", T0 + 3000)
    hermes.skill_view("s-auto", "handoff-doc", T0 + 4000)
    hermes.commit()

    store = temp_store(tmp_path, "attribution")
    report = backfilled(store, hermes.path)
    assert (report.sessions_found, report.sessions_scanned, report.sessions_skipped) == (1, 1, 0)
    assert report.prompts == 2
    assert report.loads == 2
    assert report.rejected == zero_rejections()
    stored = load_rows(store)
    assert [row["skill_name"] for row in stored] == ["code-review", "handoff-doc"]
    assert [row["terms"] for row in stored] == [
        "review association store migration path",
        "write handoff document task",
    ]
    assert stored[0]["origin"] == "automation", "a kanban session is kept and flagged"
    assert stored[0]["source"] == "backfill"
    assert stored[0]["evidence"] == "tool"
    assert stored[0]["strength"] == 1.0
    store.close()


def test_backfill_is_idempotent(hermes, tmp_path: Path):
    hermes.session("s-a")
    hermes.user("s-a", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-a", "code-review", T0 + 2000)
    hermes.commit()
    store = temp_store(tmp_path, "idempotent")
    first = backfilled(store, hermes.path)
    before = association_stats(store)
    second = backfilled(store, hermes.path)
    assert first.loads == 1
    assert second.loads == 0, "the rerun stored nothing new"
    assert second.duplicates == first.loads
    assert second.prompts == 0
    assert first.ambient_loads == 1, "the only load sits above the ambient threshold"
    assert second.ambient_loads == 0, "the report counts this run's inserts, not the first run's"
    assert association_stats(store) == before
    store.close()


def test_backfill_counts_a_skipped_session_instead_of_aborting(hermes, tmp_path: Path):
    hermes.session("s-quiet")
    hermes.session("s-broken")
    hermes.user("s-broken", SUBSTANTIVE, T0 + 1000)
    hermes.raw_messages("s-broken", "{not json at all", T0 + 2000)
    call_id = hermes.calls("s-broken", [("skill_view", {"name": "code-review"})], T0 + 3000)[0]
    hermes.result("s-broken", call_id, "skill_view", "not json either", T0 + 3100)
    hermes.session("s-good")
    hermes.user("s-good", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-good", "code-review", T0 + 2000)
    hermes.commit()

    store = temp_store(tmp_path, "skips")
    report = backfilled(store, hermes.path)
    assert report.sessions_found == 3
    assert report.sessions_skipped == 1, "the message-less session is counted, not fatal"
    assert report.skip_reasons["no-messages"] == 1
    assert report.sessions_scanned == 2
    assert report.unreadable_results == 1, "one odd result row is counted, not fatal"
    assert report.loads == 1, "the readable session was still stored"
    store.close()


def test_backfill_refuses_an_injected_user_row_as_a_boundary_and_a_continuation_prompt(hermes, tmp_path: Path):
    hermes.session("s-mixed")
    hermes.user("s-mixed", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-mixed", "code-review", T0 + 2000)
    hermes.user(
        "s-mixed",
        '[IMPORTANT: The user has invoked the "writing" skill, indicating they want you to follow its instructions.]',
        T0 + 3000,
    )
    hermes.user("s-mixed", "continue", T0 + 4000)
    hermes.skill_view("s-mixed", "handoff-doc", T0 + 5000)
    hermes.commit()

    store = temp_store(tmp_path, "injected")
    report = backfilled(store, hermes.path)
    assert report.loads == 1, "the load after 'continue' is dropped"
    assert report.rejected["substantive-prompt"] == 1
    assert report.rejected["typed-prompt"] == 0, "the injected row is not a boundary, so the earlier prompt still answers"
    terms = [row[0] for row in rows(store, "SELECT terms FROM prompt_event ORDER BY turn")]
    assert terms == ["review association store migration path"], "neither 'continue' nor the skill body was stored"
    assert load_rows(store)[0]["skill_name"] == "code-review"
    store.close()


def test_backfill_keeps_automation_sessions_and_refuses_a_subagent_session(hermes, tmp_path: Path):
    hermes.session("s-kanban", source="kanban")
    hermes.user("s-kanban", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-kanban", "code-review", T0 + 2000)
    hermes.session("s-cron", source="cron")
    hermes.user("s-cron", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-cron", "writing", T0 + 2000)
    hermes.session("s-sub", source="subagent")
    hermes.user("s-sub", "Delegate the association store review to the child agent now", T0 + 1000)
    hermes.skill_view("s-sub", "handoff-doc", T0 + 2000)
    hermes.commit()

    store = temp_store(tmp_path, "provenance")
    report = backfilled(store, hermes.path)
    assert report.loads == 2, "both automation sessions are kept"
    assert report.rejected["session-provenance"] == 1, "the subagent session contributes nothing"
    assert report.prompts == 2, "and its brief is not in the ubiquity denominator either"
    origins = sorted(row["origin"] for row in load_rows(store))
    assert origins == ["automation", "automation"]
    store.close()


def test_a_skill_view_with_file_counts_only_after_the_skill_md_in_that_session(hermes, tmp_path: Path):
    hermes.session("s-follow-on")
    hermes.user("s-follow-on", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-follow-on", "code-review", T0 + 2000)
    hermes.user("s-follow-on", "Read the rules reference for the same skill now", T0 + 3000)
    hermes.skill_view("s-follow-on", "code-review", T0 + 4000, file_path="references/rules.md")
    hermes.session("s-orphan", source="oneshot", started_ms=T0 + 1000)
    hermes.user("s-orphan", "Read the association reference before anything else", T0 + 2000)
    hermes.skill_view("s-orphan", "code-review", T0 + 3000, file_path="references/rules.md")
    hermes.commit()

    store = temp_store(tmp_path, "follow-on")
    report = backfilled(store, hermes.path)
    assert report.rejected["explicit-load"] == 1, "a reference read with no earlier SKILL.md is not a load"
    by_strength = [
        (row["skill_name"], row["evidence"], row["strength"]) for row in load_rows(store)
    ]
    assert by_strength == [("code-review", "tool", 1.0), ("code-review", "tool", 0.5)], (
        "the orphan read is dropped and the follow-on read carries the lower strength in its own window"
    )
    store.close()


def test_a_typed_slash_name_is_a_load_at_the_same_strength(hermes, tmp_path: Path):
    hermes.session("s-slash")
    hermes.user("s-slash", "/handoff-doc write the handoff for this task", T0 + 1000)
    hermes.commit()
    store = temp_store(tmp_path, "slash")
    report = backfilled(store, hermes.path)
    assert report.loads == 1
    row = load_rows(store)[0]
    assert row["evidence"] == "slash"
    assert row["strength"] == 1.0, "a typed /name and a skill_view carry the same strength"
    assert row["skill_name"] == "handoff-doc"
    assert row["terms"] == "handoff doc write handoff task", "the typed command line is its own prompt"
    store.close()


def test_a_read_file_of_skill_md_and_a_skill_view_produce_the_same_weight(hermes, tmp_path: Path):
    hermes.session("s-view")
    hermes.user("s-view", "Review the association store and its migration path", T0 + 1000)
    hermes.skill_view("s-view", "code-review", T0 + 2000)
    hermes.session("s-read")
    hermes.user("s-read", "Review the association store and its migration path", T0 + 1000)
    hermes.read_file("s-read", "/home/wilder/.hermes/skills/handoff-doc/SKILL.md", T0 + 2000)
    hermes.commit()

    store = temp_store(tmp_path, "equal-backfill")
    seed_prompts(store, 10)
    report = backfilled(store, hermes.path)
    assert report.loads == 2
    evidence = sorted((row["skill_name"], row["evidence"]) for row in load_rows(store))
    assert evidence == [("code-review", "tool"), ("handoff-doc", "skill_md_read")]
    weights = {row.skill_id: row.weight for row in decayed_weights(store, ["review"], T0 + 2000)}
    code_review = resolve_skill_name(store, "code-review").id
    handoff = resolve_skill_name(store, "handoff-doc").id
    assert weights[code_review] == weights[handoff] == 1.0, "a skill_view and a SKILL.md read weigh the same"
    store.close()


def test_backfill_stores_tokens_and_hashes_only_and_records_provenance(hermes, tmp_path: Path):
    hermes.session("s-priv", source="desktop")
    hermes.user("s-priv", "Review ZebraCorn-42 storefront immediately", T0 + 1000)
    hermes.skill_view("s-priv", "code-review", T0 + 2000)
    hermes.commit()
    db_path = tmp_path / "privacy-backfill.db"
    store = open_store(str(db_path))
    report = backfilled(store, hermes.path)
    row = load_rows(store)[0]
    assert row["source"] == "backfill"
    assert row["origin"] == "human"
    assert row["terms"] == "review zebracorn 42 storefront immediately"
    assert report.last_scan_ms >= T0 + 900_000, "the newest session stamp is the watermark"
    assert int(read_meta(store, META_LAST_SCAN_MS)) >= T0 + 900_000
    store.close()
    payload = db_path.read_bytes()
    assert b"ZebraCorn" not in payload, "the prompt text never reached the database file"


def test_the_ubiquity_denominator_counts_prompts_that_loaded_nothing(hermes, tmp_path: Path):
    hermes.session("s-ubiquity")
    for turn in range(8):
        hermes.user(
            "s-ubiquity",
            f"Review the association store migration path number {turn}",
            T0 + (turn + 1) * 60_000,
        )
    hermes.skill_view("s-ubiquity", "code-review", T0 + 90_000)
    hermes.commit()
    store = temp_store(tmp_path, "ubiquity")
    report = backfilled(store, hermes.path)
    assert report.prompts == 8, "eight prompts were typed, so eight are in the denominator"
    assert report.loads == 1
    assert report.ambient_loads == 0, "1 of 8 is 12.5%, below the ambient threshold"
    store.close()


def test_backfill_reads_the_live_hermes_store_whole_and_records_its_wall_time(tmp_path: Path):
    state = Path(default_state_db_path())
    if not state.exists():
        pytest.skip("this machine has no Hermes session store")
    index_path = Path.home() / ".cache" / "hermes-skill-router" / "index.json"
    names: List[str] = []
    if index_path.exists():
        try:
            names = list(json.loads(index_path.read_text(encoding="utf-8")).get("names") or [])
        except (OSError, ValueError):
            names = []
    catalog = [CatalogSkill(name) for name in names[:2000]]
    store = temp_store(tmp_path, "live")
    now_ms = int(time.time() * 1000)
    report = backfill(store, str(state), catalog=catalog, now_ms=now_ms)
    assert report.sessions_found > 300, f"the live store holds {report.sessions_found} sessions"
    assert report.sessions_scanned + report.sessions_skipped == report.sessions_found
    assert report.sessions_scanned > 250
    assert report.skip_reasons["write-failed"] == 0, "a write failure would be a bug in this module"
    assert report.skip_reasons["unreadable"] == 0, "the live store's sessions are readable"
    assert report.loads > 0, "the live store carries skill_view evidence"
    assert report.ms < 60_000, f"the whole-store pass cost {report.ms} ms"
    catalog_names = {skill.name for skill in catalog}
    reads = live_skill_md_read_paths(state)
    resolvable = [path for path in reads if Path(path).parent.name in catalog_names]
    stored_kinds = sorted({row["evidence"] for row in load_rows(store)})
    if resolvable:
        assert "skill_md_read" in stored_kinds, (
            f"{len(resolvable)} live SKILL.md reads resolve to the catalog, so the backfill stores "
            f"them; it stored {stored_kinds}"
        )
    print(
        f"\nlive Hermes backfill: {report.sessions_found} sessions, "
        f"{report.sessions_scanned} scanned, {report.sessions_skipped} skipped, "
        f"{report.prompts} prompts, {report.loads} loads, {report.duplicates} duplicates, "
        f"{report.ambient_loads} ambient, {report.ms} ms, "
        f"{len(reads)} SKILL.md reads ({len(resolvable)} resolvable), evidence {stored_kinds}"
    )
    store.close()


# --------------------------------------------------------------------------------------------------
# Review round 1 fixes: the live read shape, path collisions, write failures and edge provenance
# --------------------------------------------------------------------------------------------------


def test_the_store_refuses_a_schema_version_that_is_not_a_number(tmp_path: Path):
    path = str(tmp_path / "odd-version.db")
    store = open_store(path)
    write_meta(store, META_SCHEMA_VERSION, "not-a-number")
    store.close()
    with pytest.raises(AssociationStoreError, match="not a version-1 database"):
        open_store(path)


def test_a_read_only_store_can_rank_but_refuses_maintenance_and_reset(tmp_path: Path):
    path = tmp_path / "read-only.db"
    store = open_store(str(path))
    reconcile_catalog(store, CATALOG, T0)
    seed_prompts(store, 10)
    upsert_events(store, [load_of(0)])
    store.close()

    reader = open_store(str(path), read_only=True)
    assert reader.read_only is True
    assert decayed_weights(reader, ["review"], T0 + DAY), "a read-only store still ranks"
    with pytest.raises(AssociationStoreError, match="maintenance needs a writable store"):
        maintenance(reader, now_ms=T0 + 2 * DAY)
    with pytest.raises(AssociationStoreError, match="reset needs a writable store"):
        reset_associations(reader)
    with pytest.raises(AssociationStoreError, match="open read-only"):
        upsert_events(reader, [load_of(0)])
    with pytest.raises(AssociationStoreError, match="open read-only"):
        upsert_prompts(reader, [prompt_of(0)])
    with pytest.raises(AssociationStoreError, match="open read-only"):
        reconcile_catalog(reader, CATALOG, T0)
    with pytest.raises(AssociationStoreError, match="open read-only"):
        write_meta(reader, META_LAST_SCAN_MS, "1")
    with pytest.raises(AssociationStoreError, match="open read-only"):
        recompute_associations(reader)
    with pytest.raises(AssociationStoreError, match="open read-only"):
        delete_association_database(reader)
    assert path.exists(), "a read-only handle never deletes the database file"
    with pytest.raises(AssociationStoreError, match="open read-only"):
        backfill(reader, str(tmp_path / "state.db"))
    reader.close()


def test_two_names_that_reveal_one_path_do_not_abort_the_backfill(hermes, tmp_path: Path):
    shared = "/home/wilder/.hermes/skills/renamed/SKILL.md"
    hermes.session("s-one")
    hermes.user("s-one", SUBSTANTIVE, T0 + 1000)
    first = hermes.calls("s-one", [("skill_view", {"name": "code-review"})], T0 + 2000)[0]
    hermes.result(
        "s-one",
        first,
        "skill_view",
        json.dumps({"success": True, "name": "code-review", "_source_path": shared}),
        T0 + 2100,
    )
    hermes.session("s-two", started_ms=T0 + 1000)
    hermes.user("s-two", SUBSTANTIVE, T0 + 2000)
    second = hermes.calls("s-two", [("skill_view", {"name": "handoff-doc"})], T0 + 3000)[0]
    hermes.result(
        "s-two",
        second,
        "skill_view",
        json.dumps({"success": True, "name": "handoff-doc", "_source_path": shared}),
        T0 + 3100,
    )
    hermes.commit()

    store = temp_store(tmp_path, "collision")
    report = backfilled(store, hermes.path)
    assert report.loads == 2
    assert report.sessions_scanned == 2
    assert report.skip_reasons["write-failed"] == 0, "one shared path is not a write failure"
    assert rows(store, "SELECT name, path FROM skill ORDER BY name") == [
        ("code-review", catalog_path("code-review")),
        ("handoff-doc", shared),
        ("writing", catalog_path("writing")),
    ], "the first name keeps the revealed path and the second keeps its name-keyed path"


def test_backfill_counts_a_session_whose_write_fails_instead_of_aborting(
    hermes, tmp_path: Path, monkeypatch
):
    hermes.session("s-good")
    hermes.user("s-good", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-good", "code-review", T0 + 2000)
    hermes.session("s-bad", started_ms=T0 + 1000)
    hermes.user("s-bad", "Review the writing skill reference material now", T0 + 2000)
    hermes.skill_view("s-bad", "writing", T0 + 3000)
    hermes.skill_view("s-bad", "unknown-skill", T0 + 4000)
    hermes.commit()

    real = associations_module.upsert_events

    def flaky(store, events):
        if any(event.session_id == "s-bad" for event in events):
            raise sqlite3.OperationalError("database is locked")
        return real(store, events)

    monkeypatch.setattr(associations_module, "upsert_events", flaky)
    store = temp_store(tmp_path, "write-failed")
    report = backfilled(store, hermes.path)
    assert report.skip_reasons["write-failed"] == 1
    assert report.sessions_skipped == 1
    assert report.sessions_scanned == 1
    assert report.loads == 1, "the healthy session was still stored"
    assert report.prompts == 2, "the failed session's committed prompt rows are reported, not hidden"
    assert report.rejected == zero_rejections(), "a failed session leaves no gate counts behind"
    assert rows(store, "SELECT count(*) FROM prompt_event")[0][0] == 2
    assert [row["skill_name"] for row in load_rows(store)] == ["code-review"]


def test_a_compacted_row_resets_attribution_until_the_next_typed_prompt(hermes, tmp_path: Path):
    hermes.session("s-stopped")
    hermes.user("s-stopped", SUBSTANTIVE, T0 + 1000)
    hermes.user(
        "s-stopped", "Summary of the compacted conversation so far", T0 + 2000, compacted=1
    )
    hermes.skill_view("s-stopped", "code-review", T0 + 3000)

    hermes.session("s-resumed", started_ms=T0 + 1000)
    hermes.user("s-resumed", SUBSTANTIVE, T0 + 1000)
    hermes.user(
        "s-resumed", "Summary of the compacted conversation so far", T0 + 2000, compacted=1
    )
    hermes.user("s-resumed", "Now read the writing skill reference material", T0 + 3000)
    hermes.skill_view("s-resumed", "writing", T0 + 4000)
    hermes.commit()

    store = temp_store(tmp_path, "compacted")
    report = backfilled(store, hermes.path)
    assert report.loads == 1, "the load after the compaction is dropped, the later prompt's is kept"
    assert report.rejected["typed-prompt"] == 1
    assert [(row["skill_name"], row["terms"]) for row in load_rows(store)] == [
        ("writing", "read writing skill reference material")
    ]


def test_a_read_file_under_references_is_a_follow_on_load(hermes, tmp_path: Path):
    hermes.session("s-ref")
    hermes.user("s-ref", SUBSTANTIVE, T0 + 1000)
    hermes.read_file("s-ref", "/home/wilder/.hermes/skills/code-review/SKILL.md", T0 + 2000)
    hermes.user("s-ref", "Read the rules reference for the same skill now", T0 + 3000)
    hermes.read_reference(
        "s-ref", "/home/wilder/.hermes/skills/code-review/references/rules.md", T0 + 4000
    )

    hermes.session("s-ref-orphan", started_ms=T0 + 1000)
    hermes.user("s-ref-orphan", "Read the association reference before anything else", T0 + 2000)
    hermes.read_reference(
        "s-ref-orphan", "/home/wilder/.hermes/skills/code-review/references/rules.md", T0 + 3000
    )
    hermes.commit()

    store = temp_store(tmp_path, "reference")
    report = backfilled(store, hermes.path)
    assert report.rejected["explicit-load"] == 1, "a reference read with no earlier SKILL.md is not a load"
    assert [(row["skill_name"], row["evidence"], row["strength"]) for row in load_rows(store)] == [
        ("code-review", "skill_md_read", 1.0),
        ("code-review", "tool", 0.5),
    ]


def test_an_unclassified_session_source_is_refused_and_stores_no_prompt(hermes, tmp_path: Path):
    hermes.session("s-imported", source="imported")
    hermes.user("s-imported", SUBSTANTIVE, T0 + 1000)
    hermes.skill_view("s-imported", "code-review", T0 + 2000)
    hermes.commit()

    store = temp_store(tmp_path, "unknown-source")
    report = backfilled(store, hermes.path)
    assert report.rejected["session-provenance"] == 1, "an unknown source fails closed"
    assert report.loads == 0
    assert report.prompts == 0, "an unclassified session does not enter the ubiquity denominator"
    assert rows(store, "SELECT count(*) FROM load_event")[0][0] == 0
    store.close()
