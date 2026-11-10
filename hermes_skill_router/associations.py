"""The Hermes association store, the eligibility gate and the backfill (modules M9, C9).

This is the Hermes half of the learned-association channel: the same SQLite schema, the same six
gate conditions and the same incremental-versus-recompute contract as the DSH router's
``src/associations/{store,gate,backfill}.ts``, restated against the Hermes session store. The
schema, privacy and identity rules are ``RESEARCH.md`` section 5, the gate is section 4.1 and the
attribution rule is section 2.4 and 6.2 of the pre-reranker routing research.

What the file holds is learned evidence, not conversation. A prompt is stored as its normalised
terms plus a hash of those terms, never as text, and every load keeps its evidence kind, its
session provenance (`human`, `automation`, `subagent`, or `unknown` for a source this build does
not classify), its source (`live` or `backfill`) and the prompt event it was attributed to. The raw
prompt stays in the Hermes store where it already lives, which is why debugging a bad association
means reading that store, not this database.

Isolation is by file first and by column second. Every writer checks ``meta.harness`` against its
own harness name on open, so a copied file cannot silently mix two harnesses' evidence, and this
module reads only the Hermes store at ``/home/wilder/.hermes/state.db``.

Two Hermes facts shape the port:

* The Hermes catalog the plugin indexes carries names and descriptions only (``library.py``), and
  ``skill_view`` is addressed by name. There is no catalog path to key a skill on, so a skill's
  identity here is its **name**, the one identifier the catalog, the loader and the suggestion block
  share. ``skill.path`` still exists because the schema requires it: it holds the resolved absolute
  ``SKILL.md`` path once a load reveals one, and the synthetic ``hermes:<name>`` until then. A load
  that reveals a path updates the row in place instead of creating a second row, so one skill never
  splits into two identities. The DSH store keys on the path because DSH reads files through the
  ``read`` tool; the two rules differ because the two catalogs do.
* A Hermes load has no turn to stop at. Attribution walks back from the load to the nearest
  preceding typed user row, skipping harness-injected rows and compacted summaries, and never pairs
  a load with a prompt that comes after it (section 2.4). 84% of Hermes sessions are automation
  (kanban, cron, oneshot, subagent), so automation provenance is kept and flagged instead of
  excluded: their prompts are real task descriptions, and the ablation measures their contribution.
  A subagent session contributes nothing in the first release, because its prompt is the brief its
  parent wrote; a source this build does not classify answers ``unknown`` and is refused the same
  way rather than being recorded as a subagent.

The association database sits beside the vector index, at
``/home/wilder/.cache/hermes-skill-router/associations.db`` by default, and deleting it loses only
learned data. The vector index is never written here.

``association.weight`` is an exponentially decayed sum anchored at the row's ``last_ms``, so a read
can decay it once more to the query time::

    weight = sum over observations of strength * 0.5 ** ((last_ms - observed_ms) / 30 days)
    at time t: weight * 0.5 ** ((t - last_ms) / 30 days)

Exponential decay is multiplicative, so that is exactly a per-observation sum without storing every
observation twice. It also makes an incremental insert and a full recomputation from ``load_event``
bit-identical: both apply the same recurrence to the same events in ``load_event.id`` order.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: The schema version this build writes and the only one it opens.
SCHEMA_VERSION = 1

#: The harness this store is for. A file that says anything else is refused.
HARNESS = "hermes"

#: The file name beside the vector index.
ASSOCIATION_DB_NAME = "associations.db"

#: The router's cache directory, relative to the home directory.
ASSOCIATION_CACHE_DIR = os.path.join(".cache", "hermes-skill-router")

#: The Hermes store this module reads, relative to the home directory.
HERMES_STATE_DB = os.path.join(".hermes", "state.db")

#: One writer at a time, and a writer waits this long for the other one.
BUSY_TIMEOUT_MS = 5000

#: Association decay half-life. The design's 30 days, not frecency's 14.
ASSOCIATION_HALF_LIFE_MS = 30 * 24 * 60 * 60 * 1000

#: A skill missing from the catalog keeps its rows this long before the purge.
DELETED_SKILL_RETENTION_MS = 30 * 24 * 60 * 60 * 1000

#: A prompt event and its loads are kept this long.
PROMPT_RETENTION_MS = 180 * 24 * 60 * 60 * 1000

#: ``meta`` keys.
META_HARNESS = "harness"
META_SCHEMA_VERSION = "schema_version"
META_LAST_SCAN_MS = "last_scan_ms"
META_CATALOG_FINGERPRINT = "catalog_fingerprint"

#: The evidence kinds the ``load_event`` schema admits.
LOAD_EVIDENCE_KINDS: Tuple[str, ...] = ("tool", "slash", "skill_md_read")

#: Strength of an explicit load, whatever its evidence kind. A ``skill_view`` by name, a typed
#: ``/name`` and a read of a ``SKILL.md`` path all carry exactly this, so "both count equally" is one
#: constant rather than three weights to keep aligned.
EXPLICIT_STRENGTH = 1.0

#: Strength of a reference-file read that follows the same skill's ``SKILL.md`` in the session.
FOLLOW_ON_STRENGTH = 0.5

#: Minimum characters of prompt text, before normalisation.
SUBSTANTIVE_MIN_CHARS = 20

#: Minimum normalised terms in a substantive prompt.
SUBSTANTIVE_MIN_TERMS = 3

#: A skill loaded above this share of eligible prompts is ambient.
UBIQUITY_THRESHOLD = 0.25

#: The window the ubiquity gate measures over.
UBIQUITY_WINDOW_DAYS = 30

#: The window in milliseconds, from the day count.
UBIQUITY_WINDOW_MS = UBIQUITY_WINDOW_DAYS * 24 * 60 * 60 * 1000

#: The only prompt source that is the user's own typing.
PROMPT_SOURCE_USER = "user"

#: Markers a harness writes into a user row. An injected row is never the user's typing, even when
#: the harness stored it with ``role=user``: Hermes injects a loaded skill's body, a system nudge, an
#: image caption and an out-of-band steer as user rows, and a prompt-term corpus must not learn from
#: them. ``<system-reminder>`` is the DSH marker, kept so one marker list covers both harnesses.
INJECTED_PROMPT_MARKERS: Tuple[str, ...] = (
    "<system-reminder>",
    "[IMPORTANT:",
    "[System:",
    "[OUT-OF-BAND",
    "[The user attached an image",
)

#: Session sources that are a person working in a harness surface.
HUMAN_SOURCES = frozenset({"cli", "desktop", "tui", "local"})

#: Session sources that are automation whose prompts are still task descriptions.
AUTOMATION_SOURCES = frozenset({"kanban", "cron", "oneshot"})

#: Session sources whose "prompt" is its parent's brief, which contributes nothing.
SUBAGENT_SOURCES = frozenset({"subagent"})

#: The path segment that marks a file inside a skill's ``references/`` directory.
REFERENCE_SEGMENT = "/references/"

#: One condition of the gate, in the order the design lists them.
GATE_CONDITIONS: Tuple[str, ...] = (
    "explicit-load",
    "typed-prompt",
    "substantive-prompt",
    "same-boundary",
    "session-provenance",
    "ubiquity",
)

#: The six tables of schema version 1, exactly as the design states them.
SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill (
  id            INTEGER PRIMARY KEY,
  harness       TEXT NOT NULL CHECK (harness IN ('dsh', 'hermes')),
  name          TEXT NOT NULL,
  path          TEXT NOT NULL,
  first_seen_ms INTEGER NOT NULL,
  last_seen_ms  INTEGER NOT NULL,
  deleted_at_ms INTEGER,
  UNIQUE (harness, path)
);

CREATE TABLE IF NOT EXISTS skill_alias (
  harness       TEXT NOT NULL,
  name          TEXT NOT NULL,
  skill_id      INTEGER NOT NULL REFERENCES skill(id) ON DELETE CASCADE,
  valid_from_ms INTEGER NOT NULL,
  valid_to_ms   INTEGER,
  PRIMARY KEY (harness, name, valid_from_ms)
);

CREATE TABLE IF NOT EXISTS prompt_event (
  id           INTEGER PRIMARY KEY,
  session_id   TEXT NOT NULL,
  turn         INTEGER NOT NULL,
  prompt_hash  TEXT NOT NULL,
  terms        TEXT NOT NULL,
  length       INTEGER NOT NULL,
  observed_ms  INTEGER NOT NULL,
  UNIQUE (session_id, turn, prompt_hash)
);

CREATE TABLE IF NOT EXISTS load_event (
  id              INTEGER PRIMARY KEY,
  prompt_event_id INTEGER NOT NULL REFERENCES prompt_event(id) ON DELETE CASCADE,
  skill_id        INTEGER NOT NULL REFERENCES skill(id) ON DELETE CASCADE,
  evidence        TEXT NOT NULL CHECK (evidence IN ('tool', 'slash', 'skill_md_read')),
  strength        REAL NOT NULL,
  origin          TEXT NOT NULL,
  source          TEXT NOT NULL CHECK (source IN ('live', 'backfill')),
  observed_ms     INTEGER NOT NULL,
  dedup_key       TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS load_event_skill_time ON load_event (skill_id, observed_ms);

CREATE TABLE IF NOT EXISTS association (
  skill_id  INTEGER NOT NULL REFERENCES skill(id) ON DELETE CASCADE,
  term      TEXT NOT NULL,
  weight    REAL NOT NULL DEFAULT 1.0,
  support   INTEGER NOT NULL DEFAULT 1,
  first_ms  INTEGER NOT NULL,
  last_ms   INTEGER NOT NULL,
  PRIMARY KEY (skill_id, term)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS association_term ON association (term);
"""


# --------------------------------------------------------------------------------------------------
# The gate: one pure predicate over a load event
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptEvidence:
    """The prompt a load is attributed to.

    ``text`` is used only inside this process: the store keeps the normalised terms and a hash of
    them. ``index`` is the prompt's message position in its session, the Hermes substitute for a
    turn number. ``source_kind`` is ``user`` only for the user's own typing.
    """

    text: str
    source_kind: str
    index: Optional[int] = None


@dataclass(frozen=True)
class LoadEvent:
    """One candidate load, as the backfill (or a live observer) saw it.

    ``index`` is the load's own message position, ``nearest_prompt_index`` is the position of the
    nearest preceding typed user row the backfill walked back to. The two together are the Hermes
    attribution boundary: the attributed prompt must be that row, and it must not come after the
    load. ``origin`` is the session provenance; ``None`` is unattributed, which counts as human.
    """

    evidence: str
    strength: float
    skill_id: Optional[int]
    prompt: Optional[PromptEvidence]
    index: Optional[int]
    nearest_prompt_index: Optional[int]
    skill_path: Optional[str] = None
    after_skill_md: bool = False
    origin: Optional[str] = None


@dataclass(frozen=True)
class GateContext:
    """What the gate needs beyond the event: the ubiquity counts for the trailing 30 days.

    A caller that passes neither has not evaluated condition 6, and the gate passes it: the
    association channel applies the same rule from its store's ambient set. Half a context is
    rejected as malformed rather than read as zero.
    """

    skill_prompts: Optional[int] = None
    eligible_prompts: Optional[int] = None
    ubiquity_threshold: float = UBIQUITY_THRESHOLD


def is_eligible_load(event: Any, context: Optional[GateContext] = None) -> bool:
    """Return whether this load is association evidence: true only when all six conditions hold."""
    return gate_rejection(event, context) is None


def gate_rejection(event: Any, context: Optional[GateContext] = None) -> Optional[str]:
    """Return the first condition this load fails, or ``None`` when it passes all six.

    The order is the design's order, so a rejection is always attributable to exactly one condition:
    the earliest one that does not hold. The predicate is total: anything that is not a well-formed
    event is rejected rather than thrown, because a backfill over thousands of rows must not stop on
    one odd record.
    """
    if not _is_explicit_load(event):
        return "explicit-load"
    prompt = _prompt_of(event)
    if prompt is None or not _is_typed_prompt(prompt):
        return "typed-prompt"
    if not _is_substantive_prompt(prompt.text):
        return "substantive-prompt"
    if not _is_same_boundary(event, prompt):
        return "same-boundary"
    if not _is_provenanced(event):
        return "session-provenance"
    if _is_ubiquitous(event, context):
        return "ubiquity"
    return None


def is_eligible_prompt(prompt: Any) -> bool:
    """Return whether a prompt can be the ubiquity denominator: typed and substantive.

    This is conditions 2 and 3 without a load attached, because a prompt that loaded nothing is
    still one of the eligible prompts a skill is measured against.
    """
    if not isinstance(prompt, PromptEvidence):
        return False
    if not _is_typed_prompt(prompt):
        return False
    return _is_substantive_prompt(prompt.text)


def _is_explicit_load(event: Any) -> bool:
    """Condition 1. Known evidence kind, a resolved skill, and the follow-on prerequisite.

    A reference-file read carries the lower strength and only counts when the same skill's
    ``SKILL.md`` was loaded earlier in the session. A slash command or a ``SKILL.md`` read at half
    strength would double-count the follow-on rule through an evidence kind that has no follow-on
    form, so both are refused.
    """
    if not isinstance(event, LoadEvent):
        return False
    if event.evidence not in LOAD_EVIDENCE_KINDS:
        return False
    if event.strength not in (EXPLICIT_STRENGTH, FOLLOW_ON_STRENGTH):
        return False
    if event.strength == FOLLOW_ON_STRENGTH:
        if event.evidence != "tool" or event.after_skill_md is not True:
            return False
    skill_id = event.skill_id
    return isinstance(skill_id, int) and not isinstance(skill_id, bool) and skill_id > 0


def _prompt_of(event: LoadEvent) -> Optional[PromptEvidence]:
    """Return the prompt of an event, or ``None`` when it carries none."""
    prompt = event.prompt
    if not isinstance(prompt, PromptEvidence):
        return None
    if not isinstance(prompt.text, str) or not isinstance(prompt.source_kind, str):
        return None
    if prompt.index is not None and not isinstance(prompt.index, int):
        return None
    return prompt


def _is_typed_prompt(prompt: PromptEvidence) -> bool:
    """Condition 2. The user's own typing, decided by the source kind and the content markers.

    Hermes stores injected material as ``role=user`` rows, so the source kind is not enough on its
    own: a row whose text carries one of the harness markers is refused even when a caller labelled
    it ``user``. The backfill labels such a row with its own kind instead, and this check is the
    second lock.
    """
    if prompt.source_kind != PROMPT_SOURCE_USER:
        return False
    return not any(marker in prompt.text for marker in INJECTED_PROMPT_MARKERS)


def _is_substantive_prompt(text: str) -> bool:
    """Condition 3. Long enough, enough terms, and not a continuation phrase."""
    if len(text.strip()) < SUBSTANTIVE_MIN_CHARS:
        return False
    if len(normalise_terms(text)) < SUBSTANTIVE_MIN_TERMS:
        return False
    return not is_continuation(text)


def _is_same_boundary(event: LoadEvent, prompt: PromptEvidence) -> bool:
    """Condition 4, the Hermes attribution boundary.

    The load is attributed to the nearest typed user row that precedes it, so the attributed prompt
    must be that row and must not come after the load. A load paired with an earlier prompt than the
    nearest one, or with a prompt that follows it, is dropped rather than guessed at.
    """
    index = event.index
    nearest = event.nearest_prompt_index
    if not isinstance(index, int) or isinstance(index, bool):
        return False
    if not isinstance(nearest, int) or isinstance(nearest, bool):
        return False
    if prompt.index is None or prompt.index != nearest:
        return False
    return prompt.index <= index


def _is_provenanced(event: LoadEvent) -> bool:
    """Condition 5. Human or automation provenance; a subagent session contributes nothing.

    Automation is kept and flagged because 84% of the Hermes store is automation and its prompts are
    real task descriptions. An unknown provenance fails closed, because a session this build cannot
    classify is not evidence it can explain.
    """
    origin = event.origin
    return origin is None or origin in ("human", "automation")


def _is_ubiquitous(event: LoadEvent, context: Optional[GateContext]) -> bool:
    """Condition 6. An ambient skill is excluded from the association channel."""
    if not isinstance(event, LoadEvent):
        return True
    if context is None:
        return False
    skill_prompts = context.skill_prompts
    eligible_prompts = context.eligible_prompts
    if skill_prompts is None and eligible_prompts is None:
        return False
    if not isinstance(skill_prompts, int) or not isinstance(eligible_prompts, int):
        return True
    if isinstance(skill_prompts, bool) or isinstance(eligible_prompts, bool):
        return True
    return is_ambient(skill_prompts, eligible_prompts, context.ubiquity_threshold)


def is_ambient(
    skill_prompts: float,
    eligible_prompts: float,
    threshold: float = UBIQUITY_THRESHOLD,
) -> bool:
    """Return whether a skill loaded by more than *threshold* of eligible prompts is ambient.

    A zero denominator is not ambient, because an empty window carries no evidence either way. A
    malformed count is ambient, which fails closed: an unreadable count must not admit the channel's
    most frequent skill.
    """
    for value in (skill_prompts, eligible_prompts, threshold):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return True
        if not math.isfinite(float(value)):
            return True
    if skill_prompts < 0 or eligible_prompts < 0:
        return True
    if eligible_prompts == 0 or skill_prompts == 0:
        return False
    return skill_prompts / eligible_prompts > threshold


#: The continuation phrases the design rejects. A prompt made only of these is not a task.
CONTINUATION_PHRASES = frozenset(
    {"continue", "go on", "do it", "proceed", "ok", "okay", "yes", "next"}
)

#: The tokens a continuation may be built from, for the all-tokens form.
CONTINUATION_TOKENS = frozenset(
    {
        "continue",
        "go",
        "on",
        "do",
        "it",
        "proceed",
        "ok",
        "okay",
        "yes",
        "next",
        "please",
        "thanks",
        "sure",
    }
)


def is_continuation(text: str) -> bool:
    """Return whether a prompt is a continuation rather than a task.

    Two forms are rejected: the whole prompt is one of the design's phrases, or every term it has is
    a continuation token. ``Continue the migration and add tests`` is neither, so a real instruction
    that happens to start with a continuation word stays eligible.
    """
    flat = _flatten(text)
    if flat == "":
        return True
    if flat in CONTINUATION_PHRASES:
        return True
    terms = normalise_terms(text)
    if not terms:
        return True
    return all(term in CONTINUATION_TOKENS for term in terms)


#: Words dropped before a prompt becomes terms: the standard English function-word set plus the
#: conversation fillers that appear in every task. No stemmer, so no dependency.
STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been before being but by can
    could did do does doing done for from get got had has have he her here him his how i if in into
    is it its just like me more most my no not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those through to too
    under until up us very was we were what when where which while who why will with would you your
    """.split()
)


def normalise_terms(text: str) -> List[str]:
    """Return the normalised terms of a prompt: lowercased, split on non-alphanumerics only.

    Tokens shorter than two characters and stopwords are dropped, and nothing is stemmed. This is
    the only text that reaches the association database, so it is also the privacy boundary.
    Non-Latin scripts are kept, because a Vietnamese or Chinese prompt still carries topic words.
    """
    terms: List[str] = []
    for raw in _flatten(text).split(" "):
        if not raw or len(raw) < 2:
            continue
        if raw in STOPWORDS:
            continue
        terms.append(raw)
    return terms


def _flatten(text: str) -> str:
    """Return lowercased text with punctuation and repeated whitespace flattened to one space."""
    if not isinstance(text, str):
        return ""
    characters: List[str] = []
    for char in text.lower():
        characters.append(char if char.isalnum() or char.isspace() else " ")
    return " ".join("".join(characters).split())


# --------------------------------------------------------------------------------------------------
# The store: one SQLite file beside the router's vector index
# --------------------------------------------------------------------------------------------------


class AssociationStoreError(Exception):
    """Raised for a file this build must not write: wrong harness, wrong schema, not a database."""


@dataclass
class AssociationStore:
    """One open association database, its path, harness and read-only state."""

    conn: sqlite3.Connection
    path: str
    harness: str
    read_only: bool

    def close(self) -> None:
        """Close the connection. Idempotent."""
        try:
            self.conn.close()
        except sqlite3.Error:
            pass


@dataclass(frozen=True)
class CatalogSkill:
    """A catalog row as the store needs it: the name the router emits and its path, when known."""

    name: str
    path: Optional[str] = None

    def resolved_path(self) -> str:
        """Return the catalog path, or the name-keyed fallback Hermes has when it has no path."""
        return self.path if self.path else catalog_path(self.name)


@dataclass(frozen=True)
class SkillRow:
    """A skill row, as resolution and tests read it."""

    id: int
    name: str
    path: str
    deleted_at_ms: Optional[int] = None


@dataclass(frozen=True)
class AssociationEvent:
    """One load observation to store, with the prompt it is attributed to."""

    session_id: str
    turn: int
    prompt_text: str
    prompt_observed_ms: int
    skill_name: str
    skill_path: str
    evidence: str
    strength: float
    origin: str
    source: str
    observed_ms: int


@dataclass(frozen=True)
class PromptEventInput:
    """One typed prompt, stored whether or not it loaded a skill."""

    session_id: str
    turn: int
    text: str
    observed_ms: int


@dataclass(frozen=True)
class UpsertReport:
    """What one upsert changed."""

    prompts: int = 0
    loads: int = 0
    duplicates: int = 0


@dataclass(frozen=True)
class ReconcileReport:
    """What catalog reconciliation changed."""

    changed: bool = False
    inserted: int = 0
    reactivated: int = 0
    soft_deleted: int = 0
    fingerprint: str = ""


@dataclass(frozen=True)
class MaintenanceReport:
    """What one maintenance pass did."""

    catalog: Optional[ReconcileReport] = None
    deleted_skills_removed: int = 0
    stale_prompts_removed: int = 0
    association_rows: int = 0
    vacuumed: bool = False


@dataclass(frozen=True)
class ResetReport:
    """What a reset removed."""

    skills: int = 0
    load_events_removed: int = 0
    association_rows_removed: int = 0


@dataclass(frozen=True)
class AssociationWeight:
    """One association weight for one skill, decayed to the read time."""

    skill_id: int
    weight: float
    support: int


@dataclass(frozen=True)
class AssociationStats:
    """Row counts, for a report or a test."""

    skills: int = 0
    deleted_skills: int = 0
    aliases: int = 0
    prompts: int = 0
    loads: int = 0
    associations: int = 0
    ambient_skills: int = 0


def catalog_path(name: str) -> str:
    """Return the synthetic path a Hermes skill carries until a load reveals its ``SKILL.md``.

    Hermes has no catalog path to key on, so the name is the identity and this value keeps the
    schema's path column meaningful without inventing a file. A real path, once known, replaces it
    on the same row.
    """
    return "hermes:" + name


def default_association_db_path(home: Optional[str] = None) -> str:
    """Return the default database path: the router's cache directory, beside the vector index."""
    root = os.path.expanduser(home if home is not None else "~")
    return os.path.join(root, ASSOCIATION_CACHE_DIR, ASSOCIATION_DB_NAME)


def association_db_path_for(index_cache_path: str) -> str:
    """Return the database path beside an index file at *index_cache_path*."""
    return os.path.join(os.path.dirname(os.path.expanduser(index_cache_path)), ASSOCIATION_DB_NAME)


def default_state_db_path(home: Optional[str] = None) -> str:
    """Return the Hermes session store path this module reads."""
    root = os.path.expanduser(home if home is not None else "~")
    return os.path.join(root, HERMES_STATE_DB)


def open_store(
    path: str,
    harness: str = HARNESS,
    *,
    read_only: bool = False,
    busy_timeout_ms: int = BUSY_TIMEOUT_MS,
) -> AssociationStore:
    """Open the association store, creating the schema on a fresh file.

    Raises :class:`AssociationStoreError` when the file belongs to another harness, when it is not a
    version-1 database, when it was written by a newer schema, or when it is not a SQLite database at
    all. Those are the four cases where writing would mix or destroy evidence, so none of them is
    repaired silently.
    """
    if harness != HARNESS:
        raise AssociationStoreError(
            f"associations: this build opens only the {HARNESS} harness store, not {harness}"
        )
    try:
        conn = _connect(path, read_only, busy_timeout_ms)
    except sqlite3.Error as error:
        raise AssociationStoreError(f"associations: cannot open {path}: {error}") from error
    try:
        _configure(conn, path, read_only, busy_timeout_ms)
        tables = _count_tables(conn)
        if tables == 0:
            if read_only:
                raise AssociationStoreError(f"associations: {path} is not a version-1 database yet")
            _migrate(conn, harness)
        else:
            _require_compatible(conn, harness)
            version = _read_user_version(conn)
            if version < SCHEMA_VERSION:
                if read_only:
                    raise AssociationStoreError(
                        f"associations: {path} needs a migration this read-only open cannot apply"
                    )
                _migrate(conn, harness)
    except AssociationStoreError:
        _close_quietly(conn)
        raise
    except sqlite3.Error as error:
        _close_quietly(conn)
        raise AssociationStoreError(f"associations: {path} cannot be used: {error}") from error
    return AssociationStore(conn=conn, path=path, harness=harness, read_only=read_only)


def _connect(path: str, read_only: bool, busy_timeout_ms: int) -> sqlite3.Connection:
    """Open one connection in explicit-transaction mode, read-only when asked."""
    timeout = max(0, busy_timeout_ms) / 1000.0
    if read_only:
        uri = pathlib.Path(os.path.abspath(os.path.expanduser(path))).as_uri() + "?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=timeout, isolation_level=None)
    return sqlite3.connect(
        os.path.expanduser(path), timeout=timeout, isolation_level=None
    )


def _configure(conn: sqlite3.Connection, path: str, read_only: bool, busy_timeout_ms: int) -> None:
    """Apply the pragmas one writer and its readers share."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {max(0, int(busy_timeout_ms))}")
    if read_only:
        conn.execute("PRAGMA query_only = ON")
        return
    row = conn.execute("PRAGMA journal_mode = WAL").fetchone()
    mode = None if row is None else row[0]
    if str(mode).lower() != "wal":
        raise AssociationStoreError(
            f"associations: {path} did not accept WAL (journal_mode is {mode})"
        )
    conn.execute("PRAGMA synchronous = NORMAL")


def _close_quietly(conn: sqlite3.Connection) -> None:
    """Close a connection whose open failed; nothing is worth keeping from it."""
    try:
        conn.close()
    except sqlite3.Error:
        pass


def _migrate(conn: sqlite3.Connection, harness: str) -> None:
    """Create schema version 1 on a fresh file and stamp its meta rows and user_version.

    ``executescript`` commits any open transaction before it runs, so the schema is created first
    and the stamp is written in one immediate transaction afterwards. A half-created schema
    therefore never carries a version stamp.
    """
    try:
        conn.executescript(SCHEMA_V1)
        conn.execute("BEGIN IMMEDIATE")
        for key, value in (
            (META_HARNESS, harness),
            (META_SCHEMA_VERSION, str(SCHEMA_VERSION)),
            (META_LAST_SCAN_MS, "0"),
            (META_CATALOG_FINGERPRINT, ""),
        ):
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    except sqlite3.Error as error:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise AssociationStoreError(
            f"associations: schema migration {SCHEMA_VERSION} failed: {error}"
        ) from error


def _require_compatible(conn: sqlite3.Connection, harness: str) -> None:
    """Refuse a file that another harness owns, or that this build cannot read as version 1."""
    meta = _read_meta_table(conn)
    if meta is None:
        raise AssociationStoreError(
            "associations: the file is not a version-1 database (no meta table)"
        )
    stored = meta.get(META_HARNESS)
    if stored is None:
        raise AssociationStoreError(
            "associations: the file is not a version-1 database (no meta.harness)"
        )
    if stored != harness:
        raise AssociationStoreError(
            f"associations: the file belongs to the {stored} harness, so the {harness} router "
            "refuses to write it"
        )
    version = _read_user_version(conn)
    try:
        declared = int(meta.get(META_SCHEMA_VERSION, str(version)))
    except (TypeError, ValueError):
        raise AssociationStoreError(
            "associations: the file is not a version-1 database (meta.schema_version is not a number)"
        ) from None
    if declared < 1:
        raise AssociationStoreError("associations: the file is not a version-1 database")
    if declared != version:
        raise AssociationStoreError(
            f"associations: schema version {declared} in meta does not match user_version {version}"
        )
    if version > SCHEMA_VERSION:
        raise AssociationStoreError(
            f"associations: schema version {version} is newer than this build's {SCHEMA_VERSION}"
        )


def _count_tables(conn: sqlite3.Connection) -> int:
    """Return how many non-SQLite tables the file carries."""
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _read_user_version(conn: sqlite3.Connection) -> int:
    """Return the file's ``PRAGMA user_version``."""
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row is not None else 0


def _read_meta_table(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    """Return every ``meta`` row, or ``None`` when the table is absent or unreadable."""
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return None
    return {str(row["key"]): str(row["value"]) for row in rows}


def read_meta(store: AssociationStore, key: str) -> Optional[str]:
    """Return one ``meta`` value, or ``None`` when the key is absent."""
    row = store.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def write_meta(store: AssociationStore, key: str, value: str) -> None:
    """Write one ``meta`` value."""
    store.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def _meta_int(store: AssociationStore, key: str, default: int = 0) -> int:
    """Return one ``meta`` value as an int, or *default* when it is absent or unparseable."""
    raw = read_meta(store, key)
    try:
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _skill_row(row: sqlite3.Row) -> SkillRow:
    """Turn one ``skill`` row into a :class:`SkillRow`."""
    deleted = row["deleted_at_ms"]
    return SkillRow(
        id=int(row["id"]),
        name=str(row["name"]),
        path=str(row["path"]),
        deleted_at_ms=None if deleted is None else int(deleted),
    )


def resolve_skill_path(store: AssociationStore, path: str) -> Optional[SkillRow]:
    """Return the skill row at a path, or ``None``. A soft-deleted skill still resolves."""
    row = store.conn.execute(
        "SELECT id, name, path, deleted_at_ms FROM skill WHERE harness = ? AND path = ?",
        (store.harness, path),
    ).fetchone()
    return None if row is None else _skill_row(row)


def resolve_skill_name(store: AssociationStore, name: str) -> Optional[SkillRow]:
    """Return the skill a name resolves to: a live row by name, then an open alias, then a closed one.

    In the Hermes store the name is the identity, so this is the primary lookup; the alias table
    still answers for a replayed load that used an older name. A name another live skill has since
    taken answers at the first step, so reuse of a name cannot attach one skill's history to another.
    """
    live = store.conn.execute(
        """SELECT id, name, path, deleted_at_ms FROM skill
           WHERE harness = ? AND name = ?
           ORDER BY (deleted_at_ms IS NOT NULL), last_seen_ms DESC LIMIT 1""",
        (store.harness, name),
    ).fetchone()
    if live is not None:
        return _skill_row(live)
    alias = store.conn.execute(
        """SELECT s.id, s.name, s.path, s.deleted_at_ms FROM skill_alias a
           JOIN skill s ON s.id = a.skill_id
           WHERE a.harness = ? AND a.name = ?
           ORDER BY (a.valid_to_ms IS NULL) DESC, a.valid_from_ms DESC LIMIT 1""",
        (store.harness, name),
    ).fetchone()
    return None if alias is None else _skill_row(alias)


def _open_alias(store: AssociationStore, skill_id: int, name: str, at_ms: int) -> None:
    """Record that a name names a skill from *at_ms* onwards."""
    store.conn.execute(
        """INSERT OR REPLACE INTO skill_alias (harness, name, skill_id, valid_from_ms, valid_to_ms)
           VALUES (?, ?, ?, ?, NULL)""",
        (store.harness, name, skill_id, at_ms),
    )


def _close_aliases(store: AssociationStore, skill_id: int, at_ms: int) -> None:
    """Close every open alias row of one skill."""
    store.conn.execute(
        "UPDATE skill_alias SET valid_to_ms = ? WHERE skill_id = ? AND valid_to_ms IS NULL",
        (at_ms, skill_id),
    )


def _ensure_open_alias(store: AssociationStore, skill_id: int, name: str, at_ms: int) -> None:
    """Open an alias row for a name only when no open one exists."""
    row = store.conn.execute(
        "SELECT 1 FROM skill_alias WHERE harness = ? AND name = ? AND valid_to_ms IS NULL LIMIT 1",
        (store.harness, name),
    ).fetchone()
    if row is None:
        _open_alias(store, skill_id, name, at_ms)


def is_synthetic_path(path: str) -> bool:
    """Return whether a path is the name-keyed fallback rather than a file Hermes revealed."""
    return path.startswith("hermes:")


def _available_path(
    store: AssociationStore, name: str, wanted: str, owner: Optional[SkillRow]
) -> str:
    """Return *wanted* when no other skill row holds it, else the name-keyed fallback.

    Hermes identity is the name, but the schema keeps T1's ``UNIQUE (harness, path)``. Two names
    can reveal or carry one path (a renamed skill, a plugin dir shared by aliases), and the second
    must not collide with the first: the path falls back to the synthetic one so the name keeps its
    own row, its own history, and the write never aborts the session.
    """
    if is_synthetic_path(wanted):
        return wanted
    holder = resolve_skill_path(store, wanted)
    if holder is None or (owner is not None and holder.id == owner.id):
        return wanted
    return catalog_path(name)


def ensure_skill(
    store: AssociationStore, name: str, path: Optional[str], observed_ms: int
) -> SkillRow:
    """Return the skill row for a load, inserting it when the name is new.

    Hermes identity is the name, so a row is looked up by name and its ``path`` is updated in place
    when a load reveals the real ``SKILL.md`` path. That keeps one skill one row even though the
    catalog carries no path to key on.
    """
    existing = resolve_skill_name(store, name)
    wanted = _available_path(store, name, path if path else catalog_path(name), existing)
    if existing is not None:
        if is_synthetic_path(existing.path) and not is_synthetic_path(wanted):
            store.conn.execute(
                "UPDATE skill SET path = ?, last_seen_ms = MAX(last_seen_ms, ?) WHERE id = ?",
                (wanted, observed_ms, existing.id),
            )
        else:
            store.conn.execute(
                "UPDATE skill SET last_seen_ms = MAX(last_seen_ms, ?) WHERE id = ?",
                (observed_ms, existing.id),
            )
        _ensure_open_alias(store, existing.id, name, observed_ms)
        updated = resolve_skill_name(store, name)
        return updated if updated is not None else existing
    cursor = store.conn.execute(
        """INSERT INTO skill (harness, name, path, first_seen_ms, last_seen_ms, deleted_at_ms)
           VALUES (?, ?, ?, ?, ?, NULL)""",
        (store.harness, name, wanted, observed_ms, observed_ms),
    )
    skill_id = int(cursor.lastrowid)
    _open_alias(store, skill_id, name, observed_ms)
    return SkillRow(id=skill_id, name=name, path=wanted, deleted_at_ms=None)


def catalog_fingerprint(catalog: Sequence[CatalogSkill]) -> str:
    """Return a stable hash of the catalog's (name, path) pairs, sorted so order does not matter."""
    lines = sorted(f"{skill.resolved_path()}\t{skill.name}" for skill in catalog)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def reconcile_catalog(
    store: AssociationStore,
    catalog: Sequence[CatalogSkill],
    now_ms: Optional[int] = None,
) -> ReconcileReport:
    """Reconcile the live catalog against ``skill``.

    A name the store does not know is inserted with its path when the catalog carries one, or with
    the name-keyed fallback Hermes has when it does not. A row whose skill the catalog stopped
    listing is soft deleted, and a name that comes back reactivates its row. A rename is not
    observable here: Hermes has no catalog path to follow, so a name that changes is a different
    skill, and reusing a name never merges two histories. A fingerprint in ``meta`` skips the pass
    when the catalog has not changed.
    """
    stamp = _now() if now_ms is None else int(now_ms)
    fingerprint = catalog_fingerprint(catalog)
    if read_meta(store, META_CATALOG_FINGERPRINT) == fingerprint:
        return ReconcileReport(changed=False, fingerprint=fingerprint)

    inserted = reactivated = soft_deleted = 0
    names: set = set()
    for skill in catalog:
        if skill.name in names:
            continue
        names.add(skill.name)
        existing = resolve_skill_name(store, skill.name)
        wanted = _available_path(store, skill.name, skill.resolved_path(), existing)
        if existing is None:
            cursor = store.conn.execute(
                """INSERT INTO skill (harness, name, path, first_seen_ms, last_seen_ms, deleted_at_ms)
                   VALUES (?, ?, ?, ?, ?, NULL)""",
                (store.harness, skill.name, wanted, stamp, stamp),
            )
            _open_alias(store, int(cursor.lastrowid), skill.name, stamp)
            inserted += 1
            continue
        if existing.deleted_at_ms is not None:
            store.conn.execute(
                """UPDATE skill SET deleted_at_ms = NULL, last_seen_ms = MAX(last_seen_ms, ?)
                   WHERE id = ?""",
                (stamp, existing.id),
            )
            reactivated += 1
        if is_synthetic_path(existing.path) and not is_synthetic_path(wanted):
            store.conn.execute("UPDATE skill SET path = ? WHERE id = ?", (wanted, existing.id))
        store.conn.execute(
            "UPDATE skill SET last_seen_ms = MAX(last_seen_ms, ?) WHERE id = ?",
            (stamp, existing.id),
        )
        _ensure_open_alias(store, existing.id, skill.name, stamp)

    rows = store.conn.execute(
        "SELECT id, name FROM skill WHERE harness = ? AND deleted_at_ms IS NULL",
        (store.harness,),
    ).fetchall()
    for row in rows:
        if str(row["name"]) in names:
            continue
        store.conn.execute("UPDATE skill SET deleted_at_ms = ? WHERE id = ?", (stamp, int(row["id"])))
        soft_deleted += 1

    write_meta(store, META_CATALOG_FINGERPRINT, fingerprint)
    return ReconcileReport(
        changed=True,
        inserted=inserted,
        reactivated=reactivated,
        soft_deleted=soft_deleted,
        fingerprint=fingerprint,
    )


def _insert_prompt(store: AssociationStore, prompt: PromptEventInput) -> Tuple[int, bool]:
    """Normalise and insert one prompt, returning its id and whether it was new."""
    terms = normalise_terms(prompt.text)
    joined = " ".join(terms)
    prompt_hash = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    cursor = store.conn.execute(
        """INSERT OR IGNORE INTO prompt_event
           (session_id, turn, prompt_hash, terms, length, observed_ms) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            prompt.session_id,
            prompt.turn,
            prompt_hash,
            joined,
            len(prompt.text.strip()),
            prompt.observed_ms,
        ),
    )
    row = store.conn.execute(
        "SELECT id FROM prompt_event WHERE session_id = ? AND turn = ? AND prompt_hash = ?",
        (prompt.session_id, prompt.turn, prompt_hash),
    ).fetchone()
    if row is None:
        raise AssociationStoreError("associations: the prompt event vanished after an insert")
    return int(row["id"]), int(cursor.rowcount) > 0


def upsert_prompts(store: AssociationStore, prompts: Sequence[PromptEventInput]) -> int:
    """Store eligible prompts that loaded nothing, and return how many were new.

    The ubiquity gate divides a skill's loads by the eligible prompts of the window, and most typed
    prompts load nothing. Keeping only loaded prompts would divide by a fraction of the real
    denominator and mark ordinary skills ambient. These rows carry terms and a hash, exactly like a
    prompt that did load something, so the privacy boundary is unchanged.
    """
    count = 0

    def work() -> None:
        nonlocal count
        for prompt in prompts:
            _id, inserted = _insert_prompt(store, prompt)
            count += 1 if inserted else 0

    _transaction(store, work)
    return count


def upsert_events(store: AssociationStore, events: Sequence[AssociationEvent]) -> UpsertReport:
    """Store one batch of observations.

    Idempotent by construction: ``prompt_event`` is unique on ``(session_id, turn, prompt_hash)`` and
    ``load_event.dedup_key`` covers the skill and the evidence kind, so a second backfill over the
    same session inserts nothing and reports the duplicates instead. Each inserted load also updates
    its association rows with the same recurrence a full recomputation uses.
    """
    if not events:
        return UpsertReport()
    result = {"prompts": 0, "loads": 0, "duplicates": 0}

    def work() -> None:
        for event in events:
            prompt_id, inserted = _insert_prompt(
                store,
                PromptEventInput(
                    session_id=event.session_id,
                    turn=event.turn,
                    text=event.prompt_text,
                    observed_ms=event.prompt_observed_ms,
                ),
            )
            if inserted:
                result["prompts"] += 1
            skill = ensure_skill(store, event.skill_name, event.skill_path, event.observed_ms)
            terms = normalise_terms(event.prompt_text)
            first_for_prompt = (
                store.conn.execute(
                    "SELECT count(*) FROM load_event WHERE skill_id = ? AND prompt_event_id = ?",
                    (skill.id, prompt_id),
                ).fetchone()[0]
                == 0
            )
            dedup_key = _dedup_key(
                store.harness, event.session_id, event.turn, terms, skill.id, event.evidence
            )
            cursor = store.conn.execute(
                """INSERT OR IGNORE INTO load_event
                   (prompt_event_id, skill_id, evidence, strength, origin, source, observed_ms,
                    dedup_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt_id,
                    skill.id,
                    event.evidence,
                    event.strength,
                    event.origin,
                    event.source,
                    event.observed_ms,
                    dedup_key,
                ),
            )
            if int(cursor.rowcount) == 0:
                result["duplicates"] += 1
                continue
            result["loads"] += 1
            _apply_load(store, skill.id, terms, event.strength, event.observed_ms, first_for_prompt)

    _transaction(store, work)
    return UpsertReport(**result)


def _dedup_key(
    harness: str, session_id: str, turn: int, terms: Sequence[str], skill_id: int, evidence: str
) -> str:
    """Return ``sha256(harness | session_id | turn | prompt_hash | skill_id | evidence)``."""
    prompt_hash = hashlib.sha256(" ".join(terms).encode("utf-8")).hexdigest()
    joined = "|".join([harness, session_id, str(turn), prompt_hash, str(skill_id), evidence])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _apply_load(
    store: AssociationStore,
    skill_id: int,
    terms: Sequence[str],
    strength: float,
    observed_ms: int,
    first_for_prompt: bool,
) -> None:
    """Add one observation to the association rows of one skill, in the recurrence form."""
    for term in dict.fromkeys(terms):
        row = store.conn.execute(
            "SELECT weight, support, first_ms, last_ms FROM association WHERE skill_id = ? AND term = ?",
            (skill_id, term),
        ).fetchone()
        if row is None:
            store.conn.execute(
                """INSERT INTO association (skill_id, term, weight, support, first_ms, last_ms)
                   VALUES (?, ?, ?, 1, ?, ?)""",
                (skill_id, term, strength, observed_ms, observed_ms),
            )
            continue
        weight = float(row["weight"])
        support = int(row["support"])
        first = int(row["first_ms"])
        last = int(row["last_ms"])
        if observed_ms >= last:
            nxt = weight * decay_factor(observed_ms - last) + strength
        else:
            nxt = weight + strength * decay_factor(last - observed_ms)
        store.conn.execute(
            """UPDATE association SET weight = ?, support = ?, first_ms = ?, last_ms = ?
               WHERE skill_id = ? AND term = ?""",
            (
                nxt,
                support + (1 if first_for_prompt else 0),
                min(first, observed_ms),
                max(last, observed_ms),
                skill_id,
                term,
            ),
        )


def decay_factor(elapsed_ms: float, half_life_ms: float = ASSOCIATION_HALF_LIFE_MS) -> float:
    """Return ``0.5 ** (elapsed_ms / half_life_ms)``, the decay of one observation."""
    if not half_life_ms or half_life_ms <= 0:
        return 1.0
    if elapsed_ms <= 0:
        return 1.0
    return float(0.5 ** (float(elapsed_ms) / float(half_life_ms)))


def recompute_associations(store: AssociationStore) -> int:
    """Rebuild ``association`` from ``load_event`` and return the row count.

    The only source of truth is the load rows, so a bad incremental update, a reset of one skill, or
    a purge is repaired by calling this. Events are replayed in ``load_event.id`` order, the order
    they were inserted in, so the result is bit-identical to the incremental path rather than merely
    close to it.
    """
    rows = store.conn.execute(
        """SELECT le.skill_id AS skill_id, le.prompt_event_id AS prompt_id,
                  le.observed_ms AS observed_ms, le.strength AS strength, pe.terms AS terms
           FROM load_event le JOIN prompt_event pe ON pe.id = le.prompt_event_id
           ORDER BY le.id"""
    ).fetchall()

    accumulated: Dict[Tuple[int, str], List[float]] = {}
    seen: set = set()
    for row in rows:
        skill_id = int(row["skill_id"])
        prompt_id = int(row["prompt_id"])
        observed_ms = int(row["observed_ms"])
        strength = float(row["strength"])
        seen_key = f"{skill_id}:{prompt_id}"
        first_for_prompt = seen_key not in seen
        seen.add(seen_key)
        for term in dict.fromkeys(str(row["terms"]).split(" ")):
            if not term:
                continue
            key = (skill_id, term)
            current = accumulated.get(key)
            if current is None:
                accumulated[key] = [strength, 1.0, float(observed_ms), float(observed_ms)]
                continue
            weight, support, first, last = current
            if observed_ms >= last:
                weight = weight * decay_factor(observed_ms - last) + strength
            else:
                weight = weight + strength * decay_factor(last - observed_ms)
            current[0] = weight
            current[1] = support + (1.0 if first_for_prompt else 0.0)
            current[2] = min(first, observed_ms)
            current[3] = max(last, observed_ms)

    def work() -> None:
        store.conn.execute("DELETE FROM association")
        for (skill_id, term), values in sorted(accumulated.items()):
            store.conn.execute(
                """INSERT INTO association (skill_id, term, weight, support, first_ms, last_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (skill_id, term, values[0], int(values[1]), int(values[2]), int(values[3])),
            )

    _transaction(store, work)
    return len(accumulated)


def decayed_weights(
    store: AssociationStore,
    terms: Sequence[str],
    now_ms: Optional[int] = None,
) -> List[AssociationWeight]:
    """Return association weights for a query's terms, decayed to *now_ms* and summed per skill.

    Only live skills are returned. An ambient skill is filtered out here rather than deleted from
    the table, because the ubiquity ratio has to stay measurable from the same rows; the association
    channel simply contributes no rank for it.
    """
    stamp = _now() if now_ms is None else int(now_ms)
    unique = [term for term in dict.fromkeys(terms) if term]
    if not unique:
        return []
    placeholders = ", ".join("?" for _ in unique)
    rows = store.conn.execute(
        f"""SELECT a.skill_id AS skill_id, a.weight AS weight, a.last_ms AS last_ms
            FROM association a JOIN skill s ON s.id = a.skill_id
            WHERE s.deleted_at_ms IS NULL AND a.term IN ({placeholders})""",
        tuple(unique),
    ).fetchall()
    if not rows:
        return []
    ambient = ambient_skill_ids(store, stamp)
    by_skill: Dict[int, float] = {}
    for row in rows:
        skill_id = int(row["skill_id"])
        if skill_id in ambient:
            continue
        decayed = float(row["weight"]) * decay_factor(stamp - int(row["last_ms"]))
        by_skill[skill_id] = by_skill.get(skill_id, 0.0) + decayed
    if not by_skill:
        return []
    support = _skill_support(store)
    weights = [
        AssociationWeight(skill_id=skill_id, weight=weight, support=support.get(skill_id, 0))
        for skill_id, weight in by_skill.items()
    ]
    weights.sort(key=lambda item: (-item.weight, item.skill_id))
    return weights


def _skill_support(store: AssociationStore) -> Dict[int, int]:
    """Return distinct eligible prompts that loaded each skill, over the whole retained store."""
    rows = store.conn.execute(
        """SELECT skill_id AS skill_id, count(DISTINCT prompt_event_id) AS n
           FROM load_event GROUP BY skill_id"""
    ).fetchall()
    return {int(row["skill_id"]): int(row["n"]) for row in rows}


def eligible_prompt_count(store: AssociationStore, now_ms: Optional[int] = None) -> int:
    """Return distinct eligible prompts in the trailing ubiquity window."""
    stamp = _now() if now_ms is None else int(now_ms)
    row = store.conn.execute(
        "SELECT count(*) FROM prompt_event WHERE observed_ms >= ?", (stamp - UBIQUITY_WINDOW_MS,)
    ).fetchone()
    return int(row[0]) if row is not None else 0


def skill_prompt_counts(store: AssociationStore, now_ms: Optional[int] = None) -> Dict[int, int]:
    """Return how many prompts loaded each skill in the trailing ubiquity window."""
    stamp = _now() if now_ms is None else int(now_ms)
    rows = store.conn.execute(
        """SELECT le.skill_id AS skill_id, count(DISTINCT le.prompt_event_id) AS n
           FROM load_event le JOIN prompt_event pe ON pe.id = le.prompt_event_id
           WHERE pe.observed_ms >= ? GROUP BY le.skill_id""",
        (stamp - UBIQUITY_WINDOW_MS,),
    ).fetchall()
    return {int(row["skill_id"]): int(row["n"]) for row in rows}


def ambient_skill_ids(store: AssociationStore, now_ms: Optional[int] = None) -> set:
    """Return the skills the ubiquity gate marks ambient.

    They stay in the tables so the ratio stays measurable, but they never produce an association
    rank.
    """
    stamp = _now() if now_ms is None else int(now_ms)
    prompts = eligible_prompt_count(store, stamp)
    ambient: set = set()
    if prompts == 0:
        return ambient
    for skill_id, count in skill_prompt_counts(store, stamp).items():
        if is_ambient(count, prompts):
            ambient.add(skill_id)
    return ambient


def maintenance(
    store: AssociationStore,
    *,
    catalog: Optional[Sequence[CatalogSkill]] = None,
    now_ms: Optional[int] = None,
    vacuum: bool = True,
) -> MaintenanceReport:
    """Reconcile, purge, rebuild and optionally vacuum.

    Nothing on a request path may call this. The purge removes a soft-deleted skill and its rows only
    after :data:`DELETED_SKILL_RETENTION_MS`, and prompts older than :data:`PROMPT_RETENTION_MS`;
    ``association`` is then rebuilt from what remains. A read-only open refuses here, because the
    rebuild always writes and ``sqlite3`` would otherwise raise its own ``OperationalError``.
    """
    if store.read_only:
        raise AssociationStoreError("associations: maintenance needs a writable store")
    stamp = _now() if now_ms is None else int(now_ms)
    reconciled = None if catalog is None else reconcile_catalog(store, catalog, stamp)
    counts = {"deleted": 0, "prompts": 0}

    def work() -> None:
        deleted = store.conn.execute(
            "DELETE FROM skill WHERE deleted_at_ms IS NOT NULL AND deleted_at_ms < ?",
            (stamp - DELETED_SKILL_RETENTION_MS,),
        )
        counts["deleted"] = int(deleted.rowcount)
        prompts = store.conn.execute(
            "DELETE FROM prompt_event WHERE observed_ms < ?", (stamp - PROMPT_RETENTION_MS,)
        )
        counts["prompts"] = int(prompts.rowcount)

    _transaction(store, work)
    association_rows = recompute_associations(store)
    vacuumed = False
    if vacuum and not store.read_only:
        store.conn.execute("VACUUM")
        vacuumed = True
    return MaintenanceReport(
        catalog=reconciled,
        deleted_skills_removed=counts["deleted"],
        stale_prompts_removed=counts["prompts"],
        association_rows=association_rows,
        vacuumed=vacuumed,
    )


def reset_associations(
    store: AssociationStore, *, skills: Optional[Sequence[Any]] = None
) -> ResetReport:
    """Forget associations: the named skills' loads and rows, or every learned row when none is named.

    The vector index is not touched and is not even known to this module: the association file is the
    only thing a reset may change. Named skills keep their ``skill`` row, because that row is catalog
    state rather than learned evidence. A read-only open refuses here, for the same reason
    :func:`maintenance` does.
    """
    if store.read_only:
        raise AssociationStoreError("associations: reset needs a writable store")
    counts = {"skills": 0, "load_events_removed": 0, "association_rows_removed": 0}
    if skills is None:

        def clear_all() -> None:
            counts["load_events_removed"] = int(store.conn.execute("DELETE FROM load_event").rowcount)
            counts["association_rows_removed"] = int(
                store.conn.execute("DELETE FROM association").rowcount
            )
            store.conn.execute("DELETE FROM prompt_event")
            counts["skills"] = int(store.conn.execute("DELETE FROM skill").rowcount)
            # The catalog fingerprint and the scan watermark describe rows this reset just removed.
            # Left behind, the fingerprint would make the next reconcile skip the empty skill table,
            # and the watermark would make the next incremental pass skip every session on disk.
            write_meta(store, META_CATALOG_FINGERPRINT, "")
            write_meta(store, META_LAST_SCAN_MS, "0")

        _transaction(store, clear_all)
        return ResetReport(**counts)

    targets = _resolve_reset_targets(store, skills)
    if not targets:
        return ResetReport()
    placeholders = ", ".join("?" for _ in targets)
    params = tuple(targets)

    def clear_selected() -> None:
        counts["skills"] = len(targets)
        counts["load_events_removed"] = int(
            store.conn.execute(
                f"DELETE FROM load_event WHERE skill_id IN ({placeholders})", params
            ).rowcount
        )
        counts["association_rows_removed"] = int(
            store.conn.execute(
                f"DELETE FROM association WHERE skill_id IN ({placeholders})", params
            ).rowcount
        )

    _transaction(store, clear_selected)
    recompute_associations(store)
    return ResetReport(**counts)


def _resolve_reset_targets(store: AssociationStore, targets: Sequence[Any]) -> List[int]:
    """Resolve skill ids or names to the ids a reset should forget."""
    ids: List[int] = []
    for target in targets:
        if isinstance(target, int) and not isinstance(target, bool):
            ids.append(target)
            continue
        if not isinstance(target, str):
            continue
        row = resolve_skill_name(store, target)
        if row is not None:
            ids.append(row.id)
    return list(dict.fromkeys(ids))


def delete_association_database(store: AssociationStore) -> None:
    """Delete the association database itself: the file, its WAL and its shared-memory sibling.

    This is the documented reset that loses only learned data. The store is closed by the call and
    must not be used afterwards; reopening the path creates an empty version-1 database.
    """
    path = store.path
    store.close()
    for candidate in (path, path + "-wal", path + "-shm"):
        if os.path.exists(candidate):
            os.unlink(candidate)


def association_stats(store: AssociationStore, now_ms: Optional[int] = None) -> AssociationStats:
    """Return row counts for a report or a test."""

    def count(table: str) -> int:
        row = store.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row is not None else 0

    deleted = store.conn.execute(
        "SELECT count(*) FROM skill WHERE deleted_at_ms IS NOT NULL"
    ).fetchone()
    return AssociationStats(
        skills=count("skill"),
        deleted_skills=int(deleted[0]) if deleted is not None else 0,
        aliases=count("skill_alias"),
        prompts=count("prompt_event"),
        loads=count("load_event"),
        associations=count("association"),
        ambient_skills=len(ambient_skill_ids(store, now_ms)),
    )


def _transaction(store: AssociationStore, work: Callable[[], None]) -> None:
    """Run *work* in one immediate transaction, rolling back on any exception."""
    store.conn.execute("BEGIN IMMEDIATE")
    try:
        work()
        store.conn.execute("COMMIT")
    except Exception:
        try:
            store.conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise


# --------------------------------------------------------------------------------------------------
# The backfill: walk the Hermes session store, attribute each load, write the eligible ones
# --------------------------------------------------------------------------------------------------


@dataclass
class BackfillReport:
    """What one backfill run did."""

    sessions_found: int = 0
    sessions_scanned: int = 0
    sessions_skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    prompts: int = 0
    loads: int = 0
    duplicates: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    ambient_loads: int = 0
    unreadable_results: int = 0
    last_scan_ms: int = 0
    ms: int = 0


#: Why a session row was not read or written. Every one of them is counted, never thrown.
SKIP_REASONS: Tuple[str, ...] = ("no-messages", "unreadable", "write-failed")


@dataclass
class _Session:
    """One session row as the walk needs it."""

    id: str
    source: str
    stamp_ms: int


@dataclass
class _Prompt:
    """One typed prompt of a session, at its message position."""

    index: int
    text: str
    observed_ms: int


@dataclass
class _Load:
    """One load seen in a session, before the gate decides."""

    evidence: str
    strength: float
    after_skill_md: bool
    name: Optional[str]
    path: Optional[str]
    skill_path: str
    index: int
    observed_ms: int


class _Resolver:
    """Resolve a Hermes load to one identity: the catalog name and the best-known path.

    The live catalog answers first. The store answers for a skill the catalog dropped, whose row
    survives its retention window and whose alias table still maps a name an old load used. A name
    that resolves nowhere is not keyed by name, because a reused name is not evidence of the same
    skill.
    """

    def __init__(self, store: AssociationStore, catalog: Sequence[CatalogSkill]) -> None:
        self._store = store
        self._by_name: Dict[str, Tuple[str, str]] = {}
        self._by_path: Dict[str, Tuple[str, str]] = {}
        self._by_dir: Dict[str, Tuple[str, str]] = {}
        for skill in catalog:
            resolved = (skill.name, skill.resolved_path())
            self._by_name.setdefault(skill.name, resolved)
            if skill.path:
                path = os.path.normpath(skill.path)
                self._by_path.setdefault(path, resolved)
                self._by_dir.setdefault(os.path.dirname(path), resolved)

    def by_name(self, name: str) -> Optional[Tuple[str, str]]:
        """Return the identity a skill name resolves to, or ``None``."""
        hit = self._by_name.get(name)
        if hit is not None:
            return hit
        row = resolve_skill_name(self._store, name)
        return None if row is None else (row.name, row.path)

    def by_read_path(self, path: str) -> Optional[Tuple[str, str]]:
        """Return the identity a read of a ``SKILL.md`` path resolves to, or ``None``.

        Hermes lays user skills out as ``<skills root>/<name>/SKILL.md`` and plugin skills under
        their plugin's directory, so the exact path is tried first and the parent directory name
        second. The catalog and the store both answer, in that order.
        """
        normalised = os.path.normpath(path)
        hit = self._by_path.get(normalised)
        if hit is not None:
            return hit
        directory = os.path.dirname(normalised)
        hit = self._by_dir.get(directory)
        if hit is not None:
            return hit
        row = resolve_skill_path(self._store, path)
        if row is not None:
            return (row.name, row.path)
        parent = os.path.basename(directory)
        return self.by_name(parent) if parent else None

    def by_reference_path(self, path: str) -> Optional[Tuple[str, str]]:
        """Return the identity a read of a file under ``references/`` resolves to, or ``None``.

        The file lives at ``<skill dir>/references/<file>``, so the directory named before that
        segment is the skill's directory and, for Hermes, its name. The catalog answers first when
        it carries that path; the store and its aliases answer for a skill the catalog dropped.
        A path with no resolvable skill is ignored rather than counted, exactly as T1 returns
        without appending when its directory lookup misses.
        """
        normalised = os.path.normpath(path)
        head, segment, _tail = normalised.rpartition(REFERENCE_SEGMENT)
        if not head or not segment:
            return None
        hit = self._by_dir.get(head)
        if hit is not None:
            return hit
        name = os.path.basename(head)
        return self.by_name(name) if name else None


def backfill(
    store: AssociationStore,
    state_db_path: str,
    *,
    catalog: Sequence[CatalogSkill] = (),
    now_ms: Optional[int] = None,
    max_sessions: Optional[int] = None,
) -> BackfillReport:
    """Fill the association store from the Hermes session store.

    *state_db_path* is normally ``/home/wilder/.hermes/state.db``. It is opened read-only and never
    copied: the store is shared with the live gateway. The association *store* must already be open
    for the ``hermes`` harness; this function reconciles a non-empty *catalog* into it, then writes
    only loads that pass the gate.

    The run is idempotent, skips an unreadable or message-less session, and a session whose write
    fails, counting each under its skip reason instead of aborting the pass, and rewrites nothing it
    already stored: uniqueness in the store, not a check here, is what makes a second run insert
    nothing. ``max_sessions`` reads only the newest sessions and leaves the scan watermark alone,
    because a bounded run must not make a later incremental pass skip the rest.

    Privacy: prompt text exists only in this function's memory. What reaches the store is the
    normalised terms and a hash of them, so a database read cannot recover what the user typed.
    """
    started = time.perf_counter()
    stamp = _now() if now_ms is None else int(now_ms)
    if catalog:
        reconcile_catalog(store, catalog, stamp)
    resolver = _Resolver(store, catalog)
    report = BackfillReport(
        skip_reasons={reason: 0 for reason in SKIP_REASONS},
        rejected={condition: 0 for condition in GATE_CONDITIONS},
    )

    source = _open_state_db(state_db_path)
    try:
        sessions = _list_sessions(source, max_sessions)
        report.sessions_found = _count_sessions(source)
        before = _last_load_id(store)
        for session in sessions:
            rows = _session_rows(source, session.id)
            if rows is None:
                report.skip_reasons["unreadable"] += 1
                report.sessions_skipped += 1
                continue
            if not rows:
                report.skip_reasons["no-messages"] += 1
                report.sessions_skipped += 1
                continue
            try:
                scan = _scan_session(session, rows, resolver)
                _write_session(store, session, scan, report)
            except sqlite3.Error:
                # One odd session must not stop a store-wide backfill: its transaction already
                # rolled back, so count it as skipped and keep walking.
                report.skip_reasons["write-failed"] += 1
                report.sessions_skipped += 1
                continue
            report.unreadable_results += scan["unreadable"]
            report.sessions_scanned += 1
            report.last_scan_ms = max(report.last_scan_ms, session.stamp_ms)
    finally:
        source.close()

    # The ubiquity condition is a property of the whole store, so it is decided once, after every
    # session is in, and only for the loads this run inserted.
    report.ambient_loads = _count_ambient_loads(store, before, stamp)
    if report.last_scan_ms > 0 and max_sessions is None:
        previous = _meta_int(store, META_LAST_SCAN_MS)
        if report.last_scan_ms > previous:
            write_meta(store, META_LAST_SCAN_MS, str(report.last_scan_ms))
    report.ms = int((time.perf_counter() - started) * 1000)
    return report


def _open_state_db(path: str) -> sqlite3.Connection:
    """Open the Hermes store read-only, with a query timeout, and never copy it."""
    uri = pathlib.Path(os.path.abspath(os.path.expanduser(path))).as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_MS / 1000.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    return conn


def _count_sessions(source: sqlite3.Connection) -> int:
    """Return how many sessions the store holds, read or skipped."""
    row = source.execute("SELECT count(*) FROM sessions").fetchone()
    return int(row[0]) if row is not None else 0


def _list_sessions(source: sqlite3.Connection, max_sessions: Optional[int]) -> List[_Session]:
    """Return the session rows to read, newest first, bounded when asked."""
    limit = "" if max_sessions is None else f" LIMIT {max(0, int(max_sessions))}"
    rows = source.execute(
        f"""SELECT id, source, COALESCE(last_activity_at, started_at) AS stamp
            FROM sessions ORDER BY stamp DESC, id DESC{limit}"""
    ).fetchall()
    sessions: List[_Session] = []
    for row in rows:
        stamp = row["stamp"]
        sessions.append(
            _Session(
                id=str(row["id"]),
                source=str(row["source"] or ""),
                stamp_ms=int(round(float(stamp) * 1000)) if stamp is not None else 0,
            )
        )
    return sessions


def _session_rows(source: sqlite3.Connection, session_id: str) -> Optional[List[sqlite3.Row]]:
    """Return one session's messages in order, or ``None`` when they cannot be read."""
    try:
        return source.execute(
            """SELECT id, role, content, tool_calls, tool_call_id, tool_name, timestamp,
                      display_kind, compacted, active
               FROM messages WHERE session_id = ? ORDER BY id""",
            (session_id,),
        ).fetchall()
    except sqlite3.Error:
        return None


def provenance_of(source: str) -> str:
    """Classify a session source into the provenance the gate and the store record.

    Automation sources (kanban, cron, oneshot) keep their evidence and are flagged, because 84% of
    the store is automation and its prompts are real task descriptions. A subagent session is
    refused, because its prompt is the parent's brief. A source this build does not know answers
    ``unknown`` and fails closed at the gate, and the store never records a provenance the session
    does not have by mislabelling it ``subagent``.
    """
    if source in HUMAN_SOURCES:
        return "human"
    if source in AUTOMATION_SOURCES:
        return "automation"
    if source in SUBAGENT_SOURCES:
        return "subagent"
    return "unknown"


def _scan_session(
    session: _Session, rows: Sequence[sqlite3.Row], resolver: _Resolver
) -> Dict[str, Any]:
    """Walk one session's messages and return its parsed prompts and loads.

    State is the design's: the nearest preceding typed prompt, and the skills whose ``SKILL.md`` was
    loaded so far, which is what makes a reference read countable. Records this backfill does not
    use are ignored rather than trusted.

    ``prompts`` holds every typed prompt, because attribution must stop at the nearest one even when
    the gate will refuse it: a load after ``continue`` belongs to ``continue`` and is dropped, not
    re-paired with the task prompt that came before it. ``eligible`` holds the subset the ubiquity
    denominator counts, which is conditions 2 and 3 without a load attached.
    """
    origin = provenance_of(session.source)
    prompts: List[_Prompt] = []
    eligible: List[_Prompt] = []
    loads: List[_Load] = []
    pending: Dict[str, Dict[str, Any]] = {}
    skill_md_seen: set = set()
    prompt: Optional[_Prompt] = None
    resets: List[int] = []
    unreadable = 0
    for position, row in enumerate(rows):
        role = str(row["role"] or "")
        stamp = _stamp_of(row["timestamp"])
        if role == "user":
            updated = _handle_user(
                row, position, stamp, resolver, prompts, eligible, loads, skill_md_seen, prompt
            )
            if prompt is not None and updated is None:
                resets.append(position)
            prompt = updated
        elif role == "assistant":
            pending.update(_pending_calls(row, position, stamp))
        elif role == "tool":
            call_id = row["tool_call_id"]
            call = pending.pop(str(call_id), None) if call_id is not None else None
            if call is not None:
                unreadable += _resolve_tool_result(row, call, resolver, loads, skill_md_seen)
    return {
        "origin": origin,
        "prompts": prompts,
        "eligible": eligible,
        "loads": loads,
        "resets": resets,
        "unreadable": unreadable,
    }


def _handle_user(
    row: sqlite3.Row,
    position: int,
    stamp: int,
    resolver: _Resolver,
    prompts: List[_Prompt],
    eligible: List[_Prompt],
    loads: List[_Load],
    skill_md_seen: set,
    prompt: Optional[_Prompt],
) -> Optional[_Prompt]:
    """Classify one user row, and record a typed ``/name`` as a load when it names a skill.

    A compacted or deactivated row ends the attribution window and leaves no prompt, so the loads
    that follow it are dropped rather than paired with a prompt the user had already moved past. An
    injected row is not a boundary at all: it is not the user's typing, and treating it as one would
    let a skill body or a harness nudge hide the real prompt behind it.
    """
    text = row["content"] if isinstance(row["content"], str) else ""
    kind = _prompt_kind(row, text)
    if kind == "compacted":
        return None
    if kind != PROMPT_SOURCE_USER:
        return prompt
    current = _Prompt(index=position, text=text, observed_ms=stamp)
    prompts.append(current)
    if is_eligible_prompt(PromptEvidence(text=text, source_kind=PROMPT_SOURCE_USER, index=position)):
        eligible.append(current)
    slash = _slash_name(text)
    if slash is not None:
        resolved = resolver.by_name(slash)
        if resolved is not None:
            skill_md_seen.add(resolved[0])
        loads.append(
            _Load(
                evidence="slash",
                strength=EXPLICIT_STRENGTH,
                after_skill_md=resolved is not None and resolved[0] in skill_md_seen,
                name=None if resolved is None else resolved[0],
                path=None if resolved is None else resolved[1],
                skill_path="" if resolved is None else resolved[1],
                index=position,
                observed_ms=stamp,
            )
        )
    return current


def _prompt_kind(row: sqlite3.Row, text: str) -> str:
    """Return ``user`` only for the user's own typing, and the harness's own kind otherwise.

    Hermes writes injected material into user rows: a loaded skill's body after a typed ``/name``, a
    system continuation nudge, an image caption, an out-of-band steer and a process-completion
    notice. None of it is the user's wording, so none of it becomes prompt terms. A compacted row (or
    a row a rewind deactivated) is a summary, not the message that was typed: it answers
    ``compacted``, and the walk stops attributing across it.
    """
    if row["compacted"] or not row["active"]:
        return "compacted"
    kind = row["display_kind"]
    if isinstance(kind, str) and kind:
        return kind
    for marker in INJECTED_PROMPT_MARKERS:
        if text.lstrip().startswith(marker):
            return "injected"
    return PROMPT_SOURCE_USER


def _slash_name(text: str) -> Optional[str]:
    """Return the skill name of a typed ``/name`` prompt, or ``None``."""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    if not parts:
        return None
    name = parts[0][1:]
    return name if len(name) >= 2 else None


def _pending_calls(
    row: sqlite3.Row, position: int, stamp: int
) -> Dict[str, Dict[str, Any]]:
    """Return the ``skill_view`` and ``read_file`` calls of one assistant row, keyed by call id."""
    pending: Dict[str, Dict[str, Any]] = {}
    for call in _tool_calls(row["tool_calls"]):
        name = call.get("name")
        call_id = call.get("call_id")
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        arguments = call.get("arguments")
        if name == "skill_view":
            pending[call_id] = {
                "kind": "skill_view",
                "file_path": arguments.get("file_path"),
                "position": position,
                "stamp": stamp,
            }
        elif name == "read_file":
            entry = _read_file_pending(arguments)
            if entry is not None:
                pending[call_id] = {**entry, "position": position, "stamp": stamp}
    return pending


def _read_file_pending(arguments: Any) -> Optional[Dict[str, Any]]:
    """Return the pending entry for a ``read_file`` worth following, or ``None``.

    The live Hermes store writes the argument as ``path``; older rows and the test fixture may
    carry ``file_path``, so both are read and neither is trusted alone. A ``SKILL.md`` read is a
    full-strength load and a read under ``references/`` is the follow-on form; any other file is not
    a skill load and is dropped here rather than turned into evidence.
    """
    file_path = arguments.get("file_path") or arguments.get("path")
    if not isinstance(file_path, str) or not file_path:
        return None
    if file_path.endswith("/SKILL.md"):
        kind = "read_file"
    elif REFERENCE_SEGMENT in file_path:
        kind = "read_reference"
    else:
        return None
    return {"kind": kind, "file_path": file_path}


def _tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Return the tool calls of one assistant row as ``{name, call_id, arguments}`` dicts.

    A malformed row yields no calls rather than failing the session: one odd record must not stop a
    walk over the whole store.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(parsed, list):
        return []
    calls: List[Dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments")
        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            parsed_arguments = None
        if not isinstance(parsed_arguments, dict):
            parsed_arguments = {}
        calls.append(
            {
                "name": function.get("name"),
                "call_id": item.get("call_id") or item.get("id"),
                "arguments": parsed_arguments,
            }
        )
    return calls


def _resolve_tool_result(
    row: sqlite3.Row,
    call: Dict[str, Any],
    resolver: _Resolver,
    loads: List[_Load],
    skill_md_seen: set,
) -> int:
    """Turn one matched tool result into a load, and return 1 when its content was unreadable.

    A ``skill_view`` by name is the harness's tool call, so it carries the ``tool`` evidence kind,
    exactly as a DSH ``skill`` call would; a ``read_file`` of a ``SKILL.md`` path carries
    ``skill_md_read``. The two are the design's "both count equally" case: one strength, two
    provenance kinds, so a ranking can say which one produced it. A read of a file under
    ``references/`` carries ``tool`` at the follow-on strength and needs the same skill's
    ``SKILL.md`` earlier in the session; a reference read that resolves to no skill is ignored, as
    T1 returns without appending. A ``read_file`` call is a load on its own, at the path it read, so
    it needs no result body.

    A reference read in the same prompt window as its skill's ``SKILL.md`` load is absorbed by that
    load, because ``load_event.dedup_key`` covers the skill and the evidence kind and the design
    says a skill loaded twice in one turn counts once. The follow-on weight is what a reference read
    carries when it is the window's only evidence.
    """
    if call["kind"] in ("read_file", "read_reference"):
        load = _read_file_load(call, resolver, skill_md_seen)
        if load is not None:
            loads.append(load)
        return 0
    payload = _result_json(row["content"])
    if payload is None:
        return 1
    if payload.get("success") is False:
        return 0
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return 0
    loads.append(_skill_view_load(call, payload, name, resolver, skill_md_seen))
    return 0


def _read_file_load(
    call: Dict[str, Any], resolver: _Resolver, skill_md_seen: set
) -> Optional[_Load]:
    """Return the load a ``read_file`` represents, or ``None`` when a reference read resolves none."""
    if call["kind"] == "read_reference":
        resolved = resolver.by_reference_path(call["file_path"])
        if resolved is None:
            return None
        return _Load(
            evidence="tool",
            strength=FOLLOW_ON_STRENGTH,
            after_skill_md=resolved[0] in skill_md_seen,
            name=resolved[0],
            path=resolved[1],
            skill_path=call["file_path"],
            index=call["position"],
            observed_ms=call["stamp"],
        )
    resolved = resolver.by_read_path(call["file_path"])
    if resolved is not None:
        skill_md_seen.add(resolved[0])
    return _Load(
        evidence="skill_md_read",
        strength=EXPLICIT_STRENGTH,
        after_skill_md=resolved is not None,
        name=None if resolved is None else resolved[0],
        path=None if resolved is None else resolved[1],
        skill_path=call["file_path"],
        index=call["position"],
        observed_ms=call["stamp"],
    )


def _skill_view_load(
    call: Dict[str, Any],
    payload: Dict[str, Any],
    name: str,
    resolver: _Resolver,
    skill_md_seen: set,
) -> _Load:
    """Return the load a ``skill_view`` result represents, at the strength its ``file=`` decides."""
    resolved = resolver.by_name(name)
    path = payload.get("_source_path")
    if not isinstance(path, str) or not path:
        path = None if resolved is None else resolved[1]
    skill_path = path if path else ("" if resolved is None else resolved[1])
    file_path = call["file_path"]
    if file_path is None or str(file_path).endswith("SKILL.md"):
        skill_md_seen.add(name)
        return _Load(
            evidence="tool",
            strength=EXPLICIT_STRENGTH,
            after_skill_md=True,
            name=name,
            path=path,
            skill_path=skill_path,
            index=call["position"],
            observed_ms=call["stamp"],
        )
    return _Load(
        evidence="tool",
        strength=FOLLOW_ON_STRENGTH,
        after_skill_md=name in skill_md_seen,
        name=name,
        path=path,
        skill_path=skill_path,
        index=call["position"],
        observed_ms=call["stamp"],
    )


def _result_json(content: Any) -> Optional[Dict[str, Any]]:
    """Parse a ``tool`` row's content as one JSON object, or return ``None`` when it is not one."""
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _write_session(
    store: AssociationStore,
    session: _Session,
    scan: Dict[str, Any],
    report: BackfillReport,
) -> None:
    """Gate one session's loads, store the admitted ones, and record the counts.

    The ubiquity denominator is every eligible prompt, not only the ones that loaded something. A
    subagent or unclassified session contributes neither prompt nor load, since the whole session is
    excluded; its origin is still classified truthfully. Counts reach *report* only after every
    write in the session succeeded, so a failed write is not reported as stored evidence.
    """
    origin = scan["origin"]
    stored_prompts = 0
    if origin in ("human", "automation") and scan["eligible"]:
        stored_prompts = upsert_prompts(
            store,
            [
                PromptEventInput(
                    session_id=session.id,
                    turn=prompt.index,
                    text=prompt.text,
                    observed_ms=prompt.observed_ms,
                )
                for prompt in scan["eligible"]
            ],
        )
    events: List[AssociationEvent] = []
    for load, nearest in _attributions(scan["prompts"], scan["loads"], scan["resets"]):
        event = _admitted_event(store, session, load, nearest, origin, report)
        if event is not None:
            events.append(event)
    upserted = upsert_events(store, events)
    report.prompts += stored_prompts + upserted.prompts
    report.loads += upserted.loads
    report.duplicates += upserted.duplicates


def _attributions(
    prompts: Sequence[_Prompt], loads: Sequence[_Load], resets: Sequence[int]
) -> List[Tuple[_Load, Optional[_Prompt]]]:
    """Pair each load with the nearest typed prompt at or before its position.

    Loads are sorted by position first, so the prompt cursor only moves forward: a result row that
    arrives after a later prompt can never rewind the attribution to an earlier one. A compacted row
    between the prompt and the load breaks the window, so that load is left unattributed rather than
    paired with a prompt the user had already moved past.
    """
    ordered = sorted(loads, key=lambda load: load.index)
    pairs: List[Tuple[_Load, Optional[_Prompt]]] = []
    cursor = 0
    current: Optional[_Prompt] = None
    for load in ordered:
        while cursor < len(prompts) and prompts[cursor].index <= load.index:
            current = prompts[cursor]
            cursor += 1
        if current is not None and any(current.index <= reset <= load.index for reset in resets):
            pairs.append((load, None))
            continue
        pairs.append((load, current))
    return pairs


def _admitted_event(
    store: AssociationStore,
    session: _Session,
    load: _Load,
    nearest: Optional[_Prompt],
    origin: str,
    report: BackfillReport,
) -> Optional[AssociationEvent]:
    """Gate one load and return the event to store, or ``None`` when the gate refused it."""
    resolved = None if load.name is None else resolve_skill_name(store, load.name)
    skill_id = None if resolved is None else resolved.id
    prompt = None
    if nearest is not None:
        prompt = PromptEvidence(
            text=nearest.text, source_kind=PROMPT_SOURCE_USER, index=nearest.index
        )
    event = LoadEvent(
        evidence=load.evidence,
        strength=load.strength,
        skill_id=skill_id,
        prompt=prompt,
        index=load.index,
        nearest_prompt_index=None if nearest is None else nearest.index,
        skill_path=load.skill_path,
        after_skill_md=load.after_skill_md,
        origin=origin,
    )
    reason = gate_rejection(event)
    if reason is not None:
        report.rejected[reason] += 1
        return None
    if skill_id is None or nearest is None or load.name is None:
        # Unreachable while the gate is the only decider, but never invent an identity.
        report.rejected["explicit-load"] += 1
        return None
    return AssociationEvent(
        session_id=session.id,
        turn=nearest.index,
        prompt_text=nearest.text,
        prompt_observed_ms=nearest.observed_ms,
        skill_name=load.name,
        skill_path=load.path if load.path else load.skill_path,
        evidence=load.evidence,
        strength=load.strength,
        origin=origin,
        source="backfill",
        observed_ms=load.observed_ms,
    )


def _count_ambient_loads(store: AssociationStore, watermark: int, now_ms: int) -> int:
    """Return the loads this run stored whose skill the ubiquity gate excludes.

    They stay in ``load_event``; the gate suppresses their rank, and the ratio that marks a skill
    ambient stays computable from the same rows. They are reported as ``ambient_loads``, not as a
    gate rejection: they were stored. The count covers this run's inserts only (``id > watermark``):
    the report describes one run, so a rerun reports 0 rather than the same number again.
    """
    ambient = ambient_skill_ids(store, now_ms)
    if not ambient:
        return 0
    counts = skill_prompt_counts(store, now_ms)
    eligible = eligible_prompt_count(store, now_ms)
    rows = store.conn.execute(
        "SELECT skill_id AS skill_id, count(*) AS n FROM load_event WHERE id > ? GROUP BY skill_id",
        (watermark,),
    ).fetchall()
    ambient_loads = 0
    for row in rows:
        skill_id = int(row["skill_id"])
        if skill_id not in ambient:
            continue
        if not is_ambient(counts.get(skill_id, 0), eligible):
            continue
        ambient_loads += int(row["n"])
    return ambient_loads


def _last_load_id(store: AssociationStore) -> int:
    """Return the newest ``load_event`` id, the watermark a run's inserts sit above."""
    row = store.conn.execute("SELECT COALESCE(MAX(id), 0) FROM load_event").fetchone()
    return int(row[0]) if row is not None else 0


def _stamp_of(value: Any) -> int:
    """Return an epoch-millisecond stamp from the store's epoch-second float, or 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(float(value)):
        return 0
    return int(round(float(value) * 1000))


def _now() -> int:
    """Return the current epoch milliseconds."""
    return int(time.time() * 1000)
