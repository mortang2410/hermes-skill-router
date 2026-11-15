"""Score-vs-association drift analytics (module M6, contract C5).

Compares the search score this script recomputes for every catalog skill
(semantic cosine plus lexical BM25, both min-max normalized per query and
averaged) against the historical association score replayed as of each
logged decision's timestamp, and prints the top-5 diverging prompts for
every skill whose divergence reaches :data:`DIVERGENCE_THRESHOLD`.

The logged candidate scores are not the search score: they are reranker
scores over the shortlist only, so a skill the shortlist cut never appears
in the decision log. This script therefore scores every catalog skill
itself. The catalog copy is the saved-index JSON (``names`` and
``descriptions`` arrays, exactly as ``library._cache_payload`` writes
them); the embed endpoint is required at script run, so the query and the
catalog document texts are embedded then.

The historical association score is never a later snapshot's aggregate
weight: over a store, that would report future associations as history.
The script replays ``load_event`` rows filtered by ``observed_ms``, each
counted at its observed strength and decayed from its observed time to
the log timestamp, over the query's unique normalized terms, normalized
per query by the maximum weight. Replay is taken only from a snapshot
with verifiable coverage provenance (a v2 enrollment record, no recorded
history-loss marker, and every source session through the run proven
final by the copy's ``scan_ledger`` against ``--state-db``) plus the
operator assertion ``--assert-complete-source-history``. A failed proof
fails the run with a named coverage error rather than scoring the missing
intervening events as zero.

The script is read-only: it opens the store and the session store with
``mode=ro`` connections and never writes either. Output is deterministic
for a fixed input copy.

Replay parity: the catalog copy records the ``embedModelId`` its saved
index was built with (``embed_model`` in the saved-index JSON, written by
``library`` at build time). The replay model, resolved from the explicit
flag or the workspace config, must equal that recorded model; otherwise
the run exits with a named parity error before any embedding call, never
scoring in the wrong vector space. A catalog written before the key
existed records no model, so the documented invocation without
``--embed-model`` is a parity error there too and the operator must name
the run's model explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hermes_skill_router import associations  # noqa: E402
from hermes_skill_router import bm25  # noqa: E402
from hermes_skill_router.embed import opener  # noqa: E402

#: Anomaly threshold: divergence at or above this flags a skill. Provisional.
DIVERGENCE_THRESHOLD = 0.4

#: Texts per embeddings request, the same batch size the library build uses.
BATCH_TEXTS = 16

#: The decision line the router writes per completed pass.
DECISION_PREFIX = "skill router: query="

#: ``meta`` key the destructive paths write inside their own transaction.
META_LAST_HISTORY_LOSS_MS = "last_history_loss_ms"

#: ``meta`` key phase zero of a fresh creation writes; migrated stores never
#: carry ``fresh``, which is what makes a hand-cleared marker detectable.
META_CREATION_ORIGIN = "creation_origin"
CREATION_ORIGIN_FRESH = "fresh"

#: A log-line timestamp: ISO-8601 or Python-logging asctime, UTC when the
#: zone is absent. Searched in the line prefix before the decision line.
_TIMESTAMP_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:[.,](\d{1,9}))?(Z|[+-]\d{2}:?\d{2})?"
)


class AnalyticsError(Exception):
    """A named analytics failure: the message names its kind."""


class CoverageError(AnalyticsError):
    """Replay coverage cannot be established: fail closed, never score zero."""


class IdentityError(AnalyticsError):
    """A catalog name does not identify one skill."""


class ParityError(AnalyticsError):
    """The replay would run in a different vector space than the logged run.

    The catalog copy records the ``embedModelId`` its saved index was
    built with; a replay naming any other model (or an invocation that
    cannot name the run's model at all) exits with this error instead of
    silently scoring in the wrong space.
    """


class EmbedError(AnalyticsError):
    """The embed endpoint is unreachable or answered unusable vectors."""


class StoreError(AnalyticsError):
    """The store or session store cannot be opened read-only."""


def _coverage(detail: str) -> CoverageError:
    return CoverageError(f"analytics coverage error: {detail}")


def _identity(detail: str) -> IdentityError:
    return IdentityError(f"analytics identity error: {detail}")


#--------------------------------------------------------------------------------------------------
# Inputs: catalog, decision log
#--------------------------------------------------------------------------------------------------


def document_text(name: str, description: str, max_description_chars: int) -> str:
    """Return the exact text embedded for one row: ``name``, newline, clipped description.

    Byte-identical to ``library.document_text`` for the same cap, which the
    tests pin against the live function.
    """
    return name + "\n" + description[:max_description_chars]


def read_catalog(path: str) -> Tuple[List[str], List[str], Optional[str]]:
    """Read the saved-index copy's ``names``, ``descriptions`` and build model.

    The third value is the ``embed_model`` the saved index was built with,
    or ``None`` for a catalog written before the key existed (a catalog
    carrying the key with a non-string or empty value is a parity error:
    the run's model cannot be recovered from it). A catalog repeating a
    name is an identity ambiguity and fails the run: the catalog cannot
    say which skill the name means.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise AnalyticsError(f"analytics catalog error: cannot read {path}: {exc}") from exc
    names = data.get("names") if isinstance(data, dict) else None
    descriptions = data.get("descriptions") if isinstance(data, dict) else None
    catalog_model: Optional[str] = None
    if isinstance(data, dict) and "embed_model" in data:
        recorded = data["embed_model"]
        if not isinstance(recorded, str) or not recorded.strip():
            raise ParityError(
                f"analytics parity error: catalog {path} records an unusable "
                f"embed_model {recorded!r}; the run's vector space cannot be proven"
            )
        catalog_model = recorded
    if not isinstance(names, list) or not isinstance(descriptions, list):
        raise AnalyticsError(
            f"analytics catalog error: {path} carries no names/descriptions arrays"
        )
    if len(names) != len(descriptions):
        raise AnalyticsError(
            f"analytics catalog error: {path} has {len(names)} names and "
            f"{len(descriptions)} descriptions"
        )
    if not all(isinstance(name, str) for name in names):
        raise AnalyticsError(f"analytics catalog error: {path} carries a non-string name")
    if not all(isinstance(desc, str) for desc in descriptions):
        raise AnalyticsError(
            f"analytics catalog error: {path} carries a non-string description"
        )
    seen: set = set()
    for name in names:
        if name in seen:
            raise _identity(f"catalog {path} repeats the name {name!r}")
        seen.add(name)
    return list(names), list(descriptions), catalog_model


def parse_log_timestamp_ms(prefix: str) -> Optional[int]:
    """Return the line-prefix timestamp as epoch milliseconds, or ``None``."""
    match = _TIMESTAMP_RE.search(prefix)
    if match is None:
        return None
    date_part, time_part, fraction, zone = match.groups()
    try:
        moment = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    if zone is None or zone == "Z":
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        sign = 1 if zone[0] == "+" else -1
        digits = zone[1:].replace(":", "")
        hours, minutes = int(digits[:2]), int(digits[2:4])
        offset = timezone(sign * timedelta(hours=hours, minutes=minutes))
        moment = moment.replace(tzinfo=offset)
    micros = 0
    if fraction:
        micros = int((fraction + "000000")[:6])
    return int(round(moment.replace(microsecond=micros).timestamp() * 1000))


def read_decision_log(path: str) -> List[Tuple[int, str]]:
    """Read the decision lines as ``(timestamp_ms, query)`` pairs.

    Lines that are not decision lines are ignored; a decision line whose
    ``query=`` is not a JSON string is skipped. A decision line with no
    parseable timestamp fails the run: a replay without a timestamp cannot
    be ordered against the store.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw_lines = handle.readlines()
    except OSError as exc:
        raise AnalyticsError(f"analytics log error: cannot read {path}: {exc}") from exc
    decoder = json.JSONDecoder()
    rows: List[Tuple[int, str]] = []
    for number, line in enumerate(raw_lines, 1):
        position = line.find(DECISION_PREFIX)
        if position < 0:
            continue
        rest = line[position + len(DECISION_PREFIX):].lstrip()
        try:
            value, _end = decoder.raw_decode(rest)
        except ValueError:
            continue
        if not isinstance(value, str):
            continue
        stamp = parse_log_timestamp_ms(line[:position])
        if stamp is None:
            raise AnalyticsError(
                f"analytics log error: cannot parse a timestamp from line {number} of {path}"
            )
        rows.append((stamp, value))
    return rows


#--------------------------------------------------------------------------------------------------
# Inputs: the association store copy and the Hermes session store (read-only)
#--------------------------------------------------------------------------------------------------


def open_readonly(path: str, kind: str) -> sqlite3.Connection:
    """Open one database read-only (``mode=ro`` + ``query_only``), never creating it."""
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise StoreError(f"analytics store error: the {kind} at {path} does not exist")
    uri = Path(os.path.abspath(expanded)).as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
    except sqlite3.Error as exc:
        raise StoreError(
            f"analytics store error: the {kind} at {path} cannot be opened read-only: {exc}"
        ) from exc
    return conn


def _meta_table(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return None
    return {str(row["key"]): str(row["value"]) for row in rows}


def check_provenance(conn: sqlite3.Connection) -> None:
    """Require a v2 enrollment record, no recorded history loss, and this harness.

    Order: a ``user_version``/``meta.schema_version`` disagreement is a named
    mismatch (never a pass); any presence of the loss marker fails with the
    marker named, regardless of its value (which is also why no
    millisecond-equal tie-break is attempted); a store stamped below v2
    fails as pre-v2; a migrated store whose marker was hand-deleted fails
    on the missing fresh creation provenance; a fresh v2 store from another
    harness fails on ``meta.harness`` (mirroring :func:`associations.open_store`,
    which refuses a foreign file), so a copy from another harness can never
    print clean or zero scores.
    """
    meta = _meta_table(conn)
    if meta is None:
        raise StoreError("analytics store error: the store carries no meta table")
    try:
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except (sqlite3.Error, TypeError, ValueError) as exc:
        raise StoreError(f"analytics store error: cannot read user_version: {exc}") from exc
    raw_version = meta.get("schema_version")
    declared: Optional[int] = None
    if raw_version is not None:
        try:
            declared = int(raw_version)
        except (TypeError, ValueError):
            declared = None
    if declared is None or declared != user_version:
        raise _coverage(
            "schema version mismatch: user_version is "
            f"{user_version} and meta.schema_version is {raw_version!r}"
        )
    if META_LAST_HISTORY_LOSS_MS in meta:
        raise _coverage(
            f"history-loss marker {META_LAST_HISTORY_LOSS_MS} is recorded; "
            "the analyzed history cannot be proven complete"
        )
    if declared < 2:
        raise _coverage(
            "pre-v2 store: no v2 enrollment record (schema_version is "
            f"{declared}); coverage cannot be established"
        )
    if meta.get(META_CREATION_ORIGIN) != CREATION_ORIGIN_FRESH:
        raise _coverage(
            "provenance untrustworthy: no fresh creation record "
            f"({META_CREATION_ORIGIN} is not {CREATION_ORIGIN_FRESH!r}); "
            "the store was not created under the v2 contract"
        )
    if meta.get(associations.META_HARNESS) != associations.HARNESS:
        raise _coverage(
            f"foreign harness: meta.harness is {meta.get(associations.META_HARNESS)!r}, not "
            f"{associations.HARNESS!r}; the store was not created under this router's contract"
        )


def _ledger_rows(conn: sqlite3.Connection) -> Dict[str, sqlite3.Row]:
    try:
        rows = conn.execute(
            "SELECT session_id, scanned_max_row_id, pending_calls, unreadable_results,"
            " unreadable_calls FROM scan_ledger"
        ).fetchall()
    except sqlite3.Error as exc:
        raise _coverage(f"scan_ledger cannot be read: {exc}") from exc
    return {str(row["session_id"]): row for row in rows}


def _seen_sessions(conn: sqlite3.Connection) -> List[Tuple[str, Optional[int]]]:
    try:
        rows = conn.execute("SELECT session_id, started_at_ms FROM seen_sessions").fetchall()
    except sqlite3.Error as exc:
        raise _coverage(f"seen_sessions cannot be read: {exc}") from exc
    return [(str(row["session_id"]), row["started_at_ms"]) for row in rows]


def _candidate_starts(
    store_conn: sqlite3.Connection, state_conn: sqlite3.Connection
) -> Dict[str, List[Optional[int]]]:
    """Return every start a session is known by, in epoch milliseconds."""
    starts: Dict[str, List[Optional[int]]] = {}
    try:
        rows = state_conn.execute("SELECT id, started_at FROM sessions").fetchall()
    except sqlite3.Error as exc:
        raise StoreError(f"analytics store error: sessions cannot be read: {exc}") from exc
    for row in rows:
        raw = row["started_at"]
        starts.setdefault(str(row["id"]), []).append(
            None if raw is None else int(round(float(raw) * 1000))
        )
    for session_id, started_ms in _seen_sessions(store_conn):
        starts.setdefault(session_id, []).append(
            None if started_ms is None else int(started_ms)
        )
    return starts


def _source_messages(
    state_conn: sqlite3.Connection, session_id: str
) -> List[int]:
    try:
        rows = state_conn.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
    except sqlite3.Error as exc:
        raise StoreError(f"analytics store error: messages cannot be read: {exc}") from exc
    return [int(row[0]) for row in rows]


def check_finality(
    store_conn: sqlite3.Connection, state_conn: sqlite3.Connection, latest_ms: int
) -> None:
    """Prove every candidate session is final in the copy, or fail closed.

    A candidate is every session that could contain rows at or before
    *latest_ms*: current-source sessions with ``started_at`` no later than
    that instant (NULL starts included conservatively), union the copy's
    durable ``seen_sessions`` with the same start rule. Each candidate needs
    a ledger row with zero ``pending_calls``, ``unreadable_results`` and
    ``unreadable_calls``; a non-empty scan needs ``scanned_max_row_id`` at
    least the session's current maximum message id (vacuous when the
    session is deleted) and every source message id strictly positive; an
    empty scan needs no source message rows. ``meta.last_scan_ms`` never
    gates replay.
    """
    starts = _candidate_starts(store_conn, state_conn)
    candidates = sorted(
        session_id
        for session_id, session_starts in starts.items()
        if any(start is None or start <= latest_ms for start in session_starts)
    )
    ledger = _ledger_rows(store_conn)
    for session_id in candidates:
        row = ledger.get(session_id)
        if row is None:
            raise _coverage(
                f"no scan_ledger row for candidate session {session_id!r}; "
                "the backfill never scanned it (or has not scanned it yet)"
            )
        for column in ("pending_calls", "unreadable_results", "unreadable_calls"):
            if row[column] != 0:
                raise _coverage(
                    f"session {session_id!r} ends with {column}={row[column]}; "
                    "the scan is not final"
                )
        messages = _source_messages(state_conn, session_id)
        scanned = row["scanned_max_row_id"]
        if scanned is None:
            if messages:
                raise _coverage(
                    f"session {session_id!r} has {len(messages)} source message row(s) "
                    "but an empty scan; the copy is stale"
                )
            continue
        if messages:
            if min(messages) <= 0:
                raise _coverage(
                    f"session {session_id!r} carries a non-positive message id; "
                    "ids are unsupported"
                )
            if int(scanned) < max(messages):
                raise _coverage(
                    f"session {session_id!r} has source activity after the scan "
                    f"(scanned_max_row_id={scanned}, current max={max(messages)})"
                )


#--------------------------------------------------------------------------------------------------
# Scoring: search (semantic + BM25) and replayed association
#--------------------------------------------------------------------------------------------------


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine of two vectors, ``0.0`` when either has no length.

    Byte-identical to ``library._cosine``, which the tests pin against the
    live function.
    """
    length = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for position in range(length):
        first = float(left[position])
        second = float(right[position])
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    score = dot / math.sqrt(left_norm * right_norm)
    return score if math.isfinite(score) else 0.0


def _minmax(values: Sequence[float]) -> List[float]:
    """Normalize to [0, 1] per query; a zero range maps every skill to 0."""
    scored = [float(value) for value in values]
    low = min(scored, default=0.0)
    high = max(scored, default=0.0)
    if high == low:
        return [0.0] * len(scored)
    return [(value - low) / (high - low) for value in scored]


def _parse_embeddings(raw: bytes, expected: int, url: str) -> List[List[float]]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EmbedError(
            f"analytics embed error: the embed endpoint {url} returned an unusable "
            f"body: {exc}"
        ) from exc
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) != expected:
        raise EmbedError(
            f"analytics embed error: the embed endpoint {url} returned "
            f"{len(data) if isinstance(data, list) else 'no'} vectors for {expected} texts"
        )
    vectors: List[List[float]] = []
    for entry in data:
        vector = entry.get("embedding") if isinstance(entry, dict) else None
        if not isinstance(vector, list) or not vector:
            raise EmbedError(
                f"analytics embed error: the embed endpoint {url} returned an "
                "unusable vector"
            )
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in vector
        ):
            raise EmbedError(
                f"analytics embed error: the embed endpoint {url} returned a "
                "non-numeric vector"
            )
        vectors.append([float(value) for value in vector])
    if len({len(vector) for vector in vectors}) != 1:
        raise EmbedError(
            f"analytics embed error: the embed endpoint {url} returned vectors "
            "of mixed widths"
        )
    return vectors


def embed_texts(
    url: str,
    texts: Sequence[str],
    model: str,
    timeout_s: float,
) -> List[List[float]]:
    """Embed *texts* through *url*, one vector per text in order, or raise.

    *model* and *timeout_s* are the run's own ``embedModelId`` and
    ``embedTimeoutMs`` (as seconds): the replay must use the same vector
    space and the same bound the router used. Both are optional CLI
    inputs defaulting to the workspace config's values (the
    ``plugins.entries.hermes-skill-router.settings`` entry in
    ``$HERMES_HOME/config.yaml``, default ``~/.hermes/config.yaml``),
    falling back to the builtin defaults where no custom config exists,
    so the documented invocation works unchanged and still replays in
    the same space. The explicit flags always override the workspace
    config. The workspace file is read as data only: no ``config.load``
    runs in this process, so a read-only analytics run never writes the
    router's warning-suppression state. The request reuses the router
    embed client's redirect-refusing opener; the response is used only
    when it carries exactly one non-empty vector per input at one
    common width. An unreachable endpoint and an unusable body both
    fail with the named embed error, never a partial result, and
    nothing is retried.
    """
    vectors: List[List[float]] = []
    for start in range(0, len(texts), BATCH_TEXTS):
        chunk = list(texts[start:start + BATCH_TEXTS])
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with opener().open(request, timeout=timeout_s) as response:
                raw = response.read()
        except EmbedError:
            raise
        except Exception as exc:  # HTTPError (incl. 3xx), URLError, timeout, reset
            raise EmbedError(
                f"analytics embed error: the embed endpoint {url} is unreachable: {exc}"
            ) from exc
        vectors.extend(_parse_embeddings(raw, len(chunk), url))
    return vectors


def bm25_raw_scores(index: bm25.TermIndex, text: str) -> List[float]:
    """Return the raw per-skill BM25 score the same equation ranks from.

    The arithmetic mirrors ``bm25.score_ranks`` exactly; only the scores
    are kept instead of the names. Tests pin the order against
    ``score_ranks``.
    """
    scores = [0.0] * len(index.documents)
    terms = bm25.tokenise(text)
    if not terms or index.average_length <= 0:
        return scores
    document_count = len(index.documents)
    for term in terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        document_frequency = len(postings) / 2
        inverse = math.log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for position in range(0, len(postings), 2):
            document_index = postings[position]
            frequency = postings[position + 1]
            length = index.documents[document_index].length
            normalisation = bm25.BM25_K1 * (
                1 - bm25.BM25_B + bm25.BM25_B * (length / index.average_length)
            )
            scores[document_index] += inverse * (
                (frequency * (bm25.BM25_K1 + 1)) / (frequency + normalisation)
            )
    return scores


def search_scores(
    names: Sequence[str],
    descriptions: Sequence[str],
    queries: Sequence[str],
    max_description_chars: int,
    embed_url: str,
    model: str,
    timeout_s: float,
) -> List[List[float]]:
    """Return one search score per query per catalog skill.

    Semantic cosine and lexical BM25 are each min-max normalized per query
    to [0, 1] (a zero range maps every skill to 0) and averaged into one
    search score. The lexical channel builds its term index from the same
    document texts the embedder sees, with the live router's tokenizer
    and pinned constants. *model* and *timeout_s* are the run's
    ``embedModelId``/``embedTimeoutMs`` seconds, passed through to
    :func:`embed_texts`.
    """
    documents = [
        document_text(name, description, max_description_chars)
        for name, description in zip(names, descriptions)
    ]
    index = bm25.build_term_index([(name, text) for name, text in zip(names, documents)])
    catalog_vectors = embed_texts(embed_url, documents, model, timeout_s)
    query_vectors = embed_texts(embed_url, list(queries), model, timeout_s)
    per_query: List[List[float]] = []
    for query, query_vector in zip(queries, query_vectors):
        semantic = _minmax(
            [cosine(query_vector, vector) for vector in catalog_vectors]
        )
        lexical = _minmax(bm25_raw_scores(index, query))
        per_query.append([(s + l) / 2.0 for s, l in zip(semantic, lexical)])
    return per_query


def replayed_weights(
    conn: sqlite3.Connection, query_terms: Sequence[str], ts_ms: int
) -> Dict[int, float]:
    """Return the replayed per-skill weights for one query at *ts_ms*.

    For each skill, sum the store's load events observed no later than
    *ts_ms*, each counted at its observed strength and decayed from its
    observed time to *ts_ms*, once per query term the load's prompt
    carries. Later aggregate weights are never read.
    """
    unique_terms = [term for term in dict.fromkeys(query_terms) if term]
    if not unique_terms:
        return {}
    try:
        rows = conn.execute(
            """SELECT le.skill_id AS skill_id, le.strength AS strength,
                      le.observed_ms AS observed_ms, pe.terms AS terms
               FROM load_event le JOIN prompt_event pe ON pe.id = le.prompt_event_id
               WHERE le.observed_ms <= ?""",
            (ts_ms,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise StoreError(
            f"analytics store error: load events cannot be replayed: {exc}"
        ) from exc
    wanted = set(unique_terms)
    weights: Dict[int, float] = {}
    for row in rows:
        carried = set(str(row["terms"]).split()) & wanted
        if not carried:
            continue
        contribution = float(row["strength"]) * associations.decay_factor(
            ts_ms - int(row["observed_ms"])
        )
        skill_id = int(row["skill_id"])
        weights[skill_id] = weights.get(skill_id, 0.0) + contribution * len(carried)
    return weights


def association_scores(
    conn: sqlite3.Connection,
    skill_ids: Sequence[Optional[int]],
    query_terms: Sequence[str],
    ts_ms: int,
) -> List[float]:
    """Return one normalized association score per catalog position.

    Normalized per query by the maximum weight over the catalog skills
    (a zero maximum maps every skill to 0). A catalog name resolving to
    no store row scores zero: genuinely never loaded, no history.
    """
    weights = replayed_weights(conn, query_terms, ts_ms)
    raw = [
        weights.get(skill_id, 0.0) if skill_id is not None else 0.0
        for skill_id in skill_ids
    ]
    top = max(raw, default=0.0)
    if top <= 0.0:
        return [0.0] * len(raw)
    return [value / top for value in raw]


#--------------------------------------------------------------------------------------------------
# The run
#--------------------------------------------------------------------------------------------------

ASSERTION_HELP = (
    "assert that no source session or message needed to reconstruct the "
    "analyzed history was deleted before the source was first inventoried "
    "or thereafter, including deletions before v2 enrollment"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare recomputed search scores against replayed historical "
            "association scores and print the top-5 anomaly prompts per "
            "anomalous skill."
        )
    )
    parser.add_argument("--store", required=True, help="copy of associations.db (read-only)")
    parser.add_argument("--log", required=True, help="plugin log with decision lines")
    parser.add_argument("--catalog", required=True, help="saved-index JSON copy")
    parser.add_argument("--embed-url", required=True, help="embeddings endpoint URL")
    parser.add_argument(
        "--embed-model",
        default=None,
        help="the embedModelId value the logged run used (the replay must stay in that "
        "vector space); defaults to the workspace config's embedModelId, then the "
        "builtin default where no custom config exists, when omitted. The catalog "
        "copy's recorded build model is the parity authority: a replay model that "
        "does not match it exits with a parity error, and a catalog recording no "
        "model requires this flag",
    )
    parser.add_argument(
        "--embed-timeout-ms",
        default=None,
        type=int,
        help="the embedTimeoutMs value the logged run used (100..60000); defaults to "
        "the workspace config's embedTimeoutMs, then the builtin default where no "
        "custom config exists, when omitted",
    )
    parser.add_argument(
        "--max-description-chars",
        required=True,
        type=int,
        help="the maxDescriptionChars value the logged run used (3-4000, default 400)",
    )
    parser.add_argument(
        "--state-db",
        default=associations.default_state_db_path(),
        help="Hermes session store the backfill reads (read-only)",
    )
    parser.add_argument(
        "--assert-complete-source-history",
        action="store_true",
        help=ASSERTION_HELP,
    )
    return parser


def _workspace_config_path() -> Path:
    """Return the workspace config path the router's plugin settings live in."""
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    return Path(os.path.expanduser(home)) / "config.yaml"


def _resolve_embed_defaults(
    need_model: bool = True, need_timeout: bool = True
) -> Tuple[str, int]:
    """Return ``(embed model, timeout ms)`` from the workspace config or the builtins.

    The workspace config is the same file the router reads through the plugin
    settings surface: ``plugins.entries.hermes-skill-router.settings`` in
    ``$HERMES_HOME/config.yaml`` (default ``~/.hermes/config.yaml``). A missing
    file, or a file naming no settings entry, means no custom config exists and
    the builtin defaults apply. A file that cannot be read or parsed, or that
    names an unusable ``embedModelId`` / ``embedTimeoutMs``, fails closed with
    :class:`AnalyticsError` naming the problem. No ``config.load`` runs here:
    ``load()`` persists the router's warning-suppression state, and a read-only
    analytics run must never write it.

    Only the fields the caller actually needs are validated: an explicit
    flag always overrides the workspace config (C5), so a workspace field
    the explicit flag overrides is never read for validation and its
    invalid value cannot fail the run. The file structure itself is still
    validated whenever any field is needed, because the needed field
    cannot be located in a malformed structure.
    """
    from hermes_skill_router.config import BOUNDS, DEFAULTS

    default_model = str(DEFAULTS["embedModelId"])
    default_timeout = int(DEFAULTS["embedTimeoutMs"])
    timeout_min, timeout_max = (int(bound) for bound in BOUNDS["embedTimeoutMs"])
    path = _workspace_config_path()
    if not path.exists():
        return default_model, default_timeout
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        # UnicodeError (invalid UTF-8) is a named unreadable-config failure
        # like any other, never a traceback.
        raise AnalyticsError(
            f"analytics error: cannot read workspace config {path}: {exc}"
        ) from exc
    try:
        import yaml
    except ImportError as exc:
        raise AnalyticsError(
            f"analytics error: PyYAML is required to read workspace config {path}"
        ) from exc
    try:
        data = yaml.safe_load(text)
    except Exception as exc:  # malformed YAML: a named failure, never a silent default
        raise AnalyticsError(
            f"analytics error: cannot parse workspace config {path}: {exc}"
        ) from exc
    if data is None:
        return default_model, default_timeout
    if not isinstance(data, dict):
        raise AnalyticsError(
            f"analytics error: workspace config {path} must be a mapping, "
            f"not {type(data).__name__}"
        )

    def _mapping(value: Any, where: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise AnalyticsError(
                f"analytics error: workspace config {path}: {where} must be a "
                f"mapping, not {type(value).__name__}"
            )
        return value

    plugins = _mapping(data.get("plugins"), "plugins")
    entries = _mapping(plugins.get("entries"), "plugins.entries")
    entry = _mapping(entries.get("hermes-skill-router"), "plugins.entries.hermes-skill-router")
    settings = _mapping(entry.get("settings"), "plugins.entries.hermes-skill-router.settings")
    model = default_model
    if need_model and "embedModelId" in settings:
        named = settings["embedModelId"]
        if not isinstance(named, str) or not named.strip():
            raise AnalyticsError(
                f"analytics error: workspace config {path}: embedModelId must name "
                f"a non-empty model, not {named!r}"
            )
        model = named
    timeout = default_timeout
    if need_timeout and "embedTimeoutMs" in settings:
        named_timeout = settings["embedTimeoutMs"]
        if (
            not isinstance(named_timeout, int)
            or isinstance(named_timeout, bool)
            or not timeout_min <= named_timeout <= timeout_max
        ):
            raise AnalyticsError(
                f"analytics error: workspace config {path}: embedTimeoutMs must be "
                f"between {timeout_min} and {timeout_max}, not {named_timeout!r}"
            )
        timeout = int(named_timeout)
    return model, timeout


def run(args: argparse.Namespace) -> List[str]:
    """Run the analysis and return the report lines to print."""
    if not 3 <= args.max_description_chars <= 4000:
        raise AnalyticsError(
            "analytics error: --max-description-chars must be between 3 and 4000, "
            f"not {args.max_description_chars}"
        )
    # Resolve the run's embed model/timeout from the workspace config when the
    # documented invocation omits them. The workspace config alone cannot
    # prove the run used those values, so the catalog copy's recorded build
    # model is the parity authority below: a mismatch is a named parity
    # error, never a silent replay in the wrong space. The explicit flags
    # always override the workspace config.
    model_explicit = args.embed_model is not None
    if model_explicit and not args.embed_model.strip():
        # Presence is not a value: a blank explicit model names no vector
        # space, so it is a named parity refusal like any mismatched model,
        # never a pass-through to the endpoint's own default.
        raise ParityError(
            "analytics parity error: --embed-model must name a non-empty model, "
            f"not {args.embed_model!r}; refusing to score in an unproven vector space"
        )
    if args.embed_model is None or args.embed_timeout_ms is None:
        # Only the omitted flags are resolved from the workspace config:
        # an explicit flag overrides its workspace field entirely, so
        # that field is not validated (C5: explicit flags always
        # override the workspace config).
        workspace_model, workspace_timeout = _resolve_embed_defaults(
            need_model=args.embed_model is None,
            need_timeout=args.embed_timeout_ms is None,
        )
        if args.embed_model is None:
            args.embed_model = workspace_model
        if args.embed_timeout_ms is None:
            args.embed_timeout_ms = workspace_timeout
    if (
        not isinstance(args.embed_timeout_ms, int)
        or isinstance(args.embed_timeout_ms, bool)
        or not 100 <= args.embed_timeout_ms <= 60000
    ):
        # The other explicit-flag path: presence alone never validates the
        # timeout either. The CLI's type=int already rejects a blank string,
        # but a programmatic caller could pass one; name it instead of
        # raising a TypeError from the range comparison.
        raise AnalyticsError(
            "analytics error: --embed-timeout-ms must be between 100 and 60000, "
            f"not {args.embed_timeout_ms}"
        )
    store_conn = open_readonly(args.store, "association store")
    try:
        check_provenance(store_conn)
        names, descriptions, catalog_model = read_catalog(args.catalog)
        if catalog_model is not None:
            if args.embed_model != catalog_model:
                raise ParityError(
                    f"analytics parity error: the replay model {args.embed_model!r} "
                    f"does not match the catalog's recorded build model "
                    f"{catalog_model!r}; refusing to score in the wrong vector space"
                )
        elif not model_explicit:
            raise ParityError(
                f"analytics parity error: catalog {args.catalog} records no build "
                "model and --embed-model was omitted, so the run's vector space "
                "cannot be proven; pass --embed-model explicitly"
            )
        # The timeout is not part of the parity check: it bounds one embed
        # request (liveness) and cannot select different vectors, so a wrong
        # timeout fails with the named embed error instead of silently
        # scoring in the wrong space.
        decisions = read_decision_log(args.log)
        if not decisions:
            return ["no anomalies"]
        if not args.assert_complete_source_history:
            raise _coverage(
                "missing --assert-complete-source-history: the run requires the "
                "operator assertion that no source session or message needed to "
                "reconstruct the analyzed history was deleted before the source "
                "was first inventoried or thereafter, including deletions before "
                "v2 enrollment"
            )
        state_conn = open_readonly(args.state_db, "Hermes session store")
        try:
            latest_ms = max(stamp for stamp, _query in decisions)
            check_finality(store_conn, state_conn, latest_ms)
        finally:
            state_conn.close()

        store = associations.AssociationStore(
            conn=store_conn, path=args.store, harness=associations.HARNESS, read_only=True
        )
        skill_ids: List[Optional[int]] = []
        for name in names:
            resolved = associations.resolve_skill_name(store, name)
            skill_ids.append(resolved.id if resolved is not None else None)
        queries = [query for _stamp, query in decisions]
        search = search_scores(
            names,
            descriptions,
            queries,
            args.max_description_chars,
            args.embed_url,
            args.embed_model,
            args.embed_timeout_ms / 1000.0,
        )
        divergences: Dict[str, List[Tuple[float, str]]] = {name: [] for name in names}
        for position, ((stamp, query), search_row) in enumerate(zip(decisions, search)):
            terms = associations.normalise_terms(query)
            history = association_scores(store_conn, skill_ids, terms, stamp)
            for skill_index, name in enumerate(names):
                divergence = history[skill_index] - search_row[skill_index]
                divergences[name].append((divergence, query))
        lines: List[str] = []
        for name in sorted(divergences):
            rows = divergences[name]
            if not rows or max(value for value, _query in rows) < DIVERGENCE_THRESHOLD:
                continue
            rows.sort(key=lambda item: (-item[0], json.dumps(item[1]), name))
            lines.append(f"ANOMALY {name}")
            for value, query in rows[:5]:
                lines.append(f"  {json.dumps(query)} {value:.6f}")
        if not lines:
            lines.append("no anomalies")
        return lines
    finally:
        store_conn.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        lines = run(args)
    except AnalyticsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"error: analytics store error: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
