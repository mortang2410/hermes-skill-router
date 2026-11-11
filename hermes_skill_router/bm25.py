"""The BM25 term channel and its persisted term index (module M15, contract C15).

This module is the only place that tokenises catalog text and scores BM25 ranks, and the only writer
of the ``bm25.json`` file that sits beside the vector index. It owns no settings: the scoring rules
are pinned here as module constants, so a change to one invalidates every persisted term index and
the loader refuses the old file rather than scoring it under a rule it was not built with.

The rules, fixed by the design:

* Tokens are lowercased, split on every character that is not a letter or a digit, dropped when the
  token is shorter than two characters, dropped when it is in one short English stopword list, and
  never stemmed. A one-character token in a non-Latin script is dropped as one character, not as two
  UTF-16 units.
* Scoring is Okapi BM25 with ``k1 = 1.2`` and ``b = 0.75``, with document frequency and the average
  document length computed over the same catalog.
* The document text is supplied by the caller. :mod:`hermes_skill_router.library` passes the exact
  text it embeds, so the two channels read one document format rather than two.
* The query is the composed query text the semantic channel embeds. This module does not build it.

The persisted index holds the document names and token counts, the postings and the average length,
plus two digests. ``catalog_digest`` covers the supplied document texts and the scoring constants, so
a term index built from other text is refused on load. ``payload_digest`` covers everything scoring
reads, so a truncated or hand-edited posting is refused too. A missing, unreadable, foreign or
corrupt file all load as ``None``: this channel contributes no ranks, never an exception, and the
caller fuses the remaining channels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Okapi BM25 constants. Pinned rather than configured: a change invalidates every term index.
BM25_K1 = 1.2

#: Length normalisation, pinned with :data:`BM25_K1`.
BM25_B = 0.75

#: The file name beside the vector index, in the same cache directory.
TERM_INDEX_FILENAME = "bm25.json"

#: The persisted file schema. A different version means a different scoring rule.
TERM_INDEX_VERSION = 1

#: Short English stopword list: function words that carry no skill vocabulary. Single-character
#: words are absent on purpose, because the two-character floor already removes them.
STOPWORDS = frozenset(
    """
    an and are as at be been but by can do does for from had has have if in into is it its not of on
    or our should so that the their them then there these they this those to was were what when which
    will with you your
    """.split()
)

#: One indexed document: the skill name and its token count.
Document = Tuple[str, str]


@dataclass(frozen=True)
class TermDocument:
    """One catalog row as scoring reads it: the name and its token count."""

    name: str
    length: int


@dataclass(frozen=True)
class TermIndex:
    """The persisted inverted index: what scoring reads, in catalog order."""

    version: int
    k1: float
    b: float
    documents: Tuple[TermDocument, ...]
    average_length: float
    postings: Dict[str, Tuple[int, ...]]
    catalog_digest: str
    payload_digest: str


def tokenise(text: Any) -> List[str]:
    """Return the terms of *text*: lowercased, split on non-alphanumerics, no stemming.

    A token shorter than two characters and a token in :data:`STOPWORDS` is dropped. A value that is
    not a string has no terms.
    """
    if not isinstance(text, str):
        return []
    tokens: List[str] = []
    current: List[str] = []
    for character in text.lower():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    kept: List[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in STOPWORDS:
            continue
        kept.append(token)
    return kept


def catalog_digest(rows: Sequence[Document]) -> str:
    """Return the sha256 over the document texts and the scoring constants.

    Each text is framed by its character length, so one description carrying a newline cannot
    produce the same byte stream as two rows whose texts would concatenate to it. The constants are
    part of the identity: a file built under other constants describes a different scoring rule.
    """
    digest = hashlib.sha256()
    digest.update(f"bm25:{TERM_INDEX_VERSION}:k1={BM25_K1}:b={BM25_B}\n".encode("utf-8"))
    for _name, text in rows:
        encoded = text.encode("utf-8")
        digest.update(f"{len(encoded)}:".encode("ascii"))
        digest.update(encoded)
    return digest.hexdigest()


def build_term_index(
    rows: Sequence[Document],
    cache_path: Optional[str] = None,
) -> TermIndex:
    """Tokenise *rows* and return the inverted index, writing it when *cache_path* is given.

    *rows* are ``(name, text)`` pairs in catalog order. A repeated term in one document is one
    posting with its frequency, and a term absent from every document has no posting. An unwritable
    *cache_path* raises ``OSError`` after the index is built; the caller keeps the returned value and
    reports the write failure, because the channel is usable in memory for this pass.
    """
    documents: List[TermDocument] = []
    postings: Dict[str, List[int]] = {}
    total_length = 0
    for position, (name, text) in enumerate(rows):
        tokens = tokenise(text)
        documents.append(TermDocument(name=name, length=len(tokens)))
        total_length += len(tokens)
        frequencies: Dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        for term, frequency in frequencies.items():
            postings.setdefault(term, []).extend([position, frequency])
    average_length = 0.0 if not documents else total_length / len(documents)
    frozen = {term: tuple(values) for term, values in postings.items()}
    payload = _payload(documents, average_length, frozen)
    index = TermIndex(
        version=TERM_INDEX_VERSION,
        k1=BM25_K1,
        b=BM25_B,
        documents=tuple(documents),
        average_length=average_length,
        postings=frozen,
        catalog_digest=catalog_digest(rows),
        payload_digest=_payload_digest(payload),
    )
    if cache_path is not None:
        write_term_index(cache_path, index)
    return index


def write_term_index(path: str, index: TermIndex) -> None:
    """Write *index* to *path* atomically, so a reader sees the old file or the new one."""
    directory = os.path.dirname(os.path.expanduser(path)) or "."
    os.makedirs(directory, exist_ok=True)
    temporary: Optional[str] = None
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=directory,
        prefix=".bm25-",
        suffix=".tmp",
        delete=False,
    )
    temporary = handle.name
    try:
        with handle:
            json.dump(_index_json(index), handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def load_term_index(path: str, rows: Sequence[Document]) -> Optional[TermIndex]:
    """Return the persisted index when it matches *rows*, otherwise ``None``.

    A missing file, unreadable JSON, an unexpected shape, a file built under other constants, a
    payload that no longer matches its digest, and a file built from other document texts all mean
    the same thing: there is no term index for this catalog.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    index = _parse(data)
    if index is None:
        return None
    if index.k1 != BM25_K1 or index.b != BM25_B:
        return None
    if index.catalog_digest != catalog_digest(rows):
        return None
    if index.payload_digest != _payload_digest(_payload_of(index)):
        return None
    return index


def score_ranks(index: Optional[TermIndex], text: Any, limit: int) -> List[str]:
    """Return the names ranked by descending BM25 score, ties in catalog order, at most *limit*.

    A missing index, an empty catalog, a non-positive limit and a query with no terms all return an
    empty list. A query term that no document carries contributes nothing. The function is pure:
    the same index and text return the same order twice.
    """
    if index is None or not index.documents or limit <= 0:
        return []
    terms = tokenise(text)
    if not terms or index.average_length <= 0:
        return []
    scores = [0.0] * len(index.documents)
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
            normalisation = BM25_K1 * (1 - BM25_B + BM25_B * (length / index.average_length))
            scores[document_index] += inverse * (
                (frequency * (BM25_K1 + 1)) / (frequency + normalisation)
            )
    ranked = [
        (-score, position, index.documents[position].name)
        for position, score in enumerate(scores)
        if score > 0
    ]
    ranked.sort()
    return [name for _score, _position, name in ranked[:limit]]


def _payload(
    documents: Sequence[TermDocument],
    average_length: float,
    postings: Dict[str, Tuple[int, ...]],
) -> Dict[str, Any]:
    """Return the scoring payload in the one shape the digest covers."""
    return {
        "documents": [{"name": document.name, "length": document.length} for document in documents],
        "average_length": average_length,
        "postings": {term: list(values) for term, values in postings.items()},
    }


def _payload_of(index: TermIndex) -> Dict[str, Any]:
    return _payload(index.documents, index.average_length, index.postings)


def _index_json(index: TermIndex) -> Dict[str, Any]:
    """Return the whole persisted file shape, not only the scored payload."""
    return {
        "version": index.version,
        "k1": index.k1,
        "b": index.b,
        "documents": [
            {"name": document.name, "length": document.length} for document in index.documents
        ],
        "average_length": index.average_length,
        "postings": {term: list(values) for term, values in index.postings.items()},
        "catalog_digest": index.catalog_digest,
        "payload_digest": index.payload_digest,
    }


def _payload_digest(payload: Dict[str, Any]) -> str:
    """Return the sha256 over the scoring payload, in its canonical JSON form."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse(data: Any) -> Optional[TermIndex]:
    """Return a structurally valid index from parsed JSON, or ``None``."""
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    k1 = data.get("k1")
    b = data.get("b")
    average_length = data.get("average_length")
    documents = data.get("documents")
    postings = data.get("postings")
    stored_catalog = data.get("catalog_digest")
    stored_payload = data.get("payload_digest")
    if version != TERM_INDEX_VERSION:
        return None
    if not isinstance(k1, (int, float)) or not isinstance(b, (int, float)):
        return None
    if not isinstance(average_length, (int, float)) or isinstance(average_length, bool):
        return None
    if not math.isfinite(float(average_length)) or float(average_length) < 0:
        return None
    if not isinstance(stored_catalog, str) or not isinstance(stored_payload, str):
        return None
    if not isinstance(documents, list) or not isinstance(postings, dict):
        return None
    parsed: List[TermDocument] = []
    for row in documents:
        if not isinstance(row, dict):
            return None
        name = row.get("name")
        length = row.get("length")
        if not isinstance(name, str):
            return None
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            return None
        parsed.append(TermDocument(name=name, length=length))
    frozen: Dict[str, Tuple[int, ...]] = {}
    for term, values in postings.items():
        if not isinstance(term, str) or not isinstance(values, list):
            return None
        if len(values) % 2:
            return None
        seen: set[int] = set()
        parsed_values: List[int] = []
        for position in range(0, len(values), 2):
            document_index = values[position]
            frequency = values[position + 1]
            if isinstance(document_index, bool) or not isinstance(document_index, int):
                return None
            if document_index < 0 or document_index >= len(parsed):
                return None
            if document_index in seen:
                return None
            seen.add(document_index)
            if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
                return None
            parsed_values.extend([document_index, frequency])
        frozen[term] = tuple(parsed_values)
    return TermIndex(
        version=TERM_INDEX_VERSION,
        k1=float(k1),
        b=float(b),
        documents=tuple(parsed),
        average_length=float(average_length),
        postings=frozen,
        catalog_digest=stored_catalog,
        payload_digest=stored_payload,
    )
