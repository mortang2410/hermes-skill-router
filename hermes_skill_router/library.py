"""The skill index and its persisted cache (module M4, contract C4).

The index is the ``routed`` half of the catalog ``catalog.strip_catalog`` parsed out of the live
system message: the rows the plugin removes and may suggest. Nothing here walks the filesystem.
The catalog is the set of skills Hermes will actually let the model open, including plugin skills
and gated rows, under the names ``skill_view`` accepts, so indexing it keeps the suggestion block
and the loader from ever disagreeing. A row that was kept stays listed and never enters the index,
so an eager skill, a row outside a non-empty deferred set, and every name on a ``[names only]``
line can never be suggested.

What ties a persisted index to the inputs that produced it is :func:`fingerprint`: a sha256 over
the clipped document text and the full description of every indexed row in catalog order, plus the
embedding model id, the document format version and the full endpoint URL. The full description is
part of the identity even where the embedding text clips it, because the cached descriptions are
also the rerank document text and the block text: an edit past the clip must invalidate the cache
even though it cannot change one vector. A change to a kept row cannot change it, because no kept
row is part of the hash. The cache is trusted only when its ``version``, ``fingerprint`` and ``dim``
match; vectors from another model or another endpoint would rank the wrong skills.

A saved index that disagrees with the arriving catalog is a normal edit and is rebuilt, but a
catalog holding fewer than half the rows of the saved index is not treated as an edit: a truncated
catalog read would replace a full index with a handful of rows, so the build refuses that catalog,
keeps the file, and routes from the saved index. The refusal logs one warning per refused catalog
naming both counts and the cache path, and removing the file forces the rebuild (R6).

A warm start must not touch the network, so a built index is memoised per fingerprint behind a
lock: two concurrent requests on a cold cache embed the library once, and the second one waits for
the first rather than repeating its work. A cold build sends the rows in batches of
:data:`BATCH_TEXTS`, so ``embedTimeoutMs`` bounds one request instead of the whole library.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .catalog import CatalogRows
from .config import Config

logger = logging.getLogger(__name__)

#: Cache file schema. Bumped when the file's shape changes, which invalidates every old cache.
CACHE_VERSION = 1

#: Document format version. Bumped when :func:`document_text` changes, so old vectors are rebuilt.
DOCUMENT_FORMAT_VERSION = 1

#: Texts per embeddings request. A cold build of the live 424-row catalog took 19 to 42 s as one
#: request, so the build is chunked and the timeout bounds one request instead of the whole library.
#: Sixteen-row chunks measured 1.2 to 2.0 s against 8012 and one overran 2000, so the default
#: ``embedTimeoutMs`` is 6000.
BATCH_TEXTS = 16

#: ``M5.embed_texts``: one vector per text, in order, or ``None`` when the endpoint failed.
Embedder = Callable[[Config, Sequence[str]], Optional[Sequence[Sequence[float]]]]


@dataclass(frozen=True)
class Index:
    """The embedded routed catalog: parallel names, descriptions and vectors plus their identity."""

    names: List[str]
    descriptions: List[str]
    vectors: List[List[float]]
    dim: int
    fingerprint: str


def document_text(cfg: Config, name: str, description: str) -> str:
    """Return the exact text embedded for one row: ``name``, a newline, then the description.

    The description is clipped to ``cfg.max_description_chars``, which is what the reference router's
    ``documentText`` does, so both clients embed byte-identical text for the same row and
    configuration. It is one function so the index, the fingerprint and the other client can never
    drift apart.
    """
    return name + "\n" + description[: cfg.max_description_chars]


def fingerprint(rows: CatalogRows, cfg: Config) -> str:
    """Return the sha256 that ties a cache to the rows and endpoint that produced it.

    Every row's clipped document text and its full description, in catalog order, plus the embedding
    model id, the document format version and the full endpoint URL are hashed. The full description
    is included because the cache also supplies the rerank document text and the block text, so an
    edit past the clip must invalidate it. A duplicate name contributes once, as it does to the index.
    """
    return _fingerprint(_unique(rows), cfg)


def load_or_build(cfg: Config, rows: CatalogRows, embedder: Embedder) -> Optional[Index]:
    """Return the index for *rows*, from the memo, the cache, or one chunked embedding pass.

    Returns ``None`` when there is nothing to index or any embeddings chunk failed, in which case
    the request stays as it arrived. A cache write failure is logged once and the in-memory index
    is still returned. A catalog holding fewer than half the rows of the saved index is refused
    before anything is embedded or written: the saved index is returned unchanged and one warning
    names both counts and the cache path (R6).
    """
    unique = _unique(rows)
    key = _fingerprint(unique, cfg)
    with _MEMO_LOCK:
        saved = _read_saved(cfg)
        if saved is not None and len(unique) * 2 < len(saved.names):
            _warn_refusal(cfg, len(unique), len(saved.names))
            return saved
        if not unique:
            return None
        memo = _MEMO.get(key)
        if memo is not None:
            return memo
        if saved is not None and saved.fingerprint == key and len(saved.names) == len(unique):
            _MEMO[key] = saved
            return saved
        index = _build(unique, _embed_rows(cfg, unique, embedder), key)
        if index is None:
            return None
        _write_cache(cfg, index)
        _MEMO[key] = index
        return index


def cached_row_count(cfg: Config) -> int:
    """Return the row count of the saved index, or ``0`` when there is no usable saved index."""
    with _MEMO_LOCK:
        saved = _read_saved(cfg)
    return len(saved.names) if saved is not None else 0


def shortlist(index: Index, query_vector: Sequence[float], limit: int) -> List[Tuple[str, str]]:
    """Return the *limit* indexed rows closest to *query_vector* by cosine, ties in index order.

    One stage of the routing pass, shared by the request middleware and the search tool so the same
    query and index always shortlist the same rows in the same order. An empty index, a zero limit
    or a query vector with no length returns ``[]``.
    """
    scored: List[Tuple[float, int, str, str]] = []
    for position, (name, description, vector) in enumerate(
        zip(index.names, index.descriptions, index.vectors)
    ):
        scored.append((-_cosine(query_vector, vector), position, name, description))
    scored.sort()
    return [
        (name, description)
        for _score, _position, name, description in scored[: max(0, limit)]
    ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or ``0.0`` when either has no length."""
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


def _unique(rows: CatalogRows) -> List[Tuple[str, str]]:
    """Collapse duplicate names to their first occurrence, keeping catalog order."""
    unique: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for name, description in zip(rows.names, rows.descriptions):
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, description))
    return unique


def _fingerprint(unique: Sequence[Tuple[str, str]], cfg: Config) -> str:
    digest = hashlib.sha256()
    digest.update(f"hermes-skill-router/index:{DOCUMENT_FORMAT_VERSION}\n".encode())
    digest.update(f"model:{cfg.embed_model_id}\n".encode())
    digest.update(f"endpoint:{cfg.embed_base_url}{cfg.embed_path}\n".encode())
    for name, description in unique:
        digest.update(b"text:")
        digest.update(document_text(cfg, name, description).encode("utf-8"))
        digest.update(b"\n")
        # The full description, not only the clipped tail the embedder sees: the cache also carries
        # it into the rerank documents and the block, so an edit past the clip must invalidate.
        digest.update(b"description:")
        digest.update(description.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _embed_rows(
    cfg: Config, unique: Sequence[Tuple[str, str]], embedder: Embedder
) -> Optional[List[List[float]]]:
    """Embed every row in batches of at most :data:`BATCH_TEXTS` and return vectors in row order.

    Chunking is what makes ``embedTimeoutMs`` bound one request rather than the whole library: every
    chunk must fit the timeout on its own, while the whole 27-chunk build of the live catalog is
    expected to take tens of seconds (about 38) and still succeed. Every chunk must answer with one
    vector per text it received; a chunk that fails, or answers short or with mixed widths, fails the
    whole build, because a partial index must never be returned or cached.
    """
    texts = [document_text(cfg, name, description) for name, description in unique]
    vectors: List[List[float]] = []
    for start in range(0, len(texts), BATCH_TEXTS):
        chunk = texts[start : start + BATCH_TEXTS]
        answer = embedder(cfg, chunk)
        if answer is None:
            return None
        if len(answer) != len(chunk):
            _log_once(
                "vector-count",
                f"skill index: the embedder returned {len(answer)} vectors for {len(chunk)} texts",
            )
            return None
        widths = {len(vector) for vector in answer}
        if len(widths) != 1 or 0 in widths:
            _log_once("vector-width", "skill index: the embedder returned vectors of mixed widths")
            return None
        width = widths.pop()
        if vectors and len(vectors[0]) != width:
            _log_once("vector-width", "skill index: the embedder returned vectors of mixed widths")
            return None
        vectors.extend([float(value) for value in vector] for vector in answer)
    return vectors


def _build(
    unique: Sequence[Tuple[str, str]],
    vectors: Optional[Sequence[Sequence[float]]],
    key: str,
) -> Optional[Index]:
    """Turn the embedded rows into an ``Index``, or ``None`` when the embedding failed."""
    if not vectors:
        return None
    return Index(
        names=[name for name, _description in unique],
        descriptions=[description for _name, description in unique],
        vectors=[[float(value) for value in vector] for vector in vectors],
        dim=len(vectors[0]),
        fingerprint=key,
    )


def _cache_path(cfg: Config) -> Optional[str]:
    """Return the cache path to use, or ``None`` when persistence is disabled."""
    raw = (cfg.cache_path or "").strip()
    if not raw:
        return None
    return os.path.expanduser(raw)


def _read_saved(cfg: Config) -> Optional[Index]:
    """Return the saved index when the cache file is usable, otherwise ``None``.

    A missing, unreadable, wrong-version or wrong-shaped file is not a saved list, so the caller
    proceeds with the build exactly as it did before the guard existed. A usable file is returned
    under its own stored fingerprint, because a refused catalog keeps routing on the saved
    identity rather than on the one the arrived rows would produce.
    """
    path = _cache_path(cfg)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        _log_once(
            f"cache-read:{path}",
            f"skill index: the cache at {path} is unreadable, rebuilding: {exc}",
        )
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != CACHE_VERSION:
        return None
    saved_fingerprint = data.get("fingerprint")
    if not isinstance(saved_fingerprint, str) or not saved_fingerprint:
        return None
    dim = data.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        return None
    names = data.get("names")
    descriptions = data.get("descriptions")
    vectors = data.get("vectors")
    if not isinstance(names, list) or not isinstance(descriptions, list):
        return None
    if not isinstance(vectors, list):
        return None
    if not len(names) == len(descriptions) == len(vectors):
        return None
    if not all(isinstance(name, str) for name in names):
        return None
    if not all(isinstance(description, str) for description in descriptions):
        return None
    parsed: List[List[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dim:
            return None
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in vector
        ):
            return None
        parsed.append([float(value) for value in vector])
    return Index(
        names=list(names),
        descriptions=list(descriptions),
        vectors=parsed,
        dim=dim,
        fingerprint=saved_fingerprint,
    )


def _warn_refusal(cfg: Config, arrived: int, saved: int) -> None:
    """Log one warning per refused catalog, naming both counts and the file to remove.

    Deliberately not deduplicated: each refused catalog is reported, so a catalog that is still
    truncated on the next turn is still visible to an operator.
    """
    path = _cache_path(cfg)
    logger.warning(
        "skill index: the catalog read returned %d skills, less than half of the %d-entry "
        "saved index, so it is a partial view; the saved index is reused rather than replaced. "
        "Remove %s to force a rebuild if the catalog really did shrink this far",
        arrived,
        saved,
        path,
    )


def _write_cache(cfg: Config, index: Index) -> None:
    """Write the cache atomically, or log once and keep the in-memory index."""
    path = _cache_path(cfg)
    if path is None:
        return
    directory = os.path.dirname(path) or "."
    temporary: Optional[str] = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".index-",
            suffix=".tmp",
            delete=False,
        )
        temporary = handle.name
        with handle:
            json.dump(_cache_payload(index), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        _log_once(f"cache-write:{path}", f"skill index: could not write {path}: {exc}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError as exc:
                # The destination failure is already reported; name the leftover too, so the one
                # file a refusing filesystem keeps is never silent.
                _log_once(
                    f"leftover:{temporary}",
                    f"skill index: cannot remove the temporary file {temporary}: {exc}",
                )


def _cache_payload(index: Index) -> Dict[str, object]:
    return {
        "version": CACHE_VERSION,
        "fingerprint": index.fingerprint,
        "dim": index.dim,
        "names": list(index.names),
        "descriptions": list(index.descriptions),
        "vectors": [list(vector) for vector in index.vectors],
    }


#: The in-process memo, keyed by fingerprint: one build per fingerprint per process.
_MEMO: Dict[str, Index] = {}
_MEMO_LOCK = threading.Lock()

#: Reasons already logged, so one broken path logs once instead of once per request (D8).
_LOGGED: set[str] = set()


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.warning(message)
