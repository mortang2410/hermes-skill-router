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
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import associations, bm25
from .catalog import CatalogRows
from .config import Config
from .fusion import (
    ASSOCIATION_SUPPORT_FLOOR,
    CandidateDiagnostic,
    FusionConfig,
    apply_frecency,
    cut_shortlist,
    fuse_rankings,
    resolve_fusion_config,
)

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

#: The host's own skill lookup: a ``SKILL.md`` path, or ``None`` when it refuses the name.
PathResolver = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class Index:
    """The embedded routed catalog: parallel names, descriptions and vectors plus their identity.

    ``term_index`` is the BM25 inverted index built from the same document text, or ``None`` when
    the term index could not be built or did not match. It is optional at the end so a test or a
    caller can hand over an index of vectors alone.
    """

    names: List[str]
    descriptions: List[str]
    vectors: List[List[float]]
    dim: int
    fingerprint: str
    term_index: Optional[bm25.TermIndex] = None


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

    The BM25 term index is built in the same pass and written beside the vector index. A term index
    that cannot be built does not fail the build: the returned index carries ``None``, one warning
    is logged, and the vector pass stays usable. A warm vector cache loads the term index once per
    process, and a missing file is built from the catalog rows; a foreign or unreadable one leaves
    the channel with no ranks and warns once.
    """
    unique = _unique(rows)
    key = _fingerprint(unique, cfg)
    with _MEMO_LOCK:
        saved = _read_saved(cfg)
        if saved is not None and len(unique) * 2 < len(saved.names):
            _warn_refusal(cfg, len(unique), len(saved.names))
            return _with_term_index(cfg, saved, _saved_texts(cfg, saved))
        if not unique:
            return None
        memo = _MEMO.get(key)
        if memo is not None:
            return memo
        if saved is not None and saved.fingerprint == key and len(saved.names) == len(unique):
            index = _warm_term_index(cfg, saved, unique)
            _MEMO[key] = index
            return index
        index = _build(
            unique,
            _embed_rows(cfg, unique, embedder),
            key,
            _build_term_index(cfg, unique),
        )
        if index is None:
            return None
        if _write_cache(cfg, index):
            _write_term_index(cfg, index.term_index)
        _MEMO[key] = index
        return index


def cached_row_count(cfg: Config) -> int:
    """Return the row count of the saved index, or ``0`` when there is no usable saved index."""
    with _MEMO_LOCK:
        saved = _read_saved(cfg)
    return len(saved.names) if saved is not None else 0


def path_resolver(
    cfg: Config,
    index: Index,
    resolve: Optional[PathResolver],
) -> Optional[PathResolver]:
    """Return a resolver that answers from the saved read-path map and remembers every new path.

    The host lookup behind *resolve* scans and parses every ``SKILL.md`` under the search roots to
    resolve one name, which measured about 230 ms per call on the live 420-row catalog. One pass
    must therefore never ask it twice for the same name, and a later pass or a later process must
    not ask again at all. Each index fingerprint owns one map, saved beside the index cache: a new
    catalog starts with an empty map, and an entry loaded from disk is reused only while the file it
    names still exists, so a skill that moved is resolved again instead of pointing at a missing
    file. ``None`` stays ``None``, so a caller that resolves nothing keeps that behavior.
    """
    if resolve is None:
        return None
    saved = _saved_paths(cfg, index)

    def _resolve(name: str) -> Optional[str]:
        path = saved.get(name)
        if path is not None:
            return path
        resolved = resolve(name)
        if not resolved:
            return None
        with _MEMO_LOCK:
            saved[name] = str(resolved)
        _write_paths(cfg, index, saved)
        return saved[name]

    return _resolve


def shortlist(
    index: Index,
    query_vector: Sequence[float],
    query_text: str,
    limit: int,
    *,
    cfg: Config,
    resolver: Optional[PathResolver] = None,
    now_ms: Optional[int] = None,
    on_diagnostic: Optional[Callable[[CandidateDiagnostic], None]] = None,
) -> List[Tuple[str, str]]:
    """Return the fused, bumped and cut shortlist for *query_vector* and *query_text*.

    The semantic channel is the cosine order over the index, the BM25 channel reads the term index
    the build carried, and the association channel reads the learned store open read-only. The
    channels are fused with reciprocal rank fusion, the recency bump moves a candidate whose
    catalog path *resolver* resolves, and only then is the order cut to *limit*. A disabled channel
    and an absent one are the same input, so a missing term index or store leaves the remaining
    channels routing.

    ``mode: cosine`` returns the cosine order cut to *limit* without reading the term index, opening
    the store or applying the bump. ``mode: shadow`` computes and logs the diagnostics but returns
    the cosine order, so the two published orders can be compared from the log. A zero limit or an
    empty index returns ``[]``.
    """
    if limit <= 0 or not index.names:
        return []
    cosine_order = _cosine_order(index, query_vector)
    settings = _fusion_settings(cfg)
    if settings.mode == "cosine" or not settings.enabled:
        return _described(index, cosine_order[:limit])

    bm25_order: Optional[List[str]] = None
    if settings.bm25:
        if index.term_index is None:
            _warn_term_index_missing(cfg)
        else:
            bm25_order = bm25.score_ranks(index.term_index, query_text, len(index.names))

    store = _open_store(cfg)
    association_order: Optional[List[str]] = None
    if settings.associations and store is not None:
        association_order = _association_ranks(store, index, query_text, now_ms)
    try:
        fused = fuse_rankings(
            {
                "cosine": cosine_order,
                "semantic": cosine_order if settings.semantic else None,
                "bm25": bm25_order,
                "associations": association_order,
            },
            settings,
        )
        # A zero cap moves nothing, so neither the usage read nor its path lookups can change the
        # order and both are skipped outright.
        usage = _usage_scores(store, settings, now_ms) if settings.cap_positions > 0 else {}
        moved = apply_frecency(
            fused, _gated_usage(usage, fused, resolver, limit, settings.cap_positions), settings
        )
        if settings.mode == "shadow":
            # The log carries the order live mode would return, movement and final positions
            # included, so the two published orders compare directly.
            _log_fusion_diagnostics(cut_shortlist(moved, limit))
            return _described(index, cosine_order[:limit])
    finally:
        _close_store(store)
    cut = cut_shortlist(moved, limit)
    if on_diagnostic is not None:
        for record in cut:
            on_diagnostic(record)
    return _described(index, [record.name for record in cut])


def _cosine_order(index: Index, query_vector: Sequence[float]) -> List[str]:
    """Return every indexed name by descending cosine, ties in index order."""
    scored = [
        (-_cosine(query_vector, vector), position, name)
        for position, (name, vector) in enumerate(zip(index.names, index.vectors))
    ]
    scored.sort()
    return [name for _score, _position, name in scored]


def _described(index: Index, names: Sequence[str]) -> List[Tuple[str, str]]:
    """Map names back to the catalog descriptions, in the order given."""
    descriptions = dict(zip(index.names, index.descriptions))
    return [(name, descriptions.get(name, "")) for name in names]


def _fusion_settings(cfg: Config) -> FusionConfig:
    """Resolve the fusion settings from the one place that validated them."""
    return resolve_fusion_config(
        {
            "mode": cfg.fusion_mode,
            "channels": {
                "semantic": cfg.channels_semantic,
                "bm25": cfg.channels_bm25,
                "associations": cfg.channels_associations,
            },
            "frecency": {
                "halfLifeDays": cfg.frecency_half_life_days,
                "capPositions": cfg.frecency_cap_positions,
            },
        }
    )


def _text_rows(cfg: Config, unique: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Return the ``(name, text)`` pairs the term index indexes: the embedder's own document text."""
    return [(name, document_text(cfg, name, description)) for name, description in unique]


def _saved_texts(cfg: Config, saved: Index) -> List[Tuple[str, str]]:
    """Return the saved index's rows in the same document text the term index was built with."""
    return [
        (name, document_text(cfg, name, description))
        for name, description in zip(saved.names, saved.descriptions)
    ]


def _with_term_index(
    cfg: Config,
    index: Index,
    texts: Sequence[Tuple[str, str]],
) -> Index:
    """Return *index* carrying the loaded term index, or ``None`` when none is usable."""
    return replace(index, term_index=_load_term_index(cfg, texts))


def _load_term_index(cfg: Config, texts: Sequence[Tuple[str, str]]) -> Optional[bm25.TermIndex]:
    """Load the term index beside the cache once per process, or report that there is none."""
    path = _term_path(cfg)
    if path is None:
        return None
    digest = bm25.catalog_digest(texts)
    cached = _TERM_MEMO.get(path)
    if cached is not None and cached[0] == digest:
        return cached[1]
    try:
        loaded = bm25.load_term_index(path, texts)
    except Exception:  # the channel must never fail the pass
        logger.debug("skill index: the term index load raised", exc_info=True)
        loaded = None
    if loaded is None:
        _warn_term_index_missing(cfg)
    _TERM_MEMO[path] = (digest, loaded)
    return loaded


def _warm_term_index(
    cfg: Config,
    saved: Index,
    unique: Sequence[Tuple[str, str]],
) -> Index:
    """Return *saved* carrying a usable term index, building one when the file is missing.

    A warm vector cache with no term index beside it is the state every install whose index predates
    the term index is in. The vectors are still valid, so the term index is built from the catalog
    rows and written, independently of the vector cache; otherwise the default-on BM25 channel would
    stay off until the catalog fingerprint changed. A term index that exists but is foreign,
    truncated or unreadable is left to the loader's own refusal, which warns once.
    """
    path = _term_path(cfg)
    if path is not None and not os.path.exists(path):
        term_index = _build_term_index(cfg, unique)
        if term_index is None:
            return saved
        _write_term_index(cfg, term_index)
        return replace(saved, term_index=term_index)
    return _with_term_index(cfg, saved, _text_rows(cfg, unique))


def _build_term_index(
    cfg: Config,
    unique: Sequence[Tuple[str, str]],
) -> Optional[bm25.TermIndex]:
    """Build the term index over the embedder's document text, or warn once and disable BM25."""
    try:
        return bm25.build_term_index(_text_rows(cfg, unique))
    except Exception as exc:  # a broken term index must not fail the vector pass
        logger.debug("skill index: the term index build raised", exc_info=True)
        _log_once(
            f"term-index:{_term_location(cfg)}",
            f"skill index: the term index at {_term_location(cfg)} could not be built ({exc}), so "
            "the BM25 channel contributes no ranks",
        )
        return None


def _write_term_index(cfg: Config, term_index: Optional[bm25.TermIndex]) -> None:
    """Write the term index beside the vector index, or log the write failure once."""
    path = _term_path(cfg)
    if path is None or term_index is None:
        return
    try:
        bm25.write_term_index(path, term_index)
    except OSError as exc:
        _log_once(
            f"term-index-write:{path}",
            f"skill index: could not write the term index {path}: {exc}",
        )
        return
    _TERM_MEMO[path] = (term_index.catalog_digest, term_index)


def _term_path(cfg: Config) -> Optional[str]:
    """Return the term index path beside the vector cache, or ``None`` when persistence is off."""
    raw = (cfg.cache_path or "").strip()
    if not raw:
        return None
    return os.path.join(
        os.path.dirname(os.path.expanduser(raw)),
        bm25.TERM_INDEX_FILENAME,
    )


def _term_location(cfg: Config) -> str:
    """Return the term index location for a warning, naming the in-memory case."""
    return _term_path(cfg) or "the in-memory index"


def _warn_term_index_missing(cfg: Config) -> None:
    """Log one warning per process when the BM25 channel has no usable term index."""
    location = _term_location(cfg)
    _log_once(
        f"term-index:{location}",
        f"skill index: no term index matches the catalog at {location}, so the BM25 channel "
        "contributes no ranks",
    )


def _association_path(cfg: Config) -> Optional[str]:
    """Return the association store path, or ``None`` when persistence is off."""
    raw = (cfg.association_db_path or "").strip()
    if raw:
        return os.path.expanduser(raw)
    cache = (cfg.cache_path or "").strip()
    if not cache:
        return None
    return associations.association_db_path_for(cache)


def _open_store(cfg: Config) -> Optional[associations.AssociationStore]:
    """Open the association store read-only, or report a missing or unreadable file once."""
    path = _association_path(cfg)
    if path is None:
        return None
    if not os.path.exists(path):
        _log_once(
            f"associations-missing:{path}",
            f"skill index: the association database at {path} is missing, so the association "
            "channel contributes no ranks",
        )
        return None
    try:
        return associations.open_store(path, read_only=True)
    except Exception as exc:  # a foreign or corrupt store is one channel losing its ranks
        logger.debug("skill index: the association store open raised", exc_info=True)
        _log_once(
            f"associations-unreadable:{path}",
            f"skill index: the association database at {path} is unreadable ({exc}), so the "
            "association channel contributes no ranks",
        )
        return None


def _close_store(store: Optional[associations.AssociationStore]) -> None:
    """Close a read-only store, ignoring a failure that no caller can act on."""
    if store is not None:
        store.close()


def _association_ranks(
    store: associations.AssociationStore,
    index: Index,
    query_text: str,
    now_ms: Optional[int],
) -> List[str]:
    """Return the association channel's ranked catalog names, or no rank on a read failure."""
    try:
        terms = associations.normalise_terms(query_text)
        weights = associations.decayed_weights(
            store,
            terms,
            now_ms,
            support_floor=ASSOCIATION_SUPPORT_FLOOR,
        )
    except Exception as exc:  # one channel losing its ranks must not fail the pass
        logger.debug("skill index: the association read raised", exc_info=True)
        _log_once(
            f"associations-read:{store.path}",
            f"skill index: the association database at {store.path} could not be read ({exc}), so "
            "the association channel contributes no ranks",
        )
        return []
    catalog = set(index.names)
    return [row.name for row in weights if row.name in catalog]


def _usage_scores(
    store: Optional[associations.AssociationStore],
    settings: FusionConfig,
    now_ms: Optional[int],
) -> Dict[str, float]:
    """Return the decayed usage map, or an empty map when the store contributes nothing."""
    if store is None:
        return {}
    try:
        return associations.usage_scores(store, settings.half_life_days, now_ms)
    except Exception as exc:  # frecency losing its input must not fail the pass
        logger.debug("skill index: the usage read raised", exc_info=True)
        _log_once(
            f"associations-read:{store.path}",
            f"skill index: the association database at {store.path} could not be read ({exc}), so "
            "frecency contributes no boost",
        )
        return {}


def _gated_usage(
    usage: Dict[str, float],
    records: Sequence[CandidateDiagnostic],
    resolver: Optional[PathResolver],
    limit: int,
    cap_positions: int,
) -> Dict[str, float]:
    """Keep usage rows for the candidates the bump can still carry into the shortlist.

    The bump moves a candidate up by at most *cap_positions* places, so a candidate below
    ``limit + cap_positions`` in the fused order can never reach the cut and keeps its fused
    position whatever its usage is. Only the candidates at or above that bound are looked up, and
    *records* arrive in fused order, so the scan stops there: the number of host lookups is bounded
    by the shortlist size plus the cap, never by the usage history the store carries. A name no
    channel returned is dropped as well. With no resolver every candidate is unresolved, which is
    the same rule as a path the lookup refuses: no path, no bump.
    """
    if not usage or resolver is None:
        return {}
    reachable = limit + cap_positions
    gated: Dict[str, float] = {}
    for record in records:
        if record.final_shortlist_position > reachable:
            break
        score = usage.get(record.name)
        if score is None:
            continue
        try:
            path = resolver(record.name)
        except Exception:  # the host lookup never raises, but the pass must not depend on it
            logger.debug("skill index: the path lookup for %r raised", record.name, exc_info=True)
            path = None
        if path:
            gated[record.name] = float(score)
    return gated


def _log_fusion_diagnostics(records: Sequence[CandidateDiagnostic]) -> None:
    """Log one shadow-mode line carrying the channel ranks, the movement and the position."""
    payload = [
        {
            "name": record.name,
            "semantic": record.semantic_rank,
            "bm25": record.bm25_rank,
            "associations": record.association_rank,
            "movement": record.frecency_movement,
            "position": record.final_shortlist_position,
        }
        for record in records
    ]
    logger.info(
        "skill router fusion: mode=shadow candidates=%s",
        json.dumps(payload, separators=(",", ":")),
    )


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
    term_index: Optional[bm25.TermIndex],
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
        term_index=term_index,
    )


def _cache_path(cfg: Config) -> Optional[str]:
    """Return the cache path to use, or ``None`` when persistence is disabled."""
    raw = (cfg.cache_path or "").strip()
    if not raw:
        return None
    return os.path.expanduser(raw)


def _paths_path(cfg: Config) -> Optional[str]:
    """Return the saved read-path map's file beside the vector cache, or ``None`` offline."""
    path = _cache_path(cfg)
    if path is None:
        return None
    return os.path.join(os.path.dirname(path), PATH_CACHE_FILENAME)


def _saved_paths(cfg: Config, index: Index) -> Dict[str, str]:
    """Return this process' read-path map for *index*, reading it once per cache file.

    The map belongs to one catalog in one cache directory, so it is memoised under both: two
    profiles, and two tests that share a fingerprint but not a cache path, never share a map. An
    entry whose file no longer exists is dropped, because a skill that moved changed no catalog text
    and so cannot invalidate the map by fingerprint alone. With persistence off there is no map to
    keep, and every pass starts from an empty one.
    """
    path = _paths_path(cfg)
    if path is None:
        return {}
    key = (path, index.fingerprint)
    with _MEMO_LOCK:
        saved = _PATH_MEMO.get(key)
        if saved is not None:
            return saved
        data: Any = None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            data = None
        except (OSError, ValueError) as exc:
            _log_once(
                f"paths-read:{path}",
                f"skill index: the saved read paths at {path} are unreadable ({exc}), so "
                "every path is resolved again",
            )
        saved = {}
        if (
            isinstance(data, dict)
            and data.get("version") == PATH_CACHE_VERSION
            and data.get("fingerprint") == index.fingerprint
            and isinstance(data.get("paths"), dict)
        ):
            saved = {
                name: value
                for name, value in data["paths"].items()
                if isinstance(name, str) and isinstance(value, str) and os.path.isfile(value)
            }
        _PATH_MEMO[key] = saved
        return saved


def _write_paths(cfg: Config, index: Index, paths: Dict[str, str]) -> None:
    """Save the read-path map beside the index cache, or log the failure once.

    The file is a cache: a lost update costs one later host lookup, never a wrong path, so the
    replace is atomic and not fsynced.
    """
    path = _paths_path(cfg)
    if path is None:
        return
    with _MEMO_LOCK:  # the map is shared between passes; snapshot it before iterating
        snapshot = dict(paths)
    payload = {
        "version": PATH_CACHE_VERSION,
        "fingerprint": index.fingerprint,
        "paths": snapshot,
    }
    directory = os.path.dirname(path) or "."
    temporary: Optional[str] = None
    try:
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=".paths-",
            suffix=".tmp",
            delete=False,
        )
        temporary = handle.name
        with handle:
            json.dump(payload, handle)
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        _log_once(f"paths-write:{path}", f"skill index: could not write the read paths {path}: {exc}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError as exc:
                _log_once(
                    f"paths-leftover:{temporary}",
                    f"skill index: cannot remove the temporary file {temporary}: {exc}",
                )


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


def _write_cache(cfg: Config, index: Index) -> bool:
    """Write the cache atomically, or log once and keep the in-memory index.

    Returns whether the file was replaced. The term index is written after a successful vector
    write, so a failed build never leaves a term index without the vector cache it belongs to.
    """
    path = _cache_path(cfg)
    if path is None:
        return False
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
        return True
    except OSError as exc:
        _log_once(f"cache-write:{path}", f"skill index: could not write {path}: {exc}")
        return False
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

#: The loaded term index per path and the digest it matched, so a warm process reads it once.
_TERM_MEMO: Dict[str, Tuple[str, Optional[bm25.TermIndex]]] = {}

#: The read-path map per cache file and fingerprint: one host lookup per name per catalog.
_PATH_MEMO: Dict[Tuple[str, str], Dict[str, str]] = {}

#: Saved read-path map schema. Bumped when the file's shape changes, which invalidates every map.
PATH_CACHE_VERSION = 1

#: The saved read-path map's file name, beside the vector index cache.
PATH_CACHE_FILENAME = "paths.json"

_MEMO_LOCK = threading.Lock()

#: Reasons already logged, so one broken path logs once instead of once per request (D8).
_LOGGED: set[str] = set()


def _log_once(key: str, message: str) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    logger.warning(message)
