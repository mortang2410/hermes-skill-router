"""Contract C4: the routed-only skill index, its fingerprint, cache and per-fingerprint memo.

The catalog payloads below are built in the shape the renderer emits, then partitioned by the real
``strip_catalog``, so every index in this file comes from the same path a live request uses.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows, strip_catalog  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.library import (  # noqa: E402
    CACHE_VERSION,
    document_text,
    fingerprint,
    load_or_build,
)

STABLE = "You are Hermes Agent."
MEMORY = "## Memory\n\nNothing yet."
NORMAL_PROSE = (
    "## Skills\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant to "
    "your task, you MUST load it with skill_view(name) and follow its instructions.\n"
)
TRAILING_GUIDANCE = "Only proceed without loading a skill if genuinely none are relevant to the task."

ROUTED_ROWS = (("alpha", "Alpha skill"), ("beta", "Beta skill"), ("gamma", "Gamma skill"))
BASE_ROWS = CatalogRows(("alpha", "beta"), ("Alpha skill", "Beta skill"))


class CountingEmbedder:
    """Deterministic stand-in for M5 that records every text it is asked to embed."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.texts = []
        self._lock = threading.Lock()

    def __call__(self, cfg, texts):
        with self._lock:
            self.calls += 1
            self.texts.extend(texts)
        if self.delay:
            time.sleep(self.delay)
        return [[float(len(text)), 1.0] for text in texts]


class ChunkRecordingEmbedder:
    """Record each request, and return a vector that encodes the row's position.

    Encoding the position is what proves the vectors are concatenated in row order across chunk
    boundaries. ``fail_at`` and ``short_at`` name the call index that returns ``None`` or one vector
    fewer than it was asked for.
    """

    def __init__(self, order, fail_at=None, short_at=None) -> None:
        self.position = {text: index for index, text in enumerate(order)}
        self.chunks = []
        self.fail_at = fail_at
        self.short_at = short_at

    @property
    def calls(self):
        return len(self.chunks)

    def __call__(self, cfg, texts):
        call = self.calls
        self.chunks.append(list(texts))
        if call == self.fail_at:
            return None
        vectors = [[float(self.position[text]), 0.0] for text in texts]
        return vectors[:-1] if call == self.short_at else vectors


def make_config(tmp_path, **overrides):
    settings = {"cachePath": str(tmp_path / "index.json")}
    settings.update(overrides)
    return load(settings)


@pytest.fixture(autouse=True)
def empty_memo(monkeypatch):
    """Start every test as a fresh process, so the memo never leaks between tests."""
    monkeypatch.setattr(library_module, "_MEMO", {})


def render_content(rows=ROUTED_ROWS, demoted=()):
    lines = []
    if demoted:
        # The renderer sorts categories, so a demoted "creative" line precedes "general".
        lines.append("  creative [names only]: " + ", ".join(sorted(demoted)))
    lines.append("  general:")
    for name, description in rows:
        lines.append(f"    - {name}: {description}" if description else f"    - {name}")
    block = "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"
    return (
        STABLE + "\n\n" + NORMAL_PROSE + "\n" + block + "\n\n" + TRAILING_GUIDANCE + "\n\n" + MEMORY
    )


def payload(rows=ROUTED_ROWS, demoted=()):
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": render_content(rows, demoted)},
            {"role": "user", "content": "route me"},
        ],
    }


def split(rows=ROUTED_ROWS, demoted=(), **options):
    """Strip the payload and return the real partition the index must be built from."""
    stripped, partition = strip_catalog(payload(rows, demoted), **options)
    assert partition is not None
    assert "<available_skills>" not in stripped["messages"][0]["content"]
    return partition


# ---------------------------------------------------------------------------
# The index is the routed half of the catalog, under the DSH document text.
# ---------------------------------------------------------------------------


def test_the_index_is_the_routed_rows_under_the_document_text(tmp_path):
    routed = split().routed
    embedder = CountingEmbedder()
    cfg = make_config(tmp_path)

    index = load_or_build(cfg, routed, embedder)

    assert index.names == ["alpha", "beta", "gamma"]
    assert index.descriptions == ["Alpha skill", "Beta skill", "Gamma skill"]
    assert index.vectors == [[float(len(text)), 1.0] for text in embedder.texts]
    assert index.dim == 2
    assert index.fingerprint == fingerprint(routed, cfg)
    assert embedder.calls == 1
    assert embedder.texts == ["alpha\nAlpha skill", "beta\nBeta skill", "gamma\nGamma skill"]


def test_document_text_is_name_newline_clipped_description(tmp_path):
    cfg = make_config(tmp_path)
    assert document_text(cfg, "alpha", "Alpha skill") == "alpha\nAlpha skill"
    assert document_text(cfg, "alpha", "") == "alpha\n"

    clipped = make_config(tmp_path, maxDescriptionChars=10)
    assert document_text(clipped, "alpha", "0123456789abcdef") == "alpha\n0123456789"


def test_a_long_description_is_embedded_clipped(tmp_path):
    cfg = make_config(tmp_path, maxDescriptionChars=12)
    description = "A" * 12 + "tail that must never reach the embedder"
    embedder = CountingEmbedder()

    index = load_or_build(cfg, CatalogRows(("alpha",), (description,)), embedder)

    assert document_text(cfg, "alpha", description) == "alpha\n" + "A" * 12
    assert embedder.texts == ["alpha\n" + "A" * 12]
    assert index.names == ["alpha"]


def test_descriptions_differing_only_past_the_clip_share_the_document_text_not_the_fingerprint(
    tmp_path,
):
    cfg = make_config(tmp_path, maxDescriptionChars=12)
    prefix = "A" * 12
    first = CatalogRows(("alpha", "beta"), ("Alpha skill", prefix + "tail one"))
    second = CatalogRows(("alpha", "beta"), ("Alpha skill", prefix + "tail two"))

    assert document_text(cfg, "beta", prefix + "tail one") == document_text(
        cfg, "beta", prefix + "tail two"
    )
    assert fingerprint(first, cfg) != fingerprint(second, cfg)


def test_a_description_suffix_past_the_clip_invalidates_the_cached_text(monkeypatch, tmp_path):
    cfg = make_config(tmp_path, maxDescriptionChars=12)
    prefix = "A" * 12
    first = CatalogRows(("alpha", "beta"), ("Alpha skill", prefix + "tail one"))
    second = CatalogRows(("alpha", "beta"), ("Alpha skill", prefix + "tail two"))

    warm = CountingEmbedder()
    built = load_or_build(cfg, first, warm)
    assert warm.texts == ["alpha\nAlpha skill", "beta\n" + prefix]

    monkeypatch.setattr(library_module, "_MEMO", {})
    embedder = CountingEmbedder()
    rebuilt = load_or_build(cfg, second, embedder)

    assert embedder.calls == 1, "a description edit past the clip served the cached text"
    assert embedder.texts == ["alpha\nAlpha skill", "beta\n" + prefix]
    assert rebuilt.fingerprint != built.fingerprint
    # The vectors cannot tell the two rows apart; the cached descriptions must, because they also
    # feed the rerank documents and the injected block.
    assert rebuilt.vectors == built.vectors
    assert rebuilt.descriptions == ["Alpha skill", prefix + "tail two"]
    cache = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert cache["descriptions"] == ["Alpha skill", prefix + "tail two"]
    assert cache["fingerprint"] == rebuilt.fingerprint


def test_a_shorter_clip_changes_the_fingerprint_and_rebuilds_a_long_description(monkeypatch, tmp_path):
    description = "A" * 30
    rows = CatalogRows(("alpha",), (description,))
    wide = make_config(tmp_path, maxDescriptionChars=30)
    narrow = make_config(tmp_path, maxDescriptionChars=10)

    assert document_text(wide, "alpha", description) != document_text(narrow, "alpha", description)
    assert fingerprint(rows, wide) != fingerprint(rows, narrow)

    warm = CountingEmbedder()
    load_or_build(wide, rows, warm)
    monkeypatch.setattr(library_module, "_MEMO", {})
    embedder = CountingEmbedder()
    rebuilt = load_or_build(narrow, rows, embedder)

    assert embedder.calls == 1, "the cache built from the wider clip was trusted"
    assert embedder.texts == ["alpha\n" + "A" * 10]
    assert rebuilt.fingerprint == fingerprint(rows, narrow)


def test_a_catalog_with_only_eager_rows_builds_no_index(tmp_path):
    eager_only = split(rows=(("alpha", "Alpha skill"),), eager=["alpha"])
    assert eager_only.routed.names == ()

    embedder = CountingEmbedder()
    assert load_or_build(make_config(tmp_path), eager_only.routed, embedder) is None
    assert embedder.calls == 0
    assert not (tmp_path / "index.json").exists()


def test_rows_outside_a_deferred_list_are_not_indexed(tmp_path):
    deferred = split(deferred=["beta"])
    assert deferred.routed.names == ("beta",)

    index = load_or_build(make_config(tmp_path), deferred.routed, CountingEmbedder())

    assert index.names == ["beta"]


def test_a_demoted_names_only_line_is_never_embedded_scored_or_cached(tmp_path):
    demoted = ("pdf", "xlsx")
    catalog = split(demoted=demoted)
    assert set(demoted) <= set(catalog.kept.names)
    assert not set(demoted) & set(catalog.routed.names)

    cfg = make_config(tmp_path)
    embedder = CountingEmbedder()
    index = load_or_build(cfg, catalog.routed, embedder)

    assert index.names == ["alpha", "beta", "gamma"]
    # The rerank documents M6 builds come from these rows, so this is every document that can
    # ever be scored for this catalog.
    documents = [
        f"{cfg.document_prefix}{name} - {description}"
        for name, description in zip(index.names, index.descriptions)
    ]
    for name in demoted:
        assert name not in index.names
        assert all(name not in text for text in embedder.texts)
        assert all(name not in document for document in documents)
    cache_text = (tmp_path / "index.json").read_text(encoding="utf-8")
    for name in demoted:
        assert name not in cache_text


def test_a_names_only_name_named_in_deferred_skills_stays_out_of_the_index(tmp_path):
    catalog = split(demoted=("pdf",), deferred=["pdf"])
    assert "pdf" in catalog.kept.names
    assert catalog.routed.names == ()

    embedder = CountingEmbedder()
    assert load_or_build(make_config(tmp_path), catalog.routed, embedder) is None
    assert embedder.calls == 0
    assert not (tmp_path / "index.json").exists()


def test_duplicate_names_collapse_to_the_first_occurrence(tmp_path):
    rows = CatalogRows(("alpha", "beta", "alpha"), ("First", "Beta skill", "Second"))
    embedder = CountingEmbedder()

    index = load_or_build(make_config(tmp_path), rows, embedder)

    assert index.names == ["alpha", "beta"]
    assert index.descriptions == ["First", "Beta skill"]
    assert len(index.vectors) == 2
    assert embedder.texts == ["alpha\nFirst", "beta\nBeta skill"]
    collapsed = CatalogRows(("alpha", "beta"), ("First", "Beta skill"))
    assert index.fingerprint == fingerprint(collapsed, make_config(tmp_path))


def test_empty_rows_build_nothing(tmp_path):
    embedder = CountingEmbedder()

    assert load_or_build(make_config(tmp_path), CatalogRows(), embedder) is None

    assert embedder.calls == 0
    assert not (tmp_path / "index.json").exists()


def test_a_failed_embedding_returns_none_and_writes_no_cache(tmp_path):
    def failing(cfg, texts):
        return None

    cfg = make_config(tmp_path)
    routed = split().routed

    assert load_or_build(cfg, routed, failing) is None
    assert not (tmp_path / "index.json").exists()

    # A failed build leaves no memo behind, so the next attempt tries again.
    recovered = CountingEmbedder()
    assert load_or_build(cfg, routed, recovered).names == list(routed.names)
    assert recovered.calls == 1


@pytest.mark.parametrize(
    "vectors",
    [None, [], [[1.0, 2.0]], [[1.0, 2.0], [1.0]], [[1.0, 2.0], []], [[1.0, 2.0], [3.0, 4.0, 5.0]]],
    ids=["none", "empty", "too-few", "mixed-width", "empty-vector", "extra-width"],
)
def test_an_unusable_embedding_result_builds_no_index(tmp_path, vectors):
    def embed(cfg, texts):
        return vectors

    assert load_or_build(make_config(tmp_path), BASE_ROWS, embed) is None
    assert not (tmp_path / "index.json").exists()


# ---------------------------------------------------------------------------
# A cold build is chunked, so the request timeout bounds one call, not the library.
# ---------------------------------------------------------------------------


def rows_of(count, prefix="row"):
    """Return *count* distinct routed rows with short descriptions."""
    return CatalogRows(
        tuple(f"{prefix}-{index:04d}" for index in range(count)),
        tuple(f"Description of {prefix} {index}" for index in range(count)),
    )


def texts_of(cfg, rows):
    return [
        document_text(cfg, name, description)
        for name, description in zip(rows.names, rows.descriptions)
    ]


def test_a_424_row_build_makes_twenty_seven_calls_of_at_most_sixteen(tmp_path):
    cfg = make_config(tmp_path)
    rows = rows_of(424)
    texts = texts_of(cfg, rows)
    embedder = ChunkRecordingEmbedder(texts)

    index = load_or_build(cfg, rows, embedder)

    assert embedder.calls == 27, "424 rows must be 26 full batches plus a remainder"
    assert [len(chunk) for chunk in embedder.chunks] == [16] * 26 + [8]
    assert max(len(chunk) for chunk in embedder.chunks) <= 16
    assert [text for chunk in embedder.chunks for text in chunk] == texts
    assert index.names == list(rows.names)
    # Each vector carries its row's position, so equality proves order across chunk boundaries.
    assert index.vectors == [[float(position), 0.0] for position in range(424)]
    assert index.dim == 2


@pytest.mark.parametrize(
    ("count", "expected_sizes"),
    [(1, [1]), (16, [16]), (17, [16, 1]), (32, [16, 16]), (33, [16, 16, 1])],
)
def test_batch_boundaries_never_exceed_sixteen_and_keep_row_order(tmp_path, count, expected_sizes):
    cfg = make_config(tmp_path)
    rows = rows_of(count, prefix="edge")
    texts = texts_of(cfg, rows)
    embedder = ChunkRecordingEmbedder(texts)

    index = load_or_build(cfg, rows, embedder)

    assert [len(chunk) for chunk in embedder.chunks] == expected_sizes
    assert max(len(chunk) for chunk in embedder.chunks) <= 16
    assert [text for chunk in embedder.chunks for text in chunk] == texts
    assert index.vectors == [[float(position), 0.0] for position in range(count)]


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_a_failed_chunk_fails_the_build_and_writes_no_cache(tmp_path, fail_at):
    cfg = make_config(tmp_path)
    rows = rows_of(48, prefix="fail")
    texts = texts_of(cfg, rows)
    embedder = ChunkRecordingEmbedder(texts, fail_at=fail_at)

    assert load_or_build(cfg, rows, embedder) is None

    assert embedder.calls == fail_at + 1, "the build kept going after a chunk failed"
    assert not (tmp_path / "index.json").exists()

    # A failed build leaves no memo behind, so a working build recovers.
    recovery = ChunkRecordingEmbedder(texts)
    assert load_or_build(cfg, rows, recovery).names == list(rows.names)
    assert recovery.calls == 3


def test_a_short_chunk_answer_fails_the_whole_build(tmp_path):
    cfg = make_config(tmp_path)
    rows = rows_of(20, prefix="short")
    texts = texts_of(cfg, rows)
    embedder = ChunkRecordingEmbedder(texts, short_at=1)

    assert load_or_build(cfg, rows, embedder) is None

    assert embedder.calls == 2
    assert not (tmp_path / "index.json").exists()


def test_a_chunk_with_a_different_vector_width_fails_the_whole_build(tmp_path):
    cfg = make_config(tmp_path)
    rows = rows_of(20, prefix="width")
    sizes = []

    def embed(cfg, texts):
        sizes.append(len(texts))
        width = 2 if len(sizes) == 1 else 3
        return [[0.0] * width for _text in texts]

    assert load_or_build(cfg, rows, embed) is None

    assert sizes == [16, 4]
    assert not (tmp_path / "index.json").exists()


# ---------------------------------------------------------------------------
# The fingerprint: what forces a rebuild and what cannot.
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_stable_and_hex(tmp_path):
    cfg = make_config(tmp_path)

    first = fingerprint(BASE_ROWS, cfg)

    assert first == fingerprint(BASE_ROWS, cfg)
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    assert first != fingerprint(CatalogRows(("alpha", "beta"), ("Beta skill", "Alpha skill")), cfg)


def test_the_fingerprint_collapses_duplicate_names_like_the_index(tmp_path):
    cfg = make_config(tmp_path)
    duplicated = CatalogRows(("alpha", "beta", "alpha"), ("First", "Beta", "Second"))
    collapsed = CatalogRows(("alpha", "beta"), ("First", "Beta"))

    assert fingerprint(duplicated, cfg) == fingerprint(collapsed, cfg)


def changed(kind, tmp_path):
    """Return the rows and config for one change that must reach the fingerprint."""
    if kind == "name":
        return CatalogRows(("alpha", "gamma"), ("Alpha skill", "Beta skill")), make_config(tmp_path)
    if kind == "description":
        return (
            CatalogRows(("alpha", "beta"), ("Alpha skill, rewritten", "Beta skill")),
            make_config(tmp_path),
        )
    if kind == "embedModelId":
        return BASE_ROWS, make_config(tmp_path, embedModelId="another-model")
    if kind == "embedPath":
        return BASE_ROWS, make_config(tmp_path, embedPath="/v1/other")
    return BASE_ROWS, make_config(tmp_path, embedBaseUrl="http://127.0.0.1:8013")


@pytest.mark.parametrize("kind", ["name", "description", "embedModelId", "embedBaseUrl", "embedPath"])
def test_a_change_the_index_depends_on_changes_the_fingerprint_and_forces_a_rebuild(
    monkeypatch, tmp_path, kind
):
    cfg = make_config(tmp_path)
    warm = CountingEmbedder()
    built = load_or_build(cfg, BASE_ROWS, warm)
    assert warm.calls == 1 and built is not None

    rows, changed_cfg = changed(kind, tmp_path)
    assert fingerprint(rows, changed_cfg) != built.fingerprint

    monkeypatch.setattr(library_module, "_MEMO", {})  # a fresh process reading the same cache file
    embedder = CountingEmbedder()
    rebuilt = load_or_build(changed_cfg, rows, embedder)

    assert embedder.calls == 1, "the cache was trusted after the fingerprint changed"
    assert rebuilt.fingerprint == fingerprint(rows, changed_cfg)


def test_a_changed_kept_row_leaves_the_fingerprint_and_cache_intact(monkeypatch, tmp_path):
    cfg = make_config(tmp_path)
    warm = CountingEmbedder()
    built = load_or_build(cfg, split(eager=["alpha"]).routed, warm)
    assert built.names == ["beta", "gamma"]

    rows = (("alpha", "Alpha skill, rewritten"),) + ROUTED_ROWS[1:]
    changed_catalog = split(rows=rows, eager=["alpha"])
    assert changed_catalog.routed == split(eager=["alpha"]).routed
    assert fingerprint(changed_catalog.routed, cfg) == built.fingerprint

    monkeypatch.setattr(library_module, "_MEMO", {})
    embedder = CountingEmbedder()
    cached = load_or_build(cfg, changed_catalog.routed, embedder)

    assert embedder.calls == 0, "a kept row change rebuilt the index"
    assert cached.fingerprint == built.fingerprint


# ---------------------------------------------------------------------------
# The persisted cache: written atomically, trusted only when it matches.
# ---------------------------------------------------------------------------


def test_a_cache_hit_performs_zero_embedding_calls(monkeypatch, tmp_path):
    cfg = make_config(tmp_path)
    routed = split().routed
    warm = CountingEmbedder()
    built = load_or_build(cfg, routed, warm)
    assert warm.calls == 1

    monkeypatch.setattr(library_module, "_MEMO", {})  # a restarted process
    cold = CountingEmbedder()
    cached = load_or_build(cfg, routed, cold)

    assert cold.calls == 0
    assert cached.names == built.names
    assert cached.descriptions == built.descriptions
    assert cached.vectors == built.vectors
    assert cached.dim == built.dim
    assert cached.fingerprint == built.fingerprint


def test_the_cache_is_valid_json_and_the_directory_is_created(tmp_path):
    cache_file = tmp_path / "nested" / "deeper" / "index.json"
    cfg = make_config(tmp_path, cachePath=str(cache_file))

    index = load_or_build(cfg, BASE_ROWS, CountingEmbedder())

    data = json.loads(cache_file.read_text(encoding="utf-8"))
    assert data["version"] == CACHE_VERSION
    assert data["fingerprint"] == index.fingerprint
    assert data["dim"] == index.dim
    assert data["names"] == index.names
    assert data["vectors"] == index.vectors


def test_the_cache_is_written_through_a_temporary_file(monkeypatch, tmp_path):
    cache_file = tmp_path / "index.json"
    replaced = []
    real_replace = os.replace

    def spy(source, destination):
        replaced.append((str(source), str(destination)))
        assert os.path.isdir(os.path.dirname(str(destination)))
        assert os.path.dirname(str(source)) == os.path.dirname(str(destination))
        assert os.path.exists(source)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", spy)

    index = load_or_build(make_config(tmp_path), BASE_ROWS, CountingEmbedder())

    assert index is not None
    assert len(replaced) == 1
    assert replaced[0][1] == str(cache_file)
    assert replaced[0][0] != str(cache_file)
    assert cache_file.is_file()
    assert not list(tmp_path.glob(".index-*"))


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda data: data.update(version=CACHE_VERSION + 1), id="version"),
        pytest.param(lambda data: data.update(dim=99), id="dim"),
        pytest.param(lambda data: data["vectors"].pop(), id="missing-vector"),
        pytest.param(lambda data: data["names"].append("ghost"), id="extra-name"),
        pytest.param(lambda data: data.update(fingerprint="0" * 64), id="fingerprint"),
        pytest.param(lambda data: data["vectors"][0].pop(), id="narrow-vector"),
    ],
)
def test_an_untrusted_cache_is_rebuilt(monkeypatch, tmp_path, tamper):
    cfg = make_config(tmp_path)
    routed = split().routed
    built = load_or_build(cfg, routed, CountingEmbedder())
    cache_file = tmp_path / "index.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    tamper(data)
    cache_file.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(library_module, "_MEMO", {})
    embedder = CountingEmbedder()
    rebuilt = load_or_build(cfg, routed, embedder)

    assert embedder.calls == 1
    assert rebuilt.fingerprint == built.fingerprint
    assert rebuilt.dim == 2

    monkeypatch.setattr(library_module, "_MEMO", {})
    reader = CountingEmbedder()
    assert load_or_build(cfg, routed, reader).fingerprint == built.fingerprint
    assert reader.calls == 0, "the rebuilt cache was not written back"


def test_a_corrupt_cache_file_is_rebuilt(monkeypatch, tmp_path, caplog):
    cfg = make_config(tmp_path)
    cache_file = tmp_path / "index.json"
    cache_file.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        index = load_or_build(cfg, BASE_ROWS, CountingEmbedder())

    assert index is not None
    assert any("unreadable" in record.getMessage() for record in caplog.records)
    json.loads(cache_file.read_text(encoding="utf-8"))


def test_a_cache_write_failure_logs_once_and_returns_the_index(monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "cache-is-a-directory"
    blocked.mkdir()
    local = make_config(tmp_path, cachePath=str(blocked))
    other = make_config(tmp_path, cachePath=str(blocked), embedModelId="another-model")

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        first = load_or_build(local, BASE_ROWS, CountingEmbedder())
        second = load_or_build(other, BASE_ROWS, CountingEmbedder())

    assert first is not None and second is not None
    writes = [record for record in caplog.records if "could not write" in record.getMessage()]
    assert len(writes) == 1, "the same write failure was logged more than once"
    assert not list(blocked.iterdir()), "the failed write left its temporary file behind"


def test_an_empty_cache_path_disables_persistence(tmp_path):
    cfg = make_config(tmp_path, cachePath="")

    index = load_or_build(cfg, BASE_ROWS, CountingEmbedder())

    assert index is not None
    assert list(tmp_path.iterdir()) == []


SECOND_PROCESS = """
import json
import sys

from hermes_skill_router.catalog import CatalogRows
from hermes_skill_router.config import load
from hermes_skill_router.library import load_or_build


def embed(cfg, texts):
    raise SystemExit("the second process embedded the library")


cfg = load({"cachePath": sys.argv[1]})
rows = CatalogRows(tuple(json.loads(sys.argv[2])), tuple(json.loads(sys.argv[3])))
index = load_or_build(cfg, rows, embed)
print(json.dumps({"names": index.names, "dim": index.dim, "fingerprint": index.fingerprint}))
"""


def test_the_cache_survives_a_second_process(tmp_path):
    cfg = make_config(tmp_path)
    routed = split().routed
    built = load_or_build(cfg, routed, CountingEmbedder())
    cache_file = tmp_path / "index.json"
    json.loads(cache_file.read_text(encoding="utf-8"))

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SECOND_PROCESS,
            str(cache_file),
            json.dumps(list(routed.names)),
            json.dumps(list(routed.descriptions)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "names": built.names,
        "dim": built.dim,
        "fingerprint": built.fingerprint,
    }


# ---------------------------------------------------------------------------
# The memo: one build per fingerprint, even with two threads on a cold cache.
# ---------------------------------------------------------------------------


def test_two_threads_on_a_cold_cache_embed_once(tmp_path):
    cfg = make_config(tmp_path)
    # 40 rows is three batches, so one build is three calls and a raced second build would be six.
    rows = rows_of(40, prefix="thread")
    embedder = CountingEmbedder(delay=0.05)
    barrier = threading.Barrier(2)
    results = []

    def worker():
        barrier.wait(timeout=10)
        results.append(load_or_build(cfg, rows, embedder))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert embedder.calls == 3, "a second build would have added three more calls"
    assert len(results) == 2
    assert all(result is not None for result in results)
    assert results[0].fingerprint == results[1].fingerprint


def test_a_failed_cache_replace_keeps_the_previous_cache_and_removes_its_sibling(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)
    assert load_or_build(cfg, BASE_ROWS, CountingEmbedder()) is not None
    before = (tmp_path / "index.json").read_text()
    newer = CatalogRows(
        ("alpha", "beta", "gamma"), ("Alpha skill", "Beta skill", "Gamma skill")
    )
    monkeypatch.setattr(
        library_module.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    index = load_or_build(cfg, newer, CountingEmbedder())

    assert index is not None, "the in-memory index survives a cache write failure"
    assert (tmp_path / "index.json").read_text() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["index.json"]


def test_a_refused_sibling_removal_keeps_one_file_and_names_it(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)
    assert load_or_build(cfg, BASE_ROWS, CountingEmbedder()) is not None
    before = (tmp_path / "index.json").read_text()
    newer = CatalogRows(
        ("alpha", "beta", "gamma"), ("Alpha skill", "Beta skill", "Gamma skill")
    )
    monkeypatch.setattr(
        library_module.os,
        "replace",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        library_module.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("busy"))
    )

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        index = load_or_build(cfg, newer, CountingEmbedder())

    assert index is not None
    assert (tmp_path / "index.json").read_text() == before
    siblings = [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]
    assert len(siblings) == 1, "exactly the one file the removal refused to discard"
    messages = " ".join(caplog.messages)
    assert str(siblings[0]) in messages, "the kept sibling's path is logged"
