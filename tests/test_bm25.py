"""Contract C15: the BM25 tokeniser, scorer and persisted term index.

The term index is the BM25 channel's whole state. These tests pin the tokenisation rules the
reference router fixes, the Okapi constants, the order a long document cannot win by length alone,
and the digest that refuses a file built from other text. The library owns the build call; here the
module is driven directly so a failure names the rule that broke, not a pipeline symptom.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import bm25  # noqa: E402
from hermes_skill_router.bm25 import (  # noqa: E402
    BM25_B,
    BM25_K1,
    TERM_INDEX_FILENAME,
    TERM_INDEX_VERSION,
    build_term_index,
    catalog_digest,
    load_term_index,
    score_ranks,
    tokenise,
    write_term_index,
)

CATALOG = (
    ("alpha", "Review the association store migration path"),
    ("beta", "Draft a git commit message for a bug fix"),
    ("gamma", "Write a table of contents for the guide"),
)

FORMULA_CATALOG = (
    ("short", "review store"),
    ("long", "review store migration path with many words about a route through the store"),
    ("other", "unrelated text"),
)


# ---------------------------------------------------------------------------
# Tokenisation: the rules are fixed, so they are asserted one by one.
# ---------------------------------------------------------------------------


def test_tokens_are_lowercased_and_split_on_non_alphanumerics():
    assert tokenise("Deploy_the API: v2.0 NOW!") == ["deploy", "api", "v2", "now"]
    assert tokenise("multi-word,slash/separated") == ["multi", "word", "slash", "separated"]


def test_a_one_character_token_is_dropped_in_any_script():
    assert tokenise("a 日 ab") == ["ab"]
    assert tokenise("I am a token") == ["am", "token"]


def test_one_short_stopword_list_is_dropped_and_nothing_is_stemmed():
    assert tokenise("the and of their them should") == []
    assert tokenise("running runs runner") == ["running", "runs", "runner"]


def test_non_latin_terms_are_kept():
    assert tokenise("café 日本語") == ["café", "日本語"]


def test_a_non_string_has_no_terms():
    assert tokenise(None) == []
    assert tokenise(17) == []


def test_the_constants_and_file_name_are_pinned():
    assert (BM25_K1, BM25_B) == (1.2, 0.75)
    assert TERM_INDEX_FILENAME == "bm25.json"
    assert TERM_INDEX_VERSION == 1


# ---------------------------------------------------------------------------
# Scoring: Okapi BM25 over the catalog, deterministic and length-normalised.
# ---------------------------------------------------------------------------


def test_a_short_document_beats_a_long_one_with_the_same_term():
    index = build_term_index(FORMULA_CATALOG)

    ranked = score_ranks(index, "store", 3)

    assert ranked == ["short", "long"], "length normalisation did not correct the long document"
    assert score_ranks(index, "store", 1) == ["short"]
    assert score_ranks(index, "store", 0) == []


def test_a_query_term_absent_from_every_document_contributes_nothing():
    index = build_term_index(FORMULA_CATALOG)
    assert score_ranks(index, "zebracorn", 3) == []
    assert score_ranks(index, "store zebracorn", 3) == ["short", "long"]


def test_scoring_is_deterministic_and_ties_keep_catalog_order():
    index = build_term_index(FORMULA_CATALOG)
    first = score_ranks(index, "store", 3)
    second = score_ranks(index, "store", 3)
    assert first == second

    tied = (("one", "review"), ("two", "review"), ("three", "review"))
    assert score_ranks(build_term_index(tied), "review", 3) == ["one", "two", "three"]


def test_an_empty_index_or_a_stopword_query_has_no_ranks():
    assert score_ranks(build_term_index(()), "review", 3) == []
    assert score_ranks(None, "review", 3) == []
    assert score_ranks(build_term_index(CATALOG), "the and of", 3) == []


# ---------------------------------------------------------------------------
# The persisted index: built atomically, loaded only for the rows it covers.
# ---------------------------------------------------------------------------


def test_the_index_round_trips_for_the_same_rows(tmp_path):
    path = tmp_path / "bm25.json"
    built = build_term_index(CATALOG, str(path))
    assert path.is_file()
    assert not list(tmp_path.glob(".bm25-*"))

    loaded = load_term_index(str(path), CATALOG)

    assert loaded == built
    assert loaded is not None
    assert score_ranks(loaded, "review store", 5)[0] == "alpha"


def test_a_term_index_built_from_other_rows_is_refused(tmp_path):
    path = tmp_path / "bm25.json"
    build_term_index(CATALOG, str(path))

    assert load_term_index(str(path), CATALOG[:2]) is None
    assert load_term_index(str(path), CATALOG + (("delta", "extra row"),)) is None


def test_a_missing_corrupt_or_tampered_file_loads_as_none(tmp_path):
    path = tmp_path / "bm25.json"
    built = build_term_index(CATALOG, str(path))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert load_term_index(str(tmp_path / "absent.json"), CATALOG) is None

    path.write_text("{not json", encoding="utf-8")
    assert load_term_index(str(path), CATALOG) is None

    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_term_index(str(path), CATALOG) is not None

    tampered = dict(data)
    postings = {term: list(values) for term, values in data["postings"].items()}
    first_term = next(iter(postings))
    postings[first_term][1] += 1
    tampered["postings"] = postings
    path.write_text(json.dumps(tampered), encoding="utf-8")
    assert load_term_index(str(path), CATALOG) is None, "an edited posting was trusted"

    wrong = dict(data)
    wrong["k1"] = 1.5
    path.write_text(json.dumps(wrong), encoding="utf-8")
    assert load_term_index(str(path), CATALOG) is None, "a file built under other constants loaded"

    shaped = dict(data)
    shaped["postings"] = []
    path.write_text(json.dumps(shaped), encoding="utf-8")
    assert load_term_index(str(path), CATALOG) is None
    assert built.version == TERM_INDEX_VERSION


def test_the_digest_covers_the_document_texts():
    first = catalog_digest([("alpha", "one text")])
    second = catalog_digest([("alpha", "another text")])
    assert first != second
    assert first == catalog_digest([("alpha", "one text")])
    assert catalog_digest([("alpha", "a"), ("beta", "bc")]) != catalog_digest(
        [("alpha", "ab"), ("beta", "c")]
    ), "the framing is ambiguous across rows"


def test_writing_a_term_index_leaves_no_temporary_sibling(tmp_path):
    path = tmp_path / "nested" / "bm25.json"
    index = build_term_index(CATALOG)
    write_term_index(str(path), index)

    assert path.is_file()
    assert [entry.name for entry in path.parent.iterdir()] == ["bm25.json"]
