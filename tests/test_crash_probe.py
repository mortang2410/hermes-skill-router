"""Contract C8 probe: the four bad association-store inputs and what the middleware does with them.

This file is a probe, not a fix. The crash report that names these inputs is unverified, so these
tests record what the current code actually does. Measured on the pre-change tree, whose HEAD was
1d46708:

- missing file: the middleware's open helper skips the store before calling ``open_store``; a raw
  read-only ``open_store`` raises ``AssociationStoreError`` because it cannot create a missing file.
- zero-byte file: a raw read-only ``open_store`` raises ``AssociationStoreError`` (not a version-1
  database yet); the middleware's open helper catches it and the turn routes without association
  ranks.
- version-1 schema with tables but no rows: a raw read-only ``open_store`` succeeds, the read
  returns no association rank, and the turn routes without association ranks.
- a directory in the store's place: a raw read-only ``open_store`` raises
  ``AssociationStoreError`` (cannot open the path); the middleware's open helper catches it and
  the turn routes without association ranks.

In all four cases the middleware itself does not raise. The request mapping it receives stays
byte-identical (the rewrite never mutates its input), and the semantic and BM25 channels still
return ranks, so the turn is rewritten from those two channels with no association rank.

``byte-identical`` is asserted on the request mapping the middleware receives, because the rewrite
replaces the payload rather than editing it in place. The returned payload still differs when
routing succeeds: that difference is the suggestion block, not store damage.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from hermes_skill_router import associations  # noqa: E402
from hermes_skill_router import bm25  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from tests.test_middleware_contract import (  # noqa: E402
    DISPATCH_KWARGS,
    _callback_for,
    _load_plugin_module,
    chat_payload,
    endpoints,
    settings_for,
)

#: The user text both channels can rank: "alpha" and "beta" are catalog names and BM25 terms.
QUERY = "route alpha beta"

#: The catalog rows the shortlist fixture indexes, in the catalog's own document-text shape.
INDEXED_ROWS = [("alpha", "alpha\nAlpha skill"), ("beta", "beta\nBeta skill")]


@pytest.fixture(scope="module")
def plugin_module():
    """The repository's plugin entry point, loaded the way the plugin host loads a directory."""
    return _load_plugin_module()


@pytest.fixture(autouse=True)
def _host_seams(monkeypatch):
    """The host lookup resolves every name, and no dump record leaks between the probe turns.

    The plugin entry point loads its own copy of ``hermes_skill_router.*`` under the plugin's
    package name, so the patched voice has to reach every loaded router module, exactly as the
    middleware contract test's fixture does.
    """
    from hermes_skill_router import search as search_module

    search_module.clear_dumps()
    resolver = lambda name: f"/skills/{name}/SKILL.md"
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("router") and hasattr(module, "skill_path"):
            monkeypatch.setattr(module, "skill_path", resolver)
            monkeypatch.setattr(module, "_broadcast", lambda turn_id: None)
        elif module_name.endswith("search") and hasattr(module, "clear_dumps"):
            module.clear_dumps()


def _canonical(mapping) -> bytes:
    """Return one mapping's bytes with its insertion order kept, the strict byte-identical check."""
    return json.dumps(mapping, separators=(",", ":")).encode("utf-8")


def _query_payload():
    """Return the middleware contract's payload with a query both routing channels can rank."""
    payload = chat_payload()
    payload["messages"][-1] = {"role": "user", "content": QUERY}
    return payload


def _raw_open(store_path):
    """Open the store read-only with no middleware guard, and return the exception or ``None``."""
    try:
        store = associations.open_store(str(store_path), read_only=True)
    except Exception as exc:
        return exc
    store.close()
    return None


def _shortlist_probe(cfg):
    """Run the library's open path over the bad store and return its rows and channel diagnostics."""
    index = Index(
        names=[name for name, _text in INDEXED_ROWS],
        descriptions=["Alpha skill", "Beta skill"],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        dim=2,
        fingerprint="crash-probe",
        term_index=bm25.build_term_index(INDEXED_ROWS),
    )
    records = []
    try:
        rows = library_module.shortlist(
            index, [1.0, 0.0], QUERY, 5, cfg=cfg, on_diagnostic=records.append
        )
    except Exception as exc:
        return exc, [], []
    return None, rows, records


def _probe_turn(store_path, tmp_path, plugin_module, endpoints_fixture):
    """Drive one turn through the real middleware callback with the bad store configured."""
    settings = dict(settings_for(endpoints_fixture, tmp_path / "suggestions.json"))
    settings["associationDbPath"] = str(store_path)
    cfg = load(settings)
    # The middleware loads the same mapping through read_settings, so this guard keeps the probe
    # from passing while the bad store was never configured at all.
    assert cfg.association_db_path == str(store_path)

    shortlist_raised, rows, records = _shortlist_probe(cfg)

    payload = _query_payload()
    snapshot = copy.deepcopy(payload)
    before = _canonical(payload)
    callback = _callback_for(plugin_module, settings)
    middleware_raised = None
    result = None
    try:
        result = callback(request=payload, **DISPATCH_KWARGS)
    except Exception as exc:
        middleware_raised = exc

    return SimpleNamespace(
        middleware_raised=middleware_raised,
        result=result,
        payload=payload,
        snapshot=snapshot,
        before=before,
        shortlist_raised=shortlist_raised,
        rows=rows,
        records=records,
    )


def _middleware_store_warnings(caplog, plugin_module):
    """Return the warnings the middleware's own library copy logged about the association store."""
    logger_name = f"{plugin_module.__name__}.hermes_skill_router.library"
    return [
        record
        for record in caplog.records
        if record.name == logger_name and "association database" in record.getMessage()
    ]


def _assert_routed_without_crashing(probe):
    """Assert the recorded outcome: no raise at either layer, request untouched, both channels route."""
    assert probe.middleware_raised is None, (
        f"the middleware raised on this store input: {probe.middleware_raised!r}"
    )
    assert probe.payload == probe.snapshot, "the middleware mutated the request mapping in place"
    assert _canonical(probe.payload) == probe.before, (
        "the request mapping is not byte-identical after the middleware call"
    )
    assert probe.result is not None, "the middleware returned no rewrite, so the turn did not route"
    assert set(probe.result) == {"request"}
    rewritten = probe.result["request"]
    assert rewritten["model"] == probe.payload["model"]
    assert rewritten["tools"] == probe.payload["tools"]
    assert rewritten["unrelated_key"] == probe.payload["unrelated_key"]
    assert rewritten["messages"][-1]["role"] == "system"
    assert rewritten["messages"][-1]["content"].startswith("Skills routed for this request")

    assert probe.shortlist_raised is None, (
        f"the library open path raised on this store input: {probe.shortlist_raised!r}"
    )
    assert probe.rows, "the turn shortlisted no row at all"
    assert any(record.semantic_rank is not None for record in probe.records), (
        "the semantic channel returned no rank"
    )
    assert any(record.bm25_rank is not None for record in probe.records), (
        "the BM25 channel returned no rank"
    )
    assert all(record.association_rank is None for record in probe.records), (
        "the bad store contributed an association rank"
    )


def test_probe_missing_store_does_not_raise_and_keeps_the_request_byte_identical(
    tmp_path, plugin_module, endpoints, caplog
):
    """Missing file: raw read-only open raises, the middleware's open path skips it, no crash."""
    store_path = tmp_path / "associations.db"
    assert not store_path.exists()
    raw = _raw_open(store_path)
    assert isinstance(raw, associations.AssociationStoreError), raw
    with caplog.at_level(logging.WARNING):
        probe = _probe_turn(store_path, tmp_path, plugin_module, endpoints)
    warnings = _middleware_store_warnings(caplog, plugin_module)
    assert len(warnings) == 1, "the middleware never reported the configured missing store"
    assert str(store_path) in warnings[0].getMessage()
    _assert_routed_without_crashing(probe)


def test_probe_zero_byte_store_does_not_raise_and_keeps_the_request_byte_identical(
    tmp_path, plugin_module, endpoints, caplog
):
    """Zero-byte file: raw read-only open raises, the middleware's open path catches it, no crash."""
    store_path = tmp_path / "associations.db"
    store_path.write_bytes(b"")
    raw = _raw_open(store_path)
    assert isinstance(raw, associations.AssociationStoreError), raw
    with caplog.at_level(logging.WARNING):
        probe = _probe_turn(store_path, tmp_path, plugin_module, endpoints)
    warnings = _middleware_store_warnings(caplog, plugin_module)
    assert len(warnings) == 1, "the middleware never reported the configured zero-byte store"
    assert str(store_path) in warnings[0].getMessage()
    _assert_routed_without_crashing(probe)


def test_probe_tables_but_no_rows_store_does_not_raise_and_keeps_the_request_byte_identical(
    tmp_path, plugin_module, endpoints, caplog
):
    """Valid empty store: raw read-only open succeeds, read returns nothing, no crash, no warning."""
    store_path = tmp_path / "associations.db"
    store = associations.open_store(str(store_path))
    store.close()
    raw = _raw_open(store_path)
    assert raw is None, raw
    with caplog.at_level(logging.WARNING):
        probe = _probe_turn(store_path, tmp_path, plugin_module, endpoints)
    assert _middleware_store_warnings(caplog, plugin_module) == [], (
        "a valid empty store opened with a warning, though the current code opens it silently"
    )
    _assert_routed_without_crashing(probe)


def test_probe_directory_in_place_store_does_not_raise_and_keeps_the_request_byte_identical(
    tmp_path, plugin_module, endpoints, caplog
):
    """Directory in place: raw read-only open raises, the middleware's open path catches it, no crash."""
    store_path = tmp_path / "associations.db"
    store_path.mkdir()
    raw = _raw_open(store_path)
    assert isinstance(raw, associations.AssociationStoreError), raw
    with caplog.at_level(logging.WARNING):
        probe = _probe_turn(store_path, tmp_path, plugin_module, endpoints)
    warnings = _middleware_store_warnings(caplog, plugin_module)
    assert len(warnings) == 1, "the middleware never reported the configured directory store"
    assert str(store_path) in warnings[0].getMessage()
    _assert_routed_without_crashing(probe)
