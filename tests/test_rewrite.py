"""Contract C9: the composed rewrite, strip, shortlist, rerank, select, inject, record, log.

Every test drives :func:`hermes_skill_router.router.rewrite` with a fake embedder and a fake
reranker, so the suite needs no endpoint, no model and no network. What is asserted is the shape
the middleware contract depends on: one appended system message on success, ``None`` on every
failure (which keeps the catalog), only the ``request`` key in the result, a display record that
carries selected suggestion names only, and one decision line per completed pass carrying the
query, the scored shortlist, the cutoff, the suggestions and the stage timings.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import inject as inject_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router import search as search_module  # noqa: E402
from hermes_skill_router.config import DEFAULTS, Config, load  # noqa: E402
from hermes_skill_router.inject import (DROPPED_MESSAGE, HEADER, ONE_SHOT_HEADER, PINNED)  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402

# ---------------------------------------------------------------------------
# A system message in the shape agent/prompt_builder.py emits, copied from the C3 test
# fixture so the strip is exercised against the real section, not a summary of it.
# ---------------------------------------------------------------------------

NORMAL_PROSE = (
    "## Skills\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant to your "
    "task, you MUST load it with skill_view(name) and follow its instructions.\n"
)

ONESHOT_PROSE = (
    "## Skills\n"
    "Scan the skills below and load one with skill_view(name) only when it carries domain knowledge "
    "you lack for THIS task (an API, a tool's commands, a project's conventions). Do not load "
    "general process skills (testing, debugging, review methodology) for work you already know how "
    "to do, and do not create or edit skills: this is a one-shot run with no later session to reuse "
    "them.\n"
)

GUIDANCE = "Only proceed without loading a skill if genuinely none are relevant to the task."

STABLE = "You are Hermes Agent.\n\nBe helpful."
MEMORY = "## Memory\n\nNothing yet."

ROWS = (
    ("alpha", "Alpha skill"),
    ("beta", "Beta skill"),
    ("gamma", "Gamma skill"),
    ("delta", "Delta skill"),
)

QUERY = "draft a git commit message for a bug fix"

# Index vectors: alpha points at x, beta at y, gamma at z, delta at x+y.
VECTORS = {
    "alpha\nAlpha skill": [1.0, 0.0, 0.0],
    "beta\nBeta skill": [0.0, 1.0, 0.0],
    "gamma\nGamma skill": [0.0, 0.0, 1.0],
    "delta\nDelta skill": [0.7071067811865476, 0.7071067811865476, 0.0],
}

# The query points at beta, so a shortlist of one row is beta.
QUERY_VECTOR = [0.0, 1.0, 0.0]


def _block(rows=ROWS, demoted=()):
    lines = []
    if demoted:
        lines.append(f"  creative [names only]: {', '.join(sorted(demoted))}")
    lines.append("  general:")
    lines.extend(f"    - {name}: {description}" for name, description in rows)
    return "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"


def _content(rows=ROWS, demoted=(), variant="normal"):
    if variant == "oneshot":
        # The single-query prompt has no trailing guidance paragraph (agent/prompt_builder.py).
        section = ONESHOT_PROSE + "\n" + _block(rows, demoted)
    else:
        section = NORMAL_PROSE + "\n" + _block(rows, demoted) + "\n\n" + GUIDANCE
    return STABLE + "\n\n" + section + "\n\n" + MEMORY


def _payload(rows=ROWS, demoted=(), user=QUERY, extra=(), variant="normal"):
    messages = [{"role": "system", "content": _content(rows, demoted, variant)}]
    if user is not None:
        messages.append({"role": "user", "content": user})
    messages.extend(extra)
    return {
        "model": "test-model",
        "tools": [{"type": "function", "function": {"name": "skill_view"}}],
        "unrelated_key": {"kept": True},
        "messages": messages,
    }


def make_config(tmp_path, **overrides):
    settings = {
        "cachePath": "",
        "suggestionsPath": str(tmp_path / "suggestions.json"),
    }
    settings.update(overrides)
    return load(settings)


class FakeEmbedder:
    """Answers every text from a table and records each call, so the pipeline's inputs are visible."""

    def __init__(self, table=None, default=(0.0, 1.0, 0.0), result="computed"):
        self.table = dict(VECTORS if table is None else table)
        self.default = list(default)
        self.result = result
        self.calls = []

    def __call__(self, cfg: Config, texts):
        self.calls.append(list(texts))
        if self.result != "computed":
            return self.result
        vectors = []
        for text in texts:
            if text in self.table:
                vectors.append(list(self.table[text]))
            elif text == QUERY:
                vectors.append(list(QUERY_VECTOR))
            else:
                vectors.append(list(self.default))
        return vectors

    @property
    def texts(self):
        return [text for call in self.calls for text in call]


class FakeReranker:
    """Returns fixed scores, or ``None`` when constructed with ``result=None`` and no scores."""

    def __init__(self, scores=(0.9, 0.2, 0.05, 0.01), result="scores"):
        self.scores = list(scores)
        self.result = result
        self.calls = []

    def __call__(self, cfg: Config, query, documents):
        self.calls.append((query, list(documents)))
        if self.result is None:
            return None
        return self.scores[: len(documents)]


class RecordingContext:
    """Fake plugin context: records the surface ``register`` adds and answers no settings."""

    def __init__(self):
        self.tools = []
        self.hooks = []
        self.middleware = []

    def register_tool(self, name, *args, **kwargs):
        self.tools.append(name)

    def register_hook(self, hook_name, callback):
        self.hooks.append((hook_name, callback))

    def register_middleware(self, kind, callback):
        self.middleware.append((kind, callback))

    def get_config(self, key, default=None):
        return default


def _load_plugin_module():
    """Load the repository's plugin entry point as a bare module, as a test collector does.

    Without a package, the entry point takes its own ``else`` branch and imports the top-level
    ``hermes_skill_router`` modules, so the hook callback it registers expires the same record this
    module's ``rewrite`` reads.
    """
    name = "hermes_skill_router_plugin_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A path named ``__init__.py`` makes the spec a package importer, so clear the package name the
    # entry point checks before executing it: this is the bare-module import its ``else`` branch
    # documents, and it shares the top-level ``hermes_skill_router`` modules with this test.
    module.__package__ = ""
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plugin_module():
    """A freshly loaded plugin entry point, so each test registers into its own fake context."""
    return _load_plugin_module()


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """A fresh logger set, index memo and dump record per test, so one test never leaks into another.

    The dump record is keyed by session_id and many tests here use the empty id, so without this
    clear one test's dump poisons every later test in the process. The host path lookup is replaced
    with a resolver that answers one absolute path per name, the way the real ``skill_view``
    resolver does on a host that carries every skill. A test that needs a refused or raising lookup
    patches ``router_module.skill_path`` itself.
    """
    search_module.clear_dumps()
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})
    monkeypatch.setattr(inject_module, "_PINNED_WARNED", set())
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")


def record_for(cfg):
    """Read the display record the rewrite wrote, or ``None`` when there is no file."""
    path = Path(cfg.suggestions_path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def block_of(result):
    """Return the injected block from a rewritten payload."""
    return result["request"]["messages"][-1]["content"]


def make_index(pairs=None, vectors=None):
    pairs = list(ROWS if pairs is None else pairs)
    return Index(
        names=[name for name, _description in pairs],
        descriptions=[description for _name, description in pairs],
        vectors=[list(vector) for vector in (vectors or [VECTORS[name + "\n" + description] for name, description in pairs])],
        dim=3,
        fingerprint="test-index",
    )


# ---------------------------------------------------------------------------
# Success: the section is gone, one block is appended, every top-level key survives.
# ---------------------------------------------------------------------------


def test_a_routed_request_loses_the_catalog_and_gets_one_appended_block(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()

    result = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    rewritten = result["request"]
    system = rewritten["messages"][0]["content"]
    assert "## Skills" not in system
    assert "<available_skills>" not in system
    assert "You MUST load it with skill_view" not in system
    assert GUIDANCE not in system
    assert block_of(result).startswith(HEADER)
    assert rewritten["messages"][-1]["role"] == "system"
    assert len(rewritten["messages"]) == len(payload["messages"]) + 1


def test_the_injection_guard_recognises_either_header_from_its_first_line():
    marker = ONE_SHOT_HEADER.split("\n", 1)[0]

    assert router_module.INJECTED_HEADER == marker
    assert HEADER.split("\n", 1)[0] == marker
    assert ONE_SHOT_HEADER.startswith(marker + "\n")
    assert HEADER.startswith(marker + "\n")


def test_an_already_injected_one_shot_block_is_left_alone(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload(variant="oneshot", extra=({"role": "system", "content": ONE_SHOT_HEADER},))

    assert rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert not Path(cfg.suggestions_path).exists()


def test_a_one_shot_payload_gets_the_one_shot_header(tmp_path):
    payload = _payload(variant="oneshot")

    result = rewrite(payload, cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    system = result["request"]["messages"][0]["content"]
    assert "## Skills" not in system
    assert "Do not load general process skills" not in system
    block = block_of(result)
    assert block.startswith(ONE_SHOT_HEADER)
    assert not block.startswith(HEADER)
    # The restriction the strip removed is carried by the block; the normal invitation is not.
    assert "Do not load general process skills (testing, debugging, review methodology)" in block
    assert "do not create or edit skills" in block
    assert "even partially relevant" not in block


def test_a_normal_payload_keeps_the_normal_header(tmp_path):
    result = rewrite(_payload(), cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker())

    block = block_of(result)
    assert block.startswith(HEADER)
    assert not block.startswith(ONE_SHOT_HEADER)
    assert "even partially relevant" in block


def test_the_result_has_only_the_request_key_the_middleware_replaces(tmp_path):
    result = rewrite(_payload(), cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker())

    assert set(result) == {"request"}


def test_model_tools_and_an_unrelated_top_level_key_survive(tmp_path):
    payload = _payload()

    rewritten = rewrite(
        payload, cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker()
    )["request"]

    assert rewritten["model"] == "test-model"
    assert rewritten["tools"] == payload["tools"]
    assert rewritten["unrelated_key"] == {"kept": True}


def test_the_selected_suggestion_is_rendered_with_its_description_and_read_path(tmp_path):
    result = rewrite(
        _payload(), cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker()
    )

    lines = block_of(result).split("\n")
    assert "- beta: Beta skill" in lines
    assert "  path: /skills/beta/SKILL.md" in lines
    # A probability must never reach the block.
    assert "0.9" not in block_of(result)


def test_the_index_build_is_over_routed_rows_and_the_query_is_embedded_after_it(tmp_path):
    embedder = FakeEmbedder()
    result = rewrite(_payload(), cfg=make_config(tmp_path), embed=embedder, rerank=FakeReranker())

    assert result is not None
    # One build call carrying every routed document, then one call carrying only the user text.
    assert embedder.calls[0] == [name + "\n" + description for name, description in ROWS]
    assert embedder.calls[-1] == [QUERY]


def test_the_payload_the_caller_passed_is_never_mutated(tmp_path):
    payload = _payload()
    before = copy.deepcopy(payload)

    rewrite(payload, cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker())

    assert payload == before


def test_the_same_payload_and_index_rewrite_deterministically(tmp_path):
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()

    first = rewrite(
        payload, cfg=cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker()
    )
    second = rewrite(
        payload, cfg=cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker()
    )

    assert first == second


def test_a_supplied_index_is_not_rebuilt(tmp_path):
    embedder = FakeEmbedder()

    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        index=make_index(),
        embed=embedder,
        rerank=FakeReranker(),
    )

    assert result is not None
    assert embedder.calls == [[QUERY]], "a supplied index costs only the query embedding"


def test_the_record_announcement_fires_once_with_the_turn_id(tmp_path):
    seen = []

    rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        turn_id="turn-7",
        notify=seen.append,
    )

    assert seen == ["turn-7"]


# ---------------------------------------------------------------------------
# The shortlist is cosine over the index, capped by shortlistSize before rerank.
# ---------------------------------------------------------------------------


def test_the_shortlist_caps_what_reaches_the_reranker_and_orders_by_cosine(tmp_path):
    reranker = FakeReranker(scores=(0.9,))

    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path, shortlistSize=1),
        embed=FakeEmbedder(),
        rerank=reranker,
    )

    query, documents = reranker.calls[0]
    assert query == QUERY
    assert len(documents) == 1
    assert documents[0].endswith("beta - Beta skill")
    assert documents[0].startswith(load({}).document_prefix)
    assert "- beta: Beta skill" in block_of(result)


def test_the_rerank_document_is_the_configured_prefix_plus_name_dash_description(tmp_path):
    reranker = FakeReranker(scores=(0.9,))

    rewrite(
        _payload(),
        cfg=make_config(tmp_path, shortlistSize=1, documentPrefix="PREFIX "),
        embed=FakeEmbedder(),
        rerank=reranker,
    )

    assert reranker.calls[0][1] == ["PREFIX beta - Beta skill"]


# ---------------------------------------------------------------------------
# The eager/deferred partition: kept rows are listed, never suggested or recorded.
# ---------------------------------------------------------------------------


def test_an_eager_row_is_pinned_in_the_opening_block_and_never_recorded(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"])
    reranker = FakeReranker(scores=(0.9, 0.9, 0.9, 0.9))

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=reranker)

    block = block_of(result)
    assert PINNED in block.split("\n")
    assert "- alpha: Alpha skill" in block.split("\n")
    assert "  path: /skills/alpha/SKILL.md" in block.split("\n")
    record = record_for(cfg)
    assert record["status"] == "routed"
    assert "alpha" not in record["names"]
    # The eager row never reaches the reranker, because kept rows never enter the index.
    assert all("alpha - Alpha skill" not in document for document in reranker.calls[0][1])


def test_a_skill_outside_a_non_empty_deferred_list_stays_listed(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"], deferredSkills=["beta"])
    reranker = FakeReranker(scores=(0.9,))

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=reranker)

    block = block_of(result)
    # beta is the only routed row, so it is the only rerank document and suggestion.
    assert [document.split("Skill: ")[-1] for document in reranker.calls[0][1]] == [
        "beta - Beta skill"
    ]
    assert "- beta: Beta skill" in block.split("\n")
    # gamma and delta are outside the deferred list, so they stay listed, not routed.
    assert "- gamma: Gamma skill" in block.split("\n")
    assert "- delta: Delta skill" in block.split("\n")
    record = record_for(cfg)
    assert record["names"] == ["beta"]
    for kept in ("alpha", "gamma", "delta"):
        assert kept not in record["names"]


def test_a_demoted_names_only_row_is_listed_but_never_indexed_or_recorded(tmp_path):
    cfg = make_config(tmp_path, deferredSkills=["alpha", "zeta"])
    embedder = FakeEmbedder()
    reranker = FakeReranker(scores=(0.9, 0.9, 0.9, 0.9))

    result = rewrite(_payload(demoted=("zeta",)), cfg=cfg, embed=embedder, rerank=reranker)

    block = block_of(result)
    assert "- zeta" in block.split("\n")
    # The demoted name is kept even when deferredSkills names it, and never costs a vector.
    assert not any("zeta" in text for text in embedder.calls[0])
    assert not any("zeta" in document for document in reranker.calls[0][1])
    record = record_for(cfg)
    assert "zeta" not in record["names"]
    assert record["status"] == "routed"


def test_max_suggestions_zero_lists_every_suggestion_and_the_pinned_row(tmp_path):
    cfg = make_config(tmp_path, maxSuggestions=0, eagerSkills=["alpha"])

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    assert "## Skills" not in result["request"]["messages"][0]["content"]
    block = block_of(result)
    for name in ("beta", "delta", "gamma"):
        assert any(line.startswith(f"- {name}: ") for line in block.split("\n"))
    assert PINNED in block.split("\n")
    assert "- alpha: Alpha skill" in block.split("\n")
    record = record_for(cfg)
    assert record["status"] == "routed"
    assert record["names"] == ["beta", "delta", "gamma"]


# ---------------------------------------------------------------------------
# Suppression and the opening block: both come from the payload's own messages.
# ---------------------------------------------------------------------------


def _second_turn(payload, block):
    """A later request in the same window: the earlier block, an answer, then a new user message."""
    return {
        **payload,
        "messages": [
            *payload["messages"],
            {"role": "system", "content": block},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next request"},
        ],
    }


def test_a_second_request_carrying_the_block_does_not_repeat_its_names(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()
    first = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())
    assert first is not None
    second = _second_turn(payload, block_of(first))
    before = copy.deepcopy(second)

    result = rewrite(second, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is None, "every candidate the earlier block listed was suggested again"
    assert second == before, "a suppression-emptied selection must leave the payload unchanged"
    assert record_for(cfg)["status"] == "inert"


def test_a_request_without_the_block_can_suggest_the_names_again(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()

    first = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())
    second = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert first is not None
    assert second == first, "module state, not the messages, decided the suppressed set"


def test_user_text_that_merely_quotes_the_block_does_not_suppress_its_names(tmp_path):
    cfg = make_config(tmp_path)
    quoted = (
        "please quote this:\n"
        + HEADER
        + "\n\nSuggestions:\n- beta: Beta skill\n  path: /skills/beta/SKILL.md"
    )
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "user", "content": quoted},
        ],
    }

    result = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    assert "- beta: Beta skill" in block_of(result).split("\n"), (
        "user text that opens with something else seeded the suppressed set"
    )


def test_pinned_rows_return_after_a_compaction_removes_the_block(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"])
    payload = _payload()

    opening = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())
    assert opening is not None
    assert PINNED in block_of(opening).split("\n")

    prior = HEADER + "\n\nSuggestions:\n- beta: Beta skill\n  path: /skills/beta/SKILL.md"
    later = rewrite(
        _second_turn(payload, prior), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()
    )
    assert later is not None
    assert PINNED not in block_of(later).split("\n"), "a later turn in the window listed pinned rows"
    assert "- alpha: Alpha skill" not in block_of(later).split("\n")

    # The compaction removed the earlier block, so the next window opens with the pinned rows again.
    reopened = rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())
    assert reopened is not None
    assert PINNED in block_of(reopened).split("\n")
    assert "- alpha: Alpha skill" in block_of(reopened).split("\n")


# ---------------------------------------------------------------------------
# Read paths: the caller resolves every candidate, and an unresolved row never spends a slot.
# ---------------------------------------------------------------------------


def test_a_pathless_top_row_hands_its_slot_to_the_next_admissible_row(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, maxSuggestions=2)
    monkeypatch.setattr(
        router_module,
        "skill_path",
        lambda name: None if name == "beta" else f"/skills/{name}/SKILL.md",
    )

    result = rewrite(
        _payload(),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9, 0.8, 0.7, 0.6)),
    )

    assert result is not None
    rows = [line for line in block_of(result).split("\n") if line.startswith("- ")]
    assert rows == ["- delta: Delta skill", "- alpha: Alpha skill"], (
        "a candidate whose path did not resolve consumed a cap slot"
    )
    assert "beta" not in block_of(result)
    assert record_for(cfg)["names"] == ["delta", "alpha"]


def test_a_raising_lookup_costs_that_row_and_not_the_block(tmp_path, monkeypatch):
    def resolve(name):
        if name == "beta":
            raise RuntimeError("the host lookup exploded")
        return f"/skills/{name}/SKILL.md"

    monkeypatch.setattr(router_module, "skill_path", resolve)

    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9, 0.2, 0.05, 0.01)),
    )

    assert result is not None
    rows = [line for line in block_of(result).split("\n") if line.startswith("- ")]
    assert rows == ["- delta: Delta skill", "- alpha: Alpha skill", "- gamma: Gamma skill"]


def test_every_rendered_suggestion_carries_its_absolute_read_path(tmp_path):
    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(scores=(0.9, 0.2, 0.05, 0.01)),
    )

    lines = block_of(result).split("\n")
    rendered = [line for line in lines if line.startswith("- ")]
    paths = [line for line in lines if line.startswith("  path: ")]
    assert len(rendered) == len(paths) == 4
    for name in ("beta", "delta", "alpha", "gamma"):
        assert f"  path: /skills/{name}/SKILL.md" in paths


# ---------------------------------------------------------------------------
# The partial-view guard: a refused catalog routes from the saved list.
# ---------------------------------------------------------------------------


def seed_saved_index(cfg, index=None):
    """Write a saved index the way a previous process' cache write would have."""
    library_module._write_cache(cfg, index or make_index())
    return Path(cfg.cache_path)


def test_a_refused_catalog_suggests_from_the_saved_list_with_a_resolved_path(tmp_path, caplog):
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    cache_file = seed_saved_index(cfg)
    before = cache_file.read_bytes()
    embedder = FakeEmbedder()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        result = rewrite(
            _payload(rows=(("alpha", "Alpha skill"),)),
            cfg=cfg,
            embed=embedder,
            rerank=FakeReranker(),
        )

    assert result is not None
    lines = block_of(result).split("\n")
    assert "- gamma: Gamma skill" in lines, "a saved row outside the arrived catalog was not suggested"
    assert "  path: /skills/gamma/SKILL.md" in lines
    assert "- beta: Beta skill" in lines
    assert embedder.calls == [[QUERY]], "the refused catalog was embedded"
    assert cache_file.read_bytes() == before, "the refused catalog rewrote the saved index"
    assert "gamma" in record_for(cfg)["names"]
    assert any("less than half" in message for message in caplog.messages)


def test_a_saved_row_whose_path_does_not_resolve_is_not_emitted(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, cachePath=str(tmp_path / "index.json"))
    seed_saved_index(cfg)
    monkeypatch.setattr(
        router_module,
        "skill_path",
        lambda name: None if name == "gamma" else f"/skills/{name}/SKILL.md",
    )

    result = rewrite(
        _payload(rows=(("alpha", "Alpha skill"),)),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
    )

    assert result is not None
    assert "gamma" not in block_of(result)
    assert "- beta: Beta skill" in block_of(result).split("\n")


def test_a_turn_with_no_routed_rows_stays_unchanged_against_a_larger_saved_cache(
    tmp_path, caplog
):
    cfg = make_config(
        tmp_path,
        cachePath=str(tmp_path / "index.json"),
        eagerSkills=[name for name, _description in ROWS],
    )
    cache_file = seed_saved_index(cfg)
    before = cache_file.read_bytes()
    payload = _payload()
    snapshot = copy.deepcopy(payload)
    embedder = FakeEmbedder()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.library"):
        result = rewrite(payload, cfg=cfg, embed=embedder, rerank=FakeReranker())

    assert result is None
    assert payload == snapshot
    assert embedder.calls == []
    assert cache_file.read_bytes() == before
    assert not any("less than half" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# The record: selected names only, capped and namespaced by turn.
# ---------------------------------------------------------------------------


def test_the_record_holds_the_selected_suggestion_names_only(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"])
    reranker = FakeReranker(scores=(0.9, 0.5, 0.002, 0.002))

    rewrite(
        _payload(),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=reranker,
        session_id="session-1",
        turn_id="turn-1",
    )

    record = record_for(cfg)
    assert record["schema"] == 2
    assert record["origin"] == "turn", "the turn call site passes no origin, so the writer defaults it"
    assert record["status"] == "routed"
    assert record["names"] == ["beta", "delta"]
    assert record["session_id"] == "session-1"
    assert record["turn_id"] == "turn-1"
    assert record["at"].endswith("Z")


def test_the_record_never_holds_the_users_text(tmp_path):
    cfg = make_config(tmp_path)

    rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert QUERY not in Path(cfg.suggestions_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Inert paths: every failure returns None and records an empty routed/inert decision.
# ---------------------------------------------------------------------------


def test_an_unreachable_embedder_keeps_the_request_and_records_inert(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()
    before = copy.deepcopy(payload)

    result = rewrite(
        payload, cfg=cfg, index=make_index(), embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert result is None
    assert payload == before
    record = record_for(cfg)
    assert record["status"] == "inert"
    assert record["names"] == []


def test_an_unreachable_reranker_keeps_the_request_and_records_inert(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()

    result = rewrite(
        payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker(result=None)
    )

    assert result is None
    record = record_for(cfg)
    assert record["status"] == "inert"
    assert record["names"] == []


def test_an_unbuildable_index_keeps_the_request_and_records_inert(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        result = rewrite(
            _payload(), cfg=cfg, embed=FakeEmbedder(result=None), rerank=FakeReranker()
        )

    assert result is None
    assert record_for(cfg)["status"] == "inert"
    assert any("index build failed" in message for message in caplog.messages)


def test_no_user_message_returns_none_and_records_inert(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        result = rewrite(
            _payload(user=None), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()
        )

    assert result is None
    assert record_for(cfg)["status"] == "inert"
    assert any("no user text" in message for message in caplog.messages)


def test_an_empty_user_message_returns_none(tmp_path):
    cfg = make_config(tmp_path)

    assert rewrite(_payload(user="   "), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert record_for(cfg)["status"] == "inert"


def test_an_empty_block_returns_none_and_stays_inert(tmp_path):
    cfg = make_config(tmp_path, maxSuggestions=0, cutoff=0.99)

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is None
    record = record_for(cfg)
    assert record["status"] == "inert"
    assert record["names"] == []


def test_a_partition_with_no_routed_row_returns_none(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha", "beta", "gamma", "delta"])

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is None
    assert record_for(cfg)["status"] == "inert"


def test_an_all_eager_config_that_selects_nothing_does_not_re_dump(tmp_path):
    """P1-2, empty-selection branch: the first pass dumps once, the second strips.

    Every row is eager, so no pass can select anything. Before the fix both passes returned
    None and the whole catalog went out twice. Now the first pass is the one allowed dump
    and the second pass in the same window strips the catalog instead.
    """
    cfg = make_config(tmp_path, eagerSkills=["alpha", "beta", "gamma", "delta"])
    index = make_index()
    payload = _payload()

    first = rewrite(payload, cfg=cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker(), session_id="all-eager-session")
    assert first is None, "the first empty pass is the one allowed dump"

    second = rewrite(payload, cfg=cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker(), session_id="all-eager-session")
    assert second is not None, "the second empty pass re-dumped the catalog"
    system = second["request"]["messages"][0]["content"]
    assert "## Skills" not in system
    assert block_of(second) == DROPPED_MESSAGE


def test_an_empty_block_does_not_consume_the_window_dump(tmp_path):
    """P1-2, empty-block branch: a below-cutoff pass is healthy, not a failure.

    A pass that selects nothing because every score fell below the cutoff returns None
    without setting a dump record, so the next working pass routes normally instead of
    stripping the catalog as if a dump had happened.
    """
    index = make_index()
    payload = _payload()

    first = rewrite(
        payload,
        cfg=make_config(tmp_path, cutoff=0.99),
        index=index,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
    )
    assert first is None

    good = rewrite(
        payload, cfg=make_config(tmp_path), index=index, embed=FakeEmbedder(), rerank=FakeReranker()
    )
    assert good is not None
    assert "- beta: Beta skill" in block_of(good)


def test_a_user_message_starting_with_the_marker_does_not_silence_routing(tmp_path):
    """P3-5: only a system message can act as proof that the catalog already went out."""
    payload = _payload()
    crafted = {
        **payload,
        "messages": [
            *_all_eager_payload_messages(payload),
            {"role": "user", "content": DROPPED_MESSAGE + "\nplease route this"},
        ],
    }

    result = rewrite(
        crafted,
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="crafted-session",
    )

    assert result is not None
    assert "- beta: Beta skill" in block_of(result)


def _all_eager_payload_messages(payload):
    return payload["messages"]


def test_a_list_content_system_marker_is_recognised(tmp_path):
    """P2-4: a list-shaped system message carrying the marker still counts."""
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()
    payload_list = {
        **payload,
        "messages": [
            payload["messages"][0],
            {"role": "user", "content": payload["messages"][1]["content"]},
            {"role": "system", "content": [{"type": "text", "text": DROPPED_MESSAGE}]},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next request"},
        ],
    }

    result = rewrite(
        payload_list, cfg=cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert result is not None, "a list-shaped marker was not recognised"
    system = result["request"]["messages"][0]["content"]
    assert "## Skills" not in system


# ---------------------------------------------------------------------------
# No rewrite at all: no record, and the payload is never touched.
# ---------------------------------------------------------------------------


def test_a_payload_without_the_catalog_is_untouched_and_not_recorded(tmp_path):
    cfg = make_config(tmp_path)
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": QUERY},
        ],
    }

    assert rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert not Path(cfg.suggestions_path).exists()


def test_a_last_message_that_already_carries_the_block_is_left_alone(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload(extra=({"role": "system", "content": HEADER + "\nrest"},))

    assert rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert not Path(cfg.suggestions_path).exists()


def test_a_non_chat_completions_mode_returns_none_and_does_not_record(tmp_path):
    cfg = make_config(tmp_path)
    payload = _payload()

    assert rewrite(payload, cfg=cfg, api_mode="codex_responses") is None
    assert rewrite(payload, cfg=cfg, api_mode="anthropic_messages") is None
    assert not Path(cfg.suggestions_path).exists()


def test_a_payload_carrying_its_own_api_mode_wins(tmp_path):
    cfg = make_config(tmp_path)
    payload = {**_payload(), "api_mode": "codex_responses"}

    assert rewrite(payload, cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert not Path(cfg.suggestions_path).exists()


def test_a_disabled_plugin_returns_none_and_does_not_record(tmp_path):
    cfg = make_config(tmp_path, enabled=False)

    assert rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker()) is None
    assert not Path(cfg.suggestions_path).exists()


def test_an_unsupported_api_mode_logs_once_per_process(tmp_path, caplog):
    cfg = make_config(tmp_path)
    payload = _payload()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        rewrite(payload, cfg=cfg, api_mode="codex_responses")
        rewrite(payload, cfg=cfg, api_mode="codex_responses")

    reasons = [message for message in caplog.messages if "unsupported api mode" in message]
    assert len(reasons) == 1


# ---------------------------------------------------------------------------
# The newest typed message heads the query; the capped preceding context follows, oldest first.
# ---------------------------------------------------------------------------


def test_the_newest_typed_message_heads_the_query_and_the_earlier_one_follows(tmp_path):
    payload = _payload(
        extra=(
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "second request"},
        )
    )
    embedder = FakeEmbedder()
    reranker = FakeReranker(scores=(0.9, 0.2, 0.05, 0.01))

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    query = f"second request Context: {QUERY}"
    assert embedder.calls[-1] == [query], "the embedder does not receive the composed query"
    assert reranker.calls[0][0] == query, "the reranker does not receive the composed query"


def test_the_preceding_typed_messages_join_oldest_first(tmp_path):
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "user", "content": "first older"},
            {"role": "assistant", "content": "an answer"},
            {"role": "user", "content": "second older"},
            {"role": "user", "content": "newest request"},
        ],
    }
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    query = "newest request Context: first older second older"
    assert embedder.calls[-1] == [query]
    assert reranker.calls[0][0] == query


def test_only_typed_user_messages_join_the_context(tmp_path):
    block = HEADER + "\n\nSuggestions:\n- beta: Beta skill\n  path: /skills/beta/SKILL.md"
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "system", "content": block},
            {"role": "assistant", "content": "an assistant message"},
            {"role": "user", "content": "typed earlier"},
            {"role": "system", "content": "a later system message"},
            {"role": "user", "content": "newest request"},
        ],
    }
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    query = "newest request Context: typed earlier"
    assert embedder.calls[-1] == [query]
    assert reranker.calls[0][0] == query


def test_the_preceding_context_keeps_only_whole_messages_under_the_cap(tmp_path):
    cap = router_module.RECENT_CONTEXT_CHARS
    oldest = "a" * 3000
    crossing = "b" * (cap - 1000)
    newest_context = "c" * 2000
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "user", "content": oldest},
            {"role": "user", "content": crossing},
            {"role": "user", "content": newest_context},
            {"role": "user", "content": "request"},
        ],
    }
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    query = "request Context: " + newest_context
    assert embedder.calls[-1] == [query], (
        "the scan did not keep whole messages from the newest backwards and stop at the crossing"
    )
    assert reranker.calls[0][0] == query
    assert len(query) < len(oldest) + len(crossing) + len(newest_context)


def test_an_oversize_newest_message_is_passed_in_full(tmp_path):
    cap = router_module.RECENT_CONTEXT_CHARS
    newest = "q" * (cap + 500)
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "user", "content": "small context"},
            {"role": "user", "content": newest},
        ],
    }
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    query = newest + " Context: small context"
    assert embedder.calls[-1] == [query], "the newest message was cut by the context cap"
    assert reranker.calls[0][0] == query


def test_without_a_preceding_typed_message_the_query_is_exactly_the_newest_message(tmp_path):
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content()},
            {"role": "assistant", "content": "an answer with no typed message before it"},
            {"role": "user", "content": "the newest request"},
        ],
    }
    embedder = FakeEmbedder()
    reranker = FakeReranker()

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    assert embedder.calls[-1] == ["the newest request"]
    assert reranker.calls[0][0] == "the newest request"


def test_the_recent_context_cap_is_a_module_constant_and_not_a_setting():
    assert router_module.RECENT_CONTEXT_CHARS == 10_000
    assert not any("context" in key.lower() for key in DEFAULTS), (
        "the recent context cap was added as a settings key"
    )


def test_a_content_list_supplies_its_text_parts(tmp_path):
    payload = _payload(
        user=[
            {"type": "text", "text": "first part"},
            {"type": "text", "text": "second part"},
        ]
    )
    reranker = FakeReranker(scores=(0.9, 0.2, 0.05, 0.01))

    rewrite(payload, cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=reranker)

    assert reranker.calls[0][0] == "first part\nsecond part"


# ---------------------------------------------------------------------------
# Failure policy and one log line per distinct reason.
# ---------------------------------------------------------------------------


def test_a_raising_embedder_never_escapes_the_rewrite(tmp_path):
    cfg = make_config(tmp_path)

    def boom(cfg, texts):
        raise RuntimeError("router exploded")

    assert rewrite(_payload(), cfg=cfg, embed=boom, rerank=FakeReranker()) is None
    assert record_for(cfg)["status"] == "inert"


def test_a_raising_reranker_never_escapes_the_rewrite(tmp_path):
    cfg = make_config(tmp_path)

    def boom(cfg, query, documents):
        raise RuntimeError("reranker exploded")

    assert rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=boom) is None
    assert record_for(cfg)["status"] == "inert"


def test_a_raising_display_writer_does_not_change_the_rewrite(tmp_path, monkeypatch):
    cfg = make_config(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("display exploded")

    monkeypatch.setattr(router_module, "record", boom)

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    assert block_of(result).startswith(HEADER)


def test_a_raising_notify_does_not_change_the_rewrite(tmp_path):
    def boom(turn_id):
        raise RuntimeError("no gateway")

    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        notify=boom,
    )

    assert result is not None


def test_an_embed_timeout_logs_its_own_reason_once(tmp_path, caplog):
    cfg = make_config(tmp_path)

    def slow(cfg, texts):
        raise TimeoutError("too slow")

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        rewrite(_payload(), cfg=cfg, index=make_index(), embed=slow, rerank=FakeReranker())
        rewrite(_payload(), cfg=cfg, index=make_index(), embed=slow, rerank=FakeReranker())

    reasons = [message for message in caplog.messages if "embed timeout" in message]
    assert len(reasons) == 1


def test_an_unreachable_embedder_logs_once_per_process(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        rewrite(_payload(), cfg=cfg, index=make_index(), embed=FakeEmbedder(result=None),
                rerank=FakeReranker())
        rewrite(_payload(), cfg=cfg, index=make_index(), embed=FakeEmbedder(result=None),
                rerank=FakeReranker())

    reasons = [message for message in caplog.messages if "embed unreachable" in message]
    assert len(reasons) == 1


def test_an_unreachable_reranker_logs_once_per_process(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.router"):
        rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker(result=None))
        rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker(result=None))

    reasons = [message for message in caplog.messages if "rerank unreachable" in message]
    assert len(reasons) == 1


# ---------------------------------------------------------------------------
# The decision line: one info line per completed pass, none on a failed one.
# ---------------------------------------------------------------------------

DECISION_PREFIX = "skill router: query="

# The hash of ``json.dumps(result, ensure_ascii=False, sort_keys=True)`` for the payload
# ``_payload()`` rewritten with ``FakeEmbedder()``, ``FakeReranker()``, session ``s-1`` and turn
# ``t-1``, with the test's resolver answering ``/skills/<name>/SKILL.md``. It was captured from the
# ticket 02 build, which gave every suggestion its read path and its section marker, so a change to
# the returned payload fails the test.
GOLDEN_PAYLOAD_SHA256 = "700fbbb243841d2a0d6eb7e1172e5ce3a7e1774e74997c4108d678416f46a14a"

# The display record the same call wrote, minus its unstable ``at`` timestamp.
GOLDEN_RECORD = {
    "schema": 2,
    "session_id": "s-1",
    "turn_id": "t-1",
    "status": "routed",
    "origin": "turn",
    "names": ["beta", "delta", "alpha", "gamma"],
    "note": "",
}


def _decision_lines(caplog):
    """Return the decision lines in the captured log."""
    return [message for message in caplog.messages if message.startswith(DECISION_PREFIX)]


def test_a_routed_turn_writes_one_decision_line(tmp_path, caplog):
    cfg = make_config(tmp_path, cutoff=0.1)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    lines = _decision_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith(f'{DECISION_PREFIX}"{QUERY}"'), "the query is JSON-quoted"
    assert "candidates=beta:0.9,delta:0.2,alpha:0.05,gamma:0.01" in line, (
        "every shortlisted candidate is named with the score the selector compared, including the "
        "candidates the cutoff rejected"
    )
    assert "cutoff=0.1" in line
    assert "suggestions=beta:0.9,delta:0.2" in line
    for field in ("embed_ms=", "rerank_ms=", "total_ms="):
        assert field in line, f"the decision line carries no {field}"


def test_each_routed_turn_writes_its_own_decision_line(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())
        rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert len(_decision_lines(caplog)) == 2


def test_the_logged_query_is_json_quoted(tmp_path, caplog):
    query = 'say "hi"\nnow'

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        rewrite(
            _payload(user=query),
            cfg=make_config(tmp_path),
            embed=FakeEmbedder(),
            rerank=FakeReranker(),
        )

    line = _decision_lines(caplog)[0]
    assert 'query="say \\"hi\\"\\nnow"' in line, (
        "the query is not JSON-quoted, so a quote or newline in it would break the line"
    )


def test_the_decision_line_times_the_stages(tmp_path, caplog):
    def slow_embed(cfg, texts):
        time.sleep(0.01)
        return FakeEmbedder()(cfg, texts)

    def slow_rerank(cfg, query, documents):
        time.sleep(0.01)
        return FakeReranker()(cfg, query, documents)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        rewrite(
            _payload(),
            cfg=make_config(tmp_path),
            embed=slow_embed,
            rerank=slow_rerank,
        )

    line = _decision_lines(caplog)[0]
    timings = {
        field: float(line.split(field + "=")[1].split(" ")[0])
        for field in ("embed_ms", "rerank_ms", "total_ms")
    }
    assert timings["embed_ms"] >= 5.0
    assert timings["rerank_ms"] >= 5.0
    assert timings["total_ms"] >= timings["embed_ms"] + timings["rerank_ms"] - 0.2


def test_a_completed_pass_with_no_suggestion_still_writes_its_line(tmp_path, caplog):
    cfg = make_config(tmp_path, cutoff=0.99)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is None, "nothing cleared the cutoff, so the catalog stays in the request"
    line = _decision_lines(caplog)[0]
    assert "candidates=beta:0.9,delta:0.2,alpha:0.05,gamma:0.01" in line
    assert "suggestions=-" in line, "the line says no name cleared the cutoff"


def test_a_failed_rerank_writes_no_decision_line_and_keeps_one_warning(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker(result=None))

    assert result is None
    assert _decision_lines(caplog) == []
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "hermes_skill_router.router"
    ]
    assert len(warnings) == 1
    assert "rerank unreachable" in warnings[0].getMessage()
    assert QUERY not in warnings[0].getMessage(), "a warning carries no user text"


def test_an_embed_timeout_writes_no_decision_line(tmp_path, caplog):
    def slow(cfg, texts):
        raise TimeoutError("too slow")

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(
            _payload(),
            cfg=make_config(tmp_path),
            index=make_index(),
            embed=slow,
            rerank=FakeReranker(),
        )

    assert result is None
    assert _decision_lines(caplog) == []
    assert any("embed timeout" in message for message in caplog.messages)


def test_a_rerank_timeout_writes_no_decision_line(tmp_path, caplog):
    def timed_out(cfg, query, documents):
        raise TimeoutError("too slow")

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(
            _payload(),
            cfg=make_config(tmp_path),
            embed=FakeEmbedder(),
            rerank=timed_out,
        )

    assert result is None
    assert _decision_lines(caplog) == []
    assert any("rerank unreachable" in message for message in caplog.messages)


def test_a_spent_budget_writes_no_decision_line(tmp_path, caplog):
    cfg = make_config(tmp_path, routingTimeoutMs=1000)

    def slow_embed(cfg, texts):
        time.sleep(1.05)
        return FakeEmbedder()(cfg, texts)

    with caplog.at_level(logging.INFO, logger="hermes_skill_router.router"):
        result = rewrite(
            _payload(),
            cfg=cfg,
            index=make_index(),
            embed=slow_embed,
            rerank=FakeReranker(),
        )

    assert result is None
    assert _decision_lines(caplog) == []
    assert any("routing budget 1000 ms exhausted" in message for message in caplog.messages)


def test_the_returned_payload_is_byte_identical_to_the_frozen_build(tmp_path):
    result = rewrite(
        _payload(),
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="s-1",
        turn_id="t-1",
    )

    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert hashlib.sha256(encoded.encode()).hexdigest() == GOLDEN_PAYLOAD_SHA256
    assert list(result["request"]) == ["model", "tools", "unrelated_key", "messages"], (
        "the top-level key order is part of the bytes the provider receives"
    )


def test_the_display_record_is_byte_identical_to_the_pre_change_build(tmp_path):
    cfg = make_config(tmp_path)

    rewrite(
        _payload(),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="s-1",
        turn_id="t-1",
    )

    record = record_for(cfg)
    record.pop("at")
    assert record == GOLDEN_RECORD


# ---------------------------------------------------------------------------
# The catalog reaches the model once per compaction window. A failed pass that leaves
# the catalog in place is a dump: the next turn in the same window must not dump it
# again, and no later turn suggests a name the dump already showed.
# ---------------------------------------------------------------------------


def test_two_failing_passes_dump_the_catalog_once_and_no_name_twice(tmp_path):
    """Invariant 1 and 2, decisive probe: two consecutive requests whose routing fails."""
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()

    first = rewrite(
        payload, cfg=cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )
    assert first is None, "the first failure leaves the request unchanged"

    # The first request went out with the catalog on the wire. The second request carries
    # the marker the first strip appended, below the rebuilt system message.  The marker
    # itself names no skill; the names it stands for came from the catalog the first
    # request exposed.
    payload_marker = {
        **payload,
        "messages": [
            payload["messages"][0],
            {"role": "user", "content": payload["messages"][1]["content"]},
            {"role": "system", "content": DROPPED_MESSAGE},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next request"},
        ],
    }

    second = rewrite(
        payload_marker, cfg=cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert second is not None, "the second request re-dumped the catalog"
    rewritten = second["request"]
    system = rewritten["messages"][0]["content"]
    assert "## Skills" not in system, "the catalog survived into the second request"
    assert "<available_skills>" not in system
    # The first request exposed every skill. The second request exposes no name again
    # through the catalog, and the marker it appends names no skill.
    for name, _description in ROWS:
        assert name not in str(rewritten["messages"]), (
            f"{name} reached the model in the second request"
        )
    assert len(rewritten["messages"]) == len(payload_marker["messages"]) + 1
    assert rewritten["messages"][-1]["role"] == "system"


def test_a_fresh_window_allows_the_catalog_again(tmp_path):
    """Invariant 3, decisive probe: a payload with no prior block and no durable fact opens a new window.

    The first request fails and sets the durable fact for its session. A new session with the
    same payload and no prior block has no fact, so it belongs to a new window: a failed pass
    there leaves the catalog in place again, and a working pass routes normally.
    """
    index = make_index()
    payload = _payload()

    rewrite(payload, cfg=make_config(tmp_path), index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker(), session_id="old-session")
    fresh = rewrite(payload, cfg=make_config(tmp_path), index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker(), session_id="new-session")

    assert fresh is None, "a fresh window must not strip the catalog on a failed pass"
    good = rewrite(payload, cfg=make_config(tmp_path), index=index, embed=FakeEmbedder(), rerank=FakeReranker(), session_id="third-session")
    assert good is not None
    assert "- beta: Beta skill" in block_of(good)


def test_a_dump_never_corrupts_the_request(tmp_path):
    """Invariant 4: a dump keeps every top-level key and appends one clean system message."""
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()
    second_input = {**payload, "messages": [
        payload["messages"][0],
        payload["messages"][1],
        {"role": "system", "content": DROPPED_MESSAGE},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next request"},
    ]}

    result = rewrite(
        second_input, cfg=cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert result is not None
    assert set(result) == {"request"}
    rewritten = result["request"]
    assert rewritten["model"] == "test-model"
    assert rewritten["tools"] == payload["tools"]
    assert rewritten["unrelated_key"] == {"kept": True}
    assert len(rewritten["messages"]) == len(second_input["messages"]) + 1
    last = rewritten["messages"][-1]
    assert last["role"] == "system"
    assert "Skills routed for this request" not in last["content"], (
        "the strip marker must not pose as a routed block"
    )
    assert record_for(cfg)["status"] == "inert"


def test_a_first_failure_without_a_marker_returns_none(tmp_path):
    """Invariant 4: the first failure in a window leaves the request as it arrived."""
    cfg = make_config(tmp_path)
    payload = _payload()
    before = copy.deepcopy(payload)

    result = rewrite(
        payload, cfg=cfg, index=make_index(), embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert result is None
    assert payload == before


def test_two_empty_session_tests_do_not_leak_a_dump(tmp_path):
    """Two tests that both use the empty session id must not share a dump record.

    The first test below fails routing and sets the dump record for the empty session. The
    second test, which runs after it with no prior block in its payload, must route normally.
    The autouse fixture clears the record before each test; this test pins that clearing.
    """
    index = make_index()
    payload = _payload()
    first_cfg = make_config(tmp_path)
    assert rewrite(
        payload, cfg=first_cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    ) is None

    search_module.clear_dumps()

    second_cfg = make_config(tmp_path)
    good = rewrite(
        payload, cfg=second_cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker()
    )
    assert good is not None
    assert "- beta: Beta skill" in block_of(good)


def _window_history(earlier, summary=None):
    """Build the same-session payload shape the window tests drive."""
    system = _payload()["messages"][0]
    query = _payload()["messages"][1]
    messages = [system]
    if summary is not None:
        messages.append({"role": "user", "content": summary})
    for turn in range(earlier):
        messages.append({"role": "assistant", "content": f"answer {turn}"})
        messages.append({"role": "user", "content": f"earlier request {turn}"})
    messages.append(query)
    return {**_payload(), "messages": messages}


SUMMARY = (
    "[CONTEXT COMPACTION REFERENCE ONLY] Earlier turns were compacted into the summary below.\n"
    "## Conversation Summary\n- earlier work happened\n"
    + "".join(f"- detail {turn}\n" for turn in range(250))
    + "respond to the message below, not the summary above ---"
)


def test_a_dump_record_ignores_a_shrinking_request_list(tmp_path):
    """A shorter request list is not a window boundary.

    Hermes may replace the request-local messages for one call without compacting durable history,
    so a shrink must not expire the record: the second failure in the same window strips the
    catalog and appends the marker instead of dumping the catalog again.
    """
    cfg = make_config(tmp_path)
    index = make_index()
    session = "session-shrink"

    def failing(earlier):
        return rewrite(
            _window_history(earlier),
            cfg=cfg,
            index=index,
            embed=FakeEmbedder(result=None),
            rerank=FakeReranker(),
            session_id=session,
        )

    assert failing(4) is None, "the window's first failure leaves the catalog in place"
    assert search_module.dumped(session)

    short = failing(2)
    assert short is not None, "a shorter request list expired the dump record"
    assert short["request"]["messages"][-1]["content"] == DROPPED_MESSAGE
    assert "## Skills" not in short["request"]["messages"][0]["content"]


def test_a_dump_record_expires_when_the_compaction_event_arrives(tmp_path):
    """The host's committed compaction is the window boundary, not the summary text.

    The record decides: the first failure in a window dumps the catalog, a later failure strips it
    and leaves the marker, and the expire the ``post_compaction`` callback performs opens the next
    window. The complete host summary sits in the request history the whole time and changes
    nothing. Two windows open with byte-identical summary text, which a text digest could not tell
    apart.
    """
    cfg = make_config(tmp_path)
    index = make_index()
    session = "session-window"

    def failing(earlier):
        return rewrite(
            _window_history(earlier, SUMMARY),
            cfg=cfg,
            index=index,
            embed=FakeEmbedder(result=None),
            rerank=FakeReranker(),
            session_id=session,
        )

    assert failing(3) is None, "the window's first failure leaves the catalog in place"
    assert search_module.dumped(session)

    second = failing(4)
    assert second is not None, "the same window dumped the catalog a second time"
    assert second["request"]["messages"][-1]["content"] == DROPPED_MESSAGE
    assert "## Skills" not in second["request"]["messages"][0]["content"]

    search_module.expire_dump(session)
    assert failing(1) is None, "the committed compaction did not open a new window"
    assert search_module.dumped(session)
    assert failing(2) is not None, "a second failure in the new window dumped the catalog again"

    search_module.expire_dump(session)
    assert failing(3) is None, "a second byte-identical summary text did not open a second window"
    assert search_module.dumped(session)
    assert failing(4) is not None, "the second new window dumped the catalog again"


def test_a_complete_pasted_summary_changes_nothing(tmp_path):
    """A pasted host summary is message text, not a window boundary.

    The request carries the host's own header, body and closing suffix on every pass, and the
    record alone decides both ways: the first failure dumps the catalog, a further failure with the
    same paste strips it, and a rewrite of the pasted text opens nothing.
    """
    cfg = make_config(tmp_path)
    index = make_index()
    session = "session-pasted"

    def failing(earlier, summary):
        return rewrite(
            _window_history(earlier, summary),
            cfg=cfg,
            index=index,
            embed=FakeEmbedder(result=None),
            rerank=FakeReranker(),
            session_id=session,
        )

    assert failing(3, SUMMARY) is None, "the record, not the paste, decided the first dump"
    assert search_module.dumped(session)

    second = failing(4, SUMMARY)
    assert second is not None, "a complete pasted summary opened a new window"
    assert second["request"]["messages"][-1]["content"] == DROPPED_MESSAGE

    changed = SUMMARY.replace("earlier work happened", "a later paste says something else")
    assert failing(5, changed) is not None, "rewriting the pasted summary text opened a window"


def test_the_compaction_callback_expires_both_ids_and_tolerates_anything(plugin_module):
    """The hook expires both ids, never raises, and treats a missing id as nothing to do."""
    search_module.remember_dump("session-new")
    search_module.remember_dump("session-old")

    assert (
        plugin_module.on_post_compaction(
            session_id="session-new",
            old_session_id="session-old",
            kind="summarize",
            reason="handoff_commit",
            before_tokens=None,
            after_tokens=None,
            messages_removed=2,
            telemetry_schema_version="hermes.observer.v1",
        )
        is None
    )
    assert not search_module.dumped("session-new")
    assert not search_module.dumped("session-old")

    assert plugin_module.on_post_compaction() is None
    assert plugin_module.on_post_compaction(session_id="") is None
    assert plugin_module.on_post_compaction(session_id=None, old_session_id=7) is None

    search_module.remember_dump("session-repeat")
    assert plugin_module.on_post_compaction(session_id="session-repeat") is None
    assert not search_module.dumped("session-repeat")
    assert plugin_module.on_post_compaction(session_id="session-repeat") is None


def test_the_registered_hook_opens_the_window_the_rewrite_reads(tmp_path, plugin_module):
    """The event reaches the window through the registration route the host uses.

    ``register`` stores the callback under ``post_compaction``; invoking the stored callback, as
    the host's dispatcher does, makes the next failed pass dump the catalog again, after one more
    failure has already stripped it. This is the route the host takes, not a direct call to the
    record's expire operation.
    """
    ctx = RecordingContext()
    plugin_module.register(ctx)

    assert ctx.tools == ["skill_router_search"]
    assert len(ctx.middleware) == 1
    assert [name for name, _callback in ctx.hooks] == ["post_compaction"]
    callback = ctx.hooks[0][1]

    cfg = make_config(tmp_path)
    index = make_index()
    session = "session-registered"
    payload = _window_history(2, SUMMARY)

    def failing():
        return rewrite(
            payload,
            cfg=cfg,
            index=index,
            embed=FakeEmbedder(result=None),
            rerank=FakeReranker(),
            session_id=session,
        )

    assert failing() is None, "the window's first failure leaves the catalog in place"
    assert search_module.dumped(session)

    stripped = failing()
    assert stripped is not None, "a later failure in the same window stripped the catalog"
    assert stripped["request"]["messages"][-1]["content"] == DROPPED_MESSAGE

    assert (
        callback(
            session_id=session,
            old_session_id="",
            kind="summarize",
            reason="handoff_commit",
            before_tokens=None,
            after_tokens=None,
            messages_removed=2,
            telemetry_schema_version="hermes.observer.v1",
        )
        is None
    )

    assert not search_module.dumped(session), "the registered callback did not expire the record"
    assert failing() is None, "the registered event did not open the window the rewrite reads"


def test_one_pass_resolves_each_candidate_path_at_most_once(tmp_path, monkeypatch):
    """The turn's recency gate and its selection share one resolver.

    The usage map is supplied directly because the store read is another contract's test: what
    this pins is that a name the gate resolved is never handed to the host lookup again before the
    selection, so one pass costs one lookup per distinct candidate. The empty cache path keeps the
    saved map out of it, so only the per-pass memo can prevent the repeat.
    """
    calls = []

    def counting(name):
        calls.append(name)
        return f"/skills/{name}/SKILL.md"

    monkeypatch.setattr(router_module, "skill_path", counting)
    monkeypatch.setattr(library_module, "_usage_scores", lambda *_args: {"alpha": 1.0, "beta": 0.8})

    result = rewrite(_payload(), cfg=make_config(tmp_path), index=make_index(), embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    assert len(calls) == len(set(calls)), f"a candidate was looked up twice: {calls}"
    assert sorted(calls) == ["alpha", "beta", "delta", "gamma"], (
        "a candidate was looked up twice in one pass, or a pathless candidate was dropped"
    )


def test_a_second_turn_after_a_routed_block_does_not_re_dump(tmp_path):
    """Invariant 1, routed branch: the catalog is stripped every turn, never re-dumped."""
    cfg = make_config(tmp_path)
    index = make_index()
    payload = _payload()

    first = rewrite(
        payload, cfg=cfg, index=index, embed=FakeEmbedder(), rerank=FakeReranker()
    )
    assert first is not None
    second_input = _second_turn(payload, block_of(first))

    second = rewrite(
        second_input, cfg=cfg, index=index, embed=FakeEmbedder(result=None), rerank=FakeReranker()
    )

    assert second is None or "## Skills" not in second["request"]["messages"][0]["content"]
