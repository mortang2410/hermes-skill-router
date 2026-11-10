"""Contract C9: the composed rewrite — strip, shortlist, rerank, select, inject, record.

Every test drives :func:`hermes_skill_router.router.rewrite` with a fake embedder and a fake
reranker, so the suite needs no endpoint, no model and no network. What is asserted is the shape
the middleware contract depends on: one appended system message on success, ``None`` on every
failure (which keeps the catalog), only the ``request`` key in the result, and a display record
that carries selected suggestion names only.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router.config import Config, load  # noqa: E402
from hermes_skill_router.inject import HEADER, ONE_SHOT_HEADER  # noqa: E402
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


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """A fresh logger set and index memo per test, so one test's failure never silences another."""
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})


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


def test_the_selected_suggestion_is_rendered_with_its_description_from_the_index(tmp_path):
    result = rewrite(
        _payload(), cfg=make_config(tmp_path), embed=FakeEmbedder(), rerank=FakeReranker()
    )

    lines = block_of(result).split("\n")
    assert "- beta: Beta skill" in lines
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


def test_an_eager_row_is_listed_under_also_listed_and_never_recorded(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"])
    reranker = FakeReranker(scores=(0.9, 0.9, 0.9, 0.9))

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=reranker)

    block = block_of(result)
    assert "Also listed:" in block
    assert "- alpha: Alpha skill" in block.split("\n")
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


def test_max_suggestions_zero_still_lists_the_kept_row(tmp_path):
    cfg = make_config(tmp_path, maxSuggestions=0, eagerSkills=["alpha"])

    result = rewrite(_payload(), cfg=cfg, embed=FakeEmbedder(), rerank=FakeReranker())

    assert result is not None
    assert "## Skills" not in result["request"]["messages"][0]["content"]
    block = block_of(result)
    assert "- alpha: Alpha skill" in block.split("\n")
    assert not any(line.startswith("- ") and line != "- alpha: Alpha skill" for line in block.split("\n"))
    record = record_for(cfg)
    assert record["status"] == "routed"
    assert record["names"] == []


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
    assert record["schema"] == 1
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
# The last user message is the query; a content list contributes its text parts.
# ---------------------------------------------------------------------------


def test_the_last_user_message_supplies_the_query(tmp_path):
    payload = _payload(
        extra=(
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "second request"},
        )
    )
    embedder = FakeEmbedder()
    reranker = FakeReranker(scores=(0.9, 0.2, 0.05, 0.01))

    rewrite(payload, cfg=make_config(tmp_path), embed=embedder, rerank=reranker)

    assert embedder.calls[-1] == ["second request"]
    assert reranker.calls[0][0] == "second request"


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
