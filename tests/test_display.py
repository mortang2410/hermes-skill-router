"""Contract C9's display record: names only, capped, atomic, removable, and token-free.

The record writer is the whole reason the pane and the TUI widget can show what the router picked.
These tests pin the file shape, the cap, the skip-identical rule, the removal on
``showSuggestions`` false, the idle read for a missing or corrupt file, and the invariant that
matters most to the operator: the display changes no byte of any provider request.
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

from hermes_skill_router import display as display_module  # noqa: E402
from hermes_skill_router import library as library_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import FIELDS, INERT, ROUTED, read, record  # noqa: E402
from hermes_skill_router.inject import HEADER  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402

QUERY = "draft a git commit message for a bug fix"

SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Before replying, scan the skills below.\n\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)

PAYLOAD = {
    "model": "test-model",
    "tools": [{"type": "function", "function": {"name": "skill_view"}}],
    "unrelated_key": {"kept": True},
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUERY},
    ],
}

VECTORS = {"alpha\nAlpha skill": [1.0, 0.0], "beta\nBeta skill": [0.0, 1.0]}


def make_config(tmp_path, **overrides):
    settings = {"cachePath": "", "suggestionsPath": str(tmp_path / "suggestions.json")}
    settings.update(overrides)
    return load(settings)


def read_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fake_embed(cfg, texts):
    return [list(VECTORS.get(text, [0.0, 1.0])) for text in texts]


def fake_rerank(cfg, query, documents):
    return [0.9, 0.2][: len(documents)]


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Fresh once-per-process logger sets and index memo, so tests never mask each other."""
    monkeypatch.setattr(display_module, "_LOGGED", set())
    monkeypatch.setattr(router_module, "_LOGGED", set())
    monkeypatch.setattr(library_module, "_MEMO", {})


# ---------------------------------------------------------------------------
# The file shape.
# ---------------------------------------------------------------------------


def test_a_routed_record_carries_the_schema_the_timestamp_and_the_names(tmp_path):
    cfg = make_config(tmp_path)

    written = record(
        cfg,
        status=ROUTED,
        names=["alpha", "beta"],
        session_id="session-1",
        turn_id="turn-1",
    )

    assert written is not None
    stored = read_file(cfg.suggestions_path)
    assert set(stored) == set(FIELDS)
    assert stored["schema"] == 1
    assert stored["status"] == "routed"
    assert stored["names"] == ["alpha", "beta"]
    assert stored["session_id"] == "session-1"
    assert stored["turn_id"] == "turn-1"
    assert stored["at"].endswith("Z") and "T" in stored["at"]


def test_names_are_capped_at_max_display_names(tmp_path):
    cfg = make_config(tmp_path, maxDisplayNames=3)
    names = [f"skill-{position}" for position in range(50)]

    record(cfg, status=ROUTED, names=names)

    assert read_file(cfg.suggestions_path)["names"] == ["skill-0", "skill-1", "skill-2"]


def test_blank_and_duplicate_names_are_dropped(tmp_path):
    cfg = make_config(tmp_path)

    record(cfg, status=ROUTED, names=["alpha", "  ", "alpha", "beta", 7])

    assert read_file(cfg.suggestions_path)["names"] == ["alpha", "beta"]


def test_an_inert_record_carries_no_names(tmp_path):
    cfg = make_config(tmp_path)

    record(cfg, status=INERT, names=["alpha"])

    assert read_file(cfg.suggestions_path)["names"] == []


def test_an_unknown_status_is_refused(tmp_path, caplog):
    cfg = make_config(tmp_path)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.display"):
        assert record(cfg, status="?", names=["alpha"]) is None

    assert not Path(cfg.suggestions_path).exists()
    assert any("refusing record status" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Write discipline: atomic, skip-identical, newest wins, removable.
# ---------------------------------------------------------------------------


def test_the_write_is_atomic_and_leaves_no_temporary_file(tmp_path):
    cfg = make_config(tmp_path)

    record(cfg, status=ROUTED, names=["alpha"])

    assert sorted(entry.name for entry in Path(cfg.suggestions_path).parent.iterdir()) == [
        "suggestions.json"
    ]


def test_an_identical_record_is_not_rewritten(tmp_path):
    cfg = make_config(tmp_path)
    first = record(cfg, status=ROUTED, names=["alpha"], session_id="s", turn_id="t")
    before = Path(cfg.suggestions_path).read_text(encoding="utf-8")

    second = record(cfg, status=ROUTED, names=["alpha"], session_id="s", turn_id="t")

    assert first is not None
    assert second is None
    assert Path(cfg.suggestions_path).read_text(encoding="utf-8") == before


def test_a_changed_decision_replaces_the_record(tmp_path):
    cfg = make_config(tmp_path)
    record(cfg, status=ROUTED, names=["alpha"])

    record(cfg, status=ROUTED, names=["beta"])

    assert read_file(cfg.suggestions_path)["names"] == ["beta"]


def test_the_newest_record_wins_after_two_decisions(tmp_path):
    cfg = make_config(tmp_path)
    record(cfg, status=ROUTED, names=["alpha"], session_id="session-a", turn_id="turn-a")
    record(cfg, status=ROUTED, names=["beta"], session_id="session-b", turn_id="turn-b")

    stored = read(cfg)

    assert stored["names"] == ["beta"]
    assert stored["session_id"] == "session-b"


def test_show_suggestions_false_removes_an_existing_record(tmp_path):
    shown = make_config(tmp_path)
    record(shown, status=ROUTED, names=["alpha"])
    assert Path(shown.suggestions_path).exists()

    hidden = make_config(tmp_path, showSuggestions=False)

    assert record(hidden, status=ROUTED, names=["beta"]) is None
    assert not Path(hidden.suggestions_path).exists()
    assert read(hidden) is None


def test_a_missing_record_reads_as_idle(tmp_path):
    assert read(make_config(tmp_path)) is None


@pytest.mark.parametrize(
    "body",
    [
        "{not json",
        json.dumps(["a", "list"]),
        json.dumps({"schema": 2, "status": "routed", "names": ["alpha"]}),
        json.dumps({"schema": 1, "status": "weird", "names": ["alpha"]}),
        json.dumps({"schema": 1, "status": "routed", "names": "alpha"}),
        json.dumps({"schema": 1, "status": "routed", "names": [1, 2]}),
    ],
)
def test_an_unreadable_or_wrong_shaped_record_reads_as_idle(tmp_path, body):
    cfg = make_config(tmp_path)
    Path(cfg.suggestions_path).write_text(body, encoding="utf-8")

    assert read(cfg) is None


def test_read_returns_only_the_known_fields(tmp_path):
    cfg = make_config(tmp_path)
    Path(cfg.suggestions_path).write_text(
        json.dumps(
            {
                "schema": 1,
                "at": "2026-11-10T00:00:00Z",
                "session_id": "s",
                "turn_id": "t",
                "status": "routed",
                "names": ["alpha"],
                "user_text": "must not pass through",
            }
        ),
        encoding="utf-8",
    )

    assert set(read(cfg)) == set(FIELDS)


def test_read_accepts_a_config_or_a_path(tmp_path):
    cfg = make_config(tmp_path)
    written = record(cfg, status=ROUTED, names=["alpha"])

    assert read(cfg) == read(cfg.suggestions_path) == written


def test_an_empty_suggestions_path_disables_the_record(tmp_path):
    cfg = load({"suggestionsPath": ""})

    assert display_module.path_for(cfg) is None
    assert record(cfg, status=ROUTED, names=["alpha"]) is None
    assert read(cfg) is None


def test_a_failed_write_is_swallowed_and_logged_once(tmp_path, monkeypatch, caplog):
    cfg = make_config(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(display_module.os, "replace", boom)

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.display"):
        assert record(cfg, status=ROUTED, names=["alpha"]) is None
        assert record(cfg, status=ROUTED, names=["beta"]) is None

    assert not Path(cfg.suggestions_path).exists()
    failures = [message for message in caplog.messages if "cannot write" in message]
    assert len(failures) == 1


# ---------------------------------------------------------------------------
# The zero-token invariant: the display changes no byte of the provider request.
# ---------------------------------------------------------------------------


def test_the_display_costs_no_token(tmp_path):
    """The payload with the record enabled is deep-equal to the same rewrite without it."""
    shown = make_config(tmp_path, showSuggestions=True)
    hidden_path = tmp_path / "other" / "suggestions.json"
    hidden = make_config(tmp_path, showSuggestions=True, suggestionsPath=str(hidden_path))
    payload = copy.deepcopy(PAYLOAD)

    with_display = rewrite(
        payload,
        cfg=shown,
        embed=fake_embed,
        rerank=fake_rerank,
        session_id="session-1",
        turn_id="turn-1",
        notify=lambda _turn_id: None,
    )
    # The second rewrite runs the same pipeline with the record writer replaced by a no-op.
    original_record = router_module.record
    try:
        router_module.record = lambda *args, **kwargs: None
        without_display = rewrite(
            payload,
            cfg=hidden,
            embed=fake_embed,
            rerank=fake_rerank,
            session_id="session-1",
            turn_id="turn-1",
            notify=lambda _turn_id: None,
        )
    finally:
        router_module.record = original_record

    assert with_display is not None and without_display is not None
    assert with_display == without_display
    assert Path(shown.suggestions_path).exists()
    assert not hidden_path.exists()


def test_no_record_field_appears_in_any_message(tmp_path):
    cfg = make_config(tmp_path)

    result = rewrite(
        PAYLOAD,
        cfg=cfg,
        embed=fake_embed,
        rerank=fake_rerank,
        session_id="session-secret",
        turn_id="turn-secret",
        notify=lambda _turn_id: None,
    )

    stored = read_file(cfg.suggestions_path)
    flattened = json.dumps(result["request"]["messages"])
    assert "session-secret" not in flattened
    assert "turn-secret" not in flattened
    assert stored["at"] not in flattened
    assert '"status"' not in flattened
    assert '"schema"' not in flattened
    # The block itself is the only place a name may appear, and it does, because it lists it.
    assert "- alpha: Alpha skill" in result["request"]["messages"][-1]["content"]
    assert "- beta: Beta skill" in result["request"]["messages"][-1]["content"]


def test_the_rewrite_records_routed_names_and_lists_kept_rows(tmp_path):
    cfg = make_config(tmp_path, eagerSkills=["alpha"])

    result = rewrite(
        PAYLOAD, cfg=cfg, embed=fake_embed, rerank=fake_rerank, notify=lambda _turn_id: None
    )

    stored = read_file(cfg.suggestions_path)
    assert stored["status"] == "routed"
    assert stored["names"] == ["beta"]
    assert "alpha" not in stored["names"]
    block = result["request"]["messages"][-1]["content"]
    assert block.startswith(HEADER)
    assert "- alpha: Alpha skill" in block.split("\n")
