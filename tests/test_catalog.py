"""Contract C3: catalog strip on a real-shaped system message, partition boundaries, idempotency."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import catalog as catalog_module  # noqa: E402
from hermes_skill_router.catalog import strip_catalog  # noqa: E402

# ---------------------------------------------------------------------------
# A system message in the shape agent/prompt_builder.py emits: the stable part,
# the whole skills section as one paragraph-joined part, then the next volatile part.
# ---------------------------------------------------------------------------

NORMAL_PROSE = (
    "## Skills\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant to your "
    "task, you MUST load it with skill_view(name) and follow its instructions. Err on the side of "
    "loading — it is always better to have context you don't need than to miss critical steps, "
    "pitfalls, or established workflows.\n"
    "If a skill has issues, fix it with skill_manage(action='patch').\n"
)

ONESHOT_GUIDANCE = (
    "## Skills\n"
    "Scan the skills below and load one with skill_view(name) only when it carries domain knowledge "
    "you lack for THIS task (an API, a tool's commands, a project's conventions).\n"
)

TRAILING_GUIDANCE = (
    "Only proceed without loading a skill if genuinely none are relevant to the task."
)

HIDDEN_NOTE = (
    "\n(Categories marked [names only] are outside the current coding context, so their descriptions "
    "are omitted — the skills work normally and load with skill_view(name) as usual.)"
)

STABLE = "You are Hermes Agent.\n\nBe helpful."
MEMORY = "## Memory\n\nNothing yet."

ROWS = (
    ("alpha", "Alpha skill"),
    ("beta", "Beta skill"),
    ("gamma", "Gamma skill"),
    ("delta", "Delta skill"),
    ("epsilon", ""),
)


def _block(rows=ROWS, demoted=()):
    lines = []
    if demoted:
        # The renderer sorts categories, so "creative" precedes "general"; a demoted
        # category collapses to one names-only line with the names sorted and no descriptions.
        lines.append(f"  creative [names only]: {', '.join(sorted(demoted))}")
    lines += ["  general:", "  tools: Tool helpers"]
    for name, description in rows:
        lines.append(f"    - {name}: {description}" if description else f"    - {name}")
    return "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"


def _section(variant="normal", rows=ROWS, hidden_note=False, demoted=(), shadowed_note=None):
    prose = NORMAL_PROSE if variant == "normal" else ONESHOT_GUIDANCE
    section = prose + "\n" + _block(rows, demoted)
    if variant == "normal":
        section += "\n\n" + TRAILING_GUIDANCE
    if hidden_note:
        section += HIDDEN_NOTE
    if shadowed_note is not None:
        section += shadowed_note
    return section


def _content(variant="normal", hidden_note=False, demoted=(), shadowed_note=None):
    section = _section(variant, hidden_note=hidden_note, demoted=demoted, shadowed_note=shadowed_note)
    return STABLE + "\n\n" + section + "\n\n" + MEMORY


def _payload(variant="normal", hidden_note=False, demoted=(), shadowed_note=None):
    return {
        "model": "test-model",
        "tools": [{"type": "function", "function": {"name": "skill_view"}}],
        "unrelated_key": {"kept": True},
        "messages": [
            {"role": "system", "content": _content(variant, hidden_note, demoted, shadowed_note)},
            {"role": "user", "content": "route me"},
        ],
    }


def _assert_section_gone(payload):
    content = payload["messages"][0]["content"]
    assert "## Skills" not in content
    assert "<available_skills>" not in content
    assert "</available_skills>" not in content
    assert TRAILING_GUIDANCE not in content
    assert "(Categories marked [names only]" not in content
    assert "[names only]:" not in content
    for name, description in ROWS:
        assert f"- {name}" not in content
        if description:
            assert description not in content


def test_strip_removes_the_whole_skills_section():
    payload = _payload()
    stripped, partition = strip_catalog(payload)

    assert partition is not None
    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    _assert_section_gone(stripped)
    assert partition.kept.names == ()
    assert partition.routed.names == ("alpha", "beta", "gamma", "delta", "epsilon")
    assert partition.routed.descriptions == (
        "Alpha skill",
        "Beta skill",
        "Gamma skill",
        "Delta skill",
        "",
    )


def test_strip_removes_the_single_query_variant():
    payload = _payload(variant="oneshot")
    stripped, partition = strip_catalog(payload)

    assert partition is not None
    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    _assert_section_gone(stripped)


def test_the_partition_records_which_prose_the_strip_removed():
    _normal_payload, normal = strip_catalog(_payload())
    _oneshot_payload, oneshot = strip_catalog(_payload(variant="oneshot"), eager=["alpha"])

    assert normal is not None and oneshot is not None
    assert normal.one_shot is False
    # The variant belongs to the partition, not to an empty kept list: it survives the split.
    assert oneshot.one_shot is True
    assert oneshot.kept.names == ("alpha",)
    assert not hasattr(oneshot.kept, "one_shot"), "the variant belongs on the partition"


@pytest.mark.parametrize("variant", ["normal", "oneshot"])
def test_strip_removes_the_names_only_note(variant):
    payload = _payload(variant=variant, hidden_note=True)
    stripped, _partition = strip_catalog(payload)

    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    _assert_section_gone(stripped)


def test_strip_preserves_every_top_level_key_and_other_messages():
    payload = _payload()
    stripped, _partition = strip_catalog(payload)

    assert stripped["model"] == payload["model"]
    assert stripped["tools"] is payload["tools"]
    assert stripped["unrelated_key"] == payload["unrelated_key"]
    assert stripped["messages"][1] is payload["messages"][1]
    assert len(stripped["messages"]) == len(payload["messages"])


def test_partition_empty_lists_route_every_row():
    _stripped, partition = strip_catalog(_payload())
    assert partition.kept.names == ()
    assert partition.routed.names == ("alpha", "beta", "gamma", "delta", "epsilon")


def test_partition_eager_row_is_kept_and_never_routed():
    _stripped, partition = strip_catalog(_payload(), eager=["alpha"])
    assert partition.kept.names == ("alpha",)
    assert partition.kept.descriptions == ("Alpha skill",)
    assert partition.routed.names == ("beta", "gamma", "delta", "epsilon")
    assert "alpha" not in partition.routed.names
    assert partition.unknown_names == ()


def test_partition_deferred_row_is_routed_and_the_rest_stays_listed():
    _stripped, partition = strip_catalog(_payload(), deferred=["gamma"])
    assert partition.routed.names == ("gamma",)
    assert partition.kept.names == ("alpha", "beta", "delta", "epsilon")
    assert partition.unknown_names == ()


def test_partition_name_in_both_lists_is_kept_with_one_note(caplog):
    catalog_module._LOGGED.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.catalog"):
        _stripped, partition = strip_catalog(_payload(), eager=["alpha"], deferred=["alpha", "beta"])

    assert partition.kept.names == ("alpha", "gamma", "delta", "epsilon")
    assert partition.routed.names == ("beta",)
    both_notes = [record for record in caplog.records if "both eagerSkills and deferredSkills" in record.getMessage()]
    assert len(both_notes) == 1
    assert "alpha" in both_notes[0].getMessage()


def test_partition_unknown_name_is_reported_and_changes_no_row():
    baseline = strip_catalog(_payload())[1]
    _stripped, partition = strip_catalog(_payload(), eager=["zeta"])
    assert partition.unknown_names == ("zeta",)
    assert partition.kept.names == baseline.kept.names
    assert partition.routed.names == baseline.routed.names

    _stripped, deferred_unknown = strip_catalog(_payload(), deferred=["zeta"])
    assert deferred_unknown.unknown_names == ("zeta",)
    assert deferred_unknown.kept.names == ("alpha", "beta", "gamma", "delta", "epsilon")
    assert deferred_unknown.routed.names == ()


def test_unknown_name_logs_one_note(caplog):
    catalog_module._LOGGED.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.catalog"):
        strip_catalog(_payload(), eager=["zeta"])
    notes = [record for record in caplog.records if "not in the catalog" in record.getMessage()]
    assert len(notes) == 1
    assert "zeta" in notes[0].getMessage()


def test_strip_is_idempotent():
    first, partition = strip_catalog(_payload())
    assert partition is not None
    second, second_partition = strip_catalog(first)
    assert second_partition is None
    assert second is first
    assert second["messages"][0]["content"] == first["messages"][0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        "not a mapping",
        {},
        {"messages": "nope"},
        {"messages": []},
        {"messages": [{"role": "user", "content": "no system message"}]},
        {"messages": [{"role": "system", "content": "plain prompt"}]},
        {"messages": [{"role": "system", "content": [{"type": "text", "text": "blocks"}]}]},
        # the block without the heading is not the section
        {"messages": [{"role": "system", "content": STABLE + "\n\n" + _block() + "\n\n" + MEMORY}]},
        # the heading without the block is not the section
        {"messages": [{"role": "system", "content": STABLE + "\n\n" + NORMAL_PROSE + MEMORY}]},
        # an unknown paragraph after the block leaves the section end ambiguous
        {
            "messages": [
                {
                    "role": "system",
                    "content": STABLE
                    + "\n\n"
                    + NORMAL_PROSE
                    + "\n"
                    + _block()
                    + "\n\nSomething else entirely.",
                }
            ]
        },
    ],
    ids=[
        "not-a-mapping",
        "empty-mapping",
        "messages-not-a-list",
        "no-messages",
        "no-system-message",
        "plain-system-message",
        "non-string-content",
        "block-without-heading",
        "heading-without-block",
        "unknown-paragraph-after-block",
    ],
)
def test_no_section_returns_the_payload_unchanged(payload):
    result, partition = strip_catalog(payload)
    assert partition is None
    assert result is payload


def test_only_the_first_system_message_is_examined():
    payload = {
        "messages": [
            {"role": "system", "content": "plain prompt"},
            {"role": "system", "content": _content()},
        ]
    }
    result, partition = strip_catalog(payload)
    assert partition is None
    assert result is payload
    assert result["messages"][1]["content"] == _content()


def test_mutation_disabling_the_strip_makes_the_absence_assertion_fail(monkeypatch):
    payload = _payload()
    enabled_payload, partition = strip_catalog(payload)
    assert partition is not None
    _assert_section_gone(enabled_payload)

    def _disabled_strip(payload, **kwargs):
        return payload, None

    monkeypatch.setattr(catalog_module, "strip_catalog", _disabled_strip)
    mutated_payload, mutated_partition = catalog_module.strip_catalog(payload)
    assert mutated_partition is None
    with pytest.raises(AssertionError):
        _assert_section_gone(mutated_payload)

    monkeypatch.undo()
    restored_payload, restored_partition = catalog_module.strip_catalog(payload)
    assert restored_partition is not None
    _assert_section_gone(restored_payload)


def test_duplicate_catalog_names_stay_in_routed_in_catalog_order():
    rows = (("alpha", "Alpha skill"), ("beta", "Beta skill"), ("alpha", "Alpha again"))
    payload = {
        "messages": [
            {"role": "system", "content": STABLE + "\n\n" + _section(rows=rows) + "\n\n" + MEMORY},
            {"role": "user", "content": "route me"},
        ]
    }
    stripped, partition = strip_catalog(payload)
    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    assert partition.routed.names == ("alpha", "beta", "alpha")
    assert partition.routed.descriptions == ("Alpha skill", "Beta skill", "Alpha again")


DEMOTED = ("sigma", "omega")


def test_demoted_names_only_names_are_kept_with_empty_descriptions():
    payload = _payload(demoted=DEMOTED)
    stripped, partition = strip_catalog(payload)

    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    _assert_section_gone(stripped)
    assert partition.kept.names == ("omega", "sigma")
    assert partition.kept.descriptions == ("", "")
    assert partition.routed.names == ("alpha", "beta", "gamma", "delta", "epsilon")


def test_demoted_names_only_names_are_absent_from_routed():
    _stripped, partition = strip_catalog(_payload(demoted=DEMOTED), eager=["alpha"])
    for name in DEMOTED:
        assert name not in partition.routed.names
        assert name in partition.kept.names
    assert partition.routed.names == ("beta", "gamma", "delta", "epsilon")


def test_demoted_names_only_names_stay_kept_even_when_deferred_names_them():
    payload = _payload(demoted=DEMOTED)
    _stripped, partition = strip_catalog(payload, deferred=["omega", "sigma"])

    assert partition.routed.names == ()
    assert partition.kept.names == ("omega", "sigma", "alpha", "beta", "gamma", "delta", "epsilon")
    assert partition.kept.descriptions[:2] == ("", "")
    assert partition.unknown_names == ()


def test_demoted_names_only_name_in_eager_is_kept_once():
    _stripped, partition = strip_catalog(_payload(demoted=("omega",)), eager=["omega"])
    assert partition.kept.names == ("omega",)
    assert partition.kept.descriptions == ("",)
    assert "omega" not in partition.routed.names


# The copy note this build appends after the trailing guidance, byte for byte from a live prompt.
SHADOWED_NAMES = ("external-effects-board-gate", "notion", "xurl")
SHADOWED_NOTE = (
    "\n(A copy of " + ", ".join(SHADOWED_NAMES) + " is not listed: it shares both its name and its path "
    "with a different skill in the same skills directory tier (e.g. another external_dirs entry), "
    "so skill_view cannot load it — rename one.)"
)
SHADOWED_NOTE_HYPHEN = SHADOWED_NOTE.replace("load it — rename one.", "load it - rename one.")


def test_shadowed_copy_note_leaves_with_the_section():
    payload = _payload(shadowed_note=SHADOWED_NOTE)
    stripped, partition = strip_catalog(payload)

    content = stripped["messages"][0]["content"]
    assert content == STABLE + "\n\n" + MEMORY
    assert "(A copy of" not in content
    _assert_section_gone(stripped)
    for name in SHADOWED_NAMES:
        assert name not in content
        assert name not in partition.kept.names
        assert name not in partition.routed.names
    assert partition.kept.names == ()
    assert partition.routed.names == ("alpha", "beta", "gamma", "delta", "epsilon")


@pytest.mark.parametrize("note", [SHADOWED_NOTE, SHADOWED_NOTE_HYPHEN], ids=["em-dash", "hyphen"])
def test_shadowed_copy_note_separator_variants_leave_with_the_section(note):
    stripped, _partition = strip_catalog(_payload(shadowed_note=note))
    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY


def test_shadowed_copy_note_and_names_only_note_both_leave():
    payload = _payload(hidden_note=True, demoted=DEMOTED, shadowed_note=SHADOWED_NOTE)
    stripped, partition = strip_catalog(payload)

    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    assert partition.kept.names == ("omega", "sigma")
    assert partition.kept.descriptions == ("", "")
    assert partition.routed.names == ("alpha", "beta", "gamma", "delta", "epsilon")
    for name in SHADOWED_NAMES:
        assert name not in partition.kept.names
        assert name not in partition.routed.names


def test_shadowed_copy_note_leaves_with_the_single_query_variant():
    payload = _payload(variant="oneshot", shadowed_note=SHADOWED_NOTE)
    stripped, partition = strip_catalog(payload)

    assert stripped["messages"][0]["content"] == STABLE + "\n\n" + MEMORY
    for name in SHADOWED_NAMES:
        assert name not in partition.kept.names
        assert name not in partition.routed.names


def test_shadowed_copy_note_names_stay_out_of_kept_and_routed_under_eager_config():
    _stripped, partition = strip_catalog(_payload(shadowed_note=SHADOWED_NOTE), eager=["alpha"], deferred=[])
    assert partition.kept.names == ("alpha",)
    for name in SHADOWED_NAMES:
        assert name not in partition.kept.names
        assert name not in partition.routed.names


def test_a_repeated_request_warns_once_per_unknown_name(caplog):
    catalog_module._LOGGED.clear()
    payload = _payload()
    payload["messages"][1]["content"] = "SECRET QUERY TEXT that must never be logged"

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.catalog"):
        for _ in range(3):
            strip_catalog(payload, eager=["zeta-typo"])

    warnings = [record for record in caplog.records if "zeta-typo" in record.getMessage()]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    assert all(
        "SECRET QUERY TEXT" not in record.getMessage() for record in caplog.records
    ), "a catalog warning must never carry request text"


def test_a_repeated_request_warns_once_for_a_name_in_both_lists(caplog):
    catalog_module._LOGGED.clear()
    payload = _payload()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.catalog"):
        for _ in range(3):
            strip_catalog(payload, eager=["alpha"], deferred=["alpha"])

    warnings = [record for record in caplog.records if "both eagerSkills" in record.getMessage()]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
