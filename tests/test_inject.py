"""Contract C8: the block's header, row fields, sections, recogniser and kept rows.

The selected rows carry only a name and a score, so the block's descriptions come from the index that
was built for the routed rows, and its read paths come from the host lookup the caller resolved. Every
suggestion and pinned row carries name, description and an absolute path; a row whose path does not
resolve is not emitted. Pinned rows form their own section only in a window's opening block, and the
rows that are neither stay listed under ``Also listed:`` so the strip never drops a name the model
could still open. The module applies no cap: the selector is the only place it is applied.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import inject as inject_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.inject import (  # noqa: E402
    ALSO_LISTED,
    HEADER,
    LOADER_TOOLS,
    ONE_SHOT_HEADER,
    PINNED,
    SUGGESTIONS,
    render,
    suggested_names,
)
from hermes_skill_router.library import Index  # noqa: E402


def make_config(**overrides):
    return load(overrides)


def make_index(pairs):
    """An index holding *pairs* of (name, description); render reads only its text fields."""
    return Index(
        names=[name for name, _description in pairs],
        descriptions=[description for _name, description in pairs],
        vectors=[[1.0] for _pair in pairs],
        dim=1,
        fingerprint="test",
    )


def paths_for(*names):
    """The map the caller builds after resolving every name through the host lookup."""
    return {name: f"/skills/{name}/SKILL.md" for name in names}


def lines_of(block):
    return block.split("\n")


def row_lines(block):
    return [line for line in lines_of(block) if line.startswith("- ")]


def path_lines(block):
    return [line for line in lines_of(block) if line.startswith("  path: ")]


@pytest.fixture(autouse=True)
def _fresh_pinned_warnings(monkeypatch):
    """A fresh warning set per test, so one test's omitted pinned row never silences another."""
    monkeypatch.setattr(inject_module, "_PINNED_WARNED", set())


def test_both_headers_name_every_loader_tool():
    for header in (HEADER, ONE_SHOT_HEADER):
        for tool in LOADER_TOOLS:
            assert tool in header, f"the block does not name the loader tool {tool}"


def test_the_header_opens_the_block_and_names_both_tools():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert block.startswith(HEADER)
    assert block.startswith("Skills routed for this request\n")
    # T7 quotes this header, so these three facts are the injected contract.
    assert "may help" in HEADER
    assert "skill_view" in HEADER
    assert "skills_list" in HEADER
    # The block replaces a heading T5 checks is gone, so it must not reintroduce that heading.
    assert "## Skills" not in block


def test_the_normal_header_is_unchanged():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert block.startswith(HEADER)
    assert "even partially relevant" in block
    assert "one-shot" not in block


def test_the_one_shot_header_keeps_the_restriction_the_prose_carried():
    kept = CatalogRows(("pinned",), ("Pinned skill",))

    block = render(
        [("alpha", 0.9)],
        kept,
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        one_shot=True,
        paths=paths_for("alpha"),
    )

    assert block.startswith(ONE_SHOT_HEADER)
    assert (
        "Do not load general process skills (testing, debugging, review methodology) for work you "
        "already know how to do" in block
    )
    assert "do not create or edit skills" in block
    assert "one-shot run with no later session to reuse them" in block
    # The invitation the removed prose forbade must not come back with the block.
    assert "even partially relevant" not in block
    assert "even when you already know the basics" not in block
    # The one-shot header still opens a skill and points at the full list.
    assert "skill_view" in block
    assert "skills_list" in block
    assert "- alpha: Alpha skill" in row_lines(block)
    assert "- pinned: Pinned skill" in row_lines(block)


def test_both_headers_share_their_first_line_and_differ_only_after_it():
    marker = HEADER.split("\n", 1)[0]

    assert ONE_SHOT_HEADER.split("\n", 1)[0] == marker
    assert marker == "Skills routed for this request"
    assert ONE_SHOT_HEADER != HEADER
    # The variants differ only after the shared marker line, because the injection guard reads
    # that line alone and must recognise either header.
    assert HEADER[len(marker) :] != ONE_SHOT_HEADER[len(marker) :]
    assert "## Skills" not in HEADER
    assert "## Skills" not in ONE_SHOT_HEADER

    # Both fixed texts are pinned here: the normal one is quoted by technical-documentation.md.
    assert "even partially relevant" in HEADER
    assert "Do not load general process skills" in ONE_SHOT_HEADER
    assert "Do not load general process skills" not in HEADER
    assert "even partially relevant" not in ONE_SHOT_HEADER


def test_a_suggested_row_gets_its_description_from_the_index_and_its_read_path():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths=paths_for("alpha"),
    )
    rows = row_lines(block)
    path = "/skills/alpha/SKILL.md"

    assert "- alpha: Alpha skill" in rows
    assert "  path: " + path in lines_of(block)
    # The path is absolute, which is what makes it a file the model can open.
    assert os.path.isabs(path)
    # The row carries only name, description and path, so the score must never reach the block.
    assert "0.9" not in block


def test_every_suggestion_carries_a_name_a_description_and_an_absolute_path():
    pairs = (("alpha", "Alpha skill"), ("beta", "Beta skill"))
    block = render(
        [("alpha", 0.9), ("beta", 0.8)],
        CatalogRows(),
        make_index(pairs),
        make_config(),
        paths=paths_for("alpha", "beta"),
    )

    for name, description in pairs:
        assert f"- {name}: {description}" in row_lines(block)
        assert f"  path: /skills/{name}/SKILL.md" in lines_of(block)
    assert len(path_lines(block)) == 2


def test_a_suggestion_with_no_description_keeps_its_read_path():
    block = render(
        [("ghost", 0.5)], CatalogRows(), make_index([]), make_config(), paths=paths_for("ghost")
    )

    assert "- ghost" in row_lines(block)
    assert "- ghost:" not in row_lines(block)
    assert "  path: /skills/ghost/SKILL.md" in lines_of(block)


def test_a_pathless_suggestion_is_not_rendered():
    block = render(
        [("pathless", 0.9)], CatalogRows(), make_index([("pathless", "No path")]), make_config()
    )

    assert block == ""
    assert "pathless" not in block


def test_a_relative_suggestion_path_is_not_rendered():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths={"alpha": "skills/alpha/SKILL.md"},
    )

    assert block == ""


def test_a_kept_row_is_listed_under_the_also_listed_marker():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths=paths_for("alpha"),
    )
    lines = lines_of(block)

    assert ALSO_LISTED in lines
    assert lines.index(ALSO_LISTED) > lines.index("- alpha: Alpha skill")
    assert lines[-1] == "- pinned: Pinned skill"


def test_a_kept_row_with_an_empty_description_is_listed_by_name_only():
    block = render(
        [], CatalogRows(("pinned",), ("",)), make_index([]), make_config()
    )

    assert "- pinned" in row_lines(block)
    assert "- pinned:" not in row_lines(block)


def test_a_name_in_both_lists_appears_once_as_a_suggestion():
    block = render(
        [("alpha", 0.5)],
        CatalogRows(("alpha",), ("Kept description",)),
        make_index([("alpha", "Indexed description")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert "- alpha: Indexed description" in row_lines(block)
    assert "Kept description" not in block
    assert len([line for line in row_lines(block) if line.startswith("- alpha")]) == 1


def test_a_duplicate_suggested_name_renders_once():
    block = render(
        [("alpha", 0.9), ("alpha", 0.2)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert row_lines(block) == ["- alpha: Alpha skill"]
    assert len(path_lines(block)) == 1


def test_a_duplicate_kept_name_renders_once():
    block = render(
        [],
        CatalogRows(("pinned", "pinned"), ("Pinned skill", "Second line")),
        make_index([]),
        make_config(),
    )

    assert row_lines(block) == ["- pinned: Pinned skill"]


def test_a_description_is_clipped_to_max_description_chars():
    description = "0123456789abcdef"
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", description)]),
        make_config(maxDescriptionChars=10),
        paths=paths_for("alpha"),
    )

    assert "- alpha: 0123456789" in row_lines(block)
    assert "abcdef" not in block


def test_a_clip_that_lands_on_a_space_leaves_no_trailing_whitespace():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "abc de")]),
        make_config(maxDescriptionChars=4),
        paths=paths_for("alpha"),
    )

    assert "- alpha: abc" in row_lines(block)


def test_a_kept_description_is_clipped_too():
    block = render(
        [],
        CatalogRows(("pinned",), ("0123456789abcdef",)),
        make_index([]),
        make_config(maxDescriptionChars=10),
    )

    assert "- pinned: 0123456789" in row_lines(block)


def test_a_description_with_a_newline_renders_as_one_line():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "first line\nsecond line")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert "- alpha: first line second line" in row_lines(block)
    assert "\nsecond line" not in block


def test_whitespace_runs_inside_a_description_collapse_to_single_spaces():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "first  line\n\nsecond\tline")]),
        make_config(),
        paths=paths_for("alpha"),
    )

    assert "- alpha: first line second line" in row_lines(block)


def test_the_renderer_never_applies_the_cap():
    pairs = [(f"skill-{index}", f"Skill {index}") for index in range(12)]
    block = render(
        pairs,
        CatalogRows(),
        make_index(pairs),
        make_config(maxSuggestions=5),
        paths={name: f"/skills/{name}/SKILL.md" for name, _description in pairs},
    )

    assert len(row_lines(block)) == 12, "the selector is the only place the cap is applied"


def test_twelve_rows_render_with_a_cap_of_zero():
    pairs = [(f"skill-{index}", f"Skill {index}") for index in range(12)]
    block = render(
        pairs,
        CatalogRows(),
        make_index(pairs),
        make_config(maxSuggestions=0),
        paths={name: f"/skills/{name}/SKILL.md" for name, _description in pairs},
    )

    assert len(row_lines(block)) == 12, "a cap of 0 means unlimited in the renderer too"
    assert len(path_lines(block)) == 12


def test_pinned_rows_form_their_own_section_only_in_the_opening_block():
    kept = CatalogRows(("pinned", "demoted"), ("Pinned skill", ""))
    cfg = make_config(eagerSkills=["pinned"])

    opening = render(
        [], kept, make_index([]), cfg, opening=True, paths=paths_for("pinned")
    )
    later = render([], kept, make_index([]), cfg, opening=False)

    assert PINNED in lines_of(opening)
    assert "- pinned: Pinned skill" in row_lines(opening)
    assert "  path: /skills/pinned/SKILL.md" in lines_of(opening)
    assert PINNED not in lines_of(later)
    assert "pinned" not in row_lines(later)
    # The kept row that is not eager stays listed on the later turn.
    assert "- demoted" in row_lines(later)


def test_a_pathless_pinned_row_is_omitted_with_one_warning(tmp_path, caplog):
    cfg = make_config(eagerSkills=["pinned"])
    kept = CatalogRows(("pinned",), ("Pinned skill",))

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.inject"):
        first = render([], kept, make_index([]), cfg, opening=True)
        second = render([("alpha", 0.9)], CatalogRows(), make_index([("alpha", "A")]), cfg, opening=True, paths=paths_for("alpha"))

    assert first == ""
    assert PINNED not in second
    warnings = [message for message in caplog.messages if "pinned skill" in message]
    assert len(warnings) == 1, "a pinned row with no resolvable path warns once per process"
    assert "pinned" in warnings[0]


def test_no_rendered_row_means_an_empty_block():
    cfg = make_config()

    assert render([], CatalogRows(), make_index([]), cfg) == ""


def test_a_suggestion_list_with_no_resolved_path_and_no_kept_row_is_empty():
    cfg = make_config()

    assert render([("alpha", 0.9)], CatalogRows(), make_index([("alpha", "A")]), cfg) == ""


def test_an_empty_suggestion_list_still_lists_the_kept_rows():
    block = render(
        [],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([]),
        make_config(),
    )

    assert block.startswith(HEADER)
    assert "- pinned: Pinned skill" in row_lines(block)


def test_every_kept_name_is_listed_when_the_block_is_non_empty():
    kept = CatalogRows(
        ("eager-skill", "excluded-skill", "demoted-skill"),
        ("Eager skill", "Excluded skill", ""),
    )

    block = render([], kept, make_index([]), make_config())
    rows = row_lines(block)

    assert "- eager-skill: Eager skill" in rows
    assert "- excluded-skill: Excluded skill" in rows
    assert "- demoted-skill" in rows


def test_the_given_row_order_is_preserved():
    block = render(
        [("later", 0.2), ("earlier", 0.9)],
        CatalogRows(),
        make_index([("later", "Later skill"), ("earlier", "Earlier skill")]),
        make_config(),
        paths=paths_for("later", "earlier"),
    )
    rows = row_lines(block)

    assert rows.index("- later: Later skill") < rows.index("- earlier: Earlier skill")


def test_rendering_is_deterministic():
    args = (
        [("alpha", 0.9)],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
    )

    assert render(*args, paths=paths_for("alpha")) == render(*args, paths=paths_for("alpha"))


# ---------------------------------------------------------------------------
# The recogniser: the rewrite recovers the names a prior block suggested from the block itself.
# ---------------------------------------------------------------------------


def test_the_recogniser_returns_exactly_the_suggestion_names():
    block = render(
        [("alpha", 0.9), ("beta", 0.8)],
        CatalogRows(("gamma", "delta"), ("Gamma skill", "Delta skill")),
        make_index([("alpha", "Alpha skill"), ("beta", "Beta skill")]),
        make_config(eagerSkills=["gamma"]),
        opening=True,
        paths=paths_for("alpha", "beta", "gamma"),
    )

    assert suggested_names(block) == ["alpha", "beta"]


def test_the_recogniser_never_returns_a_pinned_or_also_listed_name():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(("pinned", "demoted"), ("Pinned skill", "Demoted skill")),
        make_index([("alpha", "Alpha skill")]),
        make_config(eagerSkills=["pinned"]),
        opening=True,
        paths=paths_for("alpha", "pinned"),
    )

    assert "pinned" not in suggested_names(block)
    assert "demoted" not in suggested_names(block)
    assert suggested_names(block) == ["alpha"]


def test_a_block_that_suggested_nothing_returns_no_suggested_name():
    block = render(
        [],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([]),
        make_config(eagerSkills=["pinned"]),
        opening=True,
        paths=paths_for("pinned"),
    )

    assert suggested_names(block) == []


def test_the_suggestion_marker_opens_the_suggestions_section():
    block = render(
        [("alpha", 0.9)], CatalogRows(), make_index([("alpha", "A")]), make_config(),
        paths=paths_for("alpha"),
    )

    assert SUGGESTIONS in lines_of(block)
