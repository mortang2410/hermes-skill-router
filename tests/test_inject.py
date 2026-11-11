"""Contract C8: the injected block's header, description lookup, clipping and kept rows.

The selected rows carry only a name and a score, so the block's descriptions come from the index
that was built for the routed rows. Kept rows (eager skills and rows a non-empty deferred list
leaves listed) are appended under ``Also listed:`` so the strip never drops a name the model could
still open.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router.catalog import CatalogRows  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.inject import HEADER, ONE_SHOT_HEADER, render  # noqa: E402
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


def lines_of(block):
    return block.split("\n")


def row_lines(block):
    return [line for line in lines_of(block) if line.startswith("- ")]


def test_the_header_opens_the_block_and_names_both_tools():
    block = render([("alpha", 0.9)], CatalogRows(), make_index([("alpha", "Alpha skill")]), make_config())

    assert block.startswith(HEADER)
    assert block.startswith("Skills routed for this request\n")
    # T7 quotes this header, so these three facts are the injected contract.
    assert "may help" in HEADER
    assert "skill_view" in HEADER
    assert "skills_list" in HEADER
    # The block replaces a heading T5 checks is gone, so it must not reintroduce that heading.
    assert "## Skills" not in block


def test_the_normal_header_is_unchanged():
    block = render([("alpha", 0.9)], CatalogRows(), make_index([("alpha", "Alpha skill")]), make_config())

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


def test_a_suggested_row_gets_its_description_from_the_index():
    block = render(
        [("alpha", 0.9)], CatalogRows(), make_index([("alpha", "Alpha skill")]), make_config()
    )

    assert "- alpha: Alpha skill" in row_lines(block)
    # The row carries only name and score, so the score must never reach the block.
    assert "0.9" not in block


def test_a_suggested_row_absent_from_the_index_is_listed_by_name():
    block = render([("ghost", 0.5)], CatalogRows(), make_index([]), make_config())

    assert "- ghost" in row_lines(block)
    assert "- ghost:" not in row_lines(block)


def test_a_kept_row_is_listed_under_the_also_listed_marker():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([("alpha", "Alpha skill")]),
        make_config(),
    )
    lines = lines_of(block)

    assert "Also listed:" in lines
    assert lines.index("Also listed:") > lines.index("- alpha: Alpha skill")
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
    )

    assert row_lines(block) == ["- alpha: Alpha skill"]


def test_a_duplicate_does_not_spend_a_suggestion_slot():
    block = render(
        [("alpha", 0.9), ("alpha", 0.5), ("beta", 0.4)],
        CatalogRows(),
        make_index([("alpha", "Alpha skill"), ("beta", "Beta skill")]),
        make_config(maxSuggestions=2),
    )

    assert row_lines(block) == ["- alpha: Alpha skill", "- beta: Beta skill"]


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
    )

    assert "- alpha: 0123456789" in row_lines(block)
    assert "abcdef" not in block


def test_a_clip_that_lands_on_a_space_leaves_no_trailing_whitespace():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "abc de")]),
        make_config(maxDescriptionChars=4),
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
    )

    assert "- alpha: first line second line" in row_lines(block)
    assert "\nsecond line" not in block


def test_whitespace_runs_inside_a_description_collapse_to_single_spaces():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(),
        make_index([("alpha", "first  line\n\nsecond\tline")]),
        make_config(),
    )

    assert "- alpha: first line second line" in row_lines(block)


def test_max_suggestions_caps_suggested_rows_only():
    block = render(
        [("alpha", 0.9)],
        CatalogRows(("pinned",), ("Pinned skill",)),
        make_index([("alpha", "Alpha skill")]),
        make_config(maxSuggestions=0),
    )

    assert block.startswith(HEADER)
    assert "- alpha: Alpha skill" not in row_lines(block)
    assert "- pinned: Pinned skill" in row_lines(block)


def test_no_rendered_row_means_an_empty_block():
    cfg = make_config(maxSuggestions=0)

    assert render([("alpha", 0.9)], CatalogRows(), make_index([("alpha", "Alpha skill")]), cfg) == ""
    assert render([], CatalogRows(), make_index([]), cfg) == ""


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

    assert render(*args) == render(*args)
