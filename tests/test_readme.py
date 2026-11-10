"""Ticket T7: the README documents every config key with its default, bound and effect.

The test reads the table between the two ``config-defaults`` markers in README.md and compares it
with ``DEFAULTS`` from the plugin config module. A key added to the module without a README row
fails, a README row for a key that does not exist fails, and a changed default fails. A row whose
bound or effect cell is empty fails too, so the table cannot drift into a bare key list.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router.config import DEFAULTS  # noqa: E402
from hermes_skill_router.inject import HEADER, ONE_SHOT_HEADER  # noqa: E402

README = ROOT / "README.md"
LICENSE_FILE = ROOT / "LICENSE"

START = "<!-- config-defaults:start -->"
END = "<!-- config-defaults:end -->"
COLUMNS = ("key", "default", "bound", "effect")


def _rows() -> list[list[str]]:
    """Return every table row between the config-defaults markers, cells stripped."""
    text = README.read_text(encoding="utf-8")
    assert START in text, f"README.md has no {START} marker"
    assert END in text, f"README.md has no {END} marker"
    region = text.split(START, 1)[1].split(END, 1)[0]
    rows = []
    for line in region.splitlines():
        line = line.strip()
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
    assert rows, "the config-defaults region carries no markdown table"
    return rows


def _plain(cell: str) -> str:
    """Return a table cell without its surrounding code backticks and outer padding.

    Text inside the backticks is returned verbatim, so a default that ends in a space (the rerank
    document prefix does) is compared exactly instead of being trimmed on one side.
    """
    text = cell.strip()
    if len(text) >= 2 and text.startswith("`") and text.endswith("`"):
        return text[1:-1]
    return text


def _rendered(value: object) -> str:
    """Return the default as the README spells it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return "empty" if not value else ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def test_the_table_header_names_the_four_columns() -> None:
    header = [cell.lower() for cell in _rows()[0]]
    assert header == list(COLUMNS)


def test_every_config_key_appears_once_with_its_default() -> None:
    rows = _rows()
    assert len(rows) >= 2, "the config table has no body rows"
    body = rows[2:]  # the header and the separator row
    documented = [_plain(row[0]) for row in body]
    assert len(documented) == len(set(documented)), "a config key appears twice in README.md"
    assert set(documented) == set(DEFAULTS), (
        "the README table and DEFAULTS disagree: "
        f"missing {sorted(set(DEFAULTS) - set(documented))}, "
        f"extra {sorted(set(documented) - set(DEFAULTS))}"
    )
    for row in body:
        key = _plain(row[0])
        assert _plain(row[1]) == _rendered(DEFAULTS[key]), (
            f"{key}: README default {_plain(row[1])!r} does not match DEFAULTS {DEFAULTS[key]!r}"
        )


def test_every_row_has_a_bound_and_an_effect() -> None:
    rows = _rows()[2:]
    for row in rows:
        assert len(row) == 4, f"row for {_plain(row[0])} does not carry four cells: {row}"
        key = _plain(row[0])
        assert _plain(row[2]), f"{key}: the bound cell is empty"
        assert _plain(row[3]), f"{key}: the effect cell is empty"


def _prose() -> str:
    """Return the README with every whitespace run collapsed, for phrase checks across line wraps."""
    return " ".join(README.read_text(encoding="utf-8").split())


def test_the_readme_quotes_both_headers_and_their_shared_first_line() -> None:
    shared = HEADER.split("\n", 1)[0]
    assert shared == ONE_SHOT_HEADER.split("\n", 1)[0], (
        "the normal and one-shot headers no longer share the first line the "
        "double-injection guard uses"
    )
    readme = README.read_text(encoding="utf-8")
    assert HEADER in readme, (
        "README.md does not quote the normal-turn header exactly as the plugin sends it"
    )
    assert ONE_SHOT_HEADER in readme, (
        "README.md does not quote the one-shot header exactly as the plugin sends it"
    )
    assert shared in readme, "README.md does not show the first line both headers share"


def test_the_readme_documents_the_display_caption() -> None:
    prose = _prose()
    assert prose.count("carries the count") >= 2, (
        "README.md does not say both display surfaces carry the count in the heading"
    )
    assert prose.count("muted caption line") >= 2, (
        "README.md does not describe the muted caption on both display surfaces"
    )
    assert "turn <turn_id> · <local HH:MM>" in prose, (
        "README.md does not show the caption's turn id and local clock time format"
    )
    assert prose.count("shown only for a routed record") >= 2, (
        "README.md does not say the caption is shown only for a routed record"
    )


def test_the_readme_documents_once_only_logging_and_url_reduction() -> None:
    prose = _prose()
    assert "unknown eager or deferred name warns once per process" in prose, (
        "README.md does not say an unknown eager or deferred name warns once per process"
    )
    assert "Each rejected setting is logged once per process" in prose, (
        "README.md does not say a rejected setting is logged once per process"
    )
    assert "reduced to its scheme and host" in prose, (
        "README.md does not say an endpoint URL is reduced to scheme and host before logging"
    )
    assert "credentials never reach the log" in prose, (
        "README.md does not state that credentials never reach the log"
    )


def test_the_readme_documents_the_base_url_query_and_fragment_rejection() -> None:
    prose = _prose()
    assert "query or a fragment is rejected" in prose, (
        "README.md does not say a base URL with a query or fragment is rejected"
    )
    assert "falls back to the default with one warning" in prose, (
        "README.md does not say a rejected base URL falls back to the default with one warning"
    )


def test_the_mit_license_is_present_and_named_in_the_readme() -> None:
    license_text = LICENSE_FILE.read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    readme = README.read_text(encoding="utf-8")
    assert "MIT" in readme, "README.md does not name the MIT license"
    assert "LICENSE" in readme, "README.md does not point at LICENSE"


def test_the_readme_uses_no_em_dash() -> None:
    assert "\u2014" not in README.read_text(encoding="utf-8"), "README.md contains an em dash"
