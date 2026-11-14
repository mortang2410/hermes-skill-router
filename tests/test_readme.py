"""Ticket T7: the documentation records every config key with its default, bound and effect.

The README is the short version and the technical documentation carries the reference, so the tests
that pin reference material read ``technical-documentation.md``: the table between the two
``config-defaults`` markers, the two injected headers, the display caption, the once-only logging
rules and the base URL rejection. They compare the table with ``DEFAULTS`` from the plugin config
module. A key added to the module without a row fails, a row for a key that does not exist fails, and
a changed default fails. A row whose bound or effect cell is empty fails too, so the table cannot
drift into a bare key list. The prose pins carry the window boundary, the below-floor host statement
and the description-authoring rule, so a later edit cannot soften them. The README keeps the MIT
notice, the link to the reference, and the promise that neither file uses an em dash.
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
DOCS = ROOT / "technical-documentation.md"
LICENSE_FILE = ROOT / "LICENSE"

START = "<!-- config-defaults:start -->"
END = "<!-- config-defaults:end -->"
COLUMNS = ("key", "default", "bound", "effect")


def _rows() -> list[list[str]]:
    """Return every table row between the config-defaults markers, cells stripped."""
    text = DOCS.read_text(encoding="utf-8")
    assert START in text, f"technical-documentation.md has no {START} marker"
    assert END in text, f"technical-documentation.md has no {END} marker"
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
    """Return the default as the documentation spells it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value == "":
        return "empty"
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
    assert len(documented) == len(set(documented)), (
        "a config key appears twice in technical-documentation.md"
    )
    assert set(documented) == set(DEFAULTS), (
        "the documentation table and DEFAULTS disagree: "
        f"missing {sorted(set(DEFAULTS) - set(documented))}, "
        f"extra {sorted(set(documented) - set(DEFAULTS))}"
    )
    for row in body:
        key = _plain(row[0])
        assert _plain(row[1]) == _rendered(DEFAULTS[key]), (
            f"{key}: documented default {_plain(row[1])!r} does not match DEFAULTS {DEFAULTS[key]!r}"
        )


def test_every_row_has_a_bound_and_an_effect() -> None:
    rows = _rows()[2:]
    for row in rows:
        assert len(row) == 4, f"row for {_plain(row[0])} does not carry four cells: {row}"
        key = _plain(row[0])
        assert _plain(row[2]), f"{key}: the bound cell is empty"
        assert _plain(row[3]), f"{key}: the effect cell is empty"


def _prose() -> str:
    """Return the docs with every whitespace run collapsed, for phrase checks across line wraps."""
    return " ".join(DOCS.read_text(encoding="utf-8").split())


def test_the_readme_links_the_technical_documentation_and_back() -> None:
    readme = README.read_text(encoding="utf-8")
    docs = DOCS.read_text(encoding="utf-8")
    assert "(technical-documentation.md)" in readme, (
        "README.md does not link technical-documentation.md"
    )
    assert "(README.md)" in docs, "technical-documentation.md does not link README.md"


def test_the_documentation_quotes_both_headers_and_their_shared_first_line() -> None:
    shared = HEADER.split("\n", 1)[0]
    assert shared == ONE_SHOT_HEADER.split("\n", 1)[0], (
        "the normal and one-shot headers no longer share the first line the "
        "double-injection guard uses"
    )
    docs = DOCS.read_text(encoding="utf-8")
    assert HEADER in docs, (
        "technical-documentation.md does not quote the normal-turn header exactly as the plugin "
        "sends it"
    )
    assert ONE_SHOT_HEADER in docs, (
        "technical-documentation.md does not quote the one-shot header exactly as the plugin "
        "sends it"
    )
    assert shared in docs, "technical-documentation.md does not show the first line both headers share"


def test_the_documentation_documents_the_cap_default_paths_and_opening_rule() -> None:
    prose = _prose()
    assert "`maxSuggestions` defaults to `10` and accepts `0` to `50`, and `0` means unlimited" in prose, (
        "the documentation does not state the cap default and the 0 rule"
    )
    assert "Every suggestion carries its name, its description and the absolute read path" in prose, (
        "the documentation does not state the fields each suggestion carries"
    )
    assert "both name the loader tools `skill_view` and `skills_list`" in prose, (
        "the documentation does not carry the loader-tool line"
    )
    assert "A request whose messages carry no earlier block is the window's opening block" in prose, (
        "the documentation does not state the opening-block rule"
    )
    assert "a request carrying an earlier block omits them" in prose, (
        "the documentation does not state the later-turn rule for each compaction window"
    )
    assert "a request whose earlier block that compaction removed is the opening block of that window" in prose, (
        "the documentation does not state that a removed earlier block makes the next request the "
        "opening block"
    )
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "`0` means no cap" in readme, "README.md does not state the 0 rule"
    assert "`maxSuggestions` | `10`" in readme, "README.md does not carry the cap default"


def test_the_documentation_names_the_compaction_event_as_the_window_boundary() -> None:
    prose = _prose()
    assert "A committed compaction event opens a new window" in prose, (
        "the documentation does not name the committed compaction event as the window boundary"
    )
    assert "The host fires the `post_compaction` event after a compaction commits" in prose, (
        "the documentation does not name the post_compaction event the plugin registers"
    )
    assert "Every compaction kind opens a window" in prose, (
        "the documentation does not state that every compaction kind opens a window"
    )
    assert (
        "a fork expires the record of the session it left and the record of the session it entered"
        in prose
    ), "the documentation does not state that a fork expires both session ids"
    for stale in (
        "the record holds the summary's digest",
        "that summary is the window boundary",
        "The first three non-system messages are checked",
        "1,000 visible characters",
        "2,000 visible characters",
    ):
        assert stale not in prose, (
            f"the documentation restores the removed summary-scan claim: {stale!r}"
        )


def test_the_documentation_states_a_below_floor_host_registers_nothing() -> None:
    prose = _prose()
    assert (
        "Hermes compares the manifest's `requires_hermes` with its own version before it imports the "
        "plugin" in prose
    ), "the documentation does not state that the host compares the version floor before the import"
    assert "A host below the declared floor imports nothing and registers nothing" in prose, (
        "the documentation does not state what a host below the floor does"
    )
    assert "logs one warning naming both versions" in prose, (
        "the documentation does not state that the host logs one warning naming both versions"
    )


def test_the_readme_states_how_to_write_a_description_the_router_can_pick() -> None:
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "That upkeep is worth doing in plain Hermes too" in readme, (
        "the README does not say that a tidy skill library helps without the router"
    )
    assert "With this plugin it pays more" in readme, (
        "the README does not say that the same upkeep helps more with the router"
    )
    assert "the router scores the name and the description when it picks a suggestion" in readme, (
        "the README does not name the two fields the router scores"
    )
    assert (
        "plugin picks from the skill name plus the first 400 characters of the "
        "description" in readme
    ), "the README does not state that the plugin picks from the name and the first 400 characters"
    assert "configurable with `maxDescriptionChars`" in readme, (
        "the README does not state that the description cap is configurable"
    )
    assert "put the distinctive words first" in readme, (
        "the README does not state that the distinctive words belong first in a description"
    )


def test_the_readme_names_the_rerank_cost_it_measured() -> None:
    readme = " ".join(README.read_text(encoding="utf-8").split())
    assert "`tasksource/tasksource-decider-nano`" in readme, (
        "the README does not name the rerank model"
    )
    assert "298 MB of weights" in readme, "the README does not name the reranker's weight size"
    assert "8 CPU threads" in readme, "the README does not name the CPU thread count"
    assert "50-row shortlist" in readme and "10-row shortlist" in readme, (
        "the README does not give the measured time at the default shortlist size and at a short one"
    )
    assert "median 0.71 seconds over ten calls" in readme, (
        "the README does not carry the measured 50-row median it states"
    )
    assert "a 10-row shortlist took 0.15 seconds" in readme, (
        "the README does not carry the measured 10-row time it states"
    )
    assert "4.5 and 5.2 seconds" in readme, (
        "the README does not name the measured slow calls that widen the range"
    )


def test_the_documentation_documents_the_display_caption() -> None:
    prose = _prose()
    assert prose.count("carries the count") >= 2, (
        "the documentation does not say both display surfaces carry the count in the heading"
    )
    assert prose.count("muted caption line") >= 2, (
        "the documentation does not describe the muted caption on both display surfaces"
    )
    assert "turn <turn_id> · <local HH:MM>" in prose, (
        "the documentation does not show the caption's turn id and local clock time format"
    )
    assert prose.count("shown only for a routed record") >= 2, (
        "the documentation does not say the caption is shown only for a routed record"
    )


def test_the_documentation_documents_once_only_logging_and_url_reduction() -> None:
    prose = _prose()
    assert "unknown eager or deferred name warns once per process" in prose, (
        "the documentation does not say an unknown eager or deferred name warns once per process"
    )
    assert "Each rejected setting is logged once per process" in prose, (
        "the documentation does not say a rejected setting is logged once per process"
    )
    assert "reduced to its scheme and host" in prose, (
        "the documentation does not say an endpoint URL is reduced to scheme and host before logging"
    )
    assert "credentials never reach the log" in prose, (
        "the documentation does not state that credentials never reach the log"
    )


def test_the_documentation_documents_the_decision_line_and_its_privacy_boundary() -> None:
    prose = _prose()
    assert "skill router: query=" in prose, (
        "the documentation does not name the line that carries the query"
    )
    assert "request text is never logged" not in prose, (
        "the documentation still claims that request text is never logged at all"
    )
    assert "never written to the display record" in prose, (
        "the documentation does not say request text is never written to the display record"
    )
    assert "never written to the index cache" in prose, (
        "the documentation does not say request text is never written to the index cache"
    )
    assert "never appears in a warning" in prose, (
        "the documentation does not say request text never appears in a warning"
    )
    assert "The one exception is the decision line" in prose, (
        "the documentation does not name the one place the query is logged"
    )
    assert "every shortlisted candidate as `name:score`" in prose, (
        "the documentation does not say the line carries every shortlisted candidate with its score"
    )
    assert "including the candidates the cutoff rejected" in prose, (
        "the documentation does not say the rejected candidates stay in the line"
    )
    assert "the values the selector compared with the cutoff" in prose, (
        "the documentation does not say the logged scores are the ones the selector compared"
    )
    assert "writes no decision line" in prose, (
        "the documentation does not say a failed pass writes no decision line"
    )
    assert "like every warning, carries no user text" in prose, (
        "the documentation does not say the failure warnings still carry no user text"
    )
    assert "no warning contains the user's text" in prose, (
        "the Failure behaviour promise that no warning contains user text is gone"
    )


def test_the_documentation_documents_the_channels_modes_and_backfill() -> None:
    prose = _prose()
    assert "`bm25.json`" in prose, "the documentation does not name the term index file"
    assert "the exact document text the embedder sees" in prose, (
        "the documentation does not say both channels read one document text"
    )
    assert "reciprocal rank fusion" in prose, "the documentation does not name the fusion rule"
    assert "the floor of three distinct prompts" in prose, (
        "the documentation does not state the association support floor"
    )
    assert "`fusionMode: shadow` computes" in prose, (
        "the documentation does not describe the shadow mode"
    )
    assert "`fusionMode: cosine` returns the cosine order" in prose, (
        "the documentation does not describe the cosine rollback"
    )
    assert "python3 -m hermes_skill_router.associations --max-sessions 200" in prose, (
        "the documentation does not carry the backfill command"
    )
    assert "reads `~/.hermes/state.db` read-only, writes only the association store" in prose, (
        "the documentation does not state the backfill's read and write boundary"
    )
    assert "uniqueness in the store, not a check, is what" in prose, (
        "the documentation does not state why the backfill is idempotent"
    )
    assert "association channel contributes no ranks" in prose, (
        "the documentation does not say what an empty store costs"
    )


def test_the_documentation_documents_the_base_url_query_and_fragment_rejection() -> None:
    prose = _prose()
    assert "query or a fragment is rejected" in prose, (
        "the documentation does not say a base URL with a query or fragment is rejected"
    )
    assert "falls back to the default with one warning" in prose, (
        "the documentation does not say a rejected base URL falls back to the default with one "
        "warning"
    )


def test_the_mit_license_is_present_and_named_in_the_readme() -> None:
    license_text = LICENSE_FILE.read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    readme = README.read_text(encoding="utf-8")
    assert "MIT" in readme, "README.md does not name the MIT license"
    assert "LICENSE" in readme, "README.md does not point at LICENSE"


def test_both_documents_use_no_em_dash() -> None:
    for path in (README, DOCS):
        assert "\u2014" not in path.read_text(encoding="utf-8"), f"{path.name} contains an em dash"
