"""Contract C9: decision-time loaded-skill parser and its suppression input.

The parser reads structured tool calls and their matched results, never prose and never a
second catalog. These tests pin the two fixed shapes, the configured custom shape, the cost bound,
and the end-to-end fact that a loaded name is excluded while a shortlist-only loser stays
suggestable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import loader_records as lr  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.select import select  # noqa: E402

KNOWN = {"alpha", "beta"}
PATH_MAP = {
    "alpha": "/skills/alpha/SKILL.md",
    "beta": "/skills/beta/SKILL.md",
}


def _call(call_id, tool, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool, "arguments": json.dumps(arguments)},
    }


def _result(call_id, content, tool="skill_view", is_error=None):
    row = {
        "role": "tool",
        "name": tool,
        "tool_call_id": call_id,
        "content": content,
    }
    if is_error is not None:
        row["isError"] = is_error
    return row


def _results(*rows):
    return {row["tool_call_id"]: row for row in rows}


def _tools(*custom):
    return lr.tool_name_set(load({"skillLoadingTools": list(custom)}))


def test_the_cached_name_set_is_not_a_shared_mutable_dict():
    """A mutation of one returned set must not leak into the next call with the same config.

    The cache stores the dict it hands out, so a caller -- or a test's copy -- that
    edits it under CPU load changed what a later request saw: a full-suite run
    reported "beta" loaded for an isError-free result because the shared dict
    gained a custom tool mid-run. Returning a copy keeps the cache an
    implementation detail.
    """
    base = lr.tool_name_set(load({"skillLoadingTools": []}))
    base["custom_load"] = "custom"
    got = lr.tool_name_set(load({"skillLoadingTools": []}))
    assert "custom_load" not in got, "a caller's mutation leaked through the cache"
    assert got == {"skill_view": "name", "read_file": "path"}


def test_custom_names_are_normalised_and_cached():
    cfg = load({"skillLoadingTools": [" custom_load ", "custom_load", "", "second"]})
    assert cfg.skill_loading_tools == ("custom_load", "second")
    tools = lr.tool_name_set(cfg)
    assert tools == {
        "skill_view": "name",
        "read_file": "path",
        "custom_load": "custom",
        "second": "custom",
    }
    # The cache is built once per exact configured set but every call hands out its own copy,
    # so identity can no longer prove a cache hit; what a caller can observe is the cached dict.
    assert lr.tool_name_set(cfg) is not tools
    assert lr._NAME_SET_CACHE[("custom_load", "second")] == lr.tool_name_set(cfg)
    assert lr.tool_name_set(load({"skillLoadingTools": ["custom_load", "second"]})) == tools
    assert lr.tool_name_set(load({"skillLoadingTools": ["other"]})) != tools
    assert ("other",) in lr._NAME_SET_CACHE


def test_skill_view_success_with_a_known_name_loads():
    loaded = lr.loaded_skills(
        [_call("c1", "skill_view", {"name": "alpha"})],
        _results(_result("c1", json.dumps({"success": True, "name": "alpha"}))),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == ["alpha"]


def test_skill_view_failure_does_not_load():
    loaded = lr.loaded_skills(
        [_call("c2", "skill_view", {"name": "alpha"})],
        _results(_result("c2", json.dumps({"success": False, "name": "alpha"}))),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == []


def test_skill_view_result_name_must_be_known():
    loaded = lr.loaded_skills(
        [_call("c3", "skill_view", {"name": "alpha"})],
        _results(_result("c3", json.dumps({"success": True, "name": "foreign"}))),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == []


def test_read_file_reference_loads_without_a_result_name_or_success():
    loaded = lr.loaded_skills(
        [_call("c4", "read_file", {"path": "/skills/beta/references/rules.md"})],
        _results(_result("c4", "# beta instructions\n", "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == ["beta"]


def test_read_file_of_an_unrelated_file_does_not_load():
    loaded = lr.loaded_skills(
        [_call("c5", "read_file", {"path": "/tmp/notes.txt"})],
        _results(_result("c5", "notes\n", "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == []


def test_read_file_error_and_unmatched_result_do_not_load():
    error = lr.loaded_skills(
        [_call("c6", "read_file", {"path": "/skills/beta/SKILL.md"})],
        _results(_result("c6", json.dumps({"success": False}), "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    unmatched = lr.loaded_skills(
        [_call("c7", "read_file", {"path": "/skills/beta/SKILL.md"})],
        _results(),
        KNOWN, PATH_MAP, _tools(),
    )
    assert error == [] and unmatched == []


def test_read_file_json_wrapped_error_does_not_load():
    loaded = lr.loaded_skills(
        [_call("c20", "read_file", {"path": "/skills/beta/SKILL.md"})],
        _results(_result("c20", json.dumps({"content": "Error: file not found"}), "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == []


def test_read_file_json_wrapped_content_still_loads():
    loaded = lr.loaded_skills(
        [_call("c21", "read_file", {"path": "/skills/beta/SKILL.md"})],
        _results(_result("c21", json.dumps({"content": "# beta instructions\n"}), "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == ["beta"]


def test_every_failure_signal_is_a_load_from_one_probe():
    """Each failure shape stops a load, and the legacy file_path fallback still loads.

    The shapes come from the OCR round-2 probe of the same name: an isError flag, a
    JSON success:false, a truthy error key, an error-prefixed plain body on a read and
    on a custom tool, and a failed custom result. A regression that drops any one
    branch turns exactly that case into a load.
    """
    # Every JSON failure shape is driven through a custom tool with a path argument,
    # because that is the one shape where a dropped branch is observed as a load: a
    # skill_view row with a failure branch removed reads "success is not True", which
    # suppresses the name and hides the regression. The path shape reads only the error
    # flag, so success:false, isError and the error key each have to fail it on their own.
    cases = [
        ("custom tool JSON success:false does not load", _call("e1", "custom_load", {"path": "/skills/alpha/SKILL.md"}), _result("e1", json.dumps({"success": False}), "custom_load"), ()),
        ("custom tool JSON isError does not load", _call("e2", "custom_load", {"path": "/skills/alpha/SKILL.md"}), _result("e2", json.dumps({"isError": True, "name": "alpha"}), "custom_load"), ()),
        ("custom tool JSON error key does not load", _call("e3", "custom_load", {"path": "/skills/alpha/SKILL.md"}), _result("e3", json.dumps({"error": "boom"}), "custom_load"), ()),
        ("skill isError does not load", _call("e4", "skill_view", {"name": "alpha"}), _result("e4", json.dumps({"name": "alpha"}), is_error=True), ()),
        ("read text prefix error not found does not load", _call("e5", "read_file", {"path": "/skills/beta/references/rules.md"}), _result("e5", "Error: file not found"), ()),
        # A matched result with a matched name and no failure signal is a load: this is
        # the positive control, so the rows above fail only on their own signal.
        ("custom tool JSON success loads", _call("e6", "custom_load", {"path": "/skills/alpha/SKILL.md"}), _result("e6", json.dumps({"success": True}), "custom_load"), ("alpha",)),
        ("read text succeeds", _call("e7", "read_file", {"path": "/skills/beta/SKILL.md"}), _result("e7", "# beta instructions\n"), ("beta",)),
    ]
    # Each label carries the parser keyword it pins (isError, success:false, error key),
    # because pytest drops a failing row's assertion message and only the function's
    # docstring survives in the failure output.
    for label, call, result, expected in cases:
        tools = _tools("custom_load") if call["function"]["name"] == "custom_load" else _tools()
        loaded = lr.loaded_skills([call], _results(result), KNOWN, PATH_MAP, tools)
        assert tuple(loaded) == expected, label
    custom_cases = [
        ("custom tool name shape JSON success:false does not load", _call("e8", "custom_load", {"name": "alpha"}), _result("e8", json.dumps({"success": False, "name": "alpha"}), "custom_load")),
        ("custom tool error text prefix does not load", _call("e9", "custom_load", {"name": "alpha"}), _result("e9", "tool error: denied", "custom_load")),
    ]
    for label, call, result in custom_cases:
        loaded = lr.loaded_skills([call], _results(result), KNOWN, PATH_MAP, _tools("custom_load"))
        assert loaded == [], label
    legacy = lr.loaded_skills(
        [_call("e6", "read_file", {"file_path": "/skills/alpha/SKILL.md"})],
        _results(_result("e6", "# alpha instructions\n", "read_file")),
        KNOWN, PATH_MAP, _tools(),
    )
    assert legacy == ["alpha"]


def test_custom_tool_needs_its_exact_configured_name():
    wrong_name = lr.loaded_skills(
        [_call("c8", "other_load", {"name": "alpha"})],
        _results(_result("c8", json.dumps({"success": True, "name": "alpha"}), "other_load")),
        KNOWN, PATH_MAP, _tools("custom_load"),
    )
    right_name = lr.loaded_skills(
        [_call("c9", "custom_load", {"name": "alpha"})],
        _results(_result("c9", json.dumps({"success": True, "name": "alpha"}), "custom_load")),
        KNOWN, PATH_MAP, _tools("custom_load"),
    )
    assert wrong_name == [] and right_name == ["alpha"]


def test_custom_tool_path_wins_over_name():
    loaded = lr.loaded_skills(
        [_call("c10", "custom_load", {"name": "beta", "path": "/skills/alpha/SKILL.md"})],
        _results(_result("c10", "# alpha text\n", "custom_load")),
        KNOWN, PATH_MAP, _tools("custom_load"),
    )
    assert loaded == ["alpha"]


def test_prose_and_wrong_fields_never_load():
    loaded = lr.loaded_skills(
        [
            _call("c11", "custom_load", {"text": "load alpha please"}),
            _call("c12", "custom_load", {"skill": "beta"}),
        ],
        _results(
            _result("c11", json.dumps({"success": True, "name": "alpha"}), "custom_load"),
            _result("c12", json.dumps({"success": True, "name": "beta"}), "custom_load"),
        ),
        KNOWN, PATH_MAP, _tools("custom_load"),
    )
    assert loaded == []


def test_shortlist_only_loser_is_not_a_load():
    assert lr.loaded_skills([], {}, KNOWN, PATH_MAP, _tools()) == []


def test_a_call_id_resolves_once_and_names_are_unique():
    loaded = lr.loaded_skills(
        [
            _call("same", "skill_view", {"name": "alpha"}),
            _call("same", "skill_view", {"name": "beta"}),
            _call("other", "skill_view", {"name": "alpha"}),
        ],
        _results(_result("same", json.dumps({"success": True, "name": "alpha"}))),
        KNOWN, PATH_MAP, _tools(),
    )
    assert loaded == ["alpha"]


class _CountSeq:
    def __init__(self, data):
        self.data = list(data)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.data)


class _CountMap:
    def __init__(self, data):
        self._data = dict(data)
        self.iterations = 0

    def items(self):
        self.iterations += 1
        return self._data.items()


def test_cost_is_one_pass_one_index_and_bounded_lookups():
    records = _CountSeq([_call("c13", "read_file", {"path": "/skills/beta/SKILL.md"})])
    paths = _CountMap(PATH_MAP)
    loaded = lr.loaded_skills(
        records,
        _results(_result("c13", "text\n", "read_file")),
        KNOWN, paths, _tools(),
    )
    assert loaded == ["beta"]
    assert records.iterations == 1
    assert paths.iterations == 1


def test_path_candidates_scale_with_records_not_catalog():
    records = [
        _call("p1", "read_file", {"path": "/skills/beta/references/rules.md"}),
        _call("p2", "custom_load", {"file_path": "/skills/alpha/SKILL.md"}),
    ]
    assert lr.path_candidates(records, KNOWN | {"unrelated"}) == ["beta", "alpha"]
    assert lr.path_candidates([_call("p3", "skill_view", {"name": "beta"})], KNOWN) == []


def test_request_records_pairs_calls_and_results_by_call_id():
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [_call("c14", "skill_view", {"name": "alpha"})],
        },
        _result("c14", json.dumps({"success": True, "name": "alpha"})),
    ]
    records, results = lr.request_records(messages)
    assert len(records) == 1 and records[0]["id"] == "c14"
    assert set(results) == {"c14"}
    assert lr.loaded_skills(records, results, KNOWN, PATH_MAP, _tools()) == ["alpha"]


def test_loaded_names_are_a_select_suppression_input():
    cfg = load({"cutoff": 0.5, "maxSuggestions": 10})
    scores = [0.9, 0.2]
    names = ["beta", "alpha"]
    paths = {"beta": "/skills/beta/SKILL.md", "alpha": "/skills/alpha/SKILL.md"}
    without = select(scores, names, cfg, paths)
    with_load = select(scores, names, cfg, paths, loaded=["beta"])
    assert without == [("beta", 0.9)]
    assert with_load == []
