"""Contracts C5 and C6: the ``skill_router_search`` tool and its handler.

The handler is driven directly and through the host registry's dispatcher, with the HTTP clients
replaced by fakes, so these tests need no endpoint. What they pin: the registered schema, the exact
envelope for a healthy and a degraded answer, the record a healthy call writes, the failure arms,
the disabled switch, and that a search touches no file beyond the existing record and cache.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_skill_router import display  # noqa: E402
from hermes_skill_router import tools as tools_module  # noqa: E402
from hermes_skill_router.catalog import CatalogRows, Partition  # noqa: E402
from hermes_skill_router.config import load  # noqa: E402
from hermes_skill_router.display import read, record  # noqa: E402
from hermes_skill_router.search import (  # noqa: E402
    BUDGET_EXPIRED,
    EMBED_FAILED,
    NO_INDEX,
    RERANK_FAILED,
    SearchOutcome,
    forget_session,
    remember_partition,
)
from hermes_skill_router.tools import handle, register_tools  # noqa: E402

QUERY = "draft a git commit message for a bug fix"

TURN_SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Before replying, scan the skills below. If a skill matches or is even partially relevant to "
    "your task, you MUST load it with skill_view(name) and follow its instructions.\n"
    "\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "    - gamma: Gamma skill\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)
QUERY_VECTOR = [0.0, 1.0, 0.0]

ROWS = (("alpha", "Alpha skill"), ("beta", "Beta skill"), ("gamma", "Gamma skill"))

VECTORS = {
    "alpha\nAlpha skill": [1.0, 0.0, 0.0],
    "beta\nBeta skill": [0.0, 1.0, 0.0],
    "gamma\nGamma skill": [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    def __init__(self, query_result="computed"):
        self.query_result = query_result
        self.calls = []

    def __call__(self, cfg, texts):
        self.calls.append(list(texts))
        if texts == [QUERY] and self.query_result != "computed":
            return self.query_result
        return [list(VECTORS.get(text, QUERY_VECTOR)) for text in texts]


class FakeReranker:
    def __init__(self, scores=(0.9, 0.5, 0.1), result="computed"):
        self.scores = list(scores)
        self.result = result
        self.calls = []

    def __call__(self, cfg, query, documents):
        self.calls.append(list(documents))
        if self.result != "computed":
            return self.result
        return self.scores[: len(documents)]


def partition(routed=ROWS):
    return Partition(
        kept=CatalogRows(),
        routed=CatalogRows(tuple(name for name, _d in routed), tuple(d for _n, d in routed)),
    )


def settings(tmp_path, **overrides):
    values = {"cachePath": "", "suggestionsPath": str(tmp_path / "suggestions.json")}
    values.update(overrides)
    return values


@pytest.fixture
def fake_clients(monkeypatch):
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    monkeypatch.setattr(tools_module, "EMBED", embedder)
    monkeypatch.setattr(tools_module, "RERANK", reranker)
    monkeypatch.setattr(tools_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    return embedder, reranker


@pytest.fixture
def session():
    remember_partition("session-1", partition())
    yield "session-1"
    forget_session("session-1")


class RecordingContext:
    def __init__(self, values):
        self.values = dict(values)
        self.tools = []

    def get_config(self, key, default=None):
        return self.values.get(key, default)

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)


def test_register_tools_declares_one_required_query_parameter():
    ctx = RecordingContext(settings(Path("/tmp")))

    register_tools(ctx)

    assert [tool["name"] for tool in ctx.tools] == ["skill_router_search"]
    tool = ctx.tools[0]
    assert tool["schema"]["parameters"]["required"] == ["query"]
    assert tool["schema"]["parameters"]["properties"]["query"]["type"] == "string"
    assert "do not call it" in tool["description"].lower()
    assert callable(tool["handler"])


def test_an_empty_query_is_rejected_before_any_catalog_or_embedder_read(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "catalog_rows", lambda *a, **k: pytest.fail("read a catalog"))
    monkeypatch.setattr(tools_module, "EMBED", lambda *a, **k: pytest.fail("embedded a query"))

    for query in ("", "   ", "\n\t "):
        answer = json.loads(handle({"query": query}, settings=settings(tmp_path)))

        assert answer == {
            "degraded": True,
            "reason": "invalid-query",
            "detail": "the query must be a non-empty string of the task's own words",
            "skills": [],
        }


def test_a_disabled_switch_answers_disabled_and_writes_no_record(tmp_path):
    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path, enabled=False), session_id="session-1")
    )

    assert answer["reason"] == "disabled"
    assert answer["degraded"] is True
    assert not (tmp_path / "suggestions.json").exists()


def test_a_session_without_a_catalog_answers_no_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "EMBED", lambda *a, **k: pytest.fail("embedded a query"))
    forget_session("session-9")

    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-9")
    )

    assert answer["reason"] == "no-catalog"
    assert answer["skills"] == []
    assert not (tmp_path / "suggestions.json").exists()


def test_a_healthy_search_returns_the_rows_and_writes_the_search_record(
    tmp_path, fake_clients, session
):
    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    )

    assert answer == {
        "skills": [
            {"name": "beta", "description": "Beta skill", "path": "/skills/beta/SKILL.md"},
            {"name": "alpha", "description": "Alpha skill", "path": "/skills/alpha/SKILL.md"},
            {"name": "gamma", "description": "Gamma skill", "path": "/skills/gamma/SKILL.md"},
        ]
    }
    stored = read(display.path_for(settings(tmp_path)["suggestionsPath"]))
    assert stored["origin"] == "search"
    assert stored["status"] == "routed"
    assert stored["names"] == ["beta", "alpha", "gamma"]
    assert stored["session_id"] == "session-1"
    assert stored["turn_id"] == ""


def test_an_empty_selection_is_written_as_a_healthy_empty_record(tmp_path, fake_clients, session):
    fake_clients[1].scores = [0.0, 0.0, 0.0]

    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    )

    assert answer == {"skills": []}
    assert read(display.path_for(settings(tmp_path)["suggestionsPath"]))["names"] == []


@pytest.mark.parametrize("reason", [NO_INDEX, EMBED_FAILED, RERANK_FAILED, BUDGET_EXPIRED])
def test_each_failure_arm_answers_its_reason_and_writes_no_record(
    tmp_path, monkeypatch, reason
):
    remember_partition("session-1", partition())
    monkeypatch.setattr(
        tools_module, "select_rows", lambda *a, **k: SearchOutcome(reason=reason, detail="detail")
    )
    try:
        answer = json.loads(
            handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
        )
    finally:
        forget_session("session-1")

    assert answer == {"degraded": True, "reason": reason, "detail": "detail", "skills": []}
    assert not (tmp_path / "suggestions.json").exists()


def test_an_unbuildable_index_is_the_no_index_arm(tmp_path, monkeypatch, session):
    remember_partition(
        "session-1", partition(routed=(("zeta", "A skill no other test builds"),))
    )

    def failing(cfg, texts):
        return None

    monkeypatch.setattr(tools_module, "EMBED", failing)
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())

    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    )

    assert answer["reason"] == NO_INDEX


def test_show_suggestions_false_returns_rows_and_hides_the_record(tmp_path, monkeypatch, session):
    hidden = settings(tmp_path, showSuggestions=False)
    visible = load(settings(tmp_path))
    record(visible, status=display.ROUTED, origin=display.TURN, names=["old"])
    monkeypatch.setattr(tools_module, "EMBED", FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: None)

    answer = json.loads(handle({"query": QUERY}, settings=hidden, session_id="session-1"))

    assert [row["name"] for row in answer["skills"]] == ["beta", "alpha", "gamma"]
    assert not (tmp_path / "suggestions.json").exists()
    assert read(load(hidden)) is None


def test_a_refused_removal_still_returns_rows_and_the_pane_stays_idle(
    tmp_path, monkeypatch, session, caplog
):
    hidden = settings(tmp_path, showSuggestions=False)
    visible = load(settings(tmp_path))
    record(visible, status=display.ROUTED, origin=display.TURN, names=["old"])
    monkeypatch.setattr(tools_module, "EMBED", FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: None)
    monkeypatch.setattr(
        display.os,
        "unlink",
        lambda *a, **k: (_ for _ in ()).throw(OSError("busy")),
    )
    target = tmp_path / "suggestions.json"

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.display"):
        answer = json.loads(handle({"query": QUERY}, settings=hidden, session_id="session-1"))

    assert [row["name"] for row in answer["skills"]] == ["beta", "alpha", "gamma"]
    assert read(load(hidden)) is None
    stored = json.loads(target.read_text())
    assert stored["status"] == "inert" and stored["names"] == []
    assert any("cannot remove" in message for message in caplog.messages)


def test_a_double_failure_keeps_the_old_file_and_the_pane_stays_idle(
    tmp_path, monkeypatch, session, caplog
):
    hidden = settings(tmp_path, showSuggestions=False)
    visible = load(settings(tmp_path))
    record(visible, status=display.ROUTED, origin=display.TURN, names=["old"])
    monkeypatch.setattr(tools_module, "EMBED", FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: None)
    monkeypatch.setattr(
        display.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("busy"))
    )
    monkeypatch.setattr(
        display.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    target = tmp_path / "suggestions.json"

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.display"):
        answer = json.loads(handle({"query": QUERY}, settings=hidden, session_id="session-1"))

    assert [row["name"] for row in answer["skills"]] == ["beta", "alpha", "gamma"]
    assert read(load(hidden)) is None
    assert json.loads(target.read_text())["names"] == ["old"]
    siblings = [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]
    assert len(siblings) == 1
    messages = " ".join(caplog.messages)
    assert "cannot remove" in messages and "also failed" in messages
    assert str(siblings[0]) in messages


def test_a_failed_record_write_leaves_the_previous_record_and_no_sibling(
    tmp_path, monkeypatch, session
):
    cfg = load(settings(tmp_path))
    record(cfg, status=display.ROUTED, origin=display.TURN, names=["old"])
    before = (tmp_path / "suggestions.json").read_text()
    monkeypatch.setattr(tools_module, "EMBED", FakeEmbedder())
    monkeypatch.setattr(tools_module, "RERANK", FakeReranker())
    monkeypatch.setattr(tools_module, "skill_path", lambda name: None)
    monkeypatch.setattr(display.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    )

    assert [row["name"] for row in answer["skills"]] == ["beta", "alpha", "gamma"]
    assert (tmp_path / "suggestions.json").read_text() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["suggestions.json"]


def test_the_real_dispatcher_records_the_session_id_and_no_turn_id(
    tmp_path, fake_clients, monkeypatch
):
    from tools.registry import registry

    remember_partition("cli-session", partition())
    ctx = RecordingContext(settings(tmp_path))

    def register(**kwargs):
        registry.register(**kwargs)

    monkeypatch.setattr(ctx, "register_tool", register)
    register_tools(ctx)

    try:
        raw = registry.dispatch(
            "skill_router_search", {"query": QUERY}, session_id="cli-session"
        )
    finally:
        forget_session("cli-session")
    answer = json.loads(raw)

    assert [row["name"] for row in answer["skills"]] == ["beta", "alpha", "gamma"]
    stored = read(display.path_for(settings(tmp_path)["suggestionsPath"]))
    assert stored["session_id"] == "cli-session"
    assert stored["origin"] == "search"
    assert stored["turn_id"] == ""


def test_a_search_leaves_the_next_turn_unchanged(tmp_path, fake_clients, session):
    from hermes_skill_router.router import rewrite

    embedder, reranker = fake_clients
    cfg = load(settings(tmp_path))
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": TURN_SYSTEM},
            {"role": "user", "content": QUERY},
        ],
    }
    rewrite(payload, cfg=cfg, embed=embedder, rerank=reranker, session_id="session-1")
    before = read(cfg.suggestions_path)["names"]

    handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    rewrite(payload, cfg=cfg, embed=embedder, rerank=reranker, session_id="session-1")
    after = read(cfg.suggestions_path)

    assert after["names"] == before
    assert after["origin"] == "turn"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["suggestions.json"]


def test_the_tool_repeats_the_turn_selection_on_the_same_catalog(tmp_path, fake_clients):
    from hermes_skill_router.router import rewrite

    embedder, reranker = fake_clients
    cfg = load(settings(tmp_path))
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": TURN_SYSTEM},
            {"role": "user", "content": QUERY},
        ],
    }
    rewritten = rewrite(payload, cfg=cfg, embed=embedder, rerank=reranker, session_id="session-1")
    assert rewritten is not None
    turn_names = read(cfg.suggestions_path)["names"]

    answer = json.loads(
        handle({"query": QUERY}, settings=settings(tmp_path), session_id="session-1")
    )

    assert [row["name"] for row in answer["skills"]] == turn_names


LOADER_SCRIPT = """
import json
from hermes_cli import plugins as plugins_mod

manager = plugins_mod.PluginManager()
manager.discover_and_load()
plugin = manager._plugins.get("hermes-skill-router")
from tools.registry import registry

entry = registry.get_entry("skill_router_search")
print(json.dumps({
    "enabled": bool(plugin and plugin.enabled),
    "error": plugin.error if plugin else None,
    "tool": entry.name if entry else None,
    "toolset": entry.toolset if entry else None,
    "required": entry.schema["parameters"].get("required") if entry else None,
}))
"""


def _host_runtime():
    """Return the installed Hermes interpreter and workspace, or ``None`` when there is none.

    The launcher script names both, so the test loads the plugin exactly the way ``hermes`` does and
    never depends on this checkout being importable by the test interpreter.
    """
    launcher = shutil.which("hermes")
    if not launcher:
        return None
    try:
        text = Path(launcher).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"exec\s+(\S+)\s+(?:-\S+\s+)?(\S*hermes_cli/main\.py)", text)
    if match is None:
        return None
    python = Path(match.group(1))
    workspace = Path(match.group(2)).resolve().parents[1]
    if not python.exists() or not (workspace / "hermes_cli").is_dir():
        return None
    return python, workspace


def test_the_standalone_plugin_manager_registers_the_search_tool(tmp_path):
    runtime = _host_runtime()
    if runtime is None:
        pytest.skip("no installed Hermes runtime to load the plugin through")
    python, workspace = runtime
    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    os.symlink(ROOT, home / "plugins" / "hermes-skill-router")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - hermes-skill-router\n", encoding="utf-8"
    )
    script = tmp_path / "load_plugin.py"
    script.write_text(LOADER_SCRIPT, encoding="utf-8")
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(workspace),
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
    }

    result = subprocess.run(
        [str(python), str(script)],
        capture_output=True,
        text=True,
        timeout=240,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["enabled"] is True, payload["error"]
    assert payload["tool"] == "skill_router_search"
    assert payload["toolset"] == "hermes-skill-router"
    assert payload["required"] == ["query"]


def _fake_host(monkeypatch, module):
    """Install *module* as ``tools.skills_tool`` for one test, as the host would ship it."""
    package = ModuleType("tools")
    package.skills_tool = module
    monkeypatch.setitem(sys.modules, "tools", package)
    monkeypatch.setitem(sys.modules, "tools.skills_tool", module)


def test_skill_path_uses_the_public_host_lookup(monkeypatch):
    module = ModuleType("tools.skills_tool")
    seen = {}

    def skill_search_dirs():
        return [(1, Path("/skills"))], Path("/skills")

    def locate_skill(name, category, roots):
        seen["name"] = name
        seen["category"] = category
        seen["roots"] = roots
        return None, Path("/skills") / name, Path("/skills") / name / "SKILL.md"

    module.skill_search_dirs = skill_search_dirs
    module.locate_skill = locate_skill
    _fake_host(monkeypatch, module)

    assert tools_module.skill_path("alpha") == "/skills/alpha/SKILL.md"
    assert seen == {"name": "alpha", "category": None, "roots": [(1, Path("/skills"))]}


def test_a_refused_public_lookup_keeps_the_row_pathless(monkeypatch):
    module = ModuleType("tools.skills_tool")
    module.skill_search_dirs = lambda: ([(1, Path("/skills"))], Path("/skills"))
    module.locate_skill = lambda name, category, roots: (f"{name}: not found", None, None)
    _fake_host(monkeypatch, module)

    assert tools_module.skill_path("missing") is None


def test_skill_path_falls_back_to_the_private_host_lookup(monkeypatch):
    module = ModuleType("tools.skills_tool")
    module._skill_search_dirs = lambda: ([(1, Path("/project"))], [(1, Path("/all"))], Path("/all"))
    module._locate_skill = lambda name, category, project, every: (
        None,
        Path("/all") / name,
        Path("/all") / name / "SKILL.md",
    )
    _fake_host(monkeypatch, module)

    assert tools_module.skill_path("beta") == "/all/beta/SKILL.md"
