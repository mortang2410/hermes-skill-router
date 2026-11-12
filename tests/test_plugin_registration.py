"""Plugin registration, manifest and middleware-dispatch contract for the plugin shell.

The manifest is parsed with a strict loader that refuses duplicate keys. The callback is driven
through the real Hermes dispatcher when the installed Hermes is importable, using the full keyword
set ``agent/turn_api_request.py`` passes, and a deliberately narrow callback proves that the
dispatcher passes only the names a callback declares, which is why the plugin's own callback takes
``**kwargs``. The install under test is the one the running ``hermes`` launcher reports (C26), so the
hook contract runs against a host that actually has the event.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint, Response  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402
from hermes_skill_router.config import BOUNDS, DEFAULTS  # noqa: E402

MANIFEST_PATH = ROOT / "plugin.yaml"
MANIFEST_TEXT = MANIFEST_PATH.read_text(encoding="utf-8")

# The keyword set agent/turn_api_request.py passes to apply_llm_request_middleware.
DISPATCH_KWARGS = {
    "task_id": "task-1",
    "turn_id": "turn-1",
    "api_request_id": "api-request-1",
    "session_id": "session-1",
    "platform": "cli",
    "model": "test-model",
    "provider": "openai",
    "base_url": "http://127.0.0.1:8012",
    "api_mode": "chat_completions",
    "api_call_count": 1,
}

BOOL_KEYS = {
    "enabled",
    "showSuggestions",
    "channelsSemantic",
    "channelsBm25",
    "channelsAssociations",
}
LIST_KEYS = {"eagerSkills", "deferredSkills"}
INT_KEYS = {
    "embedTimeoutMs",
    "rerankTimeoutMs",
    "routingTimeoutMs",
    "shortlistSize",
    "maxSuggestions",
    "frecencyHalfLifeDays",
    "frecencyCapPositions",
    "maxDisplayNames",
    "maxDescriptionChars",
    "healthCheckMs",
}
FLOAT_KEYS = {"cutoff"}
STR_KEYS = set(DEFAULTS) - BOOL_KEYS - LIST_KEYS - INT_KEYS - FLOAT_KEYS


class RecordingContext:
    """Fake plugin context: records registrar calls and answers ``get_config`` from a dict."""

    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        self.tools = []
        self.hooks = []
        self.middleware = []
        self.commands = []

    def register_tool(self, name, *args, **kwargs):
        self.tools.append(name)

    def register_hook(self, hook_name, callback):
        self.hooks.append((hook_name, callback))

    def register_middleware(self, kind, callback):
        self.middleware.append((kind, callback))

    def register_command(self, name, *args, **kwargs):
        self.commands.append(name)

    def get_config(self, key, default=None):
        return self.settings.get(key, default)


def _load_plugin_module(name="hermes_skill_router_plugin_under_test"):
    """Load the repository's plugin entry point the way the plugin host loads a directory plugin."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _strict_load(text):
    """Parse YAML with a loader that raises on a duplicate mapping key."""

    class StrictLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(text, Loader=StrictLoader)


@pytest.fixture(scope="module")
def plugin_module():
    return _load_plugin_module()


# ---------------------------------------------------------------------------
# C26: the host tree is the install the running launcher reports, resolved once per session.
# ---------------------------------------------------------------------------

#: The one resolution and its skip reason, in a mutable box so a test can replace the cache under
#: ``monkeypatch`` without disturbing the session's real answer.
_HOST_TREE: dict[str, tuple[Optional[Path], str]] = {}
_LAUNCHER_TIMEOUT_SECONDS = 10


def _is_host_tree(tree: Path) -> bool:
    """Whether *tree* is the Hermes install these fixtures need: it must ship hermes_cli.middleware."""
    return (tree / "hermes_cli" / "middleware.py").is_file()


def _reported_install(report: str) -> Optional[Path]:
    """Return the path on the report's ``Install directory:`` line, or ``None`` when it has none."""
    for line in report.splitlines():
        label, separator, value = line.partition(":")
        if separator and label.strip() == "Install directory" and value.strip():
            return Path(value.strip())
    return None


def _shim_install(launcher: str) -> Optional[Path]:
    """Return the install the launcher shim execs, or ``None`` when its exec line names none."""
    try:
        text = Path(launcher).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    for line in text.splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        for token in tokens:
            if token.endswith("hermes_cli/main.py"):
                tree = Path(token).parents[1]
                if _is_host_tree(tree):
                    return tree
    return None


def _probe_host_tree() -> tuple[Optional[Path], str]:
    """Resolve the host tree, or return the reason the skip should name.

    An explicit ``HERMES_AGENT_ROOT`` wins. Otherwise the ``hermes`` launcher on ``PATH`` reports
    its own install, and its shim is read when a paused-update banner leaves the report without that
    line. The launcher call is bounded so a broken launcher cannot hang the suite.
    """
    override = os.environ.get("HERMES_AGENT_ROOT", "").strip()
    if override:
        tree = Path(override)
        if _is_host_tree(tree):
            return tree, ""
        return None, f"no Hermes install at {tree} (from HERMES_AGENT_ROOT)"
    launcher = shutil.which("hermes")
    if launcher is None:
        return None, "no hermes launcher on PATH and HERMES_AGENT_ROOT is unset"
    try:
        completed = subprocess.run(
            [launcher, "--version"],
            capture_output=True,
            text=True,
            timeout=_LAUNCHER_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report_error = f"`{launcher} --version` failed: {exc}"
    else:
        tree = _reported_install(completed.stdout)
        if tree is not None and _is_host_tree(tree):
            return tree, ""
        if completed.returncode != 0:
            lines = (completed.stderr or completed.stdout).strip().splitlines()
            report_error = (
                f"`{launcher} --version` exited {completed.returncode}: "
                f"{lines[-1] if lines else 'no output'}"
            )
        elif tree is not None:
            report_error = f"the reported install at {tree} has no hermes_cli/middleware.py"
        else:
            report_error = f"`{launcher} --version` printed no `Install directory:` line"
    tree = _shim_install(launcher)
    if tree is not None:
        return tree, ""
    return None, f"{report_error}; the launcher shim at {launcher} named no install"


_HOST_TOP_LEVELS = ("agent", "hermes_cli", "tools")


def _is_host_module(name: str) -> bool:
    return name.split(".")[0] in _HOST_TOP_LEVELS


@contextlib.contextmanager
def _host_modules(install: Path):
    """Import the host packages from *install*, then restore the modules this process already had.

    The environment also carries an editable copy of an older host source, and the plugin's display
    broadcast imports the host's gateway stack, which this test process cannot import. Some earlier
    test imports the older copy's ``agent`` and ``tools``; purging only ``hermes_cli`` leaves that
    stale ``agent`` in place and the running ``hermes_cli.middleware`` cannot find
    ``agent.safe_worker_policy``. Swapping all three host top-level packages only for the tests that
    assert against the running host keeps the rest of the session on its modules.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_host_module(name)}
    path = str(install)
    added = path not in sys.path
    if added:
        sys.path.insert(0, path)
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _is_host_module(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
        if added:
            sys.path.remove(path)


def resolve_host_tree() -> Path:
    """Return the install the host contract tests run against, or skip with the reason (C26).

    One resolution per test session: the first call asks the launcher and every later call reuses
    that answer. A missing host skips the calling test with the path or command that was tried, so
    the skip never hides an assertion against some other tree.
    """
    if "resolution" not in _HOST_TREE:
        _HOST_TREE["resolution"] = _probe_host_tree()
    tree, reason = _HOST_TREE["resolution"]
    if tree is None:
        pytest.skip(reason)
    return tree


@pytest.fixture
def hermes(monkeypatch, tmp_path):
    """The running host's middleware/dispatch modules, or a skip when they cannot be imported."""
    install = resolve_host_tree()
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    with _host_modules(install):
        try:
            import hermes_cli.middleware as middleware
            import hermes_cli.plugins as plugins
            import hermes_cli.plugins_dispatch as dispatch
        except Exception as exc:  # pragma: no cover - depends on the local install
            pytest.skip(f"Hermes install at {install} is not importable: {exc}")
        yield SimpleNamespace(middleware=middleware, plugins=plugins, dispatch=dispatch)


class _Dispatcher:
    """PluginDispatchMixin with its own callback table, so no live plugin is discovered."""

    def __init__(self, dispatch_module):
        self._middleware = {}
        self.failures = []
        # The mixin's methods come from the real dispatcher class.
        mixin = dispatch_module.PluginDispatchMixin
        self.invoke_middleware = mixin.invoke_middleware.__get__(self)
        # The running dispatcher slices keywords to a callback's declared names; an older host
        # passed every keyword, so keep that behaviour when the helper is absent.
        self._hook_callback_kwargs = getattr(
            mixin, "_hook_callback_kwargs", lambda callback, payload: dict(payload)
        )

    def _report_hook_failure(self, hook_name, cb, kwargs, exc, *, surface="Hook"):
        self.failures.append((hook_name, cb, exc))


def _install(hermes, dispatcher, monkeypatch):
    """Route hermes_cli.middleware through the isolated dispatcher instead of live plugins."""
    monkeypatch.setattr(
        hermes.plugins,
        "has_middleware",
        lambda kind: bool(dispatcher._middleware.get(kind)),
        raising=False,
    )
    monkeypatch.setattr(
        hermes.plugins,
        "invoke_middleware",
        lambda kind, **kwargs: dispatcher.invoke_middleware(kind, **kwargs),
        raising=False,
    )
    return dispatcher


@pytest.fixture(autouse=True)
def _host_paths(monkeypatch):
    """The host lookup resolves every name, and the display's gateway notification stays local."""
    from hermes_skill_router import router as router_module

    # The broadcast is best effort and imports the host's gateway stack, which this process cannot
    # load; the display record is the seam these tests assert.
    monkeypatch.setattr(router_module, "_broadcast", lambda turn_id: None)
    resolver = lambda name: f"/skills/{name}/SKILL.md"
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("router") and hasattr(module, "skill_path"):
            monkeypatch.setattr(module, "skill_path", resolver)


def test_register_records_one_middleware_one_search_tool_and_one_hook(plugin_module):
    ctx = RecordingContext()
    plugin_module.register(ctx)

    assert ctx.tools == ["skill_router_search"]
    assert ctx.hooks == [("post_compaction", plugin_module.on_post_compaction)]
    assert ctx.commands == []
    assert len(ctx.middleware) == 1
    kind, callback = ctx.middleware[0]
    assert kind == "llm_request"
    assert callable(callback)


def test_callback_is_declared_to_accept_the_dispatcher_keyword_set(plugin_module):
    ctx = RecordingContext()
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]

    parameters = list(inspect.signature(callback).parameters.values())
    assert parameters[0].name == "request"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters), (
        "the dispatcher calls cb(**kwargs) with no signature filtering"
    )
    assert callback(request={"messages": []}, **DISPATCH_KWARGS) is None


def test_plugin_yaml_parses_strictly_and_declares_the_capabilities():
    manifest = _strict_load(MANIFEST_TEXT)

    assert manifest["manifest_version"] == 2
    assert manifest["api_version"] == 1
    assert manifest["name"] == "hermes-skill-router"
    assert manifest["version"] == "0.2.0"
    assert 'version: "0.2.0"' in MANIFEST_TEXT, "the version literal must stay quoted"
    assert manifest["requires_hermes"] == ">=0.22.3"
    assert manifest["description"].strip()
    assert manifest["provides_tools"] == ["skill_router_search"]
    assert manifest["provides_hooks"] == ["post_compaction"]
    # The catalog capability must be declared or `hermes plugins validate` rejects the plugin.
    # This build's manifest parser logs one unknown-field warning per load; that is expected.
    assert manifest["provides_middleware"] == ["llm_request"]
    assert manifest["license"] == "MIT"


def test_strict_loader_refuses_duplicate_keys():
    with pytest.raises(yaml.YAMLError):
        _strict_load("name: one\nname: two\n")


def test_config_schema_covers_every_key_with_default_and_bound():
    manifest = _strict_load(MANIFEST_TEXT)
    schema = manifest["config_schema"]

    assert set(schema) == set(DEFAULTS)
    for key, spec in schema.items():
        assert "default" in spec, key
        if key in LIST_KEYS:
            assert spec["type"] == "list", key
            assert spec.get("items") == "str", key
            assert spec["default"] == list(DEFAULTS[key]), key
        else:
            assert spec["default"] == DEFAULTS[key], key
        if key in INT_KEYS:
            assert spec["type"] == "int", key
            assert (spec["minimum"], spec["maximum"]) == BOUNDS[key], key
        if key in FLOAT_KEYS:
            assert spec["type"] == "float", key
            assert (spec["minimum"], spec["maximum"]) == BOUNDS[key], key
        if key in BOOL_KEYS:
            assert spec["type"] == "bool", key
        if key in STR_KEYS:
            assert spec["type"] == "str", key

    assert schema["rerankProvider"]["values"] == ["cohere"]


def test_dispatcher_runs_the_callback_with_the_full_keyword_set(plugin_module, hermes, monkeypatch):
    ctx = RecordingContext()
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    dispatcher._middleware["llm_request"] = [callback]

    seen = []
    monkeypatch.setattr(
        plugin_module, "rewrite", lambda payload, **kwargs: seen.append(payload) or None
    )

    payload = {"messages": [{"role": "user", "content": "route me"}], "model": "test-model"}
    result = hermes.middleware.apply_llm_request_middleware(payload, **DISPATCH_KWARGS)

    assert dispatcher.failures == []
    assert seen == [payload], "the middleware must actually run"
    assert result.payload == payload
    assert result.changed is False


def test_dispatcher_slices_the_keyword_set_for_a_narrow_callback(hermes, monkeypatch):
    """A narrow callback gets only the names it declares, so the plugin's own callback needs ``**kwargs``.

    The running dispatcher slices the keyword set to each callback's declared parameters, so the ids
    reach the plugin only through its ``**kwargs``; a callback declared with just ``request`` still
    runs, with no failure and without the ids.
    """
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    ran = []

    def narrow(request):
        ran.append(request)
        return {"request": {"messages": []}}

    dispatcher._middleware["llm_request"] = [narrow]
    payload = {"messages": [{"role": "user", "content": "route me"}]}
    result = hermes.middleware.apply_llm_request_middleware(payload, **DISPATCH_KWARGS)

    assert ran == [payload]
    assert dispatcher.failures == []
    assert result.payload == {"messages": []}
    assert result.changed is True


def test_callback_returns_none_when_disabled(plugin_module, monkeypatch):
    ctx = RecordingContext({"enabled": False})
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]
    seen = []
    monkeypatch.setattr(
        plugin_module, "rewrite", lambda payload, **kwargs: seen.append(payload) or None
    )

    assert callback(request={"messages": []}, **DISPATCH_KWARGS) is None
    assert seen == []


def test_callback_returns_none_when_it_raises_internally(plugin_module, monkeypatch):
    ctx = RecordingContext()
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]

    def _boom(payload, **kwargs):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(plugin_module, "rewrite", _boom)

    assert callback(request={"messages": []}, **DISPATCH_KWARGS) is None


def test_hermes_plugins_validate_accepts_the_manifest(tmp_path):
    hermes = shutil.which("hermes")
    if hermes is None:
        pytest.skip("the hermes CLI is not on PATH")

    completed = subprocess.run(
        [hermes, "plugins", "validate", str(ROOT)],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "HERMES_HOME": str(tmp_path / "hermes-home")},
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Validation passed." in completed.stdout
    assert "declared middleware" in completed.stdout


def test_the_resolved_host_catalogue_contains_post_compaction(hermes):
    """The host the suite resolved can deliver the hook the plugin registers (C26)."""
    assert "post_compaction" in hermes.plugins.VALID_HOOKS


def test_the_resolver_skips_when_no_host_resolves(monkeypatch):
    """Force C26's missing-host path: no launcher and no override, so the skip names the look."""
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_HOST_TREE", {})
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.delenv("HERMES_AGENT_ROOT", raising=False)

    with pytest.raises(pytest.skip.Exception, match="no hermes launcher on PATH"):
        resolve_host_tree()


def test_the_resolver_reads_the_launcher_shim_when_the_report_lacks_the_line(monkeypatch):
    """A paused-update banner can leave the report without the install line; the shim still names it."""
    real = resolve_host_tree()
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_HOST_TREE", {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="hermes: update banner\n", stderr=""
        ),
    )

    assert resolve_host_tree() == real


def test_manifest_load_warns_once_about_the_unknown_field(hermes, caplog):
    from hermes_cli.plugins_manifest import parse_manifest_file

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        manifest = parse_manifest_file(ROOT / "plugin.yaml", ROOT, "dir", "")

    assert manifest is not None
    assert manifest.name == "hermes-skill-router"
    warnings = [
        record for record in caplog.records
        if "unknown manifest field(s) ignored" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "provides_middleware" in warnings[0].getMessage()


def test_dispatched_session_and_turn_ids_reach_the_rewrite(plugin_module, hermes, monkeypatch):
    ctx = RecordingContext()
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    dispatcher._middleware["llm_request"] = [callback]

    calls = []
    monkeypatch.setattr(
        plugin_module, "rewrite", lambda payload, **kwargs: calls.append(kwargs) or None
    )

    hermes.middleware.apply_llm_request_middleware({"messages": []}, **DISPATCH_KWARGS)

    assert dispatcher.failures == []
    assert len(calls) == 1
    assert calls[0]["session_id"] == "session-1"
    assert calls[0]["turn_id"] == "turn-1"
    assert calls[0]["cfg"].enabled is True


def test_missing_session_and_turn_ids_stay_empty_and_do_not_raise(plugin_module, monkeypatch):
    ctx = RecordingContext()
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]

    calls = []
    monkeypatch.setattr(
        plugin_module, "rewrite", lambda payload, **kwargs: calls.append(kwargs) or None
    )

    assert callback(request={"messages": []}) is None
    assert callback(request={"messages": []}, session_id="", turn_id="") is None
    assert callback(request={"messages": []}, session_id=None, turn_id=None) is None

    assert [call["session_id"] for call in calls] == ["", "", ""]
    assert [call["turn_id"] for call in calls] == ["", "", ""]


# ---------------------------------------------------------------------------
# D8: the declared api_mode gates the rewrite, not the payload's shape.
# ---------------------------------------------------------------------------

CATALOG_SYSTEM = (
    "You are Hermes Agent.\n\n"
    "## Skills\n"
    "Scan the skills below.\n\n"
    "<available_skills>\n"
    "  general:\n"
    "    - alpha: Alpha skill\n"
    "    - beta: Beta skill\n"
    "</available_skills>\n\n"
    "Only proceed without loading a skill if genuinely none are relevant to the task.\n\n"
    "## Memory\n\nNothing yet."
)

QUERY_TEXT = "draft a git commit message for a bug fix"

VECTORS = {"alpha\nAlpha skill": [1.0, 0.0], "beta\nBeta skill": [0.0, 1.0]}


def _catalog_payload():
    """A Chat Completions shaped payload that carries the catalog in a system-role message."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": CATALOG_SYSTEM},
            {"role": "user", "content": QUERY_TEXT},
        ],
    }


def _embed_responder(request):
    body = request.json()
    vectors = [VECTORS.get(text, [0.0, 1.0]) for text in body["input"]]
    return Response(
        body={
            "model": "bge-m3",
            "data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)],
        }
    )


def _rerank_responder(request):
    body = request.json()
    return Response(
        body={
            "results": [
                {"index": index, "relevance_score": 0.9 if "beta - Beta skill" in document else 0.2}
                for index, document in enumerate(body["documents"])
            ]
        }
    )


@pytest.fixture
def catalog_endpoints():
    """Loopback fakes for the whole pipeline; beta carries the query, alpha does not."""
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()
    embeddings = FakeEndpoint(_embed_responder)
    reranking = FakeEndpoint(_rerank_responder)
    try:
        yield embeddings, reranking
    finally:
        embeddings.close()
        reranking.close()
        embed_module.HEALTH.reset()
        rerank_module.HEALTH.reset()


def _catalog_settings(endpoints, record_path):
    embeddings, reranking = endpoints
    return {
        "embedBaseUrl": embeddings.base_url,
        "rerankBaseUrl": reranking.base_url,
        "cachePath": "",
        "suggestionsPath": str(record_path),
        "healthCheckMs": 0,
        "cutoff": 0.5,
    }


def _callback(plugin_module, settings=None):
    ctx = RecordingContext(settings)
    plugin_module.register(ctx)
    _kind, callback = ctx.middleware[0]
    return callback


def test_an_anthropic_mode_dispatch_leaves_a_chat_shaped_payload_untouched(
    plugin_module, catalog_endpoints, tmp_path
):
    record_path = tmp_path / "suggestions.json"
    callback = _callback(plugin_module, _catalog_settings(catalog_endpoints, record_path))
    payload = _catalog_payload()
    original = _catalog_payload()

    result = callback(request=payload, **{**DISPATCH_KWARGS, "api_mode": "anthropic_messages"})

    assert result is None
    assert payload == original
    assert not record_path.exists()


def test_a_chat_completions_dispatch_routes_the_same_payload(
    plugin_module, catalog_endpoints, tmp_path
):
    record_path = tmp_path / "suggestions.json"
    callback = _callback(plugin_module, _catalog_settings(catalog_endpoints, record_path))
    payload = _catalog_payload()

    explicit = callback(request=payload, **DISPATCH_KWARGS)
    assert set(explicit) == {"request"}
    assert "## Skills" not in explicit["request"]["messages"][0]["content"]
    assert "<available_skills>" not in explicit["request"]["messages"][0]["content"]

    without_mode = {key: value for key, value in DISPATCH_KWARGS.items() if key != "api_mode"}
    defaulted = callback(request=_catalog_payload(), **without_mode)
    assert set(defaulted) == {"request"}, "a missing api_mode falls back to chat_completions"
    assert defaulted["request"]["messages"] != _catalog_payload()["messages"]


def test_three_requests_warn_once_about_an_unknown_eager_skill(
    plugin_module, catalog_endpoints, tmp_path, caplog
):
    from hermes_skill_router import catalog as catalog_module

    catalog_module._LOGGED.clear()
    settings = _catalog_settings(catalog_endpoints, tmp_path / "suggestions.json")
    settings["eagerSkills"] = ["typo-skill"]
    callback = _callback(plugin_module, settings)
    payload = _catalog_payload()

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.catalog"):
        for _ in range(3):
            callback(request=payload, **DISPATCH_KWARGS)

    warnings = [record for record in caplog.records if "typo-skill" in record.getMessage()]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
