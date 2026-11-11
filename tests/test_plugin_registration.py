"""Plugin registration, manifest and middleware-dispatch contract for the plugin shell.

The manifest is parsed with a strict loader that refuses duplicate keys. The callback is driven
through the real Hermes dispatcher when the installed Hermes is importable, using the full keyword
set ``agent/turn_api_request.py`` passes, and a deliberately narrow callback proves that the
dispatcher skips a narrowed signature instead of failing the model call.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

BOOL_KEYS = {"enabled", "showSuggestions"}
LIST_KEYS = {"eagerSkills", "deferredSkills"}
INT_KEYS = {
    "embedTimeoutMs",
    "rerankTimeoutMs",
    "routingTimeoutMs",
    "shortlistSize",
    "maxSuggestions",
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


def _hermes_install():
    override = os.environ.get("HERMES_AGENT_ROOT")
    return Path(override) if override else Path.home() / ".hermes" / "hermes-agent"


@pytest.fixture
def hermes(monkeypatch, tmp_path):
    """The installed Hermes middleware/dispatch modules, or a skip when there is no install."""
    install = _hermes_install()
    if not (install / "hermes_cli" / "middleware.py").is_file():
        pytest.skip(f"no Hermes install at {install}")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    if str(install) not in sys.path:
        sys.path.insert(0, str(install))
    try:
        import hermes_cli.middleware as middleware
        import hermes_cli.plugins as plugins
        import hermes_cli.plugins_dispatch as dispatch
    except Exception as exc:  # pragma: no cover - depends on the local install
        pytest.skip(f"Hermes install at {install} is not importable: {exc}")
    return SimpleNamespace(middleware=middleware, plugins=plugins, dispatch=dispatch)


class _Dispatcher:
    """PluginDispatchMixin with its own callback table, so no live plugin is discovered."""

    def __init__(self, dispatch_module):
        self._middleware = {}
        self.failures = []
        # The mixin's methods come from the real dispatcher class.
        self.invoke_middleware = dispatch_module.PluginDispatchMixin.invoke_middleware.__get__(self)

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
    """The host lookup resolves every name, and no dump record leaks between tests."""
    from hermes_skill_router import search as search_module

    search_module.clear_dumps()
    resolver = lambda name: f"/skills/{name}/SKILL.md"
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("router") and hasattr(module, "skill_path"):
            monkeypatch.setattr(module, "skill_path", resolver)


def test_register_records_one_middleware_and_one_search_tool(plugin_module):
    ctx = RecordingContext()
    plugin_module.register(ctx)

    assert ctx.tools == ["skill_router_search"]
    assert ctx.hooks == []
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
    assert manifest["version"] == "0.1.0"
    assert 'version: "0.1.0"' in MANIFEST_TEXT, "the version literal must stay quoted"
    assert manifest["requires_hermes"] == ">=0.22.2"
    assert manifest["description"].strip()
    assert manifest["provides_tools"] == ["skill_router_search"]
    assert manifest["provides_hooks"] == []
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


def test_dispatcher_skips_a_callback_declared_with_only_request(hermes, monkeypatch):
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    ran = []

    def narrow(request):
        ran.append(request)
        return {"request": {"messages": []}}

    dispatcher._middleware["llm_request"] = [narrow]
    payload = {"messages": [{"role": "user", "content": "route me"}]}
    result = hermes.middleware.apply_llm_request_middleware(payload, **DISPATCH_KWARGS)

    assert ran == []
    assert len(dispatcher.failures) == 1
    assert isinstance(dispatcher.failures[0][2], TypeError)
    assert result.payload == payload
    assert result.changed is False


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
