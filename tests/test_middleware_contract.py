"""Contract C9 through the plugin callback and the real middleware dispatcher.

The middleware replaces the whole provider payload with whatever the callback returns, so this file
asserts the two things a caller depends on: the callback returns a mapping with only the ``request``
key (or ``None`` for no change), and the payload that comes out still carries ``model`` and
``tools``. The callback is driven through ``hermes_cli.middleware.apply_llm_request_middleware``
when the installed Hermes is importable, and through a stand-in with the same shape otherwise.

The endpoints are two loopback fakes from ``tests/fake_endpoint.py``, so the whole pipeline runs
with no model, no GPU and no network. ``codex_responses`` and ``anthropic_messages`` payloads must
come out identical, because their prompt is not in ``messages``.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tests.test_plugin_registration import (  # noqa: E402
    _host_modules,
    resolve_host_tree,
)
from fake_endpoint import FakeEndpoint  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402

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

# beta points at the query vector, so it is the row the shortlist and rerank both see first.
VECTORS = {"alpha\nAlpha skill": [1.0, 0.0], "beta\nBeta skill": [0.0, 1.0]}


def chat_payload():
    return {
        "model": "test-model",
        "tools": [{"type": "function", "function": {"name": "skill_view"}}],
        "unrelated_key": {"kept": True},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUERY},
        ],
    }


class RecordingContext:
    """Fake plugin context: records each registrar call and answers ``get_config`` from a dict."""

    def __init__(self, settings):
        self.settings = dict(settings)
        self.middleware = []
        self.tools = []
        self.hooks = []

    def register_middleware(self, kind, callback):
        self.middleware.append((kind, callback))

    def register_tool(self, name, **kwargs):
        self.tools.append((name, kwargs))

    def register_hook(self, hook_name, callback, *, hold_stream=False):
        self.hooks.append((hook_name, callback))

    def get_config(self, key, default=None):
        return self.settings.get(key, default)


def _load_plugin_module(name="hermes_skill_router_middleware_under_test"):
    """Load the repository's plugin entry point the way the plugin host loads a directory plugin."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plugin_module():
    return _load_plugin_module()


def _embed_responder(request):
    from fake_endpoint import Response

    body = request.json()
    vectors = [VECTORS.get(text, [0.0, 1.0]) for text in body["input"]]
    return Response(
        body={
            "model": "bge-m3",
            "data": [
                {"index": index, "embedding": vector} for index, vector in enumerate(vectors)
            ],
        }
    )


def _rerank_responder(request):
    from fake_endpoint import Response

    body = request.json()
    results = [
        {"index": index, "relevance_score": 0.9 if "beta - Beta skill" in document else 0.2}
        for index, document in enumerate(body["documents"])
    ]
    return Response(body={"results": results})


@pytest.fixture()
def endpoints():
    """One fake embeddings endpoint and one fake rerank endpoint, both on loopback."""
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


def settings_for(endpoints, suggestions_path):
    """The plugin settings that point the pipeline at the two fakes and a throwaway record file."""
    embeddings, reranking = endpoints
    return {
        "embedBaseUrl": embeddings.base_url,
        "rerankBaseUrl": reranking.base_url,
        "cachePath": "",
        "suggestionsPath": str(suggestions_path),
        "healthCheckMs": 0,
        # Only beta scores above this, so the record's name list is ordered by score and no
        # near-zero candidate rides along on the default cutoff.
        "cutoff": 0.5,
    }


def record_for(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def _callback_for(plugin_module, settings):
    ctx = RecordingContext(settings)
    plugin_module.register(ctx)
    kind, callback = ctx.middleware[0]
    assert kind == "llm_request"
    return callback


def _stand_in(callback, request, **context):
    """A stand-in with the shape of ``_apply_request_chain``: one callback, ``{payload_key: ...}``."""

    result = callback(request=copy.deepcopy(request), **context)
    if not isinstance(result, dict):
        return SimpleNamespace(payload=request, original_payload=request, changed=False)
    next_payload = result.get("request")
    if not isinstance(next_payload, dict):
        return SimpleNamespace(payload=request, original_payload=request, changed=False)
    return SimpleNamespace(
        payload=copy.deepcopy(next_payload), original_payload=request, changed=True
    )


# ---------------------------------------------------------------------------
# The callback's own contract: one key or None.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _host_paths(monkeypatch):
    """The host lookup resolves every name, and no dump record leaks between tests."""
    from hermes_skill_router import router as router_module
    from hermes_skill_router import search as search_module

    # The broadcast is best effort and imports the host's gateway stack, which this process cannot
    # load; the display record is the seam these tests assert.
    monkeypatch.setattr(router_module, "_broadcast", lambda turn_id: None)
    search_module.clear_dumps()
    resolver = lambda name: f"/skills/{name}/SKILL.md"
    for module_name, module in list(sys.modules.items()):
        if module_name.endswith("router") and hasattr(module, "skill_path"):
            monkeypatch.setattr(module, "skill_path", resolver)


def test_the_callback_returns_only_the_request_key(plugin_module, endpoints, tmp_path):
    record_path = tmp_path / "suggestions.json"
    callback = _callback_for(plugin_module, settings_for(endpoints, record_path))

    result = callback(request=chat_payload(), **DISPATCH_KWARGS)

    assert set(result) == {"request"}
    assert record_for(record_path)["status"] == "routed"
    assert record_for(record_path)["names"] == ["beta"]


def test_model_tools_and_an_unrelated_key_survive(plugin_module, endpoints, tmp_path):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    payload = chat_payload()

    rewritten = callback(request=payload, **DISPATCH_KWARGS)["request"]

    assert rewritten["model"] == "test-model"
    assert rewritten["tools"] == payload["tools"]
    assert rewritten["unrelated_key"] == {"kept": True}
    assert "## Skills" not in rewritten["messages"][0]["content"]
    assert "<available_skills>" not in rewritten["messages"][0]["content"]
    assert rewritten["messages"][-1]["role"] == "system"


def test_an_unchanged_result_is_none(plugin_module, endpoints, tmp_path):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": QUERY},
        ],
    }

    assert callback(request=payload, **DISPATCH_KWARGS) is None
    assert not (tmp_path / "suggestions.json").exists()


# ---------------------------------------------------------------------------
# The real dispatcher, and the stand-in with the same shape.
# ---------------------------------------------------------------------------


def test_the_real_dispatcher_applies_the_rewrite(
    plugin_module, hermes, endpoints, tmp_path, monkeypatch
):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    dispatcher._middleware["llm_request"] = [callback]
    payload = chat_payload()

    result = hermes.middleware.apply_llm_request_middleware(payload, **DISPATCH_KWARGS)

    assert dispatcher.failures == []
    assert result.changed is True
    assert result.payload["model"] == "test-model"
    assert result.payload["tools"] == payload["tools"]
    assert result.payload["messages"][-1]["content"].startswith("Skills routed for this request")
    assert result.original_payload == payload
    assert "## Skills" in result.original_payload["messages"][0]["content"]


def test_a_codex_responses_payload_comes_out_same(
    plugin_module, hermes, endpoints, tmp_path, monkeypatch
):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    dispatcher._middleware["llm_request"] = [callback]
    payload = {
        "model": "test-model",
        "instructions": "You are Hermes.",
        "input": [{"role": "user", "content": QUERY}],
    }
    context = {**DISPATCH_KWARGS, "api_mode": "codex_responses"}

    result = hermes.middleware.apply_llm_request_middleware(payload, **context)

    assert callback(request=payload, **context) is None
    assert result.payload == payload
    assert result.changed is False


def test_an_anthropic_messages_payload_comes_out_same(
    plugin_module, hermes, endpoints, tmp_path, monkeypatch
):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    dispatcher = _install(hermes, _Dispatcher(hermes.dispatch), monkeypatch)
    dispatcher._middleware["llm_request"] = [callback]
    payload = {
        "model": "test-model",
        "system": "You are Hermes.",
        "messages": [{"role": "user", "content": QUERY}],
        "max_tokens": 1024,
    }
    context = {**DISPATCH_KWARGS, "api_mode": "anthropic_messages"}

    result = hermes.middleware.apply_llm_request_middleware(payload, **context)

    assert result.payload == payload
    assert result.changed is False


def test_a_disabled_plugin_returns_none_through_the_callback(
    plugin_module, endpoints, tmp_path
):
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["enabled"] = False
    callback = _callback_for(plugin_module, settings)

    assert callback(request=chat_payload(), **DISPATCH_KWARGS) is None
    assert not (tmp_path / "suggestions.json").exists()


def test_a_payload_without_user_text_comes_out_unchanged_through_the_callback(
    plugin_module, endpoints, tmp_path
):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    payload = chat_payload()
    payload["messages"] = [message for message in payload["messages"] if message["role"] != "user"]

    assert callback(request=payload, **DISPATCH_KWARGS) is None


def test_the_stand_in_chain_has_the_same_shape(plugin_module, endpoints, tmp_path):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))
    payload = chat_payload()
    before = copy.deepcopy(payload)

    result = _stand_in(callback, payload, **DISPATCH_KWARGS)

    assert result.changed is True
    assert result.original_payload == before
    assert result.payload["messages"][-1]["role"] == "system"
    # The stand-in never mutates the request object the dispatcher handed it.
    assert payload == before


def test_the_callback_tolerates_the_full_keyword_set(plugin_module, endpoints, tmp_path):
    callback = _callback_for(plugin_module, settings_for(endpoints, tmp_path / "suggestions.json"))

    parameters = list(inspect.signature(callback).parameters.values())

    assert parameters[0].name == "request"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
