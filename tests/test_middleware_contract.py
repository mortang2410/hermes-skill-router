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


def test_disabled_plugin_leaves_the_request_and_state_untouched(
    plugin_module, endpoints, tmp_path
):
    """D2: with the master switch off the plugin returns None and writes nothing."""
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["enabled"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    payload = chat_payload()
    before = copy.deepcopy(payload)
    result = callback(request=payload, **DISPATCH_KWARGS)

    assert result is None
    assert payload == before, "the off path must not mutate the request"
    assert not (tmp_path / "suggestions.json").exists()
    assert not (tmp_path / "config-switches.json").exists()
    # Repeated off turns stay silent and still write nothing.
    assert callback(request=chat_payload(), **DISPATCH_KWARGS) is None
    assert not (tmp_path / "config-switches.json").exists()


def test_disabled_unsupported_modes_preserve_payload_and_deliver_nothing(
    plugin_module, endpoints, tmp_path
):
    """A disabled plugin must not mutate unsupported payloads, and a supported
    request after them carries no warning either (D2: there is none to defer)."""
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["enabled"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    for mode in ("anthropic_messages", "codex_responses"):
        payload = chat_payload()
        payload["api_mode"] = mode
        result = callback(request=payload, api_mode=mode, **{
            k: v for k, v in DISPATCH_KWARGS.items() if k != "api_mode"
        })
        assert result is None, f"disabled plugin must not mutate {mode} payloads"

    # The supported request is untouched too: D2 left no warning to deliver.
    assert callback(request=chat_payload(), **DISPATCH_KWARGS) is None
    assert not (tmp_path / "config-switches.json").exists()


def test_no_catalog_first_then_supported_still_delivers_warning(
    plugin_module, endpoints, tmp_path
):
    """Round 4 P1: the first no-catalog supported turn must deliver the warning.

    Supersedes the pin that expected ``None`` here: a supported request that
    produces no rewrite still has a valid messages channel, so the pending
    warnings ride on the original request and the first supported turn
    delivers; only unsupported or malformed requests defer.
    """
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["showSuggestions"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    no_catalog = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
    first = callback(request=no_catalog, **DISPATCH_KWARGS)
    assert first is not None, "the first supported turn must deliver, not return None"
    assert set(first) == {"request"}
    messages = first["request"]["messages"]
    assert messages[0] == {"role": "user", "content": "hi"}, "the original request stays intact"
    assert messages[-1]["role"] == "system"
    assert "showSuggestions" in messages[-1]["content"] and " is off; " in messages[-1]["content"]

    result = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert result is not None
    messages = result["request"]["messages"]
    assert not any(
        "showSuggestions" in message.get("content", "") and " is off; " in message.get("content", "")
        for message in messages
        if isinstance(message, dict)
    ), "delivery was acknowledged on the first supported turn"
    # Repeat supported turns are suppressed after first delivery: the routed
    # block is already injected, so the callback returns None.
    assert callback(request=result["request"], **DISPATCH_KWARGS) is None


def _warning_messages(result):
    """Return the switch-warning messages carried by a middleware result."""
    if not isinstance(result, dict):
        return []
    request = result.get("request")
    if not isinstance(request, dict):
        return []
    return [
        message
        for message in request.get("messages", [])
        if isinstance(message, dict)
        and " is off; " in message.get("content", "")
        and message.get("content", "").startswith(("enabled: ", "showSuggestions: "))
    ]


def test_every_no_rewrite_path_delivers_on_the_first_supported_turn(
    plugin_module, endpoints, tmp_path, monkeypatch
):
    """Round 4 P1: every no-rewrite path delivers pending warnings at most once.

    The rewrite returns ``None`` for no catalog, an already-injected block, a
    first failure, unsupported modes, and internal exceptions. Each first
    supported turn delivers once (unsupported and malformed requests defer
    instead), and every later turn is suppressed after that delivery.
    """
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["showSuggestions"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    def fresh_cache():
        for name in ("index.json", "config-switches.json", "suggestions.json", "writeback-store.json"):
            path = tmp_path / name
            if path.exists():
                path.unlink()
        # The plugin module's own search instance (the autouse fixture clears
        # only the `hermes_skill_router.search` instance), so a first-failure
        # dump mark from one scenario cannot leak into the next or the next test.
        prefix = plugin_module.__name__ + "."
        for name, module in list(sys.modules.items()):
            if (
                name.startswith(prefix)
                and name.endswith(".search")
                and hasattr(module, "clear_dumps")
            ):
                module.clear_dumps()
        return callback

    # No catalog (first request carries no catalog section).
    fresh_cache()
    no_catalog = {"messages": [{"role": "user", "content": "hi"}], "model": "test-model"}
    first = callback(request=no_catalog, **DISPATCH_KWARGS)
    assert _warning_messages(first), "no-catalog first turn must deliver"
    assert callback(request=first["request"], **DISPATCH_KWARGS) is None

    # Already injected (the last message already carries the block): the
    # warning goes before the trailing block, which stays last.
    fresh_cache()
    injected = chat_payload()
    first = callback(request=injected, **DISPATCH_KWARGS)
    block = first["request"]["messages"][-1]
    assert block["content"].startswith("Skills routed for this request")
    assert _warning_messages(first), "first routed turn must deliver"
    already = {**injected, "messages": first["request"]["messages"]}
    assert callback(request=already, **DISPATCH_KWARGS) is None

    # First failure (no user text): the rewrite returns None and the warning
    # rides on the original request, which keeps its catalog untouched.
    fresh_cache()
    no_user_text = {
        "model": "test-model",
        "messages": [message for message in chat_payload()["messages"] if message["role"] != "user"],
    }
    first = callback(request=no_user_text, **DISPATCH_KWARGS)
    assert _warning_messages(first), "first-failure turn must deliver"
    assert first["request"]["messages"][0]["content"] == chat_payload()["messages"][0]["content"]
    # The delivered warning suppresses the next turn even on this path: a
    # request that already carries it gets no second copy (a dropped-marker
    # rewrite may carry the input's message through, never append a new one),
    # and a fresh payload gets none either.
    second = callback(request=first["request"], **DISPATCH_KWARGS)
    assert len(_warning_messages(second)) <= len(_warning_messages(first)), (
        "the delivered warning must not be delivered twice on this path"
    )
    fresh = {
        "model": "test-model",
        "messages": [m for m in chat_payload()["messages"] if m["role"] != "user"],
    }
    fresh_result = callback(request=fresh, **DISPATCH_KWARGS)
    assert not _warning_messages(fresh_result), "a fresh turn after delivery stays suppressed"

    # Unsupported modes defer: the payload is preserved and delivery stays pending.
    fresh_cache()
    for mode in ("anthropic_messages", "codex_responses"):
        payload = chat_payload()
        payload["api_mode"] = mode
        assert callback(
            request=payload,
            api_mode=mode,
            **{k: v for k, v in DISPATCH_KWARGS.items() if k != "api_mode"},
        ) is None, f"unsupported mode {mode} must not deliver"
    supported = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert _warning_messages(supported), "the first supported turn after deferral delivers"
    assert callback(request=supported["request"], **DISPATCH_KWARGS) is None

    # Malformed requests (no messages list) defer the same way.
    fresh_cache()
    assert callback(request={"model": "test-model"}, **DISPATCH_KWARGS) is None
    assert callback(
        request={"model": "test-model", "messages": "not-a-list"}, **DISPATCH_KWARGS
    ) is None
    first = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert _warning_messages(first), "malformed turns defer; the next supported turn delivers"

    # Internal exception: the rewrite's own guard (router.py) returns None
    # when anything inside the pipeline raises; the first supported turn
    # still delivers on the original request, never a lost call.
    fresh_cache()
    router_globals = plugin_module.rewrite.__globals__
    real_forget = router_globals["forget_session"]

    def _raising_forget(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(router_globals, "forget_session", _raising_forget)
    first = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert _warning_messages(first), "an internal failure must still deliver"
    monkeypatch.setitem(router_globals, "forget_session", real_forget)
    recovered = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert not _warning_messages(recovered), "recovery turn carries no second warning"


def test_already_injected_first_supported_turn_delivers_pending_warning_once(
    plugin_module, endpoints, tmp_path
):
    """Round 5 P2: a pending warning on an already-injected FIRST supported turn.

    The every-no-rewrite-path test injects first and only then replays the
    already-injected payload, so it never exercises a fresh pending warning
    on an already-injected turn. Here the very first turn already carries
    the suggestion block: the warning must still deliver once, before the
    trailing block (which stays last), and the next turn is suppressed.
    """
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["showSuggestions"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    payload = chat_payload()
    payload["messages"] = [
        *payload["messages"],
        {"role": "system", "content": "Skills routed for this request\nbeta (0.9)"},
    ]
    first = callback(request=payload, **DISPATCH_KWARGS)
    warnings = _warning_messages(first)
    assert len(warnings) == 1, "an already-injected first turn delivers the warning once"
    messages = first["request"]["messages"]
    assert messages[-1]["content"].startswith("Skills routed for this request"), (
        "the suggestion block stays last so the replay guard still fires"
    )
    assert messages[-2] == warnings[0], "the warning rides before the trailing block"
    assert callback(request=first["request"], **DISPATCH_KWARGS) is None, (
        "after that delivery the already-injected request stays suppressed"
    )


def test_tool_first_load_then_supported_still_delivers_warning(
    plugin_module, endpoints, tmp_path
):
    """Round 3 P1: a search-tool config load must not consume delivery."""
    from hermes_skill_router.config import load

    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["showSuggestions"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    # The tool's load happens first, with no delivery.
    load(settings)
    callback = _callback_for(plugin_module, settings)
    result = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert result is not None
    messages = result["request"]["messages"]
    assert any(
        "showSuggestions" in message.get("content", "")
        for message in messages
        if isinstance(message, dict)
    ), "middleware after a tool-first load must still deliver"


def test_hidden_suggestions_inject_the_switch_warning_with_the_block(
    plugin_module, endpoints, tmp_path
):
    """With the display switch off the routed block still carries its warning."""
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings["showSuggestions"] = False
    settings["cachePath"] = str(tmp_path / "index.json")
    callback = _callback_for(plugin_module, settings)

    result = callback(request=chat_payload(), **DISPATCH_KWARGS)

    messages = result["request"]["messages"]
    # The suggestion block stays last so the replay guard still recognises it.
    assert messages[-1]["content"].startswith("Skills routed for this request")
    warning = messages[-2]
    assert warning["role"] == "system"
    assert "showSuggestions" in warning["content"] and " is off; " in warning["content"]
    assert not (tmp_path / "suggestions.json").exists()


@pytest.mark.parametrize("switch", ["enabled", "showSuggestions"])
def test_both_empty_paths_deliver_the_warning_through_the_middleware(
    plugin_module, endpoints, tmp_path, monkeypatch, switch
):
    """Round 4 gap: both paths empty. The ``showSuggestions`` switch still warns
    through the middleware: the warning must reach the model on the first
    supported request and stay suppressed after that delivery; the persisted
    set lives beside the index cache, independent of the disabled display.
    The ``enabled`` switch is untouched instead (D2): no delivery, no file.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    settings = settings_for(endpoints, tmp_path / "suggestions.json")
    settings[switch] = False
    settings["suggestionsPath"] = ""
    settings["cachePath"] = ""
    callback = _callback_for(plugin_module, settings)

    if switch == "enabled":
        assert callback(request=chat_payload(), **DISPATCH_KWARGS) is None
        assert not (tmp_path / ".cache" / "hermes-skill-router" / "config-switches.json").exists()
        return

    first = callback(request=chat_payload(), **DISPATCH_KWARGS)
    assert _warning_messages(first), "the first supported request delivers"
    assert (tmp_path / ".cache" / "hermes-skill-router" / "config-switches.json").exists()
    assert callback(request=first["request"], **DISPATCH_KWARGS) is None, (
        "after delivery the same both-empty request stays suppressed"
    )


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
