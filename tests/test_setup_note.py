"""Contract C2 wired: a rewrite turn names the failing setting on the display record.

The setup check lives in ``hermes_skill_router.setup_check`` so the rewrite can
write its result through the display record without importing the setup script.
The tests here drive the real HTTP clients against loopback fakes so the rank
failure and the readiness probe share one endpoint shape, and the same record
is rendered by both real surfaces through Node.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fake_endpoint import FakeEndpoint, Request, Response  # noqa: E402
from hermes_skill_router import embed as embed_module  # noqa: E402
from hermes_skill_router import rerank as rerank_module  # noqa: E402
from hermes_skill_router import router as router_module  # noqa: E402
from hermes_skill_router.display import read  # noqa: E402
from hermes_skill_router.library import Index  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_display import PAYLOAD  # noqa: E402

NODE = shutil.which("node")

INDEX = Index(
    names=["alpha", "beta"],
    descriptions=["Alpha skill", "Beta skill"],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    dim=2,
    fingerprint="test-fingerprint",
)

DESKTOP_SDK_STUB = """
export const atom = initial => {
  let value = initial
  return {
    get: () => value,
    set: next => { value = typeof next === 'function' ? next(value) : next }
  }
}
export const useValue = store => store.get()
export const PANES_AREA = 'panes'
export const DisclosureCaret = 'DisclosureCaret'
export const useQuery = options => globalThis.__q ?? {}
export const queryClient = { invalidateQueries: async options => {} }
"""

JSX_STUB = """
export const Fragment = 'Fragment'
export const jsx = (type, props) => ({ props: props ?? {}, type })
export const jsxs = jsx
"""

SDK_PACKAGE_STUB = json.dumps({"name": "@hermes/plugin-sdk", "type": "module", "main": "./index.mjs"})
REACT_PACKAGE_STUB = json.dumps(
    {
        "name": "react",
        "type": "module",
        "exports": {".": "./index.mjs", "./jsx-runtime": "./jsx-runtime.mjs"},
    }
)
REACT_INDEX_STUB = """
export const useState = initial => [typeof initial === 'function' ? initial() : initial, () => {}]
export const useEffect = () => {}
"""


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    from hermes_skill_router import search as search_module

    router_module._SETUP_CACHE.clear()
    router_module._LOGGED.clear()
    search_module.clear_dumps()
    embed_module.HEALTH.reset()
    rerank_module.HEALTH.reset()
    monkeypatch.setattr(router_module, "skill_path", lambda name: f"/skills/{name}/SKILL.md")
    try:
        yield
    finally:
        router_module._SETUP_CACHE.clear()
        embed_module.HEALTH.reset()
        rerank_module.HEALTH.reset()


def _embed_responder(path: str):
    def responder(request: Request) -> Response:
        if request.path == path:
            body = request.json()
            count = len(body.get("input", [])) or 1
            return Response(
                body={
                    "model": body.get("model", "my-embed"),
                    "data": [
                        {"index": i, "embedding": [1.0, 0.0]} for i in range(count)
                    ],
                }
            )
        return Response(status=404, body={"error": "not found"})

    return responder


def _rerank_responder(path: str, *, fail: bool):
    def responder(request: Request) -> Response:
        if request.path != path:
            return Response(status=404, body={"error": "not found"})
        if fail:
            return Response(status=500, body={"error": "broken"})
        body = request.json()
        documents = body.get("documents", [])
        return Response(
            body={
                "results": [
                    {"index": i, "relevance_score": 0.9 if i == 0 else 0.2}
                    for i in range(len(documents))
                ]
            }
        )

    return responder


def _config(tmp_path, embed: FakeEndpoint, rerank: FakeEndpoint, **overrides):
    from hermes_skill_router.config import load

    settings = {
        "embedBaseUrl": embed.base_url,
        "embedPath": "/custom/embeddings",
        "embedModelId": "my-embed",
        "rerankBaseUrl": rerank.base_url,
        "rerankPath": "/custom/rerank",
        "rerankModel": "my-rerank",
        "cachePath": "",
        "suggestionsPath": str(tmp_path / "suggestions.json"),
        "healthCheckMs": 0,
    }
    settings.update(overrides)
    return load(settings)


def _turn(cfg, **kwargs):
    return rewrite(
        PAYLOAD,
        cfg=cfg,
        index=INDEX,
        session_id="session-1",
        turn_id="turn-1",
        notify=lambda _turn_id: None,
        **kwargs,
    )


def _both_surfaces(tmp_path: Path, record: dict) -> dict:
    """Render one record through both shipped summary functions in Node."""
    assert NODE is not None, "node is required to load the shipped display halves"
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    results: dict[str, str] = {}
    # Desktop half.
    desktop_dir = tmp_path / "desktop-surface"
    for name, text in {
        "plugin.mjs": (ROOT / "desktop" / "plugin.js").read_text(encoding="utf-8"),
        "node_modules/@hermes/plugin-sdk/index.mjs": DESKTOP_SDK_STUB,
        "node_modules/@hermes/plugin-sdk/package.json": SDK_PACKAGE_STUB,
        "node_modules/react/index.mjs": REACT_INDEX_STUB,
        "node_modules/react/jsx-runtime.mjs": JSX_STUB,
        "node_modules/react/package.json": REACT_PACKAGE_STUB,
    }.items():
        target = desktop_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    desktop_driver = desktop_dir / "driver.mjs"
    desktop_driver.write_text(
        "import { summary } from './plugin.mjs'\n"
        "import { readFileSync } from 'node:fs'\n"
        "const record = JSON.parse(readFileSync(process.argv[2], 'utf8'))\n"
        "console.log(JSON.stringify(summary(record)))\n",
        encoding="utf-8",
    )
    desktop_run = subprocess.run(
        [NODE, str(desktop_driver), str(record_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(desktop_dir),
    )
    assert desktop_run.returncode == 0, desktop_run.stderr
    results["desktop"] = json.loads(desktop_run.stdout.strip().splitlines()[-1])["note"]
    # TUI half.
    tui_dir = tmp_path / "tui-surface"
    tui_dir.mkdir(parents=True, exist_ok=True)
    (tui_dir / "skill-router.mjs").write_text(
        (ROOT / "tui" / "skill-router.mjs").read_text(encoding="utf-8"), encoding="utf-8"
    )
    tui_driver = tui_dir / "driver.mjs"
    tui_driver.write_text(
        "import { summary } from './skill-router.mjs'\n"
        "import { readFileSync } from 'node:fs'\n"
        "const record = JSON.parse(readFileSync(process.argv[2], 'utf8'))\n"
        "console.log(JSON.stringify(summary(record)))\n",
        encoding="utf-8",
    )
    tui_run = subprocess.run(
        [NODE, str(tui_driver), str(record_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tui_dir),
    )
    assert tui_run.returncode == 0, tui_run.stderr
    results["tui"] = json.loads(tui_run.stdout.strip().splitlines()[-1])["note"]
    return results


def test_an_unready_rerank_endpoint_names_the_setting_on_both_surfaces(tmp_path):
    embed = FakeEndpoint(_embed_responder("/custom/embeddings"))
    rerank = FakeEndpoint(_rerank_responder("/custom/rerank", fail=True))
    try:
        cfg = _config(tmp_path, embed, rerank)
        result = _turn(cfg)
        assert result is None, "a rerank failure leaves the request unchanged"
        stored = read(cfg)
        assert stored["status"] == "inert"
        note = stored["note"]
        assert "rerankBaseUrl" in note, f"note named no failing setting: {note!r}"
        assert note != router_module.ENDPOINT_NOTE, (
            "the wired line must name the failing setting, not the generic line"
        )
        # The embed half must have passed at the custom path: if the probe used the
        # hardcoded path it would have named embedBaseUrl instead.
        assert "embedBaseUrl" not in note
        assert {request.path for request in embed.requests} == {"/custom/embeddings"}
        assert {request.path for request in rerank.requests} == {"/custom/rerank"}

        rendered = _both_surfaces(tmp_path, stored)
        assert "rerankBaseUrl" in rendered["desktop"]
        assert "rerankBaseUrl" in rendered["tui"]
        assert rendered["desktop"] == note
        assert rendered["tui"] == note
    finally:
        embed.close()
        rerank.close()


def test_a_dead_endpoint_turn_stays_within_the_routing_budget(tmp_path):
    """C1: the setup probe after a rank failure must not outlive the routing budget.

    Both endpoints answer after 0.8 s and the embed one fails, so rank burns 0.8 s
    of a 1.0 s budget. Pre-fix the probe then re-probes both endpoints at full
    timeouts (1.6 s) and the turn takes ~2.4 s; post-fix the probe is capped by
    the 0.2 s left and the turn stays under the budget plus a small slack.
    """
    import time

    def embed_responder(request: Request) -> Response:
        return Response(status=500, body={"error": "broken"}, delay=0.8)

    def rerank_responder(request: Request) -> Response:
        return Response(body={"results": []}, delay=0.8)

    embed = FakeEndpoint(embed_responder)
    rerank = FakeEndpoint(rerank_responder)
    try:
        cfg = _config(
            tmp_path,
            embed,
            rerank,
            embedTimeoutMs=5000,
            rerankTimeoutMs=5000,
            routingTimeoutMs=1000,
        )
        started = time.perf_counter()
        result = _turn(cfg)
        elapsed = time.perf_counter() - started
        assert result is None, "a dead endpoint leaves the request unchanged"
        assert read(cfg)["status"] == "inert"
        assert elapsed >= 0.7, f"the delayed endpoints must actually delay: {elapsed:.3f}s"
        assert elapsed < 1.5, (
            "the turn must stay within the 1.0s routing budget plus slack, "
            f"took {elapsed:.3f}s"
        )
    finally:
        embed.close()
        rerank.close()


def test_a_custom_endpoint_failure_tells_the_user_to_repair_the_service(tmp_path):
    embed = FakeEndpoint(_embed_responder("/custom/embeddings"))
    rerank = FakeEndpoint(_rerank_responder("/custom/rerank", fail=True))
    try:
        cfg = _config(tmp_path, embed, rerank)
        result = _turn(cfg)
        assert result is None
        note = read(cfg)["note"]
        assert "rerankBaseUrl" in note
        assert "repair" in note.lower(), (
            f"the custom note must point at the configured service: {note!r}"
        )
        assert "setup script" not in note, (
            f"the custom note must not send the user to the script: {note!r}"
        )
    finally:
        embed.close()
        rerank.close()


def test_the_bundled_setup_note_keeps_the_setup_script_line():
    from hermes_skill_router.setup_check import setup_note

    assert setup_note("embedBaseUrl", bundled=True) == (
        "Model setup needed: embedBaseUrl is not ready; run the model setup script and reload."
    )
    assert setup_note("rerankBaseUrl", bundled=True) == (
        "Model setup needed: rerankBaseUrl is not ready; run the model setup script and reload."
    )


def test_a_bundled_endpoint_failure_keeps_the_setup_script_line(tmp_path, monkeypatch):
    from hermes_skill_router import setup_check

    embed_responder = _embed_responder("/custom/embeddings")
    rerank_responder = _rerank_responder("/custom/rerank", fail=True)

    def embed(request: Request) -> Response:
        if request.path == "/health":
            return Response(body={"status": "ok", "model": "my-embed"})
        return embed_responder(request)

    def rerank(request: Request) -> Response:
        if request.path == "/health":
            return Response(status=404, body={"error": "not found"})
        return rerank_responder(request)

    embed = FakeEndpoint(embed)
    rerank = FakeEndpoint(rerank)
    try:
        monkeypatch.setitem(setup_check.BUNDLED_BASES, "embed", embed.base_url)
        monkeypatch.setitem(setup_check.BUNDLED_BASES, "rerank", rerank.base_url)
        cfg = _config(tmp_path, embed, rerank)
        result = _turn(cfg)
        assert result is None
        note = read(cfg)["note"]
        assert note == (
            "Model setup needed: rerankBaseUrl is not ready; "
            "run the model setup script and reload."
        )
    finally:
        embed.close()
        rerank.close()


def test_changing_the_health_window_reprobes_instead_of_serving_the_old_check(
    tmp_path, monkeypatch
):
    """X2: healthCheckMs is part of the setup-check cache key."""
    from hermes_skill_router import setup_check

    embed = FakeEndpoint(_embed_responder("/custom/embeddings"))
    rerank = FakeEndpoint(_rerank_responder("/custom/rerank", fail=True))
    calls: list[int] = []
    real_check = setup_check.check_setup_state

    def counting(config, **kwargs):
        calls.append(1)
        return real_check(config, **kwargs)

    monkeypatch.setattr(router_module, "check_setup_state", counting)

    def turn(cfg, session: str):
        return rewrite(
            PAYLOAD,
            cfg=cfg,
            index=INDEX,
            session_id=session,
            turn_id=f"t-{session}",
            notify=lambda _turn_id: None,
        )

    try:
        cfg = _config(tmp_path, embed, rerank, healthCheckMs=60000)
        assert turn(cfg, "s-1") is None
        assert len(calls) == 1
        # A fresh session per turn: each turn is the first failure in its window,
        # so each reaches the failure-note probe instead of the dumped-window path.
        assert turn(cfg, "s-2") is None
        assert len(calls) == 1, "a second turn inside the window must serve the cache"
        narrower = _config(tmp_path, embed, rerank, healthCheckMs=30000)
        assert turn(narrower, "s-3") is None
        assert len(calls) == 2, "changing the health window must reprobe"
    finally:
        embed.close()
        rerank.close()


def test_a_custom_path_endpoint_that_is_live_passes_the_wired_check(tmp_path):
    embed = FakeEndpoint(_embed_responder("/custom/embeddings"))
    rerank = FakeEndpoint(_rerank_responder("/custom/rerank", fail=False))
    try:
        cfg = _config(tmp_path, embed, rerank, fusionMode="cosine")
        result = _turn(cfg)
        assert result is not None, "both custom-path endpoints answer, so the turn routes"
        stored = read(cfg)
        assert stored["status"] == "routed"
        assert stored["note"] == "", (
            f"a healthy custom-path endpoint must pass the wired check, got {stored['note']!r}"
        )
    finally:
        embed.close()
        rerank.close()


def test_a_custom_path_embed_endpoint_passes_its_half_of_the_check(tmp_path):
    # Directly pin C2's path rule: healthy custom paths return ready, and the probe
    # never visits the hardcoded paths.
    from hermes_skill_router.setup_check import check_setup_state

    embed = FakeEndpoint(_embed_responder("/custom/embeddings"))
    rerank = FakeEndpoint(_rerank_responder("/custom/rerank", fail=False))
    try:
        cfg = _config(tmp_path, embed, rerank)
        assert check_setup_state(cfg) == (True, "")
        assert [request.path for request in embed.requests] == ["/custom/embeddings"]
        assert [request.path for request in rerank.requests] == ["/custom/rerank"]
    finally:
        embed.close()
        rerank.close()


def test_a_fake_transport_failure_stays_generic_and_touches_no_socket(tmp_path, monkeypatch):
    from hermes_skill_router.setup_check import check_setup_state as real_check

    def boom(config):
        raise AssertionError("the check must not run for a fake transport")

    monkeypatch.setattr(router_module, "check_setup_state", boom)
    # Build a config on default endpoints without starting any fake; the failing fake
    # reranker below must still yield the generic line with no probe.
    from hermes_skill_router.config import load

    cfg = load(
        {
            "cachePath": "",
            "suggestionsPath": str(tmp_path / "suggestions.json"),
            "healthCheckMs": 0,
        }
    )
    result = rewrite(
        PAYLOAD,
        cfg=cfg,
        index=INDEX,
        embed=lambda config, texts: [[1.0, 0.0] for _ in texts],
        rerank=lambda config, query, documents: None,
        session_id="s",
        turn_id="t",
        notify=lambda _turn_id: None,
    )
    assert result is None
    assert read(cfg)["note"] == router_module.ENDPOINT_NOTE
    assert real_check is not None
