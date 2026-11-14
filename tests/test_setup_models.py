"""Ticket 01: the setup script's ask-first flow, its pin check, and the setup-state check.

Every HTTP test drives the loopback fake from ``tests/fake_endpoint.py``, so the setup check
is exercised against real sockets with no network and no model. The script tests replace the
four seams that would otherwise leave the process: the terminal (``_ask``), the downloader
(``_fetch``), the management commands (``_run``) and the detached spawn (``_spawn``), with a
temporary HOME and config root, so no test writes a real unit, log or model.
"""

from __future__ import annotations

import hashlib
import socket
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import setup_models as setup  # noqa: E402
from fake_endpoint import FakeEndpoint, Request, Response  # noqa: E402
from hermes_skill_router.config import DEFAULTS, load  # noqa: E402

README = ROOT / "README.md"
AFTER_INSTALL = ROOT / "after-install.md"
COMMAND = "python3 tools/setup_models.py"


@pytest.fixture()
def fake():
    endpoint = FakeEndpoint()
    try:
        yield endpoint
    finally:
        endpoint.close()


def _config(port: int, *, embed_model: str = "my-embed", rerank_model: str = "my-rerank"):
    """A plugin config pointing both endpoints at the fake's loopback port."""
    return load(
        {
            "embedBaseUrl": f"http://127.0.0.1:{port}",
            "rerankBaseUrl": f"http://127.0.0.1:{port}",
            "embedModelId": embed_model,
            "rerankModel": rerank_model,
        }
    )


def _bundled(monkeypatch: pytest.MonkeyPatch, embed: FakeEndpoint, rerank: FakeEndpoint):
    """A config whose bases count as bundled, with the fakes standing in for them."""
    monkeypatch.setitem(setup.BUNDLED_BASES, "embed", embed.base_url)
    monkeypatch.setitem(setup.BUNDLED_BASES, "rerank", rerank.base_url)
    return load(
        {
            "embedBaseUrl": embed.base_url,
            "rerankBaseUrl": rerank.base_url,
            "embedModelId": "my-embed",
            "rerankModel": "my-rerank",
        }
    )


# ---------------------------------------------------------------------------------------------
# C2: the setup-state check


def test_the_bundled_bases_are_the_plugin_defaults() -> None:
    assert setup.BUNDLED_BASES == {
        "embed": DEFAULTS["embedBaseUrl"],
        "rerank": DEFAULTS["rerankBaseUrl"],
    }


def test_a_custom_endpoint_passes_with_one_live_answer_and_no_health_route(fake) -> None:
    def responder(request: Request) -> Response:
        if request.path == "/v1/embeddings":
            return Response(body={"data": []})
        if request.path == "/v1/rerank":
            return Response(body={"results": []})
        return Response(status=404, body={"error": "not found"})

    fake.answer_with(responder)
    ready, name = setup.check_setup_state(_config(fake.port))
    assert (ready, name) == (True, "")
    assert [(request.method, request.path) for request in fake.requests] == [
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/rerank"),
    ]
    assert [request.json()["model"] for request in fake.requests] == ["my-embed", "my-rerank"]


def test_a_failing_custom_embeddings_endpoint_names_the_setting(fake) -> None:
    fake.answer_with(
        lambda request: Response(status=503, body={"error": "still loading"})
        if request.path == "/v1/embeddings"
        else Response(body={"results": []})
    )
    ready, name = setup.check_setup_state(_config(fake.port))
    assert ready is False
    assert name == "embedBaseUrl"
    assert len(fake.requests) == 1, "the rerank probe must not run after the embed probe failed"


def test_a_failing_custom_rerank_endpoint_names_the_setting(fake) -> None:
    fake.answer_with(
        lambda request: Response(body={"data": []})
        if request.path == "/v1/embeddings"
        else Response(status=500, body={"error": "broken"})
    )
    ready, name = setup.check_setup_state(_config(fake.port))
    assert (ready, name) == (False, "rerankBaseUrl")


def test_a_custom_2xx_that_does_not_parse_is_not_ready(fake) -> None:
    # A 2xx status alone is not an answer: the setup check requires the body to parse.
    fake.answer_with(lambda request: Response(body="this is not json"))
    ready, name = setup.check_setup_state(_config(fake.port))
    assert (ready, name) == (False, "embedBaseUrl")


@pytest.mark.parametrize(
    "health",
    [
        {"status": "loading"},
        {"status": "ok", "model": "another-model"},
        {"status": "ok"},
        "this is not json",
        {"status": "error", "model": "my-embed"},
    ],
)
def test_a_bundled_endpoint_must_clear_status_and_model_on_health(monkeypatch, health) -> None:
    embed = FakeEndpoint()
    rerank = FakeEndpoint()
    try:
        embed.answer(health)
        rerank.answer({"status": "ok", "model": "my-rerank"})
        config = _bundled(monkeypatch, embed, rerank)
        ready, name = setup.check_setup_state(config)
        assert (ready, name) == (False, "embedBaseUrl")
        assert [request.path for request in embed.requests] == ["/health"], (
            "the live embeddings request must not run when the health route is unready"
        )
        assert [request.path for request in rerank.requests] == [], (
            "the rerank endpoint must not be probed after the embed endpoint failed"
        )
    finally:
        embed.close()
        rerank.close()


def test_a_bundled_rerank_must_clear_its_own_health_route(monkeypatch) -> None:
    embed = FakeEndpoint()
    rerank = FakeEndpoint()
    try:
        embed.answer({"status": "ok", "model": "my-embed"})
        rerank.answer({"status": "ok", "model": "another-model"})
        config = _bundled(monkeypatch, embed, rerank)
        ready, name = setup.check_setup_state(config)
        assert (ready, name) == (False, "rerankBaseUrl")
        assert [request.path for request in rerank.requests] == ["/health"]
    finally:
        embed.close()
        rerank.close()


def test_bundled_endpoints_also_need_a_live_answer(monkeypatch) -> None:
    embed = FakeEndpoint()
    rerank = FakeEndpoint()
    try:
        embed.answer_with(
            lambda request: Response(body={"status": "ok", "model": "my-embed"})
            if request.path == "/health"
            else Response(body="not json")
        )
        rerank.answer({"status": "ok", "model": "my-rerank"})
        config = _bundled(monkeypatch, embed, rerank)
        ready, name = setup.check_setup_state(config)
        assert (ready, name) == (False, "embedBaseUrl")
        assert [request.path for request in embed.requests] == ["/health", "/v1/embeddings"]
    finally:
        embed.close()
        rerank.close()


def test_bundled_endpoints_pass_health_then_live_answer(monkeypatch) -> None:
    embed = FakeEndpoint()
    rerank = FakeEndpoint()
    try:
        embed.answer_with(
            lambda request: Response(body={"status": "ok", "model": "my-embed"})
            if request.path == "/health"
            else Response(body={"data": []})
        )
        rerank.answer_with(
            lambda request: Response(body={"status": "ok", "model": "my-rerank"})
            if request.path == "/health"
            else Response(body={"results": []})
        )
        config = _bundled(monkeypatch, embed, rerank)
        assert setup.check_setup_state(config) == (True, "")
        for endpoint in (embed, rerank):
            assert [request.method for request in endpoint.requests] == ["GET", "POST"]
            assert endpoint.requests[0].path == "/health"
            assert endpoint.requests[1].json()["model"] in ("my-embed", "my-rerank")
    finally:
        embed.close()
        rerank.close()


def test_a_dead_endpoint_is_an_unready_value_not_an_exception() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = _config(port)
    ready, name = setup.check_setup_state(config)
    assert (ready, name) == (False, "embedBaseUrl")


# ---------------------------------------------------------------------------------------------
# Pins and the files the servers load


def _pins_file(path: Path, entries: Iterable[tuple[str, str, str]]) -> Path:
    lines = ["# test pins"]
    lines.extend(f"{digest} {repo} {repo_path}" for digest, repo, repo_path in entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_pinned_files(directory: Path) -> None:
    """Create the two pinned files under *directory* with content the tests can hash."""
    weights = {
        directory / "bge-m3" / "onnx" / "model.onnx_data": b"embedding weights",
        directory / "tasksource-decider-nano" / "model.safetensors": b"decider weights",
    }
    for path, content in weights.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _digest_of(directory: Path, relative: str) -> str:
    return hashlib.sha256((directory / relative).read_bytes()).hexdigest()


def test_the_setup_downloads_and_pins_the_files_the_two_servers_load() -> None:
    pins = setup._read_pins(setup.PINS_PATH)
    pinned = {(model.repo, path) for model in setup.MODELS for path in model.pinned}
    assert pinned == set(pins), "the pin file and the setup script must agree on both files"
    assert {model.role for model in setup.MODELS} == {"embedder", "reranker"}
    for model in setup.MODELS:
        assert set(model.pinned) <= set(model.files), (
            f"{model.repo}: every pinned file must also be downloaded"
        )
    embedder = next(model for model in setup.MODELS if model.role == "embedder")
    assert {"onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"} <= set(embedder.files)
    reranker = next(model for model in setup.MODELS if model.role == "reranker")
    assert {"decider.py", "model.safetensors", "config.json"} <= set(reranker.files)


def test_the_pin_parser_refuses_a_line_of_another_shape(tmp_path) -> None:
    pins = tmp_path / "pins.txt"
    pins.write_text("# comment\nabc BAAI/bge-m3 onnx/model.onnx_data extra\n", encoding="utf-8")
    with pytest.raises(setup.SetupError, match="must be"):
        setup._read_pins(pins)


def test_matching_digests_pass_the_pin_check(tmp_path, monkeypatch, capsys) -> None:
    directory = tmp_path / "models"
    _write_pinned_files(directory)
    pins = _pins_file(
        tmp_path / "pins.txt",
        [
            (
                _digest_of(directory, "bge-m3/onnx/model.onnx_data"),
                "BAAI/bge-m3",
                "onnx/model.onnx_data",
            ),
            (
                _digest_of(directory, "tasksource-decider-nano/model.safetensors"),
                "tasksource/tasksource-decider-nano",
                "model.safetensors",
            ),
        ],
    )
    monkeypatch.setattr(setup, "PINS_PATH", pins)
    setup._check_pins(directory)
    output = capsys.readouterr().out
    assert "match the pinned digests" in output


def test_a_pin_mismatch_warns_loudly_and_a_refusal_stops(tmp_path, monkeypatch, capsys) -> None:
    directory = tmp_path / "models"
    _write_pinned_files(directory)
    digest = _digest_of(directory, "bge-m3/onnx/model.onnx_data")
    wrong = "0" * 64
    pins = _pins_file(
        tmp_path / "pins.txt",
        [
            (wrong, "BAAI/bge-m3", "onnx/model.onnx_data"),
            (
                _digest_of(directory, "tasksource-decider-nano/model.safetensors"),
                "tasksource/tasksource-decider-nano",
                "model.safetensors",
            ),
        ],
    )
    monkeypatch.setattr(setup, "PINS_PATH", pins)
    monkeypatch.setattr(setup, "_ask", lambda prompt: "n")
    with pytest.raises(setup.SetupError, match="pin check"):
        setup._check_pins(directory)
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "PIN MISMATCH" in output
    assert wrong in output, "the warning must show the pinned digest"
    assert digest in output, "the warning must show the actual digest"


def test_a_pin_mismatch_continues_after_a_confirmation(tmp_path, monkeypatch, capsys) -> None:
    directory = tmp_path / "models"
    _write_pinned_files(directory)
    wrong = "0" * 64
    pins = _pins_file(
        tmp_path / "pins.txt",
        [
            (wrong, "BAAI/bge-m3", "onnx/model.onnx_data"),
            (
                _digest_of(directory, "tasksource-decider-nano/model.safetensors"),
                "tasksource/tasksource-decider-nano",
                "model.safetensors",
            ),
        ],
    )
    monkeypatch.setattr(setup, "PINS_PATH", pins)
    monkeypatch.setattr(setup, "_ask", lambda prompt: "y")
    setup._check_pins(directory)
    assert "because you confirmed" in capsys.readouterr().out


def test_a_missing_pinned_file_is_a_mismatch(tmp_path) -> None:
    pins = {("BAAI/bge-m3", "onnx/model.onnx_data"): "0" * 64}
    mismatch = setup._pin_mismatches(pins, tmp_path)
    assert [(item.repo, item.path, item.actual) for item in mismatch] == [
        ("BAAI/bge-m3", "onnx/model.onnx_data", None),
        ("tasksource/tasksource-decider-nano", "model.safetensors", None),
    ]


def test_an_existing_complete_download_is_not_fetched_again(monkeypatch, tmp_path, capsys) -> None:
    directory = tmp_path / "models"
    for model in setup.MODELS:
        for relative in model.files:
            local = directory / model.directory / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"x")
    fetched: list[str] = []
    monkeypatch.setattr(setup, "_fetch", lambda model, destination: fetched.append(model.repo))
    setup._download_models(directory)
    assert fetched == []
    assert capsys.readouterr().out.count("already downloaded") == 2


def test_the_download_wrapper_names_the_failing_repo(monkeypatch, tmp_path) -> None:
    pytest.importorskip("huggingface_hub")
    monkeypatch.setattr("huggingface_hub.snapshot_download", _raise_route_error)
    with pytest.raises(setup.SetupError, match="download BAAI/bge-m3"):
        setup._fetch(setup.MODELS[0], tmp_path)


def _raise_route_error(**kwargs: Any) -> None:
    raise RuntimeError("no route to the hub")


# ---------------------------------------------------------------------------------------------
# C1: the interactive flow


@pytest.fixture()
def clean_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return home, xdg


def _script(monkeypatch, answers: Iterable[str], events: list[tuple[str, str]] | None = None) -> None:
    queue = list(answers)

    def ask(prompt: str) -> str:
        if events is not None:
            events.append(("prompt", prompt))
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr(setup, "_ask", ask)


def _fake_run(commands: list[list[str]]):
    def run(command, step):
        commands.append(list(command))
        return None

    return run


def test_the_first_network_action_follows_the_confirmed_directory(clean_home, monkeypatch):
    home, _xdg = clean_home
    events: list[tuple[str, str]] = []
    _script(monkeypatch, ["", "y"], events)
    monkeypatch.setattr(
        setup, "_fetch", lambda model, destination: events.append(("fetch", model.repo))
    )
    monkeypatch.setattr(setup, "_check_pins", lambda directory: None)
    monkeypatch.setattr(setup, "_platform", lambda: "linux")
    monkeypatch.setattr(setup, "_run", _fake_run([]))
    assert setup.main() == 0
    kinds = [kind for kind, _payload in events]
    assert kinds == ["prompt", "prompt", "fetch", "fetch"], (
        "both downloads must follow the directory confirmation"
    )
    prompts = [payload for kind, payload in events if kind == "prompt"]
    assert "Download directory" in prompts[0]
    assert "Download and install both models into" in prompts[1]
    assert str(home / ".cache" / "hermes-skill-router" / "models") in prompts[1]


def test_answering_no_downloads_nothing(clean_home, monkeypatch, capsys):
    events: list[tuple[str, str]] = []
    _script(monkeypatch, ["", "n"], events)
    monkeypatch.setattr(
        setup, "_fetch", lambda model, destination: pytest.fail("network was used")
    )
    assert setup.main() == 0
    assert [kind for kind, _payload in events] == ["prompt", "prompt"]
    assert "Nothing was downloaded" in capsys.readouterr().out


def test_end_of_input_starts_nothing(clean_home, monkeypatch, capsys):
    _script(monkeypatch, [])
    assert setup.main() == 1
    assert "no answer on stdin" in capsys.readouterr().err


def test_a_download_failure_names_the_step_and_exits_non_zero(clean_home, monkeypatch, capsys):
    _script(monkeypatch, ["", "y"])

    def fail(model, destination):
        raise setup.SetupError(f"download {model.repo}: RuntimeError: no route to the hub")

    monkeypatch.setattr(setup, "_fetch", fail)
    assert setup.main() != 0
    assert "download BAAI/bge-m3" in capsys.readouterr().err


def test_linux_installs_and_starts_the_two_units(clean_home, monkeypatch, capsys):
    home, xdg = clean_home
    _script(monkeypatch, ["", "y"])
    commands: list[list[str]] = []
    monkeypatch.setattr(setup, "_fetch", lambda model, destination: None)
    monkeypatch.setattr(setup, "_check_pins", lambda directory: None)
    monkeypatch.setattr(setup, "_run", _fake_run(commands))
    monkeypatch.setattr(setup, "_platform", lambda: "linux")
    assert setup.main() == 0
    output = capsys.readouterr().out
    assert "systemctl --user status hermes-skill-router-embed.service" in output
    assert "systemctl --user start hermes-skill-router-embed.service" in output
    assert "systemctl --user stop hermes-skill-router-embed.service" in output
    assert "hermes-skill-router-decider.service" in output

    unit_dir = xdg / "systemd" / "user"
    embed = (unit_dir / "hermes-skill-router-embed.service").read_text(encoding="utf-8")
    assert f"Environment=HSR_MODEL_DIR={home / '.cache/hermes-skill-router/models/bge-m3'}" in embed
    assert f"WorkingDirectory={ROOT}" in embed
    assert f"ExecStart={sys.executable} -m service.embed_server" in embed
    for author_path in (
        "/home/wilder/git-repos/skill-router-and-jev",
        "/home/wilder/.scripts",
    ):
        assert author_path not in embed, "the bundled unit's author paths must be replaced"
    assert str(home / ".cache/hermes-skill-router/models/bge-m3") in embed
    decider = (unit_dir / "hermes-skill-router-decider.service").read_text(encoding="utf-8")
    assert f"Environment=DECIDER_DIR={home / '.cache/hermes-skill-router/models/tasksource-decider-nano'}" in decider
    assert f"ExecStart={sys.executable} -m service.decider_rerank_server" in decider
    assert ["systemctl", "--user", "daemon-reload"] in commands
    assert ["systemctl", "--user", "enable", "--now", "hermes-skill-router-embed.service"] in commands
    assert ["systemctl", "--user", "enable", "--now", "hermes-skill-router-decider.service"] in commands


def test_installing_the_unit_replaces_a_symlink_without_editing_its_target(clean_home, monkeypatch, tmp_path):
    home, xdg = clean_home
    unit_dir = xdg / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    target = tmp_path / "linked-unit.service"
    target.write_text("the checkout's own unit\n", encoding="utf-8")
    link = unit_dir / "hermes-skill-router-embed.service"
    link.symlink_to(target)
    monkeypatch.setattr(setup, "_run", _fake_run([]))
    setup._install_systemd_units(home / ".cache" / "hermes-skill-router" / "models")
    assert target.read_text(encoding="utf-8") == "the checkout's own unit\n", (
        "the install must not write through a symlink into the repository"
    )
    assert not link.is_symlink()
    assert "Environment=HSR_MODEL_DIR=" in link.read_text(encoding="utf-8")


def test_macos_starts_both_scripts_with_the_interpreter(clean_home, monkeypatch, capsys):
    home, _xdg = clean_home
    _script(monkeypatch, ["", "y"])
    spawned: list[tuple[list[str], dict[str, str]]] = []

    class Process:
        pid = 4242

    def spawn(command, cwd, environment, handle):
        spawned.append((list(command), dict(environment)))
        return Process()

    monkeypatch.setattr(setup, "_spawn", spawn)
    monkeypatch.setattr(setup, "_fetch", lambda model, destination: None)
    monkeypatch.setattr(setup, "_check_pins", lambda directory: None)
    monkeypatch.setattr(setup, "_platform", lambda: "darwin")
    assert setup.main() == 0
    output = capsys.readouterr().out
    assert len(spawned) == 2
    assert spawned[0][0] == [sys.executable, "-m", "service.embed_server"]
    assert spawned[1][0] == [sys.executable, "-m", "service.decider_rerank_server"]
    assert spawned[0][1]["HSR_MODEL_DIR"] == str(home / ".cache/hermes-skill-router/models/bge-m3")
    assert spawned[1][1]["DECIDER_DIR"] == str(
        home / ".cache/hermes-skill-router/models/tasksource-decider-nano"
    )
    assert f"Started the embedder as pid 4242" in output
    assert "Start both servers" in output
    assert "Stop both servers" in output
    assert "pkill -f" in output


def test_windows_prints_powershell_manage_start_and_stop_commands(clean_home, monkeypatch, capsys):
    _script(monkeypatch, ["", "y"])

    class Process:
        pid = 5252

    monkeypatch.setattr(setup, "_spawn", lambda command, cwd, environment, handle: Process())
    monkeypatch.setattr(setup, "_fetch", lambda model, destination: None)
    monkeypatch.setattr(setup, "_check_pins", lambda directory: None)
    monkeypatch.setattr(setup, "_platform", lambda: "win32")
    assert setup.main() == 0
    output = capsys.readouterr().out
    assert "Get-CimInstance Win32_Process" in output
    assert "Stop-Process -Id $_.ProcessId -Force" in output
    assert "service.embed_server" in output
    assert "service.decider_rerank_server" in output
    assert "Set-Location" in output
    assert "Start both servers" in output
    assert "Stop both servers" in output


# ---------------------------------------------------------------------------------------------
# Documentation and the no-auto-run invariant


def test_the_readme_and_after_install_name_the_setup_command() -> None:
    assert COMMAND in README.read_text(encoding="utf-8"), (
        "README.md does not name the setup command"
    )
    assert AFTER_INSTALL.is_file(), "after-install.md is missing"
    assert COMMAND in AFTER_INSTALL.read_text(encoding="utf-8"), (
        "after-install.md does not name the setup command"
    )


def test_after_install_says_the_script_is_not_automatic_and_uses_no_em_dash() -> None:
    prose = " ".join(AFTER_INSTALL.read_text(encoding="utf-8").split())
    assert "never runs automatically" in prose or "does not run automatically" in prose, (
        "after-install.md does not say the setup script is started by the user"
    )
    assert "\u2014" not in prose, "after-install.md contains an em dash"


def test_the_readme_no_longer_claims_no_reranker_is_bundled() -> None:
    prose = " ".join(README.read_text(encoding="utf-8").split())
    assert "No reranker is bundled" not in prose, (
        "README.md still says no reranker is bundled, which ticket 02 changed"
    )
    assert "decider" in prose, "README.md does not name the bundled decider rerank server"


def test_nothing_in_the_plugin_imports_or_runs_the_setup_script() -> None:
    for path in (ROOT / "__init__.py", ROOT / "plugin.yaml", *sorted((ROOT / "hermes_skill_router").glob("*.py"))):
        text = path.read_text(encoding="utf-8")
        assert "setup_models" not in text, f"{path.name} references the setup script, so it could run it"
