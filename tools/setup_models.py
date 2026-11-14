"""Interactive setup for the two bundled model servers (contract C1) plus the setup check (C2).

Run it from the checkout root::

    python3 tools/setup_models.py

It asks where the model files should live, and the first network action happens only after
that directory is confirmed: it downloads the files the bundled servers load, checks the two
pinned weight files against ``service/model-digests.txt`` (a mismatch warns loudly and needs
a second confirmation), and then starts both servers. It does not supervise them and never
runs on its own; it prints the manage, start and stop commands for this host and exits 0.

On Linux it installs two systemd user units: the checked-in
``service/hermes-skill-router-embed.service`` adapted to this checkout and this download
directory, plus the decider unit generated from the same shape. macOS and Windows start each
bundled server script with ``python3`` instead. Nothing in the plugin imports this module, so
``hermes plugins install`` can never run it by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hermes_skill_router.config import DEFAULTS, Config  # noqa: E402

#: The two pinned files are keyed in this file; the checkout test pins its shape separately.
PINS_PATH = REPO_ROOT / "service" / "model-digests.txt"

#: The checked-in unit the Linux install is built from.
EMBED_UNIT_PATH = REPO_ROOT / "service" / "hermes-skill-router-embed.service"

#: The base URLs whose servers this repository bundles. A config pointing at one of these is
#: checked with its health route and its model id; anything else is a custom endpoint and only
#: owes a correct live answer.
BUNDLED_BASES = {
    "embed": DEFAULTS["embedBaseUrl"],
    "rerank": DEFAULTS["rerankBaseUrl"],
}

#: How many CPU threads both bundled servers use, matching their own defaults.
THREADS = "8"


@dataclass(frozen=True)
class ModelSpec:
    """One bundled model: what to download, what to pin, and how the server starts."""

    role: str
    repo: str
    directory: str
    files: Sequence[str]
    pinned: Sequence[str]
    unit: str
    module: str
    log_name: str


MODELS: Sequence[ModelSpec] = (
    ModelSpec(
        role="embedder",
        repo="BAAI/bge-m3",
        directory="bge-m3",
        # The bundled embeddings server reads these three files only: the ONNX graph, its
        # external weights and the tokenizer. The rest of the 4.5 GB repository is not loaded.
        files=("onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"),
        pinned=("onnx/model.onnx_data",),
        unit="hermes-skill-router-embed.service",
        module="service.embed_server",
        log_name="embed",
    ),
    ModelSpec(
        role="reranker",
        repo="tasksource/tasksource-decider-nano",
        directory="tasksource-decider-nano",
        # decider.py and its two config files are the snapshot's inference code; AutoTokenizer
        # reads tokenizer.json and tokenizer_config.json from the same directory.
        files=(
            "decider.py",
            "config.json",
            "decider_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        pinned=("model.safetensors",),
        unit="hermes-skill-router-decider.service",
        module="service.decider_rerank_server",
        log_name="decider",
    ),
)


class SetupError(Exception):
    """One setup step failed. The message names that step so the exit code is diagnosable."""


@dataclass(frozen=True)
class PinMismatch:
    """One pinned file whose actual digest is not the expected one."""

    repo: str
    path: str
    expected: Optional[str]
    actual: Optional[str]


# ---------------------------------------------------------------------------------------------
# C2: is the installed configuration ready?


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow a redirect, so a probe never posts to a host it was not asked about."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 - see class
        return None


#: One opener for every probe; the bundled clients use the same redirect policy.
_OPENER = urllib.request.build_opener(_RefuseRedirect())


def _json_request(url: str, *, body: Optional[Mapping[str, Any]], timeout: float) -> tuple[bool, Any]:
    """Answer ``(True, decoded_json)`` for a parseable 2xx response, ``(False, None)`` otherwise.

    Never raises: a closed port, a timeout, any non-2xx status (including a refused redirect),
    and a body that is not JSON are all unready values.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    method = "GET" if body is None else "POST"
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read()
    except Exception:  # noqa: BLE001 - a probe failure is a readiness value, not an exception
        return False, None
    try:
        return True, json.loads(raw)
    except ValueError:
        return False, None


def _endpoint_ready(
    base: str,
    model: str,
    path: str,
    payload: Mapping[str, Any],
    timeout: float,
    bundled_base: str,
) -> bool:
    """Whether *base* is ready: a live answer, and for a bundled base its health rule first."""
    root = base.rstrip("/")
    if root == bundled_base.rstrip("/"):
        healthy, health = _json_request(root + "/health", body=None, timeout=timeout)
        if not healthy or not isinstance(health, dict):
            return False
        if health.get("status") != "ok" or health.get("model") != model:
            return False
    answered, _body = _json_request(root + path, body=payload, timeout=timeout)
    return answered


def check_setup_state(config: Config) -> tuple[bool, str]:
    """Return ``(True, "")`` when both configured endpoints answer, else the failing setting name.

    A custom endpoint needs only a correct answer to the plugin's real embeddings or rerank
    request: any parseable 2xx body passes, with no health route and no bundled model id. A
    bundled endpoint (its base URL is one of the plugin defaults) must first answer
    ``GET <base>/health`` with ``status == "ok"`` and the configured model, then answer the
    same tiny live request. Never raises; unready is a value. The failing name is the settings
    key (``embedBaseUrl`` or ``rerankBaseUrl``) the setup-needed line should name.
    """
    embed_base = config.embed_base_url
    if not _endpoint_ready(
        embed_base,
        config.embed_model_id,
        "/v1/embeddings",
        {"model": config.embed_model_id, "input": ["setup check"]},
        config.embed_timeout_ms / 1000.0,
        BUNDLED_BASES["embed"],
    ):
        return False, "embedBaseUrl"
    rerank_base = config.rerank_base_url
    if not _endpoint_ready(
        rerank_base,
        config.rerank_model,
        "/v1/rerank",
        {
            "model": config.rerank_model,
            "query": "setup check",
            "documents": ["setup check"],
            "top_n": 1,
        },
        config.rerank_timeout_ms / 1000.0,
        BUNDLED_BASES["rerank"],
    ):
        return False, "rerankBaseUrl"
    return True, ""


# ---------------------------------------------------------------------------------------------
# C1: ask first, download, verify the pins, install and start


def _ask(prompt: str) -> str:
    """Read one answer from the interactive terminal."""
    return input(prompt)


def _answer(prompt: str) -> str:
    """Read one answer, turning the end of input into a named setup step failure."""
    try:
        return _ask(prompt)
    except EOFError as exc:
        raise SetupError("input: no answer on stdin; the setup stopped") from exc


def _confirm(question: str, *, default: bool) -> bool:
    """Ask a yes/no question; an empty answer takes *default*."""
    suffix = " [Y/n] " if default else " [y/N] "
    reply = _answer(question + suffix).strip().lower()
    if not reply:
        return default
    return reply in ("y", "yes")


def _default_models_dir() -> Path:
    """The download directory offered when the user presses Enter."""
    return Path("~/.cache/hermes-skill-router/models").expanduser()


def _chosen_dir(answer: str, default: Path) -> Path:
    """The directory the user typed, or *default* for an empty answer. Returns an absolute path."""
    text = answer.strip()
    chosen = Path(os.path.expanduser(text)) if text else default
    return chosen.resolve()


def _fetch(model: ModelSpec, destination: Path) -> None:
    """Download *model*'s files into *destination*; a failure names the download step."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SetupError(
            "download: huggingface_hub is not installed; "
            "install it with 'python3 -m pip install huggingface_hub'"
        ) from error
    try:
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model.repo,
            local_dir=str(destination),
            allow_patterns=list(model.files),
        )
    except Exception as error:  # noqa: BLE001 - every failure is one named download step
        raise SetupError(
            f"download {model.repo}: {type(error).__name__}: {error}"
        ) from error


def _model_present(model: ModelSpec, destination: Path) -> bool:
    """Whether every file the bundled server loads is already on disk."""
    return all((destination / path).is_file() for path in model.files)


def _download_models(directory: Path) -> None:
    """Download the two models into *directory*, skipping a model whose files are all present."""
    for model in MODELS:
        destination = directory / model.directory
        if _model_present(model, destination):
            print(f"{model.repo} is already downloaded in {destination}")
            continue
        print(f"Downloading {model.repo} into {destination} ...")
        _fetch(model, destination)
        print(f"Downloaded {model.repo}")


def _sha256(path: Path) -> str:
    """Return the sha256 of *path*, read in one-megabyte chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pins(path: Path) -> dict[tuple[str, str], str]:
    """Map ``(repo, path)`` to its pinned sha256, refusing any data line of another shape."""
    pins: dict[tuple[str, str], str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        fields = entry.split()
        if len(fields) != 3:
            raise SetupError(
                f"pin check: {path.name}:{number} must be '<sha256> <repo> <path>', "
                f"got {entry!r}"
            )
        digest, repo, repo_path = fields
        pins[(repo, repo_path)] = digest
    return pins


def _pin_mismatches(
    pins: Mapping[tuple[str, str], str], directory: Path
) -> list[PinMismatch]:
    """Every pinned file under *directory* whose digest is missing, unreadable or wrong."""
    mismatches: list[PinMismatch] = []
    for model in MODELS:
        for repo_path in model.pinned:
            expected = pins.get((model.repo, repo_path))
            local = directory / model.directory / repo_path
            actual: Optional[str]
            try:
                actual = _sha256(local) if local.is_file() else None
            except OSError:
                actual = None
            if expected is None or expected != actual:
                mismatches.append(
                    PinMismatch(repo=model.repo, path=repo_path, expected=expected, actual=actual)
                )
    return mismatches


def _check_pins(directory: Path) -> None:
    """Warn loudly and ask before continuing when a pinned file does not match its digest."""
    try:
        pins = _read_pins(PINS_PATH)
    except OSError as error:
        raise SetupError(f"pin check: cannot read {PINS_PATH}: {error}") from error
    mismatches = _pin_mismatches(pins, directory)
    if not mismatches:
        print("The downloaded model files match the pinned digests in service/model-digests.txt.")
        return
    print()
    print("=" * 78)
    print("WARNING: PIN MISMATCH. The downloaded model files do not match service/model-digests.txt")
    print("=" * 78)
    for item in mismatches:
        print(f"  {item.repo} {item.path}")
        print(f"    pinned: {item.expected or 'no pin for this file'}")
        print(f"    actual: {item.actual or 'missing or unreadable'}")
    print("A mismatch means a corrupted or replaced download, or an upstream republish.")
    print("=" * 78)
    if not _confirm("Continue with these files anyway?", default=False):
        raise SetupError(
            "pin check: stopped because the downloaded files do not match service/model-digests.txt"
        )
    print("Continuing with the mismatched files because you confirmed.")


def _config_dir() -> Path:
    """The user config root, respecting ``XDG_CONFIG_HOME``."""
    raw = os.environ.get("XDG_CONFIG_HOME")
    return Path(os.path.expanduser(raw)) if raw else Path("~/.config").expanduser()


def _log_dir() -> Path:
    """Where the directly started servers write their logs."""
    return Path("~/.cache/hermes-skill-router").expanduser()


def _default_port(key: str) -> int:
    """The port the plugin default base URL for *key* names."""
    port = urlsplit(DEFAULTS[key]).port
    if port is None:
        raise SetupError(f"install: {key} has no port in its default value")
    return port


def _server_environment(model: ModelSpec, directory: Path) -> dict[str, str]:
    """The environment the bundled server reads its settings from."""
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": THREADS,
            "OMP_WAIT_POLICY": "PASSIVE",
        }
    )
    if model.role == "embedder":
        environment.update(
            {
                "HSR_MODEL_DIR": str(directory / model.directory),
                "HSR_PORT": str(_default_port("embedBaseUrl")),
                "HSR_MODEL_ID": str(DEFAULTS["embedModelId"]),
                "HSR_MAX_LENGTH": "512",
            }
        )
    else:
        environment.update(
            {
                "DECIDER_DIR": str(directory / model.directory),
                "DECIDER_THREADS": THREADS,
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
    return environment


def _replace_prefix(lines: Sequence[str], prefix: str, replacement: str) -> tuple[list[str], bool]:
    """Replace every line starting with *prefix*; report whether any line was found."""
    found = False
    replaced: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            replaced.append(replacement)
            found = True
        else:
            replaced.append(line)
    return replaced, found


def _embed_unit_text(directory: Path) -> str:
    """The checked-in embed unit adapted to this checkout and this model directory."""
    lines = EMBED_UNIT_PATH.read_text(encoding="utf-8").splitlines()
    lines, has_workdir = _replace_prefix(lines, "WorkingDirectory=", f"WorkingDirectory={REPO_ROOT}")
    lines, has_exec = _replace_prefix(
        lines, "ExecStart=", f"ExecStart={sys.executable} -m service.embed_server"
    )
    lines, has_docs = _replace_prefix(
        lines,
        "Documentation=file:",
        f"Documentation=file:{REPO_ROOT / 'service' / 'embed_server.py'}",
    )
    lines, has_model_dir = _replace_prefix(
        lines,
        "Environment=HSR_MODEL_DIR=",
        f"Environment=HSR_MODEL_DIR={directory / 'bge-m3'}",
    )
    if not (has_workdir and has_exec and has_docs and has_model_dir):
        raise SetupError(
            "install: the bundled unit at service/hermes-skill-router-embed.service lost one "
            "of its WorkingDirectory, ExecStart, Documentation or HSR_MODEL_DIR lines"
        )
    return "\n".join(lines) + "\n"


def _decider_unit_text(directory: Path) -> str:
    """The decider unit, in the same shape as the checked-in embed unit."""
    return f"""\
[Unit]
Description=Hermes skill-router decider rerank server (Cohere-compatible, loopback, CPU)
Documentation=file:{REPO_ROOT / 'service' / 'decider_rerank_server.py'}
After=network.target

[Service]
Type=simple
WorkingDirectory={REPO_ROOT}
Environment=DECIDER_DIR={directory / 'tasksource-decider-nano'}
Environment=DECIDER_THREADS={THREADS}
Environment=CUDA_VISIBLE_DEVICES=
Environment=HF_HUB_OFFLINE=1
Environment=OMP_NUM_THREADS={THREADS}
Environment=OMP_WAIT_POLICY=PASSIVE
Environment=PYTHONUNBUFFERED=1
ExecStart={sys.executable} -m service.decider_rerank_server
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""


def _write_file(path: Path, text: str) -> None:
    """Write *text* beside *path* and rename it into place.

    The rename replaces a symlink instead of writing through it, so an existing unit link into
    this checkout is never edited by a setup run.
    """
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _run(command: Sequence[str], step: str) -> subprocess.CompletedProcess:
    """Run one management command; a non-zero exit or a missing program names *step*."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SetupError(f"{step}: {type(error).__name__}: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        last = detail.splitlines()[-1] if detail else f"exit code {result.returncode}"
        raise SetupError(f"{step}: {last}")
    return result


def _install_systemd_units(directory: Path) -> None:
    """Install and start the two systemd user units on Linux."""
    unit_dir = _config_dir() / "systemd" / "user"
    embed_unit = unit_dir / MODELS[0].unit
    decider_unit = unit_dir / MODELS[1].unit
    _write_file(embed_unit, _embed_unit_text(directory))
    _write_file(decider_unit, _decider_unit_text(directory))
    print(f"Installed {embed_unit}")
    print(f"Installed {decider_unit}")
    _run(["systemctl", "--user", "daemon-reload"], "install: systemctl --user daemon-reload")
    for model in MODELS:
        _run(
            ["systemctl", "--user", "enable", "--now", model.unit],
            f"start {model.unit}",
        )
        print(f"Started {model.unit}")


def _spawn(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_handle: Any,
) -> subprocess.Popen:
    """Start one bundled server detached from this process, writing its output to *log_handle*."""
    options: dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {}
    return subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(environment),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        **options,
    )


def _start_directly(directory: Path) -> None:
    """Start both bundled server scripts with the running interpreter, for macOS and Windows."""
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        log_path = log_dir / f"{model.log_name}.log"
        handle = log_path.open("ab")
        try:
            process = _spawn(
                [sys.executable, "-m", model.module],
                REPO_ROOT,
                _server_environment(model, directory),
                handle,
            )
        except OSError as error:
            raise SetupError(f"start {model.role}: {error}") from error
        finally:
            handle.close()
        print(f"Started the {model.role} as pid {process.pid}; log: {log_path}")


def _install_and_start(directory: Path) -> None:
    """Install and start both servers with this host's mechanism."""
    platform = _platform()
    if platform.startswith("linux"):
        _install_systemd_units(directory)
    else:
        _start_directly(directory)


def _platform() -> str:
    """The host platform string, a seam so the printed commands are testable."""
    return sys.platform


def _start_commands(platform: str, directory: Path) -> list[str]:
    """The commands that start each bundled server script directly, for macOS and Windows."""
    pairs = ((MODELS[0], "HSR_MODEL_DIR"), (MODELS[1], "DECIDER_DIR"))
    if platform.startswith("win32"):
        return [
            f"Set-Location '{REPO_ROOT}'; $env:{name}='{directory / model.directory}'; "
            f"& '{sys.executable}' -m {model.module}"
            for model, name in pairs
        ]
    return [
        f"cd '{REPO_ROOT}' && {name}='{directory / model.directory}' '{sys.executable}' "
        f"-m {model.module}"
        for model, name in pairs
    ]


def _printed_commands(platform: str, directory: Path) -> dict[str, list[str]]:
    """The manage, start and stop commands for this host, as the user would type them."""
    units = " ".join(model.unit for model in MODELS)
    if platform.startswith("linux"):
        return {
            "Manage": [
                f"systemctl --user status {units}",
                f"journalctl --user -u {MODELS[0].unit} -u {MODELS[1].unit} -f",
            ],
            "Start": [f"systemctl --user start {units}"],
            "Stop": [f"systemctl --user stop {units}"],
        }
    if platform.startswith("win32"):
        patterns = [f"'*{model.module}*'" for model in MODELS]
        return {
            "Manage": [
                f"Get-CimInstance Win32_Process | Where-Object CommandLine -like {pattern} "
                "| Select-Object ProcessId, Name"
                for pattern in patterns
            ],
            "Start": _start_commands(platform, directory),
            "Stop": [
                f"Get-CimInstance Win32_Process | Where-Object CommandLine -like {pattern} "
                "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
                for pattern in patterns
            ],
        }
    patterns = [f'"{model.module}"' for model in MODELS]
    return {
        "Manage": [f"pgrep -fl {pattern}" for pattern in patterns],
        "Start": _start_commands(platform, directory),
        "Stop": [f"pkill -f {pattern}" for pattern in patterns],
    }


def _print_commands(platform: str, directory: Path) -> None:
    """Print the manage, start and stop commands for the two servers."""
    for title, commands in _printed_commands(platform, directory).items():
        print(f"\n{title} both servers:")
        for command in commands:
            print(f"  {command}")


def _intro() -> None:
    """Print what the script is about to do, before it asks anything."""
    print("Hermes skill router: model setup")
    print("This sets up the two model servers the plugin defaults point at:")
    for model in MODELS:
        base = BUNDLED_BASES["embed"] if model.role == "embedder" else BUNDLED_BASES["rerank"]
        print(f"  - {model.role}: {model.repo} for {base}")
    print("Nothing is downloaded until you confirm the download directory.")


def _setup() -> int:
    """Ask, download, verify, install, start; return the exit code."""
    _intro()
    default_dir = _default_models_dir()
    directory = _chosen_dir(_answer(f"Download directory [{default_dir}]: "), default_dir)
    if not _confirm(f"Download and install both models into {directory}?", default=True):
        print("Nothing was downloaded. Run this script again when you are ready.")
        return 0
    _download_models(directory)
    _check_pins(directory)
    _install_and_start(directory)
    _print_commands(_platform(), directory)
    print("\nSetup finished. The servers load their models in the background; check them with:")
    print(f"  curl -s {BUNDLED_BASES['embed']}/health")
    print(f"  curl -s {BUNDLED_BASES['rerank']}/health")
    return 0


def main() -> int:
    """Run the interactive setup; a failed step prints its name and exits non-zero."""
    try:
        return _setup()
    except SetupError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
