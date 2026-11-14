"""Contract C3: the pinned model-file digests the setup script checks.

The file is data for ``tools/setup_models.py``: one sha256, one Hugging Face repo and
one path inside that repo per line. These tests pin the format and the two files it
must cover, and, when the downloaded model files are present on this host, prove the
pinned values are the digests of the real bytes rather than invented constants.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service import decider_rerank_server, embed_server  # noqa: E402

PINS_PATH = ROOT / "service" / "model-digests.txt"
DIGEST = re.compile(r"[0-9a-f]{64}")

#: Where this host keeps the two downloaded files. Each verification test skips
#: when its path is absent, so the suite still passes on a checkout with no weights.
BGE_M3_WEIGHTS = Path(
    "/home/wilder/git-repos/skill-router-and-jev/.scratch/hf_local/BAAI/bge-m3/onnx/model.onnx_data"
)
DECIDER_WEIGHTS = Path(
    "/home/wilder/.cache/huggingface/hub/models--tasksource--tasksource-decider-nano/"
    "snapshots/2c7c01ffe5aba74dd15968cf38b0fdc83d1f3810/model.safetensors"
)


def _pins() -> Iterator[Tuple[str, str, str]]:
    """Yield ``(digest, repo, path)`` for every data line, refusing any other shape."""
    for number, line in enumerate(PINS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        fields = entry.split()
        assert len(fields) == 3, f"line {number} must be '<sha256> <repo> <path>', got {entry!r}"
        digest, repo, path = fields
        assert DIGEST.fullmatch(digest), f"line {number} digest is not lowercase sha256: {digest!r}"
        assert repo.count("/") == 1 and not repo.startswith("/"), f"line {number} repo is not 'owner/name': {repo!r}"
        assert path and " " not in path, f"line {number} path is empty or has a space: {path!r}"
        yield digest, repo, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_the_file_parses_and_pins_exactly_the_two_model_files() -> None:
    pins = list(_pins())
    assert len(pins) == 2
    assert {(repo, path) for _digest, repo, path in pins} == {
        ("BAAI/bge-m3", "onnx/model.onnx_data"),
        ("tasksource/tasksource-decider-nano", "model.safetensors"),
    }
    assert len({digest for digest, _repo, _path in pins}) == 2, "each model file has its own digest"


def test_the_pins_cover_the_two_bundled_model_ids() -> None:
    repos = [repo for _digest, repo, _path in _pins()]
    assert any(embed_server.DEFAULT_MODEL_ID in repo for repo in repos)
    assert any(decider_rerank_server.MODEL_ID in repo for repo in repos)


def test_the_bge_m3_pin_is_the_digest_of_the_downloaded_weights() -> None:
    pin = next(_pins())
    if not BGE_M3_WEIGHTS.is_file():
        pytest.skip(f"no downloaded weights at {BGE_M3_WEIGHTS}")
    digest, repo, path = pin
    assert (repo, path) == ("BAAI/bge-m3", "onnx/model.onnx_data")
    assert _sha256(BGE_M3_WEIGHTS) == digest


def test_the_decider_pin_is_the_digest_of_the_downloaded_weights() -> None:
    pins = list(_pins())
    pin = next(entry for entry in pins if entry[1] == "tasksource/tasksource-decider-nano")
    if not DECIDER_WEIGHTS.is_file():
        pytest.skip(f"no downloaded weights at {DECIDER_WEIGHTS}")
    digest, repo, path = pin
    assert (repo, path) == ("tasksource/tasksource-decider-nano", "model.safetensors")
    assert _sha256(DECIDER_WEIGHTS) == digest


def test_the_pin_paths_are_the_files_the_bundled_servers_load() -> None:
    pins = {path for _digest, _repo, path in _pins()}
    # embed_server reads onnx/model.onnx and keeps the weights in onnx/model.onnx_data beside it.
    assert embed_server.ModelPaths("/models/bge-m3").onnx_model.endswith(os.path.join("onnx", "model.onnx"))
    assert "onnx/model.onnx_data" in pins
    # The decider snapshot's from_pretrained call reads model.safetensors from DECIDER_DIR.
    assert decider_rerank_server.DEFAULT_MODEL_DIR.endswith(decider_rerank_server.MODEL_ID)
    assert "model.safetensors" in pins
