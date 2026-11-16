"""Contract C4 (ticket 04): curator default untouched, delta_curation narrows both surfaces.

Seeded T0/T1/T2 fixture: one changed bundled skill, one changed agent-created
skill, one disabled skill, one pinned skill, one never-active skill created
before and one after each watermark. The user candidate list is written in the
upstream row format with upstream eligibility already applied (bundled and
disabled rows absent), exactly as agent/curator.py renders it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router.curator import EMPTY_CANDIDATES_TEXT, curator_rewrite  # noqa: E402
from hermes_skill_router.router import rewrite  # noqa: E402
from tests.test_rewrite import FakeEmbedder, FakeReranker, _content, make_config  # noqa: E402

T0 = "2026-11-10T00:00:00+00:00"
T1 = "2026-11-12T00:00:00+00:00"
T2 = "2026-11-15T00:00:00+00:00"

CATALOG_ROWS = (
    ("agent-changed", "Changed agent skill"),
    ("bundled-changed", "Changed bundled skill"),
    ("disabled-changed", "Changed disabled skill"),
    ("pinned-changed", "Changed pinned skill"),
    ("never-old", "Never-active old skill"),
    ("never-new", "Never-active new skill"),
    ("old-agent", "Unchanged agent skill"),
    ("cron-agent", "Cron-referenced agent skill"),
)

USAGE = {
    "agent-changed": {
        "use_count": 3, "view_count": 1, "patch_count": 0,
        "last_used_at": "2026-11-13T00:00:00+00:00", "last_viewed_at": None,
        "last_patched_at": None, "created_at": "2026-10-01T00:00:00+00:00",
        "pinned": False,
    },
    "bundled-changed": {
        "use_count": 2, "view_count": 0, "patch_count": 0,
        "last_used_at": None, "last_viewed_at": "2026-11-13T00:00:00+00:00",
        "last_patched_at": None, "created_at": "2026-01-01T00:00:00+00:00",
        "pinned": False,
    },
    "disabled-changed": {
        "use_count": 1, "view_count": 0, "patch_count": 1,
        "last_used_at": None, "last_viewed_at": None,
        "last_patched_at": "2026-11-13T00:00:00+00:00",
        "created_at": "2026-10-01T00:00:00+00:00", "pinned": False,
    },
    "pinned-changed": {
        "use_count": 5, "view_count": 2, "patch_count": 0,
        "last_used_at": "2026-11-13T00:00:00+00:00", "last_viewed_at": None,
        "last_patched_at": None, "created_at": "2026-10-01T00:00:00+00:00",
        "pinned": True,
    },
    "never-old": {
        "use_count": 0, "view_count": 0, "patch_count": 0,
        "last_used_at": None, "last_viewed_at": None, "last_patched_at": None,
        "created_at": "2026-11-01T00:00:00+00:00", "pinned": False,
    },
    "never-new": {
        "use_count": 0, "view_count": 0, "patch_count": 0,
        "last_used_at": None, "last_viewed_at": None, "last_patched_at": None,
        "created_at": "2026-11-13T00:00:00+00:00", "pinned": False,
    },
    "old-agent": {
        "use_count": 4, "view_count": 2, "patch_count": 0,
        "last_used_at": "2026-11-11T00:00:00+00:00", "last_viewed_at": None,
        "last_patched_at": None, "created_at": "2026-10-01T00:00:00+00:00",
        "pinned": False,
    },
    "cron-agent": {
        "use_count": 2, "view_count": 0, "patch_count": 0,
        "last_used_at": "2026-11-13T00:00:00+00:00", "last_viewed_at": None,
        "last_patched_at": None, "created_at": "2026-10-01T00:00:00+00:00",
        "pinned": False,
    },
}

# Upstream-eligible rows only (agent-created, not bundled, not disabled),
# in the upstream row format; the pinned row is present upstream and the
# delta predicate must remove it.
UPSTREAM_ROWS = (
    "- agent-changed  provenance=agent  state=active  pinned=no  cron=no  activity=4  use=3  view=1  patches=0  last_activity=2026-11-13T00:00:00+00:00",
    "- cron-agent  provenance=agent  state=active  pinned=no  cron=yes  activity=2  use=2  view=0  patches=0  last_activity=2026-11-13T00:00:00+00:00",
    "- never-new  provenance=agent  state=active  pinned=no  cron=no  activity=0  use=0  view=0  patches=0  last_activity=never",
    "- old-agent  provenance=agent  state=active  pinned=no  cron=no  activity=6  use=4  view=2  patches=0  last_activity=2026-11-11T00:00:00+00:00",
    "- pinned-changed  provenance=agent  state=active  pinned=yes  cron=no  activity=7  use=5  view=2  patches=0  last_activity=2026-11-13T00:00:00+00:00",
)

CURATOR_PROMPT = (
    "You are running as Hermes' background skill CURATOR. This is a test pass.\n\n"
    f"Agent-created skills ({len(UPSTREAM_ROWS)}):\n" + "\n".join(UPSTREAM_ROWS)
)


def _payload(user=None):
    return {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": _content(CATALOG_ROWS)},
            {"role": "user", "content": user if user is not None else CURATOR_PROMPT},
        ],
    }


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """An isolated Hermes home with seeded .curator_state and .usage.json."""
    root = tmp_path / "home"
    skills = root / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    watermark = tmp_path / "curator-watermark.json"
    monkeypatch.setenv("HERMES_SKILL_ROUTER_CURATOR_WATERMARK", str(watermark))
    return root, skills, watermark


def _set_last_run(skills, value):
    (skills / ".curator_state").write_text(
        json.dumps({"last_run_at": value, "run_count": 3}), encoding="utf-8"
    )


def _set_usage(skills, usage):
    (skills / ".usage.json").write_text(json.dumps(usage), encoding="utf-8")


def _seed_watermark(path, watermark, answered, candidates=None):
    path.write_text(
        json.dumps({"watermark": watermark, "answered_run": answered, "candidates": candidates}),
        encoding="utf-8",
    )


def _delta_cfg(tmp_path):
    return make_config(tmp_path, curation="delta_curation")


def _rewrite_cfg(cfg, user=None):
    return rewrite(
        _payload(user),
        cfg=cfg,
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        platform="curator",
        session_id="curator-session",
    )


def test_default_curation_returns_the_payload_unmodified(home, tmp_path):
    """Default mode: full catalog untouched even with a fake ranking present."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)

    payload = _payload()
    result = rewrite(
        payload,
        cfg=make_config(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        platform="curator",
        session_id="curator-session",
    )
    assert result is None
    assert not watermark.exists() or json.loads(watermark.read_text())["watermark"] == T0
    assert not (tmp_path / "suggestions.json").exists(), "no record is written"


def test_first_sight_fails_open_and_seeds(home, tmp_path):
    """No watermark file: fail open, and seed the watermark with the current run."""
    _root, skills, watermark = home
    _set_last_run(skills, T1)
    _set_usage(skills, USAGE)
    assert not watermark.exists()

    assert _rewrite_cfg(_delta_cfg(tmp_path)) is None
    record = json.loads(watermark.read_text())
    assert record["watermark"] == T1
    assert record["answered_run"] == T1 and record["candidates"] is None


def test_second_run_narrows_both_surfaces(home, tmp_path):
    """After seeding, the next run narrows the system block and the user list."""
    _root, skills, watermark = home
    _set_last_run(skills, T1)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T1, T1)

    # The prune-only bump between runs: last_run_at moves with no plugin
    # involvement, so the next delta is computed against the stale boundary.
    _set_last_run(skills, T2)
    result = _rewrite_cfg(_delta_cfg(tmp_path))
    assert result is not None
    messages = result["request"]["messages"]

    system = messages[0]["content"]
    assert "## Skills" not in system and "<available_skills>" not in system
    block = messages[-1]["content"]
    for name in ("agent-changed", "bundled-changed", "disabled-changed", "never-new", "cron-agent"):
        assert f"- {name}:" in block, name
    for name in ("pinned-changed", "never-old", "old-agent"):
        assert f"- {name}:" not in block, name

    user = messages[1]["content"]
    assert "Agent-created skills (3):" in user
    for kept in ("- agent-changed", "- cron-agent", "- never-new"):
        assert kept in user, kept
    for dropped in ("- old-agent", "- pinned-changed"):
        assert dropped not in user, dropped
    # Bundled and disabled rows never enter the user candidate list, even
    # though they are in the delta set and the system block.
    assert "bundled-changed" not in user
    assert "disabled-changed" not in user
    assert not (tmp_path / "suggestions.json").exists(), "the curator writes no record"

    record = json.loads(watermark.read_text())
    assert record["watermark"] == T2 and record["answered_run"] == T2
    assert set(record["candidates"]) == {
        "agent-changed", "bundled-changed", "disabled-changed", "cron-agent", "never-new",
    }


def test_same_run_repeats_the_answer(home, tmp_path):
    """A later middleware call in the same run repeats the cached answer."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)

    first = _rewrite_cfg(_delta_cfg(tmp_path))
    second = _rewrite_cfg(_delta_cfg(tmp_path))
    assert first is not None and second is not None
    assert first["request"]["messages"] == second["request"]["messages"]


def test_dry_run_fails_open_without_advancing(home, tmp_path):
    """Dry-run banner: byte-identical fail open, watermark untouched."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)
    before = watermark.read_text()
    banner = "DRY-RUN — REPORT ONLY. DO NOT MUTATE THE SKILL LIBRARY.\n\n"
    payload = _payload(user=banner + CURATOR_PROMPT)

    result = rewrite(
        payload,
        cfg=_delta_cfg(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        platform="curator",
        session_id="curator-session",
    )
    assert result is None
    assert watermark.read_text() == before


def test_corrupt_or_missing_state_fails_open(home, tmp_path):
    _root, skills, watermark = home
    cfg = _delta_cfg(tmp_path)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)

    # Unparseable last_run_at.
    _set_last_run(skills, "not-a-timestamp")
    assert _rewrite_cfg(cfg) is None
    # Missing .curator_state.
    (skills / ".curator_state").unlink()
    assert _rewrite_cfg(cfg) is None
    # Corrupt .usage.json.
    _set_last_run(skills, T2)
    (skills / ".usage.json").write_text("{broken", encoding="utf-8")
    assert _rewrite_cfg(cfg) is None
    # Missing .usage.json.
    (skills / ".usage.json").unlink()
    assert _rewrite_cfg(cfg) is None


def test_empty_delta_fails_open(home, tmp_path):
    """Nothing newer than the watermark: byte-identical fail open."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, {name: {**record, "last_used_at": None, "last_viewed_at": None,
                               "last_patched_at": None, "created_at": T0}
                        for name, record in USAGE.items()})
    _seed_watermark(watermark, T0, T1)
    assert _rewrite_cfg(_delta_cfg(tmp_path)) is None


def test_missing_platform_with_curator_shape_narrows(home, tmp_path):
    """Prompt shape alone (no platform kwarg) still narrows."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)

    result = rewrite(
        _payload(),
        cfg=_delta_cfg(tmp_path),
        embed=FakeEmbedder(),
        rerank=FakeReranker(),
        session_id="curator-session",
    )
    assert result is not None
    assert "- agent-changed:" in result["request"]["messages"][-1]["content"]


def test_unrecognized_curation_value_falls_back_to_default(tmp_path):
    """An unknown value parses to default_curation with a note (unknown-safe)."""
    cfg = make_config(tmp_path, curation="delta_something")
    assert cfg.curation_mode == "default_curation"
    assert any("curation" in note for note in cfg.notes)


def test_manifest_declares_the_curation_value(tmp_path):
    import yaml

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    schema = manifest["config_schema"]["curation"]
    assert schema["default"] == "default_curation"
    assert set(schema["values"]) == {"default_curation", "delta_curation"}
    cfg = make_config(tmp_path, curation="delta_curation")
    assert cfg.curation_mode == "delta_curation"


def test_curator_rewrite_never_touches_upstream_state_files(home, tmp_path):
    """The plugin reads but never writes .curator_state or .usage.json."""
    _root, skills, watermark = home
    _set_last_run(skills, T2)
    _set_usage(skills, USAGE)
    _seed_watermark(watermark, T0, T1)
    state_before = (skills / ".curator_state").read_text()
    usage_before = (skills / ".usage.json").read_text()

    assert _rewrite_cfg(_delta_cfg(tmp_path)) is not None
    assert (skills / ".curator_state").read_text() == state_before
    assert (skills / ".usage.json").read_text() == usage_before