"""Ticket 03: no-record switch warnings on three channels.

With ``enabled: false`` or ``showSuggestions: false`` in effect, ``config.load``
emits one terminal line (stderr), one injected context message (a ``system``
message dict built by :func:`switch_warning_messages`, never a suggestions-block
row), and one host-logger event, each naming the switch and its effect.
Anti-spam keys on the offending ``(switch, value)`` fingerprint in the
persisted trap set, one seen set per switch: an unchanged value warns once and
then stays silent across turns and a restart, a changed value warns once more,
and fixing one switch drops only that switch's fingerprints so re-breaking the
same value warns again while the other switch stays suppressed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hermes_skill_router import config as config_module  # noqa: E402
from hermes_skill_router.config import load, switch_warning_messages  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(config_module, "_LOGGED_NOTES", set())


def _path(tmp_path):
    return str(tmp_path / "suggestions.json")


def _switch_notes(cfg):
    return tuple(
        note
        for note in cfg.notes
        if note.startswith(("enabled: ", "showSuggestions: ")) and " is off; " in note
    )


def _logger_lines(caplog):
    cfg_logger = logging.getLogger("hermes_skill_router.config")
    prefix = "hermes-skill-router config: "
    lines = []
    for record in caplog.records:
        if record.name != cfg_logger.name:
            continue
        message = record.getMessage()
        if " is off; " not in message:
            continue
        lines.append(message[len(prefix):] if message.startswith(prefix) else message)
    return lines


def _traps_file(tmp_path):
    return tmp_path / "config-traps.json"


def _seen(tmp_path):
    path = _traps_file(tmp_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("seen") if isinstance(data, dict) else data
    return {tuple(entry) for entry in entries}


def test_enabled_false_emits_three_channels_naming_switch_and_effect(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"enabled": False, "suggestionsPath": _path(tmp_path)})

    notes = _switch_notes(cfg)
    assert len(notes) == 1, f"enabled warning must be produced, got {cfg.notes!r}"
    line = notes[0]
    assert "enabled" in line
    assert "routing" in line and ("switch" in line.lower() or "off" in line.lower())

    # Terminal line: one stderr line identical to the warning line.
    err = capsys.readouterr().err
    assert [row for row in err.splitlines() if "enabled" in row] == [line]

    # Host-logger event: one config warning naming the switch.
    assert _logger_lines(caplog) == [line]

    # Injected context message: a standalone system message, never a block row.
    messages = switch_warning_messages(cfg)
    assert messages == ({"role": "system", "content": line},)
    assert not line.startswith("Skills routed for this request")
    assert "Suggestions:" not in line
    from hermes_skill_router.inject import suggested_names

    assert suggested_names(line) == []


def test_show_suggestions_false_emits_three_channels_naming_display_loss(
    tmp_path, capsys, caplog
):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"showSuggestions": False, "suggestionsPath": _path(tmp_path)})

    notes = _switch_notes(cfg)
    assert len(notes) == 1, f"showSuggestions warning must be produced, got {cfg.notes!r}"
    line = notes[0]
    assert "showSuggestions" in line
    assert "display" in line.lower() or "record" in line.lower() or "pane" in line.lower()

    assert [row for row in capsys.readouterr().err.splitlines() if row] == [line]
    assert _logger_lines(caplog) == [line]
    assert switch_warning_messages(cfg) == ({"role": "system", "content": line},)


def test_unchanged_value_warns_once_then_stays_silent_across_turns_and_restart(
    tmp_path, capsys, caplog
):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = load({"enabled": False, "showSuggestions": False, "suggestionsPath": path})
        assert len(_switch_notes(first)) == 2
        assert {note.split(":", 1)[0] for note in _switch_notes(first)} == {
            "enabled",
            "showSuggestions",
        }
        terminal_first = capsys.readouterr().err.splitlines()
        assert len([row for row in terminal_first if row]) == 2
        assert len(_logger_lines(caplog)) == 2

        caplog.clear()
        for _ in range(3):
            again = load({"enabled": False, "showSuggestions": False, "suggestionsPath": path})
            assert _switch_notes(again) == ()
            assert switch_warning_messages(again) == ()
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == [], "repeated turns must not spam the logger"

        # A restart clears only in-process memory; the persisted set still suppresses.
        config_module._LOGGED_NOTES.clear()
        restarted = load({"enabled": False, "showSuggestions": False, "suggestionsPath": path})
        assert _switch_notes(restarted) == ()
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []


def test_each_switch_warns_independently(tmp_path, capsys, caplog):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({"enabled": False, "suggestionsPath": path})
        assert [note.split(":", 1)[0] for note in _switch_notes(cfg)] == ["enabled"]
        cfg = load({"showSuggestions": False, "suggestionsPath": path})
        # The fixed switch (enabled defaults back to True) clears only its own set,
        # so the other switch still warns for its own fingerprint.
        assert [note.split(":", 1)[0] for note in _switch_notes(cfg)] == ["showSuggestions"]


def test_changed_offending_value_warns_once_more(tmp_path, capsys, caplog):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = load({"enabled": False, "suggestionsPath": path})
        assert len(_switch_notes(first)) == 1
        changed = load({"enabled": "false", "suggestionsPath": path})
        assert len(_switch_notes(changed)) == 1, (
            "a different spelling of the offending value is a new fingerprint"
        )
        assert _switch_notes(changed) != _switch_notes(first)
        again = load({"enabled": "false", "suggestionsPath": path})
        assert _switch_notes(again) == ()


def test_fixing_one_switch_drops_only_its_fingerprints(tmp_path, capsys, caplog):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = load(
            {"enabled": False, "showSuggestions": False, "suggestionsPath": path}
        )
        assert len(_switch_notes(both)) == 2
        seen = _seen(tmp_path)
        assert {pair[0] for pair in seen} == {"enabled", "showSuggestions"}

        fixed = load({"enabled": True, "showSuggestions": False, "suggestionsPath": path})
        assert _switch_notes(fixed) == ()
        seen = _seen(tmp_path)
        assert {pair[0] for pair in seen} == {"showSuggestions"}, (
            "fixing enabled must drop only its own fingerprints"
        )

        capsys.readouterr()
        caplog.clear()
        rebroken = load({"enabled": False, "showSuggestions": False, "suggestionsPath": path})
        notes = _switch_notes(rebroken)
        assert [note.split(":", 1)[0] for note in notes] == ["enabled"], (
            "re-breaking the same value warns again while the other stays suppressed"
        )
        assert len(_logger_lines(caplog)) == 1

        # Fixing the other switch now drops only its set.
        load({"enabled": False, "showSuggestions": True, "suggestionsPath": path})
        assert {pair[0] for pair in _seen(tmp_path)} == {"enabled"}


def test_both_off_then_fixing_one_rebreak_warns_while_other_stays_silent(
    tmp_path, capsys, caplog
):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        load({"enabled": False, "showSuggestions": False, "suggestionsPath": path})
        load({"enabled": True, "showSuggestions": False, "suggestionsPath": path})
        capsys.readouterr()
        caplog.clear()
        rebroken = load(
            {"enabled": False, "showSuggestions": False, "suggestionsPath": path}
        )
        assert [note.split(":", 1)[0] for note in _switch_notes(rebroken)] == ["enabled"]
        assert len(capsys.readouterr().err.splitlines()) == 1
        assert len(_logger_lines(caplog)) == 1
        assert switch_warning_messages(rebroken) == (
            {"role": "system", "content": _switch_notes(rebroken)[0]},
        )


def test_no_per_turn_spam_loop_with_repeated_turns(tmp_path, capsys, caplog):
    path = _path(tmp_path)
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        load({"showSuggestions": False, "suggestionsPath": path})
        capsys.readouterr()
        caplog.clear()
        for _ in range(5):
            load({"showSuggestions": False, "suggestionsPath": path})
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []


def test_healthy_load_creates_no_trap_file(tmp_path, capsys, caplog):
    load({"suggestionsPath": _path(tmp_path)})
    assert not _traps_file(tmp_path).exists()
    assert capsys.readouterr().err == ""
