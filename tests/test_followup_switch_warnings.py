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
from hermes_skill_router.config import (  # noqa: E402
    acknowledge_switch_deliveries,
    load,
    switch_warning_messages,
)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(config_module, "_LOGGED_NOTES", set())


def _path(tmp_path):
    return str(tmp_path / "suggestions.json")


def _cache(tmp_path):
    return str(tmp_path / "index.json")


def _load(tmp_path, settings):
    """Load *settings* with the test's isolated display and cache paths."""
    merged = {"suggestionsPath": _path(tmp_path), "cachePath": _cache(tmp_path)}
    merged.update(settings)
    return load(merged)


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


def _switches_file(tmp_path):
    return tmp_path / "config-switches.json"


def _seen(tmp_path):
    path = _switches_file(tmp_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("seen") if isinstance(data, dict) else data
    return {tuple(entry) for entry in entries}


def test_enabled_false_emits_three_channels_naming_switch_and_effect(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = _load(tmp_path, {"enabled": False})

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
        cfg = _load(tmp_path, {"showSuggestions": False})

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
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = _load(tmp_path, {"enabled": False, "showSuggestions": False})
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
            again = _load(tmp_path, {"enabled": False, "showSuggestions": False})
            assert _switch_notes(again) == ()
            # Round 3: a load alone no longer suppresses the injected channel;
            # it stays pending until a supported request actually delivers it.
            assert len(switch_warning_messages(again)) == 2
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == [], "repeated turns must not spam the logger"

        # Delivery on a supported request acknowledges the injected channel.
        acknowledge_switch_deliveries(
            [_switch_notes(first)[0], _switch_notes(first)[1]],
            _cache(tmp_path),
            first.switch_generations,
        )
        after_delivery = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert switch_warning_messages(after_delivery) == ()

        # A restart clears only in-process memory; the persisted set still suppresses.
        config_module._LOGGED_NOTES.clear()
        restarted = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert _switch_notes(restarted) == ()
        assert switch_warning_messages(restarted) == ()
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []


def test_each_switch_warns_independently(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = _load(tmp_path, {"enabled": False})
        assert [note.split(":", 1)[0] for note in _switch_notes(cfg)] == ["enabled"]
        cfg = _load(tmp_path, {"showSuggestions": False})
        # The fixed switch (enabled defaults back to True) clears only its own set,
        # so the other switch still warns for its own fingerprint.
        assert [note.split(":", 1)[0] for note in _switch_notes(cfg)] == ["showSuggestions"]


def test_changed_offending_value_warns_once_more(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = _load(tmp_path, {"enabled": False})
        assert len(_switch_notes(first)) == 1
        changed = _load(tmp_path, {"enabled": "false"})
        assert len(_switch_notes(changed)) == 1, (
            "a different spelling of the offending value is a new fingerprint"
        )
        assert _switch_notes(changed) != _switch_notes(first)
        again = _load(tmp_path, {"enabled": "false"})
        assert _switch_notes(again) == ()


def test_fixing_one_switch_drops_only_its_fingerprints(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert len(_switch_notes(both)) == 2
        seen = _seen(tmp_path)
        assert {pair[0] for pair in seen} == {"enabled", "showSuggestions"}

        fixed = _load(tmp_path, {"enabled": True, "showSuggestions": False})
        assert _switch_notes(fixed) == ()
        seen = _seen(tmp_path)
        assert {pair[0] for pair in seen} == {"showSuggestions"}, (
            "fixing enabled must drop only its own fingerprints"
        )

        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        notes = _switch_notes(rebroken)
        assert [note.split(":", 1)[0] for note in notes] == ["enabled"], (
            "re-breaking the same value warns again while the other stays suppressed"
        )
        assert len(_logger_lines(caplog)) == 1

        # Fixing the other switch now drops only its set.
        _load(tmp_path, {"enabled": False, "showSuggestions": True})
        assert {pair[0] for pair in _seen(tmp_path)} == {"enabled"}


def test_both_off_then_fixing_one_rebreak_warns_while_other_stays_silent(
    tmp_path, capsys, caplog
):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        # Simulate delivery of both warnings on a supported request, so the
        # injected channel is acknowledged before the fix/re-break cycle.
        acknowledge_switch_deliveries(
            [message["content"] for message in switch_warning_messages(first)],
            _cache(tmp_path),
            first.switch_generations,
        )
        _load(tmp_path, {"enabled": True, "showSuggestions": False})
        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert [note.split(":", 1)[0] for note in _switch_notes(rebroken)] == ["enabled"]
        assert len(capsys.readouterr().err.splitlines()) == 1
        assert len(_logger_lines(caplog)) == 1
        assert switch_warning_messages(rebroken) == (
            {"role": "system", "content": _switch_notes(rebroken)[0]},
        )


def test_no_per_turn_spam_loop_with_repeated_turns(tmp_path, capsys, caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        _load(tmp_path, {"showSuggestions": False})
        capsys.readouterr()
        caplog.clear()
        for _ in range(5):
            _load(tmp_path, {"showSuggestions": False})
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []


def test_healthy_load_creates_no_trap_file(tmp_path, capsys, caplog):
    _load(tmp_path, {})
    assert not _switches_file(tmp_path).exists()
    assert capsys.readouterr().err == ""


def test_empty_suggestions_path_still_suppresses_switch_warnings(tmp_path, capsys, caplog):
    """An empty display path warns once, not on every load (ticket 03's no-spam rule)."""
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = _load(tmp_path, {"enabled": False, "suggestionsPath": ""})
        assert len(_switch_notes(first)) == 1
        capsys.readouterr()
        caplog.clear()
        for _ in range(3):
            again = _load(tmp_path, {"enabled": False, "suggestionsPath": ""})
            assert _switch_notes(again) == ()
            # Round 3: load alone stays pending on the injected channel.
            assert len(switch_warning_messages(again)) == 1
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []
        acknowledge_switch_deliveries(
            [message["content"] for message in switch_warning_messages(first)],
            _cache(tmp_path),
            first.switch_generations,
        )

        # A restart clears only in-process memory; the cache-side set still suppresses.
        config_module._LOGGED_NOTES.clear()
        restarted = _load(tmp_path, {"enabled": False, "suggestionsPath": ""})
        assert _switch_notes(restarted) == ()
        assert switch_warning_messages(restarted) == ()
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []
        assert _switches_file(tmp_path).exists(), (
            "the seen set persists beside the cache, independent of the display path"
        )


def test_load_alone_does_not_consume_the_injected_channel(tmp_path, capsys, caplog):
    """Round 3 P1: a tool-first load must not permanently suppress delivery."""
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        tool_first = _load(tmp_path, {"enabled": False})
        assert len(switch_warning_messages(tool_first)) == 1
        # A second load (e.g. the middleware's own) still has it pending.
        middleware_load = _load(tmp_path, {"enabled": False})
        assert _switch_notes(middleware_load) == ()
        assert len(switch_warning_messages(middleware_load)) == 1
        acknowledge_switch_deliveries(
            [message["content"] for message in switch_warning_messages(middleware_load)],
            _cache(tmp_path),
            middleware_load.switch_generations,
        )
        after = _load(tmp_path, {"enabled": False})
        assert switch_warning_messages(after) == ()
        # Repeat supported turns stay suppressed after first delivery.
        again = _load(tmp_path, {"enabled": False})
        assert switch_warning_messages(again) == ()


def _delivered(tmp_path):
    path = _switches_file(tmp_path)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("delivered") if isinstance(data, dict) else []
    return {tuple(entry) for entry in entries}


def test_stale_ack_surviving_a_fix_is_rejected(tmp_path, capsys, caplog):
    """Round 4 P2 (generation-bound since round 5): load off, load fixed, ack
    the old note, load off again.

    The stale ack's pending note was captured at the pre-fix generation, so
    it is rejected even where the seen set alone would accept it: the
    re-break warns on the terminal channel and delivers on the injected
    channel again (terminal 1, injected 1), matching the generation-bound
    Deliver guard in the model.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        off = _load(tmp_path, {"enabled": False})
        notes = _switch_notes(off)
        assert len(notes) == 1
        stale = notes[0]
        assert len(switch_warning_messages(off)) == 1

        fixed = _load(tmp_path, {"enabled": True})
        assert _switch_notes(fixed) == ()
        assert _seen(tmp_path) == set() and _delivered(tmp_path) == set()

        # The pre-fix ack arrives late and must be rejected.
        acknowledge_switch_deliveries([stale], _cache(tmp_path), off.switch_generations)
        assert _delivered(tmp_path) == set(), "a stale ack must not land in delivered"

        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {"enabled": False})
        assert _switch_notes(rebroken) == (stale,)
        assert switch_warning_messages(rebroken) == ({"role": "system", "content": stale},)
        assert len(capsys.readouterr().err.splitlines()) == 1
        assert _logger_lines(caplog) == [stale]
        # The fix/rebreak ABA completes: the re-broken note's own ack lands
        # normally (its pair is in the seen set again) and suppresses delivery.
        acknowledge_switch_deliveries(
            [stale], _cache(tmp_path), rebroken.switch_generations
        )
        assert switch_warning_messages(_load(tmp_path, {"enabled": False})) == ()


@pytest.mark.parametrize("switch", ["enabled", "showSuggestions"])
def test_aba_old_note_ack_rejected_both_switches(tmp_path, capsys, caplog, switch):
    """Round 5 P2: the fix/rebreak ABA on either switch.

    Load broken A, capture its pending note and generation, fix (bumping the
    generation), re-break the same A, then ack the OLD note with the OLD
    generation: the pair is back in the seen set, but the generation bound
    rejects it, so the NEW generation's delivery still fires. The new note's
    own on-time ack then lands and suppresses.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        off = _load(tmp_path, {switch: False})
        notes = _switch_notes(off)
        assert len(notes) == 1, notes
        stale = notes[0]
        stale_generations = dict(off.switch_generations)
        assert len(switch_warning_messages(off)) == 1

        assert _switch_notes(_load(tmp_path, {switch: True})) == ()

        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {switch: False})
        assert _switch_notes(rebroken) == (stale,)
        assert switch_warning_messages(rebroken) == ({"role": "system", "content": stale},)

        # The stale ack (old note, old generation) must not land.
        acknowledge_switch_deliveries([stale], _cache(tmp_path), stale_generations)
        assert _delivered(tmp_path) == set(), (
            "an ack from a previous generation must not suppress delivery"
        )
        assert switch_warning_messages(_load(tmp_path, {switch: False})) == (
            {"role": "system", "content": stale},
        ), "the new generation's delivery stays pending after the stale ack"

        # The on-time ack at the current generation lands and suppresses.
        acknowledge_switch_deliveries(
            [stale], _cache(tmp_path), rebroken.switch_generations
        )
        assert _delivered(tmp_path) != set()
        assert switch_warning_messages(_load(tmp_path, {switch: False})) == ()


def test_aba_unaffected_sibling_ack_stays_valid(tmp_path, capsys, caplog):
    """Round 5 P2: through the ABA on one switch, the sibling's ack stays valid.

    Both switches off, capture both notes; fix one (bumping only its
    generation), re-break it, then ack the sibling's note at its unchanged
    generation: it lands. The fixed switch's old note at its old generation
    is still rejected.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        notes = {note.split(":", 1)[0]: note for note in _switch_notes(both)}
        assert set(notes) == {"enabled", "showSuggestions"}
        captured = dict(both.switch_generations)

        def fixed_enabled():
            return _load(tmp_path, {"enabled": True, "showSuggestions": False})

        fixed_enabled()
        rebroken = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert [note.split(":", 1)[0] for note in _switch_notes(rebroken)] == ["enabled"]

        acknowledge_switch_deliveries(
            [notes["showSuggestions"]], _cache(tmp_path), captured
        )
        assert {pair[0] for pair in _delivered(tmp_path)} == {"showSuggestions"}, (
            "the untouched switch's ack stays valid through the ABA"
        )
        acknowledge_switch_deliveries(
            [notes["enabled"]], _cache(tmp_path), captured
        )
        assert {pair[0] for pair in _delivered(tmp_path)} == {"showSuggestions"}, (
            "the pre-fix note of the fixed switch stays rejected"
        )


def test_stale_ack_for_the_other_switch_stays_accepted(tmp_path, capsys, caplog):
    """Round 4 P2: an ack for a switch whose seen set was untouched is accepted.

    Fixing one switch drops only its own sets; the other switch's pending ack
    is still in its seen set and lands in delivered normally.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        notes = _switch_notes(both)
        assert {note.split(":", 1)[0] for note in notes} == {"enabled", "showSuggestions"}
        show_note = next(note for note in notes if note.startswith("showSuggestions: "))

        _load(tmp_path, {"enabled": True, "showSuggestions": False})
        acknowledge_switch_deliveries([show_note], _cache(tmp_path), both.switch_generations)
        delivered = _delivered(tmp_path)
        assert {pair[0] for pair in delivered} == {"showSuggestions"}, (
            "the untouched switch's ack stays valid"
        )
        after = _load(tmp_path, {"enabled": True, "showSuggestions": False})
        assert switch_warning_messages(after) == ()


@pytest.mark.parametrize("fixed", ["enabled", "showSuggestions"])
def test_fixing_one_switch_keeps_the_other_delivered_through_delivery(
    tmp_path, capsys, caplog, fixed
):
    """Round 4 gap, parameterized over both switch directions in round 5:
    reciprocal delivered-side fix, exercised through delivery.

    Both warnings delivered and acknowledged; fixing one switch drops only its
    delivered set, so on re-break it delivers again while the other switch's
    delivered suppression survives a full load + message pass (no stale ack
    involved: the healthy ack for the re-broken switch lands normally).
    """
    other = "showSuggestions" if fixed == "enabled" else "enabled"
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        acknowledge_switch_deliveries(
            [message["content"] for message in switch_warning_messages(both)],
            _cache(tmp_path),
            both.switch_generations,
        )
        assert len(_delivered(tmp_path)) == 2
        assert switch_warning_messages(_load(tmp_path, {"enabled": False, "showSuggestions": False})) == ()

        _load(tmp_path, {fixed: True, other: False})
        assert {pair[0] for pair in _delivered(tmp_path)} == {other}

        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert [note.split(":", 1)[0] for note in _switch_notes(rebroken)] == [fixed]
        assert switch_warning_messages(rebroken) == (
            {"role": "system", "content": _switch_notes(rebroken)[0]},
        )
        # The re-broken switch's own ack lands normally and suppresses again.
        acknowledge_switch_deliveries(
            [message["content"] for message in switch_warning_messages(rebroken)],
            _cache(tmp_path),
            rebroken.switch_generations,
        )
        settled = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert switch_warning_messages(settled) == ()


def test_empty_cache_path_suppresses_via_default_location(
    tmp_path, capsys, caplog, monkeypatch
):
    """Round 3 P2: cachePath='' still suppresses once per value, across restart."""
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    settings = {"enabled": False, "cachePath": "", "suggestionsPath": ""}
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        first = load(settings)
        assert len(_switch_notes(first)) == 1
        capsys.readouterr()
        caplog.clear()
        for _ in range(3):
            again = load(settings)
            assert _switch_notes(again) == ()
        assert capsys.readouterr().err == ""
        assert _logger_lines(caplog) == []
        config_module._LOGGED_NOTES.clear()
        restarted = load(settings)
        assert _switch_notes(restarted) == ()
        assert capsys.readouterr().err == ""
        default_file = tmp_path / ".cache" / "hermes-skill-router" / "config-switches.json"
        assert default_file.exists(), "fingerprints persist in the default cache directory"


def _generations(tmp_path):
    path = _switches_file(tmp_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("generations") or {}) if isinstance(data, dict) else {}


@pytest.mark.parametrize("fixed,other", [("enabled", "showSuggestions"), ("showSuggestions", "enabled")])
def test_concurrent_fix_load_cannot_be_overwritten_by_paused_ack(
    tmp_path, capsys, caplog, monkeypatch, fixed, other
):
    """Round 6 P2: the ack's read-modify-write is serialized with the fix load.

    Barrier-scheduled interleaving: the ack thread pauses between its read
    and its write (inside the write hook), while a fix load for *fixed*
    runs. After the fix, the ack must not restore the pre-fix snapshot:
    the fixed switch's sets stay dropped (so a re-break warns again), its
    generation stays bumped, and the sibling's sets and pending ack are
    preserved. The on-time ack of the sibling still lands afterwards.
    """
    import threading

    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        both = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        notes = {note.split(":", 1)[0]: note for note in _switch_notes(both)}
        assert set(notes) == {"enabled", "showSuggestions"}
        captured = dict(both.switch_generations)
        capsys.readouterr()
        caplog.clear()

        original_write = config_module._write_switch_seen
        paused = threading.Event()
        release = threading.Event()
        writes = [0]

        def hooked_write(path, seen, delivered=None, generations=None):
            writes[0] += 1
            if writes[0] == 1:
                paused.set()
                release.wait(10)
            return original_write(path, seen, delivered, generations)

        monkeypatch.setattr(config_module, "_write_switch_seen", hooked_write)
        ack_thread = threading.Thread(
            target=lambda: acknowledge_switch_deliveries(
                [notes[fixed]], _cache(tmp_path), captured
            )
        )
        ack_thread.start()
        assert paused.wait(10), "the ack must reach its write"

        fix_thread = threading.Thread(
            target=lambda: _load(tmp_path, {fixed: True, other: False})
        )
        fix_thread.start()
        # The fix load must serialize behind the paused ack, not interleave.
        fix_thread.join(1.0)
        assert fix_thread.is_alive(), (
            "the fix load must wait for the ack transaction, not overwrite it"
        )
        release.set()
        ack_thread.join(10)
        fix_thread.join(10)
        assert not ack_thread.is_alive() and not fix_thread.is_alive()

        # The fix survived: the fixed switch's sets are dropped and its
        # generation bumped; the sibling's seen set is untouched.
        assert {pair[0] for pair in _seen(tmp_path)} == {other}, (
            "the paused ack must not restore the fixed switch's seen set"
        )
        assert _generations(tmp_path).get(fixed, 0) == captured.get(fixed, 0) + 1, (
            "the fix's generation bump must survive the interleaving"
        )
        # The sibling's on-time ack still lands and suppresses delivery.
        acknowledge_switch_deliveries([notes[other]], _cache(tmp_path), captured)
        assert {pair[0] for pair in _delivered(tmp_path)} == {other}
        # A re-break of the fixed switch warns and delivers again.
        monkeypatch.setattr(config_module, "_write_switch_seen", original_write)
        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {"enabled": False, "showSuggestions": False})
        assert [note.split(":", 1)[0] for note in _switch_notes(rebroken)] == [fixed]
        assert switch_warning_messages(rebroken) == (
            {"role": "system", "content": _switch_notes(rebroken)[0]},
        )


@pytest.mark.parametrize("switch", ["enabled", "showSuggestions"])
def test_old_switches_file_without_generations_key(tmp_path, capsys, caplog, switch):
    """Round 6 coverage: a schema-1 file written before generations existed.

    Seen and delivered are present but the generations key is absent. The
    file loads as generation zero everywhere: the old delivered stays
    suppressed pre-fix, a fix bumps to generation one, and the pre-fix
    captured ack (generation zero) is then rejected after the re-break.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        off = _load(tmp_path, {switch: False})
        notes = _switch_notes(off)
        assert len(notes) == 1
        note = notes[0]
        acknowledge_switch_deliveries([note], _cache(tmp_path), off.switch_generations)
        assert _delivered(tmp_path) != set()

        path = _switches_file(tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema"] == 1
        data.pop("generations", None)
        path.write_text(json.dumps(data), encoding="utf-8")

        again = _load(tmp_path, {switch: False})
        assert _switch_notes(again) == ()
        assert switch_warning_messages(again) == (), (
            "an old delivered fingerprint stays suppressed pre-fix"
        )

        _load(tmp_path, {switch: True})
        assert _generations(tmp_path).get(switch, 0) == 1, (
            "the first fix of an old file bumps its generation to one"
        )
        capsys.readouterr()
        caplog.clear()
        rebroken = _load(tmp_path, {switch: False})
        assert _switch_notes(rebroken) == (note,)
        # The gen-0 captured ack from before the migration-era fix is stale.
        acknowledge_switch_deliveries([note], _cache(tmp_path), dict(off.switch_generations))
        assert _delivered(tmp_path) == set(), (
            "a replayed generation-zero ack must not consume the new generation"
        )
        assert switch_warning_messages(_load(tmp_path, {switch: False})) == (
            {"role": "system", "content": note},
        )


@pytest.mark.parametrize("switch", ["enabled", "showSuggestions"])
def test_many_fix_rebreak_cycles_replayed_stale_acks_never_consume(
    tmp_path, capsys, caplog, switch
):
    """Round 6 coverage: repeated fix/rebreak cycles past the old TLA bound.

    Six cycles, each capturing the pending note and generation, fixing,
    re-breaking, and replaying every earlier captured ack. No stale ack
    may land in delivered or suppress the current generation's delivery;
    each cycle's on-time ack lands and suppresses until the next fix.
    """
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        captured_history = []
        for cycle in range(6):
            off = _load(tmp_path, {switch: False})
            notes = _switch_notes(off)
            assert len(notes) == 1, (cycle, notes)
            note = notes[0]
            captured_history.append((note, dict(off.switch_generations)))
            assert switch_warning_messages(off) == ({"role": "system", "content": note},)

            _load(tmp_path, {switch: True})
            assert _delivered(tmp_path) == set()
            capsys.readouterr()
            caplog.clear()
            rebroken = _load(tmp_path, {switch: False})
            assert _switch_notes(rebroken) == (note,)
            assert switch_warning_messages(rebroken) == (
                {"role": "system", "content": note},
            )
            # Replay every stale ack ever captured, newest first.
            for stale_note, stale_gens in reversed(captured_history):
                acknowledge_switch_deliveries([stale_note], _cache(tmp_path), stale_gens)
                assert _delivered(tmp_path) == set(), (
                    f"cycle {cycle}: a stale ack must never land in delivered"
                )
            assert switch_warning_messages(_load(tmp_path, {switch: False})) == (
                {"role": "system", "content": note},
            ), f"cycle {cycle}: stale acks must not suppress the current delivery"
            # The on-time ack lands and suppresses.
            acknowledge_switch_deliveries(
                [note], _cache(tmp_path), rebroken.switch_generations
            )
            assert _delivered(tmp_path) != set()
            assert switch_warning_messages(_load(tmp_path, {switch: False})) == ()
            # Fix again so the next cycle's off load warns anew.
            _load(tmp_path, {switch: True})
            capsys.readouterr()
            caplog.clear()


def _first_use_load_worker(settings, cache_path, suggestions_path, barrier):
    """Run one load in a child process, released with the other process."""
    barrier.wait(timeout=20)
    load({"suggestionsPath": suggestions_path, "cachePath": cache_path, **settings})


def _first_use_trap_worker(note, suggestions_path, barrier):
    """Persist one trap note in a child process, released with the other."""
    barrier.wait(timeout=20)
    config_module.acknowledge_trap_notes([note], suggestions_path)


def _run_two_first_use_processes(worker, *args_per_process):
    """Run two child processes released together; fail if either stalls."""
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    children = [
        ctx.Process(target=worker, args=(*args_per_process[i], barrier))
        for i in range(2)
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=30)
    assert all(child.exitcode == 0 for child in children), (
        f"first-use processes must both finish, exitcodes={[c.exitcode for c in children]}"
    )


def _state_lock_holder_worker(path, ready, release):
    """Hold the write-producing state lock until released."""
    with config_module._state_lock(path, create_dir=True):
        ready.set()
        release.wait(timeout=20)


def _state_lock_contender_worker(path, entered):
    """Enter the write-producing state lock and signal once inside."""
    with config_module._state_lock(path, create_dir=True):
        entered.set()


@pytest.mark.parametrize("precreate", [False, True])
def test_cross_process_first_use_state_lock_serializes(tmp_path, precreate, monkeypatch):
    """Round 7 P2: first-use (directory absent) lock entry serializes.

    The holder enters the write-producing state lock with the
    directory initially absent (or, when *precreate*, existing);
    a contender must not enter until the holder releases, so two
    simultaneous first-use processes run their read-modify-writes
    one at a time and neither clobbers the other's state.
    """
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    import multiprocessing

    absent = tmp_path / f"lock-absent-{precreate}"
    if precreate:
        absent.mkdir(parents=True)
    path = str(absent / "config-switches.json")
    ctx = multiprocessing.get_context("fork")
    ready = ctx.Event()
    release = ctx.Event()
    entered = ctx.Event()
    holder = ctx.Process(target=_state_lock_holder_worker, args=(path, ready, release))
    contender = ctx.Process(target=_state_lock_contender_worker, args=(path, entered))
    holder.start()
    try:
        assert ready.wait(timeout=20), "the holder must reach the lock"
        contender.start()
        import time

        time.sleep(1.0)
        assert not entered.is_set(), (
            "the contender must wait for the holder's directory lock"
        )
    finally:
        release.set()
        holder.join(timeout=20)
        contender.join(timeout=20)
    assert entered.is_set(), "the contender must enter after the holder releases"
    assert holder.exitcode == 0 and contender.exitcode == 0


@pytest.mark.parametrize("precreate", [False, True])
def test_cross_process_first_use_loads_write_both_switches(tmp_path, precreate, monkeypatch):
    """Round 7 P2: two simultaneous first-use switch loads serialize.

    Both processes load with both switches off, released together
    with the directory initially absent (or, when *precreate*,
    existing). The written seen file must be one whole record
    carrying both switches' entries, never a torn interleave of
    two direct writes. Repeated so a race that needs overlap
    shows up.
    """
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    for iteration in range(5):
        absent = tmp_path / f"absent-{iteration}-{precreate}"
        if precreate:
            absent.mkdir(parents=True)
        cache_path = str(absent / "index.json")
        suggestions_path = str(absent / "suggestions.json")
        settings = {"enabled": False, "showSuggestions": False}
        _run_two_first_use_processes(
            _first_use_load_worker,
            (settings, cache_path, suggestions_path),
            (settings, cache_path, suggestions_path),
        )
        seen_file = absent / "config-switches.json"
        data = json.loads(seen_file.read_text(encoding="utf-8"))
        switches = {pair[0] for pair in data["seen"]}
        assert switches == {"enabled", "showSuggestions"}, (
            f"iteration {iteration}: both first-use effects must survive, got {data!r}"
        )


def test_cross_process_first_use_trap_union_both_effects_survive(tmp_path, monkeypatch):
    """Round 7 P2: two simultaneous first-use trap unions serialize."""
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    for iteration in range(5):
        absent = tmp_path / f"trap-absent-{iteration}"
        suggestions_path = str(absent / "suggestions.json")
        _run_two_first_use_processes(
            _first_use_trap_worker,
            ("embedModelId: 'a' is not a valid string; using default 'bge-m3'", suggestions_path),
            ("rerankModel: 'x' is not a valid string; using default 'y'", suggestions_path),
        )
        traps_file = absent / "config-traps.json"
        data = json.loads(traps_file.read_text(encoding="utf-8"))
        trap_ids = {pair[0] for pair in data["seen"]}
        assert trap_ids == {"embedModelId", "rerankModel"}, (
            f"iteration {iteration}: both trap unions must survive, got {data!r}"
        )


def test_healthy_load_with_absent_directory_creates_nothing(tmp_path, capsys, caplog, monkeypatch):
    """Round 7 P2: a healthy no-write load creates no file and no directory."""
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    absent = tmp_path / "absent-healthy"
    settings = {
        "suggestionsPath": str(absent / "suggestions.json"),
        "cachePath": str(absent / "index.json"),
    }
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        load(settings)
    assert not absent.exists(), "a healthy load must create no directory"
    assert list(tmp_path.iterdir()) == [], "a healthy load must create no file either"


def test_directory_that_cannot_be_created_stays_fail_open(tmp_path, capsys, caplog, monkeypatch):
    """Round 7 P2: an uncreatable directory is documented fail-open.

    A regular file sits where the state directory should be, so
    mkdir and the directory lock both fail. The load still warns
    (fail-open under the thread lock alone) and never raises; the
    ack likewise never raises.
    """
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    cache_path = str(blocker / "index.json")
    assignments = {
        "suggestionsPath": str(blocker / "suggestions.json"),
        "cachePath": cache_path,
    }
    with caplog.at_level(logging.WARNING, logger="hermes_skill_router.config"):
        cfg = load({**assignments, "enabled": False})
    assert len(_switch_notes(cfg)) == 1, "the warning still shows under fail-open"
    # The ack is fail-open too: it never raises and writes nothing.
    acknowledge_switch_deliveries(
        [message["content"] for message in switch_warning_messages(cfg)],
        cache_path,
        cfg.switch_generations,
    )
    assert blocker.is_file(), "the blocking file must be left alone"
    assert not (blocker / "config-switches.json").exists()


def _healthy_first_use_worker(cache_path, suggestions_path, events):
    """Healthy load in a child process, paused after its lock-open attempt.

    The first lock entry pauses after the directory open (which fails
    ENOENT while the directory is absent) and before the state read;
    a write, or a second lock entry on a retry, pauses again until the
    competitor has finished. ``events`` is
    ``(open_attempted, first_writer_done, paused, competitor_done)``.
    """
    import contextlib

    open_attempted, first_writer_done, paused, competitor_done = events
    original_lock = config_module._state_lock
    original_write = config_module._write_switch_seen
    lock_entries = [0]

    @contextlib.contextmanager
    def wrapped_lock(path, create_dir=False):
        lock_entries[0] += 1
        entry = lock_entries[0]
        if entry == 2:
            paused.set()
            competitor_done.wait(timeout=20)
        with original_lock(path, create_dir=create_dir) as held:
            if entry == 1:
                open_attempted.set()
                first_writer_done.wait(timeout=20)
            yield held

    def wrapped_write(path, seen, delivered=None, generations=None):
        paused.set()
        competitor_done.wait(timeout=20)
        return original_write(path, seen, delivered, generations)

    config_module._state_lock = wrapped_lock
    config_module._write_switch_seen = wrapped_write
    load({"suggestionsPath": suggestions_path, "cachePath": cache_path})


def _competitor_fix_rebreak_worker(cache_path, suggestions_path, switch, other, out_path):
    """Fix both switches, then re-break *switch*; record its note/generations."""
    settings = {"suggestionsPath": suggestions_path, "cachePath": cache_path}
    load({**settings, "enabled": True, "showSuggestions": True})
    cfg = load({**settings, switch: False, other: True})
    notes = _switch_notes(cfg)
    assert len(notes) == 1
    Path(out_path).write_text(
        json.dumps({"note": notes[0], "generations": dict(cfg.switch_generations)}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("switch,other", [("enabled", "showSuggestions"), ("showSuggestions", "enabled")])
def test_first_use_healthy_load_cannot_clobber_competitor_generation(
    tmp_path, monkeypatch, switch, other
):
    """Round 8 F2: no-lock healthy load must not write a stale snapshot.

    Schedule: the healthy child's directory open fails ENOENT (the
    directory is absent), so it proceeds thread-lock-only; the first
    writer then creates the directory and the seen set; the healthy
    child reads that state and pauses pre-write (or pre-retry); a
    competitor process fixes and re-breaks, bumping the generation to
    1 and capturing a note at 1. The healthy child's write must not
    restore the generation-0 snapshot: the fix it performs on the
    competitor's state bumps to 2, so the competitor's captured ack
    (generation 1) stays stale after a fresh re-break.
    """
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    absent = tmp_path / "absent-first-use"
    cache_path = str(absent / "index.json")
    suggestions_path = str(absent / "suggestions.json")
    open_attempted = ctx.Event()
    first_writer_done = ctx.Event()
    paused = ctx.Event()
    competitor_done = ctx.Event()
    healthy = ctx.Process(
        target=_healthy_first_use_worker,
        args=(cache_path, suggestions_path, (open_attempted, first_writer_done, paused, competitor_done)),
    )
    healthy.start()
    competitor_out = tmp_path / "competitor.json"
    try:
        assert open_attempted.wait(timeout=20), "the healthy child must attempt its lock-open"
        first = load({"suggestionsPath": suggestions_path, "cachePath": cache_path, switch: False})
        assert len(_switch_notes(first)) == 1
        first_writer_done.set()
        assert paused.wait(timeout=20), "the healthy child must pause before its write"
        competitor = ctx.Process(
            target=_competitor_fix_rebreak_worker,
            args=(cache_path, suggestions_path, switch, other, str(competitor_out)),
        )
        competitor.start()
        competitor.join(timeout=20)
        assert competitor.exitcode == 0
        competitor_done.set()
        healthy.join(timeout=20)
        assert healthy.exitcode == 0
    finally:
        first_writer_done.set()
        competitor_done.set()
        healthy.join(timeout=20)

    competitor_data = json.loads(competitor_out.read_text(encoding="utf-8"))
    assert _generations(absent).get(switch) == 2, (
        "the healthy fix on the competitor state must bump the generation to 2, "
        f"got {_generations(absent)!r}"
    )
    fresh = load({"suggestionsPath": suggestions_path, "cachePath": cache_path, switch: False})
    acknowledge_switch_deliveries([competitor_data["note"]], cache_path, competitor_data["generations"])
    assert _delivered(absent) == set(), "the stale generation-1 ack must not land"
    assert switch_warning_messages(fresh) == (
        {"role": "system", "content": _switch_notes(fresh)[0]},
    ), "the stale ack must not suppress the current delivery"


def test_first_use_healthy_load_shared_trap_directory(tmp_path, monkeypatch):
    """Round 8 F2: the same race with traps and switches sharing a directory.

    The traps and switches files live in one initially absent
    directory. A trap ack creates the directory (the first writer in
    the schedule), a switch load writes the seen set, and a second
    trap ack lands while the no-lock healthy child is paused. Both
    trap fingerprints must survive the healthy child's switch write,
    and its generation bump must still land on the competitor state.
    """
    monkeypatch.delenv("HERMES_SKILL_ROUTER_SWITCHES_PATH", raising=False)
    monkeypatch.delenv("HERMES_SKILL_ROUTER_TRAPS_PATH", raising=False)
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    absent = tmp_path / "absent-shared"
    cache_path = str(absent / "index.json")
    suggestions_path = str(absent / "suggestions.json")
    trap1 = "embedModelId: 'a' is not a valid string; using default 'bge-m3'"
    trap2 = "rerankModel: 'x' is not a valid string; using default 'y'"
    open_attempted = ctx.Event()
    first_writer_done = ctx.Event()
    paused = ctx.Event()
    competitor_done = ctx.Event()
    healthy = ctx.Process(
        target=_healthy_first_use_worker,
        args=(cache_path, suggestions_path, (open_attempted, first_writer_done, paused, competitor_done)),
    )
    healthy.start()
    competitor_out = tmp_path / "competitor.json"
    try:
        assert open_attempted.wait(timeout=20), "the healthy child must attempt its lock-open"
        config_module.acknowledge_trap_notes([trap1], suggestions_path)
        first = load({"suggestionsPath": suggestions_path, "cachePath": cache_path, "enabled": False})
        assert len(_switch_notes(first)) == 1
        first_writer_done.set()
        assert paused.wait(timeout=20), "the healthy child must pause before its write"
        competitor = ctx.Process(
            target=_competitor_fix_rebreak_worker,
            args=(cache_path, suggestions_path, "enabled", "showSuggestions", str(competitor_out)),
        )
        competitor.start()
        competitor.join(timeout=20)
        assert competitor.exitcode == 0
        config_module.acknowledge_trap_notes([trap2], suggestions_path)
        competitor_done.set()
        healthy.join(timeout=20)
        assert healthy.exitcode == 0
    finally:
        first_writer_done.set()
        competitor_done.set()
        healthy.join(timeout=20)

    traps_file = absent / "config-traps.json"
    traps_data = json.loads(traps_file.read_text(encoding="utf-8"))
    assert {pair[0] for pair in traps_data["seen"]} == {"embedModelId", "rerankModel"}, (
        f"both trap fingerprints must survive, got {traps_data!r}"
    )
    assert _generations(absent).get("enabled") == 2, (
        f"the healthy fix must bump the generation to 2, got {_generations(absent)!r}"
    )
