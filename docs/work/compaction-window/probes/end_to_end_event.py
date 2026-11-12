#!/usr/bin/env python3
"""Prove the compaction event reaches the router's own registered callback.

The probe loads the plugins the running host would load, marks a scratch session's dump record,
then delivers the ``post_compaction`` event the host's compaction path fires. When ``--engine`` names
the handoff engine checkout it delivers the event through that engine's own ``announce_commit``,
which is the wiring ticket 04 exists for; otherwise it calls the host's fire function directly, which
still exercises fire, dispatch and the router's callback.

It exits 0 only when the record is expired by the event, and prints the record state around the call.

Usage:
    python3 end_to_end_event.py [--install PATH] [--engine PATH] [--session ID]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_INSTALL = (
    "/home/wilder/.hermes/installs/5323f763494eb399/environments/6e4b37e4dab14600ac4818bd4b5ccaec/workspace"
)


def load_fixtures(package: str):
    """Import the suite's rewrite fixtures against the modules the plugin actually loaded.

    The host loads this plugin as ``hermes_plugins.<directory>``, so a plain ``import
    hermes_skill_router`` would build a second copy of the package and a second dump record. Aliasing
    the loaded submodules under their repository names first keeps one record for both.
    """
    prefix = f"{package}.hermes_skill_router"
    for name, module in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules["hermes_skill_router" + name[len(prefix):]] = module
    path = Path(__file__).resolve().parents[4] / "tests" / "test_rewrite.py"
    spec = importlib.util.spec_from_file_location("probe_window_fixtures", path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return fixtures


def resolve_install(explicit: str | None) -> str:
    """Return the host install directory without invoking the launcher.

    The ``hermes`` shim is a two-line ``sh`` script that execs the install's own python against
    ``<install>/workspace/hermes_cli/main.py``, so its text carries the path. Reading it is passive:
    running ``hermes --version`` on this host also tries to complete a paused source update, which is
    a side effect a probe must not cause.
    """
    if explicit:
        return explicit
    shim = shutil.which("hermes")
    if shim:
        try:
            with open(shim, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            text = ""
        marker = "/hermes_cli/main.py"
        index = text.find(marker)
        if index > 0:
            for token in reversed(text[:index].split()):
                candidate = token.strip("\"'")
                if candidate.endswith("/workspace"):
                    return candidate
    return DEFAULT_INSTALL


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", default=None, help="host install directory (default: read the hermes shim)")
    parser.add_argument("--engine", default=None, help="context-handoff-engine checkout to announce through")
    parser.add_argument("--session", default=None, help="scratch session id (default: one per run)")
    args = parser.parse_args()

    install = resolve_install(args.install)
    session_id = args.session or f"probe-compaction-window-{int(time.time())}"
    sys.path.insert(0, install)
    if args.engine:
        sys.path.insert(0, args.engine)

    from hermes_cli import plugins as host_plugins

    manager = host_plugins._delivery_manager()  # discovers the profile's plugins, as the host does
    print(f"install={install}")
    print(f"post_compaction registered by the host load: {manager.has_hook('post_compaction')}")
    if not manager.has_hook("post_compaction"):
        print("FAIL: no plugin registered post_compaction, so the router's hook never loaded")
        return 1

    import importlib

    # The host loads the plugin as ``hermes_plugins.<directory name>``, so the record lives in that
    # package's search module. Importing a second copy by file path would mark a different set and
    # the probe would prove nothing, so resolve the module the loaded hook itself came from.
    hook = (manager._hooks.get("post_compaction") or [None])[0]
    package = getattr(hook, "__module__", "") or ""
    search = None
    for candidate in (f"{package}.hermes_skill_router.search",
                      "hermes_plugins.hermes_skill_router.hermes_skill_router.search"):
        try:
            search = importlib.import_module(candidate)
            break
        except ModuleNotFoundError:
            continue
    if search is None:
        loaded = sorted(k for k in sys.modules
                        if k.endswith(".search") and hasattr(sys.modules[k], "remember_dump"))
        if not loaded:
            print("FAIL: the host loaded no plugin module that owns the dump record")
            return 1
        search = sys.modules[loaded[0]]
    print(f"record module: {search.__name__}")
    dumped, remember_dump = search.dumped, search.remember_dump

    remember_dump(session_id)
    print(f"record before the event: dumped={dumped(session_id)} session={session_id}")
    if not dumped(session_id):
        print("FAIL: could not mark the scratch session, so the probe proves nothing")
        return 1

    if args.engine:
        from _announce import announce_commit

        print(f"delivering through {args.engine}/_announce.announce_commit")
        announce_commit(session_id=session_id, messages_removed=4)
    else:
        from agent.compaction_events import fire_post_compaction

        print("delivering through agent.compaction_events.fire_post_compaction")
        fire_post_compaction(
            session_id=session_id,
            old_session_id="",
            kind="summarize",
            reason="handoff_commit",
            before_tokens=None,
            after_tokens=None,
            messages_removed=4,
        )

    after = dumped(session_id)
    print(f"record after the event: dumped={after}")
    if after:
        print("FAIL: the event did not expire the record")
        return 1

    # The window's effect on the next passes, through the same rewrite the middleware calls: the
    # suite's fixtures are imported rather than copied, and the plugin's own modules are aliased
    # under their repository names first so the fixtures and the callback read one record.
    fixtures = load_fixtures(package)
    with tempfile.TemporaryDirectory(prefix="probe-window-") as tmp:
        cfg = fixtures.make_config(Path(tmp))
        index = fixtures.make_index()
        payload = fixtures._window_history(2, fixtures.SUMMARY)

        def failing():
            return fixtures.rewrite(
                payload,
                cfg=cfg,
                index=index,
                embed=fixtures.FakeEmbedder(result=None),
                rerank=fixtures.FakeReranker(),
                session_id=session_id,
            )

        first = failing()
        print(f"failed pass after the event: returned={'None (catalog kept, dumped)' if first is None else 'payload'}")
        if first is not None:
            print("FAIL: the pass after the event stripped the catalog instead of dumping it")
            return 1
        second = failing()
        tail = (second or {}).get("request", {}).get("messages", [{}])[-1].get("content", "")
        print(f"second failed pass: stripped={second is not None} marker={tail == fixtures.DROPPED_MESSAGE}")
        if second is None or tail != fixtures.DROPPED_MESSAGE:
            print("FAIL: the second pass did not strip the catalog and leave the dropped marker")
            return 1

    print("PASS: the event reached the router's registered callback, and the next passes dumped then stripped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
