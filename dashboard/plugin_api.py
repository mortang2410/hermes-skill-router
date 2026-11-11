"""The desktop display's read-only backend route (module M12, contract C9).

Mounted at ``/api/plugins/hermes-skill-router/`` by Hermes' dashboard plugin loader
(:func:`hermes_cli.web_server_dashboard._mount_plugin_api_routes`), which imports this file by path
and wraps ``router`` with its per-request profile scope. ``GET /suggestions`` answers the names-only
record :mod:`hermes_skill_router.display` writes, or ``{"status": "idle", "names": []}`` when there
is no readable record.

Three rules drive the shape of this module:

* **Read-only.** The route never writes the record, never creates the cache directory and never
  calls the router. T5 owns the writer; this file only projects what is on disk.
* **Names only.** :func:`hermes_skill_router.display.read` returns the seven known fields and drops
  anything else, so a stray ``text`` or ``messages`` key in the file can never reach a client, and
  request text is never read, stored or shown.
* **It never raises.** The dashboard's plugin contract is that a handler folds every failure into
  its "no data" answer instead of raising out to the client: a missing module, an unreadable config
  or a bad record all answer idle, so the pane renders its plain line instead of an error.

The record path follows ``plugins.entries.hermes-skill-router.settings.suggestionsPath`` in the
active profile config, with :func:`hermes_skill_router.config.load` applying the default and every
bound; tests inject a path directly through :func:`answer`.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter

logger = logging.getLogger(__name__)

#: Plugin id: the dashboard namespace (``/api/plugins/<id>/``) and the config entry that holds
#: this plugin's settings.
PLUGIN_ID = "hermes-skill-router"

#: The plugin directory this dashboard half ships inside; the plugin package lives at its root.
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

#: The answer for "there is no record to show": absent, unreadable, or corrupt all read as idle.
IDLE: Dict[str, Any] = {"status": "idle", "names": []}

#: The display module, cached after the first successful import (it is stateless between reads).
_DISPLAY: Optional[Any] = None

router = APIRouter()


def settings_from(config: Any) -> Dict[str, Any]:
    """Return this plugin's settings mapping out of one profile config, ``{}`` when absent.

    Reads the documented location ``plugins.entries.hermes-skill-router.settings`` and falls back
    to the legacy ``config`` subtree, mirroring :meth:`hermes_cli.plugins.PluginContext.get_config`.
    """
    plugins = config.get("plugins") if isinstance(config, Mapping) else None
    entries = plugins.get("entries") if isinstance(plugins, Mapping) else None
    entry = entries.get(PLUGIN_ID) if isinstance(entries, Mapping) else None
    if not isinstance(entry, Mapping):
        return {}
    for key in ("settings", "config"):
        raw = entry.get(key)
        if isinstance(raw, Mapping):
            return {str(name): value for name, value in raw.items()}
    return {}


def active_config() -> Mapping[str, Any]:
    """Return the active profile's config mapping, or ``{}`` when Hermes cannot supply it."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:  # no config API (an older Hermes, a bare import): the defaults still answer
        return {}
    return config if isinstance(config, Mapping) else {}


def settings() -> Dict[str, Any]:
    """Return this plugin's settings from the active profile config."""
    return settings_from(active_config())


def _display_module() -> Tuple[Any, Any]:
    """Return ``(display, package)`` for the record reader, reusing the plugin's loaded copies.

    The plugin itself is imported as ``hermes_plugins.<slug>``, so its ``hermes_skill_router``
    sub-package is already in :data:`sys.modules` under that prefix when the route runs. Preferring
    it keeps one copy of ``display`` (and its once-per-process logging) in the process. A process
    that imported this file alone — ``hermes plugins validate``, a test — falls back to the bare
    package with the plugin root on ``sys.path``.
    """
    global _DISPLAY
    if _DISPLAY is not None:
        package = sys.modules.get(_DISPLAY.__package__ or "")
        if package is not None and hasattr(package, "load"):
            return _DISPLAY, package
    for name, module in list(sys.modules.items()):
        if name == "hermes_skill_router.display" or name.endswith(".hermes_skill_router.display"):
            package = sys.modules.get(module.__package__ or "")
            if package is not None and hasattr(package, "load"):
                _DISPLAY = module
                return module, package
    root = str(PLUGIN_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    display = importlib.import_module("hermes_skill_router.display")
    package = sys.modules.get(display.__package__ or "")
    if package is None or not hasattr(package, "load"):
        raise RuntimeError("hermes_skill_router imported without its config loader")
    _DISPLAY = display
    return display, package


def configured() -> Any:
    """Return the record target this profile's settings name, every default and bound applied."""
    _, package = _display_module()
    return package.load(settings())


def answer(target: Any = None) -> Dict[str, Any]:
    """Return the route body for *target*: the record, or :data:`IDLE`.

    *target* is a :class:`hermes_skill_router.config.Config`, a record path, or ``None`` for the
    path the active profile configures. The record is passed through as
    :func:`hermes_skill_router.display.read` returned it — schema, timestamp, session and turn id,
    status, origin and names, and no other key — so the schema the writer wrote is the schema the
    pane receives, the origin that says a model search produced the names is projected unchanged,
    and no request text can ride along.
    """
    display, _ = _display_module()
    record = display.read(configured() if target is None else target)
    return record if record is not None else dict(IDLE)


@router.get("/suggestions")
def suggestions() -> Dict[str, Any]:
    """Answer the last routed decision: the record, or the idle answer. Never raises."""
    try:
        return answer()
    except Exception:  # plugin contract: a handler never raises out to the client
        logger.warning(
            "hermes-skill-router: the suggestions route fell back to idle", exc_info=True
        )
        return dict(IDLE)
