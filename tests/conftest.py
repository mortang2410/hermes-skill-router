"""Shared test guard: no test starts the real background heal.

Contract C4 makes a turn with an incomplete association store request the background heal, and the
heal job walks the live Hermes session store. A suite that exercised the middleware or the library
would otherwise scan the operator's real history from a daemon thread, so every test gets a no-op
C6 entry point by default.

A test that proves the launch replaces ``heal_store`` again with its own recorder (see
``tests/test_skip_heal.py``). A module that needs the real job shadows this ``no_live_heal``
fixture with a fixture of the same name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _no_heal(
    store_path: str,
    state_db_path: Optional[str] = None,
    *,
    catalog: Sequence[Any] = (),
    now_ms: Optional[int] = None,
) -> None:
    """Stand in for the C6 job entry point without touching the store or the session store."""
    return None


@pytest.fixture(autouse=True)
def no_live_heal(monkeypatch):
    """Replace the C6 job entry point in every loaded router copy, and forget old attempts."""
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith("associations"):
            continue
        if hasattr(module, "heal_store"):
            monkeypatch.setattr(module, "heal_store", _no_heal)
        if hasattr(module, "reset_heal_state"):
            module.reset_heal_state()
    yield
