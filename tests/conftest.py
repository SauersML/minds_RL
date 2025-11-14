"""Test configuration for ensuring local packages are importable."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "asyncio: mark a test function to run inside an event loop",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("asyncio")
    if marker is None:
        return None
    testfunction = pyfuncitem.obj
    if not inspect.iscoroutinefunction(testfunction):
        return None
    loop = asyncio.new_event_loop()
    try:
        loop.set_debug(False)
        loop.run_until_complete(testfunction(**pyfuncitem.funcargs))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return True


