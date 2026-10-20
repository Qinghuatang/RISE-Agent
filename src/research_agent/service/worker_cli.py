"""Command-line launcher for a configured RISE Worker process."""

from __future__ import annotations

import importlib
import logging
import os
import signal
from typing import Any

from .worker import TaskWorker


def _load_worker(specification: str) -> TaskWorker:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("RISE_WORKER_FACTORY must use the form 'module:callable'")
    module = importlib.import_module(module_name)
    factory: Any = getattr(module, attribute_name)
    worker = factory()
    if not isinstance(worker, TaskWorker):
        raise TypeError("RISE_WORKER_FACTORY must return a TaskWorker")
    return worker


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("RISE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    specification = os.environ.get("RISE_WORKER_FACTORY")
    if not specification:
        raise RuntimeError("set RISE_WORKER_FACTORY to 'module:callable'")
    worker = _load_worker(specification)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: worker.request_shutdown())
    worker.run_forever()


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    main()
