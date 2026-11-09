"""Preserve CLI exit status when Kit's fast shutdown terminates its Python process."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def run_kit_worker(
    script: str,
    arguments: list[str],
    result_file: Path,
    *,
    marker_field: str,
    require_success: bool = False,
) -> int:
    """Run Kit separately and accept only the result written by this invocation."""
    marker = uuid4().hex
    worker = subprocess.run(
        [sys.executable, script, *arguments, "--worker-id", marker], check=False
    )
    if worker.returncode:
        return worker.returncode if worker.returncode > 0 else 1
    if not result_file.is_file():
        raise RuntimeError("Kit exited without writing this run's result")
    result = json.loads(result_file.read_text())
    if result.get(marker_field) != marker:
        raise RuntimeError("Kit did not finish this run; refusing to accept an old result")
    return 0 if not require_success or result["success"] else 1
