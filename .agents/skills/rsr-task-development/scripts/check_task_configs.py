"""Validate task TOML files against the registered backends' config classes, without Kit.

Usage (repository root):
    ./app/python.sh .agents/skills/rsr-task-development/scripts/check_task_configs.py [path ...]

Each path is a TOML file or a directory searched recursively; default is ``configs/``.
For every file the inherit chain is applied, then each section's ``backend`` is resolved through
the registry and the backend's ``config_cls`` validates the section (``extra="forbid"`` catches
misspelt fields). Run it on all task configs after changing a shared Config class or default.
Exits non-zero when any section fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

from real2sim2real.core.config import BaseConfig, apply_settings_file
from real2sim2real.core.registry import (
    JOB_REGISTRY,
    PLANNER_REGISTRY,
    ROBOT_REGISTRY,
    SCENE_REGISTRY,
    SIM_WORLD_REGISTRY,
    ensure_job_backends,
    ensure_planner_backends,
    ensure_robot_backends,
    ensure_scene_backends,
    ensure_sim_world_backends,
)

SECTIONS = {
    "sim_world": (SIM_WORLD_REGISTRY, ensure_sim_world_backends),
    "scene": (SCENE_REGISTRY, ensure_scene_backends),
    "robot": (ROBOT_REGISTRY, ensure_robot_backends),
    "planner": (PLANNER_REGISTRY, ensure_planner_backends),
    "job": (JOB_REGISTRY, ensure_job_backends),
}


def main() -> int:
    roots = [Path(arg) for arg in sys.argv[1:]] or [Path("configs")]
    files = sorted(
        p for root in roots for p in ([root] if root.is_file() else root.rglob("*.toml"))
    )
    if not files:
        print(f"No TOML files under {roots}")
        return 2
    failures = 0
    for path in files:
        apply_settings_file(path)
        problems = []
        for section, (registry, ensure) in SECTIONS.items():
            ensure()
            name = BaseConfig.peek_backend(section)
            try:
                getattr(registry.get(name), "config_cls", BaseConfig).from_settings(section)
            except Exception as exc:  # noqa: BLE001 - collect every failing section
                problems.append(f"[{section}] {name}: {type(exc).__name__}: {exc}")
        failures += len(problems)
        print(f"{'FAIL' if problems else 'ok  '} {path}")
        for problem in problems:
            print(f"     {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
