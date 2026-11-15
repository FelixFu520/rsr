"""Summarize what a USD scene contains before choosing a physics engine (read-only).

Usage (repository root):
    ./app/python.sh .agents/skills/rsr-task-development/scripts/usd_inventory.py <file.usd> [...]

Prints units/up axis, prim types (joints, meshes, lights...), applied physics schemas
(articulations, drives, mimic joints, deformables, materials) and mesh collision approximations.
Starts a headless Kit only when the bare interpreter cannot import pxr. Writes no files.
"""

from __future__ import annotations

import sys
from collections import Counter


def _import_pxr():
    """Import pxr; fall back to a headless SimulationApp like the asset preparation scripts."""
    app = None
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom, UsdPhysics

    return app, Usd, UsdGeom, UsdPhysics


def summarize(path: str, Usd, UsdGeom, UsdPhysics) -> None:  # noqa: N803 - pxr module names
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    types: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    approximations: Counter[str] = Counter()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        types[prim.GetTypeName() or "<untyped>"] += 1
        for name in prim.GetAppliedSchemas():
            schemas[name] += 1
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            value = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            approximations[value or "<unset: triangle mesh for static colliders>"] += 1
    default = stage.GetDefaultPrim()
    print(f"== {path}")
    print(
        "metersPerUnit:",
        UsdGeom.GetStageMetersPerUnit(stage),
        "upAxis:",
        UsdGeom.GetStageUpAxis(stage),
    )
    print("defaultPrim:", default.GetPath() if default else None)
    print("layers used:", len(stage.GetUsedLayers()))
    print("prim types:", dict(types.most_common(30)))
    print("applied schemas:", dict(schemas.most_common(50)))
    print("mesh collision approximation:", dict(approximations))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    app, Usd, UsdGeom, UsdPhysics = _import_pxr()
    status = 0
    try:
        for path in sys.argv[1:]:
            summarize(path, Usd, UsdGeom, UsdPhysics)
    except Exception as exc:  # noqa: BLE001 - report, then keep the exit status through Kit
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        status = 1
    if app is not None:
        app.close(exit_code=status)  # Kit fast shutdown exits here
    return status


if __name__ == "__main__":
    sys.exit(main())
