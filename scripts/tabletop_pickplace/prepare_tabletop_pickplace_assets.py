"""Copy the tabletop pick-and-place inputs into assets/TabletopPickplace.

Run from the repository root with Isaac Sim's interpreter::

    ./app/python.sh scripts/tabletop_pickplace/prepare_tabletop_pickplace_assets.py

Sources (``/home/fufa/Projects/Real2Sim2Real/assets``, the assets of the source configuration
``configs/tabletop-pickplace_franka_curobo_home000.toml``) stay unchanged. Every USD entry is
copied byte for byte with its local USD / MDL / texture dependencies, keeping the source's
relative directory layout. Existing different files are never overwritten. Masses, materials and
colliders are authored on the runtime stage by ``scene_tabletop_pickplace``; no copied file is
edited. Isaac's built-in ``OmniPBR.mdl`` / ``OmniPBR_Opacity.mdl`` resolve through the Kit search
path and are not copied. The cuMotion Franka description (URDF + XRDF) is part of Isaac Sim's
``isaacsim.robot_motion.cumotion`` extension and is not copied either.

``scripts/tabletop_replay/prepare_tabletop_replay_assets.py`` reuses :func:`prepare` with
``assets/TabletopReplay`` as the target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from real2sim2real.core.logger import get_logger
from scripts.battery_disassembly.prepare_battery_disassembly_assets import (
    _copy,
    _copy_usd,
    _import_pxr,
)

log = get_logger(__name__)

SOURCE = Path("/home/fufa/Projects/Real2Sim2Real/assets")
KIT_MDL = {"OmniPBR.mdl", "OmniPBR_Opacity.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}

# (USD entry relative to SOURCE, directory whose layout is preserved). Destinations mirror SOURCE.
USD_ENTRIES = [
    ("Background/Home/home_000/interior_template.usdc", "Background/Home/home_000"),
    ("Scene/Table/table_000/Aligned.usd", "Scene/Table/table_000"),
    ("Scene/Cube/cube_000/Aligned.usd", "Scene/Cube/cube_000"),
    ("Scene/Fruit/apple/apple_000/Aligned.usd", "Scene/Fruit/apple/apple_000"),
    ("Robot/FrankaPanda/franka.usd", "Robot/FrankaPanda"),
    (
        "Sensor/Sensing/SG5/H100SA/SG5-IMX490C-5300-GMSL2-H110SA.usd",
        "Sensor/Sensing/SG5/H100SA",
    ),
]
FILES = ["Robot/FrankaPanda/franka-LICENSE.txt"]
# Prims the task code addresses inside each copied entry.
REQUIRED_PRIMS = {
    "Scene/Cube/cube_000/Aligned.usd": ["/World", "/World/body/visual"],
    "Scene/Fruit/apple/apple_000/Aligned.usd": ["/World/body", "/World/body/visual"],
    "Scene/Table/table_000/Aligned.usd": ["/World/imagetostl_mesh0", "/World/imagetostl_mesh1"],
    "Robot/FrankaPanda/franka.usd": [
        "/panda/panda_link0",
        "/panda/panda_hand",
        "/panda/panda_leftfinger",
        "/panda/panda_rightfinger",
        "/panda/panda_hand/panda_finger_joint1",
    ],
    "Sensor/Sensing/SG5/H100SA/SG5-IMX490C-5300-GMSL2-H110SA.usd": [
        "/H100SA/Camera_SG5_IMX490C_5300_GMSL2_H110SA"
    ],
}
SOURCES = {
    "Background/Home/home_000": "Real2Sim2Real assets/Background/Home/home_000 (home interior, "
    "207 mesh colliders, 11 rect lights; one dynamic body made kinematic at runtime)",
    "Scene/Table/table_000": "Real2Sim2Real assets/Scene/Table/table_000 (0.60 x 1.00 x 0.73 m "
    "image-to-mesh table, visual meshes only; colliders are authored at runtime)",
    "Scene/Cube/cube_000": "Real2Sim2Real assets/Scene/Cube/cube_000 (5 cm cube, rigid body, "
    "boundingCube collider, friction 2.0; authored mass 1 g, overridden at runtime)",
    "Scene/Fruit/apple/apple_000": "Real2Sim2Real assets/Scene/Fruit/apple/apple_000 (9.3 cm "
    "apple, rigid body, convex decomposition, no authored mass; overridden at runtime)",
    "Robot/FrankaPanda": "Real2Sim2Real assets/Robot/FrankaPanda (Isaac Sim Franka Panda USD with "
    "PhysX drives; finger_joint2 mimics finger_joint1)",
    "Sensor/Sensing/SG5/H100SA": "Real2Sim2Real assets/Sensor/Sensing/SG5/H100SA (Sensing SG5 "
    "IMX490 H110SA camera: housing meshes and an ftheta fisheye Camera prim)",
}
LICENSES = {
    "Robot/FrankaPanda": "franka-LICENSE.txt (copied alongside)",
    "Sensor/Sensing/SG5/H100SA": "NVIDIA Isaac Sim sensor asset terms (not recorded in the source)",
    "Background/Home/home_000": "Not recorded in the source project; confirm before redistribution",
    "Scene": "Not recorded in the source project; confirm before redistribution",
}


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    """Separate Kit built-in MDL names from genuinely unresolved paths."""
    unresolved = report.pop("unresolved_non_file_paths")
    builtin = sorted(set(report["kit_builtin_mdl"]) | {p for p in unresolved if p in KIT_MDL})
    report["kit_builtin_mdl"] = builtin
    report["unresolved"] = sorted(p for p in unresolved if p not in KIT_MDL)
    return report


def _verify(pxr: Any, root: Path) -> dict[str, str]:
    """Open each copied entry; used layers must stay inside ``root``; required prims exist."""
    report = {}
    for relative, _ in USD_ENTRIES:
        stage = pxr.Usd.Stage.Open(str(root / relative))
        if stage is None or not stage.GetDefaultPrim():
            raise RuntimeError(f"Copied USD has no default prim: {relative}")
        outside = [
            layer.realPath
            for layer in stage.GetUsedLayers()
            if layer.realPath and root not in Path(layer.realPath).resolve().parents
        ]
        if outside:
            raise RuntimeError(f"{relative} loads layers outside {root}: {outside}")
        missing = [p for p in REQUIRED_PRIMS.get(relative, []) if not stage.GetPrimAtPath(p)]
        if missing:
            raise RuntimeError(f"{relative} lost required prims: {missing}")
        report[relative] = str(stage.GetDefaultPrim().GetPath())
    return report


def prepare(target: Path, *, source: Path = SOURCE, task: str = "tabletop_pickplace") -> None:
    """Copy every entry and write ``provenance.json``."""
    pxr, app = _import_pxr()
    try:
        target = target.resolve()
        manifest: dict[str, str] = {}
        dependencies: dict[str, Any] = {}
        for relative, directory in USD_ENTRIES:
            entry = source / relative
            if not entry.is_file():
                raise FileNotFoundError(entry)
            dependencies[relative] = _classify(
                _copy_usd(pxr, entry, source / directory, target / directory, manifest, target)
            )
        for relative in FILES:
            _copy(source / relative, target / relative, manifest, target)
        unresolved = {k: v["unresolved"] for k, v in dependencies.items() if v["unresolved"]}
        # The Franka USD has an unselected "Robotiq_2F_85" gripper variant whose payload points
        # at a file the source project does not contain; the Default variant is complete.
        allowed = {"Robot/FrankaPanda/franka.usd": {"Robotiq_2F_85_edit.usd"}}
        unexpected = {
            k: [p for p in v if Path(p).name not in allowed.get(k, set())]
            for k, v in unresolved.items()
        }
        if any(unexpected.values()):
            raise RuntimeError(f"Unresolved dependencies: {unexpected}")
        default_prims = _verify(pxr, target)
        provenance = {
            "task": task,
            "source_root": str(source),
            "source_config": "Real2Sim2Real configs/tabletop-pickplace_franka_curobo_home000.toml",
            "sources": SOURCES,
            "licenses": LICENSES,
            "modified": {},
            "runtime_overrides": "scene_tabletop_pickplace / robot_tabletop_pickplace author "
            "masses, collider APIs, contact reports and joint states on the runtime stage only",
            "not_copied": {
                "Kit MDL": "OmniPBR.mdl / OmniPBR_Opacity.mdl from the Kit MDL search path",
                "cuMotion": "app/exts/isaacsim.robot_motion.cumotion robot_configurations/franka",
            },
            "dependencies": dependencies,
            "default_prims": default_prims,
            "sha256": dict(sorted(manifest.items())),
        }
        (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        log.info("Tabletop assets ready", target=str(target), files=len(manifest))
    except BaseException:
        # Kit's fast shutdown exits inside close(); log first and keep a failing status.
        log.exception("Preparing tabletop assets failed")
        if app is not None:
            app.close(exit_code=1)
        raise
    if app is not None:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/TabletopPickplace"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    prepare(args.target, source=args.source)


if __name__ == "__main__":
    main()
