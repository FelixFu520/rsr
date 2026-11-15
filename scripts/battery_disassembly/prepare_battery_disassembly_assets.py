"""Copy the dual-KR210 battery disassembly inputs into assets/BatteryDisassembly.

Run with Isaac Sim's interpreter from the repository root::

    ./app/python.sh scripts/battery_disassembly/prepare_battery_disassembly_assets.py \
        [--scenes C1003B1A1 C1003B1A2 ...]

Sources (``/home/fufa/Projects/Real2Sim2Real/assets``) stay unchanged and are copied byte for
byte together with every local USD / MDL / texture dependency. Existing different files are never
overwritten. ``--scenes`` selects which ``SimScenes`` backgrounds to copy (default C1003B1A1);
each background is about 0.2-5 GB. Isaac's built-in ``OmniPBR.mdl`` resolves through the Kit
search path and is not copied. Physics proxies are authored on the runtime stage by
``scene_battery_disassembly`` / ``robot_battery_disassembly``; no copied file is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from real2sim2real.core.logger import get_logger

log = get_logger(__name__)

SOURCE = Path("/home/fufa/Projects/Real2Sim2Real/assets")
ISAAC = Path("app/extsDeprecated/isaacsim.robot_motion.motion_generation")
KIT_MDL = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}

# (USD entry, its source directory, destination directory below assets/BatteryDisassembly)
USD_ENTRIES = [
    (SOURCE / "Scene/Battery/Battery.usd", SOURCE / "Scene/Battery", "Scene/Battery"),
    (
        SOURCE / "Robot/Kuka/KR210_L150/kr210_l150.usd",
        SOURCE / "Robot/Kuka/KR210_L150",
        "Robot/KR210_L150",
    ),
    (
        SOURCE / "tmp/misc/kr210_tools/crosshead_screwdriver.usdz",
        SOURCE / "tmp/misc/kr210_tools",
        "Tools",
    ),
    (SOURCE / "tmp/misc/kr210_tools/kr210_gripper.usdz", SOURCE / "tmp/misc/kr210_tools", "Tools"),
]
FILES = [
    (ISAAC / "motion_policy_configs/Kuka/kr210/kr210.urdf", "Robot/KR210_L150/kr210.urdf"),
    (
        ISAAC / "PACKAGE-LICENSES/isaacsim.robot_motion.motion_generation-LICENSE.md",
        "Robot/KR210_L150/kr210_urdf-LICENSE.md",
    ),
    (SOURCE / "Background/SimScenes/README.md", "Background/README.md"),
]
SOURCES = {
    "Background": "Real2Sim2Real assets/Background/SimScenes (30-scene factory set); the "
    "README states USD physics is kept as delivered",
    "Scene/Battery": "Real2Sim2Real assets/Scene/Battery/Battery.usd (workstation, battery pack "
    "with cover and four screws, basket); used by configs/arm_kr210_kr210_unscrew/c1003b1a*.toml",
    "Robot/KR210_L150": "Real2Sim2Real assets/Robot/Kuka/KR210_L150/kr210_l150.usd (Isaac Sim "
    "KUKA KR210 L150 USD, PhysX schemas) and Isaac Sim 6.2 deprecated motion_generation "
    "kr210.urdf (xacro kr210l150; same joint origins as the USD) for numpy IK",
    "Tools": "Real2Sim2Real assets/tmp/misc/kr210_tools (crosshead screwdriver, KR210 gripper; "
    "visual meshes only, no physics)",
}
LICENSES = {
    "Robot/KR210_L150/kr210_l150.usd": "NVIDIA Isaac Sim asset terms",
    "Robot/KR210_L150/kr210.urdf": "Isaac Sim extension license (copied alongside); geometry "
    "from ROS-Industrial kuka_experimental kr210l150 xacro (Apache-2.0)",
    "Background": "Not recorded in the source project; confirm before redistribution",
    "Scene/Battery": "Not recorded in the source project; confirm before redistribution",
    "Tools": "Not recorded in the source project; confirm before redistribution",
}
_MDL_MODULE = re.compile(r"\.::([A-Za-z_][A-Za-z0-9_]*)")
_MDL_TEXTURE = re.compile(r'texture_2d\(\s*"([^"]+)"')


def _import_pxr() -> tuple[Any, Any | None]:
    """Import pxr; start a headless Kit only when the bare interpreter lacks USD."""
    from types import SimpleNamespace

    app = None
    try:
        from pxr import Usd, UsdUtils
    except ImportError:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        from pxr import Usd, UsdUtils
    return SimpleNamespace(Usd=Usd, UsdUtils=UsdUtils), app


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(original: Path, destination: Path, manifest: dict[str, str], root: Path) -> None:
    if not original.is_file():
        raise FileNotFoundError(original)
    if destination.exists():
        if destination.stat().st_size != original.stat().st_size or _sha256(destination) != _sha256(
            original
        ):
            raise FileExistsError(f"Refusing to overwrite different asset: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    manifest[str(destination.relative_to(root))] = _sha256(destination)


def _mdl_closure(paths: list[Path]) -> list[Path]:
    """Add MDL sibling modules and ``texture_2d`` files that USD dependency scans miss."""
    found = set(paths)
    pending = [path for path in paths if path.suffix == ".mdl"]
    while pending:
        mdl = pending.pop()
        text = mdl.read_text(errors="replace")
        modules = [mdl.parent / f"{name}.mdl" for name in _MDL_MODULE.findall(text)]
        textures = [(mdl.parent / name).resolve() for name in _MDL_TEXTURE.findall(text)]
        for path in [*modules, *textures]:
            if not path.is_file():
                raise FileNotFoundError(f"{mdl} references a missing file: {path}")
            if path not in found:
                found.add(path)
                if path.suffix == ".mdl":
                    pending.append(path)
    return sorted(found)


def _copy_usd(
    pxr: Any, entry: Path, source_dir: Path, target_dir: Path, manifest: dict, root: Path
) -> dict[str, Any]:
    """Copy a USD entry and its local dependencies below ``source_dir`` keeping relative paths."""
    layers, assets, unresolved = pxr.UsdUtils.ComputeAllDependencies(str(entry))
    local, builtin = [], []
    for item in [*(layer.realPath for layer in layers), *assets]:
        if not item:
            continue
        if "://" in item:
            raise RuntimeError(f"{entry} references a remote asset: {item}")
        # usdz members are reported as archive[member]; the archive itself is copied.
        path = Path(item.split("[", 1)[0]).resolve()
        if path.name in KIT_MDL and "/kit/mdl/" in str(path):
            builtin.append(path.name)
            continue
        local.append(path)
    local = _mdl_closure(sorted(set(local)))
    source_dir = source_dir.resolve()
    outside = [str(path) for path in local if source_dir not in path.parents]
    if outside:
        raise RuntimeError(f"{entry} depends on files outside {source_dir}: {outside}")
    for path in local:
        _copy(path, target_dir / path.relative_to(source_dir), manifest, root)
    return {
        "files": len(local),
        "kit_builtin_mdl": sorted(set(builtin)),
        # Unreal export leftovers (/Game/..., /Engine/...) are attribute strings, not files.
        "unresolved_non_file_paths": sorted(set(unresolved)),
    }


def _verify(pxr: Any, root: Path, entries: list[str]) -> dict[str, str]:
    """Open each copied USD; every used layer must live inside the task asset directory."""
    report = {}
    for relative in entries:
        stage = pxr.Usd.Stage.Open(str(root / relative))
        if stage is None or not stage.GetDefaultPrim():
            raise RuntimeError(f"Copied USD has no default prim: {relative}")
        outside = [
            layer.realPath
            for layer in stage.GetUsedLayers()
            if layer.realPath
            and root not in Path(layer.realPath.split("[", 1)[0]).resolve().parents
        ]
        if outside:
            raise RuntimeError(f"{relative} loads layers outside {root}: {outside}")
        report[relative] = str(stage.GetDefaultPrim().GetPath())
    battery = pxr.Usd.Stage.Open(str(root / "Scene/Battery/Battery.usd"))
    required = [
        "/root/Workstation/Workstation",
        "/root/Pure_Battery/Battery/BatteryCover/Cover",
        "/root/Basket/Basket1",
        *(f"/root/Pure_Battery/Battery/BatteryCover/screw_0{i}" for i in range(1, 5)),
    ]
    missing = [path for path in required if not battery.GetPrimAtPath(path)]
    if missing:
        raise RuntimeError(f"Battery.usd lost required prims: {missing}")
    return report


def prepare(target: Path, scenes: list[str]) -> None:
    """Copy all inputs and write provenance.json (existing provenance is merged)."""
    pxr, app = _import_pxr()
    try:
        root = target.resolve()
        manifest: dict[str, str] = {}
        usd: dict[str, Any] = {}
        entries = []
        copied_from = {}
        for entry, source_dir, relative in USD_ENTRIES:
            destination = root / relative
            key = f"{relative}/{entry.name}"
            usd[key] = _copy_usd(pxr, entry, source_dir, destination, manifest, root)
            entries.append(key)
            copied_from[key] = str(entry)
        for scene in scenes:
            entry = SOURCE / f"Background/SimScenes/{scene}/{scene}.usd"
            key = f"Background/{scene}/{scene}.usd"
            usd[key] = _copy_usd(
                pxr, entry, entry.parent, root / f"Background/{scene}", manifest, root
            )
            entries.append(key)
            copied_from[key] = str(entry)
            preview = SOURCE / f"Background/SimScenes/preview/{scene}.png"
            if preview.is_file():
                _copy(preview, root / f"Background/preview/{scene}.png", manifest, root)
        for source, relative in FILES:
            _copy(source.resolve(), root / relative, manifest, root)
            copied_from[relative] = str(source.resolve())
        default_prims = _verify(pxr, root, entries)
        path = root / "provenance.json"
        previous = json.loads(path.read_text()) if path.is_file() else {}
        provenance = {
            "task": "battery_disassembly",
            "source_project": "/home/fufa/Projects/Real2Sim2Real",
            "sources": SOURCES,
            "licenses": LICENSES,
            "copied_from": {**previous.get("copied_from", {}), **copied_from},
            "usd_dependencies": {**previous.get("usd_dependencies", {}), **usd},
            "default_prims": {**previous.get("default_prims", {}), **default_prims},
            "modified": [],
            "note": "Byte copies; no file is edited. Collision proxies, masses, materials, "
            "screw threads and tool joints are authored on the runtime stage.",
            "sha256": dict(sorted({**previous.get("sha256", {}), **manifest}.items())),
        }
        path.write_text(json.dumps(provenance, indent=2) + "\n")
        log.info("电池拆解资产已准备", target=str(root), files=len(manifest), scenes=scenes)
    except BaseException:
        log.exception("电池拆解资产准备失败")
        if app is not None:
            app.close(exit_code=1)
        raise
    if app is not None:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/BatteryDisassembly"))
    parser.add_argument("--scenes", nargs="+", default=["C1003B1A1"])
    args = parser.parse_args()
    prepare(args.target, args.scenes)


if __name__ == "__main__":
    main()
