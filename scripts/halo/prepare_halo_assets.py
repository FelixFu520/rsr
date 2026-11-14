"""Copy the HALO G1 task inputs (crawl_v3 / crouch_v3 / sit_v1) into assets/Halo.

Run with Isaac Sim's interpreter: ``./app/python.sh scripts/halo/prepare_halo_assets.py``.
Sources stay unchanged and are copied byte for byte. Existing different files are never
overwritten. The copied set is self-contained: home_010 living room, the reference-material
29-DOF G1, the two Blender desks, the three accepted SONIC reference motions and the GEAR-SONIC
encoder/decoder. Isaac's built-in ``OmniPBR.mdl`` resolves through the Kit search path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from real2sim2real.core.logger import get_logger

log = get_logger(__name__)

HALO = Path("/home/fufa/Projects/halo/output")
# Built-in MDL modules that Kit resolves itself; they are not copied.
KIT_MDL = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}

# (source, destination below assets/Halo, USD entry whose dependencies are copied with it)
USD_ENTRIES = [
    (HALO / "v5/assets/Home010/home_010.usdc", "Home010/home_010.usdc"),
    (HALO / "crawl_v3/assets/g1_29_reference.usd", "Robot/g1_29_reference.usd"),
    (HALO / "crawl_v3/assets/Table/crawl_desk.usdc", "Desk/crawl/crawl_desk.usdc"),
    (HALO / "crouch_v3/assets/TableReference/crouch_desk.usdc", "Desk/crouch/crouch_desk.usdc"),
]
FILES = [
    (HALO / "crawl_v3/assets/robot_29_reference_manifest.json", "Robot/robot_manifest.json"),
    (HALO / "crawl_v3/configs/g1_reference_materials.json", "Robot/g1_reference_materials.json"),
    (HALO / "crawl_v3/assets/Table/dimensions.json", "Desk/crawl/dimensions.json"),
    (HALO / "crawl_v3/assets/Table/crawl_desk.blend", "Desk/crawl/crawl_desk.blend"),
    (HALO / "crouch_v3/assets/TableReference/dimensions.json", "Desk/crouch/dimensions.json"),
    (HALO / "crouch_v3/assets/TableReference/crouch_desk.blend", "Desk/crouch/crouch_desk.blend"),
    (HALO / "v3/models/model_encoder.onnx", "models/sonic/model_encoder.onnx"),
    (HALO / "v3/models/model_decoder.onnx", "models/sonic/model_decoder.onnx"),
    (HALO / "v3/models/observation_config.yaml", "models/sonic/observation_config.yaml"),
    (HALO / "v3/vendor/GR00T-WholeBodyControl/LICENSE", "models/sonic/LICENSE"),
    (HALO / "crawl_v3/reference/crawl_29_native/qpos.npy", "reference/crawl/qpos.npy"),
    (HALO / "crawl_v3/reference/crawl_29_native/spec.json", "reference/crawl/spec.json"),
    (HALO / "crouch_v3/reference/crouch_29_native/qpos.npy", "reference/crouch/qpos.npy"),
    (HALO / "crouch_v3/reference/crouch_29_native/spec.json", "reference/crouch/spec.json"),
    (HALO / "sit_v1/reference/walk_then_sit/qpos.npy", "reference/sit/qpos.npy"),
    (HALO / "sit_v1/reference/walk_then_sit/spec.json", "reference/sit/spec.json"),
]
SOURCES = {
    "Home010": "home_010 living-room input of the HALO project (identical bytes in "
    "/home/fufa/Projects/halo/home_010); used unchanged by crawl_v3, crouch_v3 and sit_v1",
    "Robot": "Isaac Sim 6.2 Robots/Unitree/G1/g1.usd (Physics=PhysX, ThreeFinger hands), "
    "flattened by halo prepare_robot.py with the reference silver materials and rebuilt "
    "shading normals (collision, mass and joints unchanged; see robot_manifest.json)",
    "Desk/crawl": "Blender desk from halo crawl_v3/scripts/build_desk.py (1.10x0.60 m, 0.83 m)",
    "Desk/crouch": "Blender desk from halo crouch_v3/scripts/build_desk.py (1.15x0.60 m, 1.13 m)",
    "models/sonic": "GEAR-SONIC encoder/decoder from NVlabs/GR00T-WholeBodyControl "
    "(commit 1e1ec4f5f27e6c3e03048f693bd7c60cce04afda); planner_sonic.onnx is not needed",
    "reference": "Accepted SONIC reference motions of crawl_v3 (crawl_29_native), crouch_v3 "
    "(crouch_29_native) and sit_v1 (walk_then_sit); references only, never replayed as states",
}
LICENSES = {
    "Robot": "NVIDIA Isaac Sim asset terms (Unitree G1 model shipped with Isaac Sim 6.2)",
    "models/sonic": "Code Apache-2.0, weights NVIDIA Open Model License (models/sonic/LICENSE)",
    "Home010": "Not recorded in the HALO project; confirm before redistribution",
    "Desk": "Generated in the HALO project (user-owned)",
    "reference": "Generated in the HALO project (user-owned)",
}


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


def _copy_usd(pxr: Any, entry: Path, destination: Path, manifest: dict, root: Path) -> dict:
    """Copy a USD file and every local dependency below its directory, keeping relative paths."""
    layers, assets, unresolved = pxr.UsdUtils.ComputeAllDependencies(str(entry))
    if unresolved:
        raise RuntimeError(f"{entry} has unresolved dependencies: {unresolved}")
    source_dir = entry.parent.resolve()
    builtin, copied = [], []
    for item in [*(layer.realPath for layer in layers), *assets]:
        if not item:
            continue
        if "://" in item:
            raise RuntimeError(f"{entry} references a remote asset: {item}")
        path = Path(item).resolve()
        if path.name in KIT_MDL and "/kit/mdl/" in str(path):
            builtin.append(path.name)
            continue
        if source_dir not in path.parents:
            raise RuntimeError(f"{entry} depends on a file outside its directory: {path}")
        target = destination.parent / path.relative_to(source_dir)
        _copy(path, target, manifest, root)
        copied.append(str(target.relative_to(root)))
    return {"files": len(copied), "kit_builtin_mdl": sorted(set(builtin))}


def _verify(pxr: Any, root: Path) -> dict:
    """Open each copied USD; every used layer must live in assets/Halo."""
    report = {}
    for _, relative in USD_ENTRIES:
        stage = pxr.Usd.Stage.Open(str(root / relative))
        if stage is None or not stage.GetDefaultPrim():
            raise RuntimeError(f"Copied USD has no default prim: {relative}")
        outside = [
            layer.realPath
            for layer in stage.GetUsedLayers()
            if layer.realPath and root.resolve() not in Path(layer.realPath).resolve().parents
        ]
        if outside:
            raise RuntimeError(f"{relative} loads layers outside assets/Halo: {outside}")
        report[relative] = str(stage.GetDefaultPrim().GetPath())
    home = pxr.Usd.Stage.Open(str(root / "Home010/home_010.usdc"))
    if not home.GetPrimAtPath("/root/sofa_main/sofa_254_002"):
        raise RuntimeError("home_010 lost the target sofa /root/sofa_main/sofa_254_002")
    return report


def prepare(target: Path) -> None:
    """Copy all HALO inputs and write provenance.json."""
    pxr, app = _import_pxr()
    try:
        root = target.resolve()
        manifest: dict[str, str] = {}
        usd = {}
        for source, relative in USD_ENTRIES:
            usd[relative] = _copy_usd(pxr, source, root / relative, manifest, root)
        for source, relative in FILES:
            _copy(source, root / relative, manifest, root)
        default_prims = _verify(pxr, root)
        provenance = {
            "task": "halo",
            "source_project": str(HALO.parent),
            "sources": SOURCES,
            "licenses": LICENSES,
            "copied_from": {relative: str(source) for source, relative in [*USD_ENTRIES, *FILES]},
            "usd_dependencies": usd,
            "default_prims": default_prims,
            "modified": [],
            "note": "Byte copies; no file is edited. Physics overrides are applied on the "
            "runtime stage by scene_halo / robot_halo.",
            "sha256": dict(sorted(manifest.items())),
        }
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        log.info("HALO 资产已准备", target=str(root), files=len(manifest))
    except BaseException:
        log.exception("HALO 资产准备失败")
        if app is not None:
            app.close(exit_code=1)
        raise
    if app is not None:
        app.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/Halo"))
    prepare(parser.parse_args().target)


if __name__ == "__main__":
    main()
