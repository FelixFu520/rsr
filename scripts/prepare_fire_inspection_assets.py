"""Copy the fire warehouse, its USD dependencies and the X30 robot into assets/FireInspection.

Run with Isaac Sim's interpreter: ``./app/python.sh scripts/prepare_fire_inspection_assets.py``.
Sources stay unchanged. Existing different files are never overwritten.
The copied fire layer keeps the original Flow prims; only its warehouse sublayer path is
rewritten to point at the copied ``003_StorageHouse/Demonstration.usd``.
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

X30_ROOT = Path("/home/fufa/Projects/x30")
FIRE_LAYER = X30_ROOT / "output/warehouse_fire/warehouse_cardbox_fire.usda"
FIRE_MANIFEST = X30_ROOT / "output/warehouse_fire/fire_manifest.json"
ROBOT_FILES = [
    X30_ROOT / "output/x30_fire_inspection.usdc",
    X30_ROOT / "output/asset_manifest.json",
    X30_ROOT / "output/LICENSE_OEM.txt",
]


def _import_pxr() -> tuple[Any, Any | None]:
    """Import pxr; start a headless Kit only when the bare interpreter lacks USD."""
    from types import SimpleNamespace

    app = None
    try:
        from pxr import Sdf, Usd, UsdUtils
    except ImportError:
        from isaacsim import SimulationApp

        app = SimulationApp({"headless": True})
        from pxr import Sdf, Usd, UsdUtils
    return SimpleNamespace(Sdf=Sdf, Usd=Usd, UsdUtils=UsdUtils), app


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(original: Path, destination: Path, manifest: dict[str, str], root: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != original.stat().st_size or _sha256(destination) != _sha256(
            original
        ):
            raise FileExistsError(f"Refusing to overwrite different asset: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    manifest[str(destination.relative_to(root))] = _sha256(destination)


def _dependencies(pxr: Any, layer_path: Path) -> tuple[list[Path], list[str], list[str]]:
    """Return local files, non-local asset paths and unresolved references of a USD file."""
    layers, assets, unresolved = pxr.UsdUtils.ComputeAllDependencies(str(layer_path))
    local: list[Path] = []
    remote: list[str] = []
    for item in [*(layer.realPath for layer in layers), *assets]:
        if not item:
            continue
        if "://" in item:
            remote.append(item)
        else:
            local.append(Path(item).resolve())
    return sorted(set(local)), sorted(set(remote)), sorted(set(unresolved))


_MDL_MODULE = re.compile(r"\.::([A-Za-z_][A-Za-z0-9_]*)")
_MDL_TEXTURE = re.compile(r'texture_2d\(\s*"([^"]+)"')


def _mdl_closure(paths: list[Path]) -> list[Path]:
    """Add files that MDL modules load themselves; USD dependency scans do not see these.

    Covers sibling module imports (``using .::OmniUe4Base import *``) and texture files
    (``texture_2d("./Textures/x.png", ...)``), applied transitively.
    """
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


def _copy_tree_deps(
    pxr: Any, entry: Path, source_dir: Path, target_dir: Path, manifest: dict, root: Path
) -> dict[str, list[str]]:
    """Copy every local USD and MDL dependency of ``entry`` that lives below ``source_dir``."""
    local, remote, unresolved = _dependencies(pxr, entry)
    local = _mdl_closure(local)
    outside = [str(path) for path in local if source_dir not in path.parents]
    if outside:
        raise RuntimeError(
            f"{entry} depends on files outside {source_dir}; copy them explicitly: {outside}"
        )
    for path in local:
        _copy(path, target_dir / path.relative_to(source_dir), manifest, root)
    return {"remote": remote, "unresolved": unresolved}


def prepare(target: Path) -> None:
    """Copy assets, rewrite the fire sublayer and verify the composed stage."""
    pxr, app = _import_pxr()
    try:
        Sdf, Usd = pxr.Sdf, pxr.Usd
        target = target.resolve()
        manifest: dict[str, str] = {}
        report: dict[str, Any] = {}

        # Warehouse referenced by the fire layer as a sublayer.
        fire_layer = Sdf.Layer.FindOrOpen(str(FIRE_LAYER))
        if fire_layer is None:
            raise FileNotFoundError(FIRE_LAYER)
        if len(fire_layer.subLayerPaths) != 1:
            raise RuntimeError(f"Expected one warehouse sublayer: {fire_layer.subLayerPaths}")
        warehouse = Path(
            Sdf.ComputeAssetPathRelativeToLayer(fire_layer, fire_layer.subLayerPaths[0])
        ).resolve()
        if not warehouse.is_file():
            raise FileNotFoundError(warehouse)
        warehouse_dir = warehouse.parent
        target_warehouse_dir = target / "warehouse" / warehouse_dir.name
        report["warehouse"] = _copy_tree_deps(
            pxr, warehouse, warehouse_dir, target_warehouse_dir, manifest, target
        )

        # Fire layer: export a copy, then point its sublayer at the copied warehouse.
        fire_copy = target / "warehouse" / FIRE_LAYER.name
        sublayer = f"./{warehouse_dir.name}/{warehouse.name}"
        if not fire_copy.exists():
            fire_copy.parent.mkdir(parents=True, exist_ok=True)
            if not fire_layer.Export(str(fire_copy)):
                raise RuntimeError(f"Failed to export {fire_copy}")
            copied = Sdf.Layer.FindOrOpen(str(fire_copy))
            copied.subLayerPaths.clear()
            copied.subLayerPaths.append(sublayer)
            copied.Save()
        else:
            existing = Sdf.Layer.FindOrOpen(str(fire_copy))  # keep alive while reading
            if list(existing.subLayerPaths) != [sublayer]:
                raise FileExistsError(f"Refusing to overwrite different fire layer: {fire_copy}")
        manifest[str(fire_copy.relative_to(target))] = _sha256(fire_copy)

        fire_info = json.loads(FIRE_MANIFEST.read_text())
        fire_info["scene"] = str(fire_copy.relative_to(target.parents[1]))
        fire_info["source_scene"] = str(FIRE_LAYER)
        fire_manifest = target / "warehouse" / "fire_manifest.json"
        fire_manifest.write_text(json.dumps(fire_info, indent=2) + "\n")
        manifest[str(fire_manifest.relative_to(target))] = _sha256(fire_manifest)

        # Flattened X30 robot and its licence/manifest.
        robot_dir = target / "robot" / "x30"
        for original in ROBOT_FILES:
            if not original.is_file():
                raise FileNotFoundError(original)
            _copy(original, robot_dir / original.name, manifest, target)
        report["robot"] = _copy_tree_deps(
            pxr, ROBOT_FILES[0], ROBOT_FILES[0].parent, robot_dir, manifest, target
        )

        # Verify the copies compose without the source directories.
        stage = Usd.Stage.Open(str(fire_copy))
        for path in (fire_info["target_prim"], "/FireFX", "/FireFX/flowEmitterSphere"):
            if not stage.GetPrimAtPath(path):
                raise RuntimeError(f"Copied fire scene is missing {path}")
        used = {layer.realPath for layer in stage.GetUsedLayers() if layer.realPath}
        leaked = sorted(p for p in used if not Path(p).resolve().is_relative_to(target))
        if leaked:
            raise RuntimeError(f"Copied scene still loads layers outside {target}: {leaked}")
        robot_stage = Usd.Stage.Open(str(robot_dir / ROBOT_FILES[0].name))
        if not robot_stage.GetDefaultPrim():
            raise RuntimeError("Copied X30 USD has no default prim")
        report["robot_default_prim"] = str(robot_stage.GetDefaultPrim().GetPath())

        (target / "provenance.json").write_text(
            json.dumps(
                {
                    "sources": {
                        "fire_layer": str(FIRE_LAYER),
                        "warehouse": str(warehouse),
                        "robot": [str(path) for path in ROBOT_FILES],
                    },
                    "modified": {
                        "warehouse/" + FIRE_LAYER.name: f"subLayerPaths -> [{sublayer}]",
                    },
                    "dependencies": report,
                    "sha256": manifest,
                },
                indent=2,
            )
            + "\n"
        )
        # Unresolved entries are Unreal "/Game/..." material names left in the source
        # warehouse, not files; the full list is kept in provenance.json.
        log.info(
            "Fire inspection assets ready",
            target=str(target),
            files=len(manifest),
            unresolved=len(report["warehouse"]["unresolved"] + report["robot"]["unresolved"]),
        )
    except BaseException:
        # Kit's fast shutdown exits the process inside close(); log first and keep the status.
        log.exception("Preparing fire inspection assets failed")
        if app is not None:
            app.close(exit_code=1)
        raise
    if app is not None:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/FireInspection"))
    prepare(parser.parse_args().target)
