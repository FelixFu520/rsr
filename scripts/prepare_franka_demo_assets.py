#!/usr/bin/env python3
"""Copy the demo's local assets and their USD dependency closure into assets/Demo.

Run with Isaac Sim python.sh. No SimulationApp or physics process is started.
Original USD files, relative paths, textures, and license notices are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path


def _usd_modules(isaac_root: Path) -> tuple[object, object]:
    """Load the installed USD Python package without starting a Kit application."""
    try:
        from pxr import Sdf, UsdUtils
    except (ImportError, ModuleNotFoundError):
        candidates = sorted((isaac_root / "extscache").glob("omni.usd.libs-*/pxr/UsdUtils"))
        if not candidates or os.environ.get("RSR_DEMO_USD_BOOTSTRAPPED"):
            raise RuntimeError("Run this tool with the installed Isaac Sim python.sh") from None
        package = candidates[-1].parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(package) + os.pathsep + env.get("PYTHONPATH", "")
        env["LD_LIBRARY_PATH"] = str(package / "bin") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["RSR_DEMO_USD_BOOTSTRAPPED"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    return Sdf, UsdUtils


def prepare_assets(isaac_root: Path, deformable_project: Path, destination: Path) -> dict:
    """Copy assets from installed Isaac and the user's reference project."""
    isaac_root = isaac_root.resolve()
    deformable_project = deformable_project.resolve()
    destination = destination.resolve()
    project_root = Path(__file__).resolve().parents[1]
    isaac_assets = isaac_root / "Assets/Isaac/6.2/Isaac"
    source_directories = {
        "Table": project_root / "assets/Scene/Table/table_000",
        "Cube": project_root / "assets/Scene/Cube/cube_000",
        "Robot/franka_emika_panda": deformable_project / "assets/newton/franka_emika_panda",
    }
    required_files = (
        isaac_assets / "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        isaac_assets / "Robots/FrankaRobotics/FrankaPanda/franka.usd",
        source_directories["Table"] / "Aligned.usd",
        source_directories["Cube"] / "Aligned.usd",
        source_directories["Robot/franka_emika_panda"] / "urdf/fr3_franka_hand.urdf",
        deformable_project / "assets/newton/unisex_shirt.usd",
        deformable_project / "assets/newton/LICENSE",
    )
    # Check before USD loading or writes. In particular, never recurse into a
    # destination nested in a source directory, nor overwrite original assets.
    source_roots = [
        isaac_assets,
        *source_directories.values(),
        deformable_project / "assets/newton",
    ]
    for source in source_roots:
        source = source.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError(f"Destination overlaps an asset source: {destination} and {source}")
        if not source.is_dir():
            raise FileNotFoundError(f"Missing asset source directory: {source}")
    for source in required_files:
        if not source.is_file():
            raise FileNotFoundError(f"Missing required asset: {source}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Destination is not a directory: {destination}")
    sdf, usd_utils = _usd_modules(isaac_root)
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, dict] = {}
    mdl_sources: set[Path] = set()

    def copy(source: Path, target: Path) -> None:
        if str(target.relative_to(destination)) in copied:
            return
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with target.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        copied[str(target.relative_to(destination))] = {
            "source": str(source),
            "bytes": target.stat().st_size,
            "sha256": digest,
        }
        if source.suffix.lower() == ".mdl":
            mdl_sources.add(source)

    builtin_materials: set[str] = set()
    for relative in (
        "Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
        "Robots/FrankaRobotics/FrankaPanda/franka.usd",
    ):
        source = isaac_assets / relative
        layers, assets, unresolved = usd_utils.ComputeAllDependencies(sdf.AssetPath(str(source)))
        missing = [item for item in unresolved if item != "OmniPBR.mdl"]
        if missing:
            raise RuntimeError(f"Unresolved USD dependencies in {source}: {missing}")
        builtin_materials.update(unresolved)
        dependencies = {Path(layer.realPath) for layer in layers if layer.realPath}
        dependencies.update(Path(item) for item in assets)
        for dependency in sorted(dependencies):
            copy(dependency, destination / "Isaac" / dependency.relative_to(isaac_assets))
            # Preserve license notices for every package contributing a dependency.
            for ancestor in dependency.parents:
                if ancestor == isaac_assets.parent:
                    break
                for notice in ancestor.glob("*LICENSE*"):
                    if notice.is_file():
                        copy(notice, destination / "Isaac" / notice.relative_to(isaac_assets))

    # USD dependency discovery sees shader source assets but does not parse MDL
    # imports or texture defaults. Include those transitive local dependencies.
    scanned_mdl: set[Path] = set()
    while pending := mdl_sources - scanned_mdl:
        source = pending.pop()
        scanned_mdl.add(source)
        text = source.read_text(encoding="utf-8-sig")
        modules = re.findall(r"\b(?:using|import)\s+\.::([A-Za-z_][\w:]*)", text)
        dependencies = [source.parent / (module.replace("::", "/") + ".mdl") for module in modules]
        textures = re.findall(r'\btexture_[23]d\s*\(\s*"([^"\n]+)"', text)
        dependencies.extend(source.parent / texture for texture in textures)
        for dependency in dependencies:
            dependency = dependency.resolve()
            copy(dependency, destination / "Isaac" / dependency.relative_to(isaac_assets))

    for name, source in source_directories.items():
        for item in sorted(source.rglob("*")):
            if item.is_file() and "__pycache__" not in item.parts:
                copy(item, destination / name / item.relative_to(source))
    copy(
        deformable_project / "assets/newton/unisex_shirt.usd",
        destination / "Cloth/unisex_shirt.usd",
    )
    copy(deformable_project / "assets/newton/LICENSE", destination / "Cloth/LICENSE")
    provenance = {
        "isaac_root": str(isaac_root),
        "deformable_project": str(deformable_project),
        "builtin_materials": sorted(builtin_materials),
        "notes": [
            "All copied assets are unmodified; relative USD dependencies are preserved.",
            "OmniPBR.mdl is supplied by Isaac Sim Kit's built-in material search path.",
            "Transitive MDL imports and texture defaults are copied as local dependencies.",
            "Warehouse retains CC BY 4.0 notices; Franka and Newton retain license files.",
            "Project table/cube had no separate license; no new license is asserted.",
            "Runtime scene scales the table and flattens the shirt; source files are unchanged.",
        ],
        "files": copied,
    }
    (destination / "manifest.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return provenance


def main() -> None:
    """Parse paths and prepare the reproducible local asset bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-root", type=Path, default=Path("/home/fufa/isaacsim"))
    parser.add_argument(
        "--deformable-project", type=Path, default=Path("/home/fufa/Projects/grasp_deformable")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).resolve().parents[1] / "assets/Demo"
    )
    args = parser.parse_args()
    result = prepare_assets(
        args.isaac_root.resolve(), args.deformable_project.resolve(), args.destination.resolve()
    )
    print(json.dumps({"destination": str(args.destination), "files": len(result["files"])}))


if __name__ == "__main__":
    main()
