"""Copy the UR5e package into the self-contained PhysicSolver asset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare_rotation_texture(target: Path) -> None:
    """Generate asymmetric axial paint and index ticks, authored specifically for this demo."""
    import numpy as np
    from PIL import Image

    target.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:1024, :512]
    pixels = np.full((1024, 512, 3), [24, 76, 210], dtype=np.uint8)
    pixels[x >= 256] = [255, 214, 32]
    pixels[x >= 448] = [245, 245, 240]
    pixels[(x >= 384) & (x < 412)] = [230, 35, 35]
    # Short black ticks preserve broad longitudinal markers that reveal angular motion.
    pixels[((y % 128) < 12) & (x >= 270) & (x < 370)] = [15, 20, 28]
    path = target / "bolt_rotation.png"
    Image.fromarray(pixels).save(path)
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "source": "Procedurally authored by prepare_screw_arm_assets.py",
                "license": "CC0-1.0",
                "purpose": "Asymmetric blue/yellow/white axial paint and red angular index",
                "physical_geometry_modified": False,
                "files": {path.name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
            },
            indent=2,
        )
        + "\n"
    )


def prepare(source: Path, target: Path) -> None:
    """Copy without modifying the source or overwriting different existing files."""
    package = source / "universalrobots_ur5e"
    if not (package / "ur5e.usda").is_file():
        raise FileNotFoundError(package / "ur5e.usda")
    manifest = {}
    for src in sorted(package.rglob("*")):
        if not src.is_file():
            continue
        relative = src.relative_to(package)
        dst = target / relative
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to replace changed asset: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        manifest[str(relative)] = {"sha256": digest, "bytes": src.stat().st_size}
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(package.resolve()),
                "upstream": "NVIDIA isaacsim-assets/universalrobots_ur5e",
                "revision": "a0d9dfbac5163a7005442ae4fe8f954ff6780093",
                "license": "BSD-3-Clause; see .metadata/license.txt",
                "modified": False,
                "tool": "Runtime-authored electric driver, axial spring, rigid hex socket",
                "files": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info("Screw arm assets copied", target=str(target), files=len(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/fufa/Projects/screwdriving_robot_usd/screwdriving_robot_usd"),
    )
    parser.add_argument("--target", type=Path, default=Path("assets/PhysicSolver/Robot/ur5e"))
    args = parser.parse_args()
    prepare(args.source, args.target)
    prepare_rotation_texture(Path("assets/PhysicSolver/Screw/M20/Textures"))
