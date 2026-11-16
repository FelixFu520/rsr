"""Copy the supplied cutting/v1 assets without modifying their source files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare(source: Path, destination: Path) -> dict:
    """Copy a self-contained snapshot; refuse to overwrite different existing assets."""
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in (
        "kitchen.usdc",
        "knife.json",
        "flesh.png",
        "banana_cutting.blend",
        "walnut.png",
        "terrazzo.png",
        "textures/walnut.png",
        "textures/terrazzo.png",
        "textures/color_0C0C0C.exr",
    ):
        original, target = source / "assets" / relative, destination / relative
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Asset snapshot differs: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files.append(dict(path=relative, sha256=digest, bytes=target.stat().st_size))
    provenance = {
        "source": str(source),
        "files": files,
        "modifications": "Copied byte for byte",
        "license": "No license supplied with the local example; redistribution rights unverified",
        "source_code": {
            name: hashlib.sha256((source / "scripts" / name).read_bytes()).hexdigest()
            for name in ("geometry.py", "run_demo.py", "build_assets.py")
        },
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    """Prepare the task-local kitchen, knife and textures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the source cutting/v1 directory")
    parser.add_argument("--destination", type=Path, default=Path("assets/PhysicSolver/Banana"))
    args = parser.parse_args()
    result = prepare(args.source, args.destination)
    get_logger(__name__).info("Banana assets ready", files=len(result["files"]))


if __name__ == "__main__":
    main()
