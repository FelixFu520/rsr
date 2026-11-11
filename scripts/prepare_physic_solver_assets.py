"""Copy the verified local demo asset bundle into assets/PhysicSolver without links."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare_assets(source: Path, destination: Path) -> dict:
    """Preserve relative USD/URDF/MDL dependencies and record a hash for every copy."""
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Asset source and destination must not overlap")
    roots = ["Isaac", "Table", "Cloth", "Cube", "Robot/franka_emika_panda"]
    files = sorted(p for name in roots for p in (source / name).rglob("*") if p.is_file())
    for name in roots:
        if not (source / name).is_dir():
            raise FileNotFoundError(source / name)
    records = {}
    for path in files:
        relative = path.relative_to(source)
        target = destination / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"Refusing to overwrite a different asset: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Asset copy verification failed: {target}")
        records[str(relative)] = {
            "source": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    result = {
        "source": str(source),
        "destination": str(destination),
        "files": records,
        "notes": [
            "USD/MDL/URDF relative dependency trees and license notices are preserved.",
            "OmniPBR.mdl is resolved through Isaac Sim's built-in material search path.",
            "Files are independent copies; original assets are not modified.",
        ],
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Prepare assets using the project's Isaac Sim Python runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("assets/Demo"))
    parser.add_argument("--destination", type=Path, default=Path("assets/PhysicSolver"))
    args = parser.parse_args()
    result = prepare_assets(args.source, args.destination)
    get_logger(__name__).info("PhysicSolver assets ready", files=len(result["files"]))


if __name__ == "__main__":
    main()
