"""Copy the selected V10 carton and dependencies, recording immutable source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare(source: Path, destination: Path) -> None:
    """Copy the normalized box02 asset without changing its geometry or textures."""
    source = source / "local_sources" / "box02"
    files = [source / "source.usd", *sorted((source / "textures").rglob("*"))]
    manifest = {}
    for original in files:
        if original.is_dir():
            continue
        relative = original.relative_to(source)
        target = destination / relative
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise FileExistsError(f"Refusing to overwrite a different asset: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "provenance.json").write_text(
        json.dumps(
            dict(
                source=str(source.resolve()),
                geometry_modified=False,
                sha256=manifest,
                scale_application="Scene config rigid_scale; source file remains unchanged",
            ),
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info(
        "V10 carton prepared", destination=str(destination), files=len(manifest)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/fufa/Projects/package/output/simulation_v10")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("assets/PhysicSolver/Package/v10_box02")
    )
    args = parser.parse_args()
    prepare(args.source, args.destination)
