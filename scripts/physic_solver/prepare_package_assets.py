"""Copy the V7 parcel and all local dependencies without changing the source project."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare(source: Path, target: Path) -> None:
    """Make an independent copy and record hashes; refuse to overwrite different files."""
    if not (source / "parcel_two_mesh.usd").is_file():
        raise FileNotFoundError(source / "parcel_two_mesh.usd")
    files = [
        source / "parcel_two_mesh.usd",
        source / "source_configuration.json",
        source / "README.md",
        source / "usd_rerun_validation.json",
    ]
    for directory in ("textures", "materials"):
        files.extend(sorted((source / directory).rglob("*")))
    manifest = {}
    for original in files:
        if not original.is_file():
            continue
        relative = original.relative_to(source)
        destination = target / relative
        if destination.exists() and destination.read_bytes() != original.read_bytes():
            raise FileExistsError(f"Refusing to overwrite different asset: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        manifest[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (target / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "sha256": manifest,
                "geometry_modified": False,
                "source_acceptance": "USD rerun incomplete; see source README",
            },
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info(
        "Independent V7 parcel ready", target=str(target), files=len(manifest)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/fufa/Projects/package/output/simulation_v7")
    )
    parser.add_argument("--target", type=Path, default=Path("assets/PhysicSolver/Package/v7"))
    args = parser.parse_args()
    prepare(args.source, args.target)
