"""Copy the V7 parcel and all local dependencies without changing the source project."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger


def prepare(source: Path, target: Path, *, rigid: bool = False) -> None:
    """Make an independent copy and record hashes; refuse to overwrite different files."""
    asset_name = "source.usd" if rigid else "parcel_two_mesh.usd"
    if not (source / asset_name).is_file():
        raise FileNotFoundError(source / asset_name)
    files = [
        source / asset_name,
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
                "source_acceptance": "Source outcomes are not inherited; new runs need validation.",
                "license": "Source project terms apply; no new license granted.",
            },
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info("Independent parcel ready", target=str(target), files=len(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/home/fufa/Projects/package/output/simulation_v7")
    )
    parser.add_argument("--target", type=Path, default=Path("assets/PhysicSolver/Package/v7"))
    parser.add_argument("--rigid", action="store_true", help="Copy a V10 local_sources carton")
    args = parser.parse_args()
    prepare(args.source, args.target, rigid=args.rigid)
