"""Copy the installed official Newton shirt, license, and provenance without modifying originals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from real2sim2real.core.logger import get_logger


def prepare_official_cloth(*, verify_upstream: bool = False) -> Path:
    """Create a verified, independent asset copy from Isaac Sim's Newton distribution."""
    packages = Path("app/exts/isaacsim.pip.newton/pip_prebundle").resolve()
    source = packages / "newton/examples/assets/unisex_shirt.usd"
    license_path = next(packages.glob("newton-*.dist-info/licenses/LICENSE.md"))
    target = Path("assets/PhysicSolver/Cloth/newton_official")
    target.mkdir(parents=True, exist_ok=True)
    for original, name in ((source, "unisex_shirt.usd"), (license_path, "LICENSE.md")):
        destination = target / name
        if destination.exists() and destination.read_bytes() != original.read_bytes():
            raise FileExistsError(
                f"Refusing to overwrite a different official asset: {destination}"
            )
        shutil.copy2(original, destination)
    # Read the matching package metadata, independently of Python's current import path.
    metadata = importlib.metadata.PathDistribution(license_path.parent.parent)
    version = metadata.version
    manifest = {
        "source": str(source),
        "newton_version": version,
        "official_repository": "https://github.com/newton-physics/newton",
        "source_relative_path": "newton/examples/assets/unisex_shirt.usd",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": source.stat().st_size,
        "license": "Apache-2.0",
        "geometry_modified": False,
    }
    provenance_path = target / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous.get("sha256") == manifest["sha256"]:
            for key in ("upstream_verified_url", "upstream_sha256"):
                if key in previous:
                    manifest[key] = previous[key]
    if verify_upstream:
        url = (
            "https://raw.githubusercontent.com/newton-physics/newton/main/"
            "newton/examples/assets/unisex_shirt.usd"
        )
        with urlopen(url, timeout=20) as response:
            digest = hashlib.sha256(response.read()).hexdigest()
        if digest != manifest["sha256"]:
            raise ValueError("Upstream asset differs from the installed Newton release")
        manifest.update(upstream_verified_url=url, upstream_sha256=digest)
    provenance_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return target / "unisex_shirt.usd"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-upstream", action="store_true", help="Compare GitHub asset SHA-256"
    )
    args = parser.parse_args()
    get_logger(__name__).info(
        "Official Newton cloth ready",
        path=str(prepare_official_cloth(verify_upstream=args.verify_upstream)),
    )
