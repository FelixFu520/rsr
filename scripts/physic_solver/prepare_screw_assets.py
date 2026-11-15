"""Download the Newton example's Factory M20 thread meshes into the task asset closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from real2sim2real.core.logger import get_logger

REVISION = "aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3"
BASE = f"https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/{REVISION}/"
FILES = {
    "bolt.obj": "assets/factory/mesh/factory_nut_bolt/factory_bolt_m20_loose.obj",
    "nut.obj": "assets/factory/mesh/factory_nut_bolt/factory_nut_m20_loose_subdiv_3x.obj",
    "LICENSE.txt": "LICENSE.txt",
}


def prepare(target: Path) -> None:
    """Copy immutable local meshes; hashes identify the precise downloaded revision."""
    target.mkdir(parents=True, exist_ok=True)
    previous_path = target / "provenance.json"
    previous = json.loads(previous_path.read_text())["files"] if previous_path.exists() else {}
    manifest = {}
    for name, relative in FILES.items():
        destination = target / name
        if not destination.exists():
            with urllib.request.urlopen(BASE + relative, timeout=60) as response:
                content = response.read()
            destination.write_bytes(content)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if name in previous and digest != previous[name]["sha256"]:
            raise FileExistsError(f"Existing asset differs from its recorded hash: {destination}")
        manifest[name] = dict(
            url=BASE + relative, sha256=hashlib.sha256(destination.read_bytes()).hexdigest()
        )
    (target / "provenance.json").write_text(
        json.dumps(
            dict(
                source="NVIDIA IsaacGymEnvs Factory; same meshes as Newton nut_bolt_hydro example",
                files=manifest,
                geometry_modified=False,
                units="metres",
                license="See LICENSE.txt",
            ),
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info("Screw assets prepared", target=str(target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("assets/PhysicSolver/Screw/M20"))
    prepare(parser.parse_args().target)
