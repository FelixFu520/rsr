"""Copy local Kimodo and text encoder checkpoints into the self-contained HALO assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from real2sim2real.core.logger import get_logger

LOG = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = (
    "Kimodo-G1-RP-v1",
    "Meta-Llama-3-8B-Instruct",
    "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
)


def sha256(path: Path) -> str:
    """Hash a checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Copy checkpoints, refuse different existing assets, and record source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Existing local models directory"
    )
    parser.add_argument("--destination", type=Path, default=ROOT / "assets/Halo/models/kimodo")
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate directory trees")
    for name in MODEL_NAMES:
        if not (source / name).is_dir():
            raise FileNotFoundError(source / name)
    records = []
    for name in MODEL_NAMES:
        LOG.info("copy_kimodo_model", model=name)
        for original in sorted((source / name).rglob("*")):
            if not original.is_file() or "__pycache__" in original.parts:
                continue
            relative = original.relative_to(source)
            target = destination / relative
            source_hash = sha256(original)
            if target.exists():
                if not target.is_file() or sha256(target) != source_hash:
                    raise FileExistsError(f"Refusing to overwrite different model file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                if sha256(target) != source_hash:
                    raise RuntimeError(f"Copy verification failed: {target}")
            records.append(
                {"path": str(relative), "bytes": target.stat().st_size, "sha256": source_hash}
            )
    manifest = {
        "source": str(source),
        "files": records,
        "modified": [],
        "notes": "Byte-for-byte copies. The generator loads the Llama base and both adapters "
        "explicitly; historical absolute paths in adapter metadata are not used.",
        "licenses": {
            "Kimodo-G1-RP-v1": "NVIDIA Open Model License; see checkpoint LICENSE",
            "Meta-Llama-3-8B-Instruct": "Meta Llama 3 Community License; see checkpoint LICENSE",
            "LLM2Vec": "McGill-NLP model cards and licenses retained with the adapters",
        },
    }
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info(
        "kimodo_models_ready",
        destination=str(destination),
        files=len(records),
        bytes=sum(item["bytes"] for item in records),
    )


if __name__ == "__main__":
    main()
