"""Independently check a tabletop replay output directory (no Kit).

Usage: ./app/python.sh scripts/tabletop_replay/validate_tabletop_replay.py <output_dir>

Runs every check of ``scripts/tabletop_pickplace/validate_tabletop_pickplace.py`` on each
``episode_XXX`` and then recomputes the replay fidelity from the published files: the replayed
``observation.state`` and final object pose against the source dataset named in ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from real2sim2real.utils.tabletop_replay_utils import (
    ReplayAcceptanceConfig,
    load_source_episode,
    replay_fidelity,
)
from scripts.tabletop_pickplace.validate_tabletop_pickplace import validate_episode


def validate_replay_episode(episode: Path) -> dict[str, Any]:
    report = validate_episode(episode)
    result = json.loads((episode / "result.json").read_text())
    root = episode / "lerobot_v3"
    configuration = json.loads(
        (root / "meta/physics/episode_000000/configuration.json").read_text()
    )
    info = json.loads((root / "meta/info.json").read_text())
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0))
    source = load_source_episode(Path(result["planner"]["source_dataset"]))
    if source.fps != info["fps"]:
        raise ValueError(f"Replay fps {info['fps']} differs from the source fps {source.fps}")
    states = np.asarray(data["observation.state"].to_pylist(), dtype=float)
    final = np.asarray(data["observation.object_pose"].to_pylist()[-1], dtype=float)
    acceptance = ReplayAcceptanceConfig(**configuration["job"]["replay_acceptance"])
    fidelity = replay_fidelity(states, final, source, acceptance)
    stored = {k: result["checks"][k] for k in fidelity["checks"]}
    if stored != fidelity["checks"] or not all(stored.values()):
        raise ValueError(f"Replay fidelity: recomputed {fidelity['checks']}, stored {stored}")
    report["replay"] = {"source_dataset": str(source.path), **fidelity}
    (episode / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def validate(output: Path) -> dict[str, Any]:
    run = json.loads((output / "result.json").read_text())
    episodes = sorted(p for p in output.glob("episode_*") if (p / "lerobot_v3").is_dir())
    if not run["success"] or len(episodes) != run["num_episodes"]:
        raise ValueError(f"Run did not pass: {run.get('error')} ({len(episodes)} datasets)")
    reports = [validate_replay_episode(p) for p in episodes]
    summary = {
        "validated": True,
        "episodes": [
            {
                "episode": r["episode"],
                "frames": r["frames"],
                "source_dataset": r["replay"]["source_dataset"],
                **r["replay"]["metrics"],
            }
            for r in reports
        ],
        "visual_review": "pending_visual_inspection",
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        summary = validate(args.output)
    except Exception as exc:  # noqa: BLE001 - CLI reports the failure and exits non-zero
        print(f"validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
