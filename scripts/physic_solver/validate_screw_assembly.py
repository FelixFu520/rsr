"""Independently verify thread lead, sustained contact, final hold and optional RGB data."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.lerobot_v3 import validate_dataset


def validate(output: Path) -> dict:
    """Recompute thread acceptance from the saved measured joint trajectory."""
    config = json.loads((output / "configuration.json").read_text())
    result = json.loads((output / "result.json").read_text())
    rows = json.loads((output / "trajectory.json").read_text())
    settings = config["planner"]
    q = np.array([[row["axial_displacement_m"], row["spindle_angle_rad"]] for row in rows])
    t = np.array([row["time_s"] for row in rows])
    screw = [i for i, row in enumerate(rows) if row["phase"] == "screw"]
    hold = [i for i, row in enumerate(rows) if row["phase"] == "hold"]
    if len(q) < 2 or not screw or not hold or not np.isfinite(q).all():
        raise ValueError("Incomplete or non-finite screw measurements")
    np.testing.assert_allclose(np.diff(t), config["sim_world"]["physics_dt"], atol=1e-9)
    turns = abs(q[hold[0], 1] - q[screw[0], 1]) / (2 * np.pi)
    advance = q[screw[0], 0] - q[hold[0], 0]
    pitch = config["scene"]["thread_pitch_m"]
    pitch_error = abs(advance - turns * pitch) / max(turns * pitch, 1e-12)
    drift = float(np.ptp(q[hold, 0]))
    lead_residual = q[:, 0] - q[0, 0] - (q[:, 1] - q[0, 1]) * pitch / (2 * np.pi)
    unthreaded = float(np.max(np.abs(lead_residual)))
    contacts = np.array([rows[i]["contacts"] for i in screw])
    checks = dict(
        physical_result_passed=bool(result["success"]),
        sufficient_turns=bool(turns >= settings["minimum_turns"]),
        sufficient_advance=bool(advance >= settings["minimum_advance_m"]),
        measured_lead_matches_pitch=bool(pitch_error <= settings["maximum_pitch_error_fraction"]),
        sustained_thread_contact=bool(np.mean(contacts > 0) >= 0.95),
        stable_final_hold=drift <= settings["maximum_hold_drift_m"],
        no_unthreaded_push_through=unthreaded
        <= settings.get("maximum_unthreaded_advance_m", 0.002),
        completed_duration=bool(
            t[-1] >= sum(settings[key] for key in ("engage_s", "screw_s", "hold_s"))
        ),
        below_contact_capacity=bool(
            max(row["contacts"] for row in rows) < settings["contact_capacity"]
        ),
    )
    for key, measured in (
        ("turns", turns),
        ("axial_advance_m", advance),
        ("pitch_error_fraction", pitch_error),
    ):
        np.testing.assert_allclose(result[key], measured, rtol=1e-7, atol=1e-10)
    assets = {
        name: hashlib.sha256(Path(config["scene"][f"{name}_mesh_path"]).read_bytes()).hexdigest()
        for name in ("nut", "bolt")
    }
    report = dict(
        success=all(checks.values()),
        checks=checks,
        turns=float(turns),
        axial_advance_m=float(advance),
        pitch_error_fraction=float(pitch_error),
        hold_drift_m=drift,
        maximum_unthreaded_advance_m=unthreaded,
        asset_sha256=assets,
    )
    if (output / "lerobot_v3").exists():
        cameras = {
            camera["name"]: tuple(camera["resolution"]) for camera in config["scene"]["cameras"]
        }
        data = validate_dataset(output, expected_cameras=cameras)
        report["dataset"] = data
        import pyarrow.parquet as pq

        table = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
        indices = np.rint(
            np.array(table["timestamp"].to_pylist()) / config["sim_world"]["physics_dt"]
        ).astype(int)
        np.testing.assert_array_equal(
            np.array(table["observation.state"].to_pylist(), np.float32),
            q[indices].astype(np.float32),
        )
        commands = np.array(
            [
                [
                    q[i, 0]
                    if rows[i]["phase"] == "hold"
                    else config["robot"]["axial_preload_target_m"],
                    settings["spindle_speed_rad_s"] if rows[i]["phase"] == "screw" else 0.0,
                ]
                for i in indices
            ],
            np.float32,
        )
        np.testing.assert_array_equal(np.array(table["action"].to_pylist(), np.float32), commands)
        report["recorded_state_action_alignment"] = True
        qa = output / "qa"
        qa.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (960, 264), "#161616")
        draw = ImageDraw.Draw(sheet)
        video = next(iter(data["videos"].values()))
        for i, time_s in enumerate(
            (0, settings["engage_s"] + settings["screw_s"] * 0.5, t[-1] - 0.1)
        ):
            encoded = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    str(time_s),
                    "-i",
                    video,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-",
                ]
            )
            picture = Image.open(io.BytesIO(encoded)).convert("RGB").resize((320, 240))
            sheet.paste(picture, (i * 320, 24))
            draw.text((i * 320 + 8, 5), f"assembly t={time_s:.2f}s", fill="white")
        sheet.save(qa / "keyframes.jpg", quality=95)
        review = qa / "visual_review.json"
        report["visual_inspection"] = (
            "passed"
            if review.exists() and json.loads(review.read_text()).get("passed")
            else "pending"
        )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    get_logger(__name__).info("Screw assembly independently validated", **report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report = validate(parser.parse_args().output)
    raise SystemExit(0 if report["success"] else 1)
