"""Validate the recorded parcel dataset and audit saved solver geometry without Kit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.lerobot_v3 import validate_dataset
from real2sim2real.utils.physic_solver_utils import check_package_containment


def audit(output: Path) -> dict:
    """Check data alignment, replayed geometry and explicitly labelled task outcome."""
    result = json.loads((output / "result.json").read_text())
    report = validate_dataset(
        output,
        allow_failure=True,
        expected_cameras={"overview": (640, 480), "wrist": (640, 480)},
    )
    runs = sorted((output / "diagnostics").glob("*/particles.npz"))
    if not runs:
        raise ValueError("No saved particle trajectory")
    path = runs[-1]
    configuration = json.loads((path.parent / "configuration.json").read_text())
    if json.loads((path.parent / "result.json").read_text()) != result:
        raise ValueError("Root result does not match the audited run")
    if (
        result["rigid_solver"] != configuration["sim_world"]["newton_solver"]
        or result["contact_solver"] != "vbd"
    ):
        raise ValueError("Unexpected solver selection")
    mode = configuration["robot"]["control_mode"]
    if mode == "position":
        names = [f"fr3_joint{i}_target_rad" for i in range(1, 8)] + [
            "left_finger_target_m",
            "right_finger_target_m",
        ]
    else:
        names = [f"fr3_joint{i}_velocity_rad_s" for i in range(1, 8)] + [
            "left_finger_velocity_m_s",
            "right_finger_velocity_m_s",
        ]
    info = json.loads((output / "lerobot_v3/meta/info.json").read_text())
    if info["features"]["action"]["names"] != names:
        raise ValueError("Action metadata does not match the active controller")
    if (
        configuration["robot"]["backend"] == "franka_newton"
        and result["gripper_geometry"] != "native_Franka_fingers"
    ):
        raise ValueError("Native Franka profile contains a different gripper")
    asset = Path(configuration["scene"]["cloth_usd_path"])
    if hashlib.sha256(asset.read_bytes()).hexdigest() != result["asset_sha256"]:
        raise ValueError("Parcel asset changed since collection")
    with np.load(path) as stored:
        positions, times = stored["positions"], stored["times"]
        triangles, masses = stored["triangles"], stored["masses"]
        n = int(stored["garment_count"])
        checkpoints = stored["checkpoint_times_s"].tolist()
    rows = [
        json.loads(line) for line in (path.parent / "trajectory.jsonl").read_text().splitlines()
    ]
    phase_times = {}
    for row in rows:
        phase_times.setdefault(row["phase"], []).append(row["time_s"])
    if not phase_times.get("transfer"):
        raise ValueError("No measured transfer phase")
    if not np.isfinite(positions).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid saved particle trajectory")
    coms = np.stack(
        [
            np.average(positions[:, :n], axis=1, weights=masses[:n]),
            np.average(positions[:, n:], axis=1, weights=masses[n:]),
        ],
        axis=1,
    )
    baseline = int(np.abs(times - checkpoints[1]).argmin())
    peaks = (coms[baseline:, :, 2] - coms[baseline, :, 2]).max(axis=0)
    dt = configuration["sim_world"]["physics_dt"]
    transfer = (times >= phase_times["transfer"][0]) & (
        times < phase_times["transfer"][-1] + dt / 2
    )
    minimum_transfer_lifts = (coms[transfer, :, 2] - coms[baseline, :, 2]).min(axis=0)
    transfer_lifted = bool(
        np.all(minimum_transfer_lifts >= configuration["planner"]["minimum_lift_m"])
    )
    containment = check_package_containment(
        positions, triangles, n, times=times, checkpoint_times_s=checkpoints
    )
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    selected_times = [
        0.0,
        phase_times["close"][-1],
        phase_times["lift"][-1],
        phase_times["transfer"][-1],
        phase_times["release"][-1],
        times[-1],
    ]
    if phase_times.get("shake"):
        start, end = phase_times["shake"][0], phase_times["shake"][-1]
        selected_times[4:4] = [start + 0.4 * (end - start), start + 0.6 * (end - start)]
    columns = 3
    panel_rows = (len(selected_times) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, panel_rows * 2 * 264), "#171717")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for row, name in enumerate(("overview", "wrist")):
        video = report["videos"][f"observation.images.{name}"]
        for column, time in enumerate(selected_times):
            frame = min(round(time * report["fps"]), report["frames"] - 1)
            png = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    video,
                    "-vf",
                    f"select=eq(n\\,{frame})",
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-threads",
                    "1",
                    "-",
                ]
            )
            picture = Image.open(io.BytesIO(png)).convert("RGB").resize((320, 240))
            x = (column % columns) * 320
            y = (row * panel_rows + column // columns) * 264
            sheet.paste(picture, (x, y + 24))
            draw.text((x + 6, y + 3), f"{name}   t={frame / report['fps']:.2f}s", font=font)
    sheet.save(qa / "keyframes.jpg", quality=95)
    audit_report = dict(
        success=bool(result["success"] and transfer_lifted and containment["passed"]),
        dataset_validated=True,
        physical_success=result["success"],
        dataset_use=report["dataset_use"],
        saved_particle_samples=len(times),
        all_saved_particles_finite=True,
        asset_sha256_verified=True,
        robot_backend=result["robot_backend"],
        gripper_geometry=result["gripper_geometry"],
        control_mode=mode,
        action_names_verified=True,
        sampled_maximum_com_lift_m=peaks.tolist(),
        sampled_transfer_minimum_com_lift_m=minimum_transfer_lifts.tolist(),
        sampled_transfer_both_meshes_lifted=transfer_lifted,
        full_rate_maximum_com_lift_m=result["maximum_com_lift_m"],
        containment=containment,
        failed_checks=[name for name, passed in result["checks"].items() if not passed],
        keyframes=str(qa / "keyframes.jpg"),
        particle_trajectory=str(path),
    )
    (qa / "physical_audit.json").write_text(json.dumps(audit_report, indent=2) + "\n")
    get_logger(__name__).info("Package dataset audited", **audit_report)
    if result["success"] and not audit_report["success"]:
        raise ValueError("Recorded package fails independent transfer or containment checks")
    return audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="outputs/physic_solver/mujoco_vbd_pickplace_package",
        type=Path,
    )
    audit(parser.parse_args().output)
