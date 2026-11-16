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


def _validate_arm_dataset(output: Path, config: dict, rows: list[dict]) -> dict:
    """Check driver data without assuming a grasping arm must move 0.1rad."""
    import pyarrow.parquet as pq

    root = output / "lerobot_v3"
    info = json.loads((root / "meta/info.json").read_text())
    episode = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    table = pq.read_table(root / "data/chunk-000/file-000.parquet")
    fps, frames = info["fps"], info["total_frames"]
    if info["codebase_version"] != "v3.0" or len(table) != frames or episode["length"] != frames:
        raise ValueError("Incomplete UR5e dataset")
    if not episode["success"] or episode["dataset_use"] != "demonstration":
        raise ValueError("UR5e episode is not an accepted demonstration")
    np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(frames))
    times = table["timestamp"].to_numpy()
    np.testing.assert_allclose(times, np.arange(frames) / fps, atol=2e-6)
    indices = np.rint(times / config["sim_world"]["physics_dt"]).astype(int)
    state_size = 16 if config["robot"]["backend"] == "dual_ur5e_screwdriver" else 8
    for key, field in (("observation.state", "joint_q"), ("action", "action")):
        values = np.array(table[key].to_pylist(), np.float32)
        expected = np.array([rows[i][field] for i in indices], np.float32)
        if values.shape != (frames, state_size) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}")
        if len(info["features"][key]["names"]) != state_size:
            raise ValueError(f"Missing coordinate names: {key}")
        np.testing.assert_array_equal(values, expected)
    videos = {}
    for camera in config["scene"]["cameras"]:
        name = camera["name"]
        path = next((root / f"videos/observation.images.{name}").rglob("*.mp4"))
        stream = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_read_frames,codec_name",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["streams"][0]
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        if (
            stream["codec_name"] != "h264"
            or int(stream["nb_read_frames"]) != frames
            or [stream["width"], stream["height"]] != camera["resolution"]
            or abs(numerator / denominator - fps) > 1e-6
        ):
            raise ValueError(f"Invalid video metadata: {name}")
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            capture_output=True,
            check=True,
        )
        if decoded.stderr:
            raise ValueError(f"Video decoding errors: {name}")
        videos[name] = str(path)
    return {"videos": videos, "frames": frames, "fps": fps, "recorded_state_action_alignment": True}


def validate(output: Path) -> dict:
    """Recompute thread acceptance from the saved measured joint trajectory."""
    config = json.loads((output / "configuration.json").read_text())
    if config["robot"]["backend"] == "dual_ur5e_screwdriver":
        return validate_dual(output, config)
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
    arm = config["robot"]["backend"] == "ur5e_screwdriver"
    if arm:
        arm_q = np.array([row["arm_q_rad"] for row in rows])
        targets = np.array([row["arm_target_rad"] for row in rows])
        checks["arm_tracks_driver"] = bool(
            np.isfinite(arm_q).all()
            and np.max(np.abs(arm_q - targets)) <= settings["maximum_arm_tracking_error_rad"]
        )
        measured = np.array([row["bolt_position_local_m"][2] for row in rows])
        np.testing.assert_allclose(q[:, 0] - q[0, 0], measured - measured[0], atol=1e-9)
        task_root = Path("assets/PhysicSolver").resolve()
        checks["task_local_asset_layers"] = bool(result["asset_layers"]) and all(
            Path(layer).resolve().is_relative_to(task_root) for layer in result["asset_layers"]
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
        data = (
            _validate_arm_dataset(output, config, rows)
            if arm
            else validate_dataset(output, expected_cameras=cameras)
        )
        report["dataset"] = data
        import pyarrow.parquet as pq

        table = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
        indices = np.rint(
            np.array(table["timestamp"].to_pylist()) / config["sim_world"]["physics_dt"]
        ).astype(int)
        recorded = np.array([row["joint_q"] for row in rows]) if arm else q
        np.testing.assert_array_equal(
            np.array(table["observation.state"].to_pylist(), np.float32),
            recorded[indices].astype(np.float32),
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
        if arm:
            commands = np.array([rows[i]["action"] for i in indices], np.float32)
        np.testing.assert_array_equal(np.array(table["action"].to_pylist(), np.float32), commands)
        report["recorded_state_action_alignment"] = True
        qa = output / "qa"
        qa.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (960, 264 * len(data["videos"])), "#161616")
        draw = ImageDraw.Draw(sheet)
        for view_row, (view, video) in enumerate(data["videos"].items()):
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
                if np.asarray(picture).max() < 5 or np.asarray(picture).std() < 1:
                    raise ValueError(f"Blank review frame: {view} at {time_s}")
                y = view_row * 264
                sheet.paste(picture, (i * 320, y + 24))
                draw.text((i * 320 + 8, y + 5), f"{view} t={time_s:.2f}s", fill="white")
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


def validate_dual(output: Path, config: dict) -> dict:
    """Recompute free-nut pickup and relative thread motion from saved body poses."""
    from real2sim2real.utils.battery_disassembly_utils import quat_to_matrix
    from real2sim2real.utils.physic_solver_utils import (
        dual_screw_phase,
        evaluate_dual_screw_rows,
        read_obj_triangles,
    )

    result = json.loads((output / "result.json").read_text())
    rows = json.loads((output / "trajectory.json").read_text())
    settings = config["planner"]
    times = np.array([row["time_s"] for row in rows])
    np.testing.assert_allclose(np.diff(times), config["sim_world"]["physics_dt"], atol=1e-9)
    vertices, _ = read_obj_triangles(config["scene"]["nut_mesh_path"])
    mesh_center = (vertices.min(0) + vertices.max(0)) * 0.5
    mesh_center *= [1, -1, -1]  # Nut mesh is flipped in the free body's local frame.
    nut_bottom = -vertices[:, 2].max()
    bolt_vertices, _ = read_obj_triangles(config["scene"]["bolt_mesh_path"])
    bolt_tip = np.array([0, 0, -bolt_vertices[:, 2].max()])
    raw_yaw, raw_z = [], []
    for row in rows:
        if row["phase"] != dual_screw_phase(row["time_s"], settings)[0]:
            raise ValueError("Phase order disagrees with recorded physical time")
        nut = np.array(row["nut_pose_local_xyzw"])
        bolt = np.array(row["bolt_pose_local_xyzw"])
        holder = np.array(row["holder_pose_local_xyzw"])
        nr, br, hr = [quat_to_matrix(pose[[6, 3, 4, 5]]) for pose in (nut, bolt, holder)]
        relative = nr.T @ br
        raw_yaw.append(np.arctan2(relative[1, 0], relative[0, 0]))
        raw_z.append((nr.T @ (bolt[:3] - nut[:3]))[2])
        if settings.get("minimum_tip_protrusion_m", 0) > 0:
            measured_tip = nut_bottom - (nr.T @ (bolt[:3] + br @ bolt_tip - nut[:3]))[2]
            np.testing.assert_allclose(row["tip_protrusion_m"], measured_tip, atol=1e-9)
        np.testing.assert_allclose(row["nut_center_local_m"], nut[:3] + nr @ mesh_center, atol=1e-7)
        np.testing.assert_allclose(
            row["nut_relative_holder_m"], hr.T @ (nut[:3] - holder[:3]), atol=1e-9
        )
    np.testing.assert_allclose(
        [r["spindle_angle_rad"] for r in rows], np.unwrap(raw_yaw), atol=1e-8
    )
    np.testing.assert_allclose([r["axial_displacement_m"] for r in rows], raw_z, atol=1e-9)
    report = evaluate_dual_screw_rows(rows, settings, config["scene"]["thread_pitch_m"])
    for key in (
        "turns",
        "axial_advance_m",
        "pitch_error_fraction",
        "minimum_nut_lift_m",
        "maximum_nut_grasp_slip_m",
        "bilateral_grip_fraction",
    ):
        np.testing.assert_allclose(report[key], result[key], rtol=1e-7, atol=1e-9)
    task_root = Path("assets/PhysicSolver").resolve()
    if settings.get("minimum_tip_protrusion_m", 0) > 0:
        np.testing.assert_allclose(
            report["minimum_hold_tip_protrusion_m"],
            result["minimum_hold_tip_protrusion_m"],
            atol=1e-9,
        )
    if settings.get("release_s", 0) > 0:
        for key in (
            "minimum_released_finger_m",
            "released_thread_contact_fraction",
            "minimum_released_tip_protrusion_m",
        ):
            np.testing.assert_allclose(report[key], result[key], atol=1e-9)
    report["checks"].update(
        job_physical_checks_passed=bool(result["success"]),
        nut_not_attached=result["nut_attachments_used"] is False,
        below_contact_capacity=result["maximum_contacts"] < settings["contact_capacity"],
        task_local_asset_layers=bool(result["asset_layers"])
        and all(
            Path(layer).resolve().is_relative_to(task_root) for layer in result["asset_layers"]
        ),
    )
    if texture_path := config["scene"].get("bolt_texture_path"):
        texture = Path(texture_path).resolve()
        manifest = json.loads((texture.parent / "provenance.json").read_text())
        digest = hashlib.sha256(texture.read_bytes()).hexdigest()
        report["bolt_texture_sha256"] = digest
        report["checks"]["task_local_bolt_texture_verified"] = bool(
            texture.is_relative_to(task_root)
            and digest == manifest["files"][texture.name]["sha256"]
        )
    if (output / "lerobot_v3").exists():
        data = _validate_arm_dataset(output, config, rows)
        report["dataset"] = data
        report["recorded_state_action_alignment"] = True
        qa = output / "qa"
        qa.mkdir(exist_ok=True)
        start_screw = sum(settings[f"{name}_s"] for name in ("clamp", "lift", "align", "engage"))
        captures = [
            0.0,
            settings["clamp_s"],
            settings["clamp_s"] + settings["lift_s"],
            start_screw + settings["screw_s"] * 0.5,
            times[-1] - 0.1,
        ]
        if settings.get("release_s", 0) > 0:
            release_start = start_screw + settings["screw_s"] + settings["hold_s"]
            captures[-1] = release_start - 0.1
            captures.extend([release_start + settings["release_s"] * 0.5, times[-1] - 0.1])
        sheet = Image.new("RGB", (320 * len(captures), 264 * len(data["videos"])), "#161616")
        draw = ImageDraw.Draw(sheet)
        for j, (name, video) in enumerate(data["videos"].items()):
            for i, t in enumerate(captures):
                encoded = subprocess.check_output(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        str(t),
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
                im = Image.open(io.BytesIO(encoded)).convert("RGB").resize((320, 240))
                if np.asarray(im).max() < 5 or np.asarray(im).std() < 1:
                    raise ValueError(f"Blank dual-arm review frame: {name}, {t}")
                sheet.paste(im, (i * 320, j * 264 + 24))
                draw.text(
                    (i * 320 + 8, j * 264 + 5),
                    f"{name} {dual_screw_phase(t, settings)[0]} t={t:.2f}s",
                    fill="white",
                )
        sheet.save(qa / "keyframes.jpg", quality=95)
        review = qa / "visual_review.json"
        report["visual_inspection"] = (
            "passed"
            if review.exists() and json.loads(review.read_text()).get("passed")
            else "pending"
        )
    report["success"] = all(report["checks"].values())
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    get_logger(__name__).info("Dual-arm nut pickup and threading independently validated", **report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report = validate(parser.parse_args().output)
    raise SystemExit(0 if report["success"] else 1)
