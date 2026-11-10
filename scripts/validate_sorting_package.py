"""Audit Pro-M6 LeRobot v3 video/state alignment and the recorded V7 parcel geometry."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.asset_utils import resolve_usd_path
from real2sim2real.utils.package_geometry import check_package_containment
from real2sim2real.utils.sorting_quality import (
    audit_fixed_robot_clearance,
    audit_held_shape,
    audit_jaw_penetration,
    without_bag_gap_checks,
)
from scripts.collect_physic_solver_cases import validate_dataset


def audit(output: Path) -> dict:
    """Recompute physical evidence and decode all videos without starting Isaac Kit."""
    output = output.resolve()
    result = json.loads((output / "result.json").read_text())
    candidates = sorted((output / "diagnostics").glob("*/particles.npz"))
    if not candidates:
        raise ValueError("No saved physical trajectory")
    run = candidates[-1].parent
    if json.loads((run / "result.json").read_text()) != result:
        raise ValueError("Dataset result does not match the audited physical run")
    source_hash = result.get("source_trajectory_sha256")
    if (
        source_hash
        and hashlib.sha256((run / "particles.npz").read_bytes()).hexdigest() != source_hash
    ):
        raise ValueError("Rendered trajectory differs from the recorded physical source")
    config = json.loads((run / "configuration.json").read_text())
    if "source_configuration_sha256" in result:
        source_config_path = Path(result["source_physics"]) / "configuration.json"
        if (
            hashlib.sha256(source_config_path.read_bytes()).hexdigest()
            != result["source_configuration_sha256"]
        ):
            raise ValueError("Source physical configuration changed after recording")
        expected = copy.deepcopy(json.loads(source_config_path.read_text()))
        overrides = result.get("observation_camera_pose_overrides", {})
        observed = set()
        for section in ("scene", "robot"):
            for camera in expected[section]["cameras"]:
                if camera["name"] in overrides:
                    camera["pose"] = overrides[camera["name"]]
                    observed.add(camera["name"])
        if observed != set(overrides) or expected != config:
            raise ValueError("Replay changed configuration beyond declared optical poses")
    cameras = {
        camera["name"]: tuple(camera["resolution"])
        for section in ("scene", "robot")
        for camera in config[section]["cameras"]
    }
    dataset = validate_dataset(output, allow_failure=True, expected_cameras=cameras)
    info = json.loads((output / "lerobot_v3/meta/info.json").read_text())
    if result["rigid_solver"] != "mujoco" or result["contact_solver"] != "vbd":
        raise ValueError("Sorting data must use MuJoCo + VBD")
    if (
        config["planner"]["backend"] != "sorting_package"
        or info["robot_type"] != "pro_m6_696_newton"
    ):
        raise ValueError("Unexpected robot/task backend")
    if (
        hashlib.sha256(
            Path(resolve_usd_path(config["scene"]["cloth_usd_path"])).read_bytes()
        ).hexdigest()
        != result["asset_sha256"]
    ):
        raise ValueError("V7 asset changed since simulation")
    robot_asset_hash = hashlib.sha256(
        Path(resolve_usd_path(config["robot"]["usd_path"])).read_bytes()
    ).hexdigest()
    if result.get("robot_asset_sha256", robot_asset_hash) != robot_asset_hash:
        raise ValueError("Robot asset changed since simulation")
    with np.load(run / "particles.npz") as archive:
        saved = {key: archive[key] for key in archive.files}
    x, times, rest, tri = [saved[k] for k in ("positions", "times", "rest", "triangles")]
    n, mass = int(saved["garment_count"]), saved["masses"]
    from pxr import Usd

    source = Usd.Stage.Open(resolve_usd_path(config["scene"]["cloth_usd_path"]))
    authored = {
        name: np.asarray(
            source.GetPrimAtPath(f"/World/Parcel/{name}Body/{name}")
            .GetAttribute("physics:masses")
            .Get(),
            float,
        )
        for name in ("Garment", "Bag")
    }
    expected_garment_mass = config["scene"].get("garment_mass_kg", 0.2)
    expected_bag_mass = config["scene"].get("bag_mass_kg")
    if expected_bag_mass is None:
        expected_bag_mass = float(authored["Bag"].sum())
    mass_distribution_matches = bool(
        n == len(authored["Garment"])
        and len(mass) - n == len(authored["Bag"])
        and np.allclose(
            mass[:n],
            authored["Garment"] * expected_garment_mass / authored["Garment"].sum(),
            rtol=1e-6,
            atol=1e-12,
        )
        and np.allclose(
            mass[n:],
            authored["Bag"] * expected_bag_mass / authored["Bag"].sum(),
            rtol=1e-6,
            atol=1e-12,
        )
    )
    if not np.isfinite(x).all() or not np.all(mass > 0) or not np.all(np.diff(times) > 0):
        raise ValueError("Nonfinite particles, fixed vertices or nonmonotonic physical time")
    data = pq.read_table(output / "lerobot_v3/data/chunk-000/file-000.parquet")
    targets = np.arange(len(data)) / info["fps"]
    indices = np.abs(times[:, None] - targets).argmin(0)
    np.testing.assert_allclose(times[indices], targets, atol=1e-7, rtol=0)
    for feature, key in [("observation.state", "state"), ("action", "action")]:
        np.testing.assert_allclose(
            data[feature].to_pylist(), saved[key][indices], atol=1e-6, rtol=0
        )
        if info["features"][feature]["shape"] != [16]:
            raise ValueError("Pro-M6 must record fourteen arm angles and two gripper angles")
    edges = np.unique(
        np.sort(np.concatenate([tri[:, :2], tri[:, 1:], tri[:, [2, 0]]]), axis=1), axis=0
    )
    lengths = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1)
    rt = rest[tri]
    areas = np.linalg.norm(np.cross(rt[:, 1] - rt[:, 0], rt[:, 2] - rt[:, 0]), axis=1)
    maximum_stretch, minimum_area, penetration = 1.0, 1.0, 0.0
    for vertices in x:
        maximum_stretch = max(
            maximum_stretch,
            float(
                np.max(
                    np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) / lengths
                )
            ),
        )
        faces = vertices[tri]
        minimum_area = min(
            minimum_area,
            float(
                np.min(
                    np.linalg.norm(
                        np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
                    )
                    / areas
                )
            ),
        )
        for low, high in saved["support_bounds"]:
            inside = np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2]), axis=1)
            if inside.any():
                penetration = max(penetration, float(high[2] - vertices[inside, 2].min()))
    planner = config["planner"]
    groups = {side: saved[f"{side}_transfer_indices"].astype(int) for side in ("left", "right")}
    video_indices = set(indices.tolist())
    video_transfer_samples = {
        side: sorted(set(group.tolist()) & video_indices) for side, group in groups.items()
    }
    shape = audit_held_shape(
        x,
        rest,
        tri,
        mass,
        n,
        groups,
        maximum_shape_rms_m=planner["maximum_shape_rms_m"],
        maximum_bag_gap_m=planner["maximum_bag_gap_m"],
        quality_reference=json.loads(str(saved.get("quality_reference_json", "null"))),
        extra_samples={
            side: video_transfer_samples[side]
            + [
                row["sample_index"]
                for row in result.get("online_held_shape_samples", [])
                if row["side"] == side
            ]
            for side in groups
        },
    )
    shape["video_transfer_frames_checked"] = {
        side: len(samples) for side, samples in video_transfer_samples.items()
    }
    shape["scope"] = (
        "All recorded video frames during each transfer, three physical transfer samples, "
        "and saved online checkpoints; all bag vertices and face centres."
    )
    containment = check_package_containment(
        x, tri, n, times=times, checkpoint_times_s=saved["checkpoint_times_s"].tolist()
    )
    checks = dict(
        garment_mass_matches_config=abs(
            float(mass[:n].sum()) - config["scene"].get("garment_mass_kg", 0.2)
        )
        < 1e-6,
        declared_proportional_mesh_masses=mass_distribution_matches,
        bag_mass_matches_config=abs(float(mass[n:].sum()) - expected_bag_mass) < 1e-6,
        native_robot_structure=config["robot"].get("pad_width_m", 0.03) == 0.03,
        surface_stretch_bounded=maximum_stretch < planner["maximum_edge_stretch"],
        no_collapsed_triangles=minimum_area > planner["minimum_triangle_area_ratio"],
        support_penetration_bounded=penetration <= planner["maximum_table_penetration_m"],
        garment_contained_at_checkpoints=containment["passed"],
        **shape["checks"],
    )
    friction = None
    if all(
        key in saved for key in ("shape_material_mu", "shape_body", "soft_contact_mu", "belt_body")
    ):
        particle_mu = float(saved["soft_contact_mu"])
        body = saved["shape_body"]
        fingers = np.isin(body, saved["jaw_body_ids"])
        supports = (body < 0) | (body == int(saved["belt_body"]))
        expected = np.where(supports, planner["table_friction"], planner["shape_contact_friction"])
        if planner.get("platform_friction") is not None:
            if "platform_shape" not in saved:
                raise ValueError("Missing platform collider mapping for declared platform friction")
            platform_shape = int(saved["platform_shape"])
            if not 0 <= platform_shape < len(body) or body[platform_shape] >= 0:
                raise ValueError("Platform friction must identify the static platform collider")
            expected[platform_shape] = planner["platform_friction"]
        target = planner.get("gripper_package_friction")
        if target is not None:
            expected[fingers] = target**2 / particle_mu if target > 0 else 0
        checks["declared_contact_friction_applied"] = bool(
            np.isclose(particle_mu, planner["cloth_friction"])
            and np.allclose(saved["shape_material_mu"], expected, rtol=1e-6, atol=1e-8)
        )
        friction = dict(
            effective_gripper_package_mu=np.unique(
                np.sqrt(particle_mu * saved["shape_material_mu"][fingers])
            ).tolist(),
            effective_support_package_mu=np.unique(
                np.sqrt(particle_mu * saved["shape_material_mu"][supports])
            ).tolist(),
        )
        if planner.get("platform_friction") is not None:
            friction["platform_material_mu"] = float(saved["shape_material_mu"][platform_shape])
            friction["effective_platform_package_mu"] = float(
                np.sqrt(particle_mu * saved["shape_material_mu"][platform_shape])
            )
    jaws = [
        dict(
            body_id=int(body),
            vertices=saved[f"jaw_vertices_{i}"],
            triangles=saved[f"jaw_triangles_{i}"],
        )
        for i, body in enumerate(saved["jaw_body_ids"])
    ]
    finger_geometry = audit_jaw_penetration(
        x, tri, saved["body_poses"], jaws, tolerance_m=planner["maximum_table_penetration_m"]
    )
    checks["all_saved_vertices_and_face_centres_clear_of_fingers"] = finger_geometry["passed"]
    mimic = saved["mimic_pairs"]
    residual = saved["articulation_q"][:, mimic[:, 0].astype(int)] - (
        saved["articulation_q"][:, mimic[:, 1].astype(int)] * mimic[:, 2]
    )
    checks["mimic_constraints_maintained"] = bool(np.max(np.abs(residual)) < 0.03)
    native_mimic_audit = None
    if "native_mimic_residuals" in saved and saved["native_mimic_residuals"].size:
        native = saved["native_mimic_residuals"]
        steps = config["sim_world"]["newton_substeps"] * planner.get(
            "mujoco_substeps_per_vbd_step", 1
        )
        expected_dt = config["sim_world"]["physics_dt"] / steps
        checks["native_substep_mimic_recording_complete"] = bool(
            native.shape == ((len(times) - 1) * steps, len(mimic))
            and np.isfinite(native).all()
            and np.isclose(float(saved["native_mimic_dt_s"]), expected_dt, rtol=1e-9)
        )
        checks["native_substep_mimic_constraints_maintained"] = bool(np.max(np.abs(native)) < 0.03)
        native_mimic_audit = dict(
            samples=len(native),
            dt_s=float(saved["native_mimic_dt_s"]),
            maximum_error_rad=float(np.max(np.abs(native))),
        )
    rows = [json.loads(line) for line in (run / "trajectory.jsonl").read_text().splitlines()]
    placements = {}
    fixed_robot = [
        dict(
            shape_id=int(shape),
            vertices=saved[f"fixed_robot_vertices_{i}"],
            triangles=saved[f"fixed_robot_triangles_{i}"],
        )
        for i, shape in enumerate(saved.get("fixed_robot_shape_ids", []))
    ]
    for side, support_index in (("left", 1), ("right", 2)):
        resting = [row for row in rows if row["phase"] == f"{side}_rest"]
        checks[f"{side}_rest_observed"] = bool(resting)
        if not resting:
            continue
        sample = int(np.abs(times - resting[-1]["time_s"]).argmin())
        vertices = x[sample]
        low, high = saved["support_bounds"][support_index]
        goal = (low + high) / 2
        if side == "right":
            goal[:2] = config["scene"]["outfeed_target_xy"]
        elif config["scene"].get("platform_target_xy") is not None:
            goal[:2] = config["scene"]["platform_target_xy"]
        coms = np.array(
            [
                np.average(vertices[:n], axis=0, weights=mass[:n]),
                np.average(vertices[n:], axis=0, weights=mass[n:]),
            ]
        )
        label = vertices[saved["label_triangles"]]
        normal = np.cross(label[:, 1] - label[:, 0], label[:, 2] - label[:, 0]).sum(0)
        label_up = float(normal[2] / max(np.linalg.norm(normal), 1e-12))
        footprint = bool(np.all((vertices[:, :2] >= low[:2]) & (vertices[:, :2] <= high[:2])))
        error = float(np.max(np.linalg.norm(coms[:, :2] - goal[:2], axis=1)))
        height = float(np.max(coms[:, 2] - high[2]))
        fixed_clearance = (
            audit_fixed_robot_clearance(
                vertices,
                tri,
                fixed_robot,
                float(np.max(saved["particle_radius"])),
                saved["shape_margin"],
            )
            if fixed_robot
            else None
        )
        checks[f"{side}_released_from_entire_robot"] = bool(
            fixed_clearance and fixed_clearance["passed"] and resting[-1]["robot_contacts"] == 0
        )
        checks[f"{side}_label_up"] = label_up >= planner["label_up_min_cos"]
        checks[f"{side}_destination_support"] = (
            footprint and error <= planner["maximum_place_error_m"] and height < 0.07
        )
        placements[side] = dict(
            label_up_cos=label_up,
            footprint_supported=footprint,
            maximum_com_xy_error_m=error,
            maximum_com_height_m=height,
            fixed_robot_clearance=fixed_clearance,
        )
    outfeed_low, outfeed_high = saved["support_bounds"][2]
    final_coms = np.array(
        [
            np.average(x[-1, :n], axis=0, weights=mass[:n]),
            np.average(x[-1, n:], axis=0, weights=mass[n:]),
        ]
    )
    checks["parcel_exited_outfeed"] = bool(
        x[-1, :, 0].max() < outfeed_low[0]
        and np.all((final_coms[:, 1] >= outfeed_low[1]) & (final_coms[:, 1] <= outfeed_high[1]))
    )
    strict_checks = checks.copy()
    if not planner.get("require_bag_gap_quality", True):
        checks = without_bag_gap_checks(checks, shape)
    if result["success"] and not all(checks.values()):
        raise ValueError(f"Successful label contradicts independent geometry audit: {checks}")
    selected = [0.0]
    for phase in (
        "left_close",
        "left_transfer",
        "left_release",
        "right_close",
        "right_transfer",
        "right_release",
        "right_conveyor",
    ):
        matching = [row["time_s"] for row in rows if row["phase"] == phase]
        if matching:
            selected.append(matching[len(matching) // 2])
    # A failed lift can precede every transfer/release checkpoint. Always show
    # the last actual video frame so the QA sheet contains the stopping state.
    selected = sorted(set([*selected, float(targets[-1])]))
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    columns = 4
    panel_rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (320 * columns, 264 * panel_rows * len(cameras)), "#181818")
    draw = ImageDraw.Draw(sheet)
    for view, name in enumerate(cameras):
        video = dataset["videos"][f"observation.images.{name}"]
        for column, time in enumerate(selected):
            frame = min(round(time * info["fps"]), len(data) - 1)
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
            left, top = (column % columns) * 320, (view * panel_rows + column // columns) * 264
            sheet.paste(picture, (left, top + 24))
            draw.text((left + 6, top + 4), f"{name}  t={frame / info['fps']:.2f}s", fill="white")
    sheet.save(qa / "keyframes.jpg", quality=95)
    montage = qa / "synchronized_four_view.mp4"
    inputs = [
        value
        for name in ("overview", "head", "left_wrist", "right_wrist")
        for value in ("-i", dataset["videos"][f"observation.images.{name}"])
    ]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            str(montage),
        ],
        check=True,
        capture_output=True,
    )
    report = dict(
        dataset_validated=True,
        physical_success=result["success"],
        acceptance_profile=result.get("acceptance_profile", "strict_v7"),
        workflow_success=result.get("workflow_success", result["success"]),
        strict_quality_success=result.get("strict_quality_success", result["success"]),
        dataset_use=dataset["dataset_use"],
        video_frames=len(data),
        physics_samples=len(times),
        contact_friction=friction,
        garment_mass_kg=float(mass[:n].sum()),
        bag_mass_kg=float(mass[n:].sum()),
        state_action_video_alignment_verified=True,
        observation_camera_pose_overrides=result.get("observation_camera_pose_overrides", {}),
        independent_geometry_checks=checks,
        independent_geometry_passed=all(checks.values()),
        strict_independent_geometry_checks=strict_checks,
        strict_independent_geometry_passed=all(strict_checks.values()),
        maximum_edge_stretch=maximum_stretch,
        minimum_triangle_area_ratio=minimum_area,
        maximum_support_penetration_m=penetration,
        held_shape_audit=shape,
        finger_geometry_audit=finger_geometry,
        maximum_sampled_mimic_error_rad=float(np.max(np.abs(residual))),
        native_substep_mimic_audit=native_mimic_audit,
        independent_placements=placements,
        containment=containment,
        failed_checks=[name for name, value in result["checks"].items() if not value],
        videos=dataset["videos"],
        keyframes=str(qa / "keyframes.jpg"),
        synchronized_four_view=str(montage),
        scope=(
            "Geometry at 60 Hz; containment at phase boundaries; shape/gap at every recorded "
            "transfer video frame plus physical checkpoints. No continuous-time proof "
            "or hardware load certification."
        ),
    )
    (qa / "physical_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    get_logger(__name__).info(
        "Sorting video and physics audited",
        physical_success=result["success"],
        failed_checks=report["failed_checks"],
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("outputs/PhysicSolver/mujoco_vbd_sorting_package"),
    )
    parser.add_argument("--usd-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.usd_worker:
        root = Path(sys.executable).resolve().parents[3]
        libraries = sorted(root.glob("extscache/omni.usd.libs-*/pxr/Gf"))
        if not libraries:
            raise RuntimeError("Run this tool with the installed Isaac Sim python.sh")
        library = libraries[-1].parent.parent
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(library) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["LD_LIBRARY_PATH"] = (
            str(library / "bin") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
        raise SystemExit(
            subprocess.call(
                [sys.executable, __file__, *sys.argv[1:], "--usd-worker"], env=environment
            )
        )
    report = audit(args.output)
    raise SystemExit(
        0 if report["physical_success"] and report["independent_geometry_passed"] else 1
    )
