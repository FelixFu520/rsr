"""Independently check a fire-inspection output directory and its LeRobot v3 dataset.

Usage: ./app/python.sh scripts/validate_fire_inspection.py outputs/xunjian-debug
Writes ``validation.json`` and ``qa/keyframes.jpg``; exits non-zero when a check fails.
Needs only numpy, pyarrow, Pillow and the system ffmpeg/ffprobe (no Kit).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from real2sim2real.utils.patrol_geometry import fire_pixel_score


def _ffprobe(video: Path) -> dict:
    out = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height,codec_name,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(out.stdout)["streams"][0]


def _frames(video: Path, first: int, count: int, size: tuple[int, int]) -> np.ndarray:
    """Decode ``count`` frames starting at ``first`` as an (N, H, W, 3) uint8 array."""
    width, height = size
    raw = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(video),
            "-vf", f"select='between(n,{first},{first + count - 1})'", "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True, capture_output=True, timeout=300,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def _red_fraction(frames: np.ndarray) -> np.ndarray:
    rgb = frames.astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return ((r > 60) & (r > 2 * g) & (r > 2 * b)).mean(axis=(1, 2))


def _check_alarm(dataset, result, planner, names, action, xy, yaw, fps, checks, details) -> None:
    """Stop facing the fire for detect_hold_s, strobe for alarm_s, all before the photo."""
    alarm = action[:, names.index("alarm")] > 0.5
    frames = np.flatnonzero(alarm)
    checks["alarm_recorded"] = frames.size > 0
    if not frames.size:
        return
    start, end = int(frames[0]), int(frames[-1])
    details["alarm_frames"] = [start, end]
    checks["alarm_contiguous"] = frames.size == end - start + 1
    expected = planner["alarm_s"] * fps
    checks["alarm_duration"] = abs(frames.size - expected) <= 1
    moving = (np.abs(action[:, names.index("cmd_vx")]) > 1e-3) | (
        np.abs(action[:, names.index("cmd_wz")]) > 1e-3
    )
    checks["alarm_robot_still"] = not moving[start : end + 1].any()
    still_before = 0
    while start - still_before - 1 >= 0 and not moving[start - still_before - 1]:
        still_before += 1
    details["still_before_alarm_s"] = still_before / fps
    checks["held_before_alarm"] = still_before >= round(planner["detect_hold_s"] * fps) - 1
    fire = np.asarray(result["photo"]["fire_world_xyz"][:2]) if result.get("photo") else None
    if fire is not None:
        bearing = np.arctan2(fire[1] - xy[start, 1], fire[0] - xy[start, 0])
        error = abs(float(np.angle(np.exp(1j * (bearing - yaw[start])))))
        details["alarm_heading_error_rad"] = error
        checks["alarm_facing_fire"] = error <= planner["face_tolerance_rad"] + 0.01
    photo = details.get("photo_frame")
    checks["alarm_before_photo"] = photo is not None and end < photo
    # The strobe must be visible: red pixels in the follow view rise when the beacon is on.
    key = "observation.images.follow"
    height, width = json.loads((dataset / "meta/info.json").read_text())["features"][key]["shape"][
        :2
    ]
    video = dataset / "videos" / key / "chunk-000/file-000.mp4"
    lead = round(planner["detect_hold_s"] * fps / 2)
    clip = _frames(video, start - lead, lead + frames.size + 1, (width, height))
    red = _red_fraction(clip)
    hold_red = float(red[:lead].mean())
    strobe = red[lead:]  # alarm frames plus one (the beacon shows one frame after the command)
    # The beacon toggles on physics steps, not frame boundaries, so count flashes from the
    # measured signal instead of assuming an on/off pattern per frame.
    peak, low = float(strobe.max()), float(strobe.min())
    threshold = low + 0.25 * (peak - low)  # lit frames vary ±50 % with render noise
    lit = strobe >= threshold
    flashes = int(lit[0]) + int(np.sum(lit[1:] & ~lit[:-1]))
    expected_flashes = round(planner["alarm_s"] * planner["alarm_flash_hz"])
    details["follow_red_fraction"] = {
        "hold": hold_red,
        "alarm_peak": peak,
        "alarm_min": low,
        "flashes": flashes,
        "expected_flashes": expected_flashes,
    }
    checks["beacon_visible_in_video"] = bool(
        peak > 0.005 and peak > 10 * hold_red and low < 0.75 * peak
    )
    checks["beacon_flash_count"] = abs(flashes - expected_flashes) <= 1


def validate(output: Path) -> dict:
    checks: dict[str, bool] = {}
    details: dict = {}
    result = json.loads((output / "result.json").read_text())
    dataset = output / "lerobot_v3"
    checks["result_success"] = bool(result.get("success"))
    checks["dataset_exists"] = (dataset / "meta/info.json").is_file()
    if not checks["dataset_exists"]:
        return {"checks": checks, "details": details, "passed": False}

    info = json.loads((dataset / "meta/info.json").read_text())
    fps = int(info["fps"])
    table = pq.read_table(dataset / "data/chunk-000/file-000.parquet").to_pydict()
    frames = len(table["frame_index"])
    state = np.asarray(table["observation.state"], dtype=float)
    action = np.asarray(table["action"], dtype=float)
    timestamps = np.asarray(table["timestamp"], dtype=float)
    checks["codebase_v3"] = info.get("codebase_version") == "v3.0"
    checks["frame_count_matches_info"] = frames == info["total_frames"]
    checks["timestamps_regular"] = bool(
        np.allclose(np.diff(timestamps), 1 / fps, atol=1e-4) and timestamps[0] == 0
    )
    checks["state_action_finite"] = bool(np.isfinite(state).all() and np.isfinite(action).all())
    names = info["features"]["action"]["names"]
    photo_flags = action[:, names.index("photo_trigger")]
    checks["single_photo_trigger"] = int(photo_flags.sum()) == 1
    details["photo_frame"] = int(np.argmax(photo_flags)) if photo_flags.any() else None
    state_names = info["features"]["observation.state"]["names"]
    xy = state[:, [state_names.index("base_x"), state_names.index("base_y")]]
    details["path_length_m"] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * fps
    details["max_speed_mps"] = float(speed.max()) if len(speed) else 0.0
    outcome = json.loads((dataset / "meta/task_outcome.json").read_text())
    planner = outcome["configuration"]["planner"]
    limit = 1.25 * max(planner["cruise_speed_mps"], planner["approach_speed_mps"])
    details["speed_limit_mps"] = limit
    checks["base_speed_within_limit"] = details["max_speed_mps"] <= limit

    # Task string is the operator's sentence; one task, one episode.
    tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pydict()
    details["task"] = tasks["task"]
    checks["task_is_command"] = tasks["task"] == [result["command"]["text"]]
    checks["single_episode"] = info["total_episodes"] == 1 and set(table["episode_index"]) == {0}

    # Recorded velocity commands must explain the recorded base motion (body-frame vx, wz).
    yaw = state[:, state_names.index("base_yaw")]
    vx = action[:, names.index("cmd_vx")]
    wz = action[:, names.index("cmd_wz")]
    predicted = xy[:-1] + (vx[:-1] / fps)[:, None] * np.c_[np.cos(yaw[:-1]), np.sin(yaw[:-1])]
    # Commands ramp within a frame; allow the 1/fps accel-limited change.
    accel = planner["max_accel_mps2"]
    tolerance = 0.5 * accel / fps**2 + 0.01
    motion_error = np.linalg.norm(predicted - xy[1:], axis=1)
    details["command_motion_error_max_m"] = float(motion_error.max())
    checks["commands_explain_motion"] = bool(motion_error.max() <= tolerance)
    dyaw = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1])))
    yaw_error = np.abs(dyaw - wz[:-1] / fps)
    yaw_tolerance = 0.5 * planner["max_yaw_accel_rad_s2"] / fps**2 + 0.005
    details["command_yaw_error_max_rad"] = float(yaw_error.max())
    checks["commands_explain_yaw"] = bool(yaw_error.max() <= yaw_tolerance)

    # The photo frame: robot stopped, facing the fire within the face tolerance.
    if details["photo_frame"] is not None:
        k = details["photo_frame"]
        fire = np.asarray(result["photo"]["fire_world_xyz"][:2])
        bearing = np.arctan2(fire[1] - xy[k, 1], fire[0] - xy[k, 0])
        heading_error = abs(float(np.angle(np.exp(1j * (bearing - yaw[k])))))
        details["photo_frame_heading_error_rad"] = heading_error
        checks["photo_frame_facing_fire"] = heading_error <= planner["face_tolerance_rad"] + 0.01
        checks["photo_frame_stopped"] = abs(vx[k]) < 1e-3 and abs(wz[k]) < 1e-3

    for name in ("map/patrol_map.png", "map/patrol_map_result.png", "map/occupancy.npz"):
        checks[f"{name}_exists"] = (output / name).is_file()

    _check_alarm(dataset, result, planner, names, action, xy, yaw, fps, checks, details)

    episodes = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    details["episode_success"] = episodes.get("success")
    details["dataset_use"] = episodes.get("dataset_use")

    videos = {}
    for key, feature in info["features"].items():
        if feature["dtype"] != "video":
            continue
        video = dataset / "videos" / key / "chunk-000/file-000.mp4"
        stream = _ffprobe(video)
        height, width = feature["shape"][:2]
        videos[key] = stream
        checks[f"{key}_decodes_all_frames"] = int(stream["nb_read_frames"]) == frames
        checks[f"{key}_resolution"] = (stream["width"], stream["height"]) == (width, height)
    details["videos"] = videos

    photo = result.get("photo") or {}
    image_path = Path(photo.get("image", ""))
    checks["photo_file"] = image_path.is_file()
    if checks["photo_file"]:
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        x0, y0, x1, y1 = photo["window_px"]
        recheck = fire_pixel_score(rgb, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0)
        details["photo_recheck"] = recheck
        checks["photo_fire_pixels_reproduce"] = (
            abs(recheck["window_fraction"] - photo["window_fraction"]) < 1e-6
        )
        checks["photo_shows_fire"] = bool(photo.get("pass"))

    # Contact sheet: first, photo and last frame of every video.
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    alarm_frames = details.get("alarm_frames") or [0, 0]
    picks = sorted({0, alarm_frames[0] + 1, details["photo_frame"] or 0, frames - 1})
    tiles = []
    for key in videos:
        video = dataset / "videos" / key / "chunk-000/file-000.mp4"
        size = (int(videos[key]["width"]), int(videos[key]["height"]))
        tiles.append(
            [
                Image.fromarray(_frames(video, index, 1, size)[0]).resize((320, 240))
                for index in picks
            ]
        )
    if tiles:
        sheet = Image.new("RGB", (320 * len(picks), 240 * len(tiles)))
        for row, images in enumerate(tiles):
            for col, image in enumerate(images):
                sheet.paste(image, (col * 320, row * 240))
        sheet.save(qa / "keyframes.jpg", quality=90)
        details["keyframes"] = {"file": str(qa / "keyframes.jpg"), "frames": picks}
    checks = {name: bool(ok) for name, ok in checks.items()}  # numpy bools are not JSON
    passed = all(checks.values())
    return {"checks": checks, "details": details, "passed": passed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("outputs/xunjian"))
    args = parser.parse_args()
    report = validate(args.output)
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"passed": report["passed"], "checks": report["checks"]}, indent=2))
    sys.exit(0 if report["passed"] else 1)
