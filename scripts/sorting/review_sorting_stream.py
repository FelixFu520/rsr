"""Build synchronized previews and phase-labelled frames for visual sorting-stream review."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from real2sim2real.core.logger import get_logger


def prepare_review(output: Path) -> None:
    """Extract actual recorded frames; leave visual acceptance pending until inspected."""
    result = json.loads((output / "result.json").read_text())
    if not result["success"]:
        raise ValueError("Visual delivery requires an accepted continuous source")
    source = Path(result["source_physics"])
    if not source.is_absolute():
        source = output / source
    rows = json.loads((source / "stream_trajectory.json").read_text())
    dataset = output / "lerobot_v3"
    info = json.loads((dataset / "meta/info.json").read_text())
    cameras = ("overview", "head", "left_wrist", "right_wrist")
    videos = {
        name: dataset
        / info["video_path"].format(
            video_key=f"observation.images.{name}",
            chunk_index=0,
            file_index=0,
        )
        for name in cameras
    }
    qa = output / "qa"
    frames = qa / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    selections = []
    for kind in ("soft", "rigid"):
        own = [row for row in rows if row["parcel"] == kind]

        grouped: dict[str, list[dict]] = {}
        for row in own:
            grouped.setdefault(row["phase"], []).append(row)
        phase = grouped.__getitem__

        incoming = phase("left_infeed")
        release = [row for row in phase("right_release") if row["robot_contacts"] == 0]
        chosen = [
            ("remote_infeed", incoming[0]),
            ("pickup_arrival", incoming[-1]),
            ("left_grasp", phase("left_transfer")[len(phase("left_transfer")) // 2]),
            ("platform_release", phase("left_rest")[-1]),
            ("right_grasp", phase("right_transfer")[len(phase("right_transfer")) // 2]),
            ("aerial_release", release[0]),
            ("outfeed_transport", phase("right_conveyor")[0]),
            ("ground", own[-1]),
        ]
        for label, row in chosen:
            index = min(info["total_frames"] - 1, round(row["time_s"] * info["fps"]))
            record = dict(
                parcel=kind,
                phase=label,
                video_frame=index,
                time_s=index / info["fps"],
                source_time_s=row["time_s"],
                images={},
            )
            views = cameras if label in {"left_grasp", "right_grasp"} else cameras[:2]
            for camera in views:
                destination = frames / f"{kind}_{label}_{camera}.png"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        f"{record['time_s']:.9f}",
                        "-i",
                        str(videos[camera]),
                        "-frames:v",
                        "1",
                        str(destination),
                    ],
                    check=True,
                )
                record["images"][camera] = str(destination.relative_to(output))
            selections.append(record)
    for kind in ("soft", "rigid"):
        selected = [row for row in selections if row["parcel"] == kind]
        sheet = Image.new("RGB", (1280, 1056), "#181818")
        draw = ImageDraw.Draw(sheet)
        for step, row in enumerate(selected):
            for camera_index, camera in enumerate(cameras[:2]):
                tile = step * 2 + camera_index
                x, y = tile % 4 * 320, tile // 4 * 264
                with Image.open(output / row["images"][camera]) as original:
                    sheet.paste(original.resize((320, 240)), (x, y + 24))
                draw.text(
                    (x + 5, y + 6),
                    f"{row['phase']} | {camera} | {row['time_s']:.2f}s",
                    fill="white",
                )
        sheet.save(qa / f"{kind}_workflow_keyframes.jpg", quality=93)
    grasps = [row for row in selections if row["phase"] in {"left_grasp", "right_grasp"}]
    sheet = Image.new("RGB", (1280, 1056), "#181818")
    draw = ImageDraw.Draw(sheet)
    for step, row in enumerate(grasps):
        for column, camera in enumerate(cameras):
            x, y = column * 320, step * 264
            with Image.open(output / row["images"][camera]) as original:
                sheet.paste(original.resize((320, 240)), (x, y + 24))
            draw.text((x + 5, y + 6), f"{row['parcel']} {row['phase']} | {camera}", fill="white")
    sheet.save(qa / "grasp_four_view_keyframes.jpg", quality=93)
    command = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "2"]
    for video in videos.values():
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            "[0:v]scale=640:480[v0];[1:v]scale=640:480[v1];"
            "[2:v]scale=640:480[v2];[3:v]scale=640:480[v3];"
            "[v0][v1]hstack=inputs=2[top];[v2][v3]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "21",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            str(qa / "synchronized_four_view.mp4"),
        ]
    )
    subprocess.run(command, check=True)
    (qa / "visual_review.json").write_text(
        json.dumps(
            dict(
                status="pending_visual_inspection",
                selected_frames=selections,
                scope="Actual synchronized videos and selected recorded frames; no physics rerun",
            ),
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info("Sorting visual review artifacts ready", output=str(qa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare_review(args.output)
