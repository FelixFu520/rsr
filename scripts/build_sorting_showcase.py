"""Build a presentation video with a large overview and three synchronized sensor views."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from real2sim2real.core.logger import get_logger


def build_showcase(output: Path) -> Path:
    """Preserve the full timeline in a 16:9 layout without cropping any source view."""
    result = json.loads((output / "result.json").read_text())
    for filename in ("stream_audit.json", "illumination_audit.json"):
        if not json.loads((output / "qa" / filename).read_text())["passed"]:
            raise ValueError("Showcase requires physical/data and illumination audits")
    if not result["success"] or result.get("diagnostic_replay", False):
        raise ValueError("Cannot present an unaccepted source as a completed demonstration")
    root = output / "lerobot_v3"
    info = json.loads((root / "meta/info.json").read_text())
    cameras = ("overview", "head", "left_wrist", "right_wrist")
    videos = {
        name: root
        / info["video_path"].format(
            video_key=f"observation.images.{name}", chunk_index=0, file_index=0
        )
        for name in cameras
    }
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.is_file():
        raise FileNotFoundError(font)
    spawn = float(result["carton_spawn_time_s"])

    def label(text: str, *, size: int = 23, when: str | None = None) -> str:
        expression = (
            f"drawtext=fontfile={font}:text='{text}':fontsize={size}:fontcolor=white:"
            "box=1:boxcolor=black@0.50:boxborderw=8:x=16:y=16"
        )
        return expression + (f":enable='{when}'" if when else "")

    filters = [
        "[0:v]scale=1440:1080:flags=lanczos,setsar=1,"
        + label("柔性包裹 · 双臂分拣", size=32, when=f"lt(t,{spawn:.9f})")
        + ","
        + label("刚性纸箱 · 双臂分拣", size=32, when=f"gte(t,{spawn:.9f})")
        + "[main]"
    ]
    for index, (name, title) in enumerate(
        zip(cameras[1:], ("头部视角", "左腕视角", "右腕视角"), strict=True), start=1
    ):
        filters.append(
            f"[{index}:v]scale=480:360:flags=lanczos,setsar=1," + label(title) + f"[{name}]"
        )
    filters.extend(
        [
            "[head][left_wrist][right_wrist]vstack=inputs=3[side]",
            "[main][side]hstack=inputs=2[out]",
        ]
    )
    destination = output / "showcase_1920x1080.mp4"
    command = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "2"]
    for video in videos.values():
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    subprocess.run(command, check=True)
    native_overview = output / "overview_1920x1440.mp4"
    temporary_overview = native_overview.with_suffix(".tmp.mp4")
    shutil.copy2(videos["overview"], temporary_overview)
    temporary_overview.replace(native_overview)
    (output / "qa/showcase_layout.json").write_text(
        json.dumps(
            dict(
                resolution=[1920, 1080],
                overview=[1440, 1080],
                sensor_views=[480, 360],
                frame_rate=info["fps"],
                frames=info["total_frames"],
                source_views=list(cameras),
                crops_applied=False,
                full_timeline_preserved=True,
                labels="Parcel type in overview; head, left wrist, right wrist in the sidebar",
                audio=False,
            ),
            indent=2,
        )
        + "\n"
    )
    get_logger(__name__).info("Sorting showcase ready", video=str(destination))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_showcase(args.output)
