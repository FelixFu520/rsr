"""Compose this run's validated videos with measured robot/solver labels; no simulation.

Run: ./app/python.sh scripts/physic_solver/create_video_montages.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from real2sim2real.core.logger import get_logger
from real2sim2real.utils.physic_solver_utils import physic_solver_case_name

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/physic_solver"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FPS, WIDTH, HEIGHT = 15, 1920, 1080
LOG = get_logger(__name__)


def read(path: Path) -> dict:
    """Read a saved JSON artifact."""
    return json.loads(path.read_text())


def probe(path: Path) -> dict:
    """Read actual video metadata without relying on filename labels."""
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result)["streams"][0]


@dataclass
class Clip:
    """One physical episode, including all its camera videos."""

    source: Path
    title: str
    robot: str
    contact: str
    details: list[str]
    videos: dict[str, Path]
    frames: int
    passed: bool
    duration: float


def episode(relative: str, title: str, details: list[str], *, failed: bool = False) -> Clip:
    canonical = physic_solver_case_name(Path(relative).name)
    source = (ROOT / "outputs/debug/physic_solver" if failed else OUTPUT) / canonical
    object_id = canonical.split("_")[2]
    for prefix in ("V7｜", "V8｜", "V9｜", "V10｜"):
        title = title.replace(prefix, f"{object_id}｜")
    if not title.startswith(object_id):
        title = f"{object_id}｜{title}"
    result = read(source / "result.json")
    video_only = not (source / "lerobot_v3/meta/info.json").is_file()
    if video_only:
        diagnostic = read(source / "validation.json")["dataset"]
        assert failed and diagnostic["dataset_use"] == "diagnostic_video_only"
        assert not diagnostic["joint_state_action_exported"]
        info = dict(fps=diagnostic["fps"], total_frames=diagnostic["frames"])
    else:
        info = read(source / "lerobot_v3/meta/info.json")
    configs = list(source.glob("configuration.json")) or sorted(
        source.glob("diagnostics/*/configuration.json")
    )
    config = read(configs[-1])
    backend = config["robot"]["backend"]
    solver = result.get("rigid_solver", config["sim_world"]["newton_solver"])
    rigid = {
        "mujoco": "MuJoCo",
        "mujoco_cpu": "MuJoCo CPU",
        "xpbd": "Newton XPBD",
        "featherstone": "Featherstone",
        "tgs": "PhysX TGS",
    }.get(solver, solver)
    model = {
        "franka_physx": "Franka Panda",
        "franka_newton": "Franka FR3 原装指",
        "franka_package_pads": "Franka FR3＋长夹垫",
        "franka_carton_gripper": "Franka FR3＋宽行程夹爪",
        "screw_spindle": "导向机构＋连续旋转主轴",
    }[backend]
    coupling = result.get("coupling", "")
    if backend == "screw_spindle":
        contact = "接触：Hydroelastic / SDF"
    elif backend == "franka_carton_gripper":
        contact = "接触：原生 MuJoCo 刚体接触（双向）"
    elif result.get("contact_solver") == "tgs":
        contact = "接触：PhysX TGS 刚体接触"
    elif result.get("contact_solver") == "xpbd":
        contact = "接触 / 柔体：统一 Newton XPBD"
    else:
        direction = "双向代理" if "two_way" in coupling else "单向"
        contact = f"接触 / 柔体：Newton VBD（{direction}）"
    videos = (
        {key.split(".")[-1]: Path(path) for key, path in diagnostic["videos"].items()}
        if video_only
        else {
            p.parent.parent.name.removeprefix("observation.images."): p
            for p in sorted(source.glob("lerobot_v3/videos/*/chunk-000/file-000.mp4"))
        }
    )
    assert videos, source
    assert result["success"] is (not failed), source
    assert info["fps"] == FPS
    for path in videos.values():
        metadata = probe(path)
        assert int(metadata["nb_frames"]) == info["total_frames"], path
        assert metadata["avg_frame_rate"] == "15/1", path
        assert (metadata["width"], metadata["height"]) == (640, 480), path
    return Clip(
        source,
        title,
        f"机器人：{model}｜{rigid}",
        contact,
        details,
        videos,
        info["total_frames"],
        not failed,
        info["total_frames"] / FPS,
    )


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    value: str,
    size: int,
    color="#e7edf6",
    *,
    bold: bool = False,
    width: int = 1860,
) -> None:
    """Fit a complete readable caption into its allocated region."""
    font = ImageFont.truetype(BOLD if bold else FONT, size)
    while draw.textbbox((0, 0), value, font=font)[2] > width and size > 12:
        size -= 1
        font = ImageFont.truetype(BOLD if bold else FONT, size)
    draw.text(xy, value, font=font, fill=color)


def compose(
    name: str,
    title: str,
    subtitle: str,
    clips: list[Clip],
    stage: Path,
    *,
    pair: bool = False,
    notes: list[str] | None = None,
) -> dict:
    """Create one 1080p original-speed comparison board and verify the complete video."""
    count = max(c.frames for c in clips)
    duration = count / FPS
    board = Image.new("RGB", (WIDTH, HEIGHT), "#0b1220")
    draw = ImageDraw.Draw(board)
    title_color = "#e7edf6" if all(clip.passed for clip in clips) else "#ff8787"
    text(draw, (24, 12), title, 37, title_color, bold=True)
    text(draw, (25, 66), subtitle, 23, "#aabbd2")
    placements = []
    if pair:
        clip = clips[0]
        assert len(clip.videos) == 2, "The full-size paired layout requires exactly two cameras"
        text(draw, (26, 112), clip.robot, 29)
        text(draw, (26, 156), clip.contact, 28, "#9fd6ef")
        for column, (camera, video) in enumerate(clip.videos.items()):
            x = column * 960
            text(draw, (x + 18, 211), camera.upper(), 25, "#aabbd2")
            placements.append((video, x, 250, 960, 720, clip.duration, True))
        for index, note in enumerate(notes or []):
            text(draw, (26, 980 + index * 34), note, 23, "#e8b971")
    else:
        columns = 2 if len(clips) == 4 else 3
        cell_width = WIDTH // columns
        for index, clip in enumerate(clips):
            assert set(clip.videos) <= {"overview", "wrist"}, "Unexpected camera needs a new layout"
            x, y = index % columns * cell_width, 112 + index // columns * 464
            draw.rectangle((x + 2, y + 2, x + cell_width - 3, y + 461), fill="#111e30")
            text(draw, (x + 10, y + 1), clip.title, 26, bold=True, width=cell_width - 105)
            text(
                draw,
                (x + cell_width - 75, y + 4),
                "通过" if clip.passed else "未通过",
                23,
                "#7fe1bb" if clip.passed else "#ff8787",
                width=74,
            )
            text(draw, (x + 10, y + 34), clip.robot, 22, width=cell_width - 20)
            text(draw, (x + 10, y + 61), clip.contact, 21, "#9fd6ef", width=cell_width - 20)
            placements.append((clip.videos["overview"], x, y + 104, 480, 360, clip.duration, True))
            side_x, side_w = x + 486, cell_width - 492
            text(draw, (side_x, y + 91), "腕部视角", 18, "#aabbd2", width=side_w)
            if "wrist" in clip.videos:
                w = min(side_w // 2 * 2, 224)
                h = w * 3 // 8 * 2
                placements.append(
                    (clip.videos["wrist"], side_x, y + 120, w, h, clip.duration, False)
                )
            else:
                h = 78
                text(draw, (side_x, y + 126), "本例仅全景", 17, "#aabbd2", width=side_w)
            for line, detail in enumerate([*clip.details, f"时长 {clip.duration:.2f}s"]):
                text(draw, (side_x, y + 132 + h + line * 30), detail, 18, "#ced9e8", width=side_w)
        if len(clips) < columns * 2:
            x, y = len(clips) % columns * cell_width, 112 + len(clips) // columns * 464
            text(draw, (x + 26, y + 28), "本组说明", 29, bold=True, width=cell_width - 52)
            for index, note in enumerate(notes or []):
                text(
                    draw, (x + 26, y + 92 + index * 47), note, 23, "#aabbd2", width=cell_width - 52
                )
        text(
            draw,
            (22, 1043),
            "全程原速、同时开始｜主画面：全景；副画面：腕部｜较短片段结束后保持末帧，并标记“已结束”",
            22,
            "#aabbd2",
        )
    overlay = stage / f"{name}_labels.png"
    board.save(overlay)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-filter_complex_threads",
        "1",
        "-loop",
        "1",
        "-framerate",
        str(FPS),
        "-i",
        str(overlay),
    ]
    filters = [f"[0:v]trim=end_frame={count},setpts=PTS-STARTPTS,format=yuv420p[base]"]
    previous = "base"
    for index, (path, x, y, w, h, source_duration, ended_label) in enumerate(placements, 1):
        command += ["-threads", "1", "-i", str(path)]
        chain = (
            f"[{index}:v]fps={FPS},setpts=PTS-STARTPTS,"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,"
            f"tpad=stop_mode=clone:stop_duration={duration},trim=end_frame={count}"
        )
        if ended_label and source_duration < duration:
            chain += (
                f",drawtext=fontfile='{FONT}':text='已结束 · 保持末帧':fontsize=20:"
                f"fontcolor=white:box=1:boxcolor=black@0.8:boxborderw=5:x=10:y=h-34:"
                f"enable='gte(t,{source_duration:.8f})'"
            )
        filters.append(chain + f"[v{index}]")
        filters.append(f"[{previous}][v{index}]overlay=x={x}:y={y}:shortest=1[b{index}]")
        previous = f"b{index}"
    graph = stage / f"{name}_filter.txt"
    graph.write_text(";\n".join(filters))
    output = stage / f"{name}.mp4"
    command += [
        "-filter_complex_script",
        str(graph),
        "-map",
        f"[{previous}]",
        "-an",
        "-frames:v",
        str(count),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-movflags",
        "+faststart",
        str(output),
    ]
    with (stage / f"{name}.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    metadata = verify(output, count)
    previews = []
    for stamp in sorted({0, round(duration * 0.48, 2), round(duration - 0.1, 2)}):
        preview = stage / f"{name}_{stamp:.2f}.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                str(stamp),
                "-i",
                str(output),
                "-frames:v",
                "1",
                "-threads",
                "1",
                "-y",
                str(preview),
            ],
            check=True,
        )
        previews.append(str(preview.relative_to(ROOT)))
    LOG.info("Video montage encoded and decoded", output=name, frames=count)
    return dict(
        file=output.name,
        title=title,
        frames=count,
        duration_s=duration,
        video=metadata,
        preview_images=previews,
        visual_review="pending",
        cases=[
            dict(
                source=str(c.source.relative_to(ROOT)),
                title=c.title,
                robot=c.robot,
                contact=c.contact,
                passed=c.passed,
                duration_s=c.duration,
                videos={
                    k: dict(
                        path=str(v.relative_to(ROOT)),
                        sha256=hashlib.sha256(v.read_bytes()).hexdigest(),
                    )
                    for k, v in c.videos.items()
                },
            )
            for c in clips
        ],
    )


def verify(path: Path, frames: int) -> dict:
    """Fully decode the composed H.264 stream and check its promised dimensions and timeline."""
    metadata = probe(path)
    assert metadata["codec_name"] == "h264" and metadata["pix_fmt"] == "yuv420p"
    assert (metadata["width"], metadata["height"]) == (WIDTH, HEIGHT)
    assert metadata["avg_frame_rate"] == "15/1" and int(metadata["nb_frames"]) == frames
    assert math.isclose(float(metadata["duration"]), frames / FPS, abs_tol=0.001)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path), "-f", "null", "-"],
        check=True,
    )
    return {
        k: metadata[k]
        for k in (
            "codec_name",
            "pix_fmt",
            "width",
            "height",
            "avg_frame_rate",
            "nb_frames",
            "duration",
        )
    }


def frame_hashes(path: Path) -> list[str]:
    """Hash every decoded video frame to check the final concatenation in presentation order."""
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "framemd5",
            "-",
        ],
        text=True,
    )
    return [
        line.rsplit(",", 1)[1].strip() for line in decoded.splitlines() if not line.startswith("#")
    ]


def main() -> None:
    """Include all formal grasp episodes, newly recorded controls and an assembly appendix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path)
    args = parser.parse_args()
    summary = read(OUTPUT / "collection_summary.json")
    assert summary["complete"] and summary["fresh_physics"], "Fresh collection is incomplete"
    assert summary["resolution"] == [640, 480]
    for row in summary["cases"]:
        if row["status"] not in {"accepted", "diagnostic_verified"}:
            raise ValueError(f"Case has not completed actual visual review: {row['case']}")
        source = Path(row["output"])
        note = read(source / "run_note.json")
        assert note["recapture_run"] == summary["run_id"] and note["fresh_physics"]
        assert read(source / "qa/visual_review.json")["passed"]
    stage = args.stage or ROOT / "outputs/debug/physic_solver/video_montage" / datetime.now(
        UTC
    ).strftime("%Y%m%dT%H%M%S_%fZ")
    stage = stage.resolve()
    stage.mkdir(parents=True, exist_ok=False)

    def package(key: str, title: str, notes: list[str]) -> Clip:
        return episode(key, title, notes)

    solver_cases = [
        episode("tgs_tgs_pickplace_rigid", "PhysX 刚体方块", ["刚体", "无附着"]),
        episode("featherstone_vbd_pickplace_cloth", "Featherstone＋VBD 衣服", ["衣服", "单向耦合"]),
        episode("tgs_vbd_pickplace_cloth", "PhysX TGS＋VBD 衣服", ["混合引擎", "单向耦合"]),
        episode("mujoco_vbd_pickplace_cloth", "MuJoCo＋VBD 衣服", ["衣服", "单向耦合"]),
        episode("xpbd_vbd_pickplace_cloth", "XPBD＋VBD 衣服", ["衣服", "单向耦合"]),
        episode("mujoco_vbd_pickplace_package", "MuJoCo＋VBD 原包裹", ["原装指垫", "无附着"]),
    ]
    positions = [
        package(key, title, notes)
        for key, title, notes in [
            ("v7_center", "V7｜中央抓取", ["中央", "长夹垫"]),
            ("v8_offset_x35", "V8｜X 偏移 35mm", ["X=35mm", "无附着"]),
            ("v8_offset_y55", "V8｜Y 偏移 55mm", ["Y=55mm", "原装指"]),
            ("v8_yaw45", "V8｜偏航 45°", ["X=35mm", "原装指"]),
            ("v8_yaw90", "V8｜偏航 90°", ["Y=175mm", "优化抓点", "双向代理"]),
            ("v8_offset_tilt10", "V8｜偏航 30°＋俯仰 10°", ["X=35mm", "优化抓点"]),
        ]
    ]
    weights = [
        package(
            f"v9_mass{mass}g",
            f"V9｜衣物 {mass}g",
            [f"衣物 {mass}g", "袋膜20.5g", "X=35mm", "无附着"],
        )
        for mass in (100, 200, 350, 500)
    ]
    kinds = [
        package(key, title, ["柔性包裹", "无附着"])
        for key, title in [
            ("v10_station_clothing", "V10｜站点衣物"),
            ("v10_station_green", "V10｜绿色包裹"),
            ("v10_pink_tshirt", "V10｜粉色 T 恤"),
            ("v10_frosted_top", "V10｜磨砂上衣"),
            ("v10_black_hoodie", "V10｜黑色连帽衫"),
        ]
    ]
    shake_carton = [
        package(key, title, notes)
        for key, title, notes in [
            ("shake_none", "摇晃｜无附着", ["X=35mm", "无附着"]),
            ("shake_spring", "摇晃｜中央弹簧辅助", ["中央抓取", "有限力弹簧", "释放时关闭"]),
            ("shake_spring_x35", "摇晃｜X35 弹簧辅助", ["X=35mm", "有限力弹簧", "释放时关闭"]),
            ("v10_box01", "V10｜纸箱 01", ["350g", "原尺寸刚体", "无附着"]),
            ("v10_box02", "V10｜纸箱 02", ["250g", "原尺寸刚体", "无附着"]),
        ]
    ]
    groups = [
        (
            "01_solver_comparison",
            "01｜机器人＋接触算法对比",
            "六条已通过的基础抓放实验；型号、控制与材料不同，不作为单因素性能排名。",
            solver_cases,
        ),
        (
            "02_package_positions",
            "02｜包裹抓取位置与姿态",
            "含中央、位置偏移、偏航与组合倾斜；90°及组合倾斜为已记录的优化抓点。",
            positions,
        ),
        (
            "03_package_weights",
            "03｜包裹重量变化",
            "衣物质量 100 / 200 / 350 / 500g；各袋膜约 20.5g，X35mm 抓点。",
            weights,
        ),
        (
            "04_package_types",
            "04｜不同柔性包裹",
            "package02–06 五种柔性包裹；实际夹爪及接触算法按各片段生效配置标注。",
            kinds,
        ),
        (
            "05_shake_and_cartons",
            "05｜摇晃、附着与刚性纸箱",
            "弹簧辅助与无附着明确区分；纸箱采用原生 MuJoCo 刚体接触。",
            shake_carton,
        ),
    ]
    expected = {
        p.parent
        for p in OUTPUT.glob("*/result.json")
        if p.parent.name != physic_solver_case_name("mujoco_hydroelastic_screw")
    }
    included = [c.source for *_, clips in groups for c in clips]
    assert len(included) == len(set(included)) == 26 and set(included) == expected
    targets = [OUTPUT / f"{name}.mp4" for name, *_ in groups]
    targets += [
        OUTPUT / n
        for n in (
            "06_failed_grasp_controls.mp4",
            "07_screw_assembly_appendix.mp4",
            "grasp_all_comparison_labeled.mp4",
            "grasp_montage_index.json",
        )
    ]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite an existing montage")
    records = []
    for name, title, subtitle, clips in groups:
        records.append(
            compose(
                name,
                title,
                subtitle,
                clips,
                stage,
                notes=[
                    "每条为已通过的独立实验。",
                    "全程原速；腕部可能被夹爪遮挡。",
                    "数值验收与失败历史见任务文档。",
                    "单条通过不代表统计可靠性。",
                ],
            )
        )
        (stage / "progress.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    failed = [
        episode(
            "featherstone_vbd_pickplace_package",
            "Featherstone 包裹｜未通过",
            ["失败对照", "无附着"],
            failed=True,
        ),
        episode(
            "xpbd_xpbd_pickplace_cloth",
            "统一 XPBD 衣服｜未通过",
            ["失败对照", "无附着"],
            failed=True,
        ),
    ]
    records.append(
        compose(
            "06_failed_grasp_controls",
            "06｜失败抓取对照",
            "本轮重新采集的诊断视频；明确未通过验收，不作为成功演示。",
            failed,
            stage,
            notes=[
                "显示这次实际物理结果。",
                "对照失败与成功组合分别标注。",
                "成功基础组合见第 01 部。",
            ],
        )
    )
    screw = episode("mujoco_hydroelastic_screw", "连续旋转主轴装配", ["固定螺母"])
    screw_audit = read(screw.source / "validation.json")
    screw_motion = f"{screw_audit['turns']:.3f} 圈 / {screw_audit['axial_advance_m'] * 1000:.3f}mm"
    records.append(
        compose(
            "07_screw_assembly_appendix",
            "07｜附篇：固定螺母＋连续旋转主轴",
            f"螺纹装配，非抓取片段；{screw_motion}，未覆盖自由拾取和指定预紧力。",
            [screw],
            stage,
            pair=True,
            notes=["装配全景（左）＋螺纹接触近景（右），原速播放。"],
        )
    )
    concat = stage / "concat.txt"
    concat.write_text("".join(f"file '{r['file']}'\n" for r in records[:6]))
    master = stage / "grasp_all_comparison_labeled.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(master),
        ],
        check=True,
    )
    master_frames = sum(r["frames"] for r in records[:6])
    verify(master, master_frames)
    ordered_hashes = [
        digest for record in records[:6] for digest in frame_hashes(stage / record["file"])
    ]
    assert len(ordered_hashes) == master_frames
    assert frame_hashes(master) == ordered_hashes, "Concatenation changed decoded frame order"
    concat_audit = dict(
        passed=True,
        frames=master_frames,
        ordered_parts=[record["file"] for record in records[:6]],
        decoded_frame_md5_sequence_sha256=hashlib.sha256(
            "\n".join(ordered_hashes).encode()
        ).hexdigest(),
    )
    (stage / "concat_frame_audit.json").write_text(json.dumps(concat_audit, indent=2) + "\n")
    for record in records:
        (stage / record["file"]).replace(OUTPUT / record["file"])
    master.replace(OUTPUT / master.name)
    report = dict(
        created_utc=datetime.now(UTC).isoformat(),
        recapture_run=summary["run_id"],
        source_resolution=summary["resolution"],
        fresh_physics=True,
        concat_frame_audit=concat_audit,
        master=master.name,
        duration_s=master_frames / FPS,
        passed_grasp_cases=26,
        failed_grasp_video_cases=len(failed),
        grasp_source_camera_videos=(
            sum(len(c.videos) for *_, clips in groups for c in clips)
            + sum(len(c.videos) for c in failed)
        ),
        appendix_source_camera_videos=len(screw.videos),
        total_source_camera_videos=(
            sum(len(c.videos) for *_, clips in groups for c in clips)
            + sum(len(c.videos) for c in failed)
            + len(screw.videos)
        ),
        visual_review="pending",
        original_speed=True,
        short_clip_padding="last_frame_with_ended_label",
        montage_videos=records,
        staging=str(stage.relative_to(ROOT)),
        scope=(
            "All26 formal grasp episodes and both freshly recorded failed controls; "
            "screw assembly is a separate appendix, excluded from grasp master. "
            "Repeated tuning attempts are not experiment representatives."
        ),
    )
    (OUTPUT / "grasp_montage_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    LOG.info(
        "Montage assembly finished; visual review required", output=str(OUTPUT), stage=str(stage)
    )


if __name__ == "__main__":
    main()
