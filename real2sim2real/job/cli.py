"""挂载到平台 CLI: ``rsr collect ...``.

示例: ``rsr collect run`` 或 ``rsr collect run -c settings.toml``.

"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from real2sim2real.core.logger import LogConfig, configure_logging, get_logger

log = get_logger(__name__)

app = typer.Typer(
    help="数据采集任务编排",
    no_args_is_help=True,
)

_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="settings.toml 路径; 也可用环境变量 RSR_SETTINGS_FILE",
)


def _apply_config_file(config: Path | None) -> None:
    """把 ``-c`` 路径写入环境变量, 并清掉 dynaconf 缓存.

    保证后续 ``from_settings`` 读到新配置文件; 若路径不存在则转为 Typer 参数错误.
    成功后按新 settings 重新配置日志.

    Args:
        config: ``-c`` 传入的 settings 路径;None 则沿用环境变量 / cwd.

    Returns:
        None.
    """
    if config is None:
        return  # 走 cwd settings.toml / 默认
    from real2sim2real.core.config import apply_settings_file  # 延迟导入,避免 CLI 启动即拉 dynaconf

    try:
        path = apply_settings_file(config)  # 设 RSR_SETTINGS_FILE 并清缓存
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc  # 用户看得到路径错误
    configure_logging(LogConfig.from_settings("logging"), force=True)  # -c 后级别可能变
    log.info(f"加载配置文件: {path}")


@app.command()
def run(
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """采集入口: 初始化场景、布置场景、放置机器人并开始采集.

    可选 ``-c`` / ``--config`` 指定配置文件; 未传则走环境变量或默认 settings.
    调用链::

        -c config → AutoJob.from_config → BaseJob.run
        -> init_world -> arrange_scene -> place_robot -> start_simulation

    Args:
        config: ``-c`` 配置文件路径;None 则 ``RSR_SETTINGS_FILE`` 或默认 settings.

    Returns:
        None.
    """
    from real2sim2real.job.base import AutoJob  # 延迟导入,list-backends 不必拉 Job

    _apply_config_file(config)

    job = AutoJob.from_config()  # 按 [job].backend 选型
    job.run()  # 五段生命周期,finally close


@app.command("render-sorting")
def render_sorting(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=False,
            resolve_path=True,
            help="新视频与 LeRobot v3 的输出目录",
        ),
    ],
    source: Annotated[
        Path | None,
        typer.Option(
            "--source",
            exists=True,
            file_okay=False,
            resolve_path=True,
            help="可选：复用完整物理轨迹；不指定则先生成到输出目录的 physics/",
        ),
    ] = None,
    render_profile: Annotated[
        Path,
        typer.Option(
            "--render-profile",
            exists=True,
            dir_okay=False,
            resolve_path=True,
            help="高清路径追踪参数 JSON",
        ),
    ] = Path("configs/sorting_stream_showcase_render.json"),
    fps: Annotated[int, typer.Option("--fps", min=1, help="四路共享的采样率")] = 15,
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """先完成连续分拣物理，再按 overview、head、左腕、右腕依次渲染整段。"""
    import json
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "render_sorting_package.py"
    if not script.is_file():
        raise typer.BadParameter("此命令需要在包含 scripts/render_sorting_package.py 的项目中运行")
    if (output / "lerobot_v3").exists():
        raise typer.BadParameter("正式数据已存在，请选择新的 --output")
    from scripts.render_sorting_package import ReplayRenderProfile

    try:
        ReplayRenderProfile.model_validate_json(render_profile.read_text())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--render-profile") from exc
    if source is None:
        from real2sim2real.job.job_sorting import SortingJobConfig

        config = config or Path("configs/PhysicSolver_mujoco_vbd_sorting_stream_showcase.toml")
        _apply_config_file(config)
        job_config = SortingJobConfig.from_settings("job")
        source = output / "physics"
        if source.exists():
            raise typer.BadParameter(
                "physics/ 已存在；复用完整轨迹请指定 --source，否则选择新的 --output"
            )
        physics_script = script.with_name("check_sorting_package.py")
        if not physics_script.is_file():
            raise typer.BadParameter(f"缺少物理采集脚本：{physics_script}")
        log.info("先生成完整物理轨迹，不采集 RGB", output=str(source))
        solved = subprocess.run(
            [
                sys.executable,
                str(physics_script),
                "--config",
                str(config.resolve()),
                "--output",
                str(source),
                "--max-steps",
                str(job_config.max_steps),
            ],
            check=False,
        )
        if solved.returncode:
            result_path = source / "result.json"
            try:
                outcome = json.loads(result_path.read_text())
                if not isinstance(outcome, dict):
                    outcome = {}
            except (OSError, ValueError):
                outcome = {}
            log.error(
                "分拣物理阶段失败，未启动渲染",
                exit_code=solved.returncode,
                result=str(result_path),
                failure_reason=outcome.get("failure_reason"),
                failed_checks=outcome.get("failed_checks"),
                physics_duration_s=outcome.get("physics_duration_s"),
                maximum_edge_stretch=outcome.get("maximum_edge_stretch"),
            )
            raise typer.Exit(solved.returncode if solved.returncode > 0 else 1)
    elif source == output:
        raise typer.BadParameter("--output 不能覆盖物理源目录")
    log.info("按相机依次渲染整段", source=str(source), output=str(output))
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(source),
            "--output",
            str(output),
            "--render-profile",
            str(render_profile),
            "--fps",
            str(fps),
        ],
        check=False,
    )
    if completed.returncode:
        raise typer.Exit(completed.returncode if completed.returncode > 0 else 1)


def register(main_app: typer.Typer) -> None:
    """通过 entry_points 将本模块挂到平台 CLI ``rsr collect``.

    插件约定由 ``real2sim2real.cli.commands`` 加载, 调用本函数注册子命令组.

    Args:
        main_app: 根 Typer app (通常是 ``rsr``), 用于挂载 ``collect`` 子命令组.

    Returns:
        None.
    """
    main_app.add_typer(app, name="collect")  # `rsr collect run`
