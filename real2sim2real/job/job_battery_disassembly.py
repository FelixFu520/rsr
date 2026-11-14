"""Dual KR210 battery disassembly Job (PhysX): unscrew four screws into the basket, then move the
cover onto the pallet, and export one LeRobot v3 episode after the physical checks pass.

Migrated from Real2Sim2Real ``Kr210BatteryJob`` (configs/arm_kr210_kr210_unscrew). The source
moved the arms kinematically and attached screws / cover to the tools by script; here every
motion comes from PhysX: joint drives, bit-recess and pad-cover contacts, helical threads and a
magnetic bit force (see ``scene_battery_disassembly`` and ``planner_battery_disassembly``).

Lifecycle hooks:

- ``__init__``: PhysX only, integer physics Hz, recording fps divides it, refuse existing data.
- ``after_place_robot_hook``: nominal IK plan of both arms (fails before any physics step).
- ``before_start_simulation_hook``: physics views, home state, contact reports.
- ``simulate``: one planner call and one physics step per ``physics_dt``; RGB of every camera at
  ``fps``; full-rate state log; acceptance; LeRobot v3 export.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from real2sim2real.core.config import BaseConfig
from real2sim2real.job.base import BaseJob
from real2sim2real.utils.battery_disassembly_utils import inside_box, tilt_deg
from real2sim2real.utils.cameras import validate_camera_list
from real2sim2real.utils.lerobot_v3 import LeRobotV3Recorder, capture_viewport_frame


class BatteryRecorderConfig(BaseModel):
    """LeRobot v3 export; images at ``fps`` (must divide the physics rate)."""

    model_config = ConfigDict(extra="forbid")
    fps: int = Field(default=15, gt=0)
    record_rgb: bool = Field(default=True, description="false: physics only (debug), no dataset")
    save_failed_episodes: bool = False
    log_stride: int = Field(default=4, ge=1, description="physics steps per state-log row")
    warmup_captures: int = Field(default=3, ge=0, description="unrecorded renders per camera")


class BatteryAcceptanceConfig(BaseModel):
    """Physical acceptance limits."""

    model_config = ConfigDict(extra="forbid")
    min_turns_fraction: FiniteFloat = Field(default=0.95, gt=0, le=1)
    max_lead_error_fraction: FiniteFloat = Field(default=0.02, gt=0)
    max_bit_slip_fraction: FiniteFloat = Field(default=0.10, gt=0, lt=1)
    rest_window_s: FiniteFloat = Field(default=1.5, gt=0, description="final window for rest")
    max_rest_motion_m: FiniteFloat = Field(default=0.002, gt=0)
    max_cover_bottom_error_m: FiniteFloat = Field(default=0.01, gt=0)
    max_cover_tilt_deg: FiniteFloat = Field(default=3.0, gt=0)
    max_cover_slip_m: FiniteFloat = Field(default=0.01, gt=0)
    pallet_edge_margin_m: FiniteFloat = Field(default=0.0, ge=0)
    max_arm_contact_impulse_ns: FiniteFloat = Field(default=1e-3, ge=0)
    max_unexpected_tool_impulse_ns: FiniteFloat = Field(default=0.05, ge=0)


class BatteryDisassemblyJobConfig(BaseConfig):
    """Task text, output directory, limits, recorder and acceptance."""

    backend: str = "battery_disassembly"
    task: str = (
        "Remove four screws from the battery cover, place them in the basket, then remove the "
        "battery cover."
    )
    output_dir: str = "outputs/battery_disassembly/kr210_physx_c1003b1a1"
    overwrite_dataset: bool = Field(default=False, description="debug output only")
    timeout_s: FiniteFloat = Field(default=14400.0, gt=0)
    max_physics_steps: int | None = Field(default=None, gt=0, description="smoke tests")
    inspect_gui: bool = False
    stall_dump_s: FiniteFloat = Field(
        default=120.0,
        ge=0,
        description="write all thread stacks to diagnostics/<run>/stall_tracebacks.txt when one "
        "physics step + capture takes longer (0 disables); diagnostics only",
    )
    viewport_camera: str = Field(default="", description="GUI viewport camera prim (optional)")
    recorder: BatteryRecorderConfig = Field(default_factory=BatteryRecorderConfig)
    acceptance: BatteryAcceptanceConfig = Field(default_factory=BatteryAcceptanceConfig)


class _ContactFeed:
    """Accumulate PhysX contact impulses by actor category pair."""

    def __init__(self, classify: Any) -> None:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self.classify = classify
        self.to_path = PhysicsSchemaTools.intToSdfPath
        self.impulses: dict[str, float] = {}
        self.first_time: dict[str, float] = {}
        self.time = 0.0
        self.cache: dict[int, str] = {}
        interface = get_physx_simulation_interface()
        self.subscription = interface.subscribe_contact_report_events(self._callback)

    def _category(self, actor: int) -> str:
        if actor not in self.cache:
            self.cache[actor] = self.classify(str(self.to_path(actor))) or "environment"
        return self.cache[actor]

    def _callback(self, headers: Any, data: Any) -> None:
        for header in headers:
            a, b = sorted((self._category(header.actor0), self._category(header.actor1)))
            start = header.contact_data_offset
            impulse = 0.0
            for j in range(start, start + header.num_contact_data):
                impulse += float(np.linalg.norm(np.asarray(data[j].impulse, dtype=float)))
            key = f"{a}|{b}"
            self.impulses[key] = self.impulses.get(key, 0.0) + impulse
            if impulse > 0 and key not in self.first_time:
                self.first_time[key] = self.time

    def close(self) -> None:
        self.subscription = None


class BatteryDisassemblyJob(BaseJob):
    """One physically simulated disassembly episode."""

    config_cls = BatteryDisassemblyJobConfig

    def __init__(self, config: BatteryDisassemblyJobConfig | None = None) -> None:
        super().__init__(config=config)
        world = self.sim_world.config
        if self.sim_world.physics_engine != "physx":
            raise ValueError("battery_disassembly uses PhysX; set sim_world.physics_engine='physx'")
        hz = 1.0 / world.physics_dt
        stride = hz / self.config.recorder.fps
        if not math.isclose(stride, round(stride), abs_tol=1e-8, rel_tol=0):
            raise ValueError("recorder.fps must divide the physics rate")
        self.image_stride = round(stride)
        self.cameras = validate_camera_list([*self.scene.cameras, *self.robot.cameras])
        if self.config.recorder.record_rgb and not self.cameras:
            raise ValueError("record_rgb needs at least one camera")
        self.output = Path(self.config.output_dir).expanduser().resolve()
        dataset = self.output / "lerobot_v3"
        if dataset.exists():
            if not self.config.overwrite_dataset:
                raise FileExistsError(
                    f"Dataset already exists in {self.output}; choose job.output_dir"
                )
            shutil.rmtree(dataset)
        self.contacts: _ContactFeed | None = None
        self._stall_file: Any = None

    # --- hooks ---------------------------------------------------------------------------------
    def after_place_robot_hook(self) -> None:
        """Solve the nominal IK programs; unreachable layouts fail here."""
        self.planner.build(self.scene, self.robot)

    def before_start_simulation_hook(self) -> None:
        self.sim_world.reset()
        self.sim_world.play()
        self.scene.initialize_runtime()
        self.robot.initialize_runtime()
        self.planner.begin_episode(self.scene, self.robot)

        def classify(path: str) -> str | None:
            return self.robot.contact_owner(path) or self.scene.contact_owner(path)

        self.contacts = _ContactFeed(classify)

    def setup_gui_hook(self) -> None:
        if not self.sim_world.config.headless and self.config.viewport_camera:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.config.viewport_camera

    def close(self) -> None:
        if self.contacts is not None:
            self.contacts.close()
            self.contacts = None
        super().close()

    # --- main loop -----------------------------------------------------------------------------
    def simulate(self) -> None:
        run = self.output / "diagnostics" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        run.mkdir(parents=True)
        configs = {
            name: getattr(self, name).config.model_dump(mode="json")
            for name in ("sim_world", "scene", "robot", "planner")
        }
        configs["job"] = self.config.model_dump(mode="json")
        (run / "configuration.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False))
        rec_cfg = self.config.recorder
        recorder = None
        if rec_cfg.record_rgb:
            first = self.cameras[0].resolution
            recorder = LeRobotV3Recorder(
                self.output / "lerobot_v3",
                fps=rec_cfg.fps,
                width=first[0],
                height=first[1],
                task=self.config.task,
                robot_type=self.robot.recording_robot_type,
                joint_names=self.robot.recording_joint_names,
                action_names=self.planner.action_names,
                cameras={c.name: c.resolution for c in self.cameras},
                extra_features=self._extra_features(),
            )
        if recorder is not None:
            self._warm_up_cameras(run)
        started = time.monotonic()
        timings = dict(control=0.0, capture=0.0, physics=0.0, export=0.0)
        log: dict[str, list] = {}
        result: dict[str, Any] = {}
        failure: str | None = None
        step = 0
        try:
            next_progress = time.monotonic() + 30.0
            while True:
                self._arm_stall_dump(run)
                if not self.sim_world.is_running():
                    raise RuntimeError("Isaac Sim application closed")
                if time.monotonic() - started > self.config.timeout_s:
                    raise TimeoutError(f"Episode exceeded timeout_s={self.config.timeout_s}")
                t0 = time.monotonic()
                done = self.planner.on_physics_step(self.robot)
                released = self.scene.update_threads(step)
                timings["control"] += time.monotonic() - t0
                if step % rec_cfg.log_stride == 0 or done:
                    self._log_row(log, step)
                if released:
                    self.state.setdefault("released", {}).update({str(i): step for i in released})
                if recorder is not None and step % self.image_stride == 0:
                    t0 = time.monotonic()
                    self._capture(recorder)
                    recorder.add_frame(*self._frame(), self._extras(step))
                    timings["capture"] += time.monotonic() - t0
                if done or (
                    self.config.max_physics_steps and step >= self.config.max_physics_steps
                ):
                    break
                t0 = time.monotonic()
                self.sim_world.step(render=False)
                self.contacts.time = (step + 1) * self.sim_world.config.physics_dt
                timings["physics"] += time.monotonic() - t0
                step += 1
                if time.monotonic() >= next_progress:
                    self.log.info(
                        "拆解进度",
                        t=round(step * self.sim_world.config.physics_dt, 2),
                        phase=self.planner.state["phase"],
                        images=len(recorder.rows) if recorder else 0,
                        wall_s=round(time.monotonic() - started, 1),
                        **{k: round(v, 1) for k, v in timings.items()},
                    )
                    next_progress = time.monotonic() + 30.0
            recording = self._save_log(run, log)
            self._save_evidence(run)
            result = self._evaluate(recording)
            if recorder is not None:
                if not result["success"] and not rec_cfg.save_failed_episodes:
                    raise RuntimeError(f"Battery disassembly failed: {result['failed_checks']}")
                recorder.update_extra(len(recorder.rows) - 1, "next.success", result["success"])
                t0 = time.monotonic()
                dataset = recorder.finalize(episode_success=result["success"])
                timings["export"] = time.monotonic() - t0
                result["dataset"] = str(dataset)
                result["dataset_use"] = "demonstration" if result["success"] else "diagnostic"
                target = dataset / "meta" / "physics" / "episode_000000"
                target.mkdir(parents=True)
                for name in ("state_log.npz", "configuration.json", "layout.json", "contacts.json"):
                    shutil.copy2(run / name, target / name)
                (target / "acceptance.json").write_text(json.dumps(result, indent=2, default=str))
                (dataset / "meta" / "task_outcome.json").write_text(
                    json.dumps({"result": result, "configuration": configs}, indent=2, default=str)
                )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if recorder is not None:
                recorder.discard(run)
            if log and not (run / "state_log.npz").exists():
                self._save_log(run, log)
            raise
        finally:
            self._disarm_stall_dump()
            self._write_result(run, result, failure, recorder, started, timings, step)

    def _arm_stall_dump(self, run: Path) -> None:
        """Re-arm the faulthandler watchdog (Kit capture stalls seen on this machine)."""
        if not self.config.stall_dump_s:
            return
        import faulthandler

        if self._stall_file is None:
            self._stall_file = (run / "stall_tracebacks.txt").open("a")
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(
            self.config.stall_dump_s, repeat=True, file=self._stall_file
        )

    def _disarm_stall_dump(self) -> None:
        if self._stall_file is not None:
            import faulthandler

            faulthandler.cancel_dump_traceback_later()
            self._stall_file.close()
            path = Path(self._stall_file.name)
            if path.stat().st_size == 0:
                path.unlink()
            self._stall_file = None

    # --- per-step helpers ------------------------------------------------------------------------
    def _log_row(self, log: dict[str, list], step: int) -> None:
        sensors = self.robot.read_state()
        slide, turn = self.scene.thread_states()
        positions, orientations = self.scene.screw_poses()
        tip, axis = self.robot.tip_pose()
        row = dict(
            step=step,
            time=step * self.sim_world.config.physics_dt,
            screw_q=sensors["screw_q"], grip_q=sensors["grip_q"],
            screw_qd=sensors["screw_qd"], grip_qd=sensors["grip_qd"],
            screw_target=self.robot.targets["screw"], grip_target=self.robot.targets["grip"],
            float=sensors["float"], spindle=sensors["spindle"], spindle_qd=sensors["spindle_qd"],
            finger_p=sensors["finger_p"], finger_n=sensors["finger_n"],
            fingers=self.robot.finger_poses().reshape(-1),
            thread_slide=slide, thread_turn=turn,
            screw_pos=positions.reshape(-1), screw_quat=orientations.reshape(-1),
            screw_vel=self.scene.screw_velocities().reshape(-1),
            cover=self.scene.cover_pose(), tip=tip,
            magnet=np.r_[float(self.planner.magnet), self.planner.magnet_force],
            engaged=np.asarray(self.scene.engaged, dtype=float),
            phase_index=self.planner.state["phase_index"],
            action=self.planner.state["last_action"],
        )  # fmt: skip
        self._last_row = row
        for key, value in row.items():
            log.setdefault(key, []).append(value)

    def _frame(self) -> tuple[np.ndarray, np.ndarray]:
        sensors = self.robot.read_state()
        state = np.r_[sensors["screw_q"], sensors["grip_q"], sensors["float"], sensors["spindle"],
                      sensors["finger_p"], sensors["finger_n"]]  # fmt: skip
        return state, np.asarray(self.planner.state["last_action"], dtype=float)

    def _extra_features(self) -> dict[str, dict[str, Any]]:
        n = len(self.scene.screw_paths)
        return {
            "observation.screw_positions": {
                "dtype": "float32", "shape": [3 * n],
                "names": [f"screw_{i + 1}.{a}" for i in range(n) for a in "xyz"],
            },
            "observation.thread_slide_m": {
                "dtype": "float32", "shape": [n], "names": [f"screw_{i + 1}" for i in range(n)]
            },
            "observation.cover_pose": {
                "dtype": "float32", "shape": [7], "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]
            },
            "observation.phase_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "next.success": {"dtype": "bool", "shape": [1], "names": None},
        }  # fmt: skip

    def _extras(self, step: int) -> dict[str, Any]:
        positions, _ = self.scene.screw_poses()
        slide, _ = self.scene.thread_states()
        return {
            "observation.screw_positions": positions.reshape(-1),
            "observation.thread_slide_m": slide,
            "observation.cover_pose": self.scene.cover_pose(),
            "observation.phase_index": self.planner.state["phase_index"],
            "next.done": bool(self.planner.state.get("done")),
            "next.success": False,
        }

    def _warm_up_cameras(self, run: Path) -> None:
        """Render every camera a few times before frame 0 (the first RTX frame is underexposed)."""
        warm = run / "camera_warmup"
        warm.mkdir()
        for _ in range(self.config.recorder.warmup_captures):
            for camera in self.cameras:
                capture_viewport_frame(
                    self.sim_world, warm / f"{camera.name}.png", width=camera.resolution[0],
                    height=camera.resolution[1], camera_prim=camera.prim_path,
                )  # fmt: skip
        if self.sim_world.physics_steps != 0:
            raise RuntimeError("Camera warm-up advanced physics")

    def _capture(self, recorder: LeRobotV3Recorder) -> None:
        step = self.sim_world.physics_steps
        for camera in self.cameras:
            capture_viewport_frame(
                self.sim_world,
                recorder.frame_path(camera.name),
                width=camera.resolution[0],
                height=camera.resolution[1],
                camera_prim=camera.prim_path,
            )
        if self.sim_world.physics_steps != step:
            raise RuntimeError("Camera capture advanced physics")

    def _save_evidence(self, run: Path) -> None:
        """Layout and contact totals needed to recompute the acceptance offline."""
        (run / "layout.json").write_text(json.dumps(layout_json(self.scene.layout), indent=2))
        contacts = {
            "impulses_ns": dict(self.contacts.impulses) if self.contacts else {},
            "first_time_s": dict(self.contacts.first_time) if self.contacts else {},
        }
        (run / "contacts.json").write_text(json.dumps(contacts, indent=2))

    def _save_log(self, run: Path, log: dict[str, list]) -> dict[str, np.ndarray]:
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in log.items()}
        np.savez_compressed(run / "state_log.npz", **arrays)
        return arrays

    # --- acceptance ------------------------------------------------------------------------------
    def _evaluate(self, rec: dict[str, np.ndarray]) -> dict[str, Any]:
        return evaluate_battery(
            rec,
            planner_result=self.planner.result(),
            scene_layout=self.scene.layout,
            scene_config=self.scene.config,
            acceptance=self.config.acceptance,
            contacts=dict(self.contacts.impulses) if self.contacts else {},
            contact_first_time=dict(self.contacts.first_time) if self.contacts else {},
            physics_dt=self.sim_world.config.physics_dt,
        )

    def _write_result(
        self, run: Path, result: dict, failure: str | None, recorder: Any, started: float,
        timings: dict, steps: int,
    ) -> None:  # fmt: skip
        result.update(
            error=failure,
            success=bool(result.get("success") and failure is None),
            completed=bool(result) and failure is None,
            task=self.config.task,
            images=len(recorder.rows) if recorder is not None else 0,
            record_rgb=self.config.recorder.record_rgb,
            fps=self.config.recorder.fps,
            physics_steps=steps,
            physics_dt=self.sim_world.config.physics_dt if self.sim_world else None,
            elapsed_s=round(time.monotonic() - started, 1),
            timings_s={k: round(v, 2) for k, v in timings.items()},
            physics_engine="physx",
            planner=self.planner.result(),
        )
        text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        (run / "result.json").write_text(text + "\n")
        (self.output / "result.json").write_text(text + "\n")
        with (run / "trajectory.jsonl").open("w") as stream:
            for row in self.planner.rows:
                stream.write(json.dumps(row, default=float) + "\n")
        self.log.info("电池拆解结果", success=result["success"], failed=result.get("failed_checks"))


LAYOUT_KEYS = (
    "scene_matrix", "screw_heads", "basket_inner_lo", "basket_inner_hi", "cover_com_world",
    "pallet_world", "pallet_top_z",
)  # fmt: skip


def layout_json(layout: dict[str, Any]) -> dict[str, Any]:
    """JSON form of the scene layout entries used by ``evaluate_battery``."""
    return {key: np.asarray(layout[key], dtype=float).tolist() for key in LAYOUT_KEYS}


def layout_from_json(data: dict[str, Any]) -> dict[str, Any]:
    out = {key: np.asarray(data[key], dtype=float) for key in LAYOUT_KEYS}
    out["pallet_top_z"] = float(out["pallet_top_z"])
    return out


def evaluate_battery(
    rec: dict[str, np.ndarray],
    *,
    planner_result: dict[str, Any],
    scene_layout: dict[str, Any],
    scene_config: Any,
    acceptance: BatteryAcceptanceConfig,
    contacts: dict[str, float],
    contact_first_time: dict[str, float],
    physics_dt: float,
) -> dict[str, Any]:
    """Physical checks on the state log; also used by ``validate_battery_disassembly.py``."""
    acc = acceptance
    screw = scene_config.screw
    n = rec["thread_slide"].shape[1]
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    finite = all(np.isfinite(v).all() for v in rec.values())
    checks["finite_states"] = bool(finite)
    checks["sequence_completed"] = bool(planner_result.get("sequence_completed"))
    slide, turn = rec["thread_slide"], rec["thread_turn"]
    needed_turns = screw.thread_length_m / screw.pitch_m
    per_screw = []
    spindle = rec["spindle"]
    for i in range(n):
        moved = np.nonzero(slide[:, i] > 2e-4)[0]
        turns = float((turn[-1, i] - turn[0, i]) / (2 * math.pi))
        rise = float(slide[-1, i] - slide[0, i])
        lead = rise / turns if abs(turns) > 1e-9 else 0.0
        engaged = rec["engaged"][:, i] > 0.5
        out_index = int(np.argmin(engaged)) if not engaged.all() else None
        slip = None
        if len(moved) and out_index is not None:
            a, b = int(moved[0]), out_index
            spun = abs(spindle[b] - spindle[a]) / (2 * math.pi)
            screw_turns = abs(turn[b, i] - turn[a, i]) / (2 * math.pi)
            slip = float(1.0 - screw_turns / spun) if spun > 1e-9 else 1.0
        per_screw.append(
            dict(
                turns=abs(turns),
                rise_m=rise,
                lead_m=abs(lead),
                disengaged=out_index is not None,
                bit_slip_fraction=slip,
            )  # fmt: skip
        )
    metrics["screws"] = per_screw
    checks["all_threads_disengaged"] = all(s["disengaged"] for s in per_screw)
    checks["screws_turned_out_by_contact"] = all(
        s["turns"] >= acc.min_turns_fraction * needed_turns for s in per_screw
    )
    checks["thread_lead_matches_pitch"] = all(
        abs(s["lead_m"] - screw.pitch_m) <= acc.max_lead_error_fraction * screw.pitch_m
        for s in per_screw
    )
    checks["bit_drove_screw_without_slip"] = all(
        s["bit_slip_fraction"] is not None and s["bit_slip_fraction"] <= acc.max_bit_slip_fraction
        for s in per_screw
    )
    final = rec["screw_pos"][-1].reshape(n, 3)
    speeds = np.linalg.norm(rec["screw_vel"][-1].reshape(n, 3), axis=1)
    lo, hi = scene_layout["basket_inner_lo"], scene_layout["basket_inner_hi"]
    in_basket = [inside_box(p, lo, hi) for p in final]
    metrics["screw_final_positions"] = final.round(4).tolist()
    metrics["screw_final_speeds_mps"] = speeds.round(4).tolist()
    checks["screws_in_basket"] = all(in_basket)
    times = rec["time"]
    window = times >= times[-1] - acc.rest_window_s
    tail = rec["screw_pos"][window].reshape(int(window.sum()), n, 3)
    motion = np.linalg.norm(tail - tail[-1], axis=2).max(axis=0)
    cover_tail = rec["cover"][window, :3]
    metrics["screw_rest_motion_m"] = motion.round(5).tolist()
    metrics["cover_rest_motion_m"] = float(
        np.linalg.norm(cover_tail - cover_tail[-1], axis=1).max()
    )
    checks["screws_at_rest"] = bool(np.all(motion <= acc.max_rest_motion_m))
    checks["cover_at_rest"] = metrics["cover_rest_motion_m"] <= acc.max_rest_motion_m
    cover = rec["cover"][-1]
    start = rec["cover"][0]
    bottom0 = (
        scene_layout["cover_bottom_start_z"] if "cover_bottom_start_z" in scene_layout else None
    )
    pallet_top = scene_layout["pallet_top_z"]
    # cover bottom from its start height: rigid body, so the offset body→bottom is constant
    flange_bottom_start = (
        scene_layout["scene_matrix"] @ np.r_[0, 0, scene_config.cover.flange_z_m[0], 1.0]
    )[2]
    bottom = flange_bottom_start + (cover[2] - start[2])
    metrics["cover_bottom_z"] = float(bottom)
    metrics["pallet_top_z"] = float(pallet_top)
    metrics["cover_tilt_deg"] = tilt_deg(cover[3:7])
    del bottom0
    checks["cover_rests_on_pallet"] = abs(bottom - pallet_top) <= acc.max_cover_bottom_error_m
    checks["cover_level"] = metrics["cover_tilt_deg"] <= acc.max_cover_tilt_deg
    pallet = np.asarray(scene_layout["pallet_world"])
    size = np.asarray(scene_config.pallet.size_m)
    com_offset = np.asarray(scene_layout["cover_com_world"]) - start[:3]
    com_final = cover[:3] + com_offset  # small rotations only (checked by cover_level)
    metrics["cover_com_final"] = com_final.round(4).tolist()
    checks["cover_on_pallet_footprint"] = bool(
        np.all(np.abs(com_final[:2] - pallet[:2]) <= 0.5 * size[:2] - acc.pallet_edge_margin_m)
    )
    # slip of the cover in the gripper between grasp and release (gripper root ≈ finger mid-point)
    fingers = rec["fingers"].reshape(len(rec["fingers"]), 2, 3).mean(axis=1)
    phase = rec["phase_index"].astype(int)
    held = (rec["action"][:, 14] > 0) & (np.abs(rec["finger_p"]) > 1e-3)
    if held.any():
        rel = rec["cover"][held, :3] - fingers[held]
        settled = rel[min(len(rel) - 1, int(1.0 / (physics_dt * 4))) :]
        slip = float(np.max(np.linalg.norm(settled - settled[0], axis=1))) if len(settled) else 0.0
    else:
        slip = math.inf
    metrics["cover_slip_in_gripper_m"] = slip
    checks["cover_held_by_friction_without_slip"] = slip <= acc.max_cover_slip_m
    del phase
    arm = {k: v for k, v in contacts.items() if "screw_arm" in k or "grip_arm" in k}
    arm_impulse = sum(arm.values())

    def pair(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    expected = {pair("screw_tool", f"screw_{i + 1}") for i in range(n)} | {
        pair("cover", "grip_tool")
    }
    allowed_screw = {"cover", "screw_tool", "basket", *(f"screw_{i + 1}" for i in range(n))}
    stray = {
        k: v for k, v in contacts.items()
        if v > 0 and any(
            p.startswith("screw_") and p != "screw_tool"
            and (set(k.split("|")) - {p}) - allowed_screw
            for p in k.split("|")
        )
    }  # fmt: skip
    metrics["stray_screw_contacts_ns"] = stray
    checks["screws_touch_only_cover_bit_basket"] = not stray
    unexpected = {k: v for k, v in contacts.items() if "tool" in k and k not in expected and v > 0}
    metrics["contact_impulses_ns"] = {k: round(v, 6) for k, v in sorted(contacts.items())}
    metrics["contact_first_time_s"] = {
        k: round(v, 4) for k, v in sorted(contact_first_time.items())
    }
    checks["arm_links_collision_free"] = arm_impulse <= acc.max_arm_contact_impulse_ns
    checks["tools_touch_only_their_targets"] = (
        sum(unexpected.values()) <= acc.max_unexpected_tool_impulse_ns
    )
    metrics["unexpected_tool_contacts_ns"] = unexpected
    checks = {k: bool(v) for k, v in checks.items()}
    return {
        "success": all(checks.values()),
        "checks": checks,
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "metrics": metrics,
        "scope": "PhysX CPU rigid bodies and articulations with authored box/cylinder collision "
        "proxies; threads are mimic-joint helices with viscous friction; the screw is carried by a "
        "modelled magnetic bit force. No claim of real-robot validation.",
    }
