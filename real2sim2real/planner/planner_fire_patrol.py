"""Patrol planner: follow a patrol route, spot the fire, raise the alarm, photograph the fire.

Phases: ``patrol`` → ``approach`` → ``face`` → ``settle`` → ``photo`` → ``hold`` → ``alarm`` →
``done``. If the route ends without a confirmed sighting the phase becomes ``not_found``.

On a confirmed sighting the robot heads straight for the photo position without stopping.
There it faces the fire and takes the photo with the beacon off, so the beacon's red light
cannot pass for flames in the photo check. Only then it stays still for ``detect_hold_s`` and
keeps signalling for another ``alarm_s``; during both, the action ``alarm`` is 1 and the red
beacon strobes at ``alarm_flash_hz`` on one continuous clock. The episode ends after that.

The patrol route is a list of waypoints. Consecutive waypoints are joined with A* on the
occupancy grid the Job scans from PhysX collision geometry, so every leg keeps
``robot_radius_m`` clearance. Fire sighting uses ground truth: the fire point must project
inside the detection camera's field of view, within range, and a PhysX ray from the camera to
the fire must not hit anything, for ``detect_confirm_steps`` consecutive physics steps.
The photo itself is checked separately for flame-coloured pixels by the Job.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import Field, field_validator

from real2sim2real.core.config import BaseConfig
from real2sim2real.planner.base import BasePlanner
from real2sim2real.utils.patrol_geometry import (
    OccupancyGrid,
    polyline_length,
    project_point,
    wrap_angle,
)

Occluder = Callable[[np.ndarray], str | None]


class FirePatrolPlannerConfig(BaseConfig):
    """Map, route, speed limits, detection and photo parameters (metres, seconds, rad)."""

    backend: str = "fire_patrol"
    # Occupancy scan (PhysX overlap queries over the robot's height band).
    map_bounds_xy: tuple[tuple[float, float], tuple[float, float]] = ((-11.0, 29.0), (9.0, 66.0))
    map_resolution_m: float = Field(default=0.05, gt=0)
    map_z_band_m: tuple[float, float] = (0.10, 1.15)
    robot_radius_m: float = Field(default=0.6, gt=0, description="Circle around the footprint")
    clearance_margin_m: float = Field(default=0.3, ge=0, description="Soft extra clearance")
    # Patrol.
    route_xy: list[tuple[float, float]] = Field(default_factory=list)
    cruise_speed_mps: float = Field(default=0.8, gt=0)
    approach_speed_mps: float = Field(default=0.5, gt=0)
    max_accel_mps2: float = Field(default=0.8, gt=0)
    max_yaw_rate_rad_s: float = Field(default=0.9, gt=0)
    max_yaw_accel_rad_s2: float = Field(default=2.0, gt=0)
    heading_gain: float = Field(default=2.0, gt=0)
    turn_in_place_rad: float = Field(default=0.6, gt=0, description="Stop and turn above this")
    waypoint_tolerance_m: float = Field(default=0.15, gt=0)
    # Detection.
    detect_max_distance_m: float = Field(default=10.0, gt=0)
    detect_fov_margin: float = Field(default=0.8, gt=0, le=1)
    detect_confirm_steps: int = Field(default=12, ge=1)
    # Alarm at the end, after the photo.
    detect_hold_s: float = Field(default=4.0, ge=0, description="Stay facing the fire, strobing")
    alarm_s: float = Field(default=3.0, gt=0, description="Further strobing before the end")
    alarm_flash_hz: float = Field(default=2.0, gt=0, description="Strobe rate, 50 % duty")
    # Photo.
    standoff_xy: tuple[float, float] | None = Field(
        default=None, description="Photo position; None picks one on a circle around the fire"
    )
    standoff_m: float = Field(default=2.1, gt=0.5)
    face_tolerance_rad: float = Field(default=0.03, gt=0)
    settle_s: float = Field(default=1.0, ge=0)

    @field_validator("route_xy")
    @classmethod
    def _route(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(value) < 2:
            raise ValueError("planner.route_xy needs at least two waypoints")
        return value


class FirePatrolPlanner(BasePlanner):
    """Unicycle path follower with acceleration limits and corner slow-down."""

    config_cls = FirePatrolPlannerConfig
    action_names = ["cmd_vx", "cmd_vy", "cmd_wz", "photo_trigger", "alarm"]

    def __init__(self, config: FirePatrolPlannerConfig | None = None, **_: Any) -> None:
        super().__init__()
        self.config = config or self.config_cls.from_settings("planner")
        self.state["timeout_s"] = 3600.0
        self.fire = np.zeros(3)
        self.grid: OccupancyGrid | None = None
        self.occluder: Occluder = lambda eye: None
        self.patrol_path = np.zeros((0, 2))
        self.patrol_legs: list[dict] = []
        self.dt = 1 / 120
        self.reset()

    def reset(self) -> None:
        self.state.update(
            phase="patrol",
            photo_request=False,
            alarm=False,
            beacon_on=False,
            last_action=np.zeros(len(self.action_names)),
        )
        self.path = self.patrol_path
        self.path_index = 1
        self.velocity = 0.0
        self.yaw_rate = 0.0
        self.visible_steps = 0
        self.timer = 0.0
        self.events: list[dict] = []
        self.rows: list[dict] = []
        self.detection: dict | None = None
        self.alarm: dict | None = None
        self.strobe_s = 0.0
        self.approach_path: np.ndarray | None = None
        self.photo: dict | None = None
        self.step_index = 0
        self.sight_state: dict = {}
        self.base_xy: np.ndarray | None = None
        self.base_yaw = 0.0
        self.state["max_base_drift_m"] = 0.0
        self.state["drift_steps_over_1mm"] = 0

    def init_planner(self, robot: Any) -> None:
        """Bind the robot; the route must start at the robot's configured base position."""
        self.robot = robot
        start = np.asarray(robot.config.base_xy, dtype=float)
        first = np.asarray(self.config.route_xy[0], dtype=float)
        if np.linalg.norm(start - first) > self.config.waypoint_tolerance_m:
            raise ValueError(f"route_xy[0]={first} must equal robot.base_xy={start}")
        self.state["ready"] = True

    # --- setup from the Job ---------------------------------------------------------------
    def set_map(self, grid: OccupancyGrid, fire: np.ndarray) -> dict:
        """Plan the patrol route on the occupancy grid; returns a route report."""
        self.grid = grid
        self.fire = np.asarray(fire, dtype=float)
        cfg = self.config
        waypoints = [np.asarray(p, dtype=float) for p in cfg.route_xy]
        pieces, legs = [waypoints[0][None]], []
        for index, (a, b) in enumerate(zip(waypoints[:-1], waypoints[1:], strict=True)):
            leg = grid.plan(a, b, cfg.robot_radius_m, cfg.clearance_margin_m)
            info = {"from": a.tolist(), "to": b.tolist(), "planned": leg is not None}
            if leg is not None:
                info.update(
                    length_m=round(polyline_length(leg), 2),
                    vertices=len(leg),
                    **grid.path_report(leg, cfg.robot_radius_m),
                )
                pieces.append(leg[1:])
            else:
                info["reason"] = self._why_blocked(a, b)
                legs.append(info)
                return {"ok": False, "failed_leg": index, "legs": legs}
            legs.append(info)
        self.patrol_path = np.vstack(pieces)
        self.patrol_legs = legs
        self.reset()
        return {
            "ok": True,
            "length_m": round(polyline_length(self.patrol_path), 2),
            "vertices": len(self.patrol_path),
            "legs": legs,
        }

    def _why_blocked(self, a: np.ndarray, b: np.ndarray) -> str:
        radius = self.config.robot_radius_m
        for name, point in (("start", a), ("goal", b)):
            if not self.grid.free(point, radius):
                clearance = self.grid.clearance(point)
                return f"{name} {point.tolist()} has {clearance:.2f} m clearance < {radius} m"
        return "no collision-free path between the waypoints on the occupancy grid"

    def set_runtime(self, *, occluder: Occluder, dt: float) -> None:
        """Ray-cast visibility test (camera eye → fire) and the physics step."""
        self.occluder = occluder
        self.dt = dt

    # --- detection ------------------------------------------------------------------------
    def sight(self) -> dict:
        """Ground-truth visibility of the fire from the detection camera."""
        name = self.robot.config.camera_for_detection
        camera = self.robot.camera_config(name)
        matrix = self.robot.camera_matrix(name)
        projection = project_point(
            self.fire,
            matrix,
            focal_length=camera.focal_length,
            horizontal_aperture=camera.horizontal_aperture,
            resolution=camera.resolution,
            near=camera.clipping_range[0],
            far=min(camera.clipping_range[1], self.config.detect_max_distance_m),
            margin=self.config.detect_fov_margin,
        )
        blocker = self.occluder(matrix[3, :3]) if projection.in_view else None
        return {
            "in_view": projection.in_view,
            "depth_m": projection.depth,
            "pixel": [projection.u, projection.v],
            "off_axis_rad": projection.off_axis_rad,
            "blocked_by": blocker,
            "visible": projection.in_view and blocker is None,
        }

    # --- control --------------------------------------------------------------------------
    def _corner_speed(self, index: int, speed: float) -> float:
        """Speed allowed when passing path vertex ``index``; 0 at the end or a sharp turn."""
        if index >= len(self.path) - 1:
            return 0.0
        a, b, c = self.path[index - 1], self.path[index], self.path[index + 1]
        turn = abs(
            wrap_angle(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(b[1] - a[1], b[0] - a[0]))
        )
        return speed * max(0.0, 1.0 - turn / self.config.turn_in_place_rad)

    def _follow(self, xy: np.ndarray, yaw: float, speed: float) -> bool:
        """Drive along ``self.path``; True when its last vertex is reached and stopped."""
        cfg = self.config
        while self.path_index < len(self.path):
            target = self.path[self.path_index]
            distance = float(np.linalg.norm(target - xy))
            last = self.path_index == len(self.path) - 1
            # Pass intermediate vertices once close; the final one needs the full tolerance.
            if distance > (
                cfg.waypoint_tolerance_m if last else max(cfg.waypoint_tolerance_m, 0.3)
            ):
                break
            self.path_index += 1
        if self.path_index >= len(self.path):
            self._ramp(0.0, 0.0)
            return abs(self.velocity) < 1e-3 and abs(self.yaw_rate) < 1e-3
        target = self.path[self.path_index]
        delta = target - xy
        distance = float(np.linalg.norm(delta))
        error = wrap_angle(math.atan2(delta[1], delta[0]) - yaw)
        corner = self._corner_speed(self.path_index, speed)
        # Brake so the speed at the vertex does not exceed the corner speed: v^2 = vc^2 + 2ad.
        limit = math.sqrt(corner**2 + 2 * cfg.max_accel_mps2 * distance)
        desired = 0.0 if abs(error) > cfg.turn_in_place_rad else min(speed, limit)
        desired *= max(0.0, math.cos(error))
        self._ramp(desired, cfg.heading_gain * error)
        return False

    def _turn_to(self, heading: float, yaw: float) -> bool:
        error = wrap_angle(heading - yaw)
        if abs(error) <= self.config.face_tolerance_rad and abs(self.yaw_rate) < 0.05:
            self._ramp(0.0, 0.0)
            return True
        # Rate limited by deceleration so the turn stops at the heading.
        stop_rate = math.sqrt(2 * self.config.max_yaw_accel_rad_s2 * abs(error))
        rate = min(stop_rate, self.config.heading_gain * abs(error))
        self._ramp(0.0, math.copysign(rate, error))
        return False

    def _ramp(self, velocity: float, yaw_rate: float) -> None:
        dv = self.config.max_accel_mps2 * self.dt
        dw = self.config.max_yaw_accel_rad_s2 * self.dt
        limit = self.config.max_yaw_rate_rad_s
        yaw_rate = float(np.clip(yaw_rate, -limit, limit))
        self.velocity += float(np.clip(velocity - self.velocity, -dv, dv))
        self.yaw_rate += float(np.clip(yaw_rate - self.yaw_rate, -dw, dw))

    def _set_phase(self, phase: str, **info: Any) -> None:
        self.state["phase"] = phase
        self.timer = 0.0
        self.events.append({"step": self.step_index, "phase": phase, **info})

    def on_physics_step(self, robot: Any) -> bool:
        """Command the base for the coming physics step; True when the episode has ended."""
        measured_xy, measured_yaw = robot.base_xy_yaw()
        if self.base_xy is None:
            self.base_xy, self.base_yaw = measured_xy, measured_yaw
        # The base is kinematic: control from the reference pose, and record how far PhysX
        # ended up from it (contacts could only push the base through the velocity write).
        drift = float(np.linalg.norm(measured_xy - self.base_xy))
        self.state["max_base_drift_m"] = max(self.state["max_base_drift_m"], drift)
        self.state["drift_steps_over_1mm"] += int(drift > 1e-3)
        xy, yaw = self.base_xy.copy(), self.base_yaw
        phase = self.state["phase"]
        self.state["photo_request"] = False
        sight = self.sight()
        self.sight_state = sight
        if phase == "patrol":
            self.visible_steps = self.visible_steps + 1 if sight["visible"] else 0
            if self.visible_steps >= self.config.detect_confirm_steps:
                # No stop on the way: plan to the photo position from where the robot is.
                self.detection = {"xy": xy.round(3).tolist(), "sight": _rounded(sight)}
                self._plan_approach(xy)
            elif self._follow(xy, yaw, self.config.cruise_speed_mps):
                self._set_phase("not_found", xy=xy.round(3).tolist())
        elif phase == "approach":
            if self._follow(xy, yaw, self.config.approach_speed_mps):
                self._set_phase("face", standoff_xy=xy.round(3).tolist())
        elif phase == "face":
            heading = math.atan2(self.fire[1] - xy[1], self.fire[0] - xy[0])
            # Fully settled (not just slow) so the photo and the strobing stay are motionless.
            if self._turn_to(heading, yaw) and abs(self.yaw_rate) < 1e-6:
                self._set_phase("settle", yaw=round(yaw, 4))
        elif phase == "settle":
            self._ramp(0.0, 0.0)
            if self.timer >= self.config.settle_s:
                self.state["photo_request"] = True
                self._set_phase("photo", sight=_rounded(sight))
        elif phase == "photo":  # beacon still off: its red light must not reach the photo
            self._ramp(0.0, 0.0)
            if self.photo is not None:
                self._start_alarm(xy, yaw, sight)
        elif phase == "hold":  # stay facing the fire, beacon strobing
            self._ramp(0.0, 0.0)
            if self.timer >= self.config.detect_hold_s - 1e-9:
                self.alarm["held_s"] = round(self.timer, 4)
                self._set_phase("alarm", held_s=self.alarm["held_s"])
        elif phase == "alarm":
            self._ramp(0.0, 0.0)
            if self.timer >= self.config.alarm_s - 1e-9:
                self.state["alarm"] = False
                self.alarm["end_step"] = self.step_index
                self._set_phase("done")
        else:
            self._ramp(0.0, 0.0)

        # Strobe: on for the first half of every 1/alarm_flash_hz period since the alarm began.
        self.state["beacon_on"] = bool(
            self.state["alarm"] and (self.strobe_s * self.config.alarm_flash_hz) % 1.0 < 0.5
        )
        if self.state["alarm"]:
            self.strobe_s += self.dt
        # Write the current pose plus the command; PhysX integrates it over the step exactly
        # like the reference below (constant world velocity, constant yaw rate).
        robot.set_base(xy, yaw, np.array([self.velocity, 0.0, self.yaw_rate]), self.dt)
        self.base_xy = xy + self.velocity * self.dt * np.array([math.cos(yaw), math.sin(yaw)])
        self.base_yaw = wrap_angle(yaw + self.yaw_rate * self.dt)
        self.state["last_action"] = np.array(
            [
                self.velocity,
                0.0,
                self.yaw_rate,
                float(self.state["photo_request"]),
                float(self.state["alarm"]),
            ]
        )
        self.timer += self.dt
        self.rows.append(
            {
                "step": self.step_index,
                "phase": self.state["phase"],
                "xy": xy.round(4).tolist(),
                "yaw": round(yaw, 5),
                "cmd": [round(self.velocity, 4), round(self.yaw_rate, 4)],
                "fire_in_view": sight["in_view"],
                "fire_visible": sight["visible"],
                "fire_blocked_by": sight["blocked_by"],
                "fire_depth_m": round(sight["depth_m"], 3),
                "base_drift_m": round(drift, 5),
                "alarm": self.state["alarm"],
                "beacon_on": self.state["beacon_on"],
            }
        )
        self.step_index += 1
        self.state["progress"] = {
            "phase": self.state["phase"],
            "path_vertex": self.path_index,
            "xy": xy.round(2).tolist(),
        }
        return self.state["phase"] in {"done", "not_found", "failed"}

    def _start_alarm(self, xy: np.ndarray, yaw: float, sight: dict) -> None:
        """Beacon on from the first instant of the stay; the hold phase starts with it."""
        self.state["alarm"] = True
        self.strobe_s = 0.0
        self.alarm = {
            "start_step": self.step_index,
            "xy": xy.round(3).tolist(),
            "yaw": round(yaw, 4),
            "distance_to_fire_m": round(float(np.linalg.norm(self.fire[:2] - xy)), 3),
            "fire_in_view": bool(sight["in_view"]),
            "signal": "red_beacon_strobe",
            "flash_hz": self.config.alarm_flash_hz,
            "hold_s": self.config.detect_hold_s,
            "after_hold_s": self.config.alarm_s,
            "fire_visible": bool(sight["visible"]),
        }
        self._set_phase("hold", **self.alarm)

    def _plan_approach(self, xy: np.ndarray) -> None:
        """Pick a reachable photo position with a clear view of the fire and plan to it."""
        cfg = self.config
        eye_z = float(self.robot.camera_matrix(self.robot.config.camera_for_detection)[3, 2])
        if cfg.standoff_xy is not None:
            candidates = [np.asarray(cfg.standoff_xy, float)]
        else:
            angles = np.radians(np.arange(0, 360, 10))
            ring = self.fire[:2] + cfg.standoff_m * np.c_[np.cos(angles), np.sin(angles)]
            candidates = sorted(ring, key=lambda p: float(np.linalg.norm(p - xy)))
        tried = []
        for point in candidates:
            entry = {"xy": point.round(3).tolist()}
            tried.append(entry)
            if not self.grid.free(point, cfg.robot_radius_m):
                entry["rejected"] = "no clearance"
                continue
            blocker = self.occluder(np.array([point[0], point[1], eye_z]))
            if blocker is not None:
                entry["rejected"] = f"view blocked by {blocker}"
                continue
            path = self.grid.plan(xy, point, cfg.robot_radius_m, cfg.clearance_margin_m)
            if path is None:
                entry["rejected"] = "no path"
                continue
            self.detection.update(
                {
                    "approach_from_xy": xy.round(3).tolist(),
                    "standoff_xy": point.round(3).tolist(),
                    "approach_length_m": round(polyline_length(path), 2),
                    "approach_clearance": self.grid.path_report(path, cfg.robot_radius_m),
                    "candidates_tried": len(tried),
                }
            )
            self.approach_path = path
            self.path, self.path_index = path, 1
            self._set_phase("approach", standoff_xy=point.round(3).tolist())
            return
        self._set_phase("failed", reason="no reachable photo position", tried=tried[:12])

    def photo_taken(self, info: dict) -> None:
        """Called by the Job after the photo file has been written and checked."""
        self.photo = info

    def result(self) -> dict:
        return {
            "phase": self.state["phase"],
            "fire_found": self.detection is not None,
            "detection": self.detection,
            "alarm": self.alarm,
            "photo": self.photo,
            "events": self.events,
            "route_xy": [list(p) for p in self.config.route_xy],
            "patrol_path_length_m": round(polyline_length(self.patrol_path), 2)
            if len(self.patrol_path) > 1
            else 0.0,
            "approach_path": None if self.approach_path is None else self.approach_path.tolist(),
            "max_base_drift_m": round(self.state["max_base_drift_m"], 5),
            "drift_steps_over_1mm": self.state["drift_steps_over_1mm"],
        }


def _rounded(sight: dict) -> dict:
    out = dict(sight)
    out["depth_m"] = round(float(sight["depth_m"]), 3)
    out["off_axis_rad"] = round(float(sight["off_axis_rad"]), 4)
    out["pixel"] = [round(float(v), 1) for v in sight["pixel"]]
    return out
