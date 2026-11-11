"""Kit-free checks for the fire inspection parser, X30 leg kinematics, grid planner and patrol."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from real2sim2real.job.job_fire_inspection import parse_inspection_command
from real2sim2real.planner.planner_fire_patrol import FirePatrolPlanner, FirePatrolPlannerConfig
from real2sim2real.robot.robot_x30 import (
    LEGS,
    TrotGait,
    X30GaitConfig,
    X30LegGeometry,
    leg_fk,
    leg_ik,
)
from real2sim2real.utils.cameras import CameraConfig
from real2sim2real.utils.patrol_geometry import (
    OccupancyGrid,
    fire_pixel_score,
    polyline_length,
    pose_matrix,
    project_point,
    quat_wxyz_from_yaw,
    yaw_from_quat_wxyz,
)

FIRE = np.array([2.0, 8.0, 0.99])
STANCE = np.array([0.0, math.radians(-37.24225616455078), math.radians(74.48451232910156)])


def _grid(walls: list[tuple[float, float, float, float]]) -> OccupancyGrid:
    """10 m x 12 m room at 0.05 m with closed outer walls plus extra wall boxes."""
    resolution = 0.05
    origin = np.array([-5.0, 0.0])
    rows, cols = 240, 200
    occupied = np.zeros((rows, cols), bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    for x0, y0, x1, y1 in walls:
        c0, c1 = int((x0 - origin[0]) / resolution), int((x1 - origin[0]) / resolution)
        r0, r1 = int((y0 - origin[1]) / resolution), int((y1 - origin[1]) / resolution)
        occupied[r0:r1, c0:c1] = True
    return OccupancyGrid(origin, resolution, occupied, np.zeros_like(occupied))


def _camera_matrix(xy: np.ndarray, yaw: float, forward: float = 0.23, height: float = 0.9):
    c, s = math.cos(yaw), math.sin(yaw)
    matrix = np.eye(4)
    matrix[0, :3] = (s, -c, 0.0)  # camera +X: robot right
    matrix[1, :3] = (0.0, 0.0, 1.0)  # camera +Y: up
    matrix[2, :3] = (-c, -s, 0.0)  # camera +Z: backwards (looks along -Z)
    matrix[3, :3] = (xy[0] + forward * c, xy[1] + forward * s, height)
    return matrix


class FakeRobot:
    """Integrates the commanded twist like PhysX does for the kinematic base."""

    def __init__(self, xy, yaw):
        self.xy, self.yaw = np.asarray(xy, float), yaw
        self.config = SimpleNamespace(base_xy=tuple(xy), camera_for_detection="front_rgb")
        self.camera = CameraConfig(
            name="front_rgb",
            prim_path="/World/robots/X30/cam",
            pose_relative_to="/World/robots/X30",
            resolution=(640, 480),
            focal_length=18.0,
            horizontal_aperture=36.0,
        )
        self.pending = None

    def base_xy_yaw(self):
        if self.pending is not None:  # one physics step
            xy, yaw, (vx, _, wz), dt = self.pending
            self.xy = xy + vx * dt * np.array([math.cos(yaw), math.sin(yaw)])
            self.yaw = yaw + wz * dt
            self.pending = None
        return self.xy.copy(), self.yaw

    def set_base(self, xy, yaw, command, dt=0.0):
        self.pending = (np.asarray(xy, float), yaw, command, dt)

    def camera_matrix(self, name):
        return _camera_matrix(self.xy, self.yaw)

    def base_pose(self):
        return np.array([*self.xy, 0.52]), quat_wxyz_from_yaw(self.yaw)

    def camera_config(self, name):
        return self.camera


def test_parse_command():
    assert parse_inspection_command("巡检仓库")["mission"] == "patrol_find_fire_photo"
    assert parse_inspection_command("Please patrol the warehouse")["matched"]["verb"] == "patrol"
    assert parse_inspection_command("去仓库看看哪里着火了")["matched"]["fire"] == "火"
    for text in ("你好", "", "拍张照片"):
        with pytest.raises(ValueError):
            parse_inspection_command(text)


def test_pose_and_projection():
    q = quat_wxyz_from_yaw(1.2)
    assert yaw_from_quat_wxyz(q) == pytest.approx(1.2)
    m = pose_matrix([1, 2, 3], q)
    expected = [1 + math.cos(1.2), 2 + math.sin(1.2), 3, 1]
    assert np.append([1.0, 0, 0], 1) @ m == pytest.approx(expected)
    cam = _camera_matrix(np.zeros(2), 0.0, forward=0.0, height=0.0)
    kw = dict(focal_length=18, horizontal_aperture=36, resolution=(640, 480), near=0.1, far=10)
    p = project_point([2.0, 0, 0], cam, **kw)
    assert p.in_view and p.u == pytest.approx(320) and p.v == pytest.approx(240)
    assert project_point([2.0, 1.0, 0], cam, **kw).u < 320  # +Y world is the robot's left
    assert not project_point([-2.0, 0, 0], cam, **kw).in_view


def test_fire_pixel_score():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[40:60, 40:60] = (255, 140, 30)
    assert fire_pixel_score(rgb, 50, 50, 40)["window_fraction"] == pytest.approx(0.25)


def test_leg_kinematics_match_x30_stance():
    geometry = X30LegGeometry()
    feet = {leg: leg_fk(geometry, leg, STANCE) for leg in LEGS}
    # USD foot bodies of the asset at the authored stance (TORSO frame).
    assert feet["FL"] == pytest.approx([0.2971, 0.1968, -0.4856], abs=2e-4)
    assert feet["HR"] == pytest.approx([-0.2849, -0.1968, -0.4856], abs=2e-4)
    rng = np.random.default_rng(0)
    for _ in range(500):
        leg = LEGS[rng.integers(4)]
        q = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-1.2, -0.2), rng.uniform(0.6, 2.0)])
        assert leg_ik(geometry, leg, leg_fk(geometry, leg, q)) == pytest.approx(q, abs=1e-9)


def test_trot_stance_feet_stay_on_ground():
    geometry = X30LegGeometry()
    nominal = {leg: leg_fk(geometry, leg, STANCE) for leg in LEGS}
    gait = TrotGait(X30GaitConfig(), nominal)
    dt, yaw, pos = 1 / 120, 0.0, np.zeros(2)
    previous: dict[str, tuple[bool, np.ndarray]] = {}
    slip, lift = 0.0, 0.0
    for _ in range(240):
        feet = gait.feet(np.array([0.8, 0.0, 0.3]), dt)
        c, s = math.cos(yaw), math.sin(yaw)
        for leg, foot in feet.items():
            world = pos + np.array([c * foot[0] - s * foot[1], s * foot[0] + c * foot[1]])
            lift = max(lift, foot[2] - nominal[leg][2])
            stance = foot[2] == nominal[leg][2]
            if stance and leg in previous and previous[leg][0]:
                slip = max(slip, float(np.linalg.norm(world - previous[leg][1])))
            previous[leg] = (stance, world)
        pos = pos + 0.8 * dt * np.array([c, s])
        yaw += 0.3 * dt
    assert slip < 1e-3  # metres per physics step while in stance
    assert lift == pytest.approx(0.07, abs=1e-3)
    still = gait.feet(np.zeros(3), dt)
    assert all(np.allclose(still[leg], nominal[leg]) for leg in LEGS)


def test_grid_plan_keeps_clearance():
    grid = _grid([(-5.0, 5.0, 2.5, 5.4)])  # wall with a 2.5 m gap on the right
    path = grid.plan(np.array([-2.0, 2.0]), np.array([-2.0, 10.0]), radius=0.6, margin=0.3)
    assert path is not None
    report = grid.path_report(path, 0.6)
    assert report["violations"] == 0 and report["min_clearance_m"] >= 0.6
    assert polyline_length(path) > 8.0  # went around the wall
    assert grid.plan(np.array([-2.0, 2.0]), np.array([-4.9, 2.0]), 0.6) is None  # inside wall band


def _run(planner, robot, max_steps=20000):
    for _ in range(max_steps):
        if planner.on_physics_step(robot):
            return planner.state["phase"]
        if planner.state["photo_request"]:
            assert not planner.state["beacon_on"]
            planner.photo_taken({"image": "fake.png", "pass": True})
    raise AssertionError(f"planner did not finish: {planner.state['progress']}")


def _planner(grid, route, occluder, **kw):
    config = FirePatrolPlannerConfig(route_xy=route, detect_max_distance_m=10, **kw)
    planner = FirePatrolPlanner(config)
    robot = FakeRobot(route[0], math.pi / 2)
    planner.init_planner(robot)
    assert planner.set_map(grid, FIRE)["ok"]
    planner.set_runtime(occluder=occluder, dt=1 / 120)
    return planner, robot


def test_patrol_finds_fire_and_photographs():
    grid = _grid([])
    planner, robot = _planner(
        grid, [(-3.0, 1.0), (-3.0, 11.0)], lambda eye: None, standoff_xy=(0.0, 8.0)
    )
    assert _run(planner, robot) == "done"
    assert planner.detection is not None
    assert np.linalg.norm(robot.xy - [0.0, 8.0]) < 0.2
    heading = math.atan2(FIRE[1] - robot.xy[1], FIRE[0] - robot.xy[0])
    assert abs((robot.yaw - heading + math.pi) % (2 * math.pi) - math.pi) < 0.05
    assert max(abs(row["cmd"][0]) for row in planner.rows) <= 0.8 + 1e-9
    assert planner.state["max_base_drift_m"] < 1e-9


def test_auto_standoff_and_occlusion():
    grid = _grid([])
    planner, robot = _planner(grid, [(-3.0, 1.0), (-3.0, 11.0)], lambda eye: None)
    assert _run(planner, robot) == "done"
    assert np.linalg.norm(robot.xy - FIRE[:2]) == pytest.approx(2.1, abs=0.2)
    blind, robot = _planner(grid, [(-3.0, 1.0), (-3.0, 11.0)], lambda eye: "/Root/wall")
    assert _run(blind, robot) == "not_found"


def test_alarm_after_hold_then_photo():
    grid = _grid([])
    planner, robot = _planner(
        grid, [(-3.0, 1.0), (-3.0, 11.0)], lambda eye: None, standoff_xy=(0.0, 8.0)
    )
    assert _run(planner, robot) == "done"
    rows = planner.rows
    phases = [row["phase"] for row in rows]
    order = ["patrol", "stop", "turn_to_fire", "hold", "alarm", "approach", "face", "settle"]
    assert [p for i, p in enumerate(phases) if i == 0 or phases[i - 1] != p][: len(order)] == order
    dt = 1 / 120
    hold = [row for row in rows if row["phase"] == "hold"]
    alarm = [row for row in rows if row["alarm"]]
    assert len(hold) * dt == pytest.approx(4.0, abs=2 * dt)
    assert len(alarm) * dt == pytest.approx(3.0, abs=2 * dt)
    assert all(row["cmd"] == [0.0, 0.0] for row in hold[5:] + alarm)  # still while waiting
    switches = sum(a["beacon_on"] != b["beacon_on"] for a, b in zip(rows, rows[1:], strict=False))
    assert switches == 12  # 2 Hz for 3 s, on and off
    facing = math.atan2(FIRE[1] - alarm[0]["xy"][1], FIRE[0] - alarm[0]["xy"][0])
    assert abs((alarm[0]["yaw"] - facing + math.pi) % (2 * math.pi) - math.pi) < 0.05
    assert planner.alarm["held_s"] >= 4.0 - 1e-6
    assert planner.state["last_action"].shape == (len(planner.action_names),)
