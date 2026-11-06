#!/usr/bin/env python3
"""Validate actual free-fall dynamics and render-only ownership, one Kit process per run.

Examples (run sequentially with the installed Isaac Sim runtime)::

    ./app/python.sh scripts/validate_physics_modes.py --engine physx
    ./app/python.sh scripts/validate_physics_modes.py --engine newton --solver mujoco
    ./app/python.sh scripts/validate_physics_modes.py --engine newton --solver xpbd
    ./app/python.sh scripts/validate_physics_modes.py --engine newton --solver vbd
    ./app/python.sh scripts/validate_physics_modes.py --engine newton --execution direct
    ./app/python.sh scripts/validate_physics_modes.py --engine hybrid --solver vbd

This is a contact-free engine smoke test, not the cloth-grasp acceptance test.
In hybrid mode, independent PhysX and Newton cubes validate the two step owners;
they do not claim to demonstrate cross-engine contact coupling.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from real2sim2real.core.logger import get_logger
from real2sim2real.simworld.isaacsim import IsaacsimWorld, IsaacsimWorldConfig


def _make_usd_cube(stage: Any) -> None:
    """Author a massive free body; explicit inertia also satisfies MuJoCo import."""
    from pxr import Gf, UsdGeom, UsdPhysics

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    cube = UsdGeom.Cube.Define(stage, "/World/ValidationCube")
    cube.CreateSizeAttr(0.05)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass.CreateMassAttr(0.1)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*(0.1 * 0.05**2 / 6.0,) * 3))


def _engine_timestamps() -> dict[str, int]:
    from omni.physics.core import get_physics_interface, get_physics_simulation_interface

    interface = get_physics_interface()
    simulation = get_physics_simulation_interface()
    return {
        interface.get_simulation_name(identifier): int(
            simulation.get_simulation_timestamp(identifier)
        )
        for identifier in interface.get_simulation_ids()
    }


def _runtime_diagnostics(world: IsaacsimWorld) -> dict[str, Any]:
    from isaacsim.core.simulation_manager import SimulationManager

    result = {
        "configured_dt_s": world.config.physics_dt,
        # A direct model has no Kit physics scene; querying its DT would create one.
        "manager_dt_s": None if world.uses_direct_newton else SimulationManager.get_physics_dt(),
        "manager_time_s": SimulationManager.get_simulation_time(),
        "manager_physics_steps": SimulationManager.get_num_physics_steps(),
    }
    if world.physics_engine == "newton" and not world.uses_direct_newton:
        from isaacsim.physics.newton import acquire_stage

        runtime = acquire_stage()
        result.update(
            newton_dt_s=runtime.sim_dt,
            newton_time_s=runtime.sim_time,
            newton_physics_steps=runtime.simulation_step_count,
            newton_parser_dt_s=runtime.parsing_results.get("physics_dt"),
            newton_gravity=runtime.model.gravity.numpy().tolist(),
        )
        if world.config.newton_solver == "mujoco":
            result["mujoco_cpu_model_dt_s"] = float(runtime.solver.mj_model.opt.timestep)
            if runtime.solver.mjw_model is not None:
                result["mujoco_warp_model_dt_s"] = (
                    runtime.solver.mjw_model.opt.timestep.numpy().tolist()
                )
    return result


def _managed_state_reader(world: IsaacsimWorld) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    if world.physics_engine == "newton":
        from isaacsim.physics.newton import acquire_stage

        runtime = acquire_stage()
        if runtime is None or runtime.model is None or runtime.state_0 is None:
            raise AssertionError("Newton did not build its USD model")
        indices = [
            index
            for index, label in enumerate(runtime.model.body_label)
            if str(label).endswith("/World/ValidationCube")
        ]
        if len(indices) != 1:
            raise AssertionError(f"ValidationCube mapping is ambiguous: {indices}")
        body = indices[0]

        def read_newton() -> tuple[np.ndarray, np.ndarray]:
            # Re-read the live buffer: Newton can swap state_0/state_1 after each step.
            return (
                runtime.state_0.body_q.numpy()[body, :3].copy(),
                runtime.state_0.body_qd.numpy()[body, :3].copy(),
            )

        return read_newton

    from isaacsim.core.simulation_manager import SimulationManager

    view = SimulationManager.get_physics_simulation_view().create_rigid_body_view(
        "/World/ValidationCube"
    )
    if view.count != 1:
        raise AssertionError(f"Expected one PhysX cube, got {view.count}")

    def read_physx() -> tuple[np.ndarray, np.ndarray]:
        return (
            view.get_transforms().numpy().reshape(-1, 7)[0, :3].copy(),
            view.get_velocities().numpy().reshape(-1, 6)[0, :3].copy(),
        )

    return read_physx


def _register_direct_model(
    world: IsaacsimWorld,
    *,
    cloth: bool = False,
) -> Callable[[], tuple[np.ndarray, np.ndarray]]:
    """Use a real Newton rigid body or cloth to test the application's manual stepper."""
    import newton
    import warp as wp

    solver_cls = {
        "mujoco": newton.solvers.SolverMuJoCo,
        "xpbd": newton.solvers.SolverXPBD,
        "vbd": newton.solvers.SolverVBD,
        "featherstone": newton.solvers.SolverFeatherstone,
        "semi_implicit": newton.solvers.SolverSemiImplicit,
    }[world.config.newton_solver]
    builder = newton.ModelBuilder(gravity=world.config.gravity)
    solver_cls.register_custom_attributes(builder)
    if cloth:
        builder.add_cloth_grid(
            pos=wp.vec3(0, 0, 5),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            dim_x=4,
            dim_y=4,
            cell_x=0.05,
            cell_y=0.05,
            mass=0.01,
            tri_ke=1000.0,
            tri_ka=1000.0,
            tri_kd=0.1,
            edge_ke=1.0,
            edge_kd=0.1,
            add_springs=world.config.newton_solver == "xpbd",
            spring_ke=1000.0,
            spring_kd=0.1,
        )
    else:
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity()), label="validation_cube"
        )
        builder.add_shape_box(body, hx=0.025, hy=0.025, hz=0.025)
    if world.config.newton_solver == "vbd":
        builder.color()
    model = builder.finalize(device=world.config.device or "cuda:0")
    model.ground = False
    state = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, state[0])
    newton.eval_fk(model, model.joint_q, model.joint_qd, state[1])
    options = dict(world.config.newton_solver_options)
    if world.config.newton_solver in {"xpbd", "vbd"}:
        options["iterations"] = world.config.newton_iterations
    if world.config.newton_solver == "mujoco":
        options["use_mujoco_cpu"] = wp.get_device(world.config.device or "cuda:0").is_cpu
    solver = solver_cls(model, **options)
    control = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    def step(dt: float) -> None:
        for _ in range(world.config.newton_substeps):
            state[0].clear_forces()
            pipeline.collide(state[0], contacts)
            solver.step(state[0], state[1], control, contacts, dt / world.config.newton_substeps)
            state[0], state[1] = state[1], state[0]

    world.register_newton_stepper(step)

    def read() -> tuple[np.ndarray, np.ndarray]:
        if cloth:
            positions = state[0].particle_q.numpy()
            velocities = state[0].particle_qd.numpy()
            if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
                raise AssertionError("Cloth state contains NaN/Inf")
            return positions.mean(axis=0), velocities.mean(axis=0)
        return state[0].body_q.numpy()[body, :3].copy(), state[0].body_qd.numpy()[body, :3].copy()

    return read


def _validate_fall(
    label: str,
    initial: tuple[np.ndarray, np.ndarray],
    final: tuple[np.ndarray, np.ndarray],
    duration: float,
) -> dict[str, Any]:
    position0, velocity0 = initial
    position1, velocity1 = final
    if not np.isfinite(np.concatenate((position1, velocity1))).all():
        raise AssertionError(f"{label}: non-finite dynamics state")
    expected_drop = -float(velocity0[2]) * duration + 0.5 * 9.81 * duration**2
    drop = float(position0[2] - position1[2])
    if not 0.5 * expected_drop < drop < 1.5 * expected_drop:
        raise AssertionError(f"{label}: drop={drop}, expected approximately {expected_drop}")
    velocity_change = float(velocity0[2] - velocity1[2])
    if not np.isclose(velocity_change, 9.81 * duration, rtol=0.2, atol=1e-4):
        raise AssertionError(f"{label}: gravity velocity change={velocity_change}")
    return {
        "position_start": position0.tolist(),
        "position_end": position1.tolist(),
        "velocity_start": velocity0.tolist(),
        "velocity_end": velocity1.tolist(),
        "drop_m": drop,
        "expected_drop_m": expected_drop,
    }


def validate(args: argparse.Namespace) -> None:
    """Start Kit only after CLI parsing; raise failures so cleanup preserves exit code 1."""
    config = IsaacsimWorldConfig(
        backend="isaacsim",
        headless=not args.gui,
        physics_engine=args.engine,
        newton_execution=args.execution,
        newton_solver=args.solver,
        device=args.device,
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 30.0,
        rt_subframes=1,
        warmup_frames=1,
        newton_use_cuda_graph=not args.no_cuda_graph,
        stage_usd_path=str(args.stage) if args.stage else "",
    )
    world = IsaacsimWorld(config=config)
    try:
        world.init_world()
        _make_usd_cube(world.stage)
        readers = {}
        if world.uses_direct_newton or world.physics_engine == "hybrid":
            readers["direct_newton"] = _register_direct_model(world, cloth=args.cloth)
        world.reset()
        world.play()
        if not world.uses_direct_newton:
            readers[world.physics_engine] = _managed_state_reader(world)
        initial = {name: read() for name, read in readers.items()}
        start_timestamps = _engine_timestamps()
        diagnostics_before = _runtime_diagnostics(world)
        for index in range(args.steps):
            world.step(render=index % 4 == 3)
        final = {name: read() for name, read in readers.items()}
        if world.physics_steps != args.steps:
            raise AssertionError(f"Wrong application step count: {world.physics_steps}")
        after_step_timestamps = _engine_timestamps()
        diagnostics_after = _runtime_diagnostics(world)
        get_logger("physics_validation").info(
            "PHYSICS_DYNAMICS_OBSERVED",
            initial={
                name: [value.tolist() for value in values] for name, values in initial.items()
            },
            final={name: [value.tolist() for value in values] for name, values in final.items()},
            diagnostics_before=diagnostics_before,
            diagnostics_after=diagnostics_after,
        )
        dynamics = {
            name: _validate_fall(name, initial[name], final[name], args.steps * config.physics_dt)
            for name in readers
        }
        for _ in range(8):
            world.update()
        for name, read in readers.items():
            position, velocity = read()
            np.testing.assert_allclose(position, final[name][0], rtol=0, atol=1e-7)
            np.testing.assert_allclose(velocity, final[name][1], rtol=0, atol=1e-7)
        if _engine_timestamps() != after_step_timestamps:
            raise AssertionError("Render-only update advanced a physics engine")
        if world.uses_direct_newton:
            import omni.timeline
            from pxr import UsdGeom

            if not omni.timeline.get_timeline_interface().is_stopped():
                raise AssertionError("Direct Newton left the Kit timeline running")
            sentinel = world.stage.GetPrimAtPath("/World/ValidationCube")
            position = (
                UsdGeom.Xformable(sentinel).ComputeLocalToWorldTransform(0).ExtractTranslation()
            )
            np.testing.assert_allclose(tuple(position), (0.0, 0.0, 5.0), rtol=0, atol=1e-7)
            if start_timestamps != after_step_timestamps:
                raise AssertionError(
                    "A Kit physics backend stepped during direct Newton integration"
                )
        report = {
            "status": "passed",
            "engine": args.engine,
            "execution": args.execution,
            "solver": config.physx_solver if args.engine == "physx" else args.solver,
            "direct_object": "cloth" if args.cloth else "cube",
            "physics_steps": world.physics_steps,
            "dt_s": config.physics_dt,
            "dynamics": dynamics,
            "render_only_updates": 8,
            "engine_timestamps_before": start_timestamps,
            "engine_timestamps_after": after_step_timestamps,
            "diagnostics_before": diagnostics_before,
            "diagnostics_after": diagnostics_after,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        get_logger("physics_validation").info("PHYSICS_VALIDATION_OK", **report)
    except Exception as exc:
        missing_vbd_schema = (
            args.engine == "newton"
            and args.execution == "usd"
            and args.solver == "vbd"
            and "Cannot find a valid schema" in str(exc.__cause__)
        )
        failure = {
            "status": "unsupported" if missing_vbd_schema else "failed",
            "engine": args.engine,
            "execution": args.execution,
            "solver": args.solver,
            "error": f"{type(exc).__name__}: {exc}",
            "cause": str(exc.__cause__) if exc.__cause__ is not None else None,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        get_logger("physics_validation").error("PHYSICS_VALIDATION_FAILED", **failure)
        raise
    finally:
        world.destroy_world()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("physx", "newton", "hybrid"), default="newton")
    parser.add_argument("--execution", choices=("usd", "direct"), default="usd")
    parser.add_argument(
        "--solver",
        choices=("mujoco", "xpbd", "vbd", "featherstone", "semi_implicit"),
        default="xpbd",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument(
        "--cloth", action="store_true", help="Direct/hybrid Newton cloth centroid test"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stage", type=Path, help="Optional environment stage for engine-switch checks"
    )
    args = parser.parse_args()
    if not 4 <= args.steps <= 60:
        parser.error("--steps must be between 4 and 60 to remain above the ground")
    if args.cloth and (
        args.engine == "physx"
        or (args.engine == "newton" and args.execution != "direct")
        or args.solver not in {"xpbd", "vbd"}
    ):
        parser.error("--cloth requires Newton direct/hybrid with solver xpbd/vbd")
    validate(args)


if __name__ == "__main__":
    main()
