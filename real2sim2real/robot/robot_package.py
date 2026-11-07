"""Franka articulation with the supplied V7 experiment's long parallel jaw geometry."""

from __future__ import annotations

from typing import Any

from pydantic import Field, FiniteFloat, PositiveFloat

from real2sim2real.robot.robot_physic_solver import FrankaNewtonRobot, FrankaNewtonRobotConfig


class PackagePadRobotConfig(FrankaNewtonRobotConfig):
    """V7-sized simulation jaws, expressed in each mirrored finger's local frame."""

    backend: str = "franka_package_pads"
    pad_half_extents_m: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.03, 0.007, 0.042)
    pad_center_m: tuple[FiniteFloat, FiniteFloat, FiniteFloat] = (0.0, 0.007, 0.055)
    pad_density_kg_m3: FiniteFloat = Field(default=1000.0, gt=0)


class PackagePadRobot(FrankaNewtonRobot):
    """Attach the V7 14×60×84 mm boxes to actual Featherstone finger bodies."""

    config_cls = PackagePadRobotConfig

    def add_newton_robot(self, builder: Any, *, scale: float = 100.0) -> None:
        """Replace finger collision proxies while retaining articulation and finger visuals."""
        import newton
        import warp as wp

        super().add_newton_robot(builder, scale=scale)
        bodies = [
            i
            for i, label in enumerate(builder.body_label)
            if label.endswith(("fr3_leftfinger", "fr3_rightfinger"))
        ]
        if len(bodies) != 2:
            raise ValueError("V7 jaws require two independent Franka finger bodies")
        collision = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        for index, body in enumerate(builder.shape_body):
            if body in bodies:
                builder.shape_flags[index] &= ~collision
        self.package_pad_shapes = []
        extents = [value * scale for value in self.config.pad_half_extents_m]
        center = [value * scale for value in self.config.pad_center_m]
        for body in bodies:
            shape = builder.add_shape_box(
                body,
                xform=wp.transform(center, wp.quat_identity()),
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=self.config.pad_density_kg_m3 / scale**3,
                    mu=self.config.finger_friction,
                    margin=0.0002 * scale,
                ),
                label="V7_package_pad",
            )
            self.package_pad_shapes.append(shape)
