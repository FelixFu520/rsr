"""GPU control kernels, imported only after the Newton extension is enabled."""

import warp as wp


@wp.kernel
def track_joint_velocity(
    target: wp.array[float],
    position: wp.array[float],
    velocity: wp.array[float],
):
    """Limit arm rates to rad/s and finger rates to centimetres per second."""
    index = wp.tid()
    limit = 2.0
    if index >= 7:
        limit = 8.0
    velocity[index] = wp.clamp(5.0 * (target[index] - position[index]), -limit, limit)
