"""Buffered Newton VBD stepping for externally prescribed kinematic colliders."""

from __future__ import annotations

from typing import Any

import numpy as np


class KinematicVBDStepper:
    """Replay the same VBD substeps in a CUDA graph, updating collider inputs each frame."""

    def __init__(
        self,
        model: Any,
        solver: Any,
        pipeline: Any,
        contacts: Any,
        state_0: Any,
        state_1: Any,
        control: Any,
        substeps: int,
        dt: float,
        *,
        use_graph: bool = True,
    ) -> None:
        import warp as wp

        self.wp = wp
        self.model, self.solver, self.pipeline, self.contacts = model, solver, pipeline, contacts
        self.state_0, self.state_1, self.control = state_0, state_1, control
        self.substeps, self.dt = substeps, dt
        self.use_graph = use_graph and model.device.is_cuda and substeps % 2 == 0
        self.graph = None
        self.warmed = False
        count = substeps * model.body_count
        self.q_in = wp.empty(count, dtype=wp.transform, device=model.device)
        self.q_out = wp.empty(count, dtype=wp.transform, device=model.device)
        self.velocity = wp.empty(count, dtype=wp.spatial_vector, device=model.device)

    def step(self, q_in: np.ndarray, q_out: np.ndarray, velocity: np.ndarray) -> None:
        """Advance one frame from explicitly supplied substep collider transforms/velocities."""
        self.q_in.assign(np.asarray(q_in, dtype=np.float32).reshape(-1, 7))
        self.q_out.assign(np.asarray(q_out, dtype=np.float32).reshape(-1, 7))
        self.velocity.assign(np.asarray(velocity, dtype=np.float32).reshape(-1, 6))
        if self.use_graph and self.warmed:
            if self.graph is None:
                with self.wp.ScopedCapture(device=self.model.device) as capture:
                    self._simulate()
                self.graph = capture.graph
            self.wp.capture_launch(self.graph)
        else:
            self._simulate()
            self.warmed = True

    def _simulate(self) -> None:
        self.solver.rebuild_bvh(self.state_0)
        count = self.model.body_count
        for index in range(self.substeps):
            offset = index * count
            self.wp.copy(self.state_0.body_q, self.q_in, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_q, self.q_out, src_offset=offset, count=count)
            self.wp.copy(self.state_0.body_qd, self.velocity, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_qd, self.velocity, src_offset=offset, count=count)
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.wp.copy(self.state_0.body_q, self.q_out, src_offset=offset, count=count)
            self.wp.copy(self.state_1.body_q, self.q_out, src_offset=offset, count=count)
