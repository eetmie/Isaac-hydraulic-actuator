# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Demo reference policy for a valve held into a mechanical end stop."""

from __future__ import annotations

import numpy as np


class EndStopGuard:
    """Hold a contacted boundary only while its own valve pushes into it.

    This is a reference policy, not learned restitution or a physical contact
    model. Neutral and reversal release immediately. Interior motion is unchanged.
    The integrating actuator remains responsible for the final position clamp.
    """

    def __init__(self, limits: np.ndarray, dt: float, command_threshold: float = 0.05) -> None:
        """Initialize with per-axis lower/upper limits [rad] and timestep [s]."""
        self.limits = np.asarray(limits)
        if self.limits.ndim != 2 or self.limits.shape[1] != 2 or not np.isfinite(self.limits).all():
            raise ValueError("Limits must be a finite [joints, 2] array")
        if np.any(self.limits[:, 0] >= self.limits[:, 1]) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Limits must be ordered and dt positive")
        if not 0 < command_threshold < 1:
            raise ValueError("Command threshold must be between zero and one")
        self.dt = dt
        self.threshold = command_threshold
        self.latched = np.zeros(len(self.limits), dtype=np.int8)

    def reset(self) -> None:
        """Clear contact history after a reset or control-mode change."""
        self.latched[:] = 0

    def apply(self, position: np.ndarray, velocity: np.ndarray, command: np.ndarray) -> np.ndarray:
        """Return guarded reference velocity [rad/s] from state [rad] and valve [-1,1]."""
        q, v, u = np.asarray(position), np.asarray(velocity), np.asarray(command)
        if any(a.shape != self.latched.shape for a in (q, v, u)):
            raise ValueError("State, velocity and valve must match the joint count")
        pushing = np.where(u > self.threshold, 1, np.where(u < -self.threshold, -1, 0))
        self.latched[self.latched != pushing] = 0
        # Already at a boundary: suppress inward model recoil while the valve
        # still pushes out. Do not use a broad near-limit zone to damp motion.
        at_lower = (q <= self.limits[:, 0] + 1e-6) & (pushing == -1)
        at_upper = (q >= self.limits[:, 1] - 1e-6) & (pushing == 1)
        self.latched[at_lower] = -1
        self.latched[at_upper] = 1
        result = np.where(self.latched != 0, np.zeros_like(v), v)
        # Preserve the first accepted arrival increment, then hold next step.
        proposed = q + result * self.dt
        self.latched[(proposed < self.limits[:, 0]) & (pushing == -1)] = -1
        self.latched[(proposed > self.limits[:, 1]) & (pushing == 1)] = 1
        return result
