# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""A bounded, smooth bucket-angle sweep around a fixed cutting-tip position."""

from __future__ import annotations

import math

import numpy as np
import torch


class FixedTipRotation:
    """Three-point angle sweeps with quintic timing, plus a tip position feedback loop.

    Translation stays fixed. Each cycle goes 0 -> +amplitude -> -amplitude -> 0,
    relative to the initial bucket angle. Segment endpoints have zero speed and
    acceleration. Positive pitch follows the policy's positive-Y convention.
    """

    def __init__(self, initial_pose, amplitude_deg=5.0, rate_deg_s=3.0, cycles=3, kp=3.0):
        if not 0 < amplitude_deg <= 10 or not 0 < rate_deg_s <= 3 or cycles < 1 or kp <= 0:
            raise ValueError("Demo bounds: 0 < amplitude <= 10 deg, 0 < rate <= 3 deg/s, cycles >= 1")
        if not np.isfinite([amplitude_deg, rate_deg_s, cycles, kp]).all() or int(cycles) != cycles:
            raise ValueError("Finite parameters and an integer cycle count are required")
        self.initial = np.asarray(initial_pose, dtype=float).copy()
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected a finite [x, z, angle] starting pose")
        self.amplitude = math.radians(amplitude_deg)
        self.rate = math.radians(rate_deg_s)
        self.cycles, self.kp = int(cycles), kp
        # Maximum derivative of the unit quintic is 1.875.
        self.short_s = 1.875 * self.amplitude / self.rate
        self.cycle_s = 4 * self.short_s
        self.duration = self.cycles * self.cycle_s + 2.0

    def reference(self, t):
        """Return angle [rad] and angular feedforward [rad/s] at elapsed time [s]."""
        if t < 0 or t >= self.cycles * self.cycle_s:
            return self.initial[2], 0.0
        phase = t % self.cycle_s
        if phase < self.short_s:
            start, delta, elapsed, duration = 0.0, self.amplitude, phase, self.short_s
        elif phase < 3 * self.short_s:
            start, delta = self.amplitude, -2 * self.amplitude
            elapsed, duration = phase - self.short_s, 2 * self.short_s
        else:
            start, delta = -self.amplitude, self.amplitude
            elapsed, duration = phase - 3 * self.short_s, self.short_s
        s = elapsed / duration
        f = 10 * s**3 - 15 * s**4 + 6 * s**5
        derivative = 30 * s**2 * (1 - s) ** 2
        return self.initial[2] + start + delta * f, delta * derivative / duration

    def command(self, t, pose, speed_limit=0.03, pitch_rate_limit=0.3):
        """Desired twist [m/s, m/s, rad/s] with bounded position/angle correction."""
        theta, rate = self.reference(t)
        linear = self.kp * (self.initial[:2] - np.asarray(pose)[:2])
        linear *= min(1, speed_limit / max(1e-12, float(np.linalg.norm(linear))))
        error = math.atan2(math.sin(theta - pose[2]), math.cos(theta - pose[2]))
        return np.r_[linear, np.clip(rate + self.kp * error, -pitch_rate_limit, pitch_rate_limit)]

    def validate(self, kin, q):
        """Require the entire angular sweep to be reachable from q [rad]."""
        targets = torch.tensor(self.initial, device=q.device, dtype=q.dtype).repeat(41, 1)
        targets[:, 2] += torch.linspace(-self.amplitude, self.amplitude, 41, device=q.device)
        _, reached = kin.inverse(targets, q[:1])
        if not bool(reached.all()):
            raise ValueError("Fixed-tip sweep is not reachable with joint/collision margins at this pose")
