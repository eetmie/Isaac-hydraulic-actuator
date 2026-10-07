# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared timed circle reference and the simulation-tuned joint PID path."""

from __future__ import annotations

import math

import numpy as np
import torch

from .pid import BatchedPID, PidGains, robot_joint_command
from .tasks import dls_step


class CircleTrajectory:
    """X/Z circles [m], constant cutting-lip angle [rad], trapezoidal arc speed [m/s]."""

    def __init__(
        self,
        initial_pose,
        radius_m=0.05,
        speed_m_s=0.02,
        direction="ccw",
        cycles=1,
        accel_m_s2=0.5,
        lead_s=1.0,
        tail_s=2.0,
    ):
        self.initial = np.asarray(initial_pose, dtype=np.float64).copy()
        if self.initial.shape != (3,) or not np.isfinite(self.initial).all():
            raise ValueError("Expected finite [x, z, blade angle] in meters/radians")
        values = [radius_m, speed_m_s, accel_m_s2, lead_s, tail_s, cycles]
        if not np.isfinite(values).all() or min(values[:3]) <= 0 or min(values[3:5]) < 0:
            raise ValueError("Circle dimensions, timing and speed must be finite and positive")
        if direction not in ("cw", "ccw") or isinstance(cycles, bool) or int(cycles) != cycles or cycles < 1:
            raise ValueError("Use cw/ccw and a positive integer cycle count")
        self.radius, self.speed, self.accel = radius_m, speed_m_s, accel_m_s2
        self.direction, self.cycles = direction, int(cycles)
        self.lead, self.tail = lead_s, tail_s
        self.center = self.initial[:2] - np.array([radius_m, 0.0])
        self.distance = 2 * math.pi * radius_m
        self.ramp_s = min(speed_m_s / accel_m_s2, math.sqrt(self.distance / accel_m_s2))
        self.peak_speed = self.ramp_s * accel_m_s2
        self.cruise_s = max(0.0, (self.distance - self.peak_speed * self.ramp_s) / self.peak_speed)
        self.lap_s = 2 * self.ramp_s + self.cruise_s
        self.move_end = self.lead + self.cycles * self.lap_s
        self.duration = self.move_end + self.tail

    def reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return timed tip pose [m, m, rad] and feedforward twist [m/s, m/s, rad/s]."""
        if not math.isfinite(t):
            raise ValueError("Reference time must be finite")
        if t <= self.lead or t >= self.move_end:
            return self.initial.copy(), np.zeros(3)
        phase = (t - self.lead) % self.lap_s
        if phase < self.ramp_s:
            arc, speed = 0.5 * self.accel * phase**2, self.accel * phase
        elif phase < self.ramp_s + self.cruise_s:
            arc = 0.5 * self.peak_speed * self.ramp_s + self.peak_speed * (phase - self.ramp_s)
            speed = self.peak_speed
        else:
            decel_t = phase - self.ramp_s - self.cruise_s
            arc = self.distance - 0.5 * self.accel * (self.ramp_s - decel_t) ** 2
            speed = max(0.0, self.peak_speed - self.accel * decel_t)
        sense = 1 if self.direction == "ccw" else -1
        angle = sense * arc / self.radius
        pose = np.r_[
            self.center + self.radius * np.array([math.cos(angle), math.sin(angle)]), self.initial[2]
        ]
        twist = np.array([-math.sin(angle), math.cos(angle), 0.0]) * sense * speed
        return pose, twist

    def command(self, t: float, pose: np.ndarray, kp=3.0) -> np.ndarray:
        """Feedforward plus bounded position feedback [m/s, m/s, rad/s] for the MLP."""
        target, feedforward = self.reference(t)
        error = target - pose
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        command = feedforward + kp * error
        command[:2] *= min(1.0, 0.06 / max(1e-12, float(np.linalg.norm(command[:2]))))
        command[2] = np.clip(command[2], -0.3, 0.3)
        return command

    def validate(self, kin, q: torch.Tensor) -> None:
        """Check an entire circle from measured joint angles [rad], before enabling the pump."""
        angle = torch.linspace(0, 2 * math.pi, 129, device=q.device)
        targets = torch.tensor(self.initial, device=q.device, dtype=q.dtype).repeat(len(angle), 1)
        targets[:, 0] += self.radius * (angle.cos() - 1)
        targets[:, 1] += self.radius * angle.sin()
        solved, reached = kin.inverse(targets, q[:1])
        if not bool(reached.all()):
            raise ValueError("Circle is outside joint/collision margins at this starting pose")
        _, jac = kin.pose_jacobian(solved)
        weighted = jac[:, :, :3] * q.new_tensor([1.0, 1.0, 0.2])[None, :, None]
        if not torch.isfinite(weighted).all() or bool((torch.linalg.cond(weighted) > 100).any()):
            raise ValueError("Circle passes too close to a kinematic singularity")


class CircleJointPID:
    """Measured joint angles → DLS target → robot-equivalent PID → normalized valves.

    Uses the same planar DLS and PID arithmetic as tune_pid.py. The robot's
    direct-command thread owns PWM output; its pose smoother is bypassed.
    """

    def __init__(self, gains: PidGains, kin):
        self.gains, self.kin = gains, kin
        self.pid = BatchedPID(
            1,
            3,
            gains.kp,
            gains.ki,
            gains.kd,
            deriv_filter_tau=gains.deriv_filter_tau,
            min_output=gains.output_limits[0],
            max_output=gains.output_limits[1],
        )

    def reset(self) -> None:
        """Clear integral and derivative state before motion."""
        self.pid.reset()

    def valves(self, q: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """Return boom/arm/bucket valves [-1, 1] from measured q [rad] and tip target [m, m, rad]."""
        next_q = q[:, :3] + dls_step(self.kin, q, target, self.gains.ik_lambda)
        return robot_joint_command(self.pid, next_q, q[:, :3], dt)

    def joint_valves(self, q: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
        """PID for a smooth start-pose approach in joint coordinates [rad]."""
        return robot_joint_command(self.pid, target, q[:, :3], dt)


class StartMove:
    """Quintic joint interpolation with zero endpoint speed, bounded to 0.05 rad/s."""

    def __init__(self, initial, target, kin):
        self.initial = initial[:1, :3].clone()
        self.target = target[:1, :3].clone()
        self.duration = max(3.0, 1.875 * float((self.target - self.initial).abs().max()) / 0.05)
        samples = initial[:1].repeat(129, 1)
        samples[:, :3] = self.initial + torch.linspace(0, 1, 129)[:, None] * (self.target - self.initial)
        if not bool(kin.valid(samples).all()):
            raise ValueError("Start-pose approach crosses a physical joint/collision bound")

    def reference(self, elapsed):
        phase = min(1.0, max(0.0, elapsed / self.duration))
        blend = phase**3 * (10 - 15 * phase + 6 * phase**2)
        return self.initial + blend * (self.target - self.initial)
