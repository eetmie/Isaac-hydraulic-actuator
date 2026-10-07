# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""One observation layout for the simulated plant and measured hardware state."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def assemble_observation(q, velocity_history, command_history, u_stride, kin, command, point="tip"):
    """Assemble policy inputs from angles [rad], rates [rad/s], and twist commands [m/s, m/s, rad/s].

    ``point`` is the tracked point (``kinematics.TRACKED_POINTS``); pose and twist are that point's.
    """
    pose, jac = kin.pose_jacobian(q) if point == "tip" else kin.pose_jacobian(q, point)
    twist = torch.einsum("nij,nj->ni", jac[:, :, :3], velocity_history[:, 0, :3])
    return torch.cat(
        (
            q,
            velocity_history[:, ::2].flatten(1),
            command_history[:, ::u_stride].flatten(1),
            pose[:, :2],
            pose[:, 2:].sin(),
            pose[:, 2:].cos(),
            twist,
            command,
            command - twist,
        ),
        1,
    )


class MeasuredHistory:
    """100 Hz histories. Each measurement includes the command sent during the preceding interval."""

    def __init__(
        self,
        count: int,
        device: str,
        velocity_samples: int = 41,
        command_samples: int = 61,
        u_stride: int = 3,
    ):
        self.q = torch.zeros(count, 4, device=device)
        self.v = torch.zeros(count, velocity_samples, 4, device=device)
        self.u = torch.zeros(count, command_samples, 3, device=device)
        self.u_stride = u_stride

    def reset(self, ids, q):
        """Reset selected streams to angles [rad] with empty motion/command history."""
        self.q[ids] = q
        self.v[ids] = 0
        self.u[ids] = 0

    def push(self, q, velocity, preceding_command):
        """Append one 0.01 s measurement: angles [rad], rates [rad/s], normalized commands."""
        self.v[:, 1:] = self.v[:, :-1].clone()
        self.u[:, 1:] = self.u[:, :-1].clone()
        self.q.copy_(q)
        self.v[:, 0] = velocity
        self.u[:, 0] = preceding_command

    def observe(self, kin, command, point="tip"):
        """Return the shared observation vector for a twist command [m/s, m/s, rad/s]."""
        return assemble_observation(self.q, self.v, self.u, self.u_stride, kin, command, point)


@dataclass
class SensorSettings:
    """Independent link gyro noise/bias [rad/s], angle noise [rad], sensor delay [100 Hz ticks]."""

    gyro_noise_std: float = 0.005
    gyro_bias_max: float = 0.003
    angle_noise_std: float = 0.003
    delay_max_steps: int = 2

    def __post_init__(self):
        import math

        if any(
            not math.isfinite(x) or x < 0
            for x in (self.gyro_noise_std, self.gyro_bias_max, self.angle_noise_std)
        ):
            raise ValueError("Sensor uncertainties must be finite and nonnegative")
        if not isinstance(self.delay_max_steps, int) or not 0 <= self.delay_max_steps <= 10:
            raise ValueError("Sensor delay must be an integer in [0, 10]")


class SensorObservation:
    """Persistent per-sample sensor errors, separate from physical plant state and reward."""

    def __init__(self, plant, settings: SensorSettings):
        self.cfg = settings
        self.plant = plant
        self.history = MeasuredHistory(
            plant.count,
            plant.device,
            plant.model.v_history.shape[1],
            plant.u_cmd_history.shape[1],
            plant.model.source.u_stride,
        )
        self.bias = torch.zeros_like(plant.q)
        self.delay = torch.zeros(plant.count, device=plant.device, dtype=torch.long)
        self.rows = torch.arange(plant.count, device=plant.device)
        self.q_queue = torch.zeros(plant.count, settings.delay_max_steps + 1, 4, device=plant.device)
        self.v_queue = torch.zeros_like(self.q_queue)

    def reset(self, ids):
        """Reset sensor histories and randomize link biases and packet delay for selected episodes."""
        self.bias[ids] = (2 * torch.rand(len(ids), 4, device=self.plant.device) - 1) * self.cfg.gyro_bias_max
        self.delay[ids] = torch.randint(self.cfg.delay_max_steps + 1, (len(ids),), device=self.plant.device)
        self.q_queue[ids] = self.plant.q[ids, None]
        self.v_queue[ids] = 0
        self.history.reset(ids, self.plant.q[ids])

    def prime(self, ids):
        """Fill selected histories with the plant's own past motion, for episodes that start mid-motion."""
        plant = self.plant
        self.history.v[ids] = torch.cat(
            (plant.v[ids, None], plant.model.v_history[ids, :-1] * plant.speed_scale[ids, None]), dim=1
        )
        self.history.u[ids] = plant.u_cmd_history[ids]
        self.v_queue[ids] = plant.v[ids, None]

    def step(self):
        """Sample sensors once after a physical 100 Hz plant step."""
        # Shared link errors induce correlations between adjacent relative joints.
        link_error = self.bias + torch.randn_like(self.bias) * self.cfg.gyro_noise_std
        relative_error = torch.cat((link_error[:, 1:] - link_error[:, :-1], link_error[:, :1]), 1)
        for queue, sample in ((self.q_queue, self.plant.q), (self.v_queue, self.plant.v + relative_error)):
            queue[:, 1:] = queue[:, :-1].clone()
            queue[:, 0] = sample
        q = self.q_queue[self.rows, self.delay] + torch.randn_like(self.plant.q) * self.cfg.angle_noise_std
        self.history.push(q, self.v_queue[self.rows, self.delay], self.plant.u_cmd)

    def observe(self, command):
        """Return hardware-style observations without altering the plant [SI units]."""
        return self.history.observe(self.plant.kinematics, command, self.plant.tracked_point)

    def contract(self):
        """Describe the training sensor perturbations."""
        return asdict(self.cfg)
