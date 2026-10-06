"""Batched Torch port of the robot's joint PID (kaivuriprokkis ``modules/pid.py``).

Gains are tensors broadcast to ``[envs, joints]``, so every environment can run its own tuning. The arithmetic
follows ``PIDController.compute`` line for line, so gains tuned here can go straight into the robot's
``control_config.yaml``; ``training/test_pid.py`` checks this against the robot file itself.
"""

from __future__ import annotations

import math

import torch


class BatchedPID:
    """``PIDController`` for ``[envs, joints]``: derivative on measurement with a first-order filter, integral
    limits and sign-aware clamping anti-windup. Units follow the caller (rad and valve fraction here)."""

    def __init__(
        self,
        count: int,
        joints: int,
        kp,
        ki,
        kd,
        *,
        min_output: float = -1.0,
        max_output: float = 1.0,
        deriv_filter_tau=0.10,
        i_min=None,
        i_max=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        shape = (count, joints)

        def full(value):
            return torch.as_tensor(value, device=device, dtype=dtype).expand(shape).clone()

        self.kp, self.ki, self.kd = full(kp), full(ki), full(kd)
        self.min_output, self.max_output = min_output, max_output
        self.tau = full(deriv_filter_tau).clamp_min(0.0)
        if i_min is None or i_max is None:
            # The robot default: what the actuator can deliver through the I term alone; unbounded without I.
            nonzero = self.ki != 0
            safe_ki = torch.where(nonzero, self.ki, 1.0)
            i_min = torch.where(nonzero, min_output / safe_ki, -math.inf)
            i_max = torch.where(nonzero, max_output / safe_ki, math.inf)
        i_min, i_max = full(i_min), full(i_max)
        self.i_min, self.i_max = torch.minimum(i_min, i_max), torch.maximum(i_min, i_max)

        self.integral = torch.zeros(shape, device=device, dtype=dtype)
        self.last_value = torch.zeros_like(self.integral)
        self.has_last = torch.zeros(shape, device=device, dtype=torch.bool)
        self.d_filt = torch.zeros_like(self.integral)

    def reset(self, ids: torch.Tensor | None = None, keep_integral: bool = False) -> None:
        """Clear the state of the selected environments (all by default)."""
        ids = slice(None) if ids is None else ids
        if not keep_integral:
            self.integral[ids] = 0
        self.has_last[ids] = False
        self.d_filt[ids] = 0

    def compute(self, setpoint: torch.Tensor, current: torch.Tensor, dt: float) -> torch.Tensor:
        """Return the clamped output for one tick of ``dt`` seconds."""
        dt = max(float(dt), 1e-3)
        error = setpoint - current

        raw_d = torch.where(self.has_last, -(current - self.last_value) / dt, 0.0)
        alpha = dt / (self.tau + dt)
        self.d_filt = torch.where(self.tau == 0, raw_d, self.d_filt + alpha * (raw_d - self.d_filt))
        d_term = self.kd * self.d_filt

        tentative = torch.minimum(torch.maximum(self.integral + error * dt, self.i_min), self.i_max)
        unsat = self.kp * error + self.ki * tentative + d_term
        out = unsat.clamp(self.min_output, self.max_output)

        # Hold the integrator only while saturated AND the error pushes further into the limit.
        same_sign = ((unsat > 0) & (error > 0)) | ((unsat < 0) & (error < 0))
        self.integral = torch.where((out != unsat) & same_sign, self.integral, tentative)
        self.last_value = current.clone()
        self.has_last[:] = True
        return out


def robot_joint_command(pid: BatchedPID, target: torch.Tensor, q: torch.Tensor, dt: float) -> torch.Tensor:
    """Valve commands exactly as ``ExcavatorController`` calls its PIDs: ``compute(0, -wrap(target - q))``.

    The measurement it hands over is the negated wrapped error, so the "derivative on measurement" is really the
    derivative of the tracking error: a moving IK target does drive the D term on the robot.
    """
    error = torch.atan2(torch.sin(target - q), torch.cos(target - q))
    return pid.compute(torch.zeros_like(error), -error, dt)
