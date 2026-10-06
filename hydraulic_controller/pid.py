"""Batched Torch port of the robot's joint PID (kaivuriprokkis ``modules/pid.py``) and its control loop.

Gains are tensors broadcast to ``[envs, joints]``, so every environment can run its own tuning. The arithmetic
follows ``PIDController.compute`` line for line, so gains tuned here can go straight into the robot's
``control_config.yaml``; ``training/test_pid.py`` checks this against the robot file itself.

``JointPIDController`` is the robot's whole loop as a ``closed_loop`` controller: for tip lines the
damped-least-squares IK step ``dq = J^T (J J^T + lambda^2 I)^-1 e`` toward the reference pose, ``q + dq`` as the
PID target (so the PID sees ``dq``), then the PID; joint scenarios feed the joint reference straight in. Adaptive
damping and joint-limit repulsion are left out: lambda barely moves (0.001 to 0.002), and the task paths stay
clear of the repulsion margins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .tasks import DT, dls_step

ROBOT_DERIV_FILTER_TAU = 0.10  # PIDController's default; ExcavatorController does not override it


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


@dataclass
class PidGains:
    """Per-joint (boom, arm, bucket) gains in robot units: valve fraction per rad, per rad*s, per rad/s."""

    kp: list[float]
    ki: list[float]
    kd: list[float]
    deriv_filter_tau: list[float] = field(default_factory=lambda: [ROBOT_DERIV_FILTER_TAU] * 3)
    output_limits: tuple[float, float] = (-1.0, 1.0)
    ik_lambda: float = 0.001


def load_robot_gains(control_yaml: Path) -> PidGains:
    """Read boom/arm/bucket gains, output limits and the DLS lambda from a robot ``control_config.yaml``."""
    import yaml

    cfg = yaml.safe_load(Path(control_yaml).read_text())
    pid = [cfg["pid"][f"joint{i}"] for i in (1, 2, 3)]
    ctrl = cfg.get("controller", {})
    ik = cfg.get("ik", {})
    if ik.get("method", "dls") != "dls":
        raise ValueError(f"the sim loop models the dls IK method, the robot is set to {ik['method']!r}")
    return PidGains(
        kp=[float(j["kp"]) for j in pid],
        ki=[float(j["ki"]) for j in pid],
        kd=[float(j["kd"]) for j in pid],
        output_limits=(float(ctrl.get("output_limits_min", -1.0)), float(ctrl.get("output_limits_max", 1.0))),
        ik_lambda=float(ik.get("params", {}).get("lambda_val", 0.001)),
    )


class JointPIDController:
    """The robot's IK + joint-PID loop for ``closed_loop.rollout``.

    ``kp``/``ki``/``kd`` default to ``gains``; pass [C, 3] tensors to give each of ``C`` copies of the scenario
    set its own candidate (rows ``c * S + s``). Filter time constant, output limits and lambda come from ``gains``.
    """

    def __init__(self, gains: PidGains, kp=None, ki=None, kd=None):
        self.gains = gains
        self.candidates = [kp, ki, kd]

    def reset(self, task) -> None:
        device = task.q0.device
        self.task = task
        gains = [
            torch.as_tensor(value if value is not None else [default], dtype=torch.float32, device=device)
            for value, default in zip(self.candidates, (self.gains.kp, self.gains.ki, self.gains.kd))
        ]
        per_copy = task.count // len(gains[0])
        kp, ki, kd = (g.repeat_interleave(per_copy, 0) for g in gains)
        lo, hi = self.gains.output_limits
        self.pid = BatchedPID(
            task.count,
            3,
            kp,
            ki,
            kd,
            deriv_filter_tau=torch.tensor(self.gains.deriv_filter_tau, device=device),
            min_output=lo,
            max_output=hi,
            device=device,
        )
        self.pose0 = task.kin.pose_jacobian(task.q0)[0]

    def __call__(self, k: int, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        task = self.task
        tip_target = torch.where(task.is_tip[:, None], task.tip_ref[k], self.pose0)
        ik_target = q[:, :3] + dls_step(task.kin, q, tip_target, self.gains.ik_lambda)
        target = torch.where(task.is_tip[:, None], ik_target, task.q_ref[k])
        return robot_joint_command(self.pid, target, q[:, :3], DT)
