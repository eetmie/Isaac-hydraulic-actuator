"""Sampling MPC (MPPI) on the learned actuator network, as a ``closed_loop`` controller.

Every ``replan_ticks`` the controller rolls ``samples`` valve plans through its own copy of the actuator network
for ``horizon_ticks`` 100 Hz steps. The plans are piecewise constant over ``knot_ticks``, and each starts from
the measured history, the same angles, rates and past commands the robot has. A plan's cost is the squared tip
position and bucket-angle error against the upcoming reference, plus a valve-change penalty and soft joint-limit
walls. The new plan is the exponentially cost-weighted mean of the samples (Williams et al., "Information
Theoretic MPC for Model-Based Reinforcement Learning", ICRA 2017). The next plan starts from this one shifted by
one knot.

The plans live in a deadband-compensated command ``w`` in [-1, 1], not in raw valve opening. ``|w| > w0`` maps
linearly onto ``[edge, 1]``, where ``edge`` is the spool's measured deadband edge for that sign. Below ``w0`` a
steep ramp crosses the deadband, so the map stays continuous. Without this, small sampled changes inside the
deadband do nothing, the cost is flat there, and the solver sits still with an error. The edges come from
``speed_limits.spool_curves`` at HOME (where the spool reaches ``deadband_fraction`` of its full-valve speed).
Sampling and averaging happen in ``w``. The valve-change penalty is charged on the real opening, so a flip across
the deadband costs what it costs the hardware.

Costs are normalized per environment, ``(J - min J) / (median J - min J)``, so ``temperature`` means the same
thing whether the tip is 1 mm or 50 mm off.

The prediction model defaults to the plant's own nominal network. In the comparison its only mismatch is then
the hidden valve gain, offset, delay and speed perturbation, so pass ``model`` to plan with a different network
when that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .core import HOME, HydraulicModelBatch, wrap_angle
from .observations import MeasuredHistory
from .speed_limits import SpeedMapConfig, curve_summary, spool_curves
from .tasks import PID_JOINTS


@dataclass(frozen=True)
class MPPIConfig:
    """Sampling, horizon [100 Hz ticks] and cost weights (errors scaled by ``pos_scale_m`` / ``pitch_scale_rad``)."""

    samples: int = 256
    horizon_ticks: int = 40
    knot_ticks: int = 5
    replan_ticks: int = 5
    noise_std: float = 0.15
    temperature: float = 0.2
    pos_scale_m: float = 0.005
    pitch_scale_rad: float = 0.0175
    pitch_weight: float = 0.5
    valve_change_weight: float = 2.0
    effort_weight: float = 2.0  # on the real opening; zero valves hold still on this plant
    position_tolerance_m: float = 0.0  # tip errors below this cost nothing
    limit_margin_rad: float = 0.03
    limit_weight: float = 100.0
    velocity_limit: float = 2.0
    deadband_fraction: float = 0.03
    ramp_w: float = 0.05  # |w| over which the map crosses the deadband
    smoothing: float = 0.3  # executed w = smoothing * solved + (1 - smoothing) * previous; 1 = no filter
    seed: int = 0


class MPPIController:
    def __init__(self, model: str | Path, cfg: MPPIConfig = MPPIConfig()):
        if cfg.horizon_ticks % cfg.knot_ticks or cfg.replan_ticks != cfg.knot_ticks:
            raise ValueError("horizon must be whole knots, and the plan is re-solved once per knot")
        self.model_path, self.cfg = Path(model), cfg

    def reset(self, task) -> None:
        if not bool(task.is_tip.all()):
            raise ValueError("this MPC tracks tip motion; give it tip-line tasks only")
        cfg, device = self.cfg, task.q0.device
        n, knots = task.count, cfg.horizon_ticks // cfg.knot_ticks
        self.task, self.knots = task, knots
        self.model = HydraulicModelBatch(self.model_path, n * cfg.samples, str(device), task.dt)
        source = self.model.source
        self.history = MeasuredHistory(
            n, str(device), self.model.v_history.shape[1] + 1, self.model.u_history.shape[1], source.u_stride
        )
        self.history.reset(torch.arange(n, device=device), task.q0)
        self.plan = torch.zeros(n, knots, 3, device=device)
        self.u = torch.zeros(n, 3, device=device)
        self.w = torch.zeros(n, 3, device=device)
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed)
        self.limits = task.kin.limits
        home = torch.tensor([HOME], device=device)
        u_grid, curves = spool_curves(self.model_path, task.kin, home, SpeedMapConfig())
        edges = curve_summary(u_grid.cpu(), curves[0].cpu(), cfg.deadband_fraction)
        self.edges = torch.tensor(
            [[edges[f"{joint}_{side}"]["deadband_u"] for joint in PID_JOINTS] for side in ("neg", "pos")],
            device=device,
        )  # [2, 3]: negative, positive side

    @torch.no_grad()
    def __call__(self, k: int, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.history.push(q, v, self.u)
        if k % self.cfg.replan_ticks == 0:
            self.plan = self._solve(k)
            self.w = self.cfg.smoothing * self.plan[:, 0] + (1 - self.cfg.smoothing) * self.w
            self.u = self.valves(self.w)
            self.plan = torch.cat(
                (self.plan[:, 1:], self.plan[:, -1:]), dim=1
            )  # warm start for the next solve
        return self.u

    def valves(self, w: torch.Tensor) -> torch.Tensor:
        """Deadband-compensated command w [..., 3] -> valve opening u [..., 3], continuous and odd."""
        edge = torch.where(w < 0, self.edges[0], self.edges[1])
        size, ramp = w.abs(), self.cfg.ramp_w
        opening = torch.where(size < ramp, edge * size / ramp, edge + (1 - edge) * (size - ramp) / (1 - ramp))
        return torch.sign(w) * opening

    def seed_model(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the measured history into every sample's model row; returns the current (q, v) per row.

        ``predict`` shifts its buffers and then writes the current values in front, so the buffers are seeded with
        what lies one tick back: rates from the second sample on, and the commands sent before this tick.
        """
        h, m, model = self.history, self.cfg.samples, self.model
        model.q_history[:] = h.q.repeat_interleave(m, 0)[:, None]
        model.v_history[:] = h.v[:, 1:].repeat_interleave(m, 0)
        model.u_history[:] = h.u.repeat_interleave(m, 0)
        model.primed[:] = True
        return h.q.repeat_interleave(m, 0).clone(), h.v[:, 0].repeat_interleave(m, 0).clone()

    def _solve(self, k: int) -> torch.Tensor:
        cfg, task, model = self.cfg, self.task, self.model
        n, m, device = task.count, cfg.samples, self.plan.device
        noise = torch.randn(n, m, self.knots, 3, generator=self.generator, device=device) * cfg.noise_std
        noise[:, 0] = 0  # keep the warm-started plan itself among the samples
        plans = (self.plan[:, None] + noise).clamp(-1.0, 1.0)  # [N, M, K, 3]

        q, vel = self.seed_model()

        steps = task.tip_ref.shape[0]
        cost = torch.zeros(n * m, device=device)
        flat = plans.reshape(n * m, self.knots, 3)
        valves = self.valves(flat)
        for tick in range(cfg.horizon_ticks):
            u = valves[:, tick // cfg.knot_ticks]
            vel = model.predict(q, vel, u).clamp(-cfg.velocity_limit, cfg.velocity_limit)
            q = (q + task.dt * vel).clamp(self.limits[:, 0], self.limits[:, 1])  # the plant's end stops
            pose, _ = task.kin.pose_jacobian(q)
            reference = task.tip_ref[min(k + tick, steps - 1)].repeat_interleave(m, 0)
            miss = ((pose[:, :2] - reference[:, :2]).norm(dim=1) - cfg.position_tolerance_m).clamp_min(0)
            position = (miss / cfg.pos_scale_m) ** 2
            pitch = (wrap_angle(pose[:, 2] - reference[:, 2]) / cfg.pitch_scale_rad) ** 2
            below = (self.limits[:3, 0] + cfg.limit_margin_rad - q[:, :3]).clamp_min(0)
            above = (q[:, :3] - self.limits[:3, 1] + cfg.limit_margin_rad).clamp_min(0)
            walls = ((below + above) / cfg.limit_margin_rad).square().sum(1)
            cost += position + cfg.pitch_weight * pitch + cfg.limit_weight * walls
            cost += cfg.effort_weight * u.square().sum(1)
        # Valve changes are charged on the real opening, so a flip across the deadband costs what it costs the
        # hardware, not the small step it is in w.
        previous = torch.cat((self.u[:, None].repeat_interleave(m, 0), valves[:, :-1]), dim=1)
        cost += cfg.valve_change_weight * (valves - previous).square().sum((1, 2))

        cost = torch.nan_to_num(cost.reshape(n, m), nan=float("inf"))
        best = cost.min(1, keepdim=True).values
        spread = (cost.median(1, keepdim=True).values - best).clamp_min(1e-9)
        weights = torch.softmax(-(cost - best) / (spread * cfg.temperature), dim=1)
        return (weights[:, :, None, None] * plans).sum(1)
