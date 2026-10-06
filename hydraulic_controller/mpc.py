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
    cuda_graph: bool = True  # capture the horizon rollout once and replay it (CUDA only; halves the solve)
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
        n, m, knots = task.count, cfg.samples, cfg.horizon_ticks // cfg.knot_ticks
        self.task, self.knots = task, knots
        source = HydraulicModelBatch(self.model_path, 1, str(device), task.dt).source
        if source.hist_q != 1 or not source._has_q:
            raise ValueError("the MPC predictor expects a model fed the current angles only")
        self.source, self.net = source, source._model.requires_grad_(False).eval()
        # The prediction buffers, one row per (environment, sample); same layout as HydraulicModelBatch.
        self.v_buffer = torch.zeros(n * m, len(source._qdot_buf), 4, device=device)
        self.u_buffer = torch.zeros(n * m, len(source._u_buf), 3, device=device)
        self.history = MeasuredHistory(
            n, str(device), self.v_buffer.shape[1] + 1, self.u_buffer.shape[1], source.u_stride
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
        # Fixed inputs of one horizon rollout, so it can be captured as a CUDA graph and replayed.
        self.plans_in = torch.zeros(n * m, knots, 3, device=device)
        self.reference_in = torch.zeros(cfg.horizon_ticks, n, 3, device=device)
        self.u_in = torch.zeros(n, 3, device=device)
        self.graph, self.cost_out = None, None
        self.use_graph = cfg.cuda_graph and device.type == "cuda"

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
        """Load the measured history into every sample's prediction row; returns the current (q, v) per row.

        ``predict`` shifts its buffers and then writes the current values in front, so the buffers are seeded with
        what lies one tick back: rates from the second sample on, and the commands sent before this tick.
        """
        h, m = self.history, self.cfg.samples
        self.v_buffer.copy_(h.v[:, 1:].repeat_interleave(m, 0))
        self.u_buffer.copy_(h.u.repeat_interleave(m, 0))
        return h.q.repeat_interleave(m, 0), h.v[:, 0].repeat_interleave(m, 0)

    def predict(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``HydraulicModelBatch.predict`` without its data-dependent priming, so a CUDA graph can hold it."""
        src = self.source
        self.v_buffer[:, 1:] = self.v_buffer[:, :-1].clone()
        self.u_buffer[:, 1:] = self.u_buffer[:, :-1].clone()
        self.v_buffer[:, 0], self.u_buffer[:, 0] = v, u
        features = [q]
        if src.hist_qdot:
            features.append(self.v_buffer[:, :: src.qdot_stride].flatten(1))
        features.append(self.u_buffer[:, :: src.u_stride].flatten(1))
        x = torch.cat(features, dim=1)
        prediction = self.net((x - src._x_mean) / src._x_std) * src._y_std + src._y_mean
        return v + prediction if src.target_mode == "delta_velocity" else prediction

    def horizon_cost(self) -> torch.Tensor:
        """Cost [N * M] of ``plans_in`` (w) against ``reference_in``, starting from the measured history."""
        cfg, task, m = self.cfg, self.task, self.cfg.samples
        q, vel = self.seed_model()
        valves = self.valves(self.plans_in)
        cost = torch.zeros(len(q), device=q.device)
        for tick in range(cfg.horizon_ticks):
            u = valves[:, tick // cfg.knot_ticks]
            vel = self.predict(q, vel, u).clamp(-cfg.velocity_limit, cfg.velocity_limit)
            q = (q + task.dt * vel).clamp(self.limits[:, 0], self.limits[:, 1])  # the plant's end stops
            pose, _ = task.kin.pose_jacobian(q)
            reference = self.reference_in[tick].repeat_interleave(m, 0)
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
        previous = torch.cat((self.u_in.repeat_interleave(m, 0)[:, None], valves[:, :-1]), dim=1)
        return cost + cfg.valve_change_weight * (valves - previous).square().sum((1, 2))

    def _horizon_cost_graphed(self) -> torch.Tensor:
        if self.graph is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm up off the capture, as torch.cuda.graphs requires
                for _ in range(3):
                    self.horizon_cost()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.cost_out = self.horizon_cost()
        self.graph.replay()
        return self.cost_out.clone()

    def _solve(self, k: int) -> torch.Tensor:
        cfg, task = self.cfg, self.task
        n, m, device = task.count, cfg.samples, self.plan.device
        noise = torch.randn(n, m, self.knots, 3, generator=self.generator, device=device) * cfg.noise_std
        noise[:, 0] = 0  # keep the warm-started plan itself among the samples
        plans = (self.plan[:, None] + noise).clamp(-1.0, 1.0)  # [N, M, K, 3]
        steps = task.tip_ref.shape[0]
        ticks = torch.arange(k, k + cfg.horizon_ticks, device=device).clamp(max=steps - 1)
        self.plans_in.copy_(plans.reshape(n * m, self.knots, 3))
        self.reference_in.copy_(task.tip_ref[ticks])
        self.u_in.copy_(self.u)
        cost = self._horizon_cost_graphed() if self.use_graph else self.horizon_cost()

        cost = torch.nan_to_num(cost.reshape(n, m), nan=float("inf"))
        best = cost.min(1, keepdim=True).values
        spread = (cost.median(1, keepdim=True).values - best).clamp_min(1e-9)
        weights = torch.softmax(-(cost - best) / (spread * cfg.temperature), dim=1)
        return (weights[:, :, None, None] * plans).sum(1)
