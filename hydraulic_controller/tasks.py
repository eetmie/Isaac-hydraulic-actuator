"""Closed-loop tasks shared by every controller: scenarios, references and what the plant could follow at all.

Scenario families, each from HOME and optional extra start poses, under every ``benchmark.PLANTS``
perturbation:

* **joint_step / joint_ramp**: one joint steps or ramps while the other two hold (joint-space controllers only).
* **tip_line**: a straight tip line with a trapezoidal speed profile and the bucket angle held.
* **tip_circle** (opt-in): one full circle, counterclockwise or clockwise in the X/Z plane, starting on the circle at
  the start pose with its center ``circle_radius_m`` toward -X (as ``benchmark.run_trajectories``), trapezoidal
  along the arc, bucket angle held.

Feasibility is decided before any controller runs. A reference path must stay inside the joint limits and clear
of collisions, and the plant must be fast enough to follow it: ramps against the single-spool full-valve speed,
tip lines against ``speed_limits.achievable_tip_speeds`` along their path, flow sharing included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .benchmark import PLANTS
from .core import HOME, measure_joint_speed_limits, wrap_angle
from .kinematics import ExcavatorKinematics
from .speed_limits import SpeedMapConfig, achievable_tip_speeds, direction_vectors

PID_JOINTS = ("boom", "arm", "bucket")  # the robot's pid joint1..joint3, which are sim q[:, 0:3]
FAMILIES = ("joint_step", "joint_ramp", "tip_line")  # the default task set; tip_circle is opt-in
TIP_FAMILIES = ("tip_line", "tip_circle")
DT = 0.01
# Central working poses besides HOME, as tip (x, z) [m] at the HOME bucket angle: tucked in, low, high. Tuning and
# comparison run from these so the reach edges, where tracking speed dies, do not decide anything.
WORKING_STARTS = ((0.42, -0.05), (0.50, -0.25), (0.55, 0.12))


@dataclass(frozen=True)
class TaskConfig:
    """Scenario sizes [deg, deg/s, m, m/s] and timing [s]."""

    step_deg: tuple[float, ...] = (3.0, 10.0)
    ramp_deg_s: tuple[float, ...] = (5.0, 10.0)  # full-valve boom-down is only ~12 deg/s
    ramp_travel_deg: float = 10.0
    tip_speeds_m_s: tuple[float, ...] = (0.02, 0.04, 0.07)
    tip_travel_m: float = 0.10
    tip_accel_m_s2: float = 0.5  # pathing_config max_accel_mps2
    circle_speeds_m_s: tuple[float, ...] = (0.02,)
    circle_radius_m: float = 0.05
    # Start poses: HOME (unless include_home is off), then tips (x, z) [m] at the HOME bucket angle, then joint
    # angles [rad, 4 per pose].
    include_home: bool = True
    start_tips: tuple[tuple[float, float], ...] = ()
    start_q: tuple[tuple[float, ...], ...] = ()
    families: tuple[str, ...] = FAMILIES
    lead_s: float = 1.0  # hold before the move starts, so the controller holds the start pose first
    duration_s: float = 7.0
    tail_s: float = 1.0  # final window for steady-state metrics
    settle_band_deg: float = 0.25
    settle_band_mm: float = 2.0
    # Required / achievable speed above which no controller can track: tip lines use the solved tip speed along
    # their path (speed_limits), joint ramps the single-spool full-valve speed.
    feasible_speed_ratio: float = 0.9
    path_samples: int = 8
    path_joint_margin: float = 0.02  # rad; a reference path must stay this far inside the joint limits
    sensor_noise_deg: float = 0.0  # white noise on the joint angles every controller sees
    plants: tuple[str, ...] = tuple(PLANTS)


@dataclass(frozen=True)
class Scenario:
    family: str  # "joint_step", "joint_ramp", "tip_line" or "tip_circle"
    joint: str  # moved joint, the line direction ("+X", "-Z", ...) or the circle sense ("ccw", "cw")
    size: float  # step deg, ramp deg/s, or tip speed m/s
    plant: str
    start: int = 0  # index into the start poses (0 is HOME when it is included)

    @property
    def is_tip(self) -> bool:
        return self.family in TIP_FAMILIES

    @property
    def name(self) -> str:
        unit = {"joint_step": "deg", "joint_ramp": "deg/s", "tip_line": "mm/s", "tip_circle": "mm/s"}[
            self.family
        ]
        size = self.size * 1000 if self.is_tip else self.size
        where = f"@s{self.start}" if self.start else ""
        return f"{self.family}:{self.joint}:{size:g}{unit}{where}"


def trapezoid(distance: float, speed: float, accel: float, t: torch.Tensor) -> torch.Tensor:
    """Distance travelled at times t [s] along a trapezoidal (or triangular) speed profile."""
    t_acc = min(speed / accel, math.sqrt(distance / accel))
    peak = accel * t_acc
    t_cruise = max(0.0, (distance - peak * t_acc) / peak)
    total = 2 * t_acc + t_cruise
    t = t.clamp(0.0, total)
    accel_part = 0.5 * accel * t.clamp(max=t_acc) ** 2
    cruise_part = peak * (t - t_acc).clamp(0.0, t_cruise)
    t_dec = (t - t_acc - t_cruise).clamp(0.0, t_acc)
    return accel_part + cruise_part + peak * t_dec - 0.5 * accel * t_dec**2


def start_poses(kin: ExcavatorKinematics, cfg: TaskConfig, device: str) -> torch.Tensor:
    """HOME, the IK solutions of ``cfg.start_tips`` at the HOME bucket angle, then ``cfg.start_q``; [K, 4] rad."""
    home = torch.tensor([HOME], device=device)
    parts = [home] if cfg.include_home else []
    if cfg.start_tips:
        pitch = kin.pose_jacobian(home)[0][0, 2]
        targets = torch.tensor([[x, z, 0.0] for x, z in cfg.start_tips], device=device)
        targets[:, 2] = pitch
        q, valid = kin.inverse(targets, home)
        if not valid.all():
            bad = [tip for tip, ok in zip(cfg.start_tips, valid.tolist()) if not ok]
            raise ValueError(f"start tips not reachable at the HOME bucket angle: {bad}")
        parts.append(q)
    if cfg.start_q:
        parts.append(torch.tensor(cfg.start_q, device=device, dtype=torch.float32))
    if not parts:
        raise ValueError("no start poses: include HOME or give start_tips / start_q")
    return torch.cat(parts)


def build_scenarios(cfg: TaskConfig, starts: int = 1) -> list[Scenario]:
    out = []
    for plant in cfg.plants:
        for start in range(starts):
            for joint in PID_JOINTS:
                for sign in (1.0, -1.0):
                    out += [Scenario("joint_step", joint, sign * a, plant, start) for a in cfg.step_deg]
                    out += [Scenario("joint_ramp", joint, sign * r, plant, start) for r in cfg.ramp_deg_s]
            for direction in ("+X", "-X", "+Z", "-Z"):
                out += [Scenario("tip_line", direction, v, plant, start) for v in cfg.tip_speeds_m_s]
            for sense in ("ccw", "cw"):
                out += [Scenario("tip_circle", sense, v, plant, start) for v in cfg.circle_speeds_m_s]
    return [s for s in out if s.family in cfg.families]


def reference(cfg: TaskConfig, scenarios: list[Scenario], q0: torch.Tensor, pose0: torch.Tensor):
    """Joint targets [T, N, 3] rad and tip targets [T, N, 3] (m, m, rad) from each row's start q0 / pose0.

    A row's unused target stays NaN.
    """
    device = q0.device
    steps = round(cfg.duration_s / DT)
    t = torch.arange(1, steps + 1, device=device, dtype=torch.float32) * DT - cfg.lead_s
    q_ref = torch.full((steps, len(scenarios), 3), math.nan, device=device)
    tip_ref = torch.full_like(q_ref, math.nan)
    for n, s in enumerate(scenarios):
        if s.family == "tip_line":
            axis = 0 if s.joint[1] == "X" else 1
            sign = 1.0 if s.joint[0] == "+" else -1.0
            tip_ref[:, n] = pose0[n]
            tip_ref[:, n, axis] += sign * trapezoid(cfg.tip_travel_m, s.size, cfg.tip_accel_m_s2, t)
            continue
        if s.family == "tip_circle":
            radius = cfg.circle_radius_m
            arc = trapezoid(2 * math.pi * radius, s.size, cfg.tip_accel_m_s2, t)
            theta = (1.0 if s.joint == "ccw" else -1.0) * arc / radius
            tip_ref[:, n] = pose0[n]
            tip_ref[:, n, 0] += radius * (theta.cos() - 1)  # center sits radius toward -X of the start
            tip_ref[:, n, 1] += radius * theta.sin()
            continue
        j = PID_JOINTS.index(s.joint)
        q_ref[:, n] = q0[n, :3]
        if s.family == "joint_step":
            q_ref[:, n, j] += torch.where(t >= 0, math.radians(s.size), 0.0)
        else:
            travel = math.copysign(math.radians(cfg.ramp_travel_deg), s.size)
            q_ref[:, n, j] += (t.clamp(min=0) * math.radians(s.size)).clamp(-abs(travel), abs(travel))
    return q_ref, tip_ref


def dls_step(kin: ExcavatorKinematics, q: torch.Tensor, tip_target: torch.Tensor, lam: float) -> torch.Tensor:
    """Robot ``_ik_dls`` on the planar (x, z, pitch) task over boom/arm/bucket [rad]."""
    pose, jacobian = kin.pose_jacobian(q)
    j = jacobian[:, :, :3]
    error = tip_target - pose
    error[:, 2] = wrap_angle(error[:, 2])
    a = j @ j.transpose(1, 2) + lam * lam * torch.eye(3, device=q.device)
    return (j.transpose(1, 2) @ torch.linalg.solve(a, error[:, :, None])).squeeze(-1)


@torch.no_grad()
def required_speed_ratio(
    model: Path,
    kin: ExcavatorKinematics,
    cfg: TaskConfig,
    scenarios: list[Scenario],
    q0: torch.Tensor,
    q_ref,
    tip_ref,
    capacity: torch.Tensor,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Required over achievable speed on the nominal plant (steps: inf), and whether each reference path stays
    valid (in limits with ``path_joint_margin``, collision-free). Both are controller-independent.

    Joint ramps compare their rate with the single-spool full-valve speed at their start pose (``capacity`` is
    [N, 3, 2]). Tip lines are solved kinematically (DLS iterated to convergence each tick); their speed is
    compared with the slowest ``achievable_tip_speeds`` at ``path_samples`` poses along that path, which counts
    flow sharing between joints.
    """
    q = q0.clone()
    is_tip = torch.tensor([s.is_tip for s in scenarios], device=q_ref.device)
    path, valid = [], torch.ones(len(scenarios), dtype=torch.bool, device=q.device)
    for k in range(q_ref.shape[0]):
        for _ in range(5):
            q[:, :3] += torch.where(is_tip[:, None], dls_step(kin, q, torch.nan_to_num(tip_ref[k]), lam), 0.0)
        step = torch.where(is_tip[:, None], q[:, :3], q_ref[k])
        path.append(step)
        valid &= kin.valid(torch.cat((step, q0[:, 3:]), dim=1), cfg.path_joint_margin)
    path = torch.stack(path)
    speed = torch.diff(path, dim=0) / DT  # [T-1, N, 3]
    limit = torch.where(speed < 0, capacity[:, :, 0], capacity[:, :, 1])
    ratio = (speed.abs() / limit).amax(dim=(0, 2))

    paths = {}  # one solve per (family, direction, speed, start); every plant shares the nominal answer
    for n, s in enumerate(scenarios):
        if s.is_tip:
            paths.setdefault((s.family, s.joint, s.size, s.start), []).append(n)
    if paths:
        # Lines check their own direction; circles turn, so each sample takes the nearest of 12 directions.
        bins = (
            (0.0, 90.0, 180.0, 270.0)
            if all(key[0] == "tip_line" for key in paths)
            else tuple(range(0, 360, 30))
        )
        line_bins = {"+X": 0.0, "+Z": 90.0, "-X": 180.0, "-Z": 270.0}
        poses, columns = [], []
        for (family, direction, _, _), rows in paths.items():
            n = rows[0]
            step = torch.diff(tip_ref[:, n, :2], dim=0)
            moving = (step.norm(dim=1) > 0).nonzero().squeeze(1)
            if len(moving) == 0:
                raise ValueError("a tip path never starts moving: duration_s must exceed lead_s")
            picks = moving[torch.linspace(0, len(moving) - 1, cfg.path_samples).round().long()]
            sample = torch.zeros(cfg.path_samples, 4, device=path.device)
            sample[:, :3] = path[picks, n]
            poses.append(sample)
            if family == "tip_line":
                columns.append(
                    torch.full((cfg.path_samples,), bins.index(line_bins[direction]), device=path.device)
                )
            else:
                heading = torch.rad2deg(torch.atan2(step[picks, 1], step[picks, 0])) % 360.0
                columns.append(((heading / (360.0 / len(bins))).round().long()) % len(bins))
        speed_cfg = SpeedMapConfig(directions_deg=bins)
        reach, _ = achievable_tip_speeds(
            model, kin, torch.cat(poses), direction_vectors(speed_cfg.directions_deg, path.device), speed_cfg
        )
        reach = reach.reshape(len(paths), cfg.path_samples, len(bins))
        for i, ((_, _, size, _), rows) in enumerate(paths.items()):
            slowest = reach[i].gather(1, columns[i][:, None]).min()
            ratio[rows] = size / slowest if slowest > 0 else math.inf
    is_step = torch.tensor([s.family == "joint_step" for s in scenarios], device=q_ref.device)
    return torch.where(is_step, math.inf, ratio), valid


@dataclass
class Prepared:
    """Scenarios with their references, start poses, plant perturbations and controller-independent checks."""

    cfg: TaskConfig
    model: Path
    kin: ExcavatorKinematics
    scenarios: list[Scenario]
    q0: torch.Tensor  # [S, 4] rad
    q_ref: torch.Tensor  # [T, S, 3] rad, NaN for tip lines
    tip_ref: torch.Tensor  # [T, S, 3] (m, m, rad), NaN for joint scenarios
    speed_ratio: torch.Tensor  # [S]
    path_valid: torch.Tensor  # [S]
    capacity: torch.Tensor  # [S, 3, 2] rad/s, single-spool full valve at the start pose

    @property
    def feasible(self) -> torch.Tensor:
        """Paths a perfect controller could follow: fast enough plant and a valid reference."""
        is_step = torch.tensor([s.family == "joint_step" for s in self.scenarios], device=self.q0.device)
        return self.path_valid & (is_step | (self.speed_ratio <= self.cfg.feasible_speed_ratio))

    def subset(self, keep: torch.Tensor) -> "Prepared":
        rows = keep.nonzero().squeeze(1)
        return Prepared(
            self.cfg,
            self.model,
            self.kin,
            [self.scenarios[i] for i in rows.tolist()],
            self.q0[rows],
            self.q_ref[:, rows],
            self.tip_ref[:, rows],
            self.speed_ratio[rows],
            self.path_valid[rows],
            self.capacity[rows],
        )


@torch.no_grad()
def prepare(cfg: TaskConfig, model: Path, asset: Path, device: str, lam: float = 0.001) -> Prepared:
    """Build scenarios and everything about them that does not depend on the controller."""
    kin = ExcavatorKinematics(asset, device)
    starts = start_poses(kin, cfg, device)
    scenarios = build_scenarios(cfg, len(starts))
    index = torch.tensor([s.start for s in scenarios], device=device)
    q0 = starts[index]
    q_ref, tip_ref = reference(cfg, scenarios, q0, kin.pose_jacobian(q0)[0])
    capacity = torch.stack([measure_joint_speed_limits(model, kin, q[None].repeat(4, 1)) for q in starts])
    capacity = capacity[index]
    ratio, valid = required_speed_ratio(model, kin, cfg, scenarios, q0, q_ref, tip_ref, capacity, lam)
    return Prepared(cfg, Path(model), kin, scenarios, q0, q_ref, tip_ref, ratio, valid, capacity)
