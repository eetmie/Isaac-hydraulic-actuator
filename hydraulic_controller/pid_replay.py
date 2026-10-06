"""Replay the robot's joint PID gains on the learned hydraulic plant.

The loop mirrors ``ExcavatorController`` at 100 Hz: a target, then the PID on the wrapped joint error, then the
valves. Two scenario families run as one batch, each under every ``benchmark.PLANTS`` perturbation:

* **joint** — one joint steps or ramps while the other two hold. This is the bare PID dynamics.
* **tip** — straight tip lines with a trapezoidal speed profile and the bucket angle held. Each tick takes the
  robot's damped-least-squares IK step ``dq = J^T (J J^T + lambda^2 I)^-1 e``. ``q + dq`` becomes the PID target,
  so the PID sees ``dq``, as on the robot. Adaptive damping and joint-limit repulsion are left out: lambda
  barely moves (0.001 to 0.002), and the lines stay clear of the repulsion margins.

Sensing is ideal: the PID sees the plant's own joint angles, with no IMU noise or delay.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .benchmark import PLANTS
from .core import HOME, HydraulicPlant, measure_joint_speed_limits, sha256, wrap_angle
from .kinematics import ExcavatorKinematics
from .pid import BatchedPID, robot_joint_command
from .speed_limits import SpeedMapConfig, achievable_tip_speeds, direction_vectors

PID_JOINTS = ("boom", "arm", "bucket")  # the robot's pid joint1..joint3, which are sim q[:, 0:3]
ROBOT_DERIV_FILTER_TAU = 0.10  # PIDController's default; ExcavatorController does not override it
DT = 0.01


@dataclass
class PidGains:
    """Per-joint gains in robot units: valve fraction per rad, per rad*s, per rad/s."""

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
        raise ValueError(f"replay models the dls IK method, the robot is set to {ik['method']!r}")
    return PidGains(
        kp=[float(j["kp"]) for j in pid],
        ki=[float(j["ki"]) for j in pid],
        kd=[float(j["kd"]) for j in pid],
        output_limits=(float(ctrl.get("output_limits_min", -1.0)), float(ctrl.get("output_limits_max", 1.0))),
        ik_lambda=float(ik.get("params", {}).get("lambda_val", 0.001)),
    )


@dataclass(frozen=True)
class ReplayConfig:
    """Scenario sizes [deg, deg/s, m, m/s] and timing [s]."""

    step_deg: tuple[float, ...] = (3.0, 10.0)
    ramp_deg_s: tuple[float, ...] = (5.0, 10.0)  # full-valve boom-down is only ~12 deg/s
    ramp_travel_deg: float = 10.0
    tip_speeds_m_s: tuple[float, ...] = (0.02, 0.04, 0.07)
    tip_travel_m: float = 0.10
    tip_accel_m_s2: float = 0.5  # pathing_config max_accel_mps2
    lead_s: float = 1.0  # hold before the move starts, so the PID holds HOME first
    duration_s: float = 7.0
    tail_s: float = 1.0  # final window for steady-state metrics
    settle_band_deg: float = 0.25
    settle_band_mm: float = 2.0
    # Required / achievable speed above which no controller can track: tip lines use the solved tip speed along
    # their path (speed_limits), joint ramps the single-spool full-valve speed.
    feasible_speed_ratio: float = 0.9
    path_samples: int = 8
    plants: tuple[str, ...] = tuple(PLANTS)


@dataclass(frozen=True)
class Scenario:
    family: str  # "joint_step", "joint_ramp" or "tip_line"
    joint: str  # moved joint, or the tip direction ("+X", "-Z", ...)
    size: float  # step deg, ramp deg/s, or tip speed m/s
    plant: str

    @property
    def name(self) -> str:
        unit = {"joint_step": "deg", "joint_ramp": "deg/s", "tip_line": "mm/s"}[self.family]
        size = self.size * 1000 if self.family == "tip_line" else self.size
        return f"{self.family}:{self.joint}:{size:g}{unit}"


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


def build_scenarios(cfg: ReplayConfig) -> list[Scenario]:
    out = []
    for plant in cfg.plants:
        for joint in PID_JOINTS:
            for sign in (1.0, -1.0):
                out += [Scenario("joint_step", joint, sign * a, plant) for a in cfg.step_deg]
                out += [Scenario("joint_ramp", joint, sign * r, plant) for r in cfg.ramp_deg_s]
        for direction in ("+X", "-X", "+Z", "-Z"):
            out += [Scenario("tip_line", direction, v, plant) for v in cfg.tip_speeds_m_s]
    return out


def reference(cfg: ReplayConfig, scenarios: list[Scenario], home_pose: torch.Tensor, device: str):
    """Joint targets [T, N, 3] rad and tip targets [T, N, 3] (m, m, rad); a row's unused target stays NaN."""
    steps = round(cfg.duration_s / DT)
    t = torch.arange(1, steps + 1, device=device, dtype=torch.float32) * DT - cfg.lead_s
    home = torch.tensor(HOME[:3], device=device)
    q_ref = torch.full((steps, len(scenarios), 3), math.nan, device=device)
    tip_ref = torch.full_like(q_ref, math.nan)
    for n, s in enumerate(scenarios):
        if s.family == "tip_line":
            axis = 0 if s.joint[1] == "X" else 1
            sign = 1.0 if s.joint[0] == "+" else -1.0
            tip_ref[:, n] = home_pose
            tip_ref[:, n, axis] += sign * trapezoid(cfg.tip_travel_m, s.size, cfg.tip_accel_m_s2, t)
            continue
        j = PID_JOINTS.index(s.joint)
        q_ref[:, n] = home
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
    cfg: ReplayConfig,
    scenarios: list[Scenario],
    q_ref,
    tip_ref,
    capacity: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Required over achievable speed on the nominal plant, per scenario (steps: inf). Controller-independent.

    Joint ramps compare their rate with the single-spool full-valve speed at HOME. Tip lines are solved
    kinematically (DLS iterated to convergence each tick); their speed is compared with the slowest
    ``achievable_tip_speeds`` at ``path_samples`` poses along that path, which counts flow sharing between joints.
    """
    q = torch.tensor(HOME, device=q_ref.device).repeat(len(scenarios), 1)
    is_tip = torch.tensor([s.family == "tip_line" for s in scenarios], device=q_ref.device)
    path = []
    for k in range(q_ref.shape[0]):
        for _ in range(5):
            q[:, :3] += torch.where(is_tip[:, None], dls_step(kin, q, torch.nan_to_num(tip_ref[k]), lam), 0.0)
        path.append(torch.where(is_tip[:, None], q[:, :3], q_ref[k]))
    path = torch.stack(path)
    speed = torch.diff(path, dim=0) / DT  # [T-1, N, 3]
    limit = torch.where(speed < 0, capacity[:, 0], capacity[:, 1])
    ratio = (speed.abs() / limit).amax(dim=(0, 2))

    lines = {}  # one solve per (direction, speed); every plant shares the nominal answer
    for n, s in enumerate(scenarios):
        if s.family == "tip_line":
            lines.setdefault((s.joint, s.size), []).append(n)
    if lines:
        poses, directions = [], []
        for (direction, _), rows in lines.items():
            n = rows[0]
            moving = (torch.diff(tip_ref[:, n, :2], dim=0).norm(dim=1) > 0).nonzero().squeeze(1)
            picks = moving[torch.linspace(0, len(moving) - 1, cfg.path_samples).round().long()]
            q = torch.zeros(cfg.path_samples, 4, device=path.device)
            q[:, :3] = path[picks, n]
            poses.append(q)
            directions.append({"+X": 0.0, "+Z": 90.0, "-X": 180.0, "-Z": 270.0}[direction])
        speed_cfg = SpeedMapConfig(directions_deg=tuple(directions))
        reach, _ = achievable_tip_speeds(
            model, kin, torch.cat(poses), direction_vectors(speed_cfg.directions_deg, path.device), speed_cfg
        )  # [L * samples, L]: every pose against every line's direction
        reach = reach.reshape(len(lines), cfg.path_samples, len(lines))
        for i, ((_, size), rows) in enumerate(lines.items()):
            slowest = reach[i, :, i].min()
            ratio[rows] = size / slowest if slowest > 0 else math.inf
    is_step = torch.tensor([s.family == "joint_step" for s in scenarios], device=q_ref.device)
    return torch.where(is_step, math.inf, ratio)


@torch.no_grad()
def run_replay(
    gains: PidGains, cfg: ReplayConfig, model: Path, asset: Path, device: str = "cuda:0"
) -> tuple[list[Scenario], dict[str, torch.Tensor]]:
    """Roll every scenario out once; returns per-tick traces on the CPU."""
    kin = ExcavatorKinematics(asset, device)
    scenarios = build_scenarios(cfg)
    count = len(scenarios)
    plant = HydraulicPlant(model, kin, count, device)
    for n, s in enumerate(scenarios):
        p = PLANTS[s.plant]
        plant.valve_gain[n] = p["gain"]
        plant.valve_offset[n] = p["offset"]
        plant.action_delay[n] = p["delay"]
        plant.speed_scale[n, :3] = p["speed"]
    plant.reset(torch.arange(count, device=device))

    home_pose = kin.pose_jacobian(torch.tensor([HOME], device=device))[0][0]
    q_ref, tip_ref = reference(cfg, scenarios, home_pose, device)
    is_tip = torch.tensor([s.family == "tip_line" for s in scenarios], device=device)
    home = torch.tensor([HOME], device=device)
    capacity = measure_joint_speed_limits(model, kin, home.repeat(4, 1))
    speed_ratio = required_speed_ratio(model, kin, cfg, scenarios, q_ref, tip_ref, capacity, gains.ik_lambda)

    lo, hi = gains.output_limits
    pid = BatchedPID(
        count,
        3,
        torch.tensor(gains.kp, device=device),
        torch.tensor(gains.ki, device=device),
        torch.tensor(gains.kd, device=device),
        deriv_filter_tau=torch.tensor(gains.deriv_filter_tau, device=device),
        min_output=lo,
        max_output=hi,
        device=device,
    )
    trace = {key: [] for key in ("q", "u", "q_target", "tip", "tip_target")}
    for k in range(q_ref.shape[0]):
        q = plant.q[:, :3]
        tip_target = torch.where(is_tip[:, None], tip_ref[k], home_pose)
        ik_target = q + dls_step(kin, plant.q, tip_target, gains.ik_lambda)
        target = torch.where(is_tip[:, None], ik_target, q_ref[k])
        u = robot_joint_command(pid, target, q, DT)
        plant.invalid |= ~torch.isfinite(u).all(dim=1)
        plant.u_cmd.copy_(torch.nan_to_num(u))
        plant.step()
        trace["q"].append(plant.q[:, :3].clone())
        trace["u"].append(u)
        trace["q_target"].append(target)
        trace["tip"].append(kin.pose_jacobian(plant.q)[0])
        trace["tip_target"].append(tip_target)
    traces = {key: torch.stack(value).cpu() for key, value in trace.items()}
    traces["invalid"] = plant.invalid.cpu()
    traces["speed_ratio"] = speed_ratio.cpu()
    traces["capacity_rad_s"] = capacity.cpu()
    traces["at_limit"] = (
        (traces["q"] <= kin.limits[:3, 0].cpu() + 1e-6) | (traces["q"] >= kin.limits[:3, 1].cpu() - 1e-6)
    ).any(dim=(0, 2))
    return scenarios, traces


def _settle_time(err: torch.Tensor, band: float, t: torch.Tensor) -> float:
    """Time after the move starts at which |err| last leaves the band [s]; NaN if it never settles."""
    outside = (err.abs() > band).nonzero()
    if len(outside) == 0:
        return 0.0
    last = int(outside[-1])
    return math.nan if last == len(err) - 1 else float(t[last + 1])


def metrics(cfg: ReplayConfig, scenarios: list[Scenario], traces: dict[str, torch.Tensor]) -> list[dict]:
    """One row per scenario. Angles in deg, tip errors in mm, times in s after the move starts."""
    steps = traces["q"].shape[0]
    t = torch.arange(1, steps + 1) * DT - cfg.lead_s
    moving = t >= 0
    tail = t >= t[-1] - cfg.tail_s
    rows = []
    for n, s in enumerate(scenarios):
        u = traces["u"][:, n]
        du = (u[1:] - u[:-1]).abs().sum(1)
        row = {
            "scenario": s.name,
            "family": s.family,
            "joint": s.joint,
            "size": s.size,
            "plant": s.plant,
            "speed_ratio": float(traces["speed_ratio"][n]),
            "feasible": bool(traces["speed_ratio"][n] <= cfg.feasible_speed_ratio)
            or s.family == "joint_step",
            "saturated_frac": float((u.abs() >= 0.999).any(1)[moving].float().mean()),
            "valve_travel_per_s": float(du[moving[1:]].sum() / (moving.sum() * DT)),
            "tail_valve_travel_per_s": float(du[tail[1:]].sum() / (tail.sum() * DT)),
            "hit_limit": bool(traces["at_limit"][n]),
            "invalid": bool(traces["invalid"][n]),
        }
        if s.family == "tip_line":
            axis = 0 if s.joint[1] == "X" else 1
            sign = 1.0 if s.joint[0] == "+" else -1.0
            err = (traces["tip"][:, n, :2] - traces["tip_target"][:, n, :2]) * 1000  # mm
            norm = err.norm(dim=1)
            ref_speed = (traces["tip_target"][1:, n, axis] - traces["tip_target"][:-1, n, axis]).abs() / DT
            cruise = torch.cat((torch.tensor([False]), ref_speed >= 0.99 * s.size))
            done = moving & (t >= (cfg.tip_travel_m / s.size + s.size / cfg.tip_accel_m_s2))
            row.update(
                mean_err_mm=float(norm[moving].mean()),
                max_err_mm=float(norm[moving].max()),
                cruise_lag_mm=float((-sign * err[cruise, axis]).mean()) if cruise.any() else math.nan,
                end_overshoot_mm=float((sign * err[done, axis]).clamp(min=0).max())
                if done.any()
                else math.nan,
                settle_s=_settle_time(norm[moving], cfg.settle_band_mm, t[moving]),
                final_err_mm=float(norm[tail].mean()),
                max_pitch_err_deg=float(
                    torch.rad2deg(wrap_angle(traces["tip"][:, n, 2] - traces["tip_target"][:, n, 2]))[moving]
                    .abs()
                    .max()
                ),
            )
        else:
            j = PID_JOINTS.index(s.joint)
            q = torch.rad2deg(traces["q"][:, n])
            target = torch.rad2deg(traces["q_target"][:, n])
            err = target[:, j] - q[:, j]
            travel = s.size if s.family == "joint_step" else math.copysign(cfg.ramp_travel_deg, s.size)
            start = q[~moving, j][-1] if (~moving).any() else q[0, j]
            progress = (q[:, j] - start) / travel
            others = [i for i in range(3) if i != j]
            row.update(
                overshoot_pct=float((progress[moving].max() - 1).clamp(min=0) * 100),
                settle_s=_settle_time(err[moving], max(cfg.settle_band_deg, 0.02 * abs(travel)), t[moving]),
                final_err_deg=float(err[tail].abs().mean()),
                tail_p2p_deg=float(q[tail, j].max() - q[tail, j].min()),
                coupling_deg=float((q[:, others] - q[:1, others]).abs().max()),
            )
            if s.family == "joint_step":
                reached = lambda frac: (progress[moving] >= frac).nonzero()  # noqa: E731
                r10, r90 = reached(0.1), reached(0.9)
                row["rise_s"] = float((r90[0] - r10[0]) * DT) if len(r10) and len(r90) else math.nan
            else:
                ramping = moving & (t <= cfg.ramp_travel_deg / abs(s.size))
                row["ramp_lag_deg"] = float((math.copysign(1, s.size) * err[ramping]).mean())
        rows.append(row)
    return rows


def write_report(
    out_dir: Path, gains: PidGains, cfg: ReplayConfig, model: Path, rows: list[dict], traces
) -> None:
    """Write metrics.csv, summary.json, traces.pt and replay.png into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "gains": asdict(gains),
        "config": asdict(cfg),
        "model": str(model),
        "model_sha256": sha256(Path(model) / "mlp_state_dict.pt"),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(traces, out_dir / "traces.pt")


def plot_report(out_path: Path, cfg: ReplayConfig, scenarios: list[Scenario], traces) -> None:
    """Largest positive step and fastest positive ramp per joint, plus the fastest +X/+Z tip lines, all plants."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = (torch.arange(1, traces["q"].shape[0] + 1) * DT - cfg.lead_s).numpy()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    colors = dict(zip(cfg.plants, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    def pick(family, joint, size):
        return {
            s.plant: n
            for n, s in enumerate(scenarios)
            if (s.family, s.joint, s.size) == (family, joint, size)
        }

    big, fast = max(cfg.step_deg), max(cfg.ramp_deg_s)
    for row, joint in enumerate(PID_JOINTS):
        j = PID_JOINTS.index(joint)
        for col, (family, size, title) in enumerate(
            (
                ("joint_step", big, f"{joint}: +{big:g} deg step"),
                ("joint_ramp", fast, f"{joint}: +{fast:g} deg/s ramp"),
            )
        ):
            ax = axes[row, col]
            for plant, n in pick(family, joint, size).items():
                ax.plot(t, torch.rad2deg(traces["q"][:, n, j]), color=colors[plant], lw=1.2, label=plant)
                if plant == "nominal":
                    ax.plot(t, torch.rad2deg(traces["q_target"][:, n, j]), "k--", lw=1, label="target")
            ax.set(title=title, xlabel="s", ylabel="deg")
            ax.grid(alpha=0.3)
        ax = axes[row, 2]
        for plant, n in pick("joint_step", joint, big).items():
            ax.plot(t, traces["u"][:, n, j], color=colors[plant], lw=1)
        ax.set(title=f"{joint}: valve during the step", xlabel="s", ylabel="u", ylim=(-1.05, 1.05))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    v = max(cfg.tip_speeds_m_s)
    for col, direction in enumerate(("+X", "+Z")):
        ax = axes[3, col]
        axis = 0 if direction[1] == "X" else 1
        for plant, n in pick("tip_line", direction, v).items():
            err = (traces["tip"][:, n, axis] - traces["tip_target"][:, n, axis]) * 1000
            ax.plot(t, err, color=colors[plant], lw=1.2, label=plant)
        ax.set(
            title=f"tip line {direction} at {v * 1000:g} mm/s: error along the line", xlabel="s", ylabel="mm"
        )
        ax.grid(alpha=0.3)
    ax = axes[3, 2]
    for direction, style in (("+X", "-"), ("+Z", ":")):
        n = pick("tip_line", direction, v)["nominal"]
        for j, joint in enumerate(PID_JOINTS):
            ax.plot(t, traces["u"][:, n, j], style, lw=1, label=f"{direction} {joint}")
    ax.set(title=f"nominal tip-line valves at {v * 1000:g} mm/s", xlabel="s", ylabel="u", ylim=(-1.05, 1.05))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
