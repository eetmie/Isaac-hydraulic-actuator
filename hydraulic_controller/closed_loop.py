"""One closed loop for every controller: the plant at 100 Hz, measured joint state in, valve commands out.

A controller is any object with

* ``reset(task: TaskBatch)``: called once with the references of every environment, and
* ``__call__(k, q, v) -> u``: tick ``k``, measured angles ``q`` [N, 4] rad (with the configured sensor noise),
  measured rates ``v`` [N, 4] rad/s; returns valve commands ``u`` [N, 3] in [-1, 1].

The controller sees no plant internals. It keeps whatever history it needs, as it must on the robot. ``rollout``
can tile the scenarios ``copies`` times so that one batch scores many candidates (the PID tuner's population):
row ``c * S + s`` is copy ``c`` of scenario ``s``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

from .benchmark import PLANTS
from .core import HydraulicPlant, sha256, wrap_angle
from .kinematics import ExcavatorKinematics
from .tasks import DT, PID_JOINTS, Prepared, Scenario, TaskConfig, prepare

COST_TERMS = ("track", "final", "overshoot", "tail_p2p", "valve_travel", "invalid")
REPORT_TERMS = ("tip_mean_mm", "tip_max_mm", "tip_final_mm", "pitch_mean_deg")


@dataclass
class TaskBatch:
    """What a controller may know in advance: start state and the full references, per environment."""

    kin: ExcavatorKinematics
    q0: torch.Tensor  # [N, 4] rad
    q_ref: torch.Tensor  # [T, N, 3] rad; NaN for tip lines
    tip_ref: torch.Tensor  # [T, N, 3] (m, m, rad); NaN for joint scenarios
    is_tip: torch.Tensor  # [N]
    dt: float = DT

    @property
    def count(self) -> int:
        return len(self.q0)

    def tip_velocity(self) -> torch.Tensor:
        """Reference tip twist [T, N, 3] by backward difference (zero before the first tick)."""
        diff = torch.diff(self.tip_ref, dim=0, prepend=self.tip_ref[:1]) / self.dt
        diff[..., 2] = 0.0  # the bucket angle is held
        return torch.nan_to_num(diff)


class Controller(Protocol):
    def reset(self, task: TaskBatch) -> None: ...

    def __call__(self, k: int, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


@torch.no_grad()
def rollout(
    prep: Prepared,
    controller: Controller,
    copies: int = 1,
    record: bool = False,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Run every scenario (``copies`` times) as one batch; returns per-(copy, scenario) terms [C, S].

    Errors are in degree equivalents: joint angles in deg (all three joints), tip position in mm / 10 (1 deg at
    the ~0.6 m reach is ~10 mm) plus half the bucket-angle error in deg.

    * ``track``: mean error once the move starts; ``final``: mean error over the last ``tail_s``;
    * ``overshoot``: largest travel past the end of the move (moved joint, or tip along a line; 0 for circles);
    * ``tail_p2p``: summed peak-to-peak joint motion over the last ``tail_s`` (hunting, limit cycles);
    * ``valve_travel``: summed valve travel per second (chatter);
    * ``invalid``: 1 when the plant went non-finite or onto a joint limit.

    ``REPORT_TERMS`` add plain tip errors in mm and the bucket angle error in deg (tip lines; NaN otherwise).
    With ``record`` (``copies == 1``) it also returns per-tick traces on the CPU.
    """
    cfg, kin, device = prep.cfg, prep.kin, prep.q0.device
    c, s = copies, len(prep.scenarios)
    count = c * s
    if record and c != 1:
        raise ValueError("traces are recorded for a single copy only")

    def tile(x):
        return x.repeat(c, *([1] * (x.dim() - 1)))

    def tile_time(x):  # [T, S, ...] -> [T, C*S, ...]
        return x.repeat(1, c, *([1] * (x.dim() - 2)))

    plant = HydraulicPlant(prep.model, kin, count, str(device))
    for n, sc in enumerate(prep.scenarios):
        p = PLANTS[sc.plant]
        rows = torch.arange(n, count, s, device=device)
        plant.valve_gain[rows] = p["gain"]
        plant.valve_offset[rows] = p["offset"]
        plant.action_delay[rows] = p["delay"]
        plant.speed_scale[rows, :3] = p["speed"]
    q0 = tile(prep.q0)
    plant.reset(torch.arange(count, device=device), q0)

    is_tip = tile(torch.tensor([sc.is_tip for sc in prep.scenarios], device=device))
    is_line = tile(torch.tensor([sc.family == "tip_line" for sc in prep.scenarios], device=device))
    q_ref, tip_ref = tile_time(prep.q_ref), tile_time(prep.tip_ref)
    controller.reset(TaskBatch(kin, q0, q_ref, tip_ref, is_tip))

    moved = tile(
        torch.tensor(
            [PID_JOINTS.index(sc.joint) if not sc.is_tip else 0 for sc in prep.scenarios],
            device=device,
        )
    )
    line_axis = tile(
        torch.tensor([0 if sc.joint.endswith("X") else 1 for sc in prep.scenarios], device=device)
    )
    rows = torch.arange(count, device=device)
    q_end, tip_end = q_ref[-1], tip_ref[-1]
    move_sign = torch.where(
        is_tip,
        torch.sign(tip_end[rows, line_axis] - tip_ref[0][rows, line_axis]),
        torch.sign(q_end[rows, moved] - q_ref[0][rows, moved]),
    )
    pose0 = kin.pose_jacobian(q0)[0]

    steps = prep.q_ref.shape[0]
    t = torch.arange(1, steps + 1, device=device) * DT - cfg.lead_s
    n_moving, n_tail = int((t >= 0).sum()), int((t >= t[-1] - cfg.tail_s).sum())
    terms = {key: torch.zeros(count, device=device) for key in COST_TERMS + REPORT_TERMS}
    tail_lo = torch.full((count, 3), math.inf, device=device)
    tail_hi = torch.full((count, 3), -math.inf, device=device)
    noise = torch.Generator(device=device).manual_seed(seed)
    noise_std = math.radians(cfg.sensor_noise_deg)
    u_prev = torch.zeros(count, 3, device=device)
    trace = {key: [] for key in ("q", "u", "q_target", "tip", "tip_target")} if record else None
    for k in range(steps):
        q_seen = plant.q.clone()
        if noise_std > 0:
            # Common random numbers: every copy sees the same noise on the same scenario.
            q_seen[:, :3] += tile(torch.randn(s, 3, generator=noise, device=device)) * noise_std
        u = controller(k, q_seen, plant.v.clone())
        plant.invalid |= ~torch.isfinite(u).all(dim=1)
        u = torch.nan_to_num(u).clamp(-1.0, 1.0)
        plant.u_cmd.copy_(u)
        plant.step()

        q = plant.q[:, :3]
        tip = kin.pose_jacobian(plant.q)[0]
        tip_target = torch.where(is_tip[:, None], tip_ref[k], pose0)
        joint_err = torch.rad2deg((q_ref[k] - q).abs()).sum(1)
        tip_mm = (tip[:, :2] - tip_target[:, :2]).norm(dim=1) * 1000
        pitch_deg = torch.rad2deg(wrap_angle(tip[:, 2] - tip_target[:, 2]).abs())
        err = torch.where(is_tip, tip_mm / 10 + 0.5 * pitch_deg, torch.nan_to_num(joint_err))
        if t[k] >= 0:
            terms["track"] += err / n_moving
            terms["tip_mean_mm"] += tip_mm / n_moving
            terms["pitch_mean_deg"] += pitch_deg / n_moving
            terms["tip_max_mm"] = torch.maximum(terms["tip_max_mm"], tip_mm)
            past = torch.where(
                is_tip,
                (tip[rows, line_axis] - tip_end[rows, line_axis]) * move_sign * 100,
                torch.rad2deg((q[rows, moved] - q_end[rows, moved]) * move_sign),
            )
            past = torch.where(
                is_tip & ~is_line, 0.0, past
            )  # a circle ends where it started: nothing to pass
            terms["overshoot"] = torch.maximum(terms["overshoot"], torch.nan_to_num(past).clamp_min(0))
        if t[k] >= t[-1] - cfg.tail_s:
            terms["final"] += err / n_tail
            terms["tip_final_mm"] += tip_mm / n_tail
            tail_lo, tail_hi = torch.minimum(tail_lo, q), torch.maximum(tail_hi, q)
        terms["valve_travel"] += (u - u_prev).abs().sum(1) / cfg.duration_s
        u_prev = u
        if record:
            trace["q"].append(q.clone())
            trace["u"].append(u)
            trace["q_target"].append(torch.where(is_tip[:, None], math.nan, q_ref[k]))
            trace["tip"].append(tip)
            trace["tip_target"].append(tip_target)

    at_limit = (
        (plant.q[:, :3] <= kin.limits[:3, 0] + 1e-6) | (plant.q[:, :3] >= kin.limits[:3, 1] - 1e-6)
    ).any(1)
    terms["tail_p2p"] = torch.rad2deg(tail_hi - tail_lo).sum(1)
    terms["invalid"] = (plant.invalid | at_limit).float()
    for key in REPORT_TERMS:
        terms[key] = torch.where(is_tip, terms[key], math.nan)
    terms = {key: value.reshape(c, s) for key, value in terms.items()}
    if not record:
        return terms, None
    traces = {key: torch.stack(value).cpu() for key, value in trace.items()}
    traces["invalid"] = plant.invalid.cpu()
    traces["speed_ratio"] = prep.speed_ratio.cpu()
    traces["path_valid"] = prep.path_valid.cpu()
    traces["capacity_rad_s"] = prep.capacity.cpu()
    traces["at_limit"] = (
        (traces["q"] <= kin.limits[:3, 0].cpu() + 1e-6) | (traces["q"] >= kin.limits[:3, 1].cpu() - 1e-6)
    ).any(dim=(0, 2))
    return terms, traces


@torch.no_grad()
def run_single(
    controller: Controller,
    cfg: TaskConfig,
    model: Path,
    asset: Path,
    device: str = "cuda:0",
    lam: float = 0.001,
) -> tuple[list[Scenario], dict[str, torch.Tensor]]:
    """Prepare and roll out every scenario once with one controller; returns per-tick traces on the CPU."""
    prep = prepare(cfg, model, asset, device, lam)
    _, traces = rollout(prep, controller, record=True)
    return prep.scenarios, traces


def _settle_time(err: torch.Tensor, band: float, t: torch.Tensor) -> float:
    """Time after the move starts at which |err| last leaves the band [s]; NaN if it never settles."""
    outside = (err.abs() > band).nonzero()
    if len(outside) == 0:
        return 0.0
    last = int(outside[-1])
    return math.nan if last == len(err) - 1 else float(t[last + 1])


def metrics(cfg: TaskConfig, scenarios: list[Scenario], traces: dict[str, torch.Tensor]) -> list[dict]:
    """One row per joint scenario or tip line (circles are scored by ``comparison``). Angles in deg, tip errors in
    mm, times in s after the move starts."""
    steps = traces["q"].shape[0]
    t = torch.arange(1, steps + 1) * DT - cfg.lead_s
    moving = t >= 0
    tail = t >= t[-1] - cfg.tail_s
    rows = []
    for n, s in enumerate(scenarios):
        if s.family == "tip_circle":
            continue
        u = traces["u"][:, n]
        du = (u[1:] - u[:-1]).abs().sum(1)
        row = {
            "scenario": s.name,
            "family": s.family,
            "joint": s.joint,
            "size": s.size,
            "plant": s.plant,
            "speed_ratio": float(traces["speed_ratio"][n]),
            "start": s.start,
            "feasible": bool(traces["path_valid"][n])
            and (s.family == "joint_step" or bool(traces["speed_ratio"][n] <= cfg.feasible_speed_ratio)),
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
    out_dir: Path, controller: dict, cfg: TaskConfig, model: Path, rows: list[dict], traces
) -> None:
    """Write metrics.csv, summary.json, traces.pt and replay.png into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "controller": controller,
        "config": asdict(cfg),
        "model": str(model),
        "model_sha256": sha256(Path(model) / "mlp_state_dict.pt"),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(traces, out_dir / "traces.pt")


def plot_report(out_path: Path, cfg: TaskConfig, scenarios: list[Scenario], traces) -> None:
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
            if (s.family, s.joint, s.size, s.start) == (family, joint, size, 0)
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
