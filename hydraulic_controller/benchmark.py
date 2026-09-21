"""Deterministic free-space benchmark for a trained hydraulic controller.

Two families of tests, each under nominal and perturbed valves:

* **Held commands** — ramp to a constant tip velocity, reverse, then stop. Reports the steady-state speed ratio,
  velocity error, cross-track motion, stopping residual and joint drift while holding.
* **Trajectories** — circles and straight lines with quintic time scaling and a proportional position loop,
  as in Egli & Hutter (RA-L 2022, Table V): mean/max position error, orientation error and eta = e_max / v_max.

Valve statistics (per-policy-step change, saturation) are reported for every test, because a policy that chatters
drives the actuator model far outside its training data and its simulated accuracy is then meaningless.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .core import HydraulicPlant, measure_joint_speed_limits, sha256, wrap_angle
from .kinematics import ExcavatorKinematics
from .policy import ControllerPolicy
from .trajectory import quintic

PLANTS = {
    "nominal": {"gain": 1.0, "offset": 0.0, "delay": 0, "speed": 1.0},
    "slow_valves": {"gain": 0.9, "offset": -0.02, "delay": 2, "speed": 1.0},
    "fast_valves": {"gain": 1.1, "offset": 0.02, "delay": 0, "speed": 1.0},
    # Beyond valve perturbations: the measured model-vs-machine speed spread (p10/p90 about 0.8/1.3).
    "machine_20pct_slower": {"gain": 1.0, "offset": 0.0, "delay": 1, "speed": 0.8},
    "machine_30pct_faster": {"gain": 1.0, "offset": 0.0, "delay": 1, "speed": 1.3},
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Speeds [mm/s], durations [s] and geometry [m] of the benchmark."""

    held_speeds_mm_s: tuple[float, ...] = (10, 20, 40, 60, 80, 100, 120)
    leg_s: float = 2.2
    stop_s: float = 2.6
    trajectory_speeds_mm_s: tuple[float, ...] = (10, 20, 40, 60)
    circle_radius: float = 0.05
    line_length: float = 0.15
    kp: float = 3.0
    kp_angle: float = 3.0
    linear_accel: float = 0.25
    poses: int = 8
    plants: tuple[str, ...] = tuple(PLANTS)


def benchmark_poses(kin: ExcavatorKinematics, count: int, clearance: float = 0.09) -> torch.Tensor:
    """Deterministic valid poses whose surroundings (+/- clearance in X and Z) are reachable at fixed bucket angle."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    lo, hi = kin.limits[:3, 0].cpu() + 0.25, kin.limits[:3, 1].cpu() - 0.25
    chosen = []
    offsets = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]) * clearance
    for _ in range(4000):
        q = torch.zeros(1, 4)
        q[0, :3] = lo + (hi - lo) * torch.rand(3, generator=generator)
        q = q.to(kin.device)
        if not bool(kin.valid(q, 0.1)[0]):
            continue
        pose, _ = kin.pose_jacobian(q)
        targets = pose.repeat(len(offsets), 1)
        targets[:, :2] += offsets.to(kin.device)
        _, reached = kin.inverse(targets, q)
        if bool(reached.all()):
            chosen.append(q[0])
        if len(chosen) == count:
            return torch.stack(chosen)
    raise RuntimeError(f"Found only {len(chosen)} benchmark poses")


class _Rollout:
    """Closed-loop simulation of one batch with the deployment loop: governor, policy, zero-order hold."""

    def __init__(self, policy: ControllerPolicy, kin, starts: torch.Tensor, plant_name: str):
        self.policy, self.kin = policy, kin
        self.settings = policy.settings
        self.count = len(starts)
        self.plant = HydraulicPlant(policy.model_path, kin, self.count, str(starts.device), self.settings)
        self.plant.reset(torch.arange(self.count, device=starts.device), starts)
        perturbation = PLANTS[plant_name]
        self.plant.valve_gain.fill_(perturbation["gain"])
        self.plant.valve_offset.fill_(perturbation["offset"])
        self.plant.action_delay.fill_(perturbation["delay"])
        self.plant.speed_scale[:, :3] = perturbation["speed"]
        self.governor = policy.governor(kin)
        self.command = torch.zeros(self.count, 3, device=starts.device)
        self.intervention = torch.zeros(self.count, device=starts.device)

    def run(self, seconds: float, requested_fn):
        """Advance; ``requested_fn(t, pose, twist)`` is evaluated at the policy rate. Returns 100 Hz logs."""
        logs = {key: [] for key in ("pose", "twist", "u", "command", "intervention", "q")}
        du, previous = [], self.plant.u_cmd.clone()
        limit = torch.zeros(self.count, dtype=torch.bool, device=self.plant.q.device)
        for step in range(int(round(seconds / self.settings.dt))):
            if step % self.settings.decimation == 0:
                pose, twist = self.plant.tip_state()
                requested = requested_fn(step * self.settings.dt, pose, twist)
                self.command, self.intervention = self.governor(self.plant.q, self.plant.v, requested)
                self.plant.begin_action(self.policy(self.plant.observe(self.command)))
                du.append((self.plant.u_cmd - previous).abs())
                previous = self.plant.u_cmd.clone()
            self.plant.step()
            limit |= self.plant.limit_hit
            pose, twist = self.plant.tip_state()
            for key, value in (
                ("pose", pose),
                ("twist", twist),
                ("u", self.plant.u_cmd.clone()),
                ("command", self.command),
                ("intervention", self.intervention),
                ("q", self.plant.q.clone()),
            ):
                logs[key].append(value)
        logs = {key: torch.stack(value) for key, value in logs.items()}
        logs["du"] = torch.stack(du)
        failed = self.plant.invalid | limit | self.kin.colliding(self.plant.q)
        logs["failed"] = failed
        return logs


def _valve_stats(logs) -> dict[str, torch.Tensor]:
    """Per-environment valve-command statistics: change per policy step and saturation."""
    du = logs["du"].permute(1, 0, 2).reshape(logs["du"].shape[1], -1)
    return {
        "du_p50": du.median(dim=1).values,
        "du_p99": torch.quantile(du, 0.99, dim=1),
        "du_gt_0p1_fraction": (du > 0.1).float().mean(1),
        "saturation_fraction": (logs["u"].abs() > 0.98).float().mean((0, 2)),
    }


def feasible_twist(kin, q: torch.Tensor, twist: torch.Tensor, limits: torch.Tensor, margin: float = 0.85):
    """Whether +/- the planar twist [m/s, m/s, rad/s] needs joint rates within ``margin`` of the speed limits."""
    _, jacobian = kin.pose_jacobian(q)
    rates = torch.linalg.solve(jacobian[:, :, :3], twist[:, :, None]).squeeze(-1).abs()
    slowest = limits.min(dim=1).values
    return (rates <= margin * slowest).all(dim=1)


def run_held(policy, kin, poses, plant_name: str, cfg: BenchmarkConfig, limits: torch.Tensor) -> list[dict]:
    """Ramp to +v, reverse to -v, then stop, along 8 directions from every pose."""
    device = poses.device
    angles = torch.arange(8, device=device) * math.pi / 4
    directions = torch.stack((angles.cos(), angles.sin()), dim=1)
    rows = []
    dt = policy.settings.dt
    for speed_mm_s in cfg.held_speeds_mm_s:
        speed = speed_mm_s / 1000
        starts = poses.repeat_interleave(len(directions), 0)
        dirs = directions.repeat(len(poses), 1)
        rollout = _Rollout(policy, kin, starts, plant_name)
        requested = torch.zeros(len(starts), 3, device=device)

        def reference(t, pose, twist):
            goal = speed if t < cfg.leg_s else (-speed if t < 2 * cfg.leg_s else 0.0)
            current = (requested[:, :2] * dirs).sum(1)
            step = cfg.linear_accel * policy.settings.policy_dt
            current = current + (goal - current).clamp(-step, step)
            requested[:, :2] = current[:, None] * dirs
            return requested.clone()

        logs = rollout.run(2 * cfg.leg_s + cfg.stop_s, reference)
        twist = logs["twist"][:, :, :2]
        along = (twist * dirs).sum(-1)
        cross = twist[..., 0] * -dirs[:, 1] + twist[..., 1] * dirs[:, 0]
        steady = [slice(int((cfg.leg_s - 1.0) / dt), int(cfg.leg_s / dt))]
        steady.append(slice(int((2 * cfg.leg_s - 1.0) / dt), int(2 * cfg.leg_s / dt)))
        gain = (along[steady[0]].mean(0) / speed - along[steady[1]].mean(0) / speed) / 2
        error = torch.cat([(twist[s] - logs["command"][s, :, :2]).norm(dim=-1) for s in steady])
        cross_mm_s = torch.cat([cross[s].abs() for s in steady]).mean(0) * 1000
        hold = slice(int((2 * cfg.leg_s + 1.0) / dt), None)
        residual = twist[hold].norm(dim=-1).square().mean(0).sqrt() * 1000
        drift = (logs["q"][-1, :, :3] - logs["q"][hold.start, :, :3]).abs().amax(1) * 180 / math.pi
        governed = logs["intervention"][: int(2 * cfg.leg_s / dt)].mean(0)
        valves = _valve_stats(logs)
        feasible = feasible_twist(
            kin, starts, torch.cat((dirs * speed, torch.zeros(len(starts), 1, device=device)), dim=1), limits
        )
        for i in range(len(starts)):
            rows.append(
                {
                    "plant": plant_name,
                    "speed_mm_s": speed_mm_s,
                    "pose": i // len(directions),
                    "feasible": bool(feasible[i]),
                    "direction_deg": float(angles[i % len(directions)] * 180 / math.pi),
                    "speed_ratio": float(gain[i]),
                    "velocity_rmse_mm_s": float(error[:, i].square().mean().sqrt() * 1000),
                    "cross_track_mm_s": float(cross_mm_s[i]),
                    "stop_residual_mm_s": float(residual[i]),
                    "hold_drift_deg": float(drift[i]),
                    "governor_fraction": float(governed[i]),
                    "failed": bool(logs["failed"][i]),
                    **{key: float(value[i]) for key, value in valves.items()},
                }
            )
    return rows


def run_trajectories(policy, kin, poses, plant_name: str, cfg: BenchmarkConfig, keep_trace: bool = False):
    """Circles (both senses) and four straight lines per pose, with quintic time scaling and a P position loop."""
    device = poses.device
    count = len(poses)
    shapes = ["circle_ccw", "circle_cw", "line_+x", "line_-x", "line_+z", "line_-z"]
    rows, traces = [], {}
    for speed_mm_s in cfg.trajectory_speeds_mm_s:
        speed = speed_mm_s / 1000
        starts = poses.repeat(len(shapes), 1)
        rollout = _Rollout(policy, kin, starts, plant_name)
        pose0, _ = kin.pose_jacobian(starts)
        shape_id = torch.arange(len(shapes), device=device).repeat_interleave(count)
        length = torch.where(shape_id < 2, 2 * math.pi * cfg.circle_radius, cfg.line_length)
        duration = length / speed
        total = float(duration.max()) + 1.0
        sense = torch.where(shape_id == 1, -1.0, 1.0)
        line_dir = torch.zeros(len(starts), 2, device=device)
        line_dir[shape_id == 2, 0], line_dir[shape_id == 3, 0] = 1.0, -1.0
        line_dir[shape_id == 4, 1], line_dir[shape_id == 5, 1] = 1.0, -1.0
        center = pose0[:, :2].clone()
        center[:, 0] -= cfg.circle_radius

        def path(t: float):
            tau = (t / duration).clamp(0, 1)
            s, s_dot = quintic(tau)
            s_dot = s_dot / duration
            distance, rate = s * length, s_dot * length
            theta = sense * distance / cfg.circle_radius
            circle_p = center + cfg.circle_radius * torch.stack((theta.cos(), theta.sin()), dim=1)
            circle_v = (rate * sense)[:, None] * torch.stack((-theta.sin(), theta.cos()), dim=1)
            line_p = pose0[:, :2] + distance[:, None] * line_dir
            line_v = rate[:, None] * line_dir
            is_circle = (shape_id < 2)[:, None]
            return torch.where(is_circle, circle_p, line_p), torch.where(is_circle, circle_v, line_v)

        def reference(t, pose, twist):
            p_ref, v_ref = path(t)
            requested = torch.zeros(len(starts), 3, device=device)
            requested[:, :2] = v_ref + cfg.kp * (p_ref - pose[:, :2])
            requested[:, 2] = cfg.kp_angle * wrap_angle(pose0[:, 2] - pose[:, 2])
            return requested

        logs = rollout.run(total, reference)
        steps = logs["pose"].shape[0]
        times = torch.arange(steps, device=device) * policy.settings.dt
        reference_p = torch.stack([path(float(t))[0] for t in times])
        position_error = (logs["pose"][:, :, :2] - reference_p).norm(dim=-1)
        # Distance to the path geometry, independent of timing: a slowed tip that stays on the path is not charged.
        actual = logs["pose"][:, :, :2]
        radial = ((actual - center).norm(dim=-1) - cfg.circle_radius).abs()
        along = ((actual - pose0[:, :2]) * line_dir).sum(-1).clamp(0, cfg.line_length)
        lateral = (actual - (pose0[:, :2] + along[..., None] * line_dir)).norm(dim=-1)
        path_deviation = torch.where(shape_id < 2, radial, lateral)
        orientation_error = wrap_angle(logs["pose"][:, :, 2] - pose0[:, 2]).abs()
        active = times[:, None] <= duration[None, :]
        valves = _valve_stats(logs)
        for i in range(len(starts)):
            mask = active[:, i]
            e = position_error[mask, i]
            v_max = float(logs["twist"][mask, i, :2].norm(dim=-1).max())
            rows.append(
                {
                    "plant": plant_name,
                    "shape": shapes[int(shape_id[i])],
                    "speed_mm_s": speed_mm_s,
                    "pose": i % count,
                    "ep_avg_mm": float(e.mean() * 1000),
                    "ep_max_mm": float(e.max() * 1000),
                    "path_dev_avg_mm": float(path_deviation[mask, i].mean() * 1000),
                    "path_dev_max_mm": float(path_deviation[mask, i].max() * 1000),
                    "v_max_mm_s": v_max * 1000,
                    "eo_avg_deg": float(orientation_error[mask, i].mean() * 180 / math.pi),
                    "eo_max_deg": float(orientation_error[mask, i].max() * 180 / math.pi),
                    "eta_s": float(e.max()) / max(v_max, 1e-6),
                    "final_error_mm": float(position_error[-1, i] * 1000),
                    "governor_fraction": float(logs["intervention"][mask, i].mean()),
                    "failed": bool(logs["failed"][i]),
                    **{key: float(value[i]) for key, value in valves.items()},
                }
            )
        if keep_trace:
            traces[speed_mm_s] = {
                "shape_id": shape_id.cpu(),
                "actual": logs["pose"][:, :, :2].cpu(),
                "reference": reference_p.cpu(),
                "u": logs["u"].cpu(),
            }
    return rows, traces


def _median(rows, key):
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return statistics.median(values) if values else float("nan")


def summarize(held: list[dict], trajectories: list[dict]) -> dict:
    """Median metrics per plant and speed, ignoring heavily governed or failed conditions."""
    summary = {"held": [], "trajectory": []}
    for plant in sorted({row["plant"] for row in held}):
        for speed in sorted({row["speed_mm_s"] for row in held}):
            rows = [r for r in held if r["plant"] == plant and r["speed_mm_s"] == speed]
            usable = [
                r for r in rows if not r["failed"] and r["governor_fraction"] < 0.1 and r["feasible"]
            ] or rows
            summary["held"].append(
                {"plant": plant, "speed_mm_s": speed, "conditions": len(rows), "usable": len(usable)}
                | {
                    key: _median(usable, key)
                    for key in (
                        "speed_ratio",
                        "velocity_rmse_mm_s",
                        "cross_track_mm_s",
                        "stop_residual_mm_s",
                        "hold_drift_deg",
                        "du_p50",
                        "du_p99",
                        "saturation_fraction",
                    )
                }
                | {"failures": sum(r["failed"] for r in rows)}
            )
    for plant in sorted({row["plant"] for row in trajectories}):
        for speed in sorted({row["speed_mm_s"] for row in trajectories}):
            rows = [r for r in trajectories if r["plant"] == plant and r["speed_mm_s"] == speed]
            usable = [r for r in rows if not r["failed"] and r["governor_fraction"] < 0.1] or rows
            summary["trajectory"].append(
                {"plant": plant, "speed_mm_s": speed, "conditions": len(rows), "usable": len(usable)}
                | {
                    key: _median(usable, key)
                    for key in (
                        "ep_avg_mm",
                        "ep_max_mm",
                        "path_dev_avg_mm",
                        "path_dev_max_mm",
                        "v_max_mm_s",
                        "eo_avg_deg",
                        "eo_max_deg",
                        "eta_s",
                    )
                }
                | {
                    "ep_max_p90_mm": sorted(r["ep_max_mm"] for r in usable)[int(0.9 * (len(usable) - 1))],
                    "failures": sum(r["failed"] for r in rows),
                }
            )
    return summary


def recommend_speed(summary: dict, max_error_mm: float = 10.0) -> float:
    """Fastest trajectory speed whose nominal and perturbed median max error stays within ``max_error_mm``."""
    speeds = sorted({row["speed_mm_s"] for row in summary["trajectory"]})
    good = [
        speed
        for speed in speeds
        if all(
            row["ep_max_mm"] <= max_error_mm and row["failures"] == 0
            for row in summary["trajectory"]
            if row["speed_mm_s"] == speed
        )
    ]
    return max(good) if good else min(speeds)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summary: dict, traces: dict, trajectories: list[dict], circle_radius: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
    panels = (
        (axes[0, 0], "held", "speed_ratio", "Steady speed ratio (1 = perfect)"),
        (axes[0, 1], "held", "velocity_rmse_mm_s", "Steady velocity RMSE [mm/s]"),
        (axes[0, 2], "held", "du_p99", "Valve change per policy step, p99"),
        (axes[1, 0], "trajectory", "ep_avg_mm", "Trajectory mean position error [mm]"),
        (axes[1, 1], "trajectory", "ep_max_mm", "Trajectory max position error [mm]"),
    )
    for axis, family, key, label in panels:
        for plant in sorted({row["plant"] for row in summary[family]}):
            rows = [row for row in summary[family] if row["plant"] == plant]
            axis.plot([r["speed_mm_s"] for r in rows], [r[key] for r in rows], marker="o", label=plant)
        axis.set(xlabel="Speed [mm/s]", ylabel=label)
        axis.grid(alpha=0.25)
    axes[0, 0].axhline(1.0, color="black", linewidth=0.8)
    axes[0, 0].legend()

    # Valves behave differently per direction, so circles are always shown for both senses.
    senses = (("circle_ccw", "counterclockwise", "tab:blue"), ("circle_cw", "clockwise", "tab:orange"))
    axis = axes[0, 3]
    nominal = [row for row in trajectories if row["plant"] == "nominal" and not row["failed"]]
    for shape, label, color in senses:
        speeds = sorted({row["speed_mm_s"] for row in nominal})
        values = [
            [row["path_dev_max_mm"] for row in nominal if row["shape"] == shape and row["speed_mm_s"] == s]
            for s in speeds
        ]
        axis.plot(
            speeds, [statistics.median(v) for v in values], marker="o", color=color, label=f"{label}, median"
        )
        axis.plot(
            speeds,
            [max(v) for v in values],
            marker="x",
            linestyle="--",
            color=color,
            label=f"{label}, worst pose",
        )
    axis.set(
        xlabel="Speed [mm/s]", ylabel="Circle radius error, max [mm]", title="Nominal circles by direction"
    )
    axis.legend()
    axis.grid(alpha=0.25)
    if traces:
        speed = sorted(traces)[len(traces) // 2]
        trace = traces[speed]
        theta = torch.linspace(0, 2 * math.pi, 200)
        radius_mm = 1000 * circle_radius
        for axis, (shape, label, _) in zip(axes[1, 2:], senses):
            shape_index = 0 if shape == "circle_ccw" else 1
            for index in torch.nonzero(trace["shape_id"] == shape_index).flatten().tolist():
                center = trace["reference"][0, index] - torch.tensor([circle_radius, 0.0])
                axis.plot(*(1000 * (trace["actual"][:, index] - center)).T, linewidth=1.0)
            axis.plot(
                radius_mm * theta.cos(), radius_mm * theta.sin(), color="black", linestyle="--", linewidth=1
            )
            axis.set(
                title=f"Nominal {label} circles at {speed:g} mm/s, all poses",
                xlabel="X from center [mm]",
                ylabel="Z from center [mm]",
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.25)
    figure.suptitle("Hydraulic tip-velocity controller benchmark")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_benchmark(
    checkpoint: str | Path, output: str | Path, device: str = "cuda:0", cfg: BenchmarkConfig | None = None
) -> Path:
    """Run held and trajectory tests for every plant variant and write CSV, JSON and a figure."""
    cfg = cfg or BenchmarkConfig()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = ControllerPolicy(checkpoint, device)
    kin = ExcavatorKinematics(policy.asset_path, device)
    kin.build_collision_grid()
    poses = benchmark_poses(kin, cfg.poses)
    limits = measure_joint_speed_limits(policy.model_path, kin, poses, policy.settings)
    print(
        f"[benchmark] full-valve joint speeds [deg/s] (retract, extend): {(limits * 180 / math.pi).tolist()}"
    )
    held, trajectories, traces = [], [], {}
    for plant in cfg.plants:
        held += run_held(policy, kin, poses, plant, cfg, limits)
        rows, plant_traces = run_trajectories(policy, kin, poses, plant, cfg, keep_trace=plant == "nominal")
        trajectories += rows
        traces = plant_traces or traces
        print(f"[benchmark] finished plant '{plant}'")
    summary = summarize(held, trajectories)
    _write_csv(output / "held_results.csv", held)
    _write_csv(output / "trajectory_results.csv", trajectories)
    _plot(output / "benchmark.png", summary, traces, trajectories, cfg.circle_radius)
    result = output / "best_speed.json"
    result.write_text(
        json.dumps(
            {
                "checkpoint": str(Path(checkpoint).resolve()),
                "checkpoint_sha256": sha256(Path(checkpoint)),
                "recommended_speed_mm_s": recommend_speed(summary),
                "selection": "fastest trajectory speed with median max error <= 10 mm on every plant variant",
                "joint_speed_limits_rad_s": limits.tolist(),
                "held_usable_rule": "not failed, governor < 10 %, command within 85 % of full-valve joint speeds",
                "config": asdict(cfg),
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    return result
