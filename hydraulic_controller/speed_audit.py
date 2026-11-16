"""Motion-window metrics and speed-stratified summaries for EE line tracking."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

SPEEDS_MM_S = (5, 10, 20, 40, 70, 100, 120)


def line_duration(speeds, distance, acceleration, lead, tail):
    """Complete every trapezoidal/triangular reference plus start/end holds."""
    if not speeds or min(speeds) <= 0 or min(distance, acceleration) <= 0:
        raise ValueError("Speeds, distance and acceleration must be positive")
    durations = []
    for speed in speeds:
        peak = min(speed, math.sqrt(distance * acceleration))
        durations.append(distance / peak + peak / acceleration)
    return math.ceil(lead + max(durations) + tail)


def motion_metrics(trace, dt, commanded_speeds):
    """Per-scenario metrics restricted to reference motion or constant-speed cruise.

    Speeds [m/s], positions [m], angle [rad]. Errors score the physical pose.
    Reference motion above 0.1 mm/s excludes stationary holds; cruise is >=98%
    of the scenario's commanded speed. Empty windows are NaN, not zero error.
    """
    target, actual = trace["tip_target"], trace["tip"]
    ref_v = torch.diff(target[..., :2], dim=0, prepend=target[:1, :, :2]) / dt
    actual_v = torch.diff(actual[..., :2], dim=0, prepend=actual[:1, :, :2]) / dt
    ref_speed = ref_v.norm(dim=-1)
    position_error = (actual[..., :2] - target[..., :2]).norm(dim=-1) * 1000
    delta = actual[..., 2] - target[..., 2]
    angle_error = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs())
    velocity_error = (actual_v - ref_v).norm(dim=-1) * 1000
    along_ratio = (actual_v * ref_v).sum(-1) / ref_speed.square().clamp_min(1e-12)
    du = torch.diff(trace["u"], dim=0, prepend=trace["u"][:1]).abs().sum(-1) / dt
    result = {}
    for name, mask in (
        ("moving", ref_speed > 1e-4),
        ("cruise", (ref_speed > 1e-4) & (ref_speed >= 0.98 * commanded_speeds[None])),
    ):
        count = mask.sum(0)
        result[f"{name}_seconds"] = count * dt
        for metric, value in (
            ("tip_mean_mm", position_error),
            ("pitch_mean_deg", angle_error),
            ("velocity_error_mm_s", velocity_error),
            ("relative_velocity_error", velocity_error / (ref_speed * 1000).clamp_min(0.1)),
            ("along_speed_ratio", along_ratio),
            ("valve_travel_per_s", du),
        ):
            mean = torch.where(mask, value, 0.0).sum(0) / count.clamp_min(1)
            result[f"{name}_{metric}"] = torch.where(count > 0, mean, math.nan)
    return result


def write_motion_report(out, prep, terms, traces):
    """Write one row per controller/scenario; keep feasibility flags alongside errors."""
    rows = []
    speeds = torch.tensor([s.size for s in prep.scenarios])
    for method, trace in traces.items():
        metrics = motion_metrics(trace, prep.cfg.duration_s / len(trace["tip"]), speeds)
        for i, scenario in enumerate(prep.scenarios):
            rows.append(
                dict(
                    controller=method,
                    speed_mm_s=scenario.size * 1000,
                    plant=scenario.plant,
                    start=scenario.start,
                    direction=scenario.joint,
                    path_valid=int(prep.path_valid[i]),
                    feasible=int(prep.feasible[i]),
                    speed_ratio=float(prep.speed_ratio[i]),
                    failures=float(terms[method]["invalid"][i]),
                    **{k: float(v[i]) for k, v in metrics.items()},
                )
            )
    with (out / "speed_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_speed_report(benchmarks: Path, out: Path):
    """Aggregate seed means separately for all vs capacity-feasible cases at each speed."""
    grouped = defaultdict(list)
    for source in sorted(benchmarks.glob("*/summary.json")):
        data = json.loads(source.read_text())
        if data.get("task_config", {}).get("families") != ["tip_line"]:
            continue
        tag = source.parent.name.split("_", 2)[-1]
        condition, sep, seed = tag.rpartition("_seed")
        if not sep:
            condition, seed = tag, "42"
        audit = source.parent / "speed_audit.csv"
        csv_path = audit if audit.exists() else source.parent / "per_scenario.csv"
        if not csv_path.exists():
            continue
        window = "motion_only" if audit.exists() else "post_start_including_holds"
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        seed_groups = defaultdict(list)
        threshold = data["task_config"].get("feasible_speed_ratio", 0.9)
        for row in rows:
            feasible = bool(int(row["path_valid"])) and float(row["speed_ratio"]) <= threshold
            for subset in ("all", "capacity_feasible") if feasible else ("all",):
                key = (condition, int(seed), row["controller"], float(row["speed_mm_s"]), subset, window)
                seed_groups[key].append(row)
        for key, members in seed_groups.items():

            def finite_mean(field):
                values = [
                    float(r.get(field, math.nan))
                    for r in members
                    if math.isfinite(float(r.get(field, math.nan)))
                ]
                return statistics.mean(values) if values else math.nan

            prefix = "moving_" if audit.exists() else ""
            grouped[key[:1] + key[2:]].append(
                dict(
                    seed=key[1],
                    cases=len(members),
                    moving_seconds=finite_mean("moving_seconds"),
                    tip_mean_mm=finite_mean(prefix + "tip_mean_mm"),
                    pitch_mean_deg=finite_mean(prefix + "pitch_mean_deg"),
                    valve_travel_per_s=finite_mean(
                        prefix + "valve_travel_per_s" if audit.exists() else "valve_travel"
                    ),
                    failures=sum(float(r["failures" if audit.exists() else "invalid"]) for r in members),
                    velocity_error_mm_s=finite_mean("moving_velocity_error_mm_s")
                    if audit.exists()
                    else math.nan,
                    cruise_along_speed_ratio=finite_mean("cruise_along_speed_ratio")
                    if audit.exists()
                    else math.nan,
                    cruise_relative_velocity_error=finite_mean("cruise_relative_velocity_error")
                    if audit.exists()
                    else math.nan,
                )
            )
    result = []
    for (condition, method, speed, subset, window), seeds in sorted(grouped.items()):
        row = dict(
            condition=condition,
            method=method,
            speed_mm_s=speed,
            subset=subset,
            metric_window=window,
            seeds=len(seeds),
            cases_per_seed=[s["cases"] for s in seeds],
        )
        for metric in (
            "moving_seconds",
            "tip_mean_mm",
            "pitch_mean_deg",
            "valve_travel_per_s",
            "failures",
            "velocity_error_mm_s",
            "cruise_along_speed_ratio",
            "cruise_relative_velocity_error",
        ):
            values = [s[metric] for s in seeds if math.isfinite(s[metric])]
            row[metric] = statistics.mean(values) if values else None
            row[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(row)
    out.mkdir(parents=True, exist_ok=True)
    (out / "by_speed.json").write_text(json.dumps(result, indent=2) + "\n")
    if result:
        with (out / "by_speed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    lines = [
        "# Tracking by speed",
        "",
        "Mean +/- sample standard deviation of seed means. Do not mix motion-only and post-start windows.",
        "Capacity-feasible means the reference stays valid and nominal required/achievable speed <=0.9; it is a model-based label, not hardware validation.",
        "All cases remain in the all-case view, including infeasible requests. Missing feasible rows mean no qualifying cases, not zero error.",
        "The same 100 mm path takes longer at low speed. Interpret angle drift alongside motion duration and the full-trial angle metrics.",
        "",
    ]
    for condition, subset, window in sorted(
        {(r["condition"], r["subset"], r["metric_window"]) for r in result}
    ):
        lines += [
            f"## {condition}: {subset} ({window})",
            "",
            "| Method | mm/s | Motion s | Tip MAE mm | Angle MAE deg | Cruise velocity error % | Valve travel/s | Failures | Seeds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in result:
            if (row["condition"], row["subset"], row["metric_window"]) != (condition, subset, window):
                continue
            cells = [
                f"{row[m]:.2f} +/- {row[m + '_std']:.2f}" if row[m] is not None else "n/a"
                for m in ("tip_mean_mm", "pitch_mean_deg", "valve_travel_per_s", "failures")
            ]
            relative = row["cruise_relative_velocity_error"]
            cells.insert(
                2,
                f"{100 * relative:.1f} +/- {100 * row['cruise_relative_velocity_error_std']:.1f}"
                if relative is not None
                else "n/a",
            )
            duration = f"{row['moving_seconds']:.2f}" if row["moving_seconds"] is not None else "n/a"
            lines.append(
                f"| {row['method']} | {row['speed_mm_s']:g} | {duration} | {' | '.join(cells)} | {row['seeds']} |"
            )
        lines.append("")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    return result
