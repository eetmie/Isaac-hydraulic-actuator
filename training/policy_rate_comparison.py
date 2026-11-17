# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Train matched pivot policies at different frequencies, then validate tracking and smoothness."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from filelock import FileLock

from hydraulic_controller import DEFAULT_CHECKPOINT
from hydraulic_controller.core import model_fingerprint, sha256
from hydraulic_controller.speed_audit import write_speed_report
from training.overnight import ROOT, Run, prevent_sleep, write_json
from training.weekend_comparison import selected_checkpoints

RATES = (10, 20, 50, 100)
SEEDS = (42, 43, 44)


def rate_settings(settings, hz, *, smoke=False):
    """Keep temporal horizons and valve-rate costs anchored to the source recipe in physical seconds."""
    if hz not in RATES:
        raise ValueError(f"Unsupported study rate: {hz}")
    source_hz = settings["policy_hz"]
    result = dict(settings)
    result.update(
        policy_hz=hz,
        tracked_point="pivot",
        pose_priority=False,
        lookahead_steps=round(settings["lookahead_steps"] / source_hz * hz),
        curriculum_steps=round(settings["curriculum_steps"] / source_hz * hz),
        motion_pool_refresh_steps=round(settings["motion_pool_refresh_steps"] / source_hz * hz),
        # L1 integrates valve travel. L2 integrates squared valve rate, so its per-step coefficient is 1/dt.
        action_rate_l2=settings["action_rate_l2"] * hz / source_hz,
    )
    if smoke:
        result["motion_pool_size"] = 64
    return result


def matched_iterations(available, rates, seeds):
    """Identical PPO updates and 1.2 s simulated rollouts at every rate, with a 10% timing margin."""
    if available <= 0 or any(rate <= 0 for rate in rates.values()):
        raise ValueError("Invalid training window or pilot timings")
    count = math.floor(available * 0.90 / sum(rates.values()) / len(seeds) / 100) * 100
    if count < 500:
        raise ValueError("Not enough time for the matched policy-rate comparison")
    return count


def command(plan, name, directory, config, steps, seed, seconds=None):
    excluded = {"model_path", "num_envs", "device", "seed", "policy_hz"}
    overrides = [f"{key}={json.dumps(value)}" for key, value in config.items() if key not in excluded]
    result = [
        ROOT / "train_controller.py",
        "--model",
        plan["model"],
        "--run_name",
        name,
        "--log_dir",
        directory,
        "--num_envs",
        64 if plan["smoke"] else 4096,
        "--max_iterations",
        steps,
        "--policy_hz",
        config["policy_hz"],
        "--rollout_s",
        1.2,
        "--save_interval",
        1 if plan["smoke"] else 100,
        "--learning_rate",
        3e-4,
        "--lr_schedule",
        "fixed",
        "--seed",
        seed,
        "--set",
        *overrides,
    ]
    if seconds is not None:
        result += ["--max_seconds", max(1, round(seconds))]
    return result


def aggregate(out, introduction=None):
    """Aggregate seed means; keep conditions separate and include smoothness and tracking together."""
    grouped = defaultdict(list)
    latest = {}
    for source in sorted((out / "benchmarks").glob("*/summary.json")):
        tag = source.parent.name.split("_", 2)[-1]
        latest[tag] = source
    for tag, source in latest.items():
        condition, _, seed = tag.rpartition("_seed")
        data = json.loads(source.read_text())
        for method, result in data["summaries"].items():
            for subset in ("all", "capacity_feasible"):
                smooth = data["smoothness"][method][subset]
                # Tracking below is the headline all-case score; feasible motion-only tracking is in speed_analysis.
                grouped[(condition, method, subset)].append(
                    dict(
                        seed=int(seed),
                        **result["all"],
                        **smooth,
                    )
                )
    rows = []
    for (condition, method, subset), members in sorted(grouped.items()):
        row = dict(condition=condition, method=method, smoothness_subset=subset, seeds=len(members))
        for key in members[0]:
            if key == "seed":
                continue
            values = [value[key] for value in members if math.isfinite(value[key])]
            row[key] = statistics.mean(values) if values else math.nan
            row[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    write_json(out / "aggregate.json", rows)
    if rows:
        with (out / "aggregate.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = (
        list(introduction)
        if introduction is not None
        else [
            "# Policy-rate comparison",
            "",
            "Fresh pivot policies at 10/20/50/100 Hz; 100 Hz sensors and hydraulic plant throughout.",
            "All rates use matched PPO updates and 1.2 s simulated rollouts. They have different numbers of policy transitions and compute costs.",
            "Preview horizon is 0.2 s; motion-pool refresh is 6 s; curriculum duration is held fixed. L1 valve travel is unchanged and L2 valve-rate cost scales with Hz/20.",
            "Reference20 is the packaged 3500-update weekend policy, not a member of the matched fresh-training comparison.",
            "Mean +/- sample standard deviation across training seeds. Physical smoothness uses 100 Hz finite differences; valve steps are sampled at actual policy updates.",
            "All-case tracking includes endpoint holds and infeasible requests. See [motion and feasibility results by speed](speed_analysis/results.md).",
            "",
            "No controller is automatically promoted or deployed. Native runtime integration and measured robot timing are separate work.",
        ]
    )
    for condition in sorted({row["condition"] for row in rows}):
        lines += [
            "",
            f"## {condition}",
            "",
            "| Method | Tip MAE mm | Angle MAE deg | Valve travel/s | Valve step p99 | Motion accel m/s² | Motion jerk m/s³ | Hold drift mm | Failures |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["condition"] != condition or row["smoothness_subset"] != "all":
                continue
            fields = (
                "tip_mean_mm",
                "pitch_mean_deg",
                "valve_travel_per_s",
                "valve_step_p99",
                "motion_accel_rms_m_s2",
                "motion_jerk_rms_m_s3",
                "hold_tip_p2p_mm",
                "failures",
            )
            cells = [f"{row[key]:.3f} +/- {row[key + '_std']:.3f}" for key in fields]
            lines.append(f"| {row['method']} | {' | '.join(cells)} |")
    rotation = out / "rotation_summary.json"
    if rotation.exists():
        lines += [
            "",
            "## Fixed-tip rotation",
            "",
            "See rotation_summary.json for worst position/angle errors and failures, including both actuator plants.",
        ]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_speed_report(out / "benchmarks", out / "speed_analysis")


def execute(run):
    plan = run.plan
    settings = plan["source_settings"]
    configs = {hz: rate_settings(settings, hz, smoke=plan["smoke"]) for hz in plan["rates"]}
    for config in configs.values():
        config.update(
            model_path=plan["model"],
            asset_path=plan["asset"],
            extra_model_paths=[str(ROOT / "models/actuator")],
        )
    rates = {}
    # Exercise the largest rollout buffer first, so GPU-capacity failures surface before allocating the night.
    for hz in sorted(configs, reverse=True):
        config = configs[hz]
        name = f"rate{hz}"
        directory = run.out / "pilots" / name
        artifact = directory / "training_result.json"
        steps = 2 if plan["smoke"] else 100
        run.step(f"pilot_{name}", command(plan, name, directory, config, steps, 42), artifact=artifact)
        rates[hz] = json.loads(artifact.read_text())["elapsed_s"] / steps
    timing_path = run.out / "timing.json"
    if timing_path.exists():
        count = json.loads(timing_path.read_text())["iterations_each"]
    else:
        count = (
            2
            if plan["smoke"]
            else matched_iterations(plan["training_end"] - time.time(), rates, plan["seeds"])
        )
        write_json(
            timing_path,
            dict(
                seconds_per_iteration=rates,
                iterations_each=count,
                simulated_rollout_s=1.2,
                pilot_weights_discarded=True,
            ),
        )
    jobs = [
        dict(name=f"rate{hz}_seed{seed}", hz=hz, seed=seed, group="matched")
        for seed in plan["seeds"]
        for hz in plan["rates"]
    ]
    write_json(run.out / "jobs.json", jobs)
    for index, job in enumerate(jobs):
        directory = run.out / "controllers" / job["name"]
        artifact = directory / "training_result.json"
        if artifact.exists():
            continue
        remaining = [
            value
            for value in jobs[index:]
            if not (run.out / "controllers" / value["name"] / "training_result.json").exists()
        ]
        seconds = (
            (plan["training_end"] - time.time() - 60)
            * rates[job["hz"]]
            / sum(rates[value["hz"]] for value in remaining)
        )
        if seconds <= 0 and not plan["smoke"]:
            raise RuntimeError("Training window expired; preserve the partial comparison")
        saved = sorted(directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
        steps = max(1, count - int(saved[-1].stem.split("_")[-1])) if saved else count
        cmd = command(
            plan,
            job["name"],
            directory,
            configs[job["hz"]],
            steps,
            job["seed"],
            None if plan["smoke"] else seconds,
        )
        if saved:
            cmd += ["--resume", saved[-1]]
        run.step(job["name"], cmd, artifact=artifact)
    selected = selected_checkpoints(run.out, jobs)
    rotations = {}
    for seed in plan["seeds"]:
        actors = [f"reference20={DEFAULT_CHECKPOINT}"] + [
            f"rate{job['hz']}={selected[job['name']]}" for job in jobs if job["seed"] == seed
        ]
        profiles = ("gyro",) if plan["smoke"] else ("standard", "gyro", "stress")
        plants = [("new", plan["model"])]
        if not plan["smoke"]:
            plants.append(("deployed", ROOT / "models/actuator"))
        for profile in profiles:
            for label, model in plants:
                for task in ("lines",) if plan["smoke"] else ("lines", "circles"):
                    name = f"{profile}_{label}_{task}_seed{seed}"
                    cmd = [
                        ROOT / "compare_controllers.py",
                        "--controllers",
                        "pid_robot",
                        "mlp",
                        "--robot_repo",
                        plan["robot_repo"],
                        "--mlp",
                        *actors,
                        "--model",
                        model,
                        "--asset",
                        plan["asset"],
                        "--sensor_profile",
                        profile,
                        "--task",
                        task,
                        "--seed",
                        731,
                        "--out",
                        run.out / "benchmarks",
                        "--run_name",
                        name,
                        "--smoothness",
                    ]
                    if task == "lines" and not plan["smoke"]:
                        cmd += ["--speed_sweep"]
                    if plan["smoke"]:
                        cmd += ["--plants", "nominal", "--speeds_mm_s", 40]
                    run.step(f"benchmark_{name}", cmd)
                    aggregate(run.out)
        for job in [value for value in jobs if value["seed"] == seed]:
            if plan["smoke"] and job["hz"] != 20:
                continue
            for label, model in plants:
                directory = run.out / "rotation" / job["name"] / label
                run.step(
                    f"rotation_smoothness_{job['name']}_{label}",
                    [
                        "-m",
                        "hydraulic_controller.rotation_benchmark",
                        "--checkpoint",
                        selected[job["name"]],
                        "--model",
                        model,
                        "--asset",
                        plan["asset"],
                        "--out",
                        directory,
                        "--poses",
                        1,
                        "--cycles",
                        1,
                        "--smoothness",
                    ],
                    artifact=directory / "rotation_results.json",
                )
                data = json.loads((directory / "rotation_results.json").read_text())
                rotations[f"{job['name']}_{label}"] = dict(
                    tip_max_mm=max(row["tip_max_mm"] for row in data["rows"]),
                    angle_max_deg=max(row["angle_max_deg"] for row in data["rows"]),
                    failures=sum(row["failed"] for row in data["rows"]),
                    **{
                        key + "_worst": max(row[key] for row in data["rows"])
                        for key in ("motion_accel_rms_m_s2", "motion_jerk_rms_m_s3", "valve_step_p99")
                    },
                )
                write_json(run.out / "rotation_summary.json", rotations)
    aggregate(run.out)
    run.status(
        status="complete", stage="done", child_pid=None, finished=datetime.now().astimezone().isoformat()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, type=Path, help="Frozen base from training.freeze_comparison"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--finish_at", required=True)
    parser.add_argument("--evaluation_hours", type=float, default=4.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    finish = datetime.fromisoformat(args.finish_at)
    if finish.tzinfo is None or args.evaluation_hours <= 0:
        parser.error("Use a timezone-aware deadline and positive evaluation reserve")
    out = args.out.resolve()
    (out / "completed").mkdir(parents=True, exist_ok=True)
    with FileLock(out / "runner.lock", timeout=0):
        plan_path = out / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            base = args.base.resolve()
            source = json.loads((base / "plan.json").read_text())
            model = Path((base / "best_actuator.txt").read_text().strip())
            settings = yaml.safe_load((DEFAULT_CHECKPOINT.parent / "params/env.yaml").read_text())
            plan = dict(
                repository=str(ROOT),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                base=str(base),
                model=str(model),
                model_files=model_fingerprint(model),
                asset=str(ROOT / "assets/excavator_bucket_rocking.usd"),
                asset_sha256=sha256(ROOT / "assets/excavator_bucket_rocking.usd"),
                robot_repo=source["robot_repo"],
                finish_at=finish.isoformat(),
                training_end=finish.timestamp() - args.evaluation_hours * 3600,
                rates=list(RATES),
                seeds=[42] if args.smoke else list(SEEDS),
                source_settings=settings,
                smoke=args.smoke,
                hardware_validated=False,
            )
            write_json(plan_path, plan)
            shutil.copy2(__file__, out / "runner_source.py")
            (out / "source.patch").write_bytes(
                subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT)
            )
        run = Run(out, plan)
        try:
            with prevent_sleep():
                execute(run)
        except BaseException as exc:
            run.status(status="failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
