"""Multi-seed rotation remedies and preference-conditioned policies, followed by Monday reports."""

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

from hydraulic_controller.speed_audit import SPEEDS_MM_S, write_speed_report
from training.friday_extension import wait_for_base
from training.overnight import ROOT, Run, prevent_sleep, write_json
from training.rotation_ablation import VARIANTS

SEEDS = (42, 43, 44)
PREFERENCES = (0.0, 0.25, 0.5, 0.75, 1.0)


def budgets(available, rates, seeds):
    """Equal update counts within each comparison group, with a runtime margin."""
    if available <= 0 or any(value <= 0 for value in rates.values()):
        raise ValueError("Invalid remaining time or timing measurements")
    costs = dict(
        main=sum(rates.values()) * len(seeds),
        deploy=rates["tip_rot4x_linear"] * len(seeds),
        priority=statistics.mean(rates.values()) * 2 * len(seeds),
    )
    shares = dict(main=0.60, deploy=0.10, priority=0.30)
    counts = {
        group: math.floor(available * 0.92 * shares[group] / cost / 500) * 500
        for group, cost in costs.items()
    }
    if min(counts.values()) < 500:
        raise ValueError("Not enough time for all matched comparison groups")
    return counts


def jobs(plan, settings, counts):
    common = dict(settings, gyro_observations=True, motion_start_prob=0.3, pose_priority=False)
    if plan["smoke"]:
        common["motion_pool_size"] = 64
    result = []
    for seed in plan["seeds"]:
        for name, point, weight, penalty in VARIANTS:
            result.append(
                dict(
                    name=f"{name}_seed{seed}",
                    method=name,
                    seed=seed,
                    group="main",
                    steps=counts["main"],
                    config=dict(
                        common, tracked_point=point, angular_weight=weight, angular_error_weight=penalty
                    ),
                )
            )
    for seed in plan["seeds"]:
        result.append(
            dict(
                name=f"deploy_tip_seed{seed}",
                method="deploy_tip",
                seed=seed,
                group="deploy",
                steps=counts["deploy"],
                config=dict(
                    common,
                    tracked_point="tip",
                    angular_weight=2.0,
                    angular_error_weight=0.5,
                    lookahead_steps=0,
                    accel_weight=0.0,
                ),
            )
        )
    for seed in plan["seeds"]:
        for point in ("tip", "pivot"):
            result.append(
                dict(
                    name=f"priority_{point}_seed{seed}",
                    method=f"priority_{point}",
                    seed=seed,
                    group="priority",
                    steps=counts["priority"],
                    config=dict(
                        common,
                        tracked_point=point,
                        lookahead_steps=0,
                        accel_weight=0.0,
                        pose_priority=True,
                        pose_angular_weight_max=2.0,
                        pose_angular_error_weight_max=0.5,
                    ),
                )
            )
    return result


def command(plan, model, name, directory, config, steps, seed, seconds=None):
    excluded = {"model_path", "num_envs", "device", "seed"}
    overrides = [f"{key}={json.dumps(value)}" for key, value in config.items() if key not in excluded]
    cmd = [
        ROOT / "train_controller.py",
        "--model",
        model,
        "--run_name",
        name,
        "--log_dir",
        directory,
        "--max_iterations",
        steps,
        "--num_envs",
        64 if plan["smoke"] else 4096,
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
        cmd += ["--max_seconds", max(1, round(seconds))]
    return cmd


def preference_label(value):
    return f"w{value:.2f}".replace(".", "p")


def selected_checkpoints(out, all_jobs):
    """Use equal checkpoint indices if a time limit shortened any member of a group."""
    ends = defaultdict(list)
    for job in all_jobs:
        value = json.loads((out / "controllers" / job["name"] / "training_result.json").read_text())
        ends[job["group"]].append(value["iteration"])
    indices = {
        group: values[0] if len(set(values)) == 1 else min(values) // 100 * 100
        for group, values in ends.items()
    }
    chosen = {}
    for job in all_jobs:
        path = out / "controllers" / job["name"] / f"model_{indices[job['group']]}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing matched checkpoint: {path}")
        chosen[job["name"]] = str(path)
    write_json(out / "selected_checkpoints.json", dict(indices=indices, checkpoints=chosen))
    return chosen


def aggregate(out):
    cells = defaultdict(list)
    flat = []
    for path in sorted((out / "benchmarks").glob("*/summary.json")):
        tag = path.parent.name.split("_", 2)[-1]
        condition, _, seed = tag.rpartition("_seed")
        data = json.loads(path.read_text())
        for method, summary in data["summaries"].items():
            row = dict(condition=condition, seed=int(seed), method=method, **summary["all"])
            flat.append(row)
            cells[(condition, method)].append(row)
    metrics = ("tip_mean_mm", "pitch_mean_deg", "tip_p95_max_mm", "valve_travel_per_s", "failures")
    summaries = {}
    for (condition, method), rows in cells.items():
        summaries.setdefault(condition, {})[method] = {
            metric: dict(
                mean=statistics.mean(r[metric] for r in rows),
                std=statistics.stdev(r[metric] for r in rows) if len(rows) > 1 else 0.0,
            )
            for metric in metrics
        }
    write_json(out / "aggregate.json", summaries)
    if flat:
        with (out / "aggregate.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    lines = [
        "# Weekend comparison",
        "",
        "Mean +/- sample standard deviation over training seeds. Evaluation noise is shared across policies and seeds.",
        "All trajectories and plant perturbations are included, including infeasible references. No candidate is hardware validated.",
        "Reward/penalty variations change angular-velocity tracking in the inner policy; reported bucket-angle error is measured in the outer pose-tracking loop.",
        "Line sweeps cover 5, 10, 20, 40, 70, 100 and 120 mm/s. See [speed-specific motion/feasibility results](speed_analysis/results.md); headline averages include endpoint holds.",
    ]
    for condition, methods in sorted(summaries.items()):
        lines += [
            "",
            f"## {condition}",
            "",
            "| Method | Tip MAE mm | Bucket pitch MAE deg | Valve travel/s | Recorded failures |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for method, values in methods.items():

            def cell(metric):
                value = values[metric]
                return f"{value['mean']:.2f} +/- {value['std']:.2f}"

            lines.append(
                f"| {method} | {cell('tip_mean_mm')} | {cell('pitch_mean_deg')} | {cell('valve_travel_per_s')} | {cell('failures')} |"
            )
    rotation = rotation_summary(out)
    if rotation:
        lines += [
            "",
            "## Fixed-tip rotation",
            "",
            "Per seed, take the worst case across starting poses and plant perturbations, then summarize across training seeds.",
            "| Condition | Method | Worst tip error mm | Worst angle error deg | Recorded failures |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for condition, methods in sorted(rotation.items()):
            for method, values in sorted(methods.items()):
                cells = [
                    f"{values[k]['mean']:.2f} +/- {values[k]['std']:.2f}"
                    for k in ("tip_max_mm", "angle_max_deg", "failures")
                ]
                lines.append(f"| {condition} | {method} | {' | '.join(cells)} |")
    lines += [
        "",
        "## Deployment target",
        "",
        "Target: current native_stack. Its existing GR00T/OpenPI loaders do not load these PPO bundles.",
        "See [the native integration handoff](../native_handoff/README.md) for the pinned source, input/action mapping and remaining work.",
        "Export checks below replay a frozen legacy Python runtime against a generated USD profile. They do not verify the active native runtime or its physical calibration.",
    ]
    export_path = out / "export_checks.json"
    if export_path.exists():
        for name, result in json.loads(export_path.read_text()).items():
            lines.append(
                f"- {name}: legacy replay {'passed' if result.get('verified') else 'FAILED'}"
                + (f" ({result['error']})" if result.get("error") else "")
            )
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summaries:
        plot_preferences(out, summaries)
    write_speed_report(out / "benchmarks", out / "speed_analysis")


def rotation_summary(out):
    cells = defaultdict(list)
    for path in sorted((out / "rotation").glob("seed*/*/rotation_results.json")):
        data = json.loads(path.read_text())
        grouped = defaultdict(list)
        for row in data["rows"]:
            grouped[f"{row['sensor_mode']}_{row['amplitude_deg']:g}deg"].append(row)
        for condition, rows in grouped.items():
            cells[(condition, path.parent.name)].append(
                dict(
                    tip_max_mm=max(r["tip_max_mm"] for r in rows),
                    angle_max_deg=max(r["angle_max_deg"] for r in rows),
                    failures=sum(r["failed"] for r in rows),
                )
            )
    result = {}
    for (condition, method), rows in cells.items():
        result.setdefault(condition, {})[method] = {
            metric: dict(
                mean=statistics.mean(r[metric] for r in rows),
                std=statistics.stdev(r[metric] for r in rows) if len(rows) > 1 else 0.0,
                worst=max(r[metric] for r in rows),
                seeds=len(rows),
            )
            for metric in ("tip_max_mm", "angle_max_deg", "failures")
        }
    write_json(out / "rotation_summary.json", result)
    return result


def plot_preferences(out, summaries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [
        c
        for c in ("standard_new_lines", "standard_new_circles", "gyro_new_lines", "gyro_new_circles")
        if c in summaries
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, condition in zip(axes.flat, conditions):
        values = summaries[condition]
        for point in ("tip", "pivot"):
            curve = [(w, values.get(f"priority_{point}_{preference_label(w)}")) for w in PREFERENCES]
            curve = [(w, value) for w, value in curve if value is not None]
            if not curve:
                continue
            x = [v["tip_mean_mm"]["mean"] for _, v in curve]
            y = [v["pitch_mean_deg"]["mean"] for _, v in curve]
            ax.errorbar(
                x,
                y,
                xerr=[v["tip_mean_mm"]["std"] for _, v in curve],
                yerr=[v["pitch_mean_deg"]["std"] for _, v in curve],
                marker="o",
                label=f"priority {point}",
            )
            for w, v in curve:
                ax.annotate(
                    f"{w:g}",
                    (v["tip_mean_mm"]["mean"], v["pitch_mean_deg"]["mean"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        for method in (
            "deployed",
            "tip_control",
            "tip_rot4x",
            "tip_rot4x_linear",
            "pivot_control",
            "deploy_tip",
        ):
            if method in values:
                v = values[method]
                ax.scatter(v["tip_mean_mm"]["mean"], v["pitch_mean_deg"]["mean"], marker="x", label=method)
        ax.set(title=condition, xlabel="Mean EE position error [mm]", ylabel="Mean bucket-angle error [deg]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.savefig(out / "preference_tradeoff.png", dpi=160)
    plt.close(fig)


def execute(run):
    p = run.plan
    wait_for_base(run, Path(p["quick"]))
    write_speed_report(Path(p["quick"]) / "benchmarks", Path(p["quick"]) / "speed_analysis")
    base = Path(p["base"])
    model = Path((base / "best_actuator.txt").read_text().strip())
    settings = yaml.safe_load((base / "controllers/lookahead_accel_lr3e-4/params/env.yaml").read_text())
    smoke_counts = dict(main=2, deploy=2, priority=2)
    prototypes = jobs(p, settings, smoke_counts)[:4]
    pilot_steps = 2 if p["smoke"] else 100
    rates = {}
    for job in prototypes:
        directory = run.out / "pilots" / job["method"]
        result = directory / "training_result.json"
        run.step(
            f"pilot_{job['method']}",
            command(p, model, job["method"], directory, job["config"], pilot_steps, 42),
            artifact=result,
        )
        rates[job["method"]] = json.loads(result.read_text())["elapsed_s"] / pilot_steps
    timing_path = run.out / "timing.json"
    if timing_path.exists():
        counts = json.loads(timing_path.read_text())["iterations"]
    else:
        counts = smoke_counts if p["smoke"] else budgets(p["training_end"] - time.time(), rates, p["seeds"])
        write_json(timing_path, dict(rates=rates, iterations=counts, pilot_weights_discarded=True))
    all_jobs = jobs(p, settings, counts)
    write_json(run.out / "jobs.json", all_jobs)

    def estimated(job):
        return rates.get(job["method"], statistics.mean(rates.values())) * job["steps"]

    for index, job in enumerate(all_jobs):
        directory = run.out / "controllers" / job["name"]
        result = directory / "training_result.json"
        if result.exists():
            continue
        remaining = [
            j
            for j in all_jobs[index:]
            if not (run.out / "controllers" / j["name"] / "training_result.json").exists()
        ]
        seconds = (
            (p["training_end"] - time.time() - 60) * estimated(job) / sum(estimated(j) for j in remaining)
        )
        if seconds <= 0 and not p["smoke"]:
            raise RuntimeError("Training window expired; preserve the partial comparison")
        cmd = command(
            p,
            model,
            job["name"],
            directory,
            job["config"],
            job["steps"],
            job["seed"],
            None if p["smoke"] else seconds,
        )
        saved = sorted(directory.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))
        if saved:
            cmd += ["--resume", saved[-1]]
            cmd[cmd.index("--max_iterations") + 1] = max(1, job["steps"] - int(saved[-1].stem.split("_")[-1]))
        run.step(job["name"], cmd, artifact=result)
    selected = selected_checkpoints(run.out, all_jobs)
    references = json.loads((base / "controller_checkpoints.json").read_text())
    choices = (0.0, 1.0) if p["smoke"] else PREFERENCES
    rotations = []
    for seed in p["seeds"]:
        actors = list(references)
        moderate = list(references)
        for job in [j for j in all_jobs if j["seed"] == seed]:
            checkpoint = selected[job["name"]]
            if job["group"] == "priority":
                for value in choices:
                    method = f"{job['method']}_{preference_label(value)}"
                    actors.append(f"{method}={checkpoint}#{value}")
                    if value in (0.0, 0.5, 1.0):
                        rotations.append(
                            dict(method=method, seed=seed, checkpoint=checkpoint, priority=value)
                        )
                    if value == 0.5:
                        moderate.append(f"{method}={checkpoint}#{value}")
            else:
                actors.append(f"{job['method']}={checkpoint}")
                moderate.append(f"{job['method']}={checkpoint}")
                rotations.append(dict(method=job["method"], seed=seed, checkpoint=checkpoint, priority=None))
        for profile in ["standard"] if p["smoke"] else ["standard", "gyro", "stress"]:
            evaluated = actors if profile != "stress" else moderate
            for label, plant in (
                [("new", model)] if p["smoke"] else [("new", model), ("deployed", ROOT / "models/actuator")]
            ):
                for task in ["lines"] if p["smoke"] else ["lines", "circles"]:
                    tag = f"{profile}_{label}_{task}_seed{seed}"
                    cmd = [
                        ROOT / "compare_controllers.py",
                        "--controllers",
                        "pid_robot",
                        "mlp",
                        "--robot_repo",
                        p["robot_repo"],
                        "--mlp",
                        *evaluated,
                        "--model",
                        plant,
                        "--asset",
                        settings["asset_path"],
                        "--task",
                        task,
                        "--sensor_profile",
                        profile,
                        "--seed",
                        731,
                        "--out",
                        run.out / "benchmarks",
                        "--run_name",
                        tag,
                    ]
                    if p["smoke"]:
                        cmd += ["--plants", "nominal", "--speeds_mm_s", 40]
                    if task == "lines":
                        cmd += ["--speed_sweep"]
                    run.step(f"benchmark_{tag}", cmd)
                    aggregate(run.out)
    for choice in rotations:
        if p["smoke"] and choice["method"] != "priority_tip_w1p00":
            continue
        directory = run.out / "rotation" / f"seed{choice['seed']}" / choice["method"]
        cmd = [
            "-m",
            "hydraulic_controller.rotation_benchmark",
            "--checkpoint",
            choice["checkpoint"],
            "--out",
            directory,
            "--model",
            model,
            "--asset",
            settings["asset_path"],
            "--poses",
            1,
            "--cycles",
            1,
        ]
        if choice["priority"] is not None:
            cmd += ["--priority", choice["priority"]]
        run.step(
            f"rotation_{choice['method']}_seed{choice['seed']}",
            cmd,
            artifact=directory / "rotation_results.json",
        )
    export_results = {}
    for job in [j for j in all_jobs if j["group"] == "deploy"]:
        directory = run.out / "exports" / job["name"]
        try:
            run.step(
                f"export_{job['name']}",
                [
                    "-m",
                    "training.verify_export",
                    "--checkpoint",
                    selected[job["name"]],
                    "--robot_repo",
                    p["robot_repo"],
                    "--out",
                    directory,
                ],
                artifact=directory / "export_check.json",
            )
            export_results[job["name"]] = json.loads((directory / "export_check.json").read_text())
        except (RuntimeError, OSError) as exc:
            export_results[job["name"]] = dict(verified=False, error=str(exc))
    write_json(run.out / "export_checks.json", export_results)
    aggregate(run.out)
    run.status(
        status="complete",
        stage="done",
        child_pid=None,
        export_checks=export_results,
        finished=datetime.now().astimezone().isoformat(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--quick", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--finish_at", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    finish = datetime.fromisoformat(args.finish_at)
    if finish.tzinfo is None:
        parser.error("Deadline must include its timezone")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "completed").mkdir(exist_ok=True)
    with FileLock(out / "runner.lock", timeout=0):
        plan_path = out / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            original = json.loads((args.base / "plan.json").read_text())
            plan = dict(
                original,
                base=str(args.base.resolve()),
                quick=str(args.quick.resolve()),
                smoke=args.smoke,
                finish_at=finish.isoformat(),
                training_end=finish.timestamp() - 6 * 3600,
                seeds=[42] if args.smoke else list(SEEDS),
                preferences=list(PREFERENCES),
                primary_comparison="Four fixed reward/geometry conditions, 3 seeds, matched PPO updates from scratch",
                priority_semantics="w=0 ignores angular-velocity reward; w=1 gives angular-velocity reward 2.0 and normalized angular-rate-error penalty 0.5; position and common costs remain active",
                deployment_target="native_stack; native PPO adapter pending, legacy Python export is reference only",
                line_evaluation_speeds_mm_s=list(SPEEDS_MM_S),
                hardware_validated=False,
            )
            write_json(plan_path, plan)
            source = out / "source"
            source.mkdir(exist_ok=True)
            shutil.copy2(__file__, source / Path(__file__).name)
            (out / "source.patch").write_bytes(
                subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT, stderr=subprocess.DEVNULL)
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
