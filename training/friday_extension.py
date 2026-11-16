"""After the initial queue, compare gyro training with and without moving resets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml
from filelock import FileLock

from training.overnight import ROOT, Run, prevent_sleep, write_json

VARIATIONS = [("gyro_lr1e-4", 0.0), ("gyro_motion30_lr1e-4", 0.3)]


def matched_iterations(available_seconds, rates, pilot_steps):
    """Match update counts while allowing timing overhead and a small runtime margin."""
    remaining = math.floor(max(0, available_seconds - 600) * 0.94 / sum(rates) / 100) * 100
    if remaining < 100:
        raise ValueError("Not enough time left for the matched controller experiments")
    return pilot_steps + remaining


def controller_command(plan, best, name, directory, resume, overrides, iterations, seconds=None):
    command = [
        ROOT / "train_controller.py",
        "--model",
        best,
        "--run_name",
        name,
        "--log_dir",
        directory,
        "--resume",
        resume,
        "--num_envs",
        64 if plan["smoke"] else 4096,
        "--max_iterations",
        iterations,
        "--save_interval",
        1 if plan["smoke"] else 100,
        "--learning_rate",
        1e-4,
        "--lr_schedule",
        "fixed",
        "--seed",
        42,
        "--set",
        *overrides,
    ]
    if seconds is not None:
        command += ["--max_seconds", seconds]
    return command


def wait_for_base(run, base):
    run.status(status="waiting", stage="waiting_for_initial_queue", base=str(base))
    while True:
        try:
            state = json.loads((base / "status.json").read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
        if state.get("status") == "complete":
            return
        if state.get("status") == "failed":
            raise RuntimeError(f"Initial queue failed: {state.get('error')}")
        if state.get("pid") and not psutil.pid_exists(state["pid"]):
            raise RuntimeError("Initial queue process exited before completing")
        run.status(base_stage=state.get("stage"), base_updated=state.get("updated"))
        time.sleep(30)


def execute(run):
    p = run.plan
    base = Path(p["base"])
    wait_for_base(run, base)
    best = Path((base / "best_actuator.txt").read_text().strip())
    common_dir = base / "controllers/lookahead_accel_lr1e-4"
    common = Path(json.loads((common_dir / "training_result.json").read_text())["checkpoint"])
    settings = yaml.safe_load((common_dir / "params/env.yaml").read_text())
    # These two runs intentionally share the same starting weights and optimizer.
    write_json(run.out / "common_parent.json", dict(checkpoint=str(common), environment=settings))
    checkpoints = json.loads((base / "controller_checkpoints.json").read_text())
    configs, pilots = {}, {}
    pilot_steps = 2 if p["smoke"] else 100
    for name, moving in VARIATIONS:
        config = dict(settings, gyro_observations=True, motion_start_prob=moving)
        if p["smoke"]:
            config["motion_pool_size"] = 64
        excluded = {"model_path", "num_envs", "device", "seed"}
        configs[name] = [f"{key}={json.dumps(value)}" for key, value in config.items() if key not in excluded]
        directory = run.out / "pilots" / name
        result = directory / "training_result.json"
        command = controller_command(p, best, name, directory, common, configs[name], pilot_steps)
        run.step(f"pilot_{name}", command, artifact=result)
        pilots[name] = json.loads(result.read_text())
    rates = {name: result["elapsed_s"] / pilot_steps for name, result in pilots.items()}
    timing = run.out / "timing.json"
    if timing.exists():
        target = json.loads(timing.read_text())["additional_iterations_each"]
    else:
        target = (
            4
            if p["smoke"]
            else matched_iterations(p["training_end"] - time.time(), list(rates.values()), pilot_steps)
        )
        write_json(
            timing,
            dict(seconds_per_iteration=rates, pilot_steps=pilot_steps, additional_iterations_each=target),
        )
    for index, (name, moving) in enumerate(VARIATIONS):
        directory = run.out / "controllers" / name
        result = directory / "training_result.json"
        if not result.exists():
            remaining_rate = sum(rates[variant] for variant, _ in VARIATIONS[index:])
            seconds = (p["training_end"] - time.time() - 120) * rates[name] / remaining_rate
            if seconds <= 0 and not p["smoke"]:
                raise RuntimeError("Extension training window expired before all variations completed")
            saved = sorted(directory.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))
            resume = saved[-1] if saved else Path(pilots[name]["checkpoint"])
            iterations = target - pilot_steps
            if saved:
                restored = int(saved[-1].stem.split("_")[-1])
                iterations = max(1, pilots[name]["iteration"] + iterations - restored)
            command = controller_command(
                p,
                best,
                name,
                directory,
                resume,
                configs[name],
                iterations,
                0.001 if p["smoke"] else round(seconds),
            )
            run.step(name, command, artifact=result)
        checkpoints.append(f"{name}={json.loads(result.read_text())['checkpoint']}")
        write_json(run.out / "controller_checkpoints.json", checkpoints)

    plants = [("deployed", ROOT / "models/actuator")]
    if best.resolve() != plants[0][1].resolve():
        plants.append(("new", best))
    for profile in ["gyro"] if p["smoke"] else ["standard", "gyro", "stress"]:
        for label, model in plants:
            for task in ["lines"] if p["smoke"] else ["lines", "circles"]:
                command = [
                    ROOT / "compare_controllers.py",
                    "--controllers",
                    "pid_robot",
                    "mlp",
                    "--robot_repo",
                    p["robot_repo"],
                    "--mlp",
                    *checkpoints,
                    "--model",
                    model,
                    "--task",
                    task,
                    "--sensor_profile",
                    profile,
                    "--out",
                    run.out / "benchmarks",
                    "--run_name",
                    f"{profile}_{label}_{task}",
                ]
                if p["smoke"]:
                    command += ["--plants", "nominal", "--speeds_mm_s", 40]
                run.step(f"benchmark_{profile}_{label}_{task}", command)
                run.report()
    run.status(
        status="complete", stage="done", child_pid=None, finished=datetime.now().astimezone().isoformat()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--finish_at", required=True, help="ISO timestamp including timezone")
    parser.add_argument("--benchmark_reserve_minutes", type=float, default=90)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    finish = datetime.fromisoformat(args.finish_at)
    if finish.tzinfo is None or args.benchmark_reserve_minutes < 0:
        parser.error("Use a timezone-aware deadline and a nonnegative benchmark reserve")
    out = (args.out or args.base / "friday_extension").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "completed").mkdir(exist_ok=True)
    with FileLock(out / "runner.lock", timeout=0):
        plan_path = out / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            base_plan = json.loads((args.base / "plan.json").read_text())
            plan = dict(
                base_plan,
                base=str(args.base.resolve()),
                smoke=args.smoke,
                finish_at=finish.isoformat(),
                training_end=finish.timestamp() - args.benchmark_reserve_minutes * 60,
                variations=[dict(name=name, moving_reset_probability=prob) for name, prob in VARIATIONS],
                purpose="Matched fine-tuning from the same parent: gyro observations; gyro plus 30% moving resets",
                common_parent="controllers/lookahead_accel_lr1e-4",
                learning_rate=1e-4,
            )
            write_json(plan_path, plan)
            source = out / "source"
            source.mkdir(exist_ok=True)
            for path in [
                Path(__file__),
                ROOT / "training/overnight.py",
                ROOT / "compare_controllers.py",
                ROOT / "hydraulic_controller/observations.py",
            ]:
                shutil.copy2(path, source / path.name)
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
