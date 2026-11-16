"""Short controlled comparison of rotation reward/penalty and pivot-based tracking."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from filelock import FileLock

from training.overnight import ROOT, Run, prevent_sleep, write_json

VARIANTS = [
    ("tip_control", "tip", 0.5, 0.0),
    ("tip_rot4x", "tip", 2.0, 0.0),
    ("tip_rot4x_linear", "tip", 2.0, 0.5),
    ("pivot_control", "pivot", 0.5, 0.0),
]
PAPER = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/74f14861-3a45-4d24-a092-10d1619ff339/content"


def execute(run):
    p = run.plan
    base = Path(p["base"])
    model = Path((base / "best_actuator.txt").read_text().strip())
    parent = base / "controllers/lookahead_accel_lr3e-4"
    settings = yaml.safe_load((parent / "params/env.yaml").read_text())
    original = json.loads((parent / "training_result.json").read_text())["checkpoint"]
    reference = json.loads((base / "controller_checkpoints.json").read_text())[0]
    checkpoints = [reference, f"previous_tip_long={original}"]
    for name, point, weight, penalty in VARIANTS:
        directory = run.out / "controllers" / name
        config = dict(settings, tracked_point=point, angular_weight=weight, angular_error_weight=penalty)
        excluded = {"model_path", "num_envs", "device", "seed"}
        overrides = [f"{key}={json.dumps(value)}" for key, value in config.items() if key not in excluded]
        command = [
            ROOT / "train_controller.py",
            "--model",
            model,
            "--run_name",
            name,
            "--log_dir",
            directory,
            "--max_iterations",
            p["iterations"],
            "--num_envs",
            64 if p["smoke"] else 4096,
            "--save_interval",
            1 if p["smoke"] else 100,
            "--learning_rate",
            3e-4,
            "--lr_schedule",
            "fixed",
            "--seed",
            42,
            "--set",
            *overrides,
        ]
        saved = sorted(directory.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))
        if saved:
            command += ["--resume", saved[-1]]
            command[command.index("--max_iterations") + 1] = max(
                1, p["iterations"] - int(saved[-1].stem.split("_")[-1])
            )
        result = directory / "training_result.json"
        run.step(name, command, artifact=result)
        checkpoints.append(f"{name}={json.loads(result.read_text())['checkpoint']}")
        write_json(run.out / "controller_checkpoints.json", checkpoints)

    plants = [("new", model)]
    if not p["smoke"]:
        plants.append(("deployed", ROOT / "models/actuator"))
    for profile in ["standard"] if p["smoke"] else ["standard", "gyro"]:
        for label, plant in plants:
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
                    plant,
                    "--asset",
                    settings["asset_path"],
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

    # Actual EE rotation, not only holding the bucket angle during translation.
    for item in checkpoints[1:] if p["smoke"] else checkpoints:
        name, _, checkpoint = item.partition("=")
        if p["smoke"] and name != "pivot_control":
            continue
        output = run.out / "rotation" / name
        run.step(
            f"rotation_{name}",
            [
                "-m",
                "hydraulic_controller.rotation_benchmark",
                "--checkpoint",
                checkpoint,
                "--out",
                output,
                "--model",
                model,
                "--asset",
                settings["asset_path"],
                "--poses",
                1,
                "--cycles",
                1,
            ],
            artifact=output / "rotation_results.json",
        )
    run.status(
        status="complete", stage="done", child_pid=None, finished=datetime.now().astimezone().isoformat()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
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
                smoke=args.smoke,
                iterations=2 if args.smoke else args.iterations,
                variants=[
                    dict(name=n, tracked_point=point, angular_weight=w, angular_error_weight=e)
                    for n, point, w, e in VARIANTS
                ],
                initialization="All four policies from scratch; same seed and update count; frozen actuator models",
                external_interface="EE pose/twist for every policy; pivot conversion happens inside the controller",
                paper=PAPER,
            )
            write_json(plan_path, plan)
            source = out / "source"
            source.mkdir(exist_ok=True)
            for path in [
                Path(__file__),
                ROOT / "train_controller.py",
                ROOT / "hydraulic_controller/env.py",
                ROOT / "hydraulic_controller/policy.py",
                ROOT / "hydraulic_controller/kinematics.py",
                ROOT / "hydraulic_controller/closed_loop.py",
                ROOT / "hydraulic_controller/rotation_benchmark.py",
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
