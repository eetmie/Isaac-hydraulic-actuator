# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate unchanged 10 Hz-trained pivot policies at faster execution rates."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from hydraulic_controller import DEFAULT_CHECKPOINT
from hydraulic_controller.core import model_fingerprint, sha256
from hydraulic_controller.policy import ControllerPolicy
from training.overnight import ROOT, Run, prevent_sleep, write_json
from training.policy_rate_comparison import aggregate

RATES = (10, 20, 50, 100)
INTRODUCTION = [
    "# Train at 10 Hz, execute at 10/20/50/100 Hz",
    "",
    "The exact same 1499 checkpoint from each of three 10 Hz training seeds is reused at every execution rate. No retraining or output filter is applied.",
    "Sensor and valve histories stay at 100 Hz; the two trained preview slots remain at +100 and +200 ms. Only the command/governor/actor update cadence changes.",
    "Line/circle tests use planned reference preview. Fixed-tip rotations supply zero preview, meaning hold the current command.",
    "Both actuator plants, gyro/stress sensor conditions and the common valve/speed perturbations are evaluated. Reference20 is the packaged, longer-trained 20 Hz policy; PID is the existing robot controller.",
    "Mean +/- sample standard deviation across training seeds. Valve travel per second, per-update jumps, jerk, holding drift and tracking errors should be considered together.",
    "Headline tracking includes infeasible requests and endpoint holds. See [motion-window and feasible-case results by speed](speed_analysis/results.md).",
    "",
    "These are simulation results. The experiment does not validate native inference timing or physical robot jitter, and no policy is deployed.",
]


def execute(run):
    plan = run.plan
    plants = [("new", plan["model"])]
    if not plan["smoke"]:
        plants.append(("deployed", ROOT / "models/actuator"))
    for profile in ("gyro",) if plan["smoke"] else ("gyro", "stress"):
        for seed in plan["seeds"]:
            checkpoint = plan["checkpoints"][str(seed)]
            actors = [f"run{hz}={checkpoint}@{hz}" for hz in RATES]
            actors.append(f"reference20={DEFAULT_CHECKPOINT}")
            for label, model in plants:
                for task in ("lines",) if plan["smoke"] else ("lines", "circles"):
                    tag = f"{profile}_{label}_{task}_seed{seed}"
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
                        tag,
                        "--smoothness",
                    ]
                    if task == "lines" and not plan["smoke"]:
                        cmd += ["--speed_sweep"]
                    if plan["smoke"]:
                        cmd += ["--plants", "nominal", "--speeds_mm_s", 40]
                    run.step(f"benchmark_{tag}", cmd)
                    aggregate(run.out, INTRODUCTION)
    rotations = {}
    for seed in plan["seeds"]:
        for hz in (100,) if plan["smoke"] else RATES:
            for label, model in plants:
                name = f"run{hz}_seed{seed}_{label}"
                directory = run.out / "rotation" / name
                run.step(
                    f"rotation_{name}",
                    [
                        "-m",
                        "hydraulic_controller.rotation_benchmark",
                        "--checkpoint",
                        plan["checkpoints"][str(seed)],
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
                        "--policy_hz",
                        hz,
                        "--smoothness",
                    ],
                    artifact=directory / "rotation_results.json",
                )
                data = json.loads((directory / "rotation_results.json").read_text())
                rotations[name] = dict(
                    tip_max_mm=max(row["tip_max_mm"] for row in data["rows"]),
                    angle_max_deg=max(row["angle_max_deg"] for row in data["rows"]),
                    failures=sum(row["failed"] for row in data["rows"]),
                    **{
                        key + "_worst": max(row[key] for row in data["rows"])
                        for key in ("motion_jerk_rms_m_s3", "valve_step_p99", "valve_step_max")
                    },
                )
                write_json(run.out / "rotation_summary.json", rotations)
    aggregate(run.out, INTRODUCTION)
    run.status(
        status="complete", stage="done", child_pid=None, finished=datetime.now().astimezone().isoformat()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_study", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    (out / "completed").mkdir(parents=True, exist_ok=True)
    with FileLock(out / "runner.lock", timeout=0):
        path = out / "plan.json"
        if path.exists():
            plan = json.loads(path.read_text())
        else:
            source = json.loads((args.source_study / "selected_checkpoints.json").read_text())
            base = json.loads((args.base / "plan.json").read_text())
            model = Path((args.base / "best_actuator.txt").read_text().strip())
            seeds = [42] if args.smoke else [42, 43, 44]
            checkpoints, fingerprints = {}, {}
            for seed in seeds:
                checkpoint = Path(source["checkpoints"][f"rate10_seed{seed}"])
                policy = ControllerPolicy(checkpoint)
                if (
                    policy.settings.policy_hz != 10
                    or policy.lookahead_steps != 2
                    or policy.tracked_point != "pivot"
                ):
                    raise ValueError(f"Unexpected source policy contract: {checkpoint}")
                checkpoints[str(seed)] = str(checkpoint)
                fingerprints[str(seed)] = sha256(checkpoint)
            plan = dict(
                source_study=str(args.source_study.resolve()),
                repository=str(ROOT),
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                training_hz=10,
                execution_rates=list(RATES),
                seeds=seeds,
                checkpoints=checkpoints,
                checkpoint_sha256=fingerprints,
                preview_offsets_s=[0.1, 0.2],
                history_hz=100,
                model=str(model),
                model_files=model_fingerprint(model),
                asset=str(ROOT / "assets/excavator_bucket_rocking.usd"),
                asset_sha256=sha256(ROOT / "assets/excavator_bucket_rocking.usd"),
                robot_repo=base["robot_repo"],
                smoke=args.smoke,
                hardware_validated=False,
            )
            write_json(path, plan)
            shutil.copy2(__file__, out / "runner_source.py")
            (out / "source.patch").write_bytes(
                subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=ROOT)
            )
        for seed, checkpoint in plan["checkpoints"].items():
            if sha256(Path(checkpoint)) != plan["checkpoint_sha256"][seed]:
                raise ValueError(f"Source checkpoint changed for seed {seed}")
        run = Run(out, plan)
        try:
            with prevent_sleep():
                execute(run)
        except BaseException as exc:
            run.status(status="failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
