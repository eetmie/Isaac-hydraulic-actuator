"""Run timed actuator phases and PPO variations using an existing Python environment.

Run with the Isaac Lab environment's Python from this repository. A frozen plan,
stage logs, completion records and a live status.json are written below --out.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
DEV = ["v2_20260909", "robot:1792605533932081-71723"]
TEST = [
    "robot:1792664731108427-pid-3968",
    "robot:1792665379525709-5530",
    "robot:1792668759594343-14974",
    "robot:1792676310556861-26765",
]
PHASES = [("a_scratch", 1e-4, 0.45), ("b_low_lr", 3e-5, 0.30), ("c_lowest_lr", 1e-5, 0.25)]
VARIATIONS = [
    ("base_lr3e-4", 3e-4, 0, 0.0, 42),
    ("lookahead_accel_lr3e-4", 3e-4, 4, 0.4, 42),
    ("lookahead_accel_lr1e-4", 1e-4, 4, 0.4, 42),
]


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        for attempt in range(7):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * 2**attempt)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_schedule(plan, out, started):
    """Allocate the original phase budgets only to actuator phases still unfinished."""
    total_weight = sum(phase["weight"] for phase in plan["phases"])
    remaining_weight = sum(
        phase["weight"] for phase in plan["phases"] if not (out / phase["name"] / "selection.json").exists()
    )
    seconds = plan["hours"] * 3600
    actuator_end = started + seconds * plan["actuator_fraction"] * remaining_weight / total_weight
    initial_controller_end = actuator_end + seconds * plan["controller_fraction"]
    controller_end = initial_controller_end + plan["extra_seed_hours"] * 3600
    return dict(
        started=stamp(),
        actuator_end=actuator_end,
        initial_controller_end=initial_controller_end,
        controller_end=controller_end,
        target_finish=controller_end + seconds * plan["evaluation_fraction"],
    )


def reference_controller(out):
    """Stage the historical controller with its exact pinned asset from Git history."""
    source = ROOT / "models/controller"
    contract = json.loads((source / "controller_contract.json").read_text())
    asset = ROOT / contract["asset_path"]
    if hashlib.sha256(asset.read_bytes()).hexdigest() == contract["asset_sha256"]:
        return source / "model_1798.pt"
    reference = out / "reference_controller"
    reference.mkdir(exist_ok=True)
    revisions = subprocess.check_output(
        ["git", "log", "--all", "--format=%H", "--", contract["asset_path"]], cwd=ROOT, text=True
    ).split()
    for revision in revisions:
        blob = subprocess.check_output(["git", "show", f"{revision}:{contract['asset_path']}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() == contract["asset_sha256"]:
            pinned = reference / "original_asset.usd"
            pinned.write_bytes(blob)
            shutil.copy2(source / "model_1798.pt", reference / "model_1798.pt")
            shutil.copytree(source / "params", reference / "params", dirs_exist_ok=True)
            shutil.copy2(source / "controller_contract.json", reference / "controller_contract.original.json")
            contract["asset_path"] = str(pinned)
            write_json(reference / "controller_contract.json", contract)
            write_json(
                reference / "provenance.json",
                dict(
                    source=str(source),
                    asset_commit=revision,
                    original_asset_sha256=contract["asset_sha256"],
                    evaluation_asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
                    note="Unchanged reference weights and asset hash; only the staged contract's asset path is relocated",
                    checkpoint_sha256=hashlib.sha256((source / "model_1798.pt").read_bytes()).hexdigest(),
                ),
            )
            return reference / "model_1798.pt"
    raise ValueError("Cannot find the reference controller's pinned asset in Git history")


@contextmanager
def prevent_sleep():
    """Hold a Windows execution request only for this process's lifetime."""
    if os.name == "nt":
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Cannot hold the Windows system-awake request")
    try:
        yield
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def make_plan(args):
    seconds = args.hours * 3600
    variations = [
        dict(name=n, lr=lr, lookahead=look, accel=accel, seed=seed) for n, lr, look, accel, seed in VARIATIONS
    ]
    phases = [
        dict(
            name=n,
            lr=lr,
            weight=weight,
            estimated_steps=round(seconds * 0.65 * weight / args.actuator_step_seconds / 1000) * 1000,
        )
        for n, lr, weight in PHASES
    ]
    if args.extra_seed_hours:
        variations.append(
            dict(name="lookahead_accel_lr1e-4_seed43", lr=1e-4, lookahead=4, accel=0.4, seed=43)
        )
    return dict(
        version=1,
        created=stamp(),
        python=sys.executable,
        repository=str(ROOT),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        dataset=str(args.dataset.resolve()),
        dataset_revision=args.dataset_revision,
        dataset_manifest_sha256=hashlib.sha256((args.dataset / "dataset.json").read_bytes()).hexdigest(),
        hours=args.hours,
        extra_seed_hours=args.extra_seed_hours,
        actuator_fraction=0.65,
        controller_fraction=0.30,
        evaluation_fraction=0.05,
        actuator_step_seconds=args.actuator_step_seconds,
        ppo_iteration_seconds=args.ppo_iteration_seconds,
        phases=phases,
        variations=variations,
        estimated_initial_ppo_iterations_each=round(seconds * 0.30 / 3 / args.ppo_iteration_seconds),
        pretrain_steps=200 if args.smoke else 20000,
        eval_every=10 if args.smoke else 1000,
        eval_cases=8 if args.smoke else 96,
        compare_cases=8 if args.smoke else 200,
        num_envs=256 if args.smoke else 4096,
        smoke=args.smoke,
        robot_repo=str(args.robot_repo.resolve()),
    )


class Run:
    def __init__(self, out, plan):
        self.out, self.plan = out, plan
        self.state = dict(status="starting", pid=os.getpid(), started=stamp())
        self.env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        self.start = time.time()
        self.status_retry_after = 0.0

    def status(self, **fields):
        self.state.update(fields, updated=stamp())
        if time.monotonic() < self.status_retry_after and fields.get("status") not in ("failed", "complete"):
            return
        try:
            write_json(self.out / "status.json", self.state)
            self.status_retry_after = 0.0
        except OSError as exc:
            # Monitoring output must never abort the training queue or wait on its child during unwinding.
            self.status_retry_after = time.monotonic() + 60
            print(
                f"{stamp()} status update unavailable; training continues: {exc}", file=sys.stderr, flush=True
            )

    def step(self, name, arguments, *, artifact=None):
        done = self.out / "completed" / f"{name}.json"
        if done.exists():
            if artifact is not None and not artifact.exists():
                raise FileNotFoundError(f"Completed stage {name} lost its artifact: {artifact}")
            print(f"{stamp()} skip completed {name}", flush=True)
            return
        log = self.out / f"{name}.log"
        command = [sys.executable, "-u", *map(str, arguments)]
        print(f"{stamp()} {name} (log: {log})", flush=True)
        started = time.time()
        self.status(status="running", stage=name, stage_started=stamp(), log=str(log), command=command)
        with log.open("ab") as stream:
            stream.write(f"\n{stamp()} {subprocess.list2cmdline(command)}\n".encode())
            stream.flush()
            with subprocess.Popen(
                command, cwd=ROOT, env=self.env, stdout=stream, stderr=subprocess.STDOUT
            ) as child:
                self.status(child_pid=child.pid)
                while child.poll() is None:
                    self.status(stage_elapsed_s=round(time.time() - started))
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
        if code:
            raise RuntimeError(f"{name} exited {code}; see {log}")
        if artifact is not None and not artifact.exists():
            raise FileNotFoundError(f"{name} did not produce {artifact}; see {log}")
        write_json(done, dict(finished=stamp(), elapsed_s=time.time() - started, command=command))

    def benchmark(self, checkpoints, best, tag):
        deployed = ROOT / "models/actuator"
        plants = [("deployed", deployed)]
        if best.resolve() != deployed.resolve():
            plants.append(("new", best))
        for label, plant in plants:
            for task in ("lines", "circles"):
                command = [
                    ROOT / "compare_controllers.py",
                    "--controllers",
                    "pid_robot",
                    "mlp",
                    "--robot_repo",
                    self.plan["robot_repo"],
                    "--mlp",
                    *checkpoints,
                    "--model",
                    plant,
                    "--task",
                    task,
                    "--out",
                    self.out / "benchmarks",
                    "--run_name",
                    f"{tag}_{label}_{task}",
                ]
                if self.plan["smoke"]:
                    command += ["--plants", "nominal", "--speeds_mm_s", 40]
                self.step(f"benchmark_{tag}_{label}_{task}", command)
        self.report()

    def report(self):
        lines = [f"# Training results\n\nUpdated {stamp()}.\n"]
        a = self.out / self.plan["phases"][0]["name"]
        for split in ("dev", "test"):
            metrics = a / f"compare_{split}_{self.plan['compare_cases']}.json"
            if not metrics.exists():
                continue
            lines += [
                f"\n## Actuators: {split}\n",
                "| Model | 10s path MAE (deg) | 30s endpoint MAE (deg) | 10s rest drift (deg) |",
                "| --- | ---: | ---: | ---: |",
            ]
            for name, row in json.loads(metrics.read_text()).items():
                lines.append(
                    f"| {name} | {row['10']['path_deg']:.3f} | {row['30']['endpoint_deg']:.3f} | {row['rest10_deg']:.3f} |"
                )
        lines.append(
            "\nController tables include all benchmark paths and perturbations, including infeasible paths.\n"
        )
        for path in sorted((self.out / "benchmarks").glob("*/summary.json")):
            relative = path.relative_to(self.out).as_posix()
            lines += [
                f"\n## [{path.parent.name}]({relative})\n",
                "| Controller | Mean tip error (mm) | Bucket pitch MAE (deg) | p95 path max (mm) | Valve travel / s | Failures |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for name, groups in json.loads(path.read_text())["summaries"].items():
                row = groups["all"]
                lines.append(
                    f"| {name} | {row['tip_mean_mm']:.2f} | {row['pitch_mean_deg']:.2f} | {row['tip_p95_max_mm']:.2f} | "
                    f"{row['valve_travel_per_s']:.2f} | {row['failures']:.3f} |"
                )
        (self.out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def execute(self):
        p = self.plan
        reference = reference_controller(self.out)
        if not (Path(p["robot_repo"]) / "configuration_files/profiles/jetson/control_config.yaml").exists():
            raise FileNotFoundError("Robot PID configuration is required for the final comparisons")
        manifest = Path(p["dataset"]) / "dataset.json"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != p["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest changed since this plan was frozen")
        marker = self.out / "schedule.json"
        if marker.exists():
            schedule = json.loads(marker.read_text())
        else:
            schedule = make_schedule(p, self.out, self.start)
            write_json(marker, schedule)
        self.status(schedule=schedule)
        self.step(
            "gpu", ["-c", "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"]
        )
        a = self.out / p["phases"][0]["name"]
        if not (a / "split_manifest.json").exists():
            self.step(
                "prepare",
                [
                    "-m",
                    "training.prepare",
                    "--dataset",
                    p["dataset"],
                    "--out",
                    a,
                    "--dev-groups",
                    *DEV,
                    "--test-groups",
                    *TEST,
                ],
                artifact=a / "split_manifest.json",
            )
        previous = ROOT / "models/actuator"
        for index, phase in enumerate(p["phases"]):
            out = self.out / phase["name"]
            out.mkdir(exist_ok=True)
            if index and not (out / "split_manifest.json").exists():
                shutil.copytree(a / "data", out / "data", dirs_exist_ok=True)
                shutil.copy2(a / "split_manifest.json", out / "split_manifest.json")
            if not (out / "selection.json").exists():
                weight_left = sum(v["weight"] for v in p["phases"][index:])
                seconds = max(60, (schedule["actuator_end"] - time.time()) * phase["weight"] / weight_left)
                # A generous iteration cap lets the clock, rather than timing noise, allocate the phase.
                steps = (
                    20 if p["smoke"] else math.ceil(seconds / p["actuator_step_seconds"] * 1.5 / 1000) * 1000
                )
                command = [
                    "-m",
                    "training.actuator",
                    "train",
                    "--out",
                    out,
                    "--model",
                    previous,
                    "--steps",
                    steps,
                    "--lr",
                    phase["lr"],
                    "--batch",
                    256,
                    "--eval_every",
                    p["eval_every"],
                    "--cases",
                    p["eval_cases"],
                ]
                command += ["--max_seconds", 1 if p["smoke"] else round(seconds)]
                if not index:
                    command += ["--scratch", "--pretrain_steps", p["pretrain_steps"]]
                self.step(phase["name"], command, artifact=out / "selection.json")
            previous = out / "model/latest"

        models = [f"deployed={ROOT / 'models/actuator'}"]
        for phase in p["phases"]:
            out = self.out / phase["name"]
            steady = out / "model_steady"
            self.step(
                f"bound_{phase['name']}",
                ["-m", "training.neutral_bound", out / "model", steady],
                artifact=steady / "model_meta.json",
            )
            models.append(f"{phase['name']}={steady}")
        self.step(
            "compare_dev",
            [
                "-m",
                "training.compare_models",
                a,
                *models,
                "--split",
                "dev",
                "--cases",
                p["compare_cases"],
                "--best_out",
                self.out / "best_actuator.txt",
            ],
            artifact=self.out / "best_actuator.txt",
        )
        self.step(
            "compare_test",
            ["-m", "training.compare_models", a, *models, "--split", "test", "--cases", p["compare_cases"]],
        )
        best = Path((self.out / "best_actuator.txt").read_text().strip())
        deployed = ROOT / "models/actuator"
        checkpoints = [f"deployed={reference}"]
        for index, variation in enumerate(p["variations"]):
            name = variation["name"]
            log_dir = self.out / "controllers" / name
            if not (log_dir / "training_result.json").exists():
                if index < 3:
                    seconds = max(60, (schedule["initial_controller_end"] - time.time()) / (3 - index) - 30)
                else:
                    seconds = max(60, schedule["controller_end"] - time.time() - 30)
                iterations = (
                    2 if p["smoke"] else math.ceil(seconds / p["ppo_iteration_seconds"] * 1.5 / 100) * 100
                )
                if index == 3 and not p["smoke"]:
                    paired = self.out / "controllers/lookahead_accel_lr1e-4/training_result.json"
                    iterations = json.loads(paired.read_text())["iteration"] + 1
                settings = [f"lookahead_steps={variation['lookahead']}", f"accel_weight={variation['accel']}"]
                if best.resolve() != deployed.resolve():
                    settings.append(f"extra_model_paths=[{deployed.as_posix()}]")
                command = [
                    ROOT / "train_controller.py",
                    "--model",
                    best,
                    "--run_name",
                    name,
                    "--log_dir",
                    log_dir,
                    "--max_iterations",
                    iterations,
                    "--num_envs",
                    p["num_envs"],
                    "--learning_rate",
                    variation["lr"],
                    "--lr_schedule",
                    "fixed",
                    "--seed",
                    variation["seed"],
                    "--save_interval",
                    1 if p["smoke"] else 100,
                    "--set",
                    *settings,
                ]
                command += ["--max_seconds", 0.001 if p["smoke"] else round(seconds)]
                # An interrupted PPO stage resumes from its last saved optimizer state.
                saved = sorted(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))
                if saved:
                    command += ["--resume", saved[-1]]
                self.step(name, command, artifact=log_dir / "training_result.json")
            result = json.loads((log_dir / "training_result.json").read_text())
            checkpoints.append(f"{name}={result['checkpoint']}")
            write_json(self.out / "controller_checkpoints.json", checkpoints)
            if index == 2:
                # Leave comparisons for the initial three before the optional extra seed starts.
                self.benchmark(checkpoints, best, "initial")
        write_json(self.out / "controller_checkpoints.json", checkpoints)
        if len(p["variations"]) > 3:
            self.benchmark(checkpoints, best, "final")
        self.status(status="complete", stage="done", child_pid=None, finished=stamp())
        print(f"{stamp()} done: {self.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/overnight")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/complete_dataset")
    parser.add_argument("--dataset_revision", default="local")
    parser.add_argument("--robot_repo", type=Path, default=ROOT.parent / "kaivuriprokkis")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--extra_seed_hours", type=float, default=0)
    parser.add_argument("--actuator_step_seconds", type=float, default=0.2)
    parser.add_argument("--ppo_iteration_seconds", type=float, default=1.05)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()
    if (
        min(args.hours, args.actuator_step_seconds, args.ppo_iteration_seconds) <= 0
        or args.extra_seed_hours < 0
    ):
        parser.error("Hours and timing rates must be positive; extra seed hours must be nonnegative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "completed").mkdir(exist_ok=True)
    with FileLock(args.out / "runner.lock", timeout=0):
        plan_path = args.out / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            plan = make_plan(args)
            write_json(plan_path, plan)
            patch = subprocess.check_output(
                ["git", "diff", "HEAD", "--"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            (args.out / "source.patch").write_bytes(patch)
            # Include untracked orchestration code in the run's source snapshot too.
            shutil.copy2(__file__, args.out / "overnight_source.py")
        if args.plan_only:
            print(json.dumps(plan, indent=2))
            return
        run = Run(args.out, plan)
        try:
            with prevent_sleep():
                run.execute()
        except BaseException as exc:
            run.status(status="failed", error=str(exc), finished=stamp())
            raise


if __name__ == "__main__":
    main()
