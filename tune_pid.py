"""Tune the robot's boom/arm/bucket PID gains with CMA-ES on the learned plant, with no simulator launched.

Starts from the robot's ``control_config.yaml`` gains and writes the tuned ones as a YAML ``pid:`` block that
drops into the same file. From the Isaac Lab root::

    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/tune_pid.py --robot_repo ../kaivuriprokkis
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/tune_pid.py --generations 10 --popsize 16  # quick look

Writes pid_gains.yaml, result.json, history.csv, convergence.png and before/after replays (the replay_pid.py
report) to ``runs/pid_tune/<time>_<run_name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def pid_yaml(gains) -> str:
    lines = ["pid:"]
    for i, joint in enumerate(("boom", "arm", "bucket"), start=1):
        lines += [
            f"  # Joint {i}: {joint.capitalize()} (tuned in sim)",
            f"  joint{i}:",
            f"    kp: {gains.kp[i - 1]:.3f}",
            f"    ki: {gains.ki[i - 1]:.3f}",
            f"    kd: {gains.kd[i - 1]:.3f}",
        ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from hydraulic_controller.closed_loop import metrics, plot_report, run_single, write_report
    from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL
    from hydraulic_controller.pid import JointPIDController, load_robot_gains
    from hydraulic_controller.pid_tuning import TuneConfig, breakdown, tune
    from hydraulic_controller.tasks import PID_JOINTS, WORKING_STARTS, TaskConfig, prepare

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot_repo", type=Path, default=ROOT.parent / "kaivuriprokkis")
    parser.add_argument("--robot", default="jetson")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=32)
    # White noise on the joint angles at 100 Hz. Default: the frame-to-frame jitter of the new-IMU digging
    # recordings while the machine stands still (0.007-0.02 deg). Their slower 0.1-0.2 deg wander is drift or
    # creep that a PID follows rather than chatters on, so it is left out.
    parser.add_argument("--sensor_noise_deg", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_name", default="tune")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pid_tune")
    args = parser.parse_args(argv)

    control_yaml = args.robot_repo / "configuration_files/profiles" / args.robot / "control_config.yaml"
    base = load_robot_gains(control_yaml)
    tune_cfg = TuneConfig(popsize=args.popsize, generations=args.generations, seed=args.seed)
    task_cfg = TaskConfig(start_tips=WORKING_STARTS, sensor_noise_deg=args.sensor_noise_deg)
    out_dir = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    prep = prepare(task_cfg, args.model, DEFAULT_ASSET, args.device, base.ik_lambda)
    keep = prep.feasible
    print(
        f"[INFO] {int(keep.sum())} of {len(prep.scenarios)} scenarios are feasible and kept "
        f"({len(WORKING_STARTS) + 1} start poses x {len(task_cfg.plants)} plants), "
        f"prepared in {time.perf_counter() - start:.0f} s"
    )
    prep = prep.subset(keep)

    start = time.perf_counter()
    result = tune(prep, base, tune_cfg)
    tuned = result["tuned"]
    print(f"[INFO] {args.generations} generations in {time.perf_counter() - start:.0f} s")

    held_seed = 10_000 + args.seed
    base_score, base_table = breakdown(prep, base, tune_cfg, held_seed)
    tuned_score, tuned_table = breakdown(prep, tuned, tune_cfg, held_seed)

    print("\ngains        " + "  ".join(f"{j:>18s}" for j in PID_JOINTS))
    for name, gains in (("robot", base), ("tuned", tuned)):
        print(
            f"{name:12s} "
            + "  ".join(f"{gains.kp[j]:6.3g}/{gains.ki[j]:5.3g}/{gains.kd[j]:5.3g}" for j in range(3))
        )
    print(f"\nheld-out score (fresh noise): robot {base_score:.3f} -> tuned {tuned_score:.3f}")
    print("mean cost terms, nominal plant | worst plant (deg-equivalent; valve_travel per s)")
    print(f"  {'family':11s} {'term':13s} {'robot':>15s} {'tuned':>15s}")
    for (family, key), value in base_table.items():
        if key == "invalid" and value["worst"] == 0 and tuned_table[(family, key)]["worst"] == 0:
            continue
        after = tuned_table[(family, key)]
        print(
            f"  {family:11s} {key:13s} {value['nominal']:6.2f} | {value['worst']:6.2f} "
            f"{after['nominal']:6.2f} | {after['worst']:6.2f}"
        )

    (out_dir / "pid_gains.yaml").write_text(pid_yaml(tuned))
    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["history"][0]))
        writer.writeheader()
        writer.writerows(result["history"])
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "robot_gains": asdict(base),
                "tuned_gains": asdict(tuned),
                "baseline_cost": result["baseline_cost"],
                "best_cost": result["best_cost"],
                "held_out": {"robot": base_score, "tuned": tuned_score},
                "terms": {
                    f"{family}:{key}": {
                        "robot": base_table[(family, key)],
                        "tuned": tuned_table[(family, key)],
                    }
                    for family, key in base_table
                },
                "scenarios": [s.name + "@" + s.plant for s in prep.scenarios],
                "task_config": asdict(task_cfg),
                "tune_config": asdict(tune_cfg),
                "model": str(args.model),
                "control_yaml": str(control_yaml),
            },
            indent=1,
        )
    )
    plot_convergence(out_dir / "convergence.png", result)

    # The standard HOME replay, noise-free, before and after: the same report replay_pid.py writes.
    plain = TaskConfig()
    for name, gains in (("robot", base), ("tuned", tuned)):
        scenarios, traces = run_single(
            JointPIDController(gains), plain, args.model, DEFAULT_ASSET, args.device, gains.ik_lambda
        )
        rows = metrics(plain, scenarios, traces)
        write_report(out_dir / f"replay_{name}", {"pid": asdict(gains)}, plain, args.model, rows, traces)
        plot_report(out_dir / f"replay_{name}" / "replay.png", plain, scenarios, traces)

    print(f"\n{pid_yaml(tuned)}")
    print(f"[INFO] report: {out_dir}")
    return 0


def plot_convergence(path: Path, result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = result["history"]
    gen = [h["generation"] for h in history]
    fig, (ax, gx) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax.plot(gen, [h["median"] for h in history], label="generation median")
    ax.plot(gen, [h["best"] for h in history], label="generation best")
    ax.plot(gen, [h["best_so_far"] for h in history], "k", label="best so far")
    ax.axhline(result["baseline_cost"], color="gray", ls="--", label="robot gains")
    ax.set(xlabel="generation", ylabel="cost", yscale="log", title="CMA-ES")
    ax.legend()
    for name in [k for k in history[0] if k.endswith(("_kp", "_ki", "_kd"))]:
        gx.plot(gen, [h[name] for h in history], label=name, ls="-" if name.endswith("kp") else ":")
    gx.set(xlabel="generation", ylabel="gain (generation best)", yscale="log", title="gains")
    gx.legend(fontsize=7, ncol=3)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
