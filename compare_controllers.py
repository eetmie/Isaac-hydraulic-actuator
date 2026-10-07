"""Compare the robot PID, the sim-tuned PID, the trained MLP policy and the MPC on the same tip tasks.

Tip lines (default) or full circles, under every valve/speed perturbation; only paths the plant can actually
follow count (see ``tasks``). Lines start from HOME and the central working poses, circles from the benchmark's
eight poses. No simulator is launched. From the Isaac Lab root::

    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/compare_controllers.py \\
        --pid_gains runs/pid_tune/<run>/pid_gains.yaml
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/compare_controllers.py --task circles \\
        --pid_gains runs/pid_tune/<run>/pid_gains.yaml --mlp proto=models/controller_proto/model_1798.pt \\
        v6=models/controller_v6/model_1798.pt

Writes summary.json, per_scenario.csv and comparison.png (plus circles.png) to
``runs/compare/<time>_<run_name>/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONTROLLERS = ("pid_robot", "pid_tuned", "mlp", "mpc")
TRACE_KEYS = ("q", "u", "q_target", "tip", "tip_target")  # per-tick traces, [T, N, ...]


def main(argv=None) -> int:
    import math

    import torch
    import yaml

    from hydraulic_controller.benchmark import benchmark_poses
    from hydraulic_controller.comparison import (
        METRICS,
        compare,
        plot_circles,
        plot_report,
        summarize_circles,
        write_report,
    )
    from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.mpc import MPPIConfig, MPPIController
    from hydraulic_controller.pid import JointPIDController, PidGains, load_robot_gains
    from hydraulic_controller.policy import ControllerPolicy, PolicyController
    from hydraulic_controller.tasks import WORKING_STARTS, TaskConfig, prepare

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--robot_repo", type=Path, default=ROOT.parent / "kaivuriprokkis")
    parser.add_argument("--robot", default="jetson")
    parser.add_argument(
        "--pid_gains", type=Path, help="pid: YAML block from tune_pid.py (needed for pid_tuned)"
    )
    parser.add_argument(
        "--mlp",
        nargs="+",
        default=[f"mlp={ROOT / 'models/controller_proto/model_1798.pt'}"],
        metavar="NAME=CHECKPOINT[@HZ]",
        help="one or more policy checkpoints to run as controllers (default: the prototype, named mlp); "
        "@HZ runs one at another policy rate than it was trained at",
    )
    parser.add_argument("--mpc_model", type=Path, default=DEFAULT_MODEL, help="network the MPC plans with")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="network the plant runs")
    parser.add_argument("--sensor_noise_deg", type=float, default=0.02)
    parser.add_argument("--task", choices=("lines", "circles"), default="lines")
    parser.add_argument(
        "--starts", choices=("working", "benchmark"), help="default: working (lines), benchmark (circles)"
    )
    parser.add_argument(
        "--speeds_mm_s", type=float, nargs="+", help="default: 20 40 70 (lines), 20 (circles)"
    )
    parser.add_argument("--plants", nargs="+", help="subset of benchmark plants; default all")
    parser.add_argument(
        "--all_lines",
        action="store_true",
        help="keep lines faster than the plant can follow (static-speed paths); still drop invalid paths",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_name", default="compare")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/compare")
    args = parser.parse_args(argv)

    robot = load_robot_gains(
        args.robot_repo / "configuration_files/profiles" / args.robot / "control_config.yaml"
    )
    controllers = {}
    for name in args.controllers:
        if name == "pid_robot":
            controllers[name] = JointPIDController(robot)
        elif name == "pid_tuned":
            if args.pid_gains is None:
                parser.error("pid_tuned needs --pid_gains")
            block = yaml.safe_load(args.pid_gains.read_text())["pid"]
            joints = [block[f"joint{i}"] for i in (1, 2, 3)]
            tuned = PidGains(
                kp=[float(j["kp"]) for j in joints],
                ki=[float(j["ki"]) for j in joints],
                kd=[float(j["kd"]) for j in joints],
                output_limits=robot.output_limits,
                ik_lambda=robot.ik_lambda,
            )
            controllers[name] = JointPIDController(tuned)
        elif name == "mlp":
            for item in args.mlp:
                label, _, path = item.partition("=")
                path, _, hz = path.partition("@")
                policy = ControllerPolicy(Path(path), args.device, allow_pivot=True)
                controllers[label] = PolicyController(policy, policy_hz=int(hz) if hz else None)
        elif name == "mpc":
            controllers[name] = MPPIController(args.mpc_model, MPPIConfig(seed=args.seed))

    starts = args.starts or ("working" if args.task == "lines" else "benchmark")
    if starts == "working":
        where = {"start_tips": WORKING_STARTS}
    else:
        kin = ExcavatorKinematics(DEFAULT_ASSET, args.device)
        where = {"include_home": False, "start_q": tuple(map(tuple, benchmark_poses(kin, 8).tolist()))}
    if args.task == "lines":
        speeds = tuple(v / 1000 for v in args.speeds_mm_s) if args.speeds_mm_s else TaskConfig.tip_speeds_m_s
        task = {"families": ("tip_line",), "tip_speeds_m_s": speeds}
    else:
        speeds = (
            tuple(v / 1000 for v in args.speeds_mm_s) if args.speeds_mm_s else TaskConfig.circle_speeds_m_s
        )
        radius, accel = TaskConfig.circle_radius_m, TaskConfig.tip_accel_m_s2
        longest = max(2 * math.pi * radius / v + v / accel for v in speeds)
        task = {
            "families": ("tip_circle",),
            "circle_speeds_m_s": speeds,
            "duration_s": math.ceil(longest + 3.0),
        }
    task_cfg = TaskConfig(**where, **task, sensor_noise_deg=args.sensor_noise_deg)
    if args.plants:
        task_cfg = TaskConfig(**{**asdict(task_cfg), "plants": tuple(args.plants)})
    prep = prepare(task_cfg, args.model, DEFAULT_ASSET, args.device, robot.ik_lambda)
    keep = prep.path_valid if args.all_lines else prep.feasible
    print(
        f"[INFO] {int(keep.sum())} of {len(prep.scenarios)} tip {args.task} kept "
        f"({'valid paths, reachable or not' if args.all_lines else 'feasible'})"
    )
    prep = prep.subset(keep)

    summaries, terms, traces = compare(prep, controllers, seed=args.seed)
    out_dir = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    info = {
        "task_config": asdict(task_cfg),
        "scenarios": [s.name + "@" + s.plant for s in prep.scenarios],
        "plant_model": str(args.model),
        "pid_robot": asdict(robot),
        "pid_gains_file": str(args.pid_gains) if args.pid_gains else None,
        "mlp_checkpoints": dict(item.partition("=")[::2] for item in args.mlp),
        "mpc": {"model": str(args.mpc_model), **asdict(MPPIConfig(seed=args.seed))},
    }
    if args.task == "circles":
        info["circles"] = {name: summarize_circles(prep, traces[name]) for name in traces}
    write_report(out_dir, prep, summaries, terms, info)
    plot_report(out_dir / "comparison.png", prep, summaries, traces)
    if args.task == "circles":
        for speed in task_cfg.circle_speeds_m_s:
            rows = torch.tensor([s.size == speed for s in prep.scenarios], device=prep.q0.device)
            one = prep.subset(rows)
            picked = {
                name: {
                    key: (value[:, rows.cpu()] if key in TRACE_KEYS else value) for key, value in tr.items()
                }
                for name, tr in traces.items()
            }
            plot_circles(out_dir / f"circles_{speed * 1000:g}mm_s.png", one, picked, speed * 1000)

    print(f"\n{len(prep.scenarios)} tip {args.task}; nominal plant | worst plant")
    print(f"{'metric':20s}" + "".join(f"{n:>18s}" for n in summaries))
    for metric in METRICS:
        cells = [
            f"{summaries[n]['nominal'][metric]:7.2f} | {summaries[n]['worst_plant'][metric]:7.2f}"
            for n in summaries
        ]
        print(f"{metric:20s}" + "".join(f"{c:>18s}" for c in cells))
    print("\nmean tip error [mm] by commanded speed, all plants")
    speeds = next(iter(summaries.values()))["by_speed_mm_s"]
    print(f"{'mm/s':>8s}" + "".join(f"{n:>12s}" for n in summaries))
    for speed in speeds:
        print(
            f"{speed:>8s}"
            + "".join(f"{summaries[n]['by_speed_mm_s'][speed]['tip_mean_mm']:12.2f}" for n in summaries)
        )
    if args.task == "circles":
        print("\nradial deviation from the circle [mm], nominal plant: mean / worst")
        for name, result in info["circles"].items():
            cells = [
                f"{sense} {result[sense]['nominal']['mean_mm']:.1f} / {result[sense]['nominal']['worst_mm']:.1f}"
                for sense in ("ccw", "cw")
            ]
            print(f"  {name:14s} " + "   ".join(cells))
    print(f"\n[INFO] report: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
