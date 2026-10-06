"""Replay the robot's joint PID gains on the learned hydraulic plant, with no simulator launched.

Gains come from the robot's ``control_config.yaml``; ``--gains`` overrides single joints for quick what-ifs.
From the Isaac Lab root::

    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/replay_pid.py --robot_repo ../kaivuriprokkis
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/replay_pid.py --robot_repo ../kaivuriprokkis --gains arm=6,0.5,0

Writes metrics.csv, summary.json, traces.pt and replay.png to ``runs/pid_replay/<time>_<run_name>/``.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_gains(values: list[str]) -> dict[str, tuple[float, float, float]]:
    out = {}
    for value in values:
        joint, _, numbers = value.partition("=")
        parts = [float(x) for x in numbers.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--gains wants joint=kp,ki,kd, got {value!r}")
        out[joint] = tuple(parts)
    return out


def main(argv=None) -> int:
    from dataclasses import asdict

    from hydraulic_controller.closed_loop import metrics, plot_report, run_single, write_report
    from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL
    from hydraulic_controller.pid import JointPIDController, load_robot_gains
    from hydraulic_controller.tasks import PID_JOINTS, TaskConfig

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--robot_repo", type=Path, default=ROOT.parent / "kaivuriprokkis")
    parser.add_argument("--robot", default="jetson")
    parser.add_argument("--gains", nargs="*", default=[], help="override joints: boom=kp,ki,kd ...")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run_name", default="robot_gains")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pid_replay")
    args = parser.parse_args(argv)

    control_yaml = args.robot_repo / "configuration_files/profiles" / args.robot / "control_config.yaml"
    gains = load_robot_gains(control_yaml)
    for joint, (kp, ki, kd) in parse_gains(args.gains).items():
        j = PID_JOINTS.index(joint)
        gains.kp[j], gains.ki[j], gains.kd[j] = kp, ki, kd
    print(f"[INFO] gains from {control_yaml}")
    for j, joint in enumerate(PID_JOINTS):
        print(f"       {joint:6s} kp={gains.kp[j]:g} ki={gains.ki[j]:g} kd={gains.kd[j]:g}")

    cfg = TaskConfig()
    controller = JointPIDController(gains)
    scenarios, traces = run_single(controller, cfg, args.model, DEFAULT_ASSET, args.device, gains.ik_lambda)
    rows = metrics(cfg, scenarios, traces)
    out_dir = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    write_report(out_dir, {"pid": asdict(gains)}, cfg, args.model, rows, traces)
    plot_report(out_dir / "replay.png", cfg, scenarios, traces)

    def fmt(value, spec=".1f"):
        return (
            "  n/a"
            if value is None or (isinstance(value, float) and math.isnan(value))
            else format(value, spec)
        )

    def need(r):
        """Required / achievable speed; '!' marks paths no controller can follow."""
        if r["family"] == "joint_step":
            return "-"
        return f"{r['speed_ratio']:.2f}" + ("" if r["feasible"] else "!")

    nominal = [r for r in rows if r["plant"] == "nominal"]
    print("\nnominal plant, joint scenarios (deg, s, valve travel per s)")
    print(
        f"{'scenario':32s} {'overshoot%':>10s} {'rise':>6s} {'settle':>7s} {'lag':>6s} {'final':>6s} "
        f"{'tailp2p':>7s} {'couple':>6s} {'sat':>5s} {'tailTV':>6s} {'need':>5s}"
    )
    for r in (r for r in nominal if r["family"] != "tip_line"):
        print(
            f"{r['scenario']:32s} {fmt(r['overshoot_pct']):>10s} {fmt(r.get('rise_s'), '.2f'):>6s} "
            f"{fmt(r['settle_s'], '.2f'):>7s} {fmt(r.get('ramp_lag_deg'), '.2f'):>6s} "
            f"{fmt(r['final_err_deg'], '.2f'):>6s} {fmt(r['tail_p2p_deg'], '.2f'):>7s} "
            f"{fmt(r['coupling_deg'], '.2f'):>6s} {fmt(r['saturated_frac'], '.2f'):>5s} "
            f"{fmt(r['tail_valve_travel_per_s'], '.2f'):>6s} {need(r):>5s}"
        )
    print("\nnominal plant, tip lines (mm, deg, s)")
    print(
        f"{'scenario':32s} {'mean':>6s} {'max':>6s} {'lag':>6s} {'oversh':>6s} {'settle':>7s} {'final':>6s} "
        f"{'pitch':>6s} {'sat':>5s} {'need':>5s}"
    )
    for r in (r for r in nominal if r["family"] == "tip_line"):
        print(
            f"{r['scenario']:32s} {fmt(r['mean_err_mm']):>6s} {fmt(r['max_err_mm']):>6s} "
            f"{fmt(r['cruise_lag_mm']):>6s} {fmt(r['end_overshoot_mm']):>6s} {fmt(r['settle_s'], '.2f'):>7s} "
            f"{fmt(r['final_err_mm']):>6s} {fmt(r['max_pitch_err_deg']):>6s} "
            f"{fmt(r['saturated_frac'], '.2f'):>5s} {need(r):>5s}"
        )
    flagged = [r["scenario"] + "@" + r["plant"] for r in rows if r["hit_limit"] or r["invalid"]]
    if flagged:
        print(f"\n[WARN] hit a joint limit or went non-finite: {', '.join(flagged)}")
    print(
        "\nneed = required / achievable speed on the nominal plant (tip lines: solved along the path with the "
        f"bucket angle held, flow sharing included; ramps: single-spool full valve); '!' = beyond "
        f"{cfg.feasible_speed_ratio:g}, so no controller can follow that path on this plant"
    )
    print(f"[INFO] report: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
