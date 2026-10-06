"""Map the tip speeds the learned plant can actually reach, before tuning any controller.

No simulator is launched. From the Isaac Lab root::

    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/measure_speeds.py
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/measure_speeds.py --plant machine_20pct_slower

Writes speed_map.json/.pt, speed_map.png and valve_curves.png to ``runs/speed_limits/<time>_<plant>/``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    import torch

    from hydraulic_controller.benchmark import PLANTS
    from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.speed_limits import (
        PID_JOINTS,
        SpeedMapConfig,
        direction_name,
        measure_speed_map,
        write_report,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--plant", choices=tuple(PLANTS), default="nominal")
    parser.add_argument("--spacing", type=float, default=0.05, help="workspace grid spacing [m]")
    parser.add_argument("--pitch_deg", type=float, help="held bucket angle [deg]; default the HOME angle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/speed_limits")
    args = parser.parse_args(argv)

    cfg = SpeedMapConfig(
        spacing=args.spacing,
        plant=args.plant,
        pitch=None if args.pitch_deg is None else math.radians(args.pitch_deg),
    )
    kin = ExcavatorKinematics(DEFAULT_ASSET, args.device)
    start = time.perf_counter()
    result = measure_speed_map(args.model, kin, cfg, args.device)
    elapsed = time.perf_counter() - start
    out_dir = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.plant}"
    write_report(out_dir, result, cfg, args.model)

    poses = int(result["valid"].sum())
    print(
        f"[INFO] {poses} workspace poses x {cfg.random_valves + len(cfg.valve_levels) ** 3} valve combinations, "
        f"plant {args.plant}, {elapsed:.1f} s"
    )
    print("\nsingle spool at HOME: full-valve speed [deg/s] / deadband edge [u]")
    for joint in PID_JOINTS:
        neg, pos = result["curves"][f"{joint}_neg"], result["curves"][f"{joint}_pos"]
        print(
            f"  {joint:6s} neg {neg['full_valve_deg_s']:6.1f} / {neg['deadband_u']:.2f}    "
            f"pos {pos['full_valve_deg_s']:6.1f} / {pos['deadband_u']:.2f}"
        )

    print("\nachievable tip speed [mm/s], bucket angle held")
    print(f"  {'direction':9s} {'HOME':>6s} {'min':>6s} {'median':>7s} {'max':>6s} {'vs 1-joint bound':>17s}")
    for i, degrees in enumerate(cfg.directions_deg):
        speed = result["tip_speed"][:, :, i][result["valid"]] * 1000
        ratio = (result["tip_speed"][:, :, i] / result["single_joint_bound"][:, :, i])[result["valid"]]
        print(
            f"  {direction_name(degrees):9s} {float(result['home_tip_speed'][i]) * 1000:6.1f} "
            f"{float(speed.min()):6.1f} {float(speed.median()):7.1f} {float(speed.max()):6.1f} "
            f"{float(torch.nanmedian(ratio)):16.2f}x"
        )
    print(
        "\nmin/median/max over the valid workspace grid; 'vs 1-joint bound' is the median share of the speed "
        "each joint alone could give, i.e. what flow sharing leaves"
    )
    print(f"[INFO] report: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
