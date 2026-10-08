"""Export a trained controller for the robot, and generate the robot profile it runs against.

The robot side lives in kaivuriprokkis ``learned_control``; it needs only the bundle and this profile. From the
Isaac Lab root::

    # Once per calibration: a bucket profile whose geometry comes from the controller USD
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/export_robot_bundle.py profile \\
        --source_profile ../kaivuriprokkis/configuration_files/profiles/jetson --robot jetson_bucket \\
        --out ../kaivuriprokkis/configuration_files/profiles/jetson_bucket
    # Per controller
    isaaclab.sh -p scripts/Isaac-hydraulic-actuator/export_robot_bundle.py export \\
        --checkpoint models/controller_v6/model_1798.pt --robot jetson_bucket \\
        --out ../kaivuriprokkis/learned_control/bundles/v6_jetson_bucket
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

NOTES = (
    "# Bucket cutting tip and link vectors extracted from the controller USD.\n"
    "# Origin: slew bearing; boom pivot Z = 78.5 mm, confirmed by production CAD.\n"
    "# Joint angles remain relative IMU angles; do not add fixed USD rotations to them.\n"
    "# pitch_offset_rad defines the cutting lip, not the final joint-frame orientation.\n"
    "# CAD IMU magnitudes: liftboom 13.832 deg; tiltboom 0.612 deg. Signs unverified.\n"
    "# Active mounting quaternions copied from the source robot profile.\n"
    "# Actor histories are translated back to training joint zeros; PID/FK use runtime zeros.\n"
)


def profile(args) -> None:
    """Write a robot profile whose arm geometry is the controller USD's; IMU/PWM calibration is copied."""
    import yaml

    from hydraulic_controller.core import DEFAULT_ASSET
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.robot_geometry import profile_from_usd

    source = yaml.safe_load((args.source_profile / "control_config.yaml").read_text())
    configuration = profile_from_usd(ExcavatorKinematics(args.asset or DEFAULT_ASSET), copy.deepcopy(source))
    configuration["robot"]["tool"]["name"] = "bucket"
    configuration["robot"]["coordinate_origin"] = "slew_bearing"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "control_config.yaml").write_text(NOTES + yaml.safe_dump(configuration, sort_keys=False))
    (args.out / "servo_config.yaml").write_bytes((args.source_profile / "servo_config.yaml").read_bytes())
    board = yaml.safe_load((args.source_profile / "profile.yaml").read_text())
    board.update(
        name=args.robot, servo_config_file="servo_config.yaml", control_config_file="control_config.yaml"
    )
    (args.out / "profile.yaml").write_text(yaml.safe_dump(board, sort_keys=False))
    print(f"Wrote {args.out}; commit it to kaivuriprokkis as the {args.robot} profile.")


def export(args) -> None:
    from hydraulic_controller.bundle import export_bundle
    from hydraulic_controller.policy import ControllerPolicy

    profile_dir = args.robot_repo / "configuration_files/profiles" / args.robot
    directory = export_bundle(ControllerPolicy(args.checkpoint), args.out, profile_dir)
    print(f"Exported {directory.resolve()} for {profile_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="mode", required=True)
    gen = commands.add_parser("profile", help="Generate a USD-geometry robot profile from a calibrated one")
    gen.add_argument("--source_profile", type=Path, required=True)
    gen.add_argument("--robot", required=True, help="Name of the new profile")
    gen.add_argument("--asset", type=Path, help="Controller USD (default: the training asset)")
    gen.add_argument("--out", type=Path, required=True)
    exp = commands.add_parser("export", help="Export a checkpoint as a robot bundle")
    exp.add_argument("--checkpoint", type=Path, required=True)
    exp.add_argument("--robot_repo", type=Path, default=ROOT.parent / "kaivuriprokkis")
    exp.add_argument("--robot", default="jetson_bucket", help="Generated profile the bundle is pinned to")
    exp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"profile": profile, "export": export}[args.mode](args)


if __name__ == "__main__":
    main()
