"""Prepare a local fixed-tip bundle and replay it with the frozen legacy Python runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from export_robot_bundle import profile
from hydraulic_controller.bundle import export_bundle
from hydraulic_controller.policy import ControllerPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--robot_repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    policy = ControllerPolicy(args.checkpoint, "cpu")
    generated = args.out / "generated_profile"
    if not generated.exists():
        profile(
            SimpleNamespace(
                source_profile=args.robot_repo / "configuration_files/profiles/jetson",
                asset=policy.asset_path,
                out=generated,
                robot="comparison_bucket",
            )
        )
    directory = args.out / "bundle"
    if not directory.exists():
        export_bundle(policy, directory, generated)
    sys.path.insert(0, str(args.robot_repo))
    from learned_control.bundle import PolicyBundle

    bundle = PolicyBundle(directory)
    bundle.check_profile(generated)
    assert bundle.verified
    report = dict(
        verified=True,
        checkpoint=str(args.checkpoint),
        bundle=str(directory),
        runtime="Frozen legacy Python learned_control reference; current native integration is separate",
        hardware_validated=False,
        reference_cases=64,
    )
    (args.out / "export_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
