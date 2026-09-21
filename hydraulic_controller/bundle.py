# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Portable actor + normalization + geometry; runtime needs only NumPy and PyTorch."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .core import ControllerSettings, sha256
from .kinematics import CommandGovernor, ExcavatorKinematics
from .observations import MeasuredHistory
from .robot_geometry import RobotKinematics, profile_from_usd
from .sensors import SENSOR_CONTRACT


def export_bundle(policy, directory: Path, profile_dir: Path):
    """Export a self-contained inference bundle, pinning robot calibration/configuration files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    geometry = ExcavatorKinematics(policy.asset_path, "cpu")
    geometry.build_collision_grid()
    np.save(directory / "collision_grid.npy", geometry._grid.cpu().numpy(), allow_pickle=False)
    import yaml

    profile = profile_from_usd(geometry, yaml.safe_load((profile_dir / "control_config.yaml").read_text()))
    RobotKinematics(profile, geometry)  # Validate the physical geometry before exporting.
    (directory / "robot_geometry.json").write_text(json.dumps(profile, indent=2) + "\n")
    (directory / "bucket_control_config.yaml").write_text(
        "# Generated from the controller's USD bucket. Origin: lower carriage.\n"
        + yaml.safe_dump(profile, sort_keys=False)
    )
    (directory / "geometry.json").write_text(json.dumps(geometry.to_dict(), indent=2) + "\n")
    torch.jit.script(copy.deepcopy(policy.inference).cpu()).save(str(directory / "actor.pt"))
    meta = json.loads((policy.model_path / "model_meta.json").read_text())
    contract = copy.deepcopy(policy.contract)
    # Absolute source paths are provenance only, not runtime requirements.
    contract.pop("asset_path", None)
    contract.pop("model_path", None)
    contract.update(
        bundle_version=2,
        checkpoint_sha256=sha256(policy.checkpoint),
        sensors=SENSOR_CONTRACT,
        sensor_trained="sensor_randomization" in policy.contract,
        velocity_samples=(meta["hist_qdot"] - 1) * meta.get("qdot_stride", 1) + 1,
        command_samples=(meta["hist_u"] - 1) * meta.get("u_stride", 1) + 1,
        u_stride=meta.get("u_stride", 1),
    )
    contract["robot_profile_sha256"] = {
        name: sha256(profile_dir / name)
        for name in ("servo_config.yaml", "control_config.yaml", "profile.yaml")
    }
    contract["geometry_source"] = "training_usd_bucket"
    contract["files"] = {
        name: sha256(directory / name)
        for name in (
            "actor.pt",
            "geometry.json",
            "robot_geometry.json",
            "bucket_control_config.yaml",
            "collision_grid.npy",
        )
    }
    (directory / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return directory


class PolicyBundle:
    """Deterministic CPU actor; ``valves`` includes the single required tanh transform."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.directory = directory.resolve()
        self.contract = c = json.loads((directory / "contract.json").read_text())
        if c["bundle_version"] != 2 or c["action_transform"] != "tanh" or c["sensors"] != SENSOR_CONTRACT:
            raise ValueError("Unsupported hardware contract")
        if set(c["robot_profile_sha256"]) != {"servo_config.yaml", "control_config.yaml", "profile.yaml"}:
            raise ValueError("Bundle must pin calibration and board configuration")
        for name in (
            "actor.pt",
            "geometry.json",
            "robot_geometry.json",
            "bucket_control_config.yaml",
            "collision_grid.npy",
        ):
            if sha256(directory / name) != c["files"][name]:
                raise ValueError(f"Bundle file changed: {name}")
        self.actor = torch.jit.load(str(directory / "actor.pt"), map_location="cpu").eval()
        self.kin = ExcavatorKinematics.from_dict(json.loads((directory / "geometry.json").read_text()))
        grid = np.load(directory / "collision_grid.npy", allow_pickle=False)
        if grid.dtype != np.bool_ or grid.ndim != 3 or len(set(grid.shape)) != 1 or grid.shape[0] < 2:
            raise ValueError("Invalid collision lookup grid")
        self.kin._grid = torch.from_numpy(grid.copy())
        self.robot_kin = RobotKinematics(
            json.loads((directory / "robot_geometry.json").read_text()), self.kin
        )
        self.settings = ControllerSettings(**c["settings"])
        self.history = MeasuredHistory(1, "cpu", c["velocity_samples"], c["command_samples"], c["u_stride"])
        self.governor = CommandGovernor(
            self.kin,
            self.settings,
            None if "joint_speed_limits_rad_s" not in c else torch.tensor(c["joint_speed_limits_rad_s"]),
            c.get("governor_joint_speed_margin", 0.8),
        )

    @torch.inference_mode()
    def valves(self, observations):
        """Return boom/arm/bucket normalized valve values [-1, 1] from a finite observation vector."""
        if (
            observations.shape != (1, self.contract["observation_dim"])
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("Invalid hardware observation")
        action = self.actor(observations)
        if action.shape != (1, 3) or not torch.isfinite(action).all():
            raise ValueError("Invalid actor output")
        return torch.tanh(action)

    def check_profile(self, profile_dir):
        """Refuse changed IMU/geometry/PWM calibration before accessing hardware."""
        for name, fingerprint in self.contract["robot_profile_sha256"].items():
            if sha256(Path(profile_dir) / name) != fingerprint:
                raise ValueError(f"Robot configuration changed since export: {name}")
