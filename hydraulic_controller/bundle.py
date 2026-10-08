# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Export a self-contained robot bundle: actor, geometry, collision grid, contract and reference cases.

The robot runs bundles with kaivuriprokkis ``learned_control`` (NumPy and PyTorch only). The bundle pins the
robot profile it was exported for, and ``reference_io.npz`` holds inputs and outputs computed here with the
training code; the robot replays them at load and refuses a bundle its runtime does not reproduce.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from .core import HOME, ControllerSettings, sha256
from .kinematics import CommandGovernor, ExcavatorKinematics
from .observations import MeasuredHistory
from .robot_geometry import RobotKinematics, profile_from_usd
from .sensors import SENSOR_CONTRACT, policy_joint_offset

FILES = (
    "actor.pt",
    "geometry.json",
    "robot_geometry.json",
    "bucket_control_config.yaml",
    "collision_grid.npy",
)
REFERENCE = "reference_io.npz"


def export_bundle(policy, directory: Path, profile_dir: Path, cases: int = 64):
    """Export ``policy`` for the robot profile in ``profile_dir``, which must be generated from its USD.

    Generate that profile with ``export_robot_bundle.py profile`` when the robot calibration changes.
    """
    import yaml

    directory, profile_dir = Path(directory), Path(profile_dir)
    if policy.contract.get("tracked_point", "tip") != "tip":
        raise ValueError("The robot runtime sends bucket-tip requests; export a tip-tracking policy")
    config = yaml.safe_load((profile_dir / "control_config.yaml").read_text())
    geometry = ExcavatorKinematics(policy.asset_path, "cpu")
    geometry.build_collision_grid()
    if not same_profile(profile_from_usd(geometry, copy.deepcopy(config)), config):
        raise ValueError(
            f"{profile_dir} is not generated from this policy's USD; run export_robot_bundle.py profile first"
        )
    profile = config  # the robot compares its profile to the bundle exactly
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "collision_grid.npy", geometry._grid.cpu().numpy(), allow_pickle=False)
    (directory / "robot_geometry.json").write_text(json.dumps(profile, indent=2) + "\n")
    (directory / "bucket_control_config.yaml").write_bytes((profile_dir / "control_config.yaml").read_bytes())
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
        geometry_source="training_usd_bucket",
        policy_joint_offset_rad=policy_joint_offset(config["imu"]).tolist() if "imu" in config else [0.0] * 4,
    )
    contract["robot_profile_sha256"] = {
        name: sha256(profile_dir / name)
        for name in ("servo_config.yaml", "control_config.yaml", "profile.yaml")
    }
    write_reference(directory, contract, cases)
    contract["files"] = {name: sha256(directory / name) for name in (*FILES, REFERENCE)}
    (directory / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return directory


def same_profile(a, b, tolerance: float = 1e-6) -> bool:
    """Equal profiles, numbers within ``tolerance`` [m, rad] (float32 FK differs across CPUs in the last bit)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_profile(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_profile(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tolerance
    return a == b


@torch.inference_mode()
def write_reference(directory: Path, contract: dict, cases: int) -> None:
    """Reference cases for the robot's load-time check, computed from the files as the robot reads them."""
    kin = ExcavatorKinematics.from_dict(json.loads((directory / "geometry.json").read_text()))
    kin._grid = torch.from_numpy(np.load(directory / "collision_grid.npy"))
    robot = RobotKinematics(json.loads((directory / "robot_geometry.json").read_text()), kin)
    settings = ControllerSettings(**contract["settings"])
    limits = contract.get("joint_speed_limits_rad_s")
    governor = CommandGovernor(
        kin,
        settings,
        None if limits is None else torch.tensor(limits),
        contract.get("governor_joint_speed_margin", 0.8),
    )
    actor = torch.jit.load(str(directory / "actor.pt"), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(2026)

    def uniform(*shape, scale=1.0):
        return (2 * torch.rand(*shape, generator=generator) - 1) * scale

    q = torch.tensor(HOME).repeat(cases, 1)
    q[:, :3] += uniform(cases, 3, scale=0.3)
    q[:, 3] = uniform(cases, scale=0.03)
    v = uniform(cases, 4, scale=0.3)
    requested = uniform(cases, 3) * torch.tensor([0.08, 0.08, 0.3])
    history = MeasuredHistory(
        cases, "cpu", contract["velocity_samples"], contract["command_samples"], contract["u_stride"]
    )
    history.q, history.v, history.u = q, uniform(*history.v.shape, scale=0.3), uniform(*history.u.shape)
    pose, jacobian = kin.pose_jacobian(q)
    robot_pose, robot_jacobian = robot.pose_jacobian(q)
    admitted, _ = governor(q, v, requested)
    observation = history.observe(kin, admitted)
    arrays = dict(
        q=q, v=v, requested=requested, v_history=history.v, u_history=history.u, pose=pose, jacobian=jacobian,
        robot_pose=robot_pose, robot_jacobian=robot_jacobian, admitted=admitted, observation=observation,
        action=actor(observation),
    )  # fmt: skip
    np.savez(
        directory / REFERENCE, **{key: value.numpy().astype(np.float32) for key, value in arrays.items()}
    )
