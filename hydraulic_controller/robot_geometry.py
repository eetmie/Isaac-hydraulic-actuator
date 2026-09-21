# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Robot-profile cutting-tip geometry and twist conversion to the policy's training tip."""

from __future__ import annotations

import copy

import torch

from .kinematics import ExcavatorKinematics


class RobotKinematics(ExcavatorKinematics):
    """Fixed-slew planar FK using the physical robot profile [m, rad].

    Generated deployment profiles match the policy's authored USD. Legacy
    profiles can also be inspected while retaining the policy observation frame.
    Collision checks still use the conservative simulation geometry; this is a
    free-space adapter, not a newly validated physical collision model.
    """

    def __init__(self, profile: dict, safety: ExcavatorKinematics):
        self.device = safety.device
        self.safety = safety
        self.profile = profile
        self.limits = safety.limits.clone()
        limits = profile["ik"]["joint_limits_relative"]
        for i, bounds in enumerate(limits[1:4]):
            if bounds is not None:
                limit = torch.deg2rad(torch.tensor(bounds, device=self.device))
                self.limits[i, 0] = torch.maximum(self.limits[i, 0], limit[0])
                self.limits[i, 1] = torch.minimum(self.limits[i, 1], limit[1])
        if torch.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Robot and policy joint limits do not overlap")
        eye = torch.eye(4, device=self.device)
        self.frames = [(eye.clone(), eye.clone(), 3)]
        joints = profile["robot"]["joints"]
        if [j["name"] for j in joints] != ["slew", "boom", "arm", "bucket"]:
            raise ValueError("Expected the fixed-slew boom/arm/bucket chain")
        for i, joint in enumerate(joints):
            expected = [0.0, 0.0, 1.0] if i == 0 else [0.0, 1.0, 0.0]
            if joint["axis"] != expected:
                raise ValueError("The hardware demo requires aligned Y-axis arm joints")
            frame = eye.clone()
            frame[:3, 3] = torch.tensor(joint["parent_to_joint_xyz"], device=self.device)
            self.frames.append((frame, eye.clone(), i - 1))
        self.tip = eye.clone()
        self.tip[:3, 3] = torch.tensor(profile["robot"]["tool"]["parent_to_tip_xyz"], device=self.device)

    def pose_jacobian(self, q):
        """Return physical tip pose [m, m, rad] and Jacobian [m/rad; rad/rad]."""
        transforms, joints = self._transforms(q)
        point = (transforms[-1] @ self.tip)[:, :3, 3]
        angle = q.sum(1, keepdim=True) + self.profile["robot"]["tool"].get("pitch_offset_rad", 0.0)
        pose = torch.cat((point[:, [0, 2]], angle), 1)
        columns = []
        for i in range(4):
            frame = joints[i]
            axis = frame[:, :3, 1]
            linear = torch.linalg.cross(axis, point - frame[:, :3, 3])[:, [0, 2]]
            columns.append(torch.cat((linear, axis[:, 1:2]), 1))
        return pose, torch.stack(columns, -1)

    def colliding(self, q):
        return self.safety.colliding(q)


def profile_from_usd(kin: ExcavatorKinematics, source: dict) -> dict:
    """Generate the planar deployment profile from the selected USD [m, rad].

    Joint rotations share Y, so constant authored rotations commute with joint
    motion. Their effective link vectors can be extracted at zero joint angles.
    This also aligns the origin to the USD lower carriage rather than the legacy
    robot profile's ground-height convention. IMU/PWM calibration is unaffected.
    """
    result = copy.deepcopy(source)
    zero = torch.zeros(1, 4, device=kin.device)
    transforms, joints = kin._transforms(zero)
    points = [joints[i][0, :3, 3] for i in range(3)]
    tip = (transforms[-1] @ kin.tip)[0, :3, 3]
    translations = [torch.zeros_like(points[0]), points[0], points[1] - points[0], points[2] - points[1]]
    for joint, translation in zip(result["robot"]["joints"], translations, strict=True):
        joint["parent_to_joint_xyz"] = translation.tolist()
    result["robot"]["tool"]["parent_to_tip_xyz"] = (tip - points[2]).tolist()
    result["robot"]["tool"]["pitch_offset_rad"] = float(kin.pose_jacobian(zero)[0][0, 2])
    equivalent = RobotKinematics(result, kin)
    generator = torch.Generator(device="cpu").manual_seed(512)
    q = (torch.rand(128, 4, generator=generator) * 2 - 1).to(kin.device)
    q[:, 3] *= 0.02
    expected_pose, expected_jac = kin.pose_jacobian(q)
    actual_pose, actual_jac = equivalent.pose_jacobian(q)
    if not torch.allclose(expected_pose[:, :2], actual_pose[:, :2], atol=2e-6, rtol=0):
        raise ValueError("USD cannot be represented by the aligned-axis robot profile")
    if not torch.allclose(expected_jac, actual_jac, atol=2e-6, rtol=0):
        raise ValueError("Extracted robot profile has a different Jacobian")
    angle_error = actual_pose[:, 2] - expected_pose[:, 2]
    if bool((torch.atan2(angle_error.sin(), angle_error.cos()).abs() > 2e-6).any()):
        raise ValueError("Extracted robot profile has a different tool angle")
    return result


def policy_twist(q, physical_twist, robot, policy_kin):
    """Map physical cutting-tip twist [m/s, m/s, rad/s] to the policy's trained tip."""
    _, real_jac = robot.pose_jacobian(q)
    _, learned_jac = policy_kin.pose_jacobian(q)
    j = real_jac[:, :, :3]
    # A weighted condition check avoids blindly amplifying commands at a singularity.
    weighted = j * q.new_tensor([1.0, 1.0, 0.2])[None, :, None]
    if not torch.isfinite(j).all() or bool((torch.linalg.cond(weighted) > 100).any()):
        raise ValueError("Physical tip Jacobian is too close to a singularity")
    rates = torch.linalg.solve(j, physical_twist[:, :, None]).squeeze(-1)
    return torch.einsum("nij,nj->ni", learned_jac[:, :, :3], rates)
