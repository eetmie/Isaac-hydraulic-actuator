# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Write a copy of the excavator USD whose boom/arm/bucket chain matches a measured robot profile.

The robot profile (kaivuriprokkis ``configuration_files/profiles/<robot>/control_config.yaml``) gives pin-to-pin
link vectors and the bucket pivot -> tool tip vector in its own IMU calibration. Training angles use the
fixed sensor contract (``hydraulic_controller.sensors``), so the profile is evaluated at
``q_robot = q_training - policy_joint_offset``. For each link the joint position in its parent body is moved
so that, at every joint angle, the USD pivots and ``ee_tip`` equal the profile's. The link's visual and collider
meshes are rotated and stretched along the pin line to follow (about the link's own pivot), and the
downstream bodies' authored poses move with the chain. Joint axes, angle zeros and the bucket angle are kept.

    python retarget_asset.py --profile <kaivuriprokkis>/configuration_files/profiles/jetson/control_config.yaml \\
        --out assets/excavator_bucket_rocking_jetson.usd
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
# Joint whose localPos0 places each link's far pivot, and the body the link's meshes belong to.
LINKS = (("revolute_tilt", "lift_boom"), ("revolute_tool", "tilt_boom"))
# Bodies (and markers) downstream of each patched joint, whose authored pose follows the chain.
DOWNSTREAM = ("tilt_boom", "tool_body", "bucket")
BODY_INDEX = {"lift_boom": 4, "tilt_boom": 5, "tool_body": 6, "bucket": 7}  # ExcavatorKinematics transforms


def chain_points(kin, q: torch.Tensor) -> torch.Tensor:
    """Boom, arm and bucket pivots and the tip [m], shape [N, 4, 3]."""
    transforms, joints = kin._transforms(q)
    tip = (transforms[-1] @ kin.tip)[:, :3, 3]
    return torch.stack([joints[i][:, :3, 3] for i in range(3)] + [tip], 1)


def profile_points(profile: dict, kin, q_training: torch.Tensor) -> torch.Tensor:
    """The profile's pivots and tip at training-calibration angles, in the profile's frame [m]."""
    from hydraulic_controller.robot_geometry import RobotKinematics
    from hydraulic_controller.sensors import policy_joint_offset

    robot = RobotKinematics(profile, kin)
    offset = torch.tensor(policy_joint_offset(profile["imu"]), device=q_training.device)
    return chain_points(robot, q_training - offset)


def stretch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 map (column vectors) taking a to b in the X/Z plane: stretch along a, then rotate about Y."""
    a2, b2 = np.array([a[0], a[2]]), np.array([b[0], b[2]])
    u = a2 / np.linalg.norm(a2)
    scale = np.eye(2) + (np.linalg.norm(b2) / np.linalg.norm(a2) - 1.0) * np.outer(u, u)
    turn = math.atan2(b2[1], b2[0]) - math.atan2(a2[1], a2[0])
    c, s = math.cos(turn), math.sin(turn)
    planar = np.array([[c, -s], [s, c]]) @ scale
    out = np.eye(3)
    out[np.ix_([0, 2], [0, 2])] = planar
    return out


def retarget(source: Path, profile: dict, out: Path) -> dict:
    from pxr import Gf, Usd, UsdGeom

    from hydraulic_controller.core import HOME
    from hydraulic_controller.kinematics import ExcavatorKinematics

    stage = Usd.Stage.Open(str(source))
    zero, home = torch.zeros(1, 4), torch.tensor([HOME])
    report = {}
    old_kin = ExcavatorKinematics(source)
    old_home = old_kin._transforms(home)[0]
    tmp = out.with_suffix(".tmp.usd")
    for joint_name, body in LINKS:
        stage.Export(str(tmp))
        kin = ExcavatorKinematics(tmp)
        usd = chain_points(kin, zero)[0]
        target = profile_points(profile, kin, zero)[0]
        link = LINKS.index((joint_name, body))  # 0: boom -> arm pivot, 1: arm -> bucket pivot
        new_far = usd[link] + (target[link + 1] - target[link])
        frame = kin._transforms(zero)[0][BODY_INDEX[body]][0]
        joint = stage.GetPrimAtPath(f"/excavator/Joints/{joint_name}")
        old_local = np.array(joint.GetAttribute("physics:localPos0").Get(), dtype=float)
        local = (torch.linalg.inv(frame) @ torch.cat((new_far, torch.ones(1))))[:3].double().numpy()
        joint.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*local.tolist()))
        report[joint_name] = {"old_m": old_local.tolist(), "new_m": local.tolist()}
        # Meshes of this link follow its pin line, about the link's own pivot (its frame origin).
        near = (torch.linalg.inv(frame) @ torch.cat((usd[link], torch.ones(1))))[:3].double().numpy()
        affine = stretch(old_local - near, local - near)
        # Gf multiplies row vectors, so the column-vector map enters transposed.
        to_body = Gf.Matrix4d(Gf.Matrix3d(*affine.T.flatten().tolist()), Gf.Vec3d(0.0))
        pivot = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*near.tolist()))
        to_body = pivot.GetInverse() * to_body * pivot
        body_prim = stage.GetPrimAtPath(f"/excavator/{body}")
        cache = UsdGeom.XformCache()
        body_world_inv = cache.GetLocalToWorldTransform(body_prim).GetInverse()
        for child in Usd.PrimRange(body_prim):
            if not child.IsA(UsdGeom.Mesh):
                continue
            parent_to_body = cache.GetLocalToWorldTransform(child.GetParent()) * body_world_inv
            local_matrix = UsdGeom.Xformable(child).GetLocalTransformation()
            new_local = local_matrix * parent_to_body * to_body * parent_to_body.GetInverse()
            xformable = UsdGeom.Xformable(child)
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp().Set(new_local)
        report[f"{body}_mesh_stretch"] = float(
            np.linalg.norm(local - near) / np.linalg.norm(old_local - near)
        )
        report[f"{body}_mesh_turn_deg"] = math.degrees(
            math.atan2(local[2] - near[2], local[0] - near[0])
            - math.atan2(old_local[2] - near[2], old_local[0] - near[0])
        )
        # Markers that sit on this joint's parent-side position.
        for marker in Usd.PrimRange(body_prim):
            ops = UsdGeom.Xformable(marker).GetOrderedXformOps() if marker.IsA(UsdGeom.Xformable) else []
            if marker.GetName().endswith("_pos") and ops and ops[0].GetOpName() == "xformOp:translate":
                current = np.array(ops[0].Get(), dtype=float)
                if np.allclose(current, old_local, atol=1e-5):
                    ops[0].Set(type(ops[0].Get())(*local.tolist()))
    # The bucket pivot -> tip vector, in the bucket body frame used by ExcavatorKinematics.
    stage.Export(str(tmp))
    kin = ExcavatorKinematics(tmp)
    usd = chain_points(kin, zero)[0]
    target = profile_points(profile, kin, zero)[0]
    tip_world = usd[2] + (target[3] - target[2])
    bucket_frame = kin._transforms(zero)[0][-1][0]
    tip_local = (torch.linalg.inv(bucket_frame) @ torch.cat((tip_world, torch.ones(1))))[:3].double().numpy()
    tip_op = UsdGeom.Xformable(stage.GetPrimAtPath("/excavator/bucket/ee_tip")).GetOrderedXformOps()[0]
    report["ee_tip"] = {"old_m": list(tip_op.Get()), "new_m": tip_local.tolist()}
    tip_op.Set(type(tip_op.Get())(*tip_local.tolist()))
    # Authored body poses are the HOME pose: move the downstream bodies with the patched chain.
    stage.Export(str(tmp))
    new_home = ExcavatorKinematics(tmp)._transforms(home)[0]
    for body in DOWNSTREAM:
        delta = (new_home[BODY_INDEX[body]][0, :3, 3] - old_home[BODY_INDEX[body]][0, :3, 3]).double().numpy()
        op = UsdGeom.Xformable(stage.GetPrimAtPath(f"/excavator/{body}")).GetOrderedXformOps()[0]
        if op.GetOpName() != "xformOp:translate":
            raise ValueError(f"{body} does not start with a translate op")
        op.Set(type(op.Get())(*(np.array(op.Get(), dtype=float) + delta).tolist()))
        report[f"{body}_home_shift_mm"] = (delta * 1000).tolist()
    tmp.unlink()
    stage.GetRootLayer().customLayerData = {
        **stage.GetRootLayer().customLayerData,
        "retargeted_from": source.name,
        "retarget_profile_note": "boom/arm pivots and ee_tip fitted to a measured kaivuriprokkis profile",
    }
    stage.Export(str(out))
    return report


def verify(out: Path, profile: dict, samples: int = 256) -> dict:
    """Largest pivot/tip and bucket-angle mismatch [mm, deg] to the profile over random training angles."""
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.robot_geometry import RobotKinematics
    from hydraulic_controller.sensors import policy_joint_offset

    kin = ExcavatorKinematics(out)
    generator = torch.Generator().manual_seed(7)
    lo, hi = kin.limits[:, 0], kin.limits[:, 1]
    q = lo + (hi - lo) * torch.rand(samples, 4, generator=generator)
    q[:, 3] = 0
    usd, robot = chain_points(kin, q), profile_points(profile, kin, q)
    # The profile's origin is the ground, the USD's the lower carriage: compare from the boom pin.
    shift = usd[:, :1] - robot[:, :1]
    error = (usd - robot - shift)[:, :, [0, 2]].norm(dim=-1) * 1000
    angle = kin.pose_jacobian(q)[0][:, 2]
    offset = torch.tensor(policy_joint_offset(profile["imu"]))
    angle_robot = RobotKinematics(profile, kin).pose_jacobian(q - offset)[0][:, 2]
    angle_error = torch.rad2deg(torch.atan2((angle - angle_robot).sin(), (angle - angle_robot).cos())).abs()
    return {
        "max_error_mm": dict(zip(("boom", "arm", "bucket", "tip"), error.amax(0).tolist(), strict=True)),
        "max_bucket_angle_error_deg": float(angle_error.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=ROOT / "assets/excavator_bucket_rocking.usd")
    parser.add_argument("--profile", type=Path, required=True, help="kaivuriprokkis control_config.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists; choose a new name (checkpoints pin their asset by hash)")
    profile = yaml.safe_load(args.profile.read_text())
    import json

    report = retarget(args.source, profile, args.out)
    report["verify"] = verify(args.out, profile)
    print(json.dumps(report, indent=2))
    if (
        max(report["verify"]["max_error_mm"].values()) > 0.1
        or report["verify"]["max_bucket_angle_error_deg"] > 1e-3
    ):
        print("[WARN] retargeted USD does not reproduce the profile within 0.1 mm / 0.001 deg")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
