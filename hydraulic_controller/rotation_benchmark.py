# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed-tip rotation benchmark under plant and causal sensor perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark import PLANTS, benchmark_poses
from .core import HOME, HydraulicPlant, sha256, wrap_angle
from .kinematics import ExcavatorKinematics
from .observations import SensorObservation, SensorSettings
from .policy import ControllerPolicy
from .robot_geometry import RobotKinematics, policy_twist
from .rotation import FixedTipRotation


@torch.inference_mode()
def benchmark(policy, out: Path, poses=4, cycles=3, robot_profile=None):
    torch.manual_seed(731)
    kin = ExcavatorKinematics(policy.asset_path, policy.device)
    kin.build_collision_grid()
    robot = kin
    if robot_profile is not None:
        import yaml

        robot = RobotKinematics(yaml.safe_load(Path(robot_profile).read_text()), kin)
    starts = torch.cat((torch.tensor([HOME], device=policy.device), benchmark_poses(kin, poses)), 0)
    rows = []
    traces = {}
    settings = policy.settings
    for amplitude in (5.0, 10.0):
        admitted_starts, start_indices = [], []
        for index, q in enumerate(starts):
            pose, _ = robot.pose_jacobian(q[None])
            sweep = FixedTipRotation(pose[0].cpu().numpy(), amplitude, cycles=cycles)
            try:
                sweep.validate(robot, q[None])
            except ValueError:
                continue
            admitted_starts.append(q)
            start_indices.append(index)
        if not admitted_starts:
            raise ValueError("No feasible rotation poses")
        initial = torch.stack(admitted_starts)
        names = list(PLANTS)
        q0 = initial.repeat(len(names), 1)
        n = len(q0)
        for sensor_mode in ("ideal", "gyro"):
            plant = HydraulicPlant(policy.model_path, kin, n, policy.device, settings)
            plant.reset(torch.arange(n, device=policy.device), q0)
            for index, name in enumerate(names):
                section = slice(index * len(initial), (index + 1) * len(initial))
                perturb = PLANTS[name]
                plant.valve_gain[section] = perturb["gain"]
                plant.valve_offset[section] = perturb["offset"]
                plant.action_delay[section] = perturb["delay"]
                plant.speed_scale[section, :3] = perturb["speed"]
            sensor_cfg = SensorSettings(**policy.contract.get("sensor_randomization", {}))
            sensors = SensorObservation(plant, sensor_cfg)
            sensors.reset(torch.arange(n, device=policy.device))
            governor = policy.governor(kin)
            initial_pose, _ = robot.pose_jacobian(plant.q)
            sweep = FixedTipRotation(initial_pose[0].cpu().numpy(), amplitude, cycles=cycles)
            max_error = torch.zeros(n, device=policy.device)
            sum_error = torch.zeros_like(max_error)
            max_angle = torch.zeros_like(max_error)
            failures = torch.zeros(n, device=policy.device, dtype=torch.bool)
            commands = torch.zeros(n, 3, device=policy.device)
            logs = []
            for step in range(round(sweep.duration / 0.01)):
                t = step * 0.01
                angle, rate = sweep.reference(t)
                delta = angle - sweep.initial[2]
                if step % settings.decimation == 0:
                    q = plant.q if sensor_mode == "ideal" else sensors.history.q
                    v = plant.v if sensor_mode == "ideal" else sensors.history.v[:, 0]
                    observed_pose, _ = robot.pose_jacobian(q)
                    commands[:, :2] = 3 * (initial_pose[:, :2] - observed_pose[:, :2])
                    commands[:, :2] *= (
                        0.03 / commands[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-9)
                    ).clamp_max(1)
                    commands[:, 2] = rate + 3 * wrap_angle(initial_pose[:, 2] + delta - observed_pose[:, 2])
                    desired = commands if robot is kin else policy_twist(q, commands, robot, kin)
                    admitted, _ = governor(q, v, desired)
                    observation = (
                        plant.observe(admitted) if sensor_mode == "ideal" else sensors.observe(admitted)
                    )
                    plant.begin_action(policy(observation))
                plant.step()
                sensors.step()
                pose, _ = robot.pose_jacobian(plant.q)
                error = (pose[:, :2] - initial_pose[:, :2]).norm(dim=1)
                max_error = torch.maximum(max_error, error)
                sum_error += error.square()
                max_angle = torch.maximum(
                    max_angle, wrap_angle(pose[:, 2] - initial_pose[:, 2] - delta).abs()
                )
                failures |= plant.invalid | plant.limit_hit | kin.colliding(plant.q)
                if step % 5 == 0:
                    logs.append(torch.cat((pose, plant.u_cmd), 1).cpu().numpy())
            for i in range(n):
                rows.append(
                    {
                        "amplitude_deg": amplitude,
                        "sensor_mode": sensor_mode,
                        "plant": names[i // len(initial)],
                        "pose_index": start_indices[i % len(initial)],
                        "tip_max_mm": float(max_error[i] * 1000),
                        "tip_rms_mm": float((sum_error[i] / round(sweep.duration / 0.01)).sqrt() * 1000),
                        "angle_max_deg": float(torch.rad2deg(max_angle[i])),
                        "failed": bool(failures[i]),
                        "q0_rad": q0[i].tolist(),
                    }
                )
            traces[f"amp{int(amplitude)}_{sensor_mode}"] = np.stack(logs)
            print(
                json.dumps(
                    {
                        "amplitude": amplitude,
                        "sensors": sensor_mode,
                        "tip_max_mm": float(max_error.max() * 1000),
                        "failures": int(failures.sum()),
                    }
                ),
                flush=True,
            )
    report = {
        "checkpoint": str(policy.checkpoint),
        "checkpoint_sha256": sha256(policy.checkpoint),
        "cycles": cycles,
        "rate_deg_s": 3,
        "rows": rows,
        "robot_profile_sha256": None if robot_profile is None else sha256(Path(robot_profile)),
        "passed": all(not r["failed"] and r["tip_max_mm"] <= 20 for r in rows),
        "hardware_validated": False,
        "acceptance": "All simulated cases <=20mm max tip error without invalid state/limit/collision",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation_results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "rotation_traces.npz", **traces)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poses", type=int, default=4)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--robot_profile", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    benchmark(
        ControllerPolicy(args.checkpoint, args.device), args.out, args.poses, args.cycles, args.robot_profile
    )


if __name__ == "__main__":
    main()
