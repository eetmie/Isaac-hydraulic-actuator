# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Causality, sensor/plant separation, export parity, and fixed-tip references."""

import json

import numpy as np
import pytest
import torch

from hydraulic_controller.core import DEFAULT_MODEL, HOME, HydraulicPlant
from hydraulic_controller.kinematics import ExcavatorKinematics
from hydraulic_controller.observations import MeasuredHistory, SensorObservation, SensorSettings
from hydraulic_controller.rotation import FixedTipRotation
from hydraulic_controller.sensors import MOUNT_PITCH, aligned_rates, causal_indices, imu_positions


def test_aligned_rates_remove_inherited_link_motion():
    np.testing.assert_allclose(
        aligned_rates(np.array([7.0, 17.0, 13.0, 33.0])), np.deg2rad([10, -4, 20, 7]), rtol=1e-6
    )
    np.testing.assert_allclose(aligned_rates(np.full(4, 20))[:3], 0)


def test_mounting_applied_exactly_once():
    pitch = np.array([0.1, -0.3, 0.7, -0.2])

    def quaternion(p):
        return np.stack((np.cos(p / 2), p * 0, np.sin(p / 2), p * 0), -1)

    raw = imu_positions(quaternion(pitch + MOUNT_PITCH))
    corrected = imu_positions(quaternion(pitch), corrected=True)
    np.testing.assert_allclose(raw, corrected, atol=1e-7)
    np.testing.assert_allclose(raw[:3], np.diff(pitch), atol=1e-7)


def test_packet_lookup_cannot_see_future_and_rejects_missing_history():
    ticks = np.array([0.012, 0.021])
    np.testing.assert_array_equal(causal_indices(np.array([0.0, 0.01, 0.02, 0.025]), ticks), [1, 2])
    np.testing.assert_array_equal(causal_indices(np.array([0.0, 0.01, 0.02, 999]), ticks), [1, 2])
    with pytest.raises(ValueError):
        causal_indices(np.array([0.01, 0.02]), np.array([0.0]))
    with pytest.raises(ValueError):
        causal_indices(np.array([0.0, 0.01]), np.array([0.05]))


@pytest.fixture(scope="module")
def kin():
    return ExcavatorKinematics()


def test_measured_history_matches_simulation_over_actions_and_resets(kin):
    plant = HydraulicPlant(DEFAULT_MODEL, kin, 2, "cpu")
    measured = MeasuredHistory(2, "cpu")
    ids = torch.arange(2)
    measured.reset(ids, plant.q)
    torch.manual_seed(810)
    for step in range(90):
        if step == 43:
            plant.reset(ids[:1])
            measured.reset(ids[:1], plant.q[:1])
        command = torch.randn(2, 3) * 0.03
        torch.testing.assert_close(measured.observe(kin, command), plant.observe(command), rtol=0, atol=0)
        if step % 5 == 0:
            plant.begin_action(torch.randn(2, 3) * 0.3)
        plant.step()
        measured.push(plant.q, plant.v, plant.u_cmd)


def test_sensor_errors_persist_in_history_and_do_not_change_physics(kin):
    plant = HydraulicPlant(DEFAULT_MODEL, kin, 2, "cpu")
    sensors = SensorObservation(plant, SensorSettings(0.01, 0.02, 0.003, 2))
    sensors.reset(torch.arange(2))
    q0, v0 = plant.q.clone(), plant.v.clone()
    sensors.step()
    q, v = sensors.history.q.clone(), sensors.history.v.clone()
    first = sensors.observe(torch.zeros(2, 3))
    torch.testing.assert_close(first, sensors.observe(torch.zeros(2, 3)), rtol=0, atol=0)
    sensors.step()
    torch.testing.assert_close(sensors.history.v[:, 1], v[:, 0])
    torch.testing.assert_close(plant.q, q0)
    torch.testing.assert_close(plant.v, v0)
    assert not torch.equal(q, q0)


def test_exported_geometry_matches_usd_fk_jacobian_and_collisions(kin):
    portable = ExcavatorKinematics.from_dict(json.loads(json.dumps(kin.to_dict())))
    q = torch.tensor([HOME, [-0.2, 1.4, -0.7, 0.01], [-0.6, 1.0, -0.4, -0.01]])
    for expected, actual in zip(kin.pose_jacobian(q), portable.pose_jacobian(q)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(kin.colliding(q), portable.colliding(q))


def test_generated_bucket_profile_and_twist_mapping_match_usd(kin):
    from hydraulic_controller.robot_geometry import RobotKinematics, policy_twist, profile_from_usd

    source = {
        "robot": {
            "joints": [
                {
                    "name": n,
                    "axis": [0.0, 0.0, 1.0] if n == "slew" else [0.0, 1.0, 0.0],
                    "parent_to_joint_xyz": [0.0, 0.0, 0.0],
                }
                for n in ("slew", "boom", "arm", "bucket")
            ],
            "tool": {},
        },
        "ik": {"joint_limits_relative": [None] * 4},
    }
    profile = profile_from_usd(kin, source)
    robot = RobotKinematics(profile, kin)
    q = torch.tensor([HOME])
    expected, jac = kin.pose_jacobian(q)
    actual, robot_jac = robot.pose_jacobian(q)
    torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=2e-6, rtol=0)
    torch.testing.assert_close(robot_jac, jac, atol=2e-6, rtol=0)
    request = torch.tensor([[0.01, -0.02, 0.05]])
    torch.testing.assert_close(policy_twist(q, request, robot, kin), request, atol=1e-6, rtol=0)


def test_rotation_reference_holds_position_and_respects_rate():
    initial = np.array([0.5, 0.1, -0.4])
    sweep = FixedTipRotation(initial, 10, cycles=3)
    for t in np.linspace(0, sweep.duration, 1201):
        theta, omega = sweep.reference(t)
        assert abs(omega) <= np.deg2rad(3) + 1e-10
        assert abs(theta - initial[2]) <= np.deg2rad(10) + 1e-10
        command = sweep.command(t, np.r_[initial[:2], theta])
        np.testing.assert_allclose(command[:2], 0, atol=1e-10)
        assert abs(command[2] - omega) < 1e-10
    assert sweep.reference(sweep.duration) == (initial[2], 0.0)
    assert sweep.command(0, initial + np.array([0.01, 0, 0]))[0] < 0


def test_gyro_environment_uses_shared_measurements(kin):
    from hydraulic_controller.env import HydraulicControlEnv, HydraulicControlEnvCfg

    cfg = HydraulicControlEnvCfg(
        num_envs=2, device="cpu", gyro_observations=True, governor_joint_speed_margin=0, randomize=False
    )
    env = HydraulicControlEnv(cfg)
    obs, _, _, _ = env.step(torch.zeros(2, 3))
    torch.testing.assert_close(obs["policy"], env.plant.observe(env.command))
    assert env.contract()["sensors"]["velocity_source"] == "aligned_y_child_minus_parent"
