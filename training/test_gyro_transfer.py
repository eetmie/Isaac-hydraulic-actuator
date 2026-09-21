# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Causality, sensor/plant separation, export parity, and fixed-tip references."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hydraulic_controller.core import DEFAULT_MODEL, HOME, HydraulicPlant
from hydraulic_controller.hardware import ImuReader, OutputGate
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


class FakeHardware:
    def __init__(self):
        self.writes = []
        self.pump = False
        self.resets = 0
        self.accept = True

    def send_named_pwm_commands(self, commands, **kwargs):
        self.writes.append(commands.copy())
        return self.accept

    def set_pump_enabled(self, enabled):
        self.pump = enabled
        return True

    def reset(self, reset_pump=False):
        self.resets += 1


@pytest.mark.parametrize("fault", ["deadman", "sensor", "policy", "nan", "range", "rejected"])
def test_output_fault_latches_pump_off_and_cannot_be_rearmed(fault):
    hardware = FakeHardware()
    now, enabled = [1.0], [True]
    gate = OutputGate(hardware, lambda: enabled[0], clock=lambda: now[0])
    gate.sensor_time = gate.policy_time = now[0]
    gate.arm()
    hardware.set_pump_enabled(True)
    gate.write({"boom": 0.2, "slew": 1.0, "trackL": 1.0})
    assert hardware.writes[-1]["boom"] == 0.2
    assert hardware.writes[-1]["slew"] == hardware.writes[-1]["trackL"] == 0
    if fault == "deadman":
        enabled[0] = False
    elif fault == "sensor":
        now[0] += 0.051
    elif fault == "policy":
        now[0] += 0.151
        gate.sensor_time = now[0]
    elif fault == "rejected":
        hardware.accept = False
    command = {"boom": float("nan") if fault == "nan" else 1.01 if fault == "range" else 0.2}
    gate.write(command)
    assert gate.fault and not gate.armed and not hardware.pump and hardware.resets
    enabled[0] = hardware.accept = True
    gate.sensor_time = gate.policy_time = now[0]
    gate.write({"boom": 1.0, "arm": 1.0, "bucket": 1.0})
    assert all(v == 0 for v in hardware.writes[-1].values())
    with pytest.raises(RuntimeError, match="latched"):
        gate.arm()


def imu_hardware(stamp=10000):
    snapshot = SimpleNamespace(
        device_ts=stamp,
        imu_by_role={r: [1.0, 0.0, 0.0, 0.0] for r in ("base", "boom", "arm", "bucket")},
        imu_gyro=[[0.0, 20.0, 0.0], [0.0, 30.0, 0.0], [0.0, 40.0, 0.0]],
        base_imu_gyro=[0.0, 10.0, 0.0],
    )
    return SimpleNamespace(
        _imu_snapshot=snapshot, _imu_joint_roles=["boom", "arm", "bucket"], is_hardware_ready=lambda: True
    )


def test_imu_reader_same_packet_units_wrap_and_freshness():
    hardware = imu_hardware(2**32 - 5000)
    reader = ImuReader(hardware)
    _, v, _ = reader.read(0.0)
    np.testing.assert_allclose(v, np.deg2rad([10.0] * 4), rtol=1e-6)
    hardware._imu_snapshot.device_ts = 5000
    reader.read(0.01)
    reader.read(0.04)
    with pytest.raises(RuntimeError, match="stale"):
        reader.read(0.061)


@pytest.mark.parametrize("failure", ["reverse", "gap", "slow", "nan"])
def test_imu_reader_rejects_bad_clocks_and_measurements(failure):
    hardware = imu_hardware()
    reader = ImuReader(hardware)
    reader.read(0.0)
    if failure == "reverse":
        hardware._imu_snapshot.device_ts -= 1
    elif failure == "gap":
        hardware._imu_snapshot.device_ts += 110000
    elif failure == "nan":
        hardware._imu_snapshot.imu_gyro[0][1] = float("nan")
    else:
        hardware._imu_snapshot.device_ts += 1000
    with pytest.raises((RuntimeError, ValueError)):
        reader.read(0.06 if failure == "slow" else 0.01)


def test_bundle_export_actor_parity_and_tamper_detection(tmp_path, kin):
    from dataclasses import asdict

    import yaml

    from hydraulic_controller.bundle import PolicyBundle, export_bundle
    from hydraulic_controller.core import ControllerSettings

    profile = {
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
    (tmp_path / "control_config.yaml").write_text(yaml.safe_dump(profile))
    (tmp_path / "servo_config.yaml").write_text("test: true\n")
    (tmp_path / "profile.yaml").write_text("board: jetson\n")
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source checkpoint fingerprint")
    actor = torch.nn.Sequential(torch.nn.Linear(164, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)).eval()
    policy = SimpleNamespace(
        asset_path=kin.path,
        model_path=DEFAULT_MODEL,
        inference=actor,
        checkpoint=checkpoint,
        contract={
            "action_transform": "tanh",
            "observation_dim": 164,
            "settings": asdict(ControllerSettings()),
        },
    )
    directory = export_bundle(policy, tmp_path / "bundle", tmp_path)
    bundle = PolicyBundle(directory)
    observation = torch.randn(1, 164)
    torch.testing.assert_close(bundle.valves(observation), torch.tanh(actor(observation)))
    bundle.check_profile(tmp_path)
    with pytest.raises(ValueError, match="observation"):
        bundle.valves(torch.full((1, 164), float("nan")))
    (tmp_path / "servo_config.yaml").write_text("test: false\n")
    with pytest.raises(ValueError, match="configuration changed"):
        bundle.check_profile(tmp_path)
    (directory / "geometry.json").write_text("{}")
    with pytest.raises(ValueError, match="Bundle file changed"):
        PolicyBundle(directory)


def test_independent_monitor_stops_without_another_output_write():
    hardware = FakeHardware()
    now = [0.0]
    gate = OutputGate(hardware, lambda: True, clock=lambda: now[0])
    gate.sensor_time = gate.policy_time = now[0]
    gate.arm()
    hardware.set_pump_enabled(True)
    now[0] = 0.051
    gate.check()
    assert not gate.armed and not hardware.pump and gate.fault == "sensor timeout"
    assert not hardware.writes


def test_live_report_requires_matching_actor_geometry_and_complete_cases(tmp_path):
    from run_robot_controller import validate_report

    bundle = SimpleNamespace(
        contract={
            "checkpoint_sha256": "actor",
            "sensor_trained": True,
            "files": {"bucket_control_config.yaml": "bucket"},
        }
    )
    report = {
        "checkpoint_sha256": "actor",
        "robot_profile_sha256": "bucket",
        "passed": True,
        "cycles": 3,
        "rate_deg_s": 3,
        "rows": [
            {"amplitude_deg": a, "sensor_mode": s, "tip_max_mm": 5.0, "failed": False}
            for a in (5.0, 10.0)
            for s in ("ideal", "gyro")
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    validate_report(bundle, path)
    for key, value in (
        ("checkpoint_sha256", "different"),
        ("robot_profile_sha256", "gripper"),
        ("rows", []),
        ("cycles", 1),
        ("passed", False),
    ):
        path.write_text(json.dumps(dict(report, **{key: value})))
        with pytest.raises(ValueError):
            validate_report(bundle, path)
