# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Circle timing, joint-based bucket geometry, PID equivalence and latched output stops."""

import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hydraulic_controller.circle import CircleJointPID, CircleTrajectory, StartMove
from hydraulic_controller.core import HOME
from hydraulic_controller.hardware import OutputGate, raw_imu_values
from hydraulic_controller.pid import JointPIDController, PidGains
from hydraulic_controller.recording import BufferedRecording
from hydraulic_controller.robot_geometry import RobotKinematics, profile_from_usd
from hydraulic_controller.sensors import MOUNT_PITCH, imu_positions, policy_joint_offset
from hydraulic_controller.tasks import TaskConfig, reference
from run_robot_circle import (
    check_motion_envelope,
    compare_runs,
    operator_enabled,
    repeat_phase,
    summarize,
)


def test_buffer_keeps_multiple_passes_and_writes_only_when_requested(tmp_path):
    fields = ["armed", "pass_index", "device_ts_us", "sample"]
    buffer = BufferedRecording(fields, 300)
    path = tmp_path / "passes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        buffer.append([True, 0, 1234567890, 1.2])
        buffer.append([True, 1, 1234567891, 2.3])
        assert len(path.read_text().splitlines()) == 1
        buffer.write_to(writer)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["pass_index"] for r in rows] == ["0", "1"]
    assert rows[0]["armed"] == "True"
    assert rows[0]["device_ts_us"] == "1234567890"
    assert buffer.samples.nbytes < 30_000_000


def test_raw_imu_logs_all_axes_in_physical_sensor_order():
    snapshot = SimpleNamespace(
        raw_accel=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        raw_gyro=[[21, 22, 23], [24, 25, 26], [27, 28, 29], [30, 31, 32]],
    )
    values = np.array(raw_imu_values(snapshot)).reshape(4, 6)
    np.testing.assert_array_equal(values[1], [4, 5, 6, 24, 25, 26])
    snapshot.raw_accel = None
    with pytest.raises(ValueError, match="raw accelerometer/gyro"):
        raw_imu_values(snapshot)


def test_rocking_does_not_trip_the_driven_joint_rate_guard():
    q = np.array([0, 1, 0, math.radians(2)])
    v = np.array([0.1, 0.2, -0.3, 3.0])
    check_motion_envelope(q, v, 3, 2)
    with pytest.raises(ValueError, match="Carriage pitch"):
        check_motion_envelope(q, v, 1, 2)
    v[1] = 2.1
    with pytest.raises(ValueError, match="arm rate"):
        check_motion_envelope(q, v, 3, 2)
    v[1] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        check_motion_envelope(q, v, 3, 2)


def test_start_move_is_smooth_rate_bounded_and_rejects_physical_bounds():
    initial = torch.tensor([[0.0, 1.0, -0.2, 0.0]])
    target = torch.tensor([[0.2, 1.5, -0.4, 0.0]])
    kin = SimpleNamespace(valid=lambda q: (q[:, :3].abs() <= 2).all(1))
    move = StartMove(initial, target, kin)
    torch.testing.assert_close(move.reference(0), initial[:, :3])
    torch.testing.assert_close(move.reference(move.duration), target[:, :3])
    times = np.linspace(0, move.duration, 1001)
    positions = torch.cat([move.reference(t) for t in times]).numpy()
    assert np.abs(np.diff(positions, axis=0) / np.diff(times)[:, None]).max() <= 0.0501
    target[0, 1] = 2.5
    with pytest.raises(ValueError, match="physical joint/collision"):
        StartMove(initial, target, kin)


def test_continuous_circle_keeps_original_center_and_restarts_after_settling():
    path = CircleTrajectory([0.5, 0.1, 2.7])
    for lap in (0, 1, 37):
        index, phase = repeat_phase(path, lap * path.duration + 2.0)
        assert index == lap
        np.testing.assert_allclose(path.reference(phase)[0], path.reference(2.0)[0], atol=1e-12)
    index, phase = repeat_phase(path, path.duration)
    assert index == 1 and phase == 0
    np.testing.assert_array_equal(path.reference(phase)[0], path.initial)


def test_autonomous_enable_survives_lb_release_b_logs_and_a_or_disconnect_stop():
    connected = [True]
    pad = SimpleNamespace(LeftBumper=False, A=False, B=True, is_connected=lambda: connected[0])
    assert not operator_enabled(pad, True, False)
    assert operator_enabled(pad, True, True)
    pad.A = True
    assert not operator_enabled(pad, True, True)
    pad.A = False
    connected[0] = False
    assert not operator_enabled(pad, True, True)
    connected[0] = True
    pad.B, pad.LeftBumper = False, True
    assert operator_enabled(pad, False, False)
    pad.LeftBumper = False
    assert not operator_enabled(pad, False, False)


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_circle_matches_existing_simulation_reference_and_velocity(direction):
    initial = np.array([0.6, 0.03, 2.7])
    path = CircleTrajectory(initial, direction=direction)
    from hydraulic_controller.tasks import Scenario

    cfg = TaskConfig(duration_s=path.duration)
    scenarios = [Scenario("tip_circle", direction, 0.02, "nominal")]
    _, targets = reference(
        cfg, scenarios, torch.zeros(1, 4), torch.tensor(initial[None], dtype=torch.float32)
    )
    for k in range(0, len(targets), 37):
        t = (k + 1) * 0.01
        pose, twist = path.reference(t)
        np.testing.assert_allclose(pose, targets[k, 0].numpy(), atol=2e-7)
        if path.lead + 0.1 < t < path.move_end - 0.1:
            delta = (path.reference(t + 1e-5)[0] - path.reference(t - 1e-5)[0]) / 2e-5
            np.testing.assert_allclose(twist, delta, atol=1e-8)
            assert np.linalg.norm(twist[:2]) <= 0.02000001
    for t in [-1, 0, path.duration]:
        pose, twist = path.reference(t)
        np.testing.assert_array_equal(pose, initial)
        np.testing.assert_array_equal(twist, np.zeros(3))


def test_circle_restarts_at_zero_velocity_each_lap():
    path = CircleTrajectory([0.6, 0.03, 0], cycles=3)
    for lap in range(4):
        pose, twist = path.reference(path.lead + lap * path.lap_s)
        np.testing.assert_allclose(pose, path.initial, atol=1e-12)
        np.testing.assert_allclose(twist, 0, atol=1e-12)
    with pytest.raises(ValueError):
        CircleTrajectory([0.6, 0.03, 0], speed_m_s=float("nan"))


@pytest.fixture(scope="module")
def geometry():
    pytest.importorskip("pxr")
    from hydraulic_controller.kinematics import ExcavatorKinematics

    kin = ExcavatorKinematics()
    kin.build_collision_grid()
    # Geometry extraction consumes only these profile sections; no external robot checkout needed.
    source = {
        "robot": {
            "joints": [
                {"name": n, "axis": a, "parent_to_joint_xyz": [0, 0, 0]}
                for n, a in [
                    ("slew", [0, 0, 1]),
                    ("boom", [0, 1, 0]),
                    ("arm", [0, 1, 0]),
                    ("bucket", [0, 1, 0]),
                ]
            ],
            "tool": {"parent_to_tip_xyz": [0, 0, 0]},
        },
        "ik": {"joint_limits_relative": [None, [-50, 30], [28, 144], [-120, 20]]},
    }
    return kin, RobotKinematics(profile_from_usd(kin, source), kin)


def test_bucket_geometry_uses_joint_vectors_and_blade_angle(geometry):
    kin, robot = geometry
    vectors = [j["parent_to_joint_xyz"] for j in robot.profile["robot"]["joints"]]
    assert vectors[0] == [0, 0, 0]
    assert vectors[1][2] == pytest.approx(0.0785, abs=1e-7)
    assert vectors[3][2] == pytest.approx(-0.0025, abs=1e-7)
    tip = robot.profile["robot"]["tool"]["parent_to_tip_xyz"]
    np.testing.assert_allclose(tip, [-0.012196396, 0, -0.132756686], atol=1e-7)
    q = torch.rand(128, 4, generator=torch.Generator().manual_seed(512)) - 0.5
    q[:, 3] *= 0.02
    expected, expected_j = kin.pose_jacobian(q)
    actual, actual_j = robot.pose_jacobian(q)
    torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=2e-6, rtol=0)
    torch.testing.assert_close(actual_j, expected_j, atol=2e-6, rtol=0)
    assert (
        torch.atan2((actual[:, 2] - expected[:, 2]).sin(), (actual[:, 2] - expected[:, 2]).cos()).abs().max()
        < 2e-6
    )


def test_circle_pid_reproduces_simulation_controller(geometry):
    _, kin = geometry
    gains = PidGains(kp=[10, 11, 5], ki=[0.4, 0.3, 0.25], kd=[0.2, 0.1, 0.05])
    q = torch.tensor([HOME])
    pose = kin.pose_jacobian(q)[0]
    targets = pose.repeat(100, 1, 1)
    targets[:, :, 0] += torch.linspace(0, 0.005, 100)[:, None]
    task = SimpleNamespace(
        kin=kin, q0=q, tip_ref=targets, q_ref=torch.zeros_like(targets), is_tip=torch.tensor([True]), count=1
    )
    simulated, hardware = JointPIDController(gains), CircleJointPID(gains, kin)
    # Match live warmup, reset, and repeated operator-enabled control ticks.
    with torch.no_grad():
        hardware.valves(q, pose, 0.01)
    hardware.reset()
    simulated.reset(task)
    for k in range(100):
        measured = q.clone()
        measured[:, 0] += 0.001 * math.sin(k / 10)
        torch.testing.assert_close(hardware.valves(measured, targets[k], 0.01), simulated(k, measured, q * 0))


def test_gate_forces_slew_tracks_zero_and_latches_release():
    clock, enabled, written, pump = [10.0], [True], [], []
    hardware = SimpleNamespace(
        send_named_pwm_commands=lambda commands, **kwargs: written.append(commands.copy()) or True,
        set_pump_enabled=lambda value: pump.append(value),
        reset=lambda **kwargs: None,
    )
    gate = OutputGate(hardware, lambda: enabled[0], lambda: clock[0])
    gate.sensor_time = gate.policy_time = clock[0]
    gate.arm()
    gate.write({"boom": 0.2, "slew": 1, "trackL": 1, "trackR": 1})
    assert written[-1] == {"boom": 0.2, "arm": 0.0, "bucket": 0.0, "slew": 0.0, "trackL": 0.0, "trackR": 0.0}
    enabled[0] = False
    gate.check()
    enabled[0] = True
    gate.write({"boom": 1})
    assert not gate.armed and not any(written[-1].values()) and pump[-1] is False
    with pytest.raises(RuntimeError, match="latched"):
        gate.arm()


def test_summary_distinguishes_radial_error_from_timing_error_and_keeps_faults():
    path = CircleTrajectory([0.6, 0.03, 0])
    rows = [
        dict(
            armed=True,
            motion_t_s=2.0 + i * 0.01,
            error_m=0.01,
            radial_error_m=0.0,
            angle_error_rad=0,
            boom_emitted_u=0.2,
            arm_emitted_u=0.2,
            bucket_emitted_u=0.0,
            compute_ms=1.0,
        )
        for i in range(3)
    ]
    result = summarize(rows, path, "operator stop")
    assert not result["completed"] and result["fault"] == "operator stop"
    assert result["tracking_rmse_mm"] == pytest.approx(10)
    assert result["radial_max_abs_mm"] == 0


def test_arm_mounting_change_preserves_the_frozen_policy_observation_frame():
    runtime_mount = MOUNT_PITCH.copy()
    runtime_mount[2] = math.radians(0.612)
    profile = {
        "mounting_offsets_quat": {
            role: [math.cos(a / 2), 0, math.sin(a / 2), 0]
            for role, a in zip(("base", "boom", "arm", "bucket"), runtime_mount, strict=True)
        }
    }
    raw_pitch = np.array([0.0, -0.4, 0.7, -0.2])

    def quats(pitch):
        return np.stack([np.cos(pitch / 2), pitch * 0, np.sin(pitch / 2), pitch * 0], axis=-1)

    physical_q = imu_positions(quats(raw_pitch - runtime_mount), corrected=True)
    training_q = imu_positions(quats(raw_pitch), corrected=False)
    offset = policy_joint_offset(profile)
    np.testing.assert_allclose(physical_q + offset, training_q, atol=1e-7)
    # The arm changes by -0.388 degrees in policy coordinates; the bucket changes oppositely.
    assert math.degrees(offset[1]) == pytest.approx(-0.388, abs=1e-5)
    assert math.degrees(offset[2]) == pytest.approx(+0.388, abs=1e-5)
    assert offset[0] == 0 and offset[3] == 0


def test_twist_mapping_uses_physical_and_training_joint_zeros_separately(geometry):
    from hydraulic_controller.robot_geometry import policy_twist

    kin, robot = geometry
    physical_q = torch.tensor([HOME])
    policy_q = physical_q + torch.tensor([0, -0.00677, 0.00677, 0])
    physical_twist = torch.tensor([[0.02, 0.0, 0.0]])
    desired = policy_twist(physical_q, physical_twist, robot, kin, policy_q=policy_q)
    real_jac = robot.pose_jacobian(physical_q)[1][:, :, :3]
    learned_jac = kin.pose_jacobian(policy_q)[1][:, :, :3]
    rates = torch.linalg.solve(real_jac, physical_twist[:, :, None])
    torch.testing.assert_close(desired, (learned_jac @ rates).squeeze(-1))


def test_comparison_keeps_stopped_trials_and_rejects_changed_calibration(tmp_path):
    logs = [tmp_path / "pid.csv", tmp_path / "mlp.csv"]
    reports = []
    for path, controller in zip(logs, ("pid_tuned", "mlp"), strict=True):
        report = {
            "controller": controller,
            "direction": "ccw",
            "radius_mm": 50,
            "speed_mm_s": 20,
            "cycles": 1,
            "profile_sha256": {"control_config.yaml": "same calibration"},
            "result": {
                "completed": controller == "pid_tuned",
                "fault": None if controller == "pid_tuned" else "stop",
                "tracking_rmse_mm": 5.0,
                "radial_max_abs_mm": 3.0,
            },
        }
        path.with_suffix(".json").write_text(json.dumps(report))
        reports.append(report)
    args = SimpleNamespace(logs=logs, out=tmp_path / "comparison", plot=False)
    compare_runs(args)
    result = json.loads(args.out.with_suffix(".json").read_text())
    assert len(result["runs"]) == 2 and result["runs"][1]["fault"] == "stop"
    assert not result["runs"][1]["completed"]
    reports[1]["profile_sha256"]["control_config.yaml"] = "changed calibration"
    logs[1].with_suffix(".json").write_text(json.dumps(reports[1]))
    with pytest.raises(ValueError, match="profile_sha256"):
        compare_runs(SimpleNamespace(logs=logs, out=tmp_path / "different", plot=False))
    assert not (tmp_path / "different.csv").exists()
