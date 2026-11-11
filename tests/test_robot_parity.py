"""The robot runtime (kaivuriprokkis ``learned_control``) against the training code, plus bundle export.

The robot repository is the sibling checkout; these tests skip without it.
"""

import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hydraulic_controller import sensors as training_sensors
from hydraulic_controller.core import DEFAULT_MODEL, HOME, ControllerSettings
from hydraulic_controller.kinematics import CommandGovernor, ExcavatorKinematics
from hydraulic_controller.observations import MeasuredHistory
from hydraulic_controller.pid import JointPIDController, PidGains
from hydraulic_controller.robot_geometry import RobotKinematics, policy_twist, profile_from_usd
from hydraulic_controller.tasks import Scenario, TaskConfig, reference

ROBOT_REPO = Path(__file__).resolve().parents[2] / "kaivuriprokkis"
if not (ROBOT_REPO / "learned_control").is_dir():
    pytest.skip("kaivuriprokkis checkout with learned_control missing", allow_module_level=True)
sys.path.insert(0, str(ROBOT_REPO))
from learned_control import circle as robot_circle  # noqa: E402
from learned_control import kinematics as robot_kinematics  # noqa: E402
from learned_control import observations as robot_observations  # noqa: E402
from learned_control import robot_geometry as robot_geometry  # noqa: E402
from learned_control import sensors as robot_sensors  # noqa: E402
from learned_control.bundle import PolicyBundle  # noqa: E402

SOURCE_PROFILE = {
    "robot": {
        "joints": [
            {"name": n, "axis": a, "parent_to_joint_xyz": [0, 0, 0]}
            for n, a in [("slew", [0, 0, 1]), ("boom", [0, 1, 0]), ("arm", [0, 1, 0]), ("bucket", [0, 1, 0])]
        ],
        "tool": {"parent_to_tip_xyz": [0, 0, 0]},
    },
    "ik": {"joint_limits_relative": [None, [-50, 30], [28, 144], [-120, 20]]},
}


@pytest.fixture(scope="module")
def geometry():
    pytest.importorskip("pxr")
    kin = ExcavatorKinematics()
    kin.build_collision_grid()
    robot = robot_kinematics.ExcavatorKinematics.from_dict(json.loads(json.dumps(kin.to_dict())))
    robot._grid = kin._grid.clone()
    profile = profile_from_usd(kin, copy.deepcopy(SOURCE_PROFILE))
    return kin, robot, profile


def random_states(count=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.tensor(HOME).repeat(count, 1)
    q[:, :3] += (torch.rand(count, 3, generator=generator) - 0.5) * 0.8
    q[:, 3] = (torch.rand(count, generator=generator) - 0.5) * 0.06
    v = (torch.rand(count, 4, generator=generator) - 0.5) * 0.6
    requested = (torch.rand(count, 3, generator=generator) - 0.5) * torch.tensor([0.2, 0.2, 0.6])
    return q, v, requested


def test_kinematics_governor_and_observations_match(geometry):
    kin, robot, _ = geometry
    q, v, requested = random_states()
    for expected, actual in zip(kin.pose_jacobian(q), robot.pose_jacobian(q)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(robot.valid(q), kin.valid(q))
    targets = kin.pose_jacobian(q[:16])[0] + 0.01
    for expected, actual in zip(kin.inverse(targets, q[:1]), robot.inverse(targets, q[:1])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    settings = ControllerSettings()
    limits = torch.tensor([[0.2, 0.3], [0.35, 0.36], [0.6, 1.0]])
    for speed_limits in (None, limits):
        training = CommandGovernor(kin, settings, speed_limits, 0.8)
        runtime = robot_kinematics.CommandGovernor(robot, settings, speed_limits, 0.8)
        for expected, actual in zip(training(q, v, requested), runtime(q, v, requested)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    history = MeasuredHistory(len(q), "cpu")
    runtime_history = robot_observations.MeasuredHistory(len(q), "cpu")
    generator = torch.Generator().manual_seed(1)
    for _ in range(70):
        sample = (
            q + torch.randn(q.shape, generator=generator) * 0.01,
            torch.randn(v.shape, generator=generator),
        )
        command = torch.rand(len(q), 3, generator=generator) * 2 - 1
        history.push(*sample, command)
        runtime_history.push(*sample, command)
    torch.testing.assert_close(runtime_history.observe(robot, requested), history.observe(kin, requested))


def test_sensor_conversions_match():
    generator = np.random.default_rng(3)
    assert robot_sensors.SENSOR_CONTRACT == training_sensors.SENSOR_CONTRACT
    quats = generator.normal(size=(64, 4, 4))
    gyro = generator.normal(size=(64, 4)) * 30
    for corrected in (False, True):
        np.testing.assert_array_equal(
            robot_sensors.imu_positions(quats, corrected=corrected),
            training_sensors.imu_positions(quats, corrected=corrected),
        )
    np.testing.assert_array_equal(robot_sensors.aligned_rates(gyro), training_sensors.aligned_rates(gyro))
    mount = {
        "mounting_offsets_quat": {
            r: [math.cos(0.1 * i), 0, math.sin(0.1 * i), 0] for i, r in enumerate(training_sensors.ROLES)
        }
    }
    np.testing.assert_array_equal(
        robot_sensors.policy_joint_offset(mount), training_sensors.policy_joint_offset(mount)
    )


def test_robot_geometry_and_twist_mapping_match(geometry):
    kin, robot, profile = geometry
    training = RobotKinematics(profile, kin)
    runtime = robot_geometry.RobotKinematics(profile, robot)
    q, _, requested = random_states(64)
    for expected, actual in zip(training.pose_jacobian(q), runtime.pose_jacobian(q)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    policy_q = q + torch.tensor([0, -0.00677, 0.00677, 0])
    torch.testing.assert_close(
        robot_geometry.policy_twist(q[:1], requested[:1], runtime, robot, policy_q=policy_q[:1]),
        policy_twist(q[:1], requested[:1], training, kin, policy_q=policy_q[:1]),
        rtol=0,
        atol=0,
    )


def test_bucket_geometry_uses_joint_vectors_and_blade_angle(geometry):
    kin, _, profile = geometry
    robot = RobotKinematics(profile, kin)
    vectors = [j["parent_to_joint_xyz"] for j in profile["robot"]["joints"]]
    assert vectors[0] == [0, 0, 0]
    assert vectors[1][2] == pytest.approx(0.0785, abs=1e-7)
    assert vectors[3][2] == pytest.approx(-0.0025, abs=1e-7)
    np.testing.assert_allclose(
        profile["robot"]["tool"]["parent_to_tip_xyz"], [-0.012196396, 0, -0.132756686], atol=1e-7
    )
    q = torch.rand(128, 4, generator=torch.Generator().manual_seed(512)) - 0.5
    q[:, 3] *= 0.02
    expected, expected_j = kin.pose_jacobian(q)
    actual, actual_j = robot.pose_jacobian(q)
    torch.testing.assert_close(actual[:, :2], expected[:, :2], atol=2e-6, rtol=0)
    torch.testing.assert_close(actual_j, expected_j, atol=2e-6, rtol=0)
    angle = actual[:, 2] - expected[:, 2]
    assert torch.atan2(angle.sin(), angle.cos()).abs().max() < 2e-6


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_robot_circle_matches_the_simulation_reference(direction):
    initial = np.array([0.6, 0.03, 2.7])
    path = robot_circle.CircleTrajectory(initial, direction=direction)
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


def test_robot_circle_pid_reproduces_the_simulation_controller(geometry):
    kin, _, profile = geometry
    robot = RobotKinematics(profile, kin)
    gains = PidGains(kp=[10, 11, 5], ki=[0.4, 0.3, 0.25], kd=[0.2, 0.1, 0.05])
    q = torch.tensor([HOME])
    pose = robot.pose_jacobian(q)[0]
    targets = pose.repeat(100, 1, 1)
    targets[:, :, 0] += torch.linspace(0, 0.005, 100)[:, None]
    task = SimpleNamespace(
        kin=robot,
        q0=q,
        tip_ref=targets,
        q_ref=torch.zeros_like(targets),
        is_tip=torch.tensor([True]),
        count=1,
    )
    simulated = JointPIDController(gains)
    hardware = robot_circle.CircleJointPID(robot_circle.PidGains(**asdict(gains)), robot)
    simulated.reset(task)
    for k in range(100):
        measured = q.clone()
        measured[:, 0] += 0.001 * math.sin(k / 10)
        torch.testing.assert_close(
            hardware.valves(measured, targets[k], 0.01), simulated(k, measured, q * 0), rtol=1e-5, atol=1e-6
        )


def test_export_is_verified_and_pinned_by_the_robot_loader(tmp_path, geometry):
    import yaml

    kin, _, profile = geometry
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
    from hydraulic_controller.bundle import export_bundle

    directory = export_bundle(policy, tmp_path / "bundle", tmp_path)
    bundle = PolicyBundle(directory)
    assert bundle.verified
    observation = torch.randn(1, 164)
    torch.testing.assert_close(bundle.valves(observation), torch.tanh(actor(observation)))
    bundle.check_profile(tmp_path)
    (tmp_path / "servo_config.yaml").write_text("test: false\n")
    with pytest.raises(ValueError, match="configuration changed"):
        bundle.check_profile(tmp_path)
    (tmp_path / "control_config.yaml").write_text(yaml.safe_dump(SOURCE_PROFILE))
    with pytest.raises(ValueError, match="not generated from this policy's USD"):
        export_bundle(policy, tmp_path / "other", tmp_path)
