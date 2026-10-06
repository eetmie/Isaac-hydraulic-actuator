"""The batched PID must reproduce the robot's ``modules/pid.py`` so sim-tuned gains transfer as-is.

The reference is the robot file itself, found through ``$KAIVURIPROKKIS`` or a sibling ``kaivuriprokkis``
checkout; without one these parity tests skip.
"""

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hydraulic_controller.pid import BatchedPID, robot_joint_command

ROOT = Path(__file__).resolve().parents[1]
ROBOT_PID = Path(os.environ.get("KAIVURIPROKKIS", ROOT.parent / "kaivuriprokkis")) / "modules/pid.py"
needs_robot = pytest.mark.skipif(not ROBOT_PID.exists(), reason=f"robot PID not found at {ROBOT_PID}")


def robot_pid_class():
    spec = importlib.util.spec_from_file_location("robot_pid", ROBOT_PID)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIDController


def random_gains(rng, count):
    """Gain sets including the zero-I, zero-D and unfiltered corners the robot code special-cases."""
    gains = []
    for i in range(count):
        kp, ki, kd, tau = rng.uniform(0, 15), rng.uniform(0, 3), rng.uniform(0, 0.5), rng.uniform(0, 0.2)
        if i % 4 == 1:
            ki = 0.0
        if i % 4 == 2:
            kd = 0.0
        if i % 4 == 3:
            tau = 0.0
        gains.append((kp, ki, kd, tau))
    return gains


@needs_robot
@pytest.mark.parametrize("explicit_limits", (False, True))
def test_batched_pid_matches_robot_pid(explicit_limits):
    """Saturating, unwinding and resetting sequences match the scalar robot PID step for step."""
    PIDController = robot_pid_class()
    rng = np.random.default_rng(3)
    envs, joints, steps = 6, 3, 600
    gains = random_gains(rng, envs * joints)
    limits = [(-0.3, 0.2)] * len(gains) if explicit_limits else [(None, None)] * len(gains)
    refs = [
        PIDController(
            kp=kp, ki=ki, kd=kd, min_output=-1.0, max_output=1.0, deriv_filter_tau=tau, Imin=lo, Imax=hi
        )
        for (kp, ki, kd, tau), (lo, hi) in zip(gains, limits)
    ]
    table = torch.tensor(gains, dtype=torch.float64).reshape(envs, joints, 4)
    pid = BatchedPID(
        envs,
        joints,
        table[..., 0],
        table[..., 1],
        table[..., 2],
        deriv_filter_tau=table[..., 3],
        i_min=-0.3 if explicit_limits else None,
        i_max=0.2 if explicit_limits else None,
        dtype=torch.float64,
    )
    # Large steps drive saturation and windup; the slow wander gives the D filter something to track.
    setpoint = np.repeat(rng.uniform(-1, 1, (steps // 100, envs, joints)), 100, axis=0)
    current = np.cumsum(rng.normal(0, 0.02, (steps, envs, joints)), axis=0)
    dts = rng.uniform(0.0, 0.02, steps)  # includes dt below the 1 ms floor
    for t in range(steps):
        if t == 350:
            pid.reset(torch.tensor([1, 4]))
            for e in (1, 4):
                for j in range(joints):
                    refs[e * joints + j].reset()
        out = pid.compute(torch.from_numpy(setpoint[t]), torch.from_numpy(current[t]), dts[t])
        expected = [
            ref.compute(float(setpoint[t].flat[i]), float(current[t].flat[i]), dt=float(dts[t]))
            for i, ref in enumerate(refs)
        ]
        np.testing.assert_allclose(out.numpy().ravel(), expected, rtol=0, atol=1e-12, err_msg=f"step {t}")
    integrals = [ref.integral_sum for ref in refs]
    np.testing.assert_allclose(pid.integral.numpy().ravel(), integrals, rtol=0, atol=1e-12)


@needs_robot
def test_robot_call_convention_wraps_the_error():
    """``robot_joint_command`` feeds the PID like ExcavatorController: setpoint 0, measurement -wrap(target - q)."""
    PIDController = robot_pid_class()
    ref = PIDController(kp=10.0, ki=1.0, kd=0.2, min_output=-1.0, max_output=1.0)
    pid = BatchedPID(1, 1, 10.0, 1.0, 0.2, dtype=torch.float64)
    q = math.pi - 0.01
    for target in (-math.pi + 0.02, -math.pi + 0.05, 0.3):
        error = math.atan2(math.sin(target - q), math.cos(target - q))
        expected = ref.compute(0.0, -error, dt=0.01)
        out = robot_joint_command(
            pid, torch.tensor([[target]], dtype=torch.float64), torch.tensor([[q]], dtype=torch.float64), 0.01
        )
        assert out.item() == pytest.approx(expected, abs=1e-12)


def test_float32_gpu_layout_runs_independent_gain_sets():
    """Each environment owns its gains: a zero-gain row stays neutral while its neighbour acts."""
    pid = BatchedPID(2, 3, torch.tensor([[0.0], [10.0]]), 0.0, 0.0)
    out = pid.compute(torch.full((2, 3), 0.05), torch.zeros(2, 3), 0.01)
    assert torch.equal(out[0], torch.zeros(3))
    assert torch.allclose(out[1], torch.full((3,), 0.5))


def test_trapezoid_reaches_the_distance_without_exceeding_speed_or_accel():
    from hydraulic_controller.pid_replay import trapezoid

    t = torch.arange(0, 400) * 0.01
    for distance, speed in ((0.10, 0.07), (0.01, 0.07)):  # trapezoidal, then triangular
        s = trapezoid(distance, speed, 0.5, t)
        v = torch.diff(s) / 0.01
        assert s[-1].item() == pytest.approx(distance, abs=1e-6)
        assert v.max().item() <= speed + 1e-4
        assert torch.diff(v).abs().max().item() / 0.01 <= 0.5 + 1e-2


def test_dls_step_iterates_onto_the_target():
    from hydraulic_controller.core import DEFAULT_ASSET, HOME
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.pid_replay import dls_step

    kin = ExcavatorKinematics(DEFAULT_ASSET, "cpu")
    q = torch.tensor([HOME])
    target = kin.pose_jacobian(q)[0] + torch.tensor([[0.03, -0.02, 0.05]])
    for _ in range(10):
        q[:, :3] += dls_step(kin, q, target, 1e-3)
    assert torch.allclose(kin.pose_jacobian(q)[0], target, atol=1e-5)


def test_replay_with_zero_gains_leaves_the_machine_at_rest():
    """With the PID silenced the valves stay neutral and the steady model must not drift."""
    from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL, HOME
    from hydraulic_controller.pid_replay import PidGains, ReplayConfig, metrics, run_replay

    if not DEFAULT_MODEL.exists():
        pytest.skip("V5 steady model artifact is not installed")
    cfg = ReplayConfig(plants=("nominal",), duration_s=1.5, tail_s=0.5, tip_speeds_m_s=(0.02,))
    scenarios, traces = run_replay(
        PidGains([0.0] * 3, [0.0] * 3, [0.0] * 3), cfg, DEFAULT_MODEL, DEFAULT_ASSET, "cpu"
    )
    assert torch.equal(traces["u"], torch.zeros_like(traces["u"]))
    assert (traces["q"] - torch.tensor(HOME[:3])).abs().max() < 1e-4
    rows = metrics(cfg, scenarios, traces)
    assert len(rows) == len(scenarios) and not any(r["invalid"] for r in rows)
