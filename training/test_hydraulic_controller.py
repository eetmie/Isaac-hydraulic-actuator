"""Contracts shared by controller training, benchmark and custom playback."""

from pathlib import Path

import numpy as np
import pytest
import torch

from actuators import HydraulicActuatorNet
from hydraulic_controller.core import HOME, ControllerSettings, HydraulicModelBatch, HydraulicPlant
from hydraulic_controller.kinematics import CommandGovernor, ExcavatorKinematics
from hydraulic_controller.trajectory import (
    DrawingSession,
    StrokeFollower,
    StrokeUnreachableError,
    quintic,
    sample_segments,
    simplify_stroke,
    smooth_stroke,
    validate_stroke,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/arm_v5_steady"
ASSET = ROOT / "assets/excavator_bucket_rocking.usd"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="V5 steady model artifact is not installed")


@pytest.fixture(scope="module")
def kin():
    return ExcavatorKinematics(ASSET, "cpu")


@needs_model
def test_batched_hydraulic_history_matches_original_wrapper():
    """Vectorizing environments must not change the trained recurrence."""
    originals = [HydraulicActuatorNet(MODEL, sim_dt=0.01) for _ in range(2)]
    batch = HydraulicModelBatch(MODEL, 2, "cpu")
    rng = np.random.default_rng(42)
    q = np.array([HOME, np.asarray(HOME) + [0.1, -0.1, 0.05, 0]], dtype=np.float32)
    v = np.zeros((2, 4), dtype=np.float32)
    for _ in range(50):
        u = rng.uniform(-1, 1, size=(2, 3)).astype(np.float32)
        q += rng.normal(0, 1e-4, size=q.shape).astype(np.float32)
        expected = np.stack([originals[i].predict_velocity(q[i], v[i], u[i]) for i in range(2)])
        actual = batch.predict(torch.from_numpy(q), torch.from_numpy(v), torch.from_numpy(u)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
        v = expected


@needs_model
def test_kinematics_tip_and_governor_contract(kin):
    q = torch.tensor([HOME])
    assert kin.valid(q).item()
    pose, jacobian = kin.pose_jacobian(q)
    assert pose.shape == (1, 3)
    assert jacobian.shape == (1, 3, 4)
    assert torch.isfinite(jacobian).all()
    plant = HydraulicPlant(MODEL, kin, 1, "cpu")
    contract = plant.contract()
    assert plant.observe(torch.zeros(1, 3)).shape == (1, contract["observation_dim"])
    request = torch.tensor([[0.5, -0.1, 0.9]])
    settings = ControllerSettings()
    admitted, intervention = CommandGovernor(kin, settings)(q, torch.zeros_like(q), request)
    assert admitted[0, :2].norm() <= settings.speed_max + 1e-6
    assert admitted[0, 2].abs() <= settings.pitch_rate_max + 1e-6
    assert 0 <= intervention.item() <= 1


@needs_model
def test_tracked_twist_excludes_carriage_rocking(kin):
    plant = HydraulicPlant(MODEL, kin, 1, "cpu")
    plant.v[:] = torch.tensor([[0.0, 0.0, 0.0, 0.3]])
    _, twist = plant.tip_state()
    assert torch.equal(twist, torch.zeros_like(twist))
    plant.v[:] = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    assert plant.tip_state()[1][0, :2].norm() > 0


@needs_model
def test_hidden_valve_perturbation_reaches_model_but_not_observation(kin):
    plant = HydraulicPlant(MODEL, kin, 2, "cpu")
    plant.valve_gain[1] = 0.5
    plant.valve_offset[1] = 0.1
    plant.action_delay[1] = 2
    plant.begin_action(torch.full((2, 3), 0.5))
    commanded = torch.tanh(torch.tensor(0.5))
    plant.step()
    assert torch.allclose(plant.u[0], commanded.expand(3))
    assert torch.allclose(plant.u[1], torch.full((3,), 0.1))  # delayed: still the neutral command plus offset
    plant.step()
    plant.step()
    assert torch.allclose(plant.u[1], (0.5 * commanded + 0.1).expand(3))
    assert torch.equal(plant.u_cmd_history[0], plant.u_cmd_history[1])


def test_collision_grid_is_conservative(kin, tmp_path):
    kin.build_collision_grid(resolution=24, pitches=(0.0,), cache_dir=tmp_path)
    generator = torch.Generator().manual_seed(3)
    lo, hi = kin.limits[:, 0], kin.limits[:, 1]
    q = lo + (hi - lo) * torch.rand(4000, 4, generator=generator)
    q[:, 3] = 0.0
    exact = kin._colliding_exact(q)
    assert exact.any(), "sample should include colliding poses"
    assert not (exact & ~kin.colliding(q)).any()
    kin._grid = None


@needs_model
def test_training_environment_steps_and_resets():
    pytest.importorskip("rsl_rl")
    from hydraulic_controller.env import HydraulicControlEnv, HydraulicControlEnvCfg

    cfg = HydraulicControlEnvCfg(num_envs=8, device="cpu", episode_length_s=0.2)
    env = HydraulicControlEnv(cfg)
    obs = env.get_observations()
    assert obs["policy"].shape == (8, env.plant.contract()["observation_dim"])
    assert obs["critic"].shape[1] == obs["policy"].shape[1] + 10
    for _ in range(4):
        obs, reward, dones, extras = env.step(torch.zeros(8, 3))
        assert torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all()
    assert dones.all() and extras["time_outs"].all()
    assert (env.episode_length_buf == 0).all()
    assert env.command[:, :2].norm(dim=1).max() <= cfg.speed_max + 1e-6
    # New checkpoints must stay loadable after the repository is moved or cloned.
    assert env.plant.contract()["model_path"] == "models/arm_v5_steady"
    assert env.plant.contract()["asset_path"] == "assets/excavator_bucket_rocking.usd"


@needs_model
def test_shipped_prototype_controller_loads_from_any_clone():
    """The demo default must not depend on this machine's paths or local training logs."""
    pytest.importorskip("rsl_rl")
    import hashlib
    import json

    from hydraulic_controller.policy import ControllerPolicy

    folder = ROOT / "models/controller_proto"
    manifest = json.loads((folder / "release_manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
    contract = json.loads((folder / "controller_contract.json").read_text())
    assert not Path(contract["model_path"]).is_absolute()
    assert not Path(contract["asset_path"]).is_absolute()
    policy = ControllerPolicy(folder / "model_1798.pt")
    assert policy.model_path == MODEL and policy.asset_path == ASSET
    actions = policy(torch.zeros(1, contract["observation_dim"]))
    assert actions.shape == (1, 3) and torch.isfinite(actions).all()


def test_quintic_time_scaling_is_rest_to_rest():
    s, s_dot = quintic(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(s_dot[[0, 2]], [0.0, 0.0])
    assert s_dot[1] == pytest.approx(1.875)


def test_follower_progress_is_constant_speed_and_independent_of_measured_state():
    path = np.array([[0.0, 0.0], [0.02, 0.0], [0.02, 0.02]])
    first = StrokeFollower(path, 0.01)
    second = StrokeFollower(path, 0.01)
    for _ in range(10):
        command_a = first.command(np.array([10.0, -10.0]), np.array([3.0, 4.0]), 0.02)
        command_b = second.command(np.array([-2.0, 8.0]), np.array([-7.0, 1.0]), 0.02)
        np.testing.assert_allclose(command_a, command_b)
        np.testing.assert_allclose(first.reference, second.reference)
        assert np.linalg.norm(command_a) == pytest.approx(0.01)
    assert first.progress == pytest.approx(0.002)


def test_accelerated_follower_starts_and_ends_at_rest():
    follower = StrokeFollower(np.array([[0.0, 0.0], [0.05, 0.0]]), 0.03, accel=0.25)
    speeds = []
    while not follower.finished:
        speeds.append(np.linalg.norm(follower.command(np.zeros(2), np.zeros(2), 0.05)))
    assert speeds[0] <= 0.25 * 0.05 + 1e-9
    assert max(speeds) == pytest.approx(0.03)
    assert speeds[-1] < 0.03
    assert follower.progress == pytest.approx(0.05)


def test_session_speed_change_retimes_the_running_path_at_bounded_acceleration():
    session = DrawingSession(np.array([0.0, 0.0, 0.0]), 0.06, kp=0.0, accel=0.25)
    session.submit(np.array([[0.0, 0.0], [0.3, 0.0]]), np.zeros(2))

    def speed():
        return np.linalg.norm(session.command(np.r_[session.reference, 0.0], np.zeros(3), 0.05)[:2])

    assert [speed() for _ in range(6)][-1] == pytest.approx(0.06)
    session.set_speed(0.02)
    slowing = [speed() for _ in range(5)]
    np.testing.assert_allclose(slowing, [0.0475, 0.035, 0.0225, 0.02, 0.02])
    session.set_speed(0.03)
    assert speed() == pytest.approx(0.03)
    with pytest.raises(ValueError):
        session.set_speed(0.0)


def test_drawing_session_corrects_position_error():
    session = DrawingSession(np.array([0.4, 0.0, 0.0]), 0.03, kp=3.0)
    command = session.command(np.array([0.41, -0.01, 0.1]), np.zeros(3), 0.05)
    np.testing.assert_allclose(command[:2], [-0.03, 0.03])
    assert command[2] == pytest.approx(-0.3)


def test_follower_display_target_stays_on_path_and_does_not_move_backwards():
    follower = StrokeFollower(np.array([[0.0, 0.0], [0.02, 0.0]]), 0.01)
    velocity = np.zeros(2)
    follower.command(np.array([0.004, 0.003]), velocity, 0.02)
    first = follower.display_target.copy()
    follower.command(np.array([0.002, -0.004]), velocity, 0.02)
    second = follower.display_target.copy()
    assert first[1] == pytest.approx(0.0)
    assert second[1] == pytest.approx(0.0)
    assert second[0] >= first[0]


def test_mouse_stroke_smoothing_reduces_jitter_and_preserves_endpoints():
    x = np.linspace(0.0, 0.05, 51)
    raw = np.column_stack((x, 0.0015 * np.sin(np.arange(len(x)) * np.pi / 2)))
    smoothed = smooth_stroke(raw)
    np.testing.assert_allclose(smoothed[[0, -1]], raw[[0, -1]])
    dense = sample_segments(smoothed, spacing=0.001)
    assert np.max(np.abs(dense[1:-1, 1])) < np.max(np.abs(raw[1:-1, 1]))
    assert len(simplify_stroke(np.array([[0, 0], [0.01, 0], [0.02, 0]], dtype=float))) == 2


@needs_model
def test_speed_scale_scales_integrated_motion(kin):
    plant = HydraulicPlant(MODEL, kin, 2, "cpu")
    plant.speed_scale[1, :3] = 0.5
    start = plant.q.clone()
    plant.begin_action(torch.full((2, 3), 1.0))
    for _ in range(40):
        plant.step()
    moved = (plant.q - start)[:, :3]
    assert (moved[0].abs() > 1e-3).all()
    # The network also sees joint position, so the ratio is close to, not exactly, the scale.
    ratio = moved[1] / moved[0]
    assert ((ratio > 0.45) & (ratio < 0.55)).all()
    torch.testing.assert_close(plant.v[1, :3], 0.5 * plant._v_model[1, :3])


def test_governor_scales_twist_to_joint_speed_limits(kin):
    q = torch.tensor([HOME])
    settings = ControllerSettings()
    limits = torch.full((3, 2), 0.05)
    request = torch.tensor([[0.1, 0.05, 0.0]])
    admitted, _ = CommandGovernor(kin, settings, limits, speed_margin=0.8)(q, torch.zeros_like(q), request)
    _, jacobian = kin.pose_jacobian(q)
    rates = torch.linalg.solve(jacobian[:, :, :3], admitted[:, :, None]).squeeze(-1)
    assert rates.abs().max() <= 0.8 * 0.05 + 1e-4
    direction = admitted[0, :2] / admitted[0, :2].norm()
    torch.testing.assert_close(direction, request[0, :2] / request[0, :2].norm(), atol=1e-3, rtol=0)


def test_drawing_box_encloses_home_tip(kin):
    q = torch.tensor([HOME])
    pose, _ = kin.pose_jacobian(q)
    box = kin.drawing_box(q)
    x, z = pose[0, :2].tolist()
    assert box[0] + 0.05 < x < box[1] - 0.05
    assert box[2] + 0.05 < z < box[3] - 0.05


def test_stroke_validation_reports_where_it_is_out_of_reach(kin):
    q = torch.tensor([HOME])
    pose, _ = kin.pose_jacobian(q)
    box = kin.drawing_box(q)
    angle = float(pose[0, 2])
    home = pose[0, :2].numpy().astype(np.float64)
    path = validate_stroke(kin, q, home + np.array([[-0.05, -0.05], [-0.10, -0.10]]), box, angle)
    np.testing.assert_allclose(path[[0, -1]], home + np.array([[-0.05, -0.05], [-0.10, -0.10]]), atol=1e-7)
    # At the home bucket angle the boom reaches its lower limit before the upper left of the box.
    with pytest.raises(StrokeUnreachableError, match="keep the stroke") as rejected:
        validate_stroke(kin, q, np.array([[0.45, 0.0], [0.40, 0.15]]), box, angle)
    assert len(rejected.value.points) and (rejected.value.points[:, 1] > 0.03).all()
    # From the narrow reachable wedge above home, a straight move to the lower left cuts the unreachable corner.
    raised, solved = kin.inverse(torch.tensor([[0.62, 0.22, angle]]), q)
    assert solved.all()
    with pytest.raises(StrokeUnreachableError, match="straight move"):
        validate_stroke(kin, raised, np.array([[0.40, 0.0], [0.40, -0.05]]), box, angle)


def test_reachable_cells_tile_the_box_and_agree_with_validation(kin):
    q = torch.tensor([HOME])
    pose, _ = kin.pose_jacobian(q)
    box = np.array([0.32, 0.72, -0.30, 0.20])
    cells = kin.reachable_cells(q, box, float(pose[0, 2]), spacing=0.02)
    assert cells.shape == (20, 25)
    home = ((pose[0, :2].numpy() - box[[0, 2]]) / 0.02).astype(int)
    assert cells[home[0], home[1]]
    assert not cells[2, -2]  # (370, 170) mm, the unreachable upper left
    assert 0.5 < cells.mean() < 0.9


def test_drawing_window_keeps_rejection_visible_during_telemetry():
    """A rejected stroke must stay explained; 20 Hz state messages used to overwrite the error at once."""
    import queue
    import tkinter

    from hydraulic_controller.draw_ui import DrawingWindow

    outgoing, incoming = queue.Queue(), queue.Queue()
    try:
        window = DrawingWindow(outgoing, incoming, [0.32, 0.72, -0.30, 0.20], 30.0)
    except tkinter.TclError:
        pytest.skip("Tk needs a display")
    try:
        # Mirrored like the Isaac camera view: the far end of the box is on the left.
        left, top, right, bottom = window.bounds
        assert window.to_canvas([0.72, 0.20]) == pytest.approx((left, top))
        assert window.to_world(right, bottom) == pytest.approx([0.32, -0.30])
        assert window.to_world(*window.to_canvas([0.5, -0.1])) == pytest.approx([0.5, -0.1])
        incoming.put({"kind": "error", "text": "Out of reach"})
        state = {"position": [0.6, 0.0], "target": [0.6, 0.0], "error_mm": 0.0, "speed_mm_s": 0.0}
        incoming.put({"kind": "state", "phase": "hold", "valves": [0.0, 0.0, 0.0], **state})
        window.poll()
        assert window.status.get() == "Out of reach"
    finally:
        window.root.destroy()


def test_drawing_window_speed_slider_sends_only_moved_speeds():
    import queue
    import tkinter

    from hydraulic_controller.draw_ui import DrawingWindow

    outgoing, incoming = queue.Queue(), queue.Queue()
    try:
        window = DrawingWindow(outgoing, incoming, [0.32, 0.72, -0.30, 0.20], 25.5, max_speed_mm_s=80.0)
    except tkinter.TclError:
        pytest.skip("Tk needs a display")
    try:
        window.root.update()
        assert outgoing.empty(), "opening the window must not round and resend --speed-mm-s"
        assert window.speed_text.get() == "25.5 mm/s"
        assert window.speed_scale.cget("to") == 80
        window.speed_scale.set(50)
        window.root.update()
        assert outgoing.get_nowait() == {"kind": "speed", "mm_s": 50.0}
        assert window.speed_text.get() == "50 mm/s"
    finally:
        window.root.destroy()
