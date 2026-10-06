"""The speed map must describe what the plant really does, not just what the solver believes."""

from pathlib import Path

import pytest
import torch

from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL, HOME, HydraulicPlant
from hydraulic_controller.kinematics import ExcavatorKinematics
from hydraulic_controller.speed_limits import (
    SpeedMapConfig,
    achievable_tip_speeds,
    curve_summary,
    direction_vectors,
    on_direction,
)

needs_model = pytest.mark.skipif(
    not Path(DEFAULT_MODEL).exists(), reason="V5 steady model artifact is not installed"
)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_on_direction_rejects_sideways_and_turning_twists():
    cfg = SpeedMapConfig()
    directions = direction_vectors([0, 90], "cpu")
    twist = torch.tensor(
        [
            [0.05, 0.004, 0.0],  # +X within 10% cross
            [0.05, 0.010, 0.0],  # +X with 20% cross
            [0.05, 0.0, 0.05],  # +X but the bucket turns 1 rad per metre
            [-0.05, 0.0, 0.0],  # backwards
        ]
    )
    along, ok = on_direction(twist, directions, cfg)
    assert along[0, 0].item() == pytest.approx(0.05)
    assert ok[:, 0].tolist() == [True, False, False, False]
    assert not ok[:, 1].any()


def test_curve_summary_finds_the_deadband_edge():
    u = torch.linspace(-1, 1, 201)
    speed = (
        torch.where(u.abs() > 0.3, (u.abs() - 0.3) / 0.7, 0.0)
        * u.sign()
        * torch.tensor([[0.2], [0.4], [0.8]])
    )
    summary = curve_summary(u, speed)
    for joint in ("boom", "arm", "bucket"):
        assert summary[f"{joint}_pos"]["deadband_u"] == pytest.approx(0.37, abs=0.011)  # 10% of full speed
        assert summary[f"{joint}_neg"]["deadband_u"] == pytest.approx(0.37, abs=0.011)
    assert summary["bucket_pos"]["full_valve_deg_s"] == pytest.approx(45.84, rel=1e-3)


@needs_model
def test_solved_valves_move_the_free_plant_as_promised():
    """Replaying the solved steady valves on an unpinned plant must go the requested way at about that speed."""
    cfg = SpeedMapConfig(directions_deg=(270, 0), random_valves=64)
    kin = ExcavatorKinematics(DEFAULT_ASSET, DEVICE)
    home = torch.tensor([HOME], device=DEVICE)
    directions = direction_vectors(cfg.directions_deg, DEVICE)
    speed, valves = achievable_tip_speeds(DEFAULT_MODEL, kin, home, directions, cfg)
    assert (speed[0] > 0.02).all()  # -Z and +X are both reachable from HOME (~155 and ~40 mm/s)

    plant = HydraulicPlant(DEFAULT_MODEL, kin, 2, DEVICE)
    plant.reset(torch.arange(2, device=DEVICE), home.repeat(2, 1))
    twists = []
    for step in range(45):
        plant.u_cmd.copy_(valves[0])
        plant.step()
        if step >= 20:  # past the ~0.2 s valve/flow rise
            twists.append(plant.tip_state()[1])
    twist = torch.stack(twists).mean(0)
    along = (twist[:, :2] * directions).sum(1)
    cross = (twist[:, :2] - along[:, None] * directions).norm(dim=1)
    # The free tip leaves the pose the speed was measured at (the arm turns ~5 deg by the end of the window).
    assert torch.allclose(along, speed[0], rtol=0.3)
    assert (cross <= 0.3 * along).all()
