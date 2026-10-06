"""Opt-in training options: several plant networks, achievable-relative commands, linear error penalties."""

import math
from pathlib import Path

import pytest
import torch

from hydraulic_controller.core import DEFAULT_ASSET, HOME, HydraulicModelBatch, MixedModelBatch
from hydraulic_controller.kinematics import ExcavatorKinematics
from hydraulic_controller.speed_limits import SpeedTable

ROOT = Path(__file__).resolve().parents[1]
STEADY, V4, V5 = (ROOT / "models" / name for name in ("arm_v5_steady", "arm_v4", "arm_v5"))
needs_models = pytest.mark.skipif(
    not all(p.exists() for p in (STEADY, V4, V5)), reason="model artifacts missing"
)


@needs_models
def test_mixed_batch_runs_each_environment_on_its_own_network():
    mixed = MixedModelBatch([STEADY, V4], 4, "cpu")
    singles = {0: HydraulicModelBatch(STEADY, 4, "cpu"), 1: HydraulicModelBatch(V4, 4, "cpu")}
    generator = torch.Generator().manual_seed(0)
    q = torch.tensor(HOME).repeat(4, 1)
    v = torch.zeros(4, 4)
    for _ in range(30):
        u = torch.rand(4, 3, generator=generator) * 2 - 1
        got = mixed.predict(q, v, u)
        expected = {k: model.predict(q, v, u) for k, model in singles.items()}
        for row in range(4):
            torch.testing.assert_close(got[row], expected[row % 2][row])
        v = got
    assert not torch.allclose(expected[0], expected[1])  # the two twins really differ


@needs_models
def test_plant_contract_names_the_extra_models():
    from hydraulic_controller.core import HydraulicPlant

    kin = ExcavatorKinematics(DEFAULT_ASSET, "cpu")
    contract = HydraulicPlant(STEADY, kin, 3, "cpu", extra_models=(V4, V5)).contract()
    assert contract["extra_model_paths"] == ["models/arm_v4", "models/arm_v5"]
    assert "extra_model_paths" not in HydraulicPlant(STEADY, kin, 3, "cpu").contract()


def test_speed_table_interpolates_and_wraps_direction():
    axes = [torch.linspace(0.0, 1.0, 3) for _ in range(3)]
    directions = torch.tensor([0.0, 90.0, 180.0, 270.0])
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # [3, 3, 3, 3]
    speed = grid.sum(-1, keepdim=True) * torch.tensor([1.0, 2.0, 3.0, 4.0])  # linear in q, per direction
    table = SpeedTable(axes, directions, speed)
    q = torch.tensor([[0.5, 0.0, 1.0, 0.0], [0.25, 0.25, 0.25, 0.0], [5.0, -1.0, 0.5, 0.0]])
    east = torch.zeros(3)
    torch.testing.assert_close(
        table.lookup(q, east), torch.tensor([1.5, 0.75, 1.5])
    )  # last row clamps to the grid
    torch.testing.assert_close(
        table.lookup(q[:1], torch.tensor([math.radians(45)])), torch.tensor([1.5 * 1.5])
    )
    torch.testing.assert_close(
        table.lookup(q[:1], torch.tensor([math.radians(315)])), torch.tensor([1.5 * 2.5])
    )


@needs_models
def test_achievable_commands_stay_inside_their_fraction_of_the_table(monkeypatch):
    from hydraulic_controller.env import HydraulicControlEnv, HydraulicControlEnvCfg

    flat = SpeedTable(
        [torch.linspace(-3, 3, 2) for _ in range(3)],
        torch.tensor([0.0, 180.0]),
        torch.full((2, 2, 2, 2), 0.05),
    )
    import hydraulic_controller.speed_limits as speed_limits

    monkeypatch.setattr(speed_limits.SpeedTable, "load_or_build", classmethod(lambda cls, *a, **k: flat))
    cfg = HydraulicControlEnvCfg(
        num_envs=512,
        device="cpu",
        command_speed_mode="achievable",
        over_limit_prob=0.0,
        zero_speed_prob=0.0,
        uniform_mix_prob=0.0,
    )
    env = HydraulicControlEnv(cfg)
    env._sample_targets(torch.arange(512))
    speed = env.target[:, :2].norm(dim=1)
    assert (speed >= 0.05 * 0.05 - 1e-6).all() and (speed <= 0.9 * 0.05 + 1e-6).all()


def test_default_config_keeps_the_original_recipe():
    from hydraulic_controller.env import HydraulicControlEnvCfg

    cfg = HydraulicControlEnvCfg()
    assert cfg.extra_model_paths == () and cfg.command_speed_mode == "uniform"
    assert cfg.linear_error_weight == 0.0 and cfg.angular_error_weight == 0.0
