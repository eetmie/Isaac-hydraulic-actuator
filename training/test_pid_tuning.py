"""The tuner's pieces: the optimizer, the gain encoding, the score and batched candidate rollouts."""

import numpy as np
import pytest
import torch

from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL
from hydraulic_controller.pid_replay import PidGains, ReplayConfig, Scenario, prepare, rollout
from hydraulic_controller.pid_tuning import CMAES, TuneConfig, score, to_gains, to_vector, vector_to_pid_gains

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def test_cmaes_solves_an_ill_conditioned_shifted_quadratic():
    target = np.array([1.0, -2.0, 0.5, 3.0, -1.0])
    scales = np.array([1.0, 10.0, 100.0, 1.0, 0.1])
    es = CMAES(np.zeros(5), 1.0, seed=1)
    for _ in range(300):
        x = es.ask()
        es.tell(x, (((x - target) * scales) ** 2).sum(1))
    assert np.allclose(es.mean, target, atol=1e-3)


def test_gain_vectors_round_trip_and_zero_gains_sit_on_the_lower_bound():
    cfg = TuneConfig()
    gains = PidGains(kp=[10.0, 10.0, 5.0], ki=[0.4, 0.4, 0.25], kd=[0.0, 0.2, 0.0])
    vector = to_vector(gains, cfg)
    kp, ki, kd = to_gains(vector[None], "cpu")
    assert torch.allclose(kp[0], torch.tensor([10.0, 10.0, 5.0]))
    assert torch.allclose(ki[0], torch.tensor([0.4, 0.4, 0.25]))
    assert torch.allclose(kd[0], torch.tensor([1e-3, 0.2, 1e-3]))
    assert vector_to_pid_gains(vector, gains).kp == [10.0, 10.0, 5.0]


def test_score_mixes_the_plant_mean_with_the_worst_plant():
    class Prep:
        scenarios = [Scenario("joint_step", "boom", 3.0, plant) for plant in ("nominal", "nominal", "slow")]

    zeros = torch.zeros(1, 3)
    terms = {key: zeros.clone() for key in ("final", "overshoot", "tail_p2p", "valve_travel", "invalid")}
    terms["track"] = torch.tensor([[1.0, 3.0, 6.0]])  # nominal mean 2, slow 6
    cfg = TuneConfig(worst_plant_weight=0.5)
    assert score(Prep, terms, cfg).item() == pytest.approx(0.5 * 4.0 + 0.5 * 6.0)


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="V5 steady model artifact is not installed")
def test_batched_candidates_score_like_single_runs():
    """Candidate c in a batch must see exactly what it would see alone (no cross-talk between rows)."""
    cfg = ReplayConfig(
        plants=("nominal", "slow_valves"),
        duration_s=2.0,
        tail_s=0.5,
        tip_speeds_m_s=(0.02,),
        sensor_noise_deg=0.02,
    )
    prep = prepare(cfg, DEFAULT_MODEL, DEFAULT_ASSET, DEVICE)
    base = PidGains([10.0, 10.0, 5.0], [0.4, 0.4, 0.25], [0.0, 0.0, 0.0])
    kp = torch.tensor([[10.0, 10.0, 5.0], [20.0, 5.0, 12.0]], device=DEVICE)
    ki = torch.tensor([[0.4, 0.4, 0.25], [1.0, 0.1, 2.0]], device=DEVICE)
    kd = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.1]], device=DEVICE)
    batched, _ = rollout(prep, kp, ki, kd, base, seed=3)
    for c in range(2):
        alone, _ = rollout(prep, kp[c : c + 1], ki[c : c + 1], kd[c : c + 1], base, seed=3)
        for key, value in alone.items():
            torch.testing.assert_close(batched[key][c], value[0], rtol=1e-4, atol=1e-5)
