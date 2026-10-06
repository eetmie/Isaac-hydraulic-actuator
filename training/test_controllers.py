"""Controllers in ``closed_loop`` must behave exactly like the loops they stand for."""

from pathlib import Path

import pytest
import torch

from hydraulic_controller.closed_loop import rollout
from hydraulic_controller.core import DEFAULT_ASSET, DEFAULT_MODEL, HydraulicPlant, wrap_angle
from hydraulic_controller.tasks import TaskConfig, prepare

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "models/controller_proto/model_1798.pt"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
needs_proto = pytest.mark.skipif(
    not (PROTO.exists() and DEFAULT_MODEL.exists()), reason="prototype controller or plant not installed"
)


def tip_task(**overrides):
    settings = dict(
        families=("tip_line",), plants=("nominal",), tip_speeds_m_s=(0.04,), duration_s=2.0, tail_s=0.5
    )
    cfg = TaskConfig(**{**settings, **overrides})
    return prepare(cfg, DEFAULT_MODEL, DEFAULT_ASSET, DEVICE)


@needs_proto
def test_policy_controller_matches_the_plant_observation_loop():
    """Measured-history observations reproduce ``plant.observe``, so valves match the benchmark loop tick for tick."""
    from hydraulic_controller.policy import ControllerPolicy, PolicyController

    policy = ControllerPolicy(PROTO, DEVICE)
    prep = tip_task()
    _, traces = rollout(prep, PolicyController(policy), record=True)

    kin, n = prep.kin, len(prep.scenarios)
    plant = HydraulicPlant(DEFAULT_MODEL, kin, n, DEVICE, policy.settings)
    plant.reset(torch.arange(n, device=DEVICE), prep.q0)
    governor = policy.governor(kin)
    v_ref = torch.nan_to_num(torch.diff(prep.tip_ref, dim=0, prepend=prep.tip_ref[:1]) / 0.01)
    expected = []
    for k in range(prep.q_ref.shape[0]):
        if k % policy.settings.decimation == 0:
            pose, _ = plant.tip_state()
            requested = torch.zeros(n, 3, device=DEVICE)
            requested[:, :2] = v_ref[k, :, :2] + 3.0 * (prep.tip_ref[k, :, :2] - pose[:, :2])
            requested[:, 2] = 3.0 * wrap_angle(prep.tip_ref[k, :, 2] - pose[:, 2])
            command, _ = governor(plant.q, plant.v, requested)
            plant.begin_action(policy(plant.observe(command)))
        plant.step()
        expected.append(plant.u_cmd.clone().cpu())
    torch.testing.assert_close(traces["u"], torch.stack(expected), rtol=1e-4, atol=1e-5)


@needs_proto
def test_policy_controller_refuses_joint_tasks():
    from hydraulic_controller.policy import ControllerPolicy, PolicyController

    cfg = TaskConfig(
        families=("joint_step",), plants=("nominal",), step_deg=(3.0,), duration_s=0.5, tail_s=0.1
    )
    prep = prepare(cfg, DEFAULT_MODEL, DEFAULT_ASSET, DEVICE)
    with pytest.raises(ValueError, match="tip-line"):
        rollout(prep, PolicyController(ControllerPolicy(PROTO, DEVICE)))


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="V5 steady model artifact is not installed")
def test_mpc_model_seeded_from_measurements_predicts_the_plant_exactly():
    """With the nominal network, the MPC's first prediction from the measured history is the plant's next step."""
    from hydraulic_controller.closed_loop import TaskBatch
    from hydraulic_controller.mpc import MPPIConfig, MPPIController

    prep = tip_task()
    n = len(prep.scenarios)
    mpc = MPPIController(DEFAULT_MODEL, MPPIConfig(samples=1))
    mpc.reset(
        TaskBatch(prep.kin, prep.q0, prep.q_ref, prep.tip_ref, torch.ones(n, dtype=torch.bool, device=DEVICE))
    )
    plant = HydraulicPlant(DEFAULT_MODEL, prep.kin, n, DEVICE)
    plant.reset(torch.arange(n, device=DEVICE), prep.q0)
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    previous = torch.zeros(n, 3, device=DEVICE)
    for _ in range(120):  # longer than every history window, with real motion in it
        mpc.history.push(plant.q, plant.v, previous)
        u = torch.rand(n, 3, generator=generator, device=DEVICE) * 1.2 - 0.6
        q, vel = mpc.seed_model()
        predicted = mpc.predict(q, vel, u).clamp(-2.0, 2.0)
        plant.u_cmd.copy_(u)
        plant.step()
        torch.testing.assert_close(predicted, plant.v, rtol=1e-4, atol=1e-5)
        previous = u


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="V5 steady model artifact is not installed")
def test_mpc_deadband_map_is_odd_continuous_and_starts_at_the_edge():
    from hydraulic_controller.closed_loop import TaskBatch
    from hydraulic_controller.mpc import MPPIController

    prep = tip_task()
    n = len(prep.scenarios)
    mpc = MPPIController(DEFAULT_MODEL)
    mpc.reset(
        TaskBatch(prep.kin, prep.q0, prep.q_ref, prep.tip_ref, torch.ones(n, dtype=torch.bool, device=DEVICE))
    )
    w = torch.linspace(-1, 1, 4001, device=DEVICE)[:, None].repeat(1, 3)
    u = mpc.valves(w)
    assert torch.equal(mpc.valves(torch.zeros(1, 3, device=DEVICE)), torch.zeros(1, 3, device=DEVICE))
    assert torch.allclose(u[[0, -1]].abs(), torch.ones(2, 3, device=DEVICE))
    assert (torch.diff(u, dim=0) >= 0).all() and torch.diff(u, dim=0).abs().max() < 0.01  # monotone, no jumps
    ramp = mpc.cfg.ramp_w
    edge_pos = mpc.valves(torch.full((1, 3), ramp, device=DEVICE))[0]
    edge_neg = -mpc.valves(torch.full((1, 3), -ramp, device=DEVICE))[0]
    assert torch.allclose(edge_pos, mpc.edges[1]) and torch.allclose(edge_neg, mpc.edges[0])
    assert ((mpc.edges > 0.1) & (mpc.edges < 0.4)).all()  # the measured deadbands, not zero or full valve


@pytest.mark.skipif(
    not (torch.cuda.is_available() and DEFAULT_MODEL.exists()), reason="needs CUDA and the plant"
)
def test_mpc_cuda_graph_replays_exactly_what_eager_computes():
    from hydraulic_controller.mpc import MPPIConfig, MPPIController

    prep = tip_task(duration_s=1.5)
    _, eager = rollout(prep, MPPIController(DEFAULT_MODEL, MPPIConfig(cuda_graph=False)), record=True)
    _, graph = rollout(prep, MPPIController(DEFAULT_MODEL, MPPIConfig(cuda_graph=True)), record=True)
    torch.testing.assert_close(graph["u"], eager["u"], rtol=0, atol=1e-6)
