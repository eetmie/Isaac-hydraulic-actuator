"""The default pivot policy keeps external requests and scores at the bucket edge."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from hydraulic_controller.benchmark import _Rollout
from hydraulic_controller.bundle import export_bundle
from hydraulic_controller.core import HOME, HydraulicPlant, sha256
from hydraulic_controller.kinematics import ExcavatorKinematics
from hydraulic_controller.policy import ControllerPolicy
from run_controller import resolve_checkpoint
from train_controller import tracking_point


@pytest.fixture(scope="module")
def policy():
    return ControllerPolicy(resolve_checkpoint(None))


def test_packaged_demo_is_pivot_with_portable_verified_dependencies(policy):
    assert policy.tracked_point == "pivot" and not policy.has_pose_priority
    assert not Path(policy.contract["model_path"]).is_absolute()
    assert not Path(policy.contract["asset_path"]).is_absolute()
    provenance = json.loads((policy.checkpoint.parent / "provenance.json").read_text())
    assert provenance["checkpoint_sha256"] == sha256(policy.checkpoint)
    assert provenance["hardware_validated"] is False
    actions = policy(torch.zeros(1, policy.contract["observation_dim"] - 3 * policy.lookahead_steps))
    assert actions.shape == (1, 3) and torch.isfinite(actions).all()


def test_bucket_rotation_changes_tip_translation_but_not_pivot_translation(policy):
    kin = ExcavatorKinematics(policy.asset_path)
    plant = HydraulicPlant(policy.model_path, kin, 1, "cpu", tracked_point="pivot")
    plant.v[:, 2] = 0.2
    assert plant.tip_state()[1][0, :2].norm() > 0.01
    torch.testing.assert_close(plant.tracking_state()[1][:, :2], torch.zeros(1, 2), atol=1e-7, rtol=0)


def test_standalone_benchmark_converts_requests_and_lookahead_but_scores_tip(policy):
    kin = ExcavatorKinematics(policy.asset_path)
    q = torch.tensor([HOME])
    rollout = _Rollout(policy, kin, q, "nominal")
    request = torch.tensor([[0.01, -0.02, 0.1]])
    captured = {}

    def governor(q, v, command):
        captured["command"] = command.clone()
        return command, torch.zeros(1)

    rollout.governor = governor

    def feedforward(t):
        return request * (1 + t)

    original_future = policy.future

    def future(ff, k, stride):
        ahead = original_future(ff, k, stride)
        expected = torch.stack(
            [kin.tip_twist_to_pivot(q, request * (j * stride * 0.01)) for j in range(1, 5)], dim=1
        )
        torch.testing.assert_close(ahead, expected, atol=1e-7, rtol=1e-5)
        return ahead

    with patch.object(policy, "future", side_effect=future):
        logs = rollout.run(0.01, lambda t, pose, twist: request, feedforward)
    torch.testing.assert_close(captured["command"], kin.tip_twist_to_pivot(q, request))
    torch.testing.assert_close(logs["pose"][0], kin.pose_jacobian(rollout.plant.q)[0])
    torch.testing.assert_close(
        logs["command"][0], policy.to_tip_twist(kin, rollout.plant.q, captured["command"])
    )


@pytest.mark.parametrize("saved", [None, "tip", "pivot"])
def test_resume_retains_tracking_semantics_even_when_old_contract_omits_point(tmp_path, saved):
    contract = {} if saved is None else {"tracked_point": saved}
    (tmp_path / "controller_contract.json").write_text(json.dumps(contract))
    checkpoint = tmp_path / "model_100.pt"
    expected = saved or "tip"
    assert tracking_point(checkpoint) == expected
    with pytest.raises(ValueError, match="Cannot resume"):
        tracking_point(checkpoint, "tip" if expected == "pivot" else "pivot")
    assert tracking_point() == "pivot"


def test_legacy_export_refuses_pivot_before_writing_a_misinterpreted_bundle(policy, tmp_path):
    with pytest.raises(ValueError, match="pivot adapter"):
        export_bundle(policy, tmp_path / "bundle", tmp_path)
    assert not (tmp_path / "bundle").exists()


def test_historical_checkpoint_without_tracking_field_remains_tip_tracking(tmp_path):
    from training.overnight import reference_controller

    # The archived controller pins an older USD. Recover that exact asset instead of bypassing its hash check.
    legacy = ControllerPolicy(reference_controller(tmp_path))
    assert "tracked_point" not in legacy.contract and legacy.tracked_point == "tip"
    kin = ExcavatorKinematics(legacy.asset_path)
    q, request = torch.tensor([HOME]), torch.tensor([[0.01, 0.0, 0.1]])
    torch.testing.assert_close(legacy.to_tracking_twist(kin, q, request), request)
    action = legacy(torch.zeros(1, legacy.contract["observation_dim"]))
    assert action.shape == (1, 3) and torch.isfinite(action).all()
