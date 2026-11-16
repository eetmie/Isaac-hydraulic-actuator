"""An externally requested EE pose/twist stays unchanged by the policy's internal tracking point."""

import math

import pytest
import torch

from hydraulic_controller.closed_loop import TaskBatch
from hydraulic_controller.core import DEFAULT_MODEL, HOME, ControllerSettings
from hydraulic_controller.kinematics import ExcavatorKinematics
from hydraulic_controller.policy import ControllerPolicy, PolicyController


def test_tip_to_pivot_twist_matches_both_jacobians():
    kin = ExcavatorKinematics()
    generator = torch.Generator().manual_seed(391)
    q = torch.tensor(HOME).repeat(32, 1) + torch.randn(32, 4, generator=generator) * 0.2
    velocity = torch.randn(32, 4, generator=generator) * 0.2
    tip = torch.einsum("nij,nj->ni", kin.pose_jacobian(q)[1], velocity)
    pivot = torch.einsum("nij,nj->ni", kin.pose_jacobian(q, "pivot")[1], velocity)
    torch.testing.assert_close(kin.tip_twist_to_pivot(q, tip), pivot, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(kin.pivot_twist_to_tip(q, pivot), tip, atol=2e-7, rtol=2e-6)


def test_reference_angular_feedforward_wraps_at_pi():
    refs = torch.tensor([[[0.5, 0.1, math.pi - 0.01]], [[0.5, 0.1, -math.pi + 0.01]]])
    task = TaskBatch(None, None, None, refs, None)
    velocity = task.tip_velocity()
    torch.testing.assert_close(velocity[0], torch.zeros(1, 3))
    torch.testing.assert_close(velocity[1, :, :2], torch.zeros(1, 2))
    torch.testing.assert_close(velocity[1, :, 2], torch.tensor([2.0]), atol=4e-5, rtol=0)


@pytest.mark.parametrize("point", ["tip", "pivot"])
def test_policy_receives_ee_request_and_angular_feedforward_without_changing_operator_target(point):
    kin = ExcavatorKinematics()
    q, v = torch.tensor([HOME]), torch.zeros(1, 4)
    target = kin.pose_jacobian(q)[0] + torch.tensor([[0.01, 0.0, 0.05]])
    captured = {}

    def governor(q, v, command):
        captured["command"] = command.clone()
        return command, torch.zeros(len(q))

    class Policy:
        settings = ControllerSettings()
        model_path = DEFAULT_MODEL
        tracked_point = point
        to_tracking_twist = ControllerPolicy.to_tracking_twist

        def governor(self, kin):
            return governor

        def future(self, *args):
            return None

        def __call__(self, *args):
            return torch.zeros(1, 3)

    task = TaskBatch(
        kin, q, torch.zeros(6, 1, 3), target[None].repeat(6, 1, 1), torch.ones(1, dtype=torch.bool)
    )
    task.tip_ref[5, :, 2] += 0.001
    original = task.tip_ref.clone()
    controller = PolicyController(Policy())
    controller.reset(task)
    controller(0, q, v)
    expected = torch.tensor([[0.03, 0.0, 0.15]])
    desired = kin.tip_twist_to_pivot(q, expected) if point == "pivot" else expected
    torch.testing.assert_close(captured["command"], desired, atol=2e-7, rtol=2e-6)
    controller(5, q, v)
    expected[:, 2] = 0.253  # 0.1 rad/s feedforward + 3 * 0.051 rad pose error.
    desired = kin.tip_twist_to_pivot(q, expected) if point == "pivot" else expected
    torch.testing.assert_close(captured["command"], desired, atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(task.tip_ref, original, rtol=0, atol=0)
