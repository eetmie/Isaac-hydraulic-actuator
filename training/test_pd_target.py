# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Target-actuator command regressions without launching Isaac Kit."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class TestTargetCommands(unittest.TestCase):
    def actuator(self, feedforward=True):
        source = Path(__file__).resolve().parents[1] / "actuators/velocity_integrated_actuator.py"
        node = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "VelocityIntegratedActuator"
        )

        class Base:
            def _validate(self, value):
                assert value.shape == (1, 3)

            def _clamp(self, value):
                return torch.clamp(value, -1, 1)

            def _current_joint_pos(self):
                return self.measured.clone()

            def _current_joint_vel(self):
                return torch.full_like(self.measured, 0.2)

        namespace = {"torch": torch, "IntegratingActuatorBase": Base, "VelocityIntegratedActuatorCfg": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        cls = namespace["VelocityIntegratedActuator"]
        actuator = object.__new__(cls)
        actuator.cfg = SimpleNamespace(velocity_feedforward=feedforward, continuous=False)
        actuator.sim_dt = 0.01
        actuator.joint_ids = [1, 2, 3]
        actuator.measured = torch.tensor([[0.0, 0.999, -1.0]])
        actuator._target_position = actuator.measured.clone()
        actuator._target_velocity = torch.zeros_like(actuator.measured)
        actuator._reference_velocity = torch.zeros_like(actuator.measured)
        actuator.sent = {}
        actuator.robot = SimpleNamespace(
            set_joint_position_target_index=lambda **kw: actuator.sent.update(position=kw["target"].clone()),
            set_joint_velocity_target_index=lambda **kw: actuator.sent.update(velocity=kw["target"].clone()),
        )
        return actuator

    def test_feedforward_matches_clamped_position_increment(self):
        actuator = self.actuator()
        actuator.apply_velocity_command(torch.tensor([[0.5, 0.5, -0.5]]))
        torch.testing.assert_close(
            actuator.sent["velocity"], torch.tensor([[0.5, 0.1, 0.0]]), atol=2e-6, rtol=0
        )
        torch.testing.assert_close(actuator.sent["position"], torch.tensor([[0.005, 1.0, -1.0]]))

    def test_legacy_default_does_not_write_velocity_target(self):
        actuator = self.actuator(False)
        actuator.apply_velocity_command(torch.ones(1, 3))
        self.assertNotIn("velocity", actuator.sent)

    def test_reset_clears_persistent_feedforward(self):
        actuator = self.actuator()
        actuator.apply_velocity_command(torch.ones(1, 3))
        actuator.reset()
        torch.testing.assert_close(actuator.sent["velocity"], torch.zeros(1, 3))
        torch.testing.assert_close(actuator.target_position, actuator.measured)

    def test_limit_reversal_has_no_windup(self):
        actuator = self.actuator()
        actuator.apply_velocity_command(torch.tensor([[0.0, 1.0, -1.0]]))
        actuator.apply_velocity_command(torch.tensor([[0.0, -1.0, 1.0]]))
        torch.testing.assert_close(actuator.sent["position"], torch.tensor([[0.0, 0.99, -0.99]]))
        torch.testing.assert_close(
            actuator.sent["velocity"], torch.tensor([[0.0, -1.0, 1.0]]), atol=2e-6, rtol=0
        )

    def test_model_reference_stops_on_clamp_without_losing_pd_increment(self):
        actuator = self.actuator()
        actuator.apply_velocity_command(torch.tensor([[0.5, 0.5, -0.5]]))
        torch.testing.assert_close(actuator.reference_velocity, torch.tensor([[0.5, 0.0, 0.0]]))
        torch.testing.assert_close(
            actuator.target_velocity, torch.tensor([[0.5, 0.1, 0.0]]), atol=2e-6, rtol=0
        )

    def test_reset_seeds_reference_from_actual_velocity(self):
        actuator = self.actuator()
        actuator.reset()
        torch.testing.assert_close(actuator.reference_velocity, torch.full((1, 3), 0.2))
        torch.testing.assert_close(actuator.sent["velocity"], torch.zeros(1, 3))

    def test_continuous_joint_wrap_preserves_reference_and_local_pd_error(self):
        actuator = self.actuator()
        actuator.cfg.continuous = True
        actuator._clamp = lambda value: value
        actuator.measured[:] = 0.02
        actuator._target_position[:] = 4 * torch.pi + 0.03
        actuator.apply_velocity_command(torch.full((1, 3), 0.5))
        torch.testing.assert_close(actuator.target_position, torch.full((1, 3), 4 * torch.pi + 0.035))
        torch.testing.assert_close(actuator.sent["position"], torch.full((1, 3), 0.035), atol=2e-6, rtol=0)
        torch.testing.assert_close(actuator.sent["velocity"], torch.full((1, 3), 0.5), atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()
