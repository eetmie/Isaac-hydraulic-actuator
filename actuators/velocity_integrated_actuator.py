"""
Velocity-Integrated Actuator (no linkage scaling).

This actuator integrates commanded joint velocities into joint position targets
each simulation step and writes them to the robot. It intentionally omits any
linkage-rate based scaling or cylinder geometry logic so it can serve as a clean
baseline for testing and comparison.

Intended usage (example):

    actuator = VelocityIntegratedActuator(
        VelocityIntegratedActuatorCfg(joint_names_expr=["revolute_lift", ...]),
        scene=scene,
        sim_dt=sim.get_physics_dt(),
    )

    # velocity_commands: shape (num_envs, num_joints)
    actuator.apply_velocity_command(velocity_commands)

Notes:
- This class treats the incoming commands strictly as joint velocities.
- No dependencies on IMUs or frame transformers.
- Clamping to soft joint limits is on by default and should stay on here. Unlike
  DirectIntegrationActuator, this class integrates into a *setpoint* the PD chases, so an
  unclamped target keeps accumulating past the physical limit. The joint would then refuse
  to move back until the whole excursion had been unwound -- a stick that does nothing for
  several seconds. The clamp is the anti-windup, not a speed limit.
- This class leaves the articulation drive live and relies on it. That is the whole point
  of the "target" route: PhysX's PD is the thing doing the work.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .integrating_actuator_base import IntegratingActuatorBase, IntegratingActuatorCfg


@configclass
class VelocityIntegratedActuatorCfg(IntegratingActuatorCfg):
    """Configuration for :class:`VelocityIntegratedActuator`."""

    pass


class VelocityIntegratedActuator(IntegratingActuatorBase):
    """Integrates joint velocities to position targets and writes to the robot."""

    cfg: VelocityIntegratedActuatorCfg

    def __init__(self, cfg: VelocityIntegratedActuatorCfg, **kwargs) -> None:
        """Initialize the actuator.

        Args:
            cfg: Joint group and clamping behaviour.
            **kwargs: Forwarded to :class:`IntegratingActuatorBase` -- ``scene``,
                ``sim_dt`` [s], and optionally ``asset_name``.
        """
        super().__init__(cfg, **kwargs)
        self._target_position = self._current_joint_pos()

    def reset(self) -> None:
        """Reset target positions to current joint positions."""
        self._target_position = self._current_joint_pos()

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """Integrate velocity commands and send joint position targets to the robot.

        Args:
            velocity_commands: Shape (num_envs, num_joints), desired joint
                velocities in joint-space units per second.
        """
        self._validate(velocity_commands)

        # Integrate v*dt into target positions, then apply the anti-windup clamp
        # so the setpoint stays inside the reachable range (see module docstring).
        self._target_position = self._clamp(self._target_position + velocity_commands * self.sim_dt)

        # Send position targets to robot PD controller
        self.robot.set_joint_position_target_index(target=self._target_position, joint_ids=self.joint_ids)

    @property
    def target_position(self) -> torch.Tensor:
        """The integrated position setpoint currently handed to the PD drive [rad]."""
        return self._target_position


__all__ = ["VelocityIntegratedActuator", "VelocityIntegratedActuatorCfg"]
