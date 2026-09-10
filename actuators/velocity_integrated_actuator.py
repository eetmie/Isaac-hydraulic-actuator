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
- Optional velocity feedforward synchronizes the PD's desired velocity with its position
  increment. ``reference_velocity`` also supports a learned reference generator independent
  of physical tracking error, without bypassing the actual articulation dynamics.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .integrating_actuator_base import IntegratingActuatorBase, IntegratingActuatorCfg


@configclass
class VelocityIntegratedActuatorCfg(IntegratingActuatorCfg):
    """Configuration for :class:`VelocityIntegratedActuator`."""

    velocity_feedforward: bool = False
    """Also send the accepted setpoint velocity [rad/s] to the PD drive.

    Opt-in preserves legacy position-only behavior. Velocity is derived after
    limit clamping so the drive does not push outward at an end stop.
    """

    continuous: bool = False
    """Use circular PD position error for unlimited revolute joints.

    The learned reference remains unwrapped [rad]. Only the position handed to
    the drive is represented near the measured angle, whose coordinate may wrap.
    Set this only for groups containing exclusively unlimited revolute joints.
    """


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
        self._target_velocity = torch.zeros_like(self._target_position)
        self._reference_velocity = self._current_joint_vel()

    def reset(self) -> None:
        """Reset target positions to current joint positions."""
        self._target_position = self._current_joint_pos()
        self._target_velocity = torch.zeros_like(self._target_position)
        self._reference_velocity = self._current_joint_vel()
        if self.cfg.velocity_feedforward:
            self.robot.set_joint_velocity_target_index(target=self._target_velocity, joint_ids=self.joint_ids)

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """Integrate velocity commands and send joint position targets to the robot.

        Args:
            velocity_commands: Shape (num_envs, num_joints), desired joint
                velocities [rad/s].
        """
        self._validate(velocity_commands)

        # Integrate v*dt into target positions, then apply the anti-windup clamp
        # so the setpoint stays inside the reachable range (see module docstring).
        previous_position = self._target_position
        proposed_position = previous_position + velocity_commands * self.sim_dt
        self._target_position = self._clamp(proposed_position)
        self._target_velocity = (self._target_position - previous_position) / self.sim_dt
        self._reference_velocity = torch.where(
            proposed_position != self._target_position, torch.zeros_like(velocity_commands), velocity_commands
        )

        # Send position targets to robot PD controller
        drive_position = self._target_position
        if self.cfg.continuous:
            measured = self._current_joint_pos()
            error = self._target_position - measured
            drive_position = measured + torch.atan2(torch.sin(error), torch.cos(error))
        self.robot.set_joint_position_target_index(target=drive_position, joint_ids=self.joint_ids)
        if self.cfg.velocity_feedforward:
            self.robot.set_joint_velocity_target_index(target=self._target_velocity, joint_ids=self.joint_ids)

    @property
    def target_position(self) -> torch.Tensor:
        """Integrated reference position [rad], unwrapped for continuous joints."""
        return self._target_position

    @property
    def target_velocity(self) -> torch.Tensor:
        """Accepted velocity after position-limit clamping [rad/s]."""
        return self._target_velocity

    @property
    def reference_velocity(self) -> torch.Tensor:
        """Model reference velocity [rad/s], stopped when a position limit clips.

        This follows DirectIntegrationActuator's reference-state convention.
        It differs from the accepted one-step PD increment when a limit is hit.
        """
        return self._reference_velocity


__all__ = ["VelocityIntegratedActuator", "VelocityIntegratedActuatorCfg"]
