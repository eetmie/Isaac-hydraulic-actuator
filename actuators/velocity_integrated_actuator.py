"""
Velocity-Integrated Actuator (base, no linkage scaling).

This actuator integrates commanded joint velocities into joint position targets
each simulation step and writes them to the robot. It intentionally omits any
linkage-rate based scaling or cylinder geometry logic so it can serve as a clean
baseline for testing and comparison.

Intended usage (example):

    actuator = VelocityIntegratedActuator(
        scene=scene,
        joint_names=["revolute_lift", "revolute_tilt", ...],
        sim_dt=sim.get_physics_dt(),
        clamp_to_limits=True,
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

from typing import List

import torch

from isaaclab.scene import InteractiveScene


class VelocityIntegratedActuator:
    """Integrates joint velocities to position targets and writes to the robot."""

    def __init__(
        self,
        scene: InteractiveScene,
        joint_names: List[str],
        sim_dt: float,
        clamp_to_limits: bool = True,
    ) -> None:
        """Initialize the actuator.

        Args:
            scene: InteractiveScene containing the robot.
            joint_names: Ordered list of joint names to control.
            sim_dt: Simulation time step used for integration.
            clamp_to_limits: If True, clamp integrated targets to soft joint limits.
        """
        self.scene = scene
        self.robot = scene["robot"]
        self.joint_names = list(joint_names)
        self.sim_dt = float(sim_dt)
        self.clamp_to_limits = bool(clamp_to_limits)

        # Resolve joint indices in the same order as provided names.
        # preserve_order=True: find_joints() defaults to articulation order, which is not
        # necessarily the order of joint_names. The learned model is order-sensitive
        # ([lift, tilt, tool]), so a silent permutation here would scramble it.
        self.joint_ids, resolved_names = self.robot.find_joints(self.joint_names, preserve_order=True)
        if resolved_names != self.joint_names:
            raise RuntimeError(
                f"Joint resolution changed order: asked for {self.joint_names}, got {resolved_names}"
            )
        self.num_joints = len(self.joint_ids)

        # Initialize target positions from current robot joint positions
        self._target_position = self.robot.data.joint_pos[:, self.joint_ids].clone()

        # Cache limits if clamping is enabled
        if self.clamp_to_limits:
            self._joint_pos_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, :].clone()
        else:
            self._joint_pos_limits = None

    def reset(self) -> None:
        """Reset target positions to current joint positions."""
        self._target_position = self.robot.data.joint_pos[:, self.joint_ids].clone()

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """Integrate velocity commands and send joint position targets to the robot.

        Args:
            velocity_commands: Tensor of shape (num_envs, num_joints) representing
                desired joint velocities in joint-space units per second.
        """
        if velocity_commands.ndim != 2:
            raise ValueError(
                f"velocity_commands must be 2D (num_envs, num_joints), got shape {tuple(velocity_commands.shape)}"
            )

        if velocity_commands.shape[1] != self.num_joints:
            raise ValueError(
                f"velocity_commands second dim ({velocity_commands.shape[1]}) does not match controlled num_joints ({self.num_joints})"
            )

        # Integrate v*dt into target positions
        self._target_position = self._target_position + velocity_commands * self.sim_dt

        # Anti-windup: keep the setpoint inside the reachable range (see module docstring)
        if self.clamp_to_limits and self._joint_pos_limits is not None:
            self._target_position = torch.clamp(
                self._target_position,
                min=self._joint_pos_limits[:, :, 0],
                max=self._joint_pos_limits[:, :, 1],
            )

        # Send position targets to robot PD controller
        self.robot.set_joint_position_target(self._target_position, joint_ids=self.joint_ids)

    @property
    def target_position(self) -> torch.Tensor:
        """The integrated position setpoint currently handed to the PD drive."""
        return self._target_position


__all__ = ["VelocityIntegratedActuator"]

