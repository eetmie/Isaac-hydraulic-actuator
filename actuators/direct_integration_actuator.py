"""
Direct-Integration Actuator — the learned model *is* the dynamics.

    q(t+1) = q(t) + dt * qdot_pred(t+1)

and that state is written straight into the sim. PhysX does not get a say. These joints
therefore stop reacting to contact: they are kinematically driven, so they push through
collisions and do not slow under load. Fine for free-space motion, wrong for digging.
Use VelocityIntegratedActuator instead when you want physics in the loop.

**You must zero the drive gains for these joints.** PhysX computes
``tau = stiffness * (targetPos - q) + damping * (targetVel - qdot)`` every substep, and
``targetVel`` is a zero buffer IsaacLab pushes unconditionally -- nothing in the public
API parks it. Parking ``targetPos`` cancels only the stiffness term, so the damping term
survives as a brake aimed at exactly the velocity you just prescribed.

Measured here (0.6 rad/s commanded on all three arm joints, K=600 D=40, dt=0.01,
position target parked every step)::

    commanded qdot         [ 0.600  0.600  0.600]
    robot.data.joint_vel   [-0.056 -0.025 -0.010]   <- annihilated, sign flipped

The joint still *looks* like it moves, because position is overwritten every step, but
the velocity readback is ~0 -- and that is what the model would be fed back. With the
gains at 0/0 the same test gives ``[0.616 0.601 0.490]``. Call
``write_joint_stiffness_to_sim(0.0, ...)`` and ``write_joint_damping_to_sim(0.0, ...)``
before stepping; see ``configure_arm_drive`` in sim_common.py.

**Read state back from this class, not from** ``robot.data`` -- use :attr:`position` /
:attr:`velocity` as the model's input, which closes the loop the same way rollout.py
does offline. PhysX still advances the prescribed velocity during ``sim.step()``, so
call :meth:`sync_to_sim` afterwards to stop it being integrated twice.
"""

from __future__ import annotations

from typing import List

import torch

from isaaclab.scene import InteractiveScene


class DirectIntegrationActuator:
    """Integrates predicted joint velocity into joint state and writes it to the sim."""

    def __init__(
        self,
        scene: InteractiveScene,
        joint_names: List[str],
        sim_dt: float,
        clamp_to_limits: bool = True,
    ) -> None:
        self.scene = scene
        self.robot = scene["robot"]
        self.joint_names = list(joint_names)
        self.sim_dt = float(sim_dt)
        self.clamp_to_limits = bool(clamp_to_limits)

        # preserve_order=True: find_joints() defaults to articulation order, which is not
        # necessarily the order of joint_names. The learned model is order-sensitive
        # ([lift, tilt, tool]), so a silent permutation here would scramble it.
        self.joint_ids, resolved_names = self.robot.find_joints(self.joint_names, preserve_order=True)
        if resolved_names != self.joint_names:
            raise RuntimeError(
                f"Joint resolution changed order: asked for {self.joint_names}, got {resolved_names}"
            )
        self.num_joints = len(self.joint_ids)

        self._position = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self._velocity = torch.zeros_like(self._position)

        if self.clamp_to_limits:
            self._joint_pos_limits = self.robot.data.soft_joint_pos_limits[:, self.joint_ids, :].clone()
        else:
            self._joint_pos_limits = None

    def reset(self) -> None:
        """Re-seed the integrated state from the robot's current pose."""
        self._position = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self._velocity = self.robot.data.joint_vel[:, self.joint_ids].clone()

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """
        Integrate the commanded velocity and write the result into the simulation.

        Args:
            velocity_commands: (num_envs, num_joints) joint velocities in rad/s.
        """
        if velocity_commands.ndim != 2:
            raise ValueError(
                f"velocity_commands must be 2D (num_envs, num_joints), got {tuple(velocity_commands.shape)}"
            )
        if velocity_commands.shape[1] != self.num_joints:
            raise ValueError(
                f"velocity_commands second dim ({velocity_commands.shape[1]}) does not match "
                f"controlled num_joints ({self.num_joints})"
            )

        vel = velocity_commands.to(self._position.device, dtype=self._position.dtype)
        self._position = self._position + vel * self.sim_dt

        if self.clamp_to_limits and self._joint_pos_limits is not None:
            lo = self._joint_pos_limits[:, :, 0]
            hi = self._joint_pos_limits[:, :, 1]
            clamped = torch.clamp(self._position, min=lo, max=hi)
            # Zero the velocity on axes we just clamped, so the state written to the sim
            # stays self-consistent (a joint pinned at its limit is not moving).
            vel = torch.where(clamped != self._position, torch.zeros_like(vel), vel)
            self._position = clamped

        self._velocity = vel
        self.sync_to_sim()

    def sync_to_sim(self) -> None:
        """Restore the authoritative learned state after a PhysX step."""
        self.robot.write_joint_state_to_sim(
            position=self._position,
            velocity=self._velocity,
            joint_ids=self.joint_ids,
        )
        # No set_joint_position_target() here on purpose. Parking the position target
        # cancels only the stiffness term; the damping term still brakes against the
        # velocity we just wrote, because joint_vel_target is a zero buffer that
        # IsaacLab pushes to PhysX every step. The drive has to be switched off at the
        # gains instead -- see the module docstring.

    @property
    def position(self) -> torch.Tensor:
        return self._position

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity


__all__ = ["DirectIntegrationActuator"]
