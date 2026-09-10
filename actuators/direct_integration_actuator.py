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
``write_joint_stiffness_to_sim_index(stiffness=0.0, ...)`` and
``write_joint_damping_to_sim_index(damping=0.0, ...)`` before stepping; see
``configure_joint_drive`` in sim_common.py.

**Read state back from this class, not from** ``robot.data`` -- use :attr:`position` /
:attr:`velocity` as the model's input, which closes the loop the same way rollout.py
does offline. PhysX still advances the prescribed velocity during ``sim.step()``, so
call :meth:`sync_to_sim` afterwards to stop it being integrated twice.
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .integrating_actuator_base import IntegratingActuatorBase, IntegratingActuatorCfg


@configclass
class DirectIntegrationActuatorCfg(IntegratingActuatorCfg):
    """Configuration for :class:`DirectIntegrationActuator`."""

    pass


class DirectIntegrationActuator(IntegratingActuatorBase):
    """Integrates predicted joint velocity into joint state and writes it to the sim."""

    cfg: DirectIntegrationActuatorCfg

    def __init__(self, cfg: DirectIntegrationActuatorCfg, **kwargs) -> None:
        """Initialize the actuator.

        Args:
            cfg: Joint group and clamping behaviour.
            **kwargs: Forwarded to :class:`IntegratingActuatorBase` -- ``scene``,
                ``sim_dt`` [s], and optionally ``asset_name``.
        """
        super().__init__(cfg, **kwargs)
        self._position = self._current_joint_pos()
        self._velocity = torch.zeros_like(self._position)

    def reset(self) -> None:
        """Re-seed the integrated state from the robot's current pose."""
        self._position = self._current_joint_pos()
        self._velocity = self._current_joint_vel()

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """
        Integrate the commanded velocity and write the result into the simulation.

        Args:
            velocity_commands: Shape (num_envs, num_joints), joint velocities in rad/s.
        """
        self._validate(velocity_commands)

        vel = velocity_commands.to(self._position.device, dtype=self._position.dtype)
        integrated = self._position + vel * self.sim_dt
        clamped = self._clamp(integrated)
        # Zero the velocity on axes we just clamped, so the state written to the sim
        # stays self-consistent (a joint pinned at its limit is not moving).
        vel = torch.where(clamped != integrated, torch.zeros_like(vel), vel)

        self._position = clamped
        self._velocity = vel
        self.sync_to_sim()

    def sync_to_sim(self) -> None:
        """Restore the authoritative learned state after a PhysX step."""
        self.robot.write_joint_position_to_sim_index(position=self._position, joint_ids=self.joint_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self._velocity, joint_ids=self.joint_ids)
        # No set_joint_position_target_index() here on purpose. Parking the position target
        # cancels only the stiffness term; the damping term still brakes against the
        # velocity we just wrote, because joint_vel_target is a zero buffer that
        # IsaacLab pushes to PhysX every step. The drive has to be switched off at the
        # gains instead -- see the module docstring.

    @property
    def position(self) -> torch.Tensor:
        """The integrated joint position the model is driving [rad]."""
        return self._position

    @property
    def velocity(self) -> torch.Tensor:
        """The joint velocity last written to the sim [rad/s]."""
        return self._velocity


__all__ = ["DirectIntegrationActuator", "DirectIntegrationActuatorCfg"]
