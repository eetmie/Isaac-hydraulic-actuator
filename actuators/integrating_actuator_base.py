"""Shared base for the two velocity-integrating actuators.

Both actuators take a predicted joint velocity and integrate it forward; they
differ only in what they do with the result -- write it into joint state, or hand
it to the articulation's PD drive as a position target. Everything before that
point -- resolving joints, caching limits, validating command shape -- is here.

These are deliberately **not** :class:`isaaclab.actuators.ActuatorBase`
subclasses. That interface is ``compute(control_action, joint_pos, joint_vel) ->
ArticulationActions``, which has nowhere to put a valve command and cannot write
joint state, so neither of these two would fit inside it. What they do borrow is
the convention: a ``@configclass`` cfg carrying ``joint_names_expr``, and the
``joint_names`` / ``joint_ids`` / ``num_joints`` / ``reset()`` surface. They are
driven from the simulation loop rather than from
``Articulation.write_data_to_sim()``.
"""

from __future__ import annotations

import re
from dataclasses import MISSING

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.utils import configclass

_REGEX_CHARS = re.compile(r"[.^$*+?{}\[\]\\|()]")


@configclass
class IntegratingActuatorCfg:
    """Configuration shared by the velocity-integrating actuators."""

    joint_names_expr: list[str] = MISSING
    """Joint names, or regular expressions matching them, in model-channel order.

    Resolution preserves this order rather than falling back to articulation
    order: these actuators feed an order-sensitive learned model, and a silent
    permutation would scramble it.
    """

    clamp_to_limits: bool = True
    """Whether to clamp the integrated result to the soft joint position limits."""


class IntegratingActuatorBase:
    """Resolves a joint group and integrates velocity commands over it."""

    cfg: IntegratingActuatorCfg
    """The configuration this actuator was built from."""

    def __init__(
        self,
        cfg: IntegratingActuatorCfg,
        *,
        scene: InteractiveScene,
        sim_dt: float,
        asset_name: str = "robot",
    ) -> None:
        """Initialize the actuator.

        Args:
            cfg: Joint group and clamping behaviour.
            scene: Scene holding the articulation.
            sim_dt: Integration period [s].
            asset_name: Key of the articulation within the scene.
        """
        self.cfg = cfg
        self.scene = scene
        self.robot = scene[asset_name]
        self.sim_dt = float(sim_dt)
        self.clamp_to_limits = bool(cfg.clamp_to_limits)

        requested = list(cfg.joint_names_expr)
        self._joint_ids, self._joint_names = self.robot.find_joints(requested, preserve_order=True)
        # Literal names must come back verbatim; a regex is allowed to expand, but
        # then the resolved order is what the model will see, so surface it.
        if not any(_REGEX_CHARS.search(name) for name in requested) and self._joint_names != requested:
            raise RuntimeError(
                f"Joint resolution changed order: asked for {requested}, got {self._joint_names}"
            )

        # Isaac Lab 3.0 returns ProxyArray from every ``.data`` property; ``.torch``
        # is the zero-copy torch view, so clone before holding on to a slice.
        if self.clamp_to_limits:
            self._joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch[
                :, self._joint_ids, :
            ].clone()
        else:
            self._joint_pos_limits = None

    @property
    def joint_names(self) -> list[str]:
        """Controlled joint names, in the order the model expects them."""
        return self._joint_names

    @property
    def joint_ids(self) -> list[int]:
        """Controlled joint indices within the articulation."""
        return self._joint_ids

    @property
    def num_joints(self) -> int:
        """Number of controlled joints."""
        return len(self._joint_ids)

    def _current_joint_pos(self) -> torch.Tensor:
        """Current positions of the controlled joints [rad], shape (num_envs, num_joints)."""
        return self.robot.data.joint_pos.torch[:, self._joint_ids].clone()

    def _current_joint_vel(self) -> torch.Tensor:
        """Current velocities of the controlled joints [rad/s], shape (num_envs, num_joints)."""
        return self.robot.data.joint_vel.torch[:, self._joint_ids].clone()

    def _validate(self, velocity_commands: torch.Tensor) -> None:
        """Reject a command whose shape does not match the controlled group."""
        if velocity_commands.ndim != 2:
            raise ValueError(
                "velocity_commands must be 2D (num_envs, num_joints), got shape "
                f"{tuple(velocity_commands.shape)}"
            )
        if velocity_commands.shape[1] != self.num_joints:
            raise ValueError(
                f"velocity_commands second dim ({velocity_commands.shape[1]}) does not match "
                f"controlled num_joints ({self.num_joints}) for {self._joint_names}"
            )

    def _clamp(self, position: torch.Tensor) -> torch.Tensor:
        """Clamp integrated positions to the soft limits, if clamping is enabled."""
        if not self.clamp_to_limits or self._joint_pos_limits is None:
            return position
        return torch.clamp(
            position,
            min=self._joint_pos_limits[:, :, 0],
            max=self._joint_pos_limits[:, :, 1],
        )

    def reset(self) -> None:
        """Re-seed the integrator from the robot's current state."""
        raise NotImplementedError

    def apply_velocity_command(self, velocity_commands: torch.Tensor) -> None:
        """Integrate one step of commanded joint velocity [rad/s].

        Args:
            velocity_commands: Shape (num_envs, num_joints), in rad/s.
        """
        raise NotImplementedError


__all__ = ["IntegratingActuatorBase", "IntegratingActuatorCfg"]
