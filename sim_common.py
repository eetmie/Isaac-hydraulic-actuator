"""The excavator demo's robot-specific layer.

Everything that knows this particular machine lives here or in sim.py: joint
names, drive gains, tool assets, and where its trained models are kept. The
``actuators/`` package below it knows none of it.

Targets Isaac Lab 3.0 / Isaac Sim 6.0.1.
"""

from __future__ import annotations

import os

import numpy as np

SIM_HZ = 100.0

# Joint groups, in the channel order their models were trained in.
ARM_JOINT_NAMES = ["revolute_lift", "revolute_tilt", "revolute_tool"]
SLEW_JOINT_NAMES = ["revolute_carriage"]
GRIPPER_JOINT_NAMES = ["revolute_gripper", "revolute_claw_1", "revolute_claw_2"]

# Trained model directories, one per joint group. The arm model ships with the
# repository; the slew one does not exist yet, and sim.py falls back when the
# directory is absent -- see its ``load_controller``.
DEFAULT_ARM_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "arm"
)
DEFAULT_SLEW_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "slew"
)

# End-effector variants shipped in assets/. The bucket asset welds a bucket to
# ``tool_body`` with a fixed joint and has no extra DOFs; the gripper asset adds
# the three GRIPPER_JOINT_NAMES joints. Both share the arm and carriage joints.
TOOL_VARIANTS = ("bucket", "gripper")
DEFAULT_TOOL = "bucket"
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

ARM_STIFFNESS = 600.0
ARM_DAMPING = 40.0
CARRIAGE_STIFFNESS = 600.0
CARRIAGE_DAMPING = 40.0
GRIPPER_STIFFNESS = 600.0
GRIPPER_DAMPING = 40.0
SOLVER_VELOCITY_ITERATIONS = 1
DISABLE_ROBOT_GRAVITY = True

NN_ARM_VEL_LIMIT_DEFAULT = 10.0
NN_ARM_VEL_WARN_DEFAULT = 2.0


def robot_usd_path(tool: str) -> str:
    """Absolute path to the excavator USD carrying the requested end effector.

    Args:
        tool: One of :data:`TOOL_VARIANTS`.

    Returns:
        Absolute path to the USD file.

    Raises:
        ValueError: If ``tool`` is not a known variant.
        FileNotFoundError: If the asset is missing from ``assets/``.
    """
    if tool not in TOOL_VARIANTS:
        raise ValueError(f"Unknown tool variant {tool!r}; expected one of {list(TOOL_VARIANTS)}")
    path = os.path.join(_ASSET_DIR, f"excavator_{tool}.usd")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing robot asset for tool {tool!r}: {path}")
    return path


def has_gripper_joints(tool: str) -> bool:
    """Whether the asset for ``tool`` exposes the articulated gripper DOFs."""
    return tool == "gripper"


def configure_joint_drive(
    robot,
    joint_ids,
    *,
    direct: bool,
    stiffness: float,
    damping: float,
) -> None:
    """Release the PhysX drive for direct state integration, otherwise enable PD.

    Zeroing the gains is not optional under direct integration: PhysX's damping
    term brakes against exactly the velocity the actuator just prescribed. See
    the module docstring of ``actuators/direct_integration_actuator.py``.

    Args:
        robot: The articulation.
        joint_ids: Indices of the joints to reconfigure.
        direct: If True, zero the gains so the learned state integration is not
            fought by the articulation drive.
        stiffness: PD stiffness to restore when not direct [N·m/rad].
        damping: PD damping to restore when not direct [N·m·s/rad].
    """
    robot.write_joint_stiffness_to_sim_index(
        stiffness=0.0 if direct else stiffness, joint_ids=joint_ids
    )
    robot.write_joint_damping_to_sim_index(
        damping=0.0 if direct else damping, joint_ids=joint_ids
    )


def sanitize_velocity_prediction(prediction, limit: float):
    """Zero non-finite output and apply the shared last-resort velocity clamp.

    Args:
        prediction: Predicted joint velocities [rad/s].
        limit: Hard clamp magnitude [rad/s].

    Returns:
        Tuple of (sanitized velocity [rad/s], max absolute magnitude [rad/s],
        whether the input was finite, whether the clamp engaged).
    """
    velocity = np.asarray(prediction, dtype=np.float32)
    if not np.all(np.isfinite(velocity)):
        return np.zeros_like(velocity), 0.0, False, False
    max_abs = float(np.max(np.abs(velocity)))
    clipped = max_abs > limit
    return np.clip(velocity, -limit, limit), max_abs, True, clipped


__all__ = [
    "ARM_DAMPING",
    "ARM_JOINT_NAMES",
    "ARM_STIFFNESS",
    "CARRIAGE_DAMPING",
    "CARRIAGE_STIFFNESS",
    "DEFAULT_ARM_MODEL_DIR",
    "DEFAULT_SLEW_MODEL_DIR",
    "DEFAULT_TOOL",
    "DISABLE_ROBOT_GRAVITY",
    "GRIPPER_DAMPING",
    "GRIPPER_JOINT_NAMES",
    "GRIPPER_STIFFNESS",
    "NN_ARM_VEL_LIMIT_DEFAULT",
    "NN_ARM_VEL_WARN_DEFAULT",
    "SIM_HZ",
    "SLEW_JOINT_NAMES",
    "SOLVER_VELOCITY_ITERATIONS",
    "TOOL_VARIANTS",
    "configure_joint_drive",
    "has_gripper_joints",
    "robot_usd_path",
    "sanitize_velocity_prediction",
]
