"""The excavator demo's robot-specific layer.

Everything that knows this particular machine lives here or in sim.py: joint
names, drive gains, tool assets, and where its trained models are kept. The
``actuators/`` package below it knows none of it.

Targets Isaac Lab 3.0 / Isaac Sim 6.0.1.
"""

from __future__ import annotations

import json
import math
import os
from argparse import Namespace

import numpy as np

SIM_HZ = 100.0

# Joint groups, in the channel order their models were trained in.
ARM_JOINT_NAMES = ["revolute_lift", "revolute_tilt", "revolute_tool"]
ROCKING_JOINT_NAMES = ["revolute_carriage_roll", "revolute_carriage_pitch"]
SLEW_JOINT_NAMES = ["revolute_carriage"]
GRIPPER_JOINT_NAMES = ["revolute_gripper", "revolute_claw_1", "revolute_claw_2"]

# Trained model directories, one per joint group. A missing optional slew
# directory enables the demo's stick-velocity fallback.
DEFAULT_ARM_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "arm_v4")
DEFAULT_SLEW_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "slew")

# End-effector variants shipped in assets/. The bucket asset welds a bucket to
# ``tool_body`` with a fixed joint and has no extra DOFs; the gripper asset adds
# the three GRIPPER_JOINT_NAMES joints. Both share the arm and carriage joints.
TOOL_VARIANTS = ("bucket", "gripper")
DEFAULT_TOOL = "bucket"
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

ARM_STIFFNESS = 600.0
ARM_DAMPING = 40.0
# Tuned target-mode arm drives. Legacy gains above remain available to callers.
ARM_TARGET_STIFFNESS = 2400.0
ARM_TARGET_DAMPING = 120.0
CARRIAGE_STIFFNESS = 600.0
CARRIAGE_DAMPING = 40.0
GRIPPER_STIFFNESS = 600.0
GRIPPER_DAMPING = 40.0
SOLVER_VELOCITY_ITERATIONS = 1
DISABLE_ROBOT_GRAVITY = True

NN_ARM_VEL_LIMIT_DEFAULT = 2.0
NN_ARM_VEL_WARN_DEFAULT = 2.0


def resolve_demo_defaults(args: Namespace) -> None:
    """Resolve explicit CLI overrides and the opt-in compliant-carriage preset.

    Stiffness is in N m/rad, damping in N m s/rad. Rocking needs physical arm
    drives: prescribing arm state does not provide the same reaction torques.
    """
    learned = getattr(args, "learned_carriage", False)
    motions = ARM_JOINT_NAMES + ROCKING_JOINT_NAMES if learned else ARM_JOINT_NAMES
    model_path = getattr(args, "model", None)
    if model_path and os.path.isfile(os.path.join(model_path, "model_meta.json")):
        with open(os.path.join(model_path, "model_meta.json"), encoding="utf-8") as stream:
            meta = json.load(stream)
        learned = bool(meta.get("learned_carriage", False))
        if learned:
            motions = meta.get("simulation_joint_names")
            if motions not in (
                ARM_JOINT_NAMES + ROCKING_JOINT_NAMES,
                ARM_JOINT_NAMES + ["revolute_carriage_pitch"],
            ):
                raise ValueError("Learned carriage artifact has an unsupported motion-channel mapping")
        else:
            motions = ARM_JOINT_NAMES
    args.learned_carriage = learned
    args.model_joint_names = list(motions)
    if learned:
        args.carriage_rocking = True
        if args.integration is None:
            args.integration = "direct"
        if args.integration != "direct":
            raise ValueError("The joint hydraulic/carriage model uses --integration direct")
        if args.physics_substeps is None:
            args.physics_substeps = 1
    rocking = args.carriage_rocking and not learned
    if args.integration is None:
        args.integration = "target" if rocking else "direct"
    if rocking and args.integration != "target":
        raise ValueError("--carriage_rocking requires --integration target for physical reaction torques")
    if args.physics_substeps is None:
        args.physics_substeps = 2 if rocking else 1
    if args.arm_stiffness is None:
        args.arm_stiffness = [10000.0 if rocking else ARM_TARGET_STIFFNESS] * 3
    if args.arm_damping is None:
        args.arm_damping = [300.0 if rocking else ARM_TARGET_DAMPING] * 3


class RealtimeReporter:
    """Report simulated seconds / wall seconds at a wall-clock cadence.

    Call after completed physics/render steps, using a monotonic wall clock.
    Initialize after startup to exclude loading. GUI pauses count as elapsed
    wall time; no report is emitted while the simulation loop is blocked.
    """

    def __init__(self, interval_s: float, *, wall_s: float, sim_s: float):
        if not math.isfinite(interval_s) or interval_s < 0:
            raise ValueError("Reporting interval must be finite and non-negative")
        self.interval_s = interval_s
        self.start_wall = self.last_wall = wall_s
        self.start_sim = self.last_sim = sim_s

    def update(self, *, wall_s: float, sim_s: float) -> str | None:
        """Return a periodic message, or None when disabled/not due."""
        elapsed = wall_s - self.last_wall
        if self.interval_s == 0 or elapsed < self.interval_s:
            return None
        rate = (sim_s - self.last_sim) / elapsed
        average = (sim_s - self.start_sim) / (wall_s - self.start_wall)
        if rate <= 0:
            comparison = "no simulated time advanced"
        elif rate < 1:
            comparison = f"{1 / rate:.2f}x slower"
        else:
            comparison = f"{rate:.2f}x faster" if rate > 1 else "real time"
        self.last_wall, self.last_sim = wall_s, sim_s
        return (
            f"[PERF] real-time={rate:.2f}x ({comparison}), average={average:.2f}x "
            f"| sim={sim_s - self.start_sim:.2f}s wall={wall_s - self.start_wall:.2f}s"
        )


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
    robot.write_joint_stiffness_to_sim_index(stiffness=0.0 if direct else stiffness, joint_ids=joint_ids)
    robot.write_joint_damping_to_sim_index(damping=0.0 if direct else damping, joint_ids=joint_ids)


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
    "ARM_TARGET_STIFFNESS",
    "ARM_TARGET_DAMPING",
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
    "RealtimeReporter",
    "SIM_HZ",
    "SLEW_JOINT_NAMES",
    "SOLVER_VELOCITY_ITERATIONS",
    "TOOL_VARIANTS",
    "configure_joint_drive",
    "has_gripper_joints",
    "robot_usd_path",
    "sanitize_velocity_prediction",
]
