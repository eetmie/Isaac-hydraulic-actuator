"""
Excavator sim -- hydraulic actuator net + gamepad teleop.

Targets Isaac Lab 3.0 / Isaac Sim 6.0.1.

USD joint names map to the model's joint order as lift=boom, tilt=arm, tool=bucket.

Gamepad bindings (Xbox layout):
  Left  stick  Y  : tilt   command
  Left  stick  X  : slew   command
  Right stick  Y  : lift   command
  Right stick  X  : tool   command
  A              : toggle MANUAL / NN command source
  B              : reset joints to default pose
  dead zone      : 30 %

MANUAL mode -- sticks scale directly to joint velocity
NN mode     -- sticks are valve commands [-1,1] -> HydraulicActuatorNet -> predicted qdot

Two learned groups: the hydraulic arm (optionally including carriage pitch or
roll/pitch) and the independent slew in models/slew. Metadata determines the
arm motion-channel mapping. If a requested slew artifact is absent, this demo
falls back to stick velocity; the generic actuator has no degraded mode.

Both command sources use the integration route selected by --integration. The
slew joint follows the arm's route once it has a model of its own; without one it
stays on the PD-target route whatever --integration says, since there is no
learned state to make authoritative.

Isaac Lab 3.0 runs headless unless a visualizer is requested. Teleop needs the
Omniverse app window for gamepad events, so this script defaults to ``--viz kit``.
Pass ``--viz none`` (or the deprecated ``--headless``) to run without a window; the
gamepad then reports no input and the sim free-runs on zero commands.

.. code-block:: bash

    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/sim.py --tool bucket
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import weakref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaaclab.app import AppLauncher

from endstop_guard import EndStopGuard
from sim_common import (  # noqa: E402
    ARM_JOINT_NAMES,
    CARRIAGE_DAMPING,
    CARRIAGE_STIFFNESS,
    DEFAULT_ARM_MODEL_DIR,
    DEFAULT_SLEW_MODEL_DIR,
    DEFAULT_TOOL,
    DISABLE_ROBOT_GRAVITY,
    GRIPPER_DAMPING,
    GRIPPER_JOINT_NAMES,
    GRIPPER_STIFFNESS,
    NN_ARM_VEL_LIMIT_DEFAULT,
    NN_ARM_VEL_WARN_DEFAULT,
    SIM_HZ,
    SLEW_JOINT_NAMES,
    SOLVER_VELOCITY_ITERATIONS,
    TOOL_VARIANTS,
    RealtimeReporter,
    configure_joint_drive,
    has_gripper_joints,
    resolve_demo_defaults,
    robot_usd_path,
    sanitize_velocity_prediction,
)

parser = argparse.ArgumentParser(description="Excavator NN sim with gamepad teleop")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--tool",
    choices=list(TOOL_VARIANTS),
    default=DEFAULT_TOOL,
    help="End effector variant to spawn (selects assets/excavator_<tool>.usd)",
)
parser.add_argument(
    "--model",
    default=DEFAULT_ARM_MODEL_DIR,
    help="Directory containing the trained arm actuator model artifact",
)
parser.add_argument(
    "--slew-model",
    default=DEFAULT_SLEW_MODEL_DIR,
    help="Directory containing the trained slew actuator model artifact. When it "
    "does not exist, the slew joint falls back to the raw stick command.",
)
parser.add_argument("--weights", default=None, help="Optional arm checkpoint state dict")
parser.add_argument(
    "--model_device", default="cpu", help="Actuator inference device; CPU is faster for the single-env demo"
)
parser.add_argument(
    "--torch_threads", type=int, default=2, help="Process-wide CPU tensor threads for this single-env demo"
)
parser.add_argument(
    "--max_steps", type=int, default=0, help="Stop after this many 100 Hz control steps; 0 runs continuously"
)
parser.add_argument(
    "--physics_substeps",
    type=int,
    default=None,
    help="Physics steps per 100 Hz control step; default 1, or 2 with carriage rocking",
)
parser.add_argument(
    "--realtime_report_interval",
    type=float,
    default=2.0,
    help="Wall seconds between sim/wall real-time multiplier prints; 0 disables (startup excluded)",
)
parser.add_argument(
    "--replay_npz", default=None, help="Replay a continuous NPZ chunk containing q, v_smooth21, u and t"
)
parser.add_argument(
    "--replay_start", type=int, default=200, help="Replay start sample, leaving measured history for warmup"
)
parser.add_argument(
    "--replay_out", default=None, help="Save replay state and command arrays to this NPZ file"
)
parser.add_argument(
    "--arm_stiffness",
    type=float,
    nargs=3,
    default=None,
    metavar=("BOOM", "ARM", "TOOL"),
    help="Position gains [N m/rad]; default 2400, or 10000 with rocking",
)
parser.add_argument(
    "--arm_damping",
    type=float,
    nargs=3,
    default=None,
    metavar=("BOOM", "ARM", "TOOL"),
    help="Velocity gains [N m s/rad]; default 120, or 300 with rocking",
)
parser.add_argument(
    "--no_target_velocity_feedforward",
    action="store_true",
    help="Use legacy position-only arm PD targets for comparison",
)
parser.add_argument(
    "--target_feedback",
    choices=("reference", "physics"),
    default="reference",
    help="Arm NN state in target mode: reference avoids recycling PD tracking error through the model; "
    "physics preserves the legacy experimental feedback route",
)
parser.add_argument(
    "--endstop_mode",
    choices=("hold", "legacy"),
    default="hold",
    help="hold: suppress learned recoil only while a valve pushes into a contacted limit; "
    "legacy: reproduce the previous clamp-only reference",
)
parser.add_argument(
    "--carriage_rocking",
    action="store_true",
    help="Opt-in compliant upper-carriage roll/pitch variant; lower carriage stays fixed",
)
parser.add_argument(
    "--rocking_frequency",
    type=float,
    default=7.4,
    help="Nominal effective carriage spring frequency at the initial pose [Hz]",
)
parser.add_argument(
    "--rocking_damping_ratio",
    type=float,
    default=0.03,
    help="Pitch drive damping ratio; default .03 gives about .057 with the tested arm drives",
)
parser.add_argument(
    "--solver_velocity_iterations",
    type=int,
    default=SOLVER_VELOCITY_ITERATIONS,
    help="PhysX articulation velocity iterations; useful for constrained target-mode tracking",
)
parser.add_argument(
    "--integration",
    choices=["direct", "target"],
    default=None,
    help="direct: learned model integrates joint state and writes it to the sim "
    "(Egli & Hutter convention -- the model IS the dynamics). "
    "target: integrate into a position target and let the articulation PD drive track it. "
    "Default direct, or target with --carriage_rocking.",
)
parser.add_argument(
    "--vel-limit",
    type=float,
    default=NN_ARM_VEL_LIMIT_DEFAULT,
    help="Hard safety clamp on NN-predicted arm velocity [rad/s]",
)
parser.add_argument(
    "--vel-warn",
    type=float,
    default=NN_ARM_VEL_WARN_DEFAULT,
    help="Warn, but do not clamp, above this NN-predicted arm speed [rad/s]",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
try:
    resolve_demo_defaults(args_cli)
except ValueError as exc:
    parser.error(str(exc))
if args_cli.vel_limit <= 0.0:
    parser.error("--vel-limit must be positive")
if args_cli.vel_warn < 0.0:
    parser.error("--vel-warn must be non-negative")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be non-negative")
if not 1 <= args_cli.physics_substeps <= 10:
    parser.error("--physics_substeps must be between 1 and 10")
if not math.isfinite(args_cli.realtime_report_interval) or args_cli.realtime_report_interval < 0:
    parser.error("--realtime_report_interval must be finite and non-negative")
if args_cli.torch_threads < 1:
    parser.error("--torch_threads must be positive")
if args_cli.replay_out and not args_cli.replay_npz:
    parser.error("--replay_out requires --replay_npz")
if not all(math.isfinite(value) and value > 0 for value in args_cli.arm_stiffness):
    parser.error("--arm_stiffness values must be finite and positive")
if not all(math.isfinite(value) and value >= 0 for value in args_cli.arm_damping):
    parser.error("--arm_damping values must be finite and non-negative")
if not 1 <= args_cli.solver_velocity_iterations <= 255:
    parser.error("--solver_velocity_iterations must be between 1 and 255")
if not math.isfinite(args_cli.rocking_frequency) or not 1 <= args_cli.rocking_frequency <= 15:
    parser.error("--rocking_frequency must be between 1 and 15 Hz")
if not math.isfinite(args_cli.rocking_damping_ratio) or not 0.01 <= args_cli.rocking_damping_ratio <= 2:
    parser.error("--rocking_damping_ratio must be between 0.01 and 2")
args_cli.num_envs = 1  # teleop always single env

# Isaac Lab 3.0 resolves the visualizer from --viz (or SimulationCfg.visualizer_cfgs) and
# runs headless when neither asks for Kit. Gamepad teleop needs the Kit app window, so
# request it unless the user made an explicit choice. --headless still wins downstream.
if not getattr(args_cli, "visualizer_explicit", False):
    args_cli.visualizer = ["kit"]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- post-launch imports ----------------------------------------------------

import numpy as np
import torch
from pxr import UsdPhysics

torch.set_num_threads(args_cli.torch_threads)

import carb
import carb.input
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.schemas import RigidBodyBaseCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

from actuators import (  # noqa: E402
    DirectIntegrationActuator,
    DirectIntegrationActuatorCfg,
    HydraulicActuatorNet,
    VelocityIntegratedActuator,
    VelocityIntegratedActuatorCfg,
)

# -- constants --------------------------------------------------------------

ARM_MODEL_DIR = args_cli.model
SLEW_MODEL_DIR = args_cli.slew_model
TOOL = args_cli.tool
ROBOT_USD = robot_usd_path(TOOL)
if args_cli.carriage_rocking:
    ROBOT_USD = ROBOT_USD.removesuffix(".usd") + "_rocking.usd"
    if not os.path.isfile(ROBOT_USD):
        raise FileNotFoundError(f"Missing opt-in rocking variant: {ROBOT_USD}")
ROCKING_JOINT_NAMES = ["revolute_carriage_roll", "revolute_carriage_pitch"]
MODEL_JOINT_NAMES = args_cli.model_joint_names
CONTROL_DECIMATION = 1  # one NN update per 100 Hz outer control step
DEAD_ZONE = 0.30
CARRIAGE_VEL_MAX = 0.8  # rad/s -- manual/fallback slew velocity scale
ARM_VEL_MAX = 0.5  # rad/s -- manual mode arm velocity scale
NN_ARM_VEL_MAX = args_cli.vel_limit
NN_ARM_VEL_WARN = args_cli.vel_warn


# -- scene ------------------------------------------------------------------


def _robot_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Drive groups for the selected tool variant.

    The bucket asset welds its bucket on with a fixed joint, so only the gripper
    variant has the three extra claw/rotate DOFs to actuate.
    """
    actuators: dict[str, ImplicitActuatorCfg] = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness=dict(zip(ARM_JOINT_NAMES, args_cli.arm_stiffness)),
            damping=dict(zip(ARM_JOINT_NAMES, args_cli.arm_damping)),
        ),
        "slew": ImplicitActuatorCfg(
            joint_names_expr=SLEW_JOINT_NAMES,
            stiffness=CARRIAGE_STIFFNESS,
            damping=CARRIAGE_DAMPING,
        ),
    }
    if has_gripper_joints(TOOL):
        actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=GRIPPER_JOINT_NAMES,
            stiffness=GRIPPER_STIFFNESS,
            damping=GRIPPER_DAMPING,
        )
    if args_cli.carriage_rocking:
        actuators["carriage_rocking"] = ImplicitActuatorCfg(
            joint_names_expr=ROCKING_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
        )
    return actuators


@configclass
class ExcavatorSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    # ground = AssetBaseCfg(
    #    prim_path="/World/defaultGroundPlane",
    #    spawn=sim_utils.GroundPlaneCfg(),
    # )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            copy_from_source=True,
            # The learned velocity already includes the real machine's gravity response.
            # RigidBodyBaseCfg is the backend-portable half of the 2.x RigidBodyPropertiesCfg.
            rigid_props=RigidBodyBaseCfg(disable_gravity=DISABLE_ROBOT_GRAVITY),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                solver_velocity_iteration_count=args_cli.solver_velocity_iterations,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # note: Isaac Lab 3.0 quaternions are (x, y, z, w); `rot` is left at identity here.
            pos=(0.0, 0.0, 0.09),
            joint_pos={
                "revolute_carriage": 0.0,
                "revolute_lift": -0.5498,  # -31.5 deg
                "revolute_tilt": 1.2549,  #  71.9 deg
                "revolute_tool": -0.7540,  # -43.2 deg
            },
            joint_vel={".*": 0.0},
        ),
        actuators=_robot_actuators(),
    )


def fix_carriage_to_world(stage, num_envs: int) -> None:
    """Weld each env's ``lower_carriage`` to the world frame.

    Done on the stage rather than through ``fix_root_link`` because the asset's
    articulation root (``/excavator``) is a plain Xform without ``RigidBodyAPI``,
    which is the one case Isaac Lab's fixed-root helper explicitly refuses.
    """
    for env_idx in range(num_envs):
        robot_prim = f"/World/envs/env_{env_idx}/Robot"
        joint_prim = f"{robot_prim}/Joints/world_fixed"
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_prim)
        fixed_joint.CreateBody1Rel().SetTargets([f"{robot_prim}/lower_carriage"])
        print(f"[INFO] Fixed joint created: {joint_prim} -> {robot_prim}/lower_carriage")


# -- joint groups -----------------------------------------------------------
#
# One learned model per joint group. Arm and slew artifacts ship with the repository.
# Which groups exist, what their joints are called and
# what to do when a model is missing are all decisions of this script -- the
# actuators below it are told, never asked.


def make_actuator(
    scene,
    joint_names: list[str],
    sim_dt: float,
    *,
    direct: bool,
    velocity_feedforward: bool = False,
    clamp_to_limits: bool = True,
    continuous: bool = False,
):
    """Build the integrating actuator for one joint group.

    Args:
        scene: Scene holding the articulation.
        joint_names: Joints of the group, in model-channel order.
        sim_dt: Integration period [s].
        direct: Route selector. True writes integrated joint state into the sim
            (the model is the dynamics); False integrates into a position target
            for the articulation's PD drive to chase.
        velocity_feedforward: Also update the desired PD velocity [rad/s].
        clamp_to_limits: Clamp integrated positions to soft joint bounds.
        continuous: Wrap only the PD error for unlimited revolute joints.

    Returns:
        A :class:`DirectIntegrationActuator` or :class:`VelocityIntegratedActuator`.
    """
    if direct:
        return DirectIntegrationActuator(
            DirectIntegrationActuatorCfg(joint_names_expr=list(joint_names), clamp_to_limits=clamp_to_limits),
            scene=scene,
            sim_dt=sim_dt,
        )
    return VelocityIntegratedActuator(
        VelocityIntegratedActuatorCfg(
            joint_names_expr=list(joint_names),
            velocity_feedforward=velocity_feedforward,
            clamp_to_limits=clamp_to_limits,
            continuous=continuous,
        ),
        scene=scene,
        sim_dt=sim_dt,
    )


def load_controller(model_dir: str, sim_dt: float, label: str, weights=None):
    """Load one learned forward model, or report that there is not one yet.

    Returning None rather than a stub is deliberate: ``HydraulicActuatorNet`` has
    no degraded mode, so the caller has to decide what a missing model means for
    its joint group. That keeps "we have not trained this one yet" out of every
    layer beneath this script.

    Args:
        model_dir: Directory holding the model artifact.
        sim_dt: Control period the model will be stepped at [s].
        label: Joint group name, for logging.
        weights: Optional checkpoint overriding the artifact's own weights.

    Returns:
        A :class:`HydraulicActuatorNet`, or None when ``model_dir`` does not exist.
    """
    if not os.path.isdir(model_dir):
        print(f"[INFO] No {label} model at {model_dir}; {label} runs on the direct stick command")
        return None
    controller = HydraulicActuatorNet(
        model_dir,
        device=args_cli.model_device,
        sim_dt=sim_dt * CONTROL_DECIMATION,
        weights=weights,
    )
    controller.reset()
    print(f"[INFO] {label.capitalize()} model: {model_dir}")
    print(
        f"[INFO]   dt={controller.dt}s  hist_qdot={controller.hist_qdot}  "
        f"hist_u={controller.hist_u}  target={controller.target_mode}"
    )
    print(f"[INFO]   joints={controller.joint_names}  commands={controller.command_names}")
    return controller


def require_channels(controller, label: str, joint_names: list[str], num_commands: int) -> None:
    """Check a model's widths against the group and sticks this script drives it with.

    The model's channel order comes from its training columns, the joint order
    comes from the USD and the command count comes from the gamepad mapping
    below; nothing links the three but this script, so they are checked at
    startup rather than discovered as wrong motion later.
    """
    if controller.num_joints != len(joint_names):
        raise ValueError(
            f"{label} model predicts {controller.num_joints} joints "
            f"({controller.joint_names}) but {len(joint_names)} are being driven "
            f"({joint_names}). Retrain with matching --qdot-cols, or point "
            "--model/--slew-model at the right artifact."
        )
    if controller.num_commands != num_commands:
        raise ValueError(
            f"{label} model expects {controller.num_commands} command channels "
            f"({controller.command_names}) but this script maps {num_commands} stick "
            "axes onto it. Retrain with matching --u-cols, or extend the gamepad "
            "mapping in this file."
        )


# -- gamepad ----------------------------------------------------------------


def _deadzone(v: float) -> float:
    if abs(v) < DEAD_ZONE:
        return 0.0
    s = 1.0 if v > 0.0 else -1.0
    return s * (abs(v) - DEAD_ZONE) / (1.0 - DEAD_ZONE)


class XboxController:
    """Minimal gamepad reader for direct valve-command teleop.

    Degrades to a zero-command stub when no Omniverse app window exists, which is
    the case whenever the script is launched without the Kit visualizer.
    """

    def __init__(self):
        get_settings_manager().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        # index: 0=lift_up 1=lift_dn 2=carriage_l 3=carriage_r 4=tilt_up 5=tilt_dn 6=tool_r 7=tool_l
        self._axes = np.zeros(8, dtype=np.float32)
        self._a_pressed = False
        self._prev_a = False
        self._b_pressed = False
        self._prev_b = False
        self._appwindow = None
        self._input = None
        self._gamepad = None
        self._sub = None

        # omni.appwindow only exists once Kit has a window, i.e. when the Kit
        # visualizer is running. Headless runs keep the zero-command stub.
        try:
            import omni.appwindow

            self._appwindow = omni.appwindow.get_default_app_window()
        except ImportError:
            self._appwindow = None
        if self._appwindow is None:
            print("[Gamepad] No app window (headless run); teleop disabled, commands stay at zero")
            return

        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda ev, *a, obj=weakref.proxy(self): obj._on_event(ev),
        )
        name = self._input.get_gamepad_name(self._gamepad)
        print(f"[Gamepad] {'Connected: ' + name if name else 'No gamepad detected'}")

    def __del__(self):
        if getattr(self, "_input", None) is not None and getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._sub)

    def _on_event(self, ev):
        GI = carb.input.GamepadInput
        v = ev.value
        m = {
            GI.LEFT_STICK_UP: (0, v),
            GI.LEFT_STICK_DOWN: (1, v),
            GI.LEFT_STICK_LEFT: (2, v),
            GI.LEFT_STICK_RIGHT: (3, v),
            GI.RIGHT_STICK_UP: (4, v),
            GI.RIGHT_STICK_DOWN: (5, v),
            GI.RIGHT_STICK_RIGHT: (6, v),
            GI.RIGHT_STICK_LEFT: (7, v),
        }
        if ev.input in m:
            idx, val = m[ev.input]
            self._axes[idx] = val
        elif ev.input == GI.A:
            self._a_pressed = v > 0.5
        elif ev.input == GI.B:
            self._b_pressed = v > 0.5
        return True

    @staticmethod
    def _axis(pos: float, neg: float) -> float:
        raw = float(pos) - float(neg)
        return _deadzone(raw)

    def read(self) -> tuple[float, float, float, float]:
        """Returns (lift_cmd, carriage_vel_norm, tilt_cmd, tool_cmd) each in [-1, 1]."""
        tilt = -self._axis(self._axes[0], self._axes[1])  # left  Y
        carriage = -self._axis(self._axes[3], self._axes[2])  # left  X
        lift = self._axis(self._axes[4], self._axes[5])  # right Y
        tool = -self._axis(self._axes[6], self._axes[7])  # right X
        return lift, carriage, tilt, tool

    def mode_toggle_requested(self) -> bool:
        cur = self._a_pressed
        rising = cur and not self._prev_a
        self._prev_a = cur
        return rising

    def reset_requested(self) -> bool:
        cur = self._b_pressed
        rising = cur and not self._prev_b
        self._prev_b = cur
        return rising


# -- main -------------------------------------------------------------------


def configure_rocking_drive(robot: Articulation) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Set effective carriage spring gains [N m/rad, N m s/rad]."""
    if not args_cli.carriage_rocking:
        return [], np.zeros(2), np.zeros(2)
    ids, _ = robot.find_joints(ROCKING_JOINT_NAMES, preserve_order=True)
    if args_cli.learned_carriage:
        configure_joint_drive(robot, ids, direct=True, stiffness=0.0, damping=0.0)
        print("[ROCKING] Neural carriage motion; physical springs and dampers disabled")
        return ids, np.zeros(2), np.zeros(2)
    inertia = robot.data.mass_matrix.torch[0].diagonal()[ids].detach().cpu().numpy()
    omega = 2 * np.pi * args_cli.rocking_frequency
    kp = inertia * omega**2
    kd = 2 * np.array([1.0, args_cli.rocking_damping_ratio], dtype=np.float32) * inertia * omega
    device = robot.data.joint_pos.torch.device
    configure_joint_drive(
        robot,
        ids,
        direct=False,
        stiffness=torch.as_tensor(kp[None], dtype=torch.float32, device=device),
        damping=torch.as_tensor(kd[None], dtype=torch.float32, device=device),
    )
    robot.set_joint_position_target_index(
        target=torch.zeros_like(robot.data.joint_pos.torch[:, ids]), joint_ids=ids
    )
    robot.set_joint_velocity_target_index(
        target=torch.zeros_like(robot.data.joint_vel.torch[:, ids]), joint_ids=ids
    )
    print(
        f"[ROCKING] Passive upper carriage: inertia={inertia} kg m2, "
        f"stiffness={kp} N m/rad, damping={kd} N m s/rad"
    )
    print("[ROCKING] Experimental linear compliance; +/-3 deg guard, not a calibrated backlash model")
    return ids, kp, kd


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / (SIM_HZ * args_cli.physics_substeps),
        render_interval=1,
        device=args_cli.device,
        # Isaac Lab 3.0: the backend config moved from `physx=` to `physics=`.
        physics=PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = ExcavatorSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    fix_carriage_to_world(sim.stage, scene.num_envs)

    sim.reset()

    robot: Articulation = scene["robot"]
    print(f"[INFO] Tool variant: {TOOL}  ({ROBOT_USD})")
    print("[INFO] Joints:", robot.joint_names)
    print("[INFO] Fixed base:", robot.is_fixed_base)

    physics_dt = sim.get_physics_dt()
    sim_dt = physics_dt * args_cli.physics_substeps
    print(
        f"[INFO] Control/model dt={sim_dt:g}s; physics dt={physics_dt:g}s "
        f"({args_cli.physics_substeps} substeps)"
    )
    direct_integration = args_cli.integration == "direct"

    # Arm joints: the learned model drives velocity. Two conventions --
    #   direct : integrate to joint state and write it (model is the dynamics, paper-style)
    #   target : integrate to a position target and let the PD drive chase it
    arm_actuator = make_actuator(
        scene,
        MODEL_JOINT_NAMES,
        physics_dt,
        direct=direct_integration,
        velocity_feedforward=not args_cli.no_target_velocity_feedforward,
    )
    configure_joint_drive(
        robot,
        arm_actuator.joint_ids,
        direct=direct_integration,
        stiffness=torch.tensor(
            [args_cli.arm_stiffness + [0.0] * (len(MODEL_JOINT_NAMES) - 3)],
            device=robot.data.joint_pos.torch.device,
        ),
        damping=torch.tensor(
            [args_cli.arm_damping + [0.0] * (len(MODEL_JOINT_NAMES) - 3)],
            device=robot.data.joint_pos.torch.device,
        ),
    )
    print(f"[INFO] Arm integration mode: {args_cli.integration}")
    if not direct_integration:
        print(
            f"[INFO] Arm PD stiffness={args_cli.arm_stiffness}, damping={args_cli.arm_damping}, "
            f"velocity feedforward={not args_cli.no_target_velocity_feedforward}, "
            f"NN feedback={args_cli.target_feedback}"
        )
    print(f"[INFO] Robot gravity disabled: {DISABLE_ROBOT_GRAVITY}")
    print(f"[INFO] NN velocity warning/clamp: {NN_ARM_VEL_WARN:g}/{NN_ARM_VEL_MAX:g} rad/s")
    print("[INFO] Arm joints resolved:", list(zip(arm_actuator.joint_ids, arm_actuator.joint_names)))

    arm_controller = load_controller(ARM_MODEL_DIR, sim_dt, "arm", weights=args_cli.weights)
    endstop_guard = (
        EndStopGuard(arm_actuator._joint_pos_limits[0, :3].cpu().numpy(), sim_dt)
        if args_cli.endstop_mode == "hold"
        else None
    )
    print(f"[INFO] Arm end-stop policy: {args_cli.endstop_mode}")
    if arm_controller is None:
        raise FileNotFoundError(
            f"No arm model at {ARM_MODEL_DIR}. The arm model ships with the repository; "
            "pass --model if it lives elsewhere."
        )
    # Three stick axes map onto the arm model: lift, tilt, tool.
    require_channels(arm_controller, "arm", MODEL_JOINT_NAMES, num_commands=3)

    # Slew uses the same generic class with its own one-channel artifact.
    # A missing optional artifact enables this demo's stick-velocity fallback.
    slew_controller = load_controller(SLEW_MODEL_DIR, sim_dt, "slew")
    slew_learned = slew_controller is not None
    if slew_learned:
        # One stick axis maps onto the slew model.
        require_channels(slew_controller, "slew", SLEW_JOINT_NAMES, num_commands=1)
    # With a model the slew joint takes the arm's integration route; without one it
    # stays on the PD-target route it has always used, whatever --integration says.
    slew_actuator = make_actuator(
        scene,
        SLEW_JOINT_NAMES,
        physics_dt,
        direct=direct_integration and slew_learned,
        clamp_to_limits=False,
        velocity_feedforward=not args_cli.no_target_velocity_feedforward,
        continuous=True,
    )
    configure_joint_drive(
        robot,
        slew_actuator.joint_ids,
        direct=direct_integration and slew_learned,
        stiffness=CARRIAGE_STIFFNESS,
        damping=CARRIAGE_DAMPING,
    )
    slew_direct_integration = direct_integration and slew_learned
    print("[INFO] Slew joint resolved:", list(zip(slew_actuator.joint_ids, slew_actuator.joint_names)))
    rocking_ids, rocking_kp, rocking_kd = configure_rocking_drive(robot)
    held_rocking = None
    if args_cli.learned_carriage:
        omitted = [name for name in ROCKING_JOINT_NAMES if name not in MODEL_JOINT_NAMES]
        if omitted:
            held_rocking = make_actuator(scene, omitted, physics_dt, direct=True)

    # Gamepad
    teleop = XboxController()

    # Default pose for reset
    default_pos = robot.data.default_joint_pos.torch.clone()
    default_vel = torch.zeros_like(default_pos)

    print("[INFO] Controls:")
    print("  Left  Y  : tilt  Right Y : lift")
    print("  Left  X  : slew        Right X : tool")
    print(
        "  A        : toggle MANUAL / NN command source"
        + ("" if slew_learned else "  (slew always direct: no model)")
    )
    print("  B        : reset")

    use_nn = True
    print("[INFO] Starting in NN mode")

    # Persistent command tensors -- updated at the control rate, applied every physics step
    device = robot.data.joint_pos.torch.device
    arm_vel_t = torch.zeros(1, arm_actuator.num_joints, device=device)
    slew_vel_t = torch.zeros(1, slew_actuator.num_joints, device=device)
    u = np.zeros(arm_controller.num_commands, dtype=np.float32)
    u_slew = np.zeros(slew_controller.num_commands if slew_learned else 1, dtype=np.float32)
    velocity_warning_active = False
    velocity_clamp_active = False

    replay = None
    replay_rows = []
    replay_times = []
    rocking_rows = []
    slew_rows = []
    if args_cli.replay_npz:
        with np.load(args_cli.replay_npz) as source:
            replay = {key: source[key] for key in ("q", "v_smooth21", "u", "t")}
            for key in ("slew_q", "slew_v", "slew_u"):
                if key in source:
                    replay[key] = source[key]
            if any(key in replay for key in ("slew_q", "slew_v", "slew_u")):
                if not all(
                    key in replay and replay[key].shape == (len(replay["q"]), 1)
                    for key in ("slew_q", "slew_v", "slew_u")
                ):
                    raise ValueError("Slew replay requires matching (samples, 1) q, v and u arrays")
        start = args_cli.replay_start
        history = max(
            arm_controller.hist_q,
            (arm_controller.hist_qdot - 1) * arm_controller.qdot_stride + 1,
            (arm_controller.hist_u - 1) * arm_controller.u_stride + 1,
        )
        if start < history or start >= len(replay["q"]) - 1:
            raise ValueError("Replay start must leave complete model history and at least one future sample")
        if not np.allclose(np.diff(replay["t"]), sim_dt, atol=1e-6):
            raise ValueError("Replay must be one continuous chunk at the simulation period")
        if not all(np.isfinite(value).all() for value in replay.values()):
            raise ValueError("Replay contains non-finite observations")
        initial_pos = default_pos.clone()
        initial_vel = default_vel.clone()
        initial_pos[0, arm_actuator.joint_ids] = torch.as_tensor(replay["q"][start], device=device)
        initial_vel[0, arm_actuator.joint_ids] = torch.as_tensor(replay["v_smooth21"][start], device=device)
        if "slew_q" in replay:
            initial_pos[0, slew_actuator.joint_ids] = torch.as_tensor(replay["slew_q"][start], device=device)
            initial_vel[0, slew_actuator.joint_ids] = torch.as_tensor(replay["slew_v"][start], device=device)
        robot.write_joint_position_to_sim_index(position=initial_pos)
        robot.write_joint_velocity_to_sim_index(velocity=initial_vel)
        robot.set_joint_position_target_index(target=initial_pos)
        # The measured velocity seeds physical state, not a persistent PD
        # feedforward command. The target actuator updates its own commands.
        robot.set_joint_velocity_target_index(target=default_vel)
        arm_actuator.reset()
        slew_actuator.reset()
        if held_rocking is not None:
            held_rocking.reset()
        arm_controller.reset()
        for row in range(start - history, start):
            arm_controller.predict_velocity(replay["q"][row], replay["v_smooth21"][row], replay["u"][row])
        if slew_learned and "slew_q" in replay:
            slew_history = max(
                slew_controller.hist_q,
                (slew_controller.hist_qdot - 1) * slew_controller.qdot_stride + 1,
                (slew_controller.hist_u - 1) * slew_controller.u_stride + 1,
            )
            if start < slew_history:
                raise ValueError("Replay start leaves insufficient slew history")
            slew_controller.reset()
            for row in range(start - slew_history, start):
                slew_controller.predict_velocity(
                    replay["slew_q"][row], replay["slew_v"][row], replay["slew_u"][row]
                )
        print(f"[INFO] Replaying {args_cli.replay_npz} from sample {start}")

    def joint_state(actuator, direct: bool, *, reference: bool = False):
        """Feed a model its own integrated state under direct integration.

        Target-reference feedback advances the same learned reference while
        PhysX tracks it independently. Physical feedback is retained as an
        explicit experiment; it recycles PD error into the forward model.
        """
        if direct:
            return actuator.position[0].cpu().numpy(), actuator.velocity[0].cpu().numpy()
        if reference:
            return actuator.target_position[0].cpu().numpy(), actuator.reference_velocity[0].cpu().numpy()
        ids = actuator.joint_ids
        return (
            robot.data.joint_pos.torch[0, ids].cpu().numpy(),
            robot.data.joint_vel.torch[0, ids].cpu().numpy(),
        )

    step = 0
    realtime = RealtimeReporter(
        args_cli.realtime_report_interval,
        wall_s=time.perf_counter(),
        sim_s=sim.get_physics_step_count() * physics_dt,
    )
    while simulation_app.is_running():
        if args_cli.max_steps and step >= args_cli.max_steps:
            break
        if replay is not None and args_cli.replay_start + step >= len(replay["q"]) - 1:
            break

        # -- mode toggle ----------------------------------------------------
        if replay is None and teleop.mode_toggle_requested():
            use_nn = not use_nn
            arm_controller.reset()
            arm_actuator.reset()
            if endstop_guard is not None:
                endstop_guard.reset()
            if slew_learned:
                slew_controller.reset()
                slew_actuator.reset()
            print(f"[step {step}] Mode -> {'NN' if use_nn else 'MANUAL'}")

        # -- reset ----------------------------------------------------------
        if replay is None and teleop.reset_requested():
            robot.write_joint_position_to_sim_index(position=default_pos)
            robot.write_joint_velocity_to_sim_index(velocity=default_vel)
            robot.set_joint_position_target_index(target=default_pos)
            robot.set_joint_velocity_target_index(target=default_vel)
            arm_controller.reset()
            arm_actuator.reset()
            if endstop_guard is not None:
                endstop_guard.reset()
            slew_actuator.reset()
            if slew_learned:
                slew_controller.reset()
            arm_vel_t[:] = 0.0
            slew_vel_t[:] = 0.0
            u[:] = 0.0
            u_slew[:] = 0.0
            scene.write_data_to_sim()
            if direct_integration:
                arm_actuator.sync_to_sim()
            if slew_direct_integration:
                slew_actuator.sync_to_sim()
            print(f"[step {step}] Reset")
            continue

        # -- update commands at the 100 Hz control rate ---------------------
        if step % CONTROL_DECIMATION == 0:
            lift_cmd, slew_cmd, tilt_cmd, tool_cmd = teleop.read()
            if replay is not None:
                lift_cmd, tilt_cmd, tool_cmd = replay["u"][args_cli.replay_start + step]
                slew_cmd = (
                    float(replay["slew_u"][args_cli.replay_start + step, 0]) if "slew_u" in replay else 0.0
                )
            # THE joint mapping: stick axes onto the arm model's channel order,
            # which is [lift, tilt, tool]. It lives here, in the demo.
            u = np.array([lift_cmd, tilt_cmd, tool_cmd], dtype=np.float32)

            if use_nn:
                q, qdot = joint_state(
                    arm_actuator, direct_integration, reference=args_cli.target_feedback == "reference"
                )
                qdot_pred = arm_controller.predict_velocity(q, qdot, u)  # rad/s
                qdot_pred, max_abs_vel, finite, clipped = sanitize_velocity_prediction(
                    qdot_pred, NN_ARM_VEL_MAX
                )
                if not finite:
                    print(f"[step {step}] [WARN] Non-finite NN velocity prediction; commanding zero")
                over_warning = max_abs_vel > NN_ARM_VEL_WARN
                if over_warning and not velocity_warning_active:
                    print(
                        f"[step {step}] [WARN] NN velocity reached {max_abs_vel:.3f} rad/s "
                        f"(warning threshold {NN_ARM_VEL_WARN:g}; still allowed)"
                    )
                if clipped and not velocity_clamp_active:
                    print(
                        f"[step {step}] [WARN] NN velocity reached {max_abs_vel:.3f} rad/s; "
                        f"clamped to {NN_ARM_VEL_MAX:g}"
                    )
                velocity_warning_active = over_warning
                velocity_clamp_active = clipped
                if endstop_guard is not None:
                    qdot_pred[:3] = endstop_guard.apply(q[:3], qdot_pred[:3], u)
                arm_vel_t = torch.from_numpy(qdot_pred).unsqueeze(0).to(device)
            else:
                arm_vel_t = torch.tensor(
                    [
                        [lift_cmd * ARM_VEL_MAX, tilt_cmd * ARM_VEL_MAX, tool_cmd * ARM_VEL_MAX]
                        + [0.0] * (arm_actuator.num_joints - 3)
                    ],
                    device=device,
                )

            if use_nn and slew_learned:
                u_slew = np.array([slew_cmd], dtype=np.float32)
                q_slew, qdot_slew = joint_state(
                    slew_actuator, slew_direct_integration, reference=args_cli.target_feedback == "reference"
                )
                slew_pred = slew_controller.predict_velocity(q_slew, qdot_slew, u_slew)
                slew_pred, _, slew_finite, _ = sanitize_velocity_prediction(slew_pred, NN_ARM_VEL_MAX)
                if not slew_finite:
                    print(f"[step {step}] [WARN] Non-finite slew velocity prediction; commanding zero")
                slew_vel_t = torch.from_numpy(slew_pred).unsqueeze(0).to(device)
            else:
                # MANUAL mode, or no slew model yet: the stick is the velocity.
                slew_vel_t = torch.tensor([[slew_cmd * CARRIAGE_VEL_MAX]], device=device)

        # Model updates stay at 100 Hz. Integrating each smaller setpoint
        # increment keeps PD position/velocity targets synchronized throughout.
        for substep in range(args_cli.physics_substeps):
            arm_actuator.apply_velocity_command(arm_vel_t)
            slew_actuator.apply_velocity_command(slew_vel_t)
            if held_rocking is not None:
                held_rocking.apply_velocity_command(torch.zeros_like(held_rocking.velocity))
            scene.write_data_to_sim()
            sim.step(render=not direct_integration and substep == args_cli.physics_substeps - 1)
            scene.update(physics_dt)
            if direct_integration:
                # Do not let PhysX integrate the prescribed state a second time.
                arm_actuator.sync_to_sim()
                if slew_direct_integration:
                    slew_actuator.sync_to_sim()
                if held_rocking is not None:
                    held_rocking.sync_to_sim()
        if direct_integration and sim.is_rendering:
            sim.render()

        if replay is not None:
            replay_times.append((sim.get_physics_step_count() * physics_dt, time.perf_counter()))
            rocking_rows.append(
                np.concatenate(
                    (
                        robot.data.joint_pos.torch[0, rocking_ids].cpu().numpy(),
                        robot.data.joint_vel.torch[0, rocking_ids].cpu().numpy(),
                    )
                )
            )
            slew_rows.append(
                np.concatenate(
                    (
                        robot.data.joint_pos.torch[0, slew_actuator.joint_ids].cpu().numpy(),
                        robot.data.joint_vel.torch[0, slew_actuator.joint_ids].cpu().numpy(),
                        np.array([slew_cmd], np.float32),
                        (
                            slew_actuator.position
                            if slew_direct_integration
                            else slew_actuator.target_position
                        )[0]
                        .cpu()
                        .numpy(),
                        (
                            slew_actuator.velocity
                            if slew_direct_integration
                            else slew_actuator.reference_velocity
                        )[0]
                        .cpu()
                        .numpy(),
                    )
                )
            )
            replay_rows.append(
                np.concatenate(
                    (
                        robot.data.joint_pos.torch[0, arm_actuator.joint_ids].cpu().numpy(),
                        robot.data.joint_vel.torch[0, arm_actuator.joint_ids].cpu().numpy(),
                        u,
                        (arm_actuator.position if direct_integration else arm_actuator.target_position)[0]
                        .cpu()
                        .numpy(),
                        (arm_actuator.velocity if direct_integration else arm_actuator.target_velocity)[0]
                        .cpu()
                        .numpy(),
                        (arm_actuator.velocity if direct_integration else arm_actuator.reference_velocity)[0]
                        .cpu()
                        .numpy(),
                    )
                )
            )

        if step % 200 == 0:
            p = robot.data.joint_pos.torch[0, arm_actuator.joint_ids].cpu().numpy()
            slew_p = robot.data.joint_pos.torch[0, slew_actuator.joint_ids].cpu().numpy()
            arm_vel_dbg = arm_vel_t.squeeze(0).detach().cpu().numpy()
            slew_vel_dbg = slew_vel_t.squeeze(0).detach().cpu().numpy()
            print(
                f"[step {step:6d}]  mode={'NN' if use_nn else 'MANUAL'}  u={u}  "
                f"qdot_cmd={arm_vel_dbg}  "
                f"lift={p[0]:.3f}  tilt={p[1]:.3f}  tool={p[2]:.3f}  "
                f"slew={slew_p[0]:.3f} ({slew_vel_dbg[0]:+.3f})  rad"
            )
        step += 1
        performance = realtime.update(
            wall_s=time.perf_counter(), sim_s=sim.get_physics_step_count() * physics_dt
        )
        if performance is not None:
            print(performance, flush=True)

    if args_cli.replay_out and replay_rows:
        from pathlib import Path

        output = Path(args_cli.replay_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = np.stack(replay_rows)
        times = np.asarray(replay_times)
        n = arm_actuator.num_joints
        np.savez_compressed(
            output,
            q=rows[:, :n],
            v=rows[:, n : 2 * n],
            u=rows[:, 2 * n : 2 * n + 3],
            q_target=rows[:, 2 * n + 3 : 3 * n + 3],
            v_target=rows[:, 3 * n + 3 : 4 * n + 3],
            v_reference=rows[:, 4 * n + 3 : 5 * n + 3],
            target_feedback=args_cli.target_feedback,
            solver_velocity_iterations=args_cli.solver_velocity_iterations,
            source=args_cli.replay_npz,
            start=args_cli.replay_start,
            dt=sim_dt,
            model=ARM_MODEL_DIR,
            integration=args_cli.integration,
            tool=TOOL,
            arm_stiffness=args_cli.arm_stiffness,
            arm_damping=args_cli.arm_damping,
            velocity_feedforward=not args_cli.no_target_velocity_feedforward,
            endstop_mode=args_cli.endstop_mode,
            physics_dt=physics_dt,
            physics_substeps=args_cli.physics_substeps,
            sim_time=times[:, 0],
            wall_time=times[:, 1] - realtime.start_wall,
            carriage_rocking=args_cli.carriage_rocking,
            learned_carriage=args_cli.learned_carriage,
            model_joint_names=MODEL_JOINT_NAMES,
            slew_state=np.stack(slew_rows),
            slew_model=SLEW_MODEL_DIR,
            slew_learned=slew_learned,
            slew_clamp_to_limits=False,
            slew_stiffness=[0.0 if slew_direct_integration else CARRIAGE_STIFFNESS],
            slew_damping=[0.0 if slew_direct_integration else CARRIAGE_DAMPING],
            rocking_state=np.stack(rocking_rows),
            rocking_stiffness=rocking_kp,
            rocking_damping=rocking_kd,
            velocity_limit=NN_ARM_VEL_MAX,
            position_limits=arm_actuator._joint_pos_limits[0].cpu().numpy(),
        )
        print(f"[INFO] Saved {len(rows)} replay steps to {output}")
    simulation_app.close()


if __name__ == "__main__":
    main()
