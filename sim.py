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

Two joint groups, one learned model each: the arm ([lift, tilt, tool], shipped in
models/arm) and the slew ([carriage], models/slew). The slew model does not exist
yet, so this script falls back to driving that joint straight from the stick --
the same behaviour it has always had. That decision is made HERE, in the demo:
HydraulicActuatorNet has no degraded mode, and nothing under actuators/ knows
that a model might be missing.

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
import os
import sys
import weakref

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_common import (  # noqa: E402
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
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
    configure_joint_drive,
    has_gripper_joints,
    robot_usd_path,
    sanitize_velocity_prediction,
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Excavator NN sim with gamepad teleop")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--tool", choices=list(TOOL_VARIANTS), default=DEFAULT_TOOL,
    help="End effector variant to spawn (selects assets/excavator_<tool>.usd)",
)
parser.add_argument(
    "--model", default=DEFAULT_ARM_MODEL_DIR,
    help="Directory containing the trained arm actuator model artifact",
)
parser.add_argument(
    "--slew-model", default=DEFAULT_SLEW_MODEL_DIR,
    help="Directory containing the trained slew actuator model artifact. When it "
         "does not exist, the slew joint falls back to the raw stick command.",
)
parser.add_argument("--weights", default=None, help="Optional arm checkpoint state dict")
parser.add_argument(
    "--integration", choices=["direct", "target"], default="direct",
    help="direct: learned model integrates joint state and writes it to the sim "
          "(Egli & Hutter convention -- the model IS the dynamics). "
          "target: integrate into a position target and let the articulation PD drive track it.",
)
parser.add_argument(
    "--vel-limit", type=float, default=NN_ARM_VEL_LIMIT_DEFAULT,
    help="Hard safety clamp on NN-predicted arm velocity [rad/s]",
)
parser.add_argument(
    "--vel-warn", type=float, default=NN_ARM_VEL_WARN_DEFAULT,
    help="Warn, but do not clamp, above this NN-predicted arm speed [rad/s]",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.vel_limit <= 0.0:
    parser.error("--vel-limit must be positive")
if args_cli.vel_warn < 0.0:
    parser.error("--vel-warn must be non-negative")
args_cli.num_envs = 1          # teleop always single env

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

ARM_MODEL_DIR    = args_cli.model
SLEW_MODEL_DIR   = args_cli.slew_model
TOOL             = args_cli.tool
ROBOT_USD        = robot_usd_path(TOOL)
CONTROL_DECIMATION = 1       # physics and NN run at the same rate
DEAD_ZONE        = 0.30
CARRIAGE_VEL_MAX = 0.8   # rad/s -- manual/fallback slew velocity scale
ARM_VEL_MAX      = 0.5   # rad/s -- manual mode arm velocity scale
NN_ARM_VEL_MAX   = args_cli.vel_limit
NN_ARM_VEL_WARN  = args_cli.vel_warn


# -- scene ------------------------------------------------------------------

def _robot_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Drive groups for the selected tool variant.

    The bucket asset welds its bucket on with a fixed joint, so only the gripper
    variant has the three extra claw/rotate DOFs to actuate.
    """
    actuators: dict[str, ImplicitActuatorCfg] = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            stiffness=ARM_STIFFNESS,
            damping=ARM_DAMPING,
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
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # note: Isaac Lab 3.0 quaternions are (x, y, z, w); `rot` is left at identity here.
            pos=(0.0, 0.0, 0.09),
            joint_pos={
                "revolute_carriage":  0.0,
                "revolute_lift":     -0.5498,  # -31.5 deg
                "revolute_tilt":      1.2549,  #  71.9 deg
                "revolute_tool":     -0.7540,  # -43.2 deg
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
# One learned model per joint group. The arm's ships with the repository; the
# slew's does not exist yet. Which groups exist, what their joints are called and
# what to do when a model is missing are all decisions of this script -- the
# actuators below it are told, never asked.

def make_actuator(scene, joint_names: list[str], sim_dt: float, *, direct: bool):
    """Build the integrating actuator for one joint group.

    Args:
        scene: Scene holding the articulation.
        joint_names: Joints of the group, in model-channel order.
        sim_dt: Integration period [s].
        direct: Route selector. True writes integrated joint state into the sim
            (the model is the dynamics); False integrates into a position target
            for the articulation's PD drive to chase.

    Returns:
        A :class:`DirectIntegrationActuator` or :class:`VelocityIntegratedActuator`.
    """
    if direct:
        return DirectIntegrationActuator(
            DirectIntegrationActuatorCfg(joint_names_expr=list(joint_names)),
            scene=scene,
            sim_dt=sim_dt,
        )
    return VelocityIntegratedActuator(
        VelocityIntegratedActuatorCfg(joint_names_expr=list(joint_names)),
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
        device=args_cli.device,
        sim_dt=sim_dt * CONTROL_DECIMATION,
        weights=weights,
    )
    controller.reset()
    print(f"[INFO] {label.capitalize()} model: {model_dir}")
    print(f"[INFO]   dt={controller.dt}s  hist_qdot={controller.hist_qdot}  "
          f"hist_u={controller.hist_u}  target={controller.target_mode}")
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
        tilt = -self._axis(self._axes[0], self._axes[1])      # left  Y
        carriage = -self._axis(self._axes[3], self._axes[2])  # left  X
        lift = self._axis(self._axes[4], self._axes[5])       # right Y
        tool = -self._axis(self._axes[6], self._axes[7])      # right X
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

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / SIM_HZ,
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

    sim_dt = sim.get_physics_dt()
    direct_integration = args_cli.integration == "direct"

    # Arm joints: the learned model drives velocity. Two conventions --
    #   direct : integrate to joint state and write it (model is the dynamics, paper-style)
    #   target : integrate to a position target and let the PD drive chase it
    arm_actuator = make_actuator(
        scene, ARM_JOINT_NAMES, sim_dt, direct=direct_integration
    )
    configure_joint_drive(
        robot, arm_actuator.joint_ids,
        direct=direct_integration, stiffness=ARM_STIFFNESS, damping=ARM_DAMPING,
    )
    print(f"[INFO] Arm integration mode: {args_cli.integration}")
    print(f"[INFO] Robot gravity disabled: {DISABLE_ROBOT_GRAVITY}")
    print(f"[INFO] NN velocity warning/clamp: {NN_ARM_VEL_WARN:g}/{NN_ARM_VEL_MAX:g} rad/s")
    print("[INFO] Arm joints resolved:", list(zip(arm_actuator.joint_ids, arm_actuator.joint_names)))

    arm_controller = load_controller(ARM_MODEL_DIR, sim_dt, "arm", weights=args_cli.weights)
    if arm_controller is None:
        raise FileNotFoundError(
            f"No arm model at {ARM_MODEL_DIR}. The arm model ships with the repository; "
            "pass --model if it lives elsewhere."
        )
    # Three stick axes map onto the arm model: lift, tilt, tool.
    require_channels(arm_controller, "arm", ARM_JOINT_NAMES, num_commands=3)

    # Slew: the same class, a different model. Until one is trained the stick
    # command drives the joint directly, exactly as it always has -- that decision
    # is made here and nowhere lower, so nothing under actuators/ has an
    # "unless there is no model" branch in it.
    slew_controller = load_controller(SLEW_MODEL_DIR, sim_dt, "slew")
    slew_learned = slew_controller is not None
    if slew_learned:
        # One stick axis maps onto the slew model.
        require_channels(slew_controller, "slew", SLEW_JOINT_NAMES, num_commands=1)
    # With a model the slew joint takes the arm's integration route; without one it
    # stays on the PD-target route it has always used, whatever --integration says.
    slew_actuator = make_actuator(
        scene, SLEW_JOINT_NAMES, sim_dt, direct=direct_integration and slew_learned
    )
    configure_joint_drive(
        robot, slew_actuator.joint_ids,
        direct=direct_integration and slew_learned,
        stiffness=CARRIAGE_STIFFNESS, damping=CARRIAGE_DAMPING,
    )
    slew_direct_integration = direct_integration and slew_learned
    print("[INFO] Slew joint resolved:", list(zip(slew_actuator.joint_ids, slew_actuator.joint_names)))

    # Gamepad
    teleop = XboxController()

    # Default pose for reset
    default_pos = robot.data.default_joint_pos.torch.clone()
    default_vel = torch.zeros_like(default_pos)

    print("[INFO] Controls:")
    print("  Left  Y  : tilt  Right Y : lift")
    print("  Left  X  : slew        Right X : tool")
    print("  A        : toggle MANUAL / NN command source"
          + ("" if slew_learned else "  (slew always direct: no model)"))
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

    def joint_state(actuator, direct: bool):
        """Feed a model its own integrated state under direct integration.

        Under the target route PhysX owns the state, so read it back from the sim
        instead. Either way this closes the loop the way rollout.py does offline.
        """
        if direct:
            return actuator.position[0].cpu().numpy(), actuator.velocity[0].cpu().numpy()
        ids = actuator.joint_ids
        return (
            robot.data.joint_pos.torch[0, ids].cpu().numpy(),
            robot.data.joint_vel.torch[0, ids].cpu().numpy(),
        )

    step = 0
    while simulation_app.is_running():

        # -- mode toggle ----------------------------------------------------
        if teleop.mode_toggle_requested():
            use_nn = not use_nn
            arm_controller.reset()
            arm_actuator.reset()
            if slew_learned:
                slew_controller.reset()
                slew_actuator.reset()
            print(f"[step {step}] Mode -> {'NN' if use_nn else 'MANUAL'}")

        # -- reset ----------------------------------------------------------
        if teleop.reset_requested():
            robot.write_joint_position_to_sim_index(position=default_pos)
            robot.write_joint_velocity_to_sim_index(velocity=default_vel)
            robot.set_joint_position_target_index(target=default_pos)
            robot.set_joint_velocity_target_index(target=default_vel)
            arm_controller.reset()
            arm_actuator.reset()
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
            # THE joint mapping: stick axes onto the arm model's channel order,
            # which is [lift, tilt, tool]. It lives here, in the demo.
            u = np.array([lift_cmd, tilt_cmd, tool_cmd], dtype=np.float32)

            if use_nn:
                q, qdot = joint_state(arm_actuator, direct_integration)
                qdot_pred = arm_controller.predict_velocity(q, qdot, u)   # rad/s
                qdot_pred, max_abs_vel, finite, clipped = sanitize_velocity_prediction(
                    qdot_pred, NN_ARM_VEL_MAX
                )
                if not finite:
                    print(f"[step {step}] [WARN] Non-finite NN velocity prediction; commanding zero")
                over_warning = max_abs_vel > NN_ARM_VEL_WARN
                if over_warning and not velocity_warning_active:
                    print(f"[step {step}] [WARN] NN velocity reached {max_abs_vel:.3f} rad/s "
                          f"(warning threshold {NN_ARM_VEL_WARN:g}; still allowed)")
                if clipped and not velocity_clamp_active:
                    print(f"[step {step}] [WARN] NN velocity reached {max_abs_vel:.3f} rad/s; "
                          f"clamped to {NN_ARM_VEL_MAX:g}")
                velocity_warning_active = over_warning
                velocity_clamp_active = clipped
                arm_vel_t = torch.from_numpy(qdot_pred).unsqueeze(0).to(device)
            else:
                arm_vel_t = torch.tensor(
                    [[lift_cmd * ARM_VEL_MAX, tilt_cmd * ARM_VEL_MAX, tool_cmd * ARM_VEL_MAX]],
                    device=device,
                )

            if use_nn and slew_learned:
                u_slew = np.array([slew_cmd], dtype=np.float32)
                q_slew, qdot_slew = joint_state(slew_actuator, slew_direct_integration)
                slew_pred = slew_controller.predict_velocity(q_slew, qdot_slew, u_slew)
                slew_pred, _, slew_finite, _ = sanitize_velocity_prediction(
                    slew_pred, NN_ARM_VEL_MAX
                )
                if not slew_finite:
                    print(f"[step {step}] [WARN] Non-finite slew velocity prediction; commanding zero")
                slew_vel_t = torch.from_numpy(slew_pred).unsqueeze(0).to(device)
            else:
                # MANUAL mode, or no slew model yet: the stick is the velocity.
                slew_vel_t = torch.tensor([[slew_cmd * CARRIAGE_VEL_MAX]], device=device)

        # -- apply held commands every physics step -------------------------
        arm_actuator.apply_velocity_command(arm_vel_t)
        slew_actuator.apply_velocity_command(slew_vel_t)

        # -- step -----------------------------------------------------------
        scene.write_data_to_sim()
        sim.step(render=not direct_integration)
        scene.update(sim_dt)
        if direct_integration:
            # PhysX advances prescribed velocity during its step. Restore the learned
            # sample before feedback, logging, and rendering so it is not integrated twice.
            arm_actuator.sync_to_sim()
            if slew_direct_integration:
                slew_actuator.sync_to_sim()
            if sim.is_rendering:
                sim.render()

        if step % 200 == 0:
            p = robot.data.joint_pos.torch[0, arm_actuator.joint_ids].cpu().numpy()
            slew_p = robot.data.joint_pos.torch[0, slew_actuator.joint_ids].cpu().numpy()
            arm_vel_dbg = arm_vel_t.squeeze(0).detach().cpu().numpy()
            slew_vel_dbg = slew_vel_t.squeeze(0).detach().cpu().numpy()
            print(f"[step {step:6d}]  mode={'NN' if use_nn else 'MANUAL'}  u={u}  "
                  f"qdot_cmd={arm_vel_dbg}  "
                  f"lift={p[0]:.3f}  tilt={p[1]:.3f}  tool={p[2]:.3f}  "
                  f"slew={slew_p[0]:.3f} ({slew_vel_dbg[0]:+.3f})  rad")
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
