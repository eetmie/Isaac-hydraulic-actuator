"""
Excavator sim — hydraulic actuator net + gamepad teleop.

USD joint names map to the model's joint order as lift=boom, tilt=arm, tool=bucket.

Gamepad bindings (Xbox layout):
  Left  stick  Y  : tilt   command
  Left  stick  X  : carriage rotate → VelocityIntegratedActuator (always direct)
  Right stick  Y  : lift   command
  Right stick  X  : tool   command
  A              : toggle MANUAL / NN command source
  B              : reset joints to default pose
  dead zone      : 30 %

MANUAL mode — sticks scale directly to arm velocity
NN mode     — sticks are valve commands [-1,1] → HydraulicActuatorNet → predicted qdot

Both command sources use the integration route selected by --integration.
"""

from __future__ import annotations

import argparse
import os
import sys
import weakref

sys.path.insert(0, os.path.dirname(__file__))
from sim_common import (  # noqa: E402
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    CARRIAGE_DAMPING,
    CARRIAGE_STIFFNESS,
    DISABLE_ROBOT_GRAVITY,
    GRIPPER_DAMPING,
    GRIPPER_JOINT_NAMES,
    GRIPPER_STIFFNESS,
    NN_ARM_VEL_LIMIT_DEFAULT,
    NN_ARM_VEL_WARN_DEFAULT,
    SIM_HZ,
    SOLVER_VELOCITY_ITERATIONS,
    configure_arm_drive,
    sanitize_velocity_prediction,
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Excavator NN sim with gamepad teleop")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--model",
    default=os.path.join(os.path.dirname(__file__), "model"),
    help="Directory containing the trained actuator model artifact",
)
parser.add_argument("--weights", default=None, help="Optional checkpoint state dict")
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────

import numpy as np
import torch
from pxr import UsdPhysics

import carb
import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from actuators import (  # noqa: E402
    HydraulicActuatorNet,
    VelocityIntegratedActuator,
    DirectIntegrationActuator,
)

# ── constants ──────────────────────────────────────────────────────────────

MODEL_DIR        = args_cli.model
ROBOT_USD        = os.path.join(os.path.dirname(__file__), "assets", "excavator.usd")
CONTROL_DECIMATION = 1       # physics and NN run at the same rate
DEAD_ZONE        = 0.30
CARRIAGE_VEL_MAX = 0.8   # rad/s
ARM_VEL_MAX      = 0.5   # rad/s — direct mode arm velocity scale
NN_ARM_VEL_MAX   = args_cli.vel_limit
NN_ARM_VEL_WARN  = args_cli.vel_warn


# ── scene ──────────────────────────────────────────────────────────────────

@configclass
class ExcavatorSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    #ground = AssetBaseCfg(
    #    prim_path="/World/defaultGroundPlane",
    #    spawn=sim_utils.GroundPlaneCfg(),
    #)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            copy_from_source=True,
            # The learned velocity already includes the real machine's gravity response.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=DISABLE_ROBOT_GRAVITY),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.09),
            joint_pos={
                "revolute_carriage":  0.0,
                "revolute_lift":     -0.5498,  # -31.5 deg
                "revolute_tilt":      1.2549,  #  71.9 deg
                "revolute_tool":     -0.7540,  # -43.2 deg
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=ARM_JOINT_NAMES,
                stiffness=ARM_STIFFNESS,
                damping=ARM_DAMPING,
            ),
            "carriage": ImplicitActuatorCfg(
                joint_names_expr=["revolute_carriage"],
                stiffness=CARRIAGE_STIFFNESS,
                damping=CARRIAGE_DAMPING,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER_JOINT_NAMES,
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
            ),
        },
    )


# ── gamepad ────────────────────────────────────────────────────────────────

def _deadzone(v: float) -> float:
    if abs(v) < DEAD_ZONE:
        return 0.0
    s = 1.0 if v > 0.0 else -1.0
    return s * (abs(v) - DEAD_ZONE) / (1.0 - DEAD_ZONE)


class XboxController:
    """Minimal gamepad reader for direct valve-command teleop."""

    def __init__(self):
        carb.settings.get_settings().set_bool(
            "/persistent/app/omniverse/gamepadCameraControl", False
        )
        self._appwindow  = omni.appwindow.get_default_app_window()
        self._input      = carb.input.acquire_input_interface()
        self._gamepad    = self._appwindow.get_gamepad(0)
        # index: 0=lift_up 1=lift_dn 2=carriage_l 3=carriage_r 4=tilt_up 5=tilt_dn 6=tool_r 7=tool_l
        self._axes = np.zeros(8, dtype=np.float32)
        self._a_pressed  = False
        self._prev_a     = False
        self._b_pressed  = False
        self._prev_b     = False

        self._sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda ev, *a, obj=weakref.proxy(self): obj._on_event(ev),
        )
        name = self._input.get_gamepad_name(self._gamepad)
        print(f"[Gamepad] {'Connected: ' + name if name else 'No gamepad detected'}")

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_sub"):
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._sub)

    def _on_event(self, ev):
        GI = carb.input.GamepadInput
        v  = ev.value
        m  = {
            GI.LEFT_STICK_UP:    (0, v),
            GI.LEFT_STICK_DOWN:  (1, v),
            GI.LEFT_STICK_LEFT:  (2, v),
            GI.LEFT_STICK_RIGHT: (3, v),
            GI.RIGHT_STICK_UP:   (4, v),
            GI.RIGHT_STICK_DOWN: (5, v),
            GI.RIGHT_STICK_RIGHT:(6, v),
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
        tilt     = -self._axis(self._axes[0], self._axes[1])  # left  Y
        carriage = -self._axis(self._axes[3], self._axes[2])  # left  X
        lift     =  self._axis(self._axes[4], self._axes[5])  # right Y
        tool     = -self._axis(self._axes[6], self._axes[7])  # right X
        return lift, carriage, tilt, tool

    def mode_toggle_requested(self) -> bool:
        cur    = self._a_pressed
        rising = cur and not self._prev_a
        self._prev_a = cur
        return rising

    def reset_requested(self) -> bool:
        cur    = self._b_pressed
        rising = cur and not self._prev_b
        self._prev_b = cur
        return rising


# ── main ───────────────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / SIM_HZ,
        render_interval=1,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = ExcavatorSceneCfg(num_envs=1, env_spacing=4.0)
    scene     = InteractiveScene(scene_cfg)

    # Fix lower_carriage to world programmatically (avoids USD edit)
    stage = sim.stage
    joint_prim = "/World/envs/env_0/Robot/Joints/world_fixed"
    lc_prim    = "/World/envs/env_0/Robot/lower_carriage"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_prim)
    fixed_joint.CreateBody1Rel().SetTargets([lc_prim])
    print(f"[INFO] Fixed joint created: {joint_prim} -> {lc_prim}")

    sim.reset()

    robot: Articulation = scene["robot"]
    print("[INFO] Joints:", robot.joint_names)
    print("[INFO] Fixed base:", robot.is_fixed_base)

    sim_dt = sim.get_physics_dt()

    # Arm joints: the learned model drives velocity. Two conventions --
    #   direct : integrate to joint state and write it (model is the dynamics, paper-style)
    #   target : integrate to a position target and let the PD drive chase it
    ArmActuator = (DirectIntegrationActuator if args_cli.integration == "direct"
                   else VelocityIntegratedActuator)
    arm_actuator = ArmActuator(
        scene=scene,
        joint_names=ARM_JOINT_NAMES,
        sim_dt=sim_dt,
        clamp_to_limits=True,
    )
    direct_integration = args_cli.integration == "direct"
    configure_arm_drive(robot, arm_actuator.joint_ids, direct=direct_integration)
    print(f"[INFO] Arm integration mode: {args_cli.integration}")
    print(f"[INFO] Robot gravity disabled: {DISABLE_ROBOT_GRAVITY}")
    print(f"[INFO] NN velocity warning/clamp: {NN_ARM_VEL_WARN:g}/{NN_ARM_VEL_MAX:g} rad/s")
    print("[INFO] Arm joints resolved:", list(zip(arm_actuator.joint_ids, ARM_JOINT_NAMES)))

    # VelocityIntegratedActuator for carriage (gamepad drives velocity directly)
    carriage_actuator = VelocityIntegratedActuator(
        scene=scene,
        joint_names=["revolute_carriage"],
        sim_dt=sim_dt,
        clamp_to_limits=True,
    )
    print("[INFO] Carriage joint resolved:", list(zip(carriage_actuator.joint_ids, carriage_actuator.joint_names)))

    # Hydraulic NN controller (predicts arm joint velocity from valve commands)
    controller = HydraulicActuatorNet(
        MODEL_DIR,
        device=args_cli.device,
        sim_dt=sim_dt * CONTROL_DECIMATION,
        weights=args_cli.weights,
    )
    controller.reset()
    print(f"[INFO] Controller ready — dt={controller.dt}s  "
          f"hist_qdot={controller.hist_qdot}  hist_u={controller.hist_u}  "
          f"target={controller.target_mode}")

    # Gamepad
    teleop = XboxController()

    # Default pose for reset
    default_pos = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(default_pos)

    print("[INFO] Controls:")
    print("  Left  Y  : tilt  Right Y : lift")
    print("  Left  X  : carriage    Right X : tool")
    print("  A        : toggle MANUAL / NN command source")
    print("  B        : reset")

    use_nn = True
    print("[INFO] Starting in NN mode")

    # Persistent command tensors — updated at the control rate, applied every physics step
    _dev           = robot.data.joint_pos.device
    arm_vel_t      = torch.zeros(1, len(ARM_JOINT_NAMES), device=_dev)
    carriage_vel_t = torch.zeros(1, 1, device=_dev)
    u              = np.zeros(3, dtype=np.float32)
    velocity_warning_active = False
    velocity_clamp_active = False

    step = 0
    while simulation_app.is_running():

        # ── mode toggle ────────────────────────────────────────────────────
        if teleop.mode_toggle_requested():
            use_nn = not use_nn
            controller.reset()
            arm_actuator.reset()
            print(f"[step {step}] Mode → {'NN' if use_nn else 'MANUAL'}")

        # ── reset ──────────────────────────────────────────────────────────
        if teleop.reset_requested():
            robot.write_joint_state_to_sim(default_pos, default_vel)
            robot.set_joint_position_target(default_pos)
            robot.set_joint_velocity_target(default_vel)
            controller.reset()
            arm_actuator.reset()
            carriage_actuator.reset()
            arm_vel_t[:] = 0.0
            carriage_vel_t[:] = 0.0
            u[:] = 0.0
            scene.write_data_to_sim()
            if direct_integration:
                arm_actuator.sync_to_sim()
            print(f"[step {step}] Reset")
            continue

        # ── update commands at the 100 Hz control rate ────────────────────
        if step % CONTROL_DECIMATION == 0:
            arm_ids = arm_actuator.joint_ids
            if direct_integration:
                q = arm_actuator.position[0].cpu().numpy()
                qdot = arm_actuator.velocity[0].cpu().numpy()
            else:
                q = robot.data.joint_pos[0, arm_ids].cpu().numpy()
                qdot = robot.data.joint_vel[0, arm_ids].cpu().numpy()

            lift_cmd, carriage_vel_norm, tilt_cmd, tool_cmd = teleop.read()
            u = np.array([lift_cmd, tilt_cmd, tool_cmd], dtype=np.float32)

            if use_nn:
                qdot_pred = controller.predict_velocity(q, qdot, u)   # [3] rad/s
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
                arm_vel_t = torch.from_numpy(qdot_pred).unsqueeze(0).to(robot.data.joint_pos.device)
            else:
                arm_vel_t = torch.tensor(
                    [[lift_cmd * ARM_VEL_MAX, tilt_cmd * ARM_VEL_MAX, tool_cmd * ARM_VEL_MAX]],
                    device=robot.data.joint_pos.device,
                )
            carriage_vel_t = torch.tensor(
                [[carriage_vel_norm * CARRIAGE_VEL_MAX]], device=robot.data.joint_pos.device
            )

        # ── apply held commands every physics step ────────────────────────
        arm_actuator.apply_velocity_command(arm_vel_t)
        carriage_actuator.apply_velocity_command(carriage_vel_t)

        # ── step ──────────────────────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step(render=not direct_integration)
        scene.update(sim.get_physics_dt())
        if direct_integration:
            # PhysX advances prescribed velocity during its step. Restore the learned
            # sample before feedback, logging, and rendering so it is not integrated twice.
            arm_actuator.sync_to_sim()
            sim.render()

        if step % 200 == 0:
            p = robot.data.joint_pos[0, arm_actuator.joint_ids].cpu().numpy()
            arm_vel_dbg = arm_vel_t.squeeze(0).detach().cpu().numpy()
            print(f"[step {step:6d}]  mode={'NN' if use_nn else 'MANUAL'}  u={u}  "
                  f"qdot_cmd={arm_vel_dbg}  "
                  f"lift={p[0]:.3f}  tilt={p[1]:.3f}  tool={p[2]:.3f}  rad")
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
