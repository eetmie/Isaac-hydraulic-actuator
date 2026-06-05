"""
MASI excavator sim — hydraulic actuator net + gamepad teleop.

Gamepad bindings (Xbox layout):
  Left  stick  Y  : tilt   command
  Left  stick  X  : carriage rotate → VelocityIntegratedActuator (always direct)
  Right stick  Y  : lift   command
  Right stick  X  : tool   command
  A              : toggle DIRECT / NN mode
  B              : reset joints to default pose
  dead zone      : 20 %

DIRECT mode  — sticks scale to rad/s directly → VelocityIntegratedActuator
NN mode      — sticks are valve commands [-1,1] → HydraulicActuatorNet
               → predicted qdot [rad/s] → VelocityIntegratedActuator
"""

from __future__ import annotations

import argparse
import os
import sys
import weakref

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MASI NN sim with gamepad teleop")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
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

sys.path.insert(0, os.path.dirname(__file__))
from actuators import HydraulicActuatorNet, VelocityIntegratedActuator  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────

MODEL_DIR        = os.path.join(os.path.dirname(__file__), "model")
ROBOT_USD        = os.path.join(os.path.dirname(__file__), "assets", "excavator.usd")
CONTROL_DECIMATION = 1       # physics and NN both at 50 Hz
ARM_JOINT_NAMES  = ["revolute_lift", "revolute_tilt", "revolute_tool"]
DEAD_ZONE        = 0.30
CARRIAGE_VEL_MAX = 0.8   # rad/s
ARM_VEL_MAX      = 0.5   # rad/s — direct mode arm velocity scale
NN_ARM_VEL_MAX   = 0.5   # rad/s — safety clamp for NN-predicted arm velocity


# ── scene ──────────────────────────────────────────────────────────────────

@configclass
class MasiTestSceneCfg(InteractiveSceneCfg):
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_velocity_iteration_count=1,
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
            "main_joints": ImplicitActuatorCfg(
                joint_names_expr=["revolute_carriage", "revolute_lift", "revolute_tilt", "revolute_tool"],
                stiffness=600.0,
                damping=40.0,
            ),
            "tool": ImplicitActuatorCfg(
                joint_names_expr=["revolute_gripper", "revolute_claw_1", "revolute_claw_2"],
                stiffness=600.0,
                damping=40.0,
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
        dt=0.02,
        render_interval=1,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 2.0], [0.0, 0.0, 0.5])

    scene_cfg = MasiTestSceneCfg(num_envs=1, env_spacing=4.0)
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

    # VelocityIntegratedActuator for arm joints (NN drives velocity)
    arm_actuator = VelocityIntegratedActuator(
        scene=scene,
        joint_names=ARM_JOINT_NAMES,
        sim_dt=sim_dt,
        clamp_to_limits=True,
    )
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
    controller = HydraulicActuatorNet(MODEL_DIR, device=args_cli.device)
    controller.reset()
    print(f"[INFO] Controller ready — dt={controller.dt}s  "
          f"hist_qdot={controller.hist_qdot}  hist_u={controller.hist_u}")

    # Gamepad
    teleop = XboxController()

    # Default pose for reset
    default_pos = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(default_pos)

    print("[INFO] Controls:")
    print("  Left  Y  : tilt  Right Y : lift")
    print("  Left  X  : carriage    Right X : tool")
    print("  A        : toggle DIRECT / NN mode")
    print("  B        : reset")

    use_nn = True
    print("[INFO] Starting in NN mode")

    # Persistent command tensors — updated at 50 Hz, applied every physics step
    _dev           = robot.data.joint_pos.device
    arm_vel_t      = torch.zeros(1, len(ARM_JOINT_NAMES), device=_dev)
    carriage_vel_t = torch.zeros(1, 1, device=_dev)
    u              = np.zeros(3, dtype=np.float32)

    step = 0
    while simulation_app.is_running():

        # ── mode toggle ────────────────────────────────────────────────────
        if teleop.mode_toggle_requested():
            use_nn = not use_nn
            controller.reset()
            arm_actuator.reset()
            print(f"[step {step}] Mode → {'NN' if use_nn else 'DIRECT'}")

        # ── reset ──────────────────────────────────────────────────────────
        if teleop.reset_requested():
            robot.write_joint_state_to_sim(default_pos, default_vel)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
            controller.reset()
            arm_actuator.reset()
            carriage_actuator.reset()
            arm_vel_t[:] = 0.0
            carriage_vel_t[:] = 0.0
            u[:] = 0.0
            print(f"[step {step}] Reset")
            continue

        # ── update commands at control rate (50 Hz) ───────────────────────
        if step % CONTROL_DECIMATION == 0:
            joint_pos = robot.data.joint_pos   # [1, n_joints]
            joint_vel = robot.data.joint_vel

            arm_ids = arm_actuator.joint_ids
            q    = joint_pos[0, arm_ids].cpu().numpy()   # rad
            qdot = joint_vel[0, arm_ids].cpu().numpy()   # rad/s

            lift_cmd, carriage_vel_norm, tilt_cmd, tool_cmd = teleop.read()
            u = np.array([lift_cmd, tilt_cmd, tool_cmd], dtype=np.float32)

            if use_nn:
                qdot_pred = controller.predict_velocity(q, qdot, u)   # [3] rad/s
                qdot_pred = np.clip(qdot_pred, -NN_ARM_VEL_MAX, NN_ARM_VEL_MAX)
                arm_vel_t = torch.from_numpy(qdot_pred).unsqueeze(0).to(joint_pos.device)
            else:
                arm_vel_t = torch.tensor(
                    [[lift_cmd * ARM_VEL_MAX, tilt_cmd * ARM_VEL_MAX, tool_cmd * ARM_VEL_MAX]],
                    device=joint_pos.device,
                )
            carriage_vel_t = torch.tensor(
                [[carriage_vel_norm * CARRIAGE_VEL_MAX]], device=joint_pos.device
            )

        # ── apply held commands every physics step ────────────────────────
        arm_actuator.apply_velocity_command(arm_vel_t)
        carriage_actuator.apply_velocity_command(carriage_vel_t)

        # ── step ──────────────────────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        if step % 200 == 0:
            p = robot.data.joint_pos[0, arm_actuator.joint_ids].cpu().numpy()
            arm_vel_dbg = arm_vel_t.squeeze(0).detach().cpu().numpy()
            print(f"[step {step:6d}]  mode={'NN' if use_nn else 'DIRECT'}  u={u}  "
                  f"qdot_cmd={arm_vel_dbg}  "
                  f"lift={p[0]:.3f}  tilt={p[1]:.3f}  tool={p[2]:.3f}  rad")
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
