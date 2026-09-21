"""Isaac scene and authoritative-state synchronization shared by train and play."""

from __future__ import annotations

import numpy as np
import torch

from .core import DEFAULT_ASSET, HOME, JOINT_NAMES


def robot_config(asset: str = str(DEFAULT_ASSET)):
    """Create the excavator articulation configuration with direct hydraulic actuation."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.schemas import RigidBodyBaseCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=asset,
            copy_from_source=True,
            rigid_props=RigidBodyBaseCfg(disable_gravity=True),
            articulation_props=PhysxArticulationRootPropertiesCfg(solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.09),
            joint_pos=dict(zip(JOINT_NAMES, HOME)),
            joint_vel={".*": 0.0},
        ),
        actuators={"prescribed": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )


def setup_scene(scene, stage, cfg):
    """Spawn the fixed-base robot and clone isolated environments."""
    from isaaclab.assets import Articulation
    from isaaclab.sim import DomeLightCfg
    from pxr import UsdPhysics

    robot = Articulation(cfg)
    anchor = UsdPhysics.FixedJoint.Define(stage, "/World/envs/env_0/Robot/Joints/world_fixed")
    anchor.CreateBody1Rel().SetTargets(["/World/envs/env_0/Robot/lower_carriage"])
    scene.clone_environments(copy_from_source=False)
    scene.articulations["robot"] = robot
    if scene.device == "cpu":
        scene.filter_collisions(global_prim_paths=[])
    light = DomeLightCfg(intensity=2500.0, color=(0.8, 0.85, 1.0))
    light.func("/World/Light", light)
    return robot


class HydraulicSceneBridge:
    """Keep PhysX synchronized to the learned state after each physics step."""

    def __init__(self, sim, robot, plant):
        from isaaclab_physx.physics.physx_manager import IsaacEvents, PhysxManager

        self.sim, self.robot, self.plant = sim, robot, plant
        self.ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
        if names != JOINT_NAMES:
            raise ValueError("USD/model joint channel order mismatch")
        self.position = robot.data.default_joint_pos.torch.clone()
        self.velocity = torch.zeros_like(self.position)
        self.post_steps = 0
        robot.write_joint_stiffness_to_sim_index(stiffness=0.0)
        robot.write_joint_damping_to_sim_index(damping=0.0)
        self.handle = PhysxManager.register_callback(
            self._post_step,
            IsaacEvents.POST_PHYSICS_STEP,
            name="hydraulic_authoritative_state",
        )
        self.sync()

    def sync(self) -> None:
        """Write authoritative q [rad] and velocity [rad/s], holding unused DOFs fixed."""
        self.position[:, self.ids] = self.plant.q
        self.velocity[:, self.ids] = self.plant.v
        self.robot.write_joint_position_to_sim_index(position=self.position)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.velocity)

    def _post_step(self, dt):
        self.sync()
        self.post_steps += 1

    def close(self) -> None:
        """Remove only this bridge's callback before closing the simulator."""
        self.handle.deregister()


class ViewportSketch:
    """Mirror the drawing window in the Kit viewport with persistent debug-draw lines.

    Points are tip X/Z [m] in the lower-carriage frame, drawn in the arm plane at the fixed carriage origin.
    Colors match the drawing window. Every call is a no-op when the app has no debug drawing (e.g. headless).
    """

    BOX_COLOR = (0.30, 0.39, 0.50, 1.0)
    REQUESTED_COLOR = (0.38, 0.66, 1.0, 1.0)
    APPROACH_COLOR = (1.0, 0.73, 0.41, 1.0)
    EXECUTED_COLOR = (0.34, 0.86, 0.66, 1.0)

    def __init__(self, origin: torch.Tensor, box: np.ndarray):
        """Acquire debug drawing and outline the drawing box.

        Args:
            origin: World position of the lower carriage [m], shape [3].
            box: Drawing rectangle ``[x_min, x_max, z_min, z_max]`` [m].
        """
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            self._draw = None
        else:
            self._draw = _debug_draw.acquire_debug_draw_interface()
        self.origin = np.asarray(origin.tolist(), dtype=np.float64)
        x0, x1, z0, z1 = box
        self._outline = [(x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0)]
        self._last = None
        self.clear()

    def _polyline(self, points, color, width: float) -> None:
        if self._draw is None or len(points) < 2:
            return
        world = [tuple(self.origin + (x, 0.0, z)) for x, z in points]
        count = len(world) - 1
        self._draw.draw_lines(world[:-1], world[1:], [color] * count, [width] * count)

    def clear(self) -> None:
        """Remove the path and the executed trace, keeping the drawing-box outline."""
        self._last = None
        if self._draw is not None:
            self._draw.clear_lines()
            self._polyline(self._outline, self.BOX_COLOR, 1.0)

    def show_path(self, path: np.ndarray) -> None:
        """Replace everything with a new requested path [m], shape [N, 2]."""
        self.clear()
        self._polyline(np.asarray(path).tolist(), self.REQUESTED_COLOR, 2.0)

    def trace(self, position: np.ndarray, phase: str) -> None:
        """Extend the executed trace to the measured tip position [m] while approaching or drawing."""
        position = tuple(float(value) for value in position)
        if self._last is not None and phase == self._last[1] and phase in ("approach", "drawing"):
            color = self.APPROACH_COLOR if phase == "approach" else self.EXECUTED_COLOR
            self._polyline([self._last[0], position], color, 4.0)
        self._last = (position, phase)
