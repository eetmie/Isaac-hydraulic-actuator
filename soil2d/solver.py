"""Batched forward soil solver with CPU execution and optional CUDA graph replay."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from . import kernels as k
from .config import Scene


class Soil:
    def __init__(
        self, scene: Scene | None = None, *, envs: int = 1, device: str | None = None, graph: bool = True
    ):
        self.scene = scene or Scene()
        self.scene.validate()
        if isinstance(envs, bool) or not isinstance(envs, int) or envs < 1:
            raise ValueError("envs must be a positive integer")
        wp.init()
        self.device = wp.get_device(device or ("cuda:0" if wp.is_cuda_available() else "cpu"))
        self.envs = envs
        s = self.scene
        spacing = s.dx / s.particles_per_axis
        axes = [np.arange(lo + spacing / 2, hi, spacing) for lo, hi in zip(s.soil_min, s.soil_max)]
        xx, zz = np.meshgrid(*axes, indexing="ij")
        initial = np.stack((xx.ravel(), zz.ravel()), axis=-1)
        if len(initial) == 0:
            raise ValueError("Initial soil contains no particles at this spacing")
        jitter = np.random.default_rng(s.seed).uniform(-0.025, 0.025, initial.shape) * spacing
        self.initial = np.repeat((initial + jitter)[None], envs, axis=0).astype(np.float32)
        self.particles = len(initial)
        self.particle_volume = spacing**2 * s.effective_width
        self.particle_mass = self.particle_volume * s.material.density
        shape = (envs, self.particles)
        nx, nz = s.cells
        grid_shape = (envs, nx + 1, nz + 1)
        self.x = wp.array(self.initial, dtype=wp.vec2, device=self.device)
        self.v = wp.zeros(shape, dtype=wp.vec2, device=self.device)
        self.f = wp.zeros(shape, dtype=wp.mat22, device=self.device)
        self.c = wp.zeros(shape, dtype=wp.mat22, device=self.device)
        self.grid_mass = wp.zeros(grid_shape, dtype=float, device=self.device)
        self.grid_momentum = wp.zeros(grid_shape, dtype=wp.vec2, device=self.device)
        self.grid_velocity = wp.zeros(grid_shape, dtype=wp.vec2, device=self.device)
        self.pose = wp.zeros(envs, dtype=wp.vec3, device=self.device)
        self.twist = wp.zeros(envs, dtype=wp.vec3, device=self.device)
        self.enabled = wp.zeros(envs, dtype=int, device=self.device)
        self.impulse = wp.zeros(envs, dtype=wp.vec3, device=self.device)
        self.boundary_impulse = wp.zeros(envs, dtype=wp.vec3, device=self.device)
        self.penetration = wp.zeros(envs, dtype=float, device=self.device)
        self.faults = wp.zeros(envs, dtype=int, device=self.device)
        walls = np.array(
            [(*a, *b) for a, b in zip(s.bucket.vertices, s.bucket.vertices[1:])], dtype=np.float32
        )
        self.walls = wp.array(walls, dtype=wp.vec4, device=self.device)
        self.params = self._parameters()
        self.graph = None
        self.reset()
        wp.load_module(module=k, device=self.device)
        if graph and self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                self._frame()
            self.graph = capture.graph

    def _parameters(self) -> k.Parameters:
        s, m = self.scene, self.scene.material
        mu, lam = m.lame
        phi = math.radians(m.friction_angle_deg)
        params = k.Parameters()
        params.dx, params.inv_dx, params.dt = s.dx, 1 / s.dx, s.dt
        params.mass, params.volume = self.particle_mass, self.particle_volume
        params.mu, params.lam = mu, lam
        params.alpha = math.sqrt(2 / 3) * 2 * math.sin(phi) / (2 - math.sin(phi))
        params.cohesion = (4 * m.cohesion_pa * math.cos(phi) / (math.sqrt(3) * (2 - math.sin(phi)))) / (
            2 * mu
        )
        params.gravity = s.gravity
        params.floor_friction, params.tool_friction = m.floor_friction, m.tool_friction
        params.xmax, params.zmax = s.width, s.height
        params.radius = s.collision_radius
        params.tool_bound = max(math.hypot(*p) for p in s.bucket.vertices) + s.collision_radius + s.dx
        params.nx, params.nz = s.cells
        return params

    def reset(self) -> None:
        self.x.assign(self.initial)
        for array in (
            self.v,
            self.c,
            self.impulse,
            self.boundary_impulse,
            self.penetration,
            self.faults,
            self.twist,
            self.enabled,
            self.grid_mass,
            self.grid_momentum,
            self.grid_velocity,
        ):
            array.zero_()
        identity = np.broadcast_to(np.eye(2, dtype=np.float32), (self.envs, self.particles, 2, 2)).copy()
        self.f.assign(identity)
        self.pose.assign(np.tile(np.array((0.2, 0.4, 0.0), dtype=np.float32), (self.envs, 1)))
        self.time = 0.0

    def _frame(self) -> None:
        self.impulse.zero_()
        self.boundary_impulse.zero_()
        self.penetration.zero_()
        for _ in range(self.scene.substeps):
            wp.launch(
                k.clear_grid,
                self.grid_mass.shape,
                [self.grid_mass, self.grid_momentum, self.grid_velocity, self.pose, self.twist, self.params],
                device=self.device,
            )
            wp.launch(
                k.p2g,
                self.x.shape,
                [
                    self.x,
                    self.v,
                    self.f,
                    self.c,
                    self.grid_mass,
                    self.grid_momentum,
                    self.faults,
                    self.params,
                ],
                device=self.device,
            )
            wp.launch(
                k.grid_update,
                self.grid_mass.shape,
                [
                    self.grid_mass,
                    self.grid_momentum,
                    self.grid_velocity,
                    self.pose,
                    self.twist,
                    self.enabled,
                    self.walls,
                    self.impulse,
                    self.boundary_impulse,
                    self.params,
                ],
                device=self.device,
            )
            wp.launch(
                k.g2p,
                self.x.shape,
                [
                    self.x,
                    self.v,
                    self.c,
                    self.grid_velocity,
                    self.pose,
                    self.twist,
                    self.enabled,
                    self.walls,
                    self.impulse,
                    self.boundary_impulse,
                    self.penetration,
                    self.faults,
                    self.params,
                ],
                device=self.device,
            )

    def _commands(self, value, *, name: str) -> np.ndarray:
        data = np.asarray(value, dtype=np.float32)
        if data.shape == (3,):
            data = np.broadcast_to(data, (self.envs, 3)).copy()
        if data.shape != (self.envs, 3) or not np.isfinite(data).all():
            raise ValueError(f"{name} must be a finite (3,) or (envs, 3) array")
        return np.ascontiguousarray(data)

    def step(self, pose=None, twist=(0.0, 0.0, 0.0), *, enabled: bool = True) -> None:
        """Advance frame_dt, following the supplied start pose and constant twist.

        Commands may be shared or different per environment. Device arrays remain
        available for a future RL adapter; host transfer is explicit in snapshot().
        """
        velocity = self._commands(twist, name="twist")
        start = self._commands(pose, name="pose") if pose is not None else None
        extent = max(math.hypot(*p) for p in self.scene.bucket.vertices)
        surface_speed = np.linalg.norm(velocity[:, :2], axis=-1) + extent * np.abs(velocity[:, 2])
        if np.max(surface_speed) * self.scene.dt / self.scene.dx > 0.25:
            raise ValueError("Bucket moves more than a quarter cell per substep; reduce dt or speed")
        if start is not None:
            self.pose.assign(start)
        self.twist.assign(velocity)
        self.enabled.assign(np.full(self.envs, int(enabled), dtype=np.int32))
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self._frame()
        self.time += self.scene.frame_dt

    def wrench(self) -> np.ndarray:
        """Mean force on bucket [Fx N, Fz N, torque CCW N m] over the last frame."""
        return self.impulse.numpy() / self.scene.frame_dt

    def snapshot(self) -> dict[str, np.ndarray | float]:
        return {
            "time_s": self.time,
            "position": self.x.numpy(),
            "velocity": self.v.numpy(),
            "deformation": self.f.numpy(),
            "bucket_pose": self.pose.numpy(),
            "wrench": self.wrench(),
            "boundary_impulse": self.boundary_impulse.numpy(),
            "max_projection_m": self.penetration.numpy(),
        }

    def check(self) -> dict:
        """Fail on nonfinite state, lost mass, inverted strain, or unstable CFL."""
        faults = self.faults.numpy()
        if faults.any():
            raise FloatingPointError(
                f"Invalid deformation, escaped stencil or nonfinite state: {faults.tolist()}; reduce dt"
            )
        x, v, f = self.x.numpy(), self.v.numpy(), self.f.numpy()
        if not all(np.isfinite(a).all() for a in (x, v, f)):
            raise FloatingPointError("Nonfinite soil state")
        determinants = np.linalg.det(f)
        if np.any(determinants <= 0):
            raise FloatingPointError("Inverted elastic deformation")
        mass = self.grid_mass.numpy().sum(axis=(1, 2), dtype=np.float64)
        expected = self.particles * self.particle_mass
        if self.time > 0 and not np.allclose(mass, expected, rtol=2e-5):
            raise FloatingPointError("Particle/grid transfer lost mass")
        speed = np.linalg.norm(v, axis=-1).max()
        cfl = (float(speed) + self.scene.material.wave_speed) * self.scene.dt / self.scene.dx
        if cfl > 0.5:
            raise FloatingPointError(f"Dynamic CFL {cfl:.3f} exceeds 0.5; reduce dt")
        return {
            "valid": True,
            "mass_kg_per_env": expected,
            "grid_mass_kg": mass.tolist(),
            "max_speed_m_s": float(speed),
            "cfl": cfl,
            "min_elastic_determinant": float(determinants.min()),
            "max_projection_m": float(self.penetration.numpy().max()),
        }
