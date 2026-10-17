"""SI units throughout: x/z in metres, angles in radians, sectional width in metres."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Material:
    density: float = 1500.0
    young_modulus: float = 20_000.0
    poisson_ratio: float = 0.2
    friction_angle_deg: float = 32.0
    cohesion_pa: float = 0.0
    tool_friction: float = 0.4
    floor_friction: float = 0.5

    @property
    def lame(self) -> tuple[float, float]:
        e, nu = self.young_modulus, self.poisson_ratio
        return e / (2 * (1 + nu)), e * nu / ((1 + nu) * (1 - 2 * nu))

    @property
    def wave_speed(self) -> float:
        mu, lam = self.lame
        return math.sqrt((lam + 2 * mu) / self.density)

    def validate(self) -> None:
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Material parameters must be finite")
        if self.density <= 0 or self.young_modulus <= 0:
            raise ValueError("Density and Young's modulus must be positive")
        if not 0 <= self.poisson_ratio < 0.45:
            raise ValueError("Poisson ratio must be in [0, 0.45)")
        if not 0 <= self.friction_angle_deg <= 60:
            raise ValueError("Friction angle must be in [0, 60] degrees")
        if min(self.cohesion_pa, self.tool_friction, self.floor_friction) < 0:
            raise ValueError("Cohesion and contact friction must be nonnegative")


@dataclass(frozen=True)
class Bucket:
    # Back, curved bottom and cutting edge; the mouth is deliberately open.
    vertices: tuple[tuple[float, float], ...] = (
        (-0.080, 0.055),
        (-0.075, -0.055),
        (0.025, -0.065),
        (0.080, -0.015),
    )
    wall_thickness: float = 0.012
    contact_padding_cells: float = 0.5

    def validate(self) -> None:
        if len(self.vertices) < 2:
            raise ValueError("Bucket needs at least two vertices")
        if not all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in self.vertices):
            raise ValueError("Bucket vertices must be finite x/z pairs")
        if any(math.dist(a, b) < 1e-8 for a, b in zip(self.vertices, self.vertices[1:])):
            raise ValueError("Bucket contains a zero-length wall")
        if not math.isfinite(self.wall_thickness) or self.wall_thickness <= 0:
            raise ValueError("Bucket wall thickness must be positive")
        if not math.isfinite(self.contact_padding_cells) or self.contact_padding_cells < 0:
            raise ValueError("Contact padding must be nonnegative")


@dataclass(frozen=True)
class Scene:
    width: float = 1.0
    height: float = 0.6
    dx: float = 0.01
    dt: float = 0.0005
    frame_dt: float = 0.02
    effective_width: float = 0.11
    gravity: float = 9.81
    soil_min: tuple[float, float] = (0.40, 0.035)
    soil_max: tuple[float, float] = (0.78, 0.225)
    particles_per_axis: int = 2
    seed: int = 7
    material: Material = field(default_factory=Material)
    bucket: Bucket = field(default_factory=Bucket)

    @property
    def cells(self) -> tuple[int, int]:
        return round(self.width / self.dx), round(self.height / self.dx)

    @property
    def substeps(self) -> int:
        return round(self.frame_dt / self.dt)

    @property
    def collision_radius(self) -> float:
        return self.bucket.wall_thickness / 2 + self.bucket.contact_padding_cells * self.dx

    def validate(self) -> None:
        self.material.validate()
        self.bucket.validate()
        for name in ("width", "height", "dx", "dt", "frame_dt", "effective_width"):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.gravity) or self.gravity < 0:
            raise ValueError("Gravity must be finite and nonnegative")
        if isinstance(self.particles_per_axis, bool) or not isinstance(self.particles_per_axis, int):
            raise ValueError("particles_per_axis must be an integer")
        if not 1 <= self.particles_per_axis <= 8:
            raise ValueError("particles_per_axis must be in [1, 8]")
        if min(self.cells) < 12:
            raise ValueError("Domain must have at least 12 cells per axis")
        for length in (self.width, self.height):
            if not math.isclose(length / self.dx, round(length / self.dx), abs_tol=1e-6):
                raise ValueError("Domain dimensions must be integer multiples of dx")
        if self.substeps < 1 or not math.isclose(self.substeps * self.dt, self.frame_dt, abs_tol=1e-9):
            raise ValueError("frame_dt must be an integer multiple of dt")
        if len(self.soil_min) != 2 or len(self.soil_max) != 2:
            raise ValueError("Initial soil bounds must be x/z pairs")
        for low, high, length in zip(self.soil_min, self.soil_max, (self.width, self.height)):
            if not (
                math.isfinite(low) and math.isfinite(high) and 2 * self.dx < low < high < length - 2 * self.dx
            ):
                raise ValueError("Initial soil must lie inside the two-cell boundary margin")
        if self.dt * self.material.wave_speed / self.dx > 0.35:
            raise ValueError("Elastic CFL exceeds 0.35; reduce dt or stiffness, or increase dx")

    def as_dict(self) -> dict:
        return asdict(self)
