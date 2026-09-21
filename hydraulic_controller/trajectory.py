"""Ordered planar strokes and a bounded geometric velocity reference."""

from __future__ import annotations

import numpy as np
import torch


def quintic(tau):
    """Rest-to-rest quintic time scaling: s(tau) and ds/dtau for tau in [0, 1] (NumPy or Torch)."""
    s = tau**3 * (10 - 15 * tau + 6 * tau**2)
    s_dot = 30 * tau**2 * (1 - tau) ** 2
    return s, s_dot


def feedback_velocity(reference, reference_velocity, position, kp: float, speed_limit: float) -> np.ndarray:
    """Feed-forward velocity plus a proportional position correction [m/s], bounded in magnitude."""
    linear = np.asarray(reference_velocity) + kp * (np.asarray(reference) - np.asarray(position))
    norm = float(np.linalg.norm(linear))
    return linear * min(1.0, speed_limit / norm) if norm > 1e-12 else linear


def simplify_stroke(points: np.ndarray, tolerance: float = 0.0005) -> np.ndarray:
    """Simplify a stroke [m] with a maximum geometric deviation [m]."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Stroke must contain finite XZ points in meters")
    if len(points) < 2:
        return points.copy()
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        a, b = pending.pop()
        if b <= a + 1:
            continue
        delta = points[b] - points[a]
        t = np.clip((points[a + 1 : b] - points[a]) @ delta / max(delta @ delta, 1e-12), 0, 1)
        distance = np.linalg.norm(points[a + 1 : b] - (points[a] + t[:, None] * delta), axis=1)
        index = a + 1 + int(np.argmax(distance))
        if distance.max() > tolerance:
            keep.add(index)
            pending.extend(((a, index), (index, b)))
    return points[sorted(keep)]


def sample_segments(points: np.ndarray, spacing: float = 0.002) -> np.ndarray:
    """Densely sample every segment [m], retaining vertices and stroke order."""
    if len(points) < 2:
        return np.asarray(points).copy()
    result = []
    for a, b in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        result.extend(a + np.arange(count)[:, None] / count * (b - a))
    result.append(points[-1])
    return np.asarray(result)


def smooth_stroke(
    points: np.ndarray, sample_spacing: float = 0.001, smoothing_radius: float = 0.004
) -> np.ndarray:
    """Low-pass a mouse stroke in physical arc length while preserving its endpoints."""
    points = simplify_stroke(points, tolerance=0.0)
    if len(points) < 3:
        return points
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(lengths)]
    total = float(distance[-1])
    if total <= 2 * sample_spacing:
        return simplify_stroke(points)
    count = max(2, int(np.ceil(total / sample_spacing)))
    arc = np.linspace(0.0, total, count + 1)
    sampled = np.column_stack([np.interp(arc, distance, points[:, axis]) for axis in range(2)])
    actual_spacing = total / count
    half_window = max(1, int(round(smoothing_radius / actual_spacing)))
    half_window = min(half_window, max(1, (len(sampled) - 1) // 2))
    ramp = np.arange(1, half_window + 2, dtype=np.float64)
    weights = np.r_[ramp, ramp[-2::-1]]
    weights /= weights.sum()
    padded = np.pad(sampled, ((half_window, half_window), (0, 0)), mode="edge")
    smoothed = np.column_stack([np.convolve(padded[:, axis], weights, mode="valid") for axis in range(2)])
    smoothed[0], smoothed[-1] = points[0], points[-1]
    # Keep the planned curve compact while bounding its deviation to 0.5 mm.
    return simplify_stroke(smoothed)


class StrokeUnreachableError(ValueError):
    """The approach or the stroke leaves the region reachable at the held bucket angle."""

    def __init__(self, message: str, points: np.ndarray):
        super().__init__(message)
        self.points = points
        """Unreachable path samples [m], shape [N, 2]."""


def validate_stroke(
    kin, q: torch.Tensor, points: np.ndarray, box: np.ndarray, angle: float | None = None
) -> np.ndarray:
    """Smooth a stroke and validate it, and the straight approach to it, against fixed-angle IK.

    Args:
        kin: Excavator kinematics.
        q: Current configuration [rad], shape [1, 4].
        points: Raw stroke [m], shape [N, 2].
        box: Drawing rectangle ``[x_min, x_max, z_min, z_max]`` [m].
        angle: Held bucket angle [rad]; defaults to the angle at ``q``.

    Returns:
        The smoothed path [m], shape [M, 2].

    Raises:
        StrokeUnreachableError: If any approach or stroke sample is unreachable.
        ValueError: If the stroke is empty or leaves the drawing box.
    """
    path = smooth_stroke(points)
    if not len(path):
        raise ValueError("Draw at least one point")
    if ((path < box[[0, 2]]) | (path > box[[1, 3]])).any():
        raise ValueError("Drawing extends outside the drawing box")
    pose, _ = kin.pose_jacobian(q[:1])
    approach = sample_segments(np.vstack((pose[0, :2].cpu().numpy(), path[:1])))
    samples = np.vstack((approach, sample_segments(path)))
    targets = pose.repeat(len(samples), 1)
    targets[:, :2] = torch.as_tensor(samples, device=q.device, dtype=q.dtype)
    if angle is not None:
        targets[:, 2] = angle
    reachable = kin.reachable(targets, q).cpu().numpy()
    if not reachable.all():
        stroke_bad = ~reachable[len(approach) :]
        if stroke_bad.any():
            x, z = 1000 * samples[len(approach) :][stroke_bad][0]
            message = f"Out of reach near X {x:.0f} mm, Z {z:.0f} mm; keep the stroke in the lit area"
        else:
            x, z = 1000 * samples[~reachable][0]
            message = f"The straight move to the stroke start is out of reach near X {x:.0f} mm, Z {z:.0f} mm"
        raise StrokeUnreachableError(message, samples[~reachable])
    return path


class StrokeFollower:
    """Advance an open-loop reference along a polyline at a cruise arc-length speed.

    With ``accel`` [m/s^2] the reference speeds up from and slows down to rest; without it the speed is constant.
    """

    def __init__(self, points: np.ndarray, speed: float, accel: float | None = None):
        self.points = simplify_stroke(points)
        if not len(self.points) or not np.isfinite(speed) or speed <= 0:
            raise ValueError("A nonempty path and positive cruise speed are required")
        self.speed = speed
        self.accel = accel
        self.current_speed = 0.0 if accel else speed
        self.lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.distance = np.r_[0.0, np.cumsum(self.lengths)]
        self.total_length = float(self.distance[-1])
        self.progress = 0.0
        self.finished = self.total_length <= 1e-10
        self.reference = self.points[0].copy()
        self.display_target = self.points[0].copy()

    def _sample(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the point and forward tangent at an arc length along the path."""
        segment = min(int(np.searchsorted(self.distance, progress, side="right") - 1), len(self.lengths) - 1)
        segment = max(0, segment)
        length = max(float(self.lengths[segment]), 1e-10)
        along = np.clip((progress - self.distance[segment]) / length, 0.0, 1.0)
        delta = self.points[segment + 1] - self.points[segment]
        return self.points[segment] + along * delta, delta / length

    def command(self, position: np.ndarray, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Return a feed-forward velocity; measured state intentionally does not affect progress."""
        del position, velocity
        if self.finished:
            self.reference = self.points[-1].copy()
            self.display_target = self.points[-1].copy()
            return np.zeros(2)
        remaining = self.total_length - self.progress
        if self.accel:
            braking = np.sqrt(2 * self.accel * remaining)
            # A lowered cruise speed is also approached at ``accel``, not in one step.
            self.current_speed = min(
                max(self.speed, self.current_speed - self.accel * dt),
                self.current_speed + self.accel * dt,
                max(braking, self.accel * dt),
            )
        else:
            self.current_speed = self.speed
        step = min(self.current_speed * dt, remaining)
        self.progress += step
        self.reference, tangent = self._sample(self.progress)
        self.display_target = self.reference.copy()
        self.finished = self.progress >= self.total_length - 1e-10
        return tangent * (step / dt)


class DrawingSession:
    """Run a straight approach and drawn stroke as one reference, closed around the measured tip position.

    As in Egli & Hutter (RA-L 2022, eq. 2), the commanded twist is the reference velocity plus ``kp`` times the
    position and bucket-angle error, so velocity tracking errors do not accumulate into path errors.
    """

    def __init__(
        self,
        initial_pose: np.ndarray,
        speed: float,
        kp: float = 3.0,
        accel: float = 0.25,
        speed_limit: float = 0.12,
        pitch_rate_limit: float = 0.3,
    ):
        self.angle = float(initial_pose[2])
        self.hold = initial_pose[:2].copy()
        self.speed = speed
        self.kp, self.accel = kp, accel
        self.speed_limit, self.pitch_rate_limit = speed_limit, pitch_rate_limit
        self.phase = "hold"
        self.pending = None
        self.follower = None
        self.path = None
        self.approach_length = 0.0
        self.reference = self.hold.copy()
        self.display_target = self.hold.copy()

    def cancel(self, position: np.ndarray):
        """Cancel pending execution and hold the current position [m]."""
        self.phase = "hold"
        self.pending = self.follower = None
        self.hold = position.copy()
        self.reference = self.hold.copy()
        self.display_target = self.hold.copy()

    def set_speed(self, speed: float):
        """Change the cruise speed [m/s], including along the path being executed."""
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("Cruise speed must be positive")
        self.speed = speed
        if self.follower is not None:
            self.follower.speed = speed

    def submit(self, path: np.ndarray, position: np.ndarray):
        """Start a constant-speed straight approach followed immediately by the stroke."""
        self.path = path
        self.pending = None
        self.approach_length = float(np.linalg.norm(path[0] - position))
        self.follower = StrokeFollower(np.vstack((position, path)), self.speed, self.accel)
        self.phase = "approach" if self.approach_length > 1e-6 else "drawing"
        self.reference = position.copy()
        self.display_target = position.copy()

    def command(self, pose: np.ndarray, twist: np.ndarray, dt: float) -> np.ndarray:
        """Compute desired tip twist [m/s, m/s, rad/s] for the current session."""
        if self.phase in ("approach", "drawing"):
            feed = self.follower.command(pose[:2], twist[:2], dt)
            self.reference = self.follower.reference.copy()
            self.display_target = self.follower.display_target.copy()
            if self.phase == "approach" and self.follower.progress >= self.approach_length:
                self.phase = "drawing"
            if self.follower.finished:
                self.hold = self.path[-1].copy()
                self.phase = "hold"
                self.display_target = self.hold.copy()
        elif self.phase == "hold":
            self.reference = self.hold.copy()
            self.display_target = self.hold.copy()
            feed = np.zeros(2)
        else:
            feed = np.zeros(2)
        linear = feedback_velocity(self.reference, feed, pose[:2], self.kp, self.speed_limit)
        error = np.arctan2(np.sin(self.angle - pose[2]), np.cos(self.angle - pose[2]))
        return np.r_[linear, np.clip(self.kp * error, -self.pitch_rate_limit, self.pitch_rate_limit)]
