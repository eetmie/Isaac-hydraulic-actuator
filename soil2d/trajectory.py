"""Explicit bucket-origin poses, suitable for replay by another soil solver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSE_COLUMNS = ("time_s", "bucket_x_m", "bucket_z_m", "bucket_angle_rad")


@dataclass
class Trajectory:
    times: np.ndarray
    poses: np.ndarray
    phases: tuple[str, ...]
    smooth: bool = True

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.poses = np.asarray(self.poses, dtype=np.float64)
        if self.times.ndim != 1 or len(self.times) < 2 or self.poses.shape != (len(self.times), 3):
            raise ValueError("Trajectory needs at least two time/pose samples")
        if not np.isfinite(self.times).all() or not np.isfinite(self.poses).all():
            raise ValueError("Trajectory contains nonfinite values")
        if self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Trajectory times must start at zero and increase strictly")
        if len(self.phases) != len(self.times) - 1:
            raise ValueError("Trajectory phase count must match its intervals")

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def sample(self, time_s: float) -> np.ndarray:
        if not np.isfinite(time_s):
            raise ValueError("Trajectory time must be finite")
        index = int(np.clip(np.searchsorted(self.times, time_s, side="right") - 1, 0, len(self.times) - 2))
        t0, t1 = self.times[index : index + 2]
        u = float(np.clip((time_s - t0) / (t1 - t0), 0, 1))
        if self.smooth:
            u = u**3 * (10 + u * (-15 + 6 * u))
        return self.poses[index] + u * (self.poses[index + 1] - self.poses[index])

    def command(self, time_s: float, frame_dt: float) -> tuple[np.ndarray, np.ndarray]:
        start = self.sample(time_s)
        return start, (self.sample(time_s + frame_dt) - start) / frame_dt

    def phase(self, time_s: float) -> str:
        index = int(np.clip(np.searchsorted(self.times, time_s, side="right") - 1, 0, len(self.phases) - 1))
        return self.phases[index]

    @classmethod
    def from_csv(cls, path: str | Path) -> Trajectory:
        with Path(path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(POSE_COLUMNS).issubset(reader.fieldnames or ()):
                raise ValueError(f"Pose CSV must contain {', '.join(POSE_COLUMNS)}")
            rows = [[float(row[key]) for key in POSE_COLUMNS] for row in reader]
        data = np.asarray(rows, dtype=float)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError("Pose CSV contains no samples")
        return cls(data[:, 0], data[:, 1:], ("replay",) * (len(data) - 1), smooth=False)


def digging_path() -> Trajectory:
    """RC-scale example: settle, approach, cut, curl/lift, transport, dump, settle."""
    return Trajectory(
        np.array([0.0, 1.0, 1.8, 2.6, 3.8, 4.8, 5.8, 6.6, 8.0]),
        np.array(
            [
                (0.22, 0.38, 0.0),
                (0.22, 0.38, 0.0),
                (0.32, 0.25, -0.15),
                (0.40, 0.13, -0.20),
                (0.58, 0.15, 0.55),
                (0.62, 0.43, 1.00),
                (0.28, 0.43, 1.00),
                (0.28, 0.43, 2.80),
                (0.28, 0.43, 2.80),
            ]
        ),
        ("settle", "approach", "penetrate", "cut and curl", "lift", "transport", "dump", "settle"),
    )
