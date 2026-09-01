"""Shared data and model definitions for the hydraulic actuator model.

This module is the single source of truth for everything the training side and
the evaluation side must agree on:

* ``WindowSpec``   -- the history geometry, and the one definition of how long a
  history is (``history_samples``) and how wide a feature vector is (``in_dim``).
* ``build_features`` -- the one definition of the ``[q | qdot | u]`` layout.
* ``Normalizer``   -- the one definition of normalization and of the four
  ``.npy`` filenames in a model directory.
* ``MLP``          -- the one training-side network definition.

``actuators/hydraulic_actuator.py`` deliberately does NOT import this module.
It is the independent second implementation of the same contract, written
against ring buffers for real-time single-step inference inside Isaac Sim.
Keeping it separate is what gives ``test_contract.py`` something real to check;
if it imported ``build_features`` the parity test would only prove that numpy
equals numpy.

Cleaning is not here -- bring your own. This module consumes Parquet logs or
legacy CSVs with the required columns, and splits the timeline wherever they
are unusable rather than dropping rows in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Column contract (matches kaivuriprokkis/simple_drive.py Parquet output)
# ----------------------------------------------------------------------------

TIME_COL = "timestamp"
U_COLS = ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]
Q_COLS = ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket"]
QDOT_COLS = ["joint_vel_boom", "joint_vel_arm", "joint_vel_bucket"]
DATA_SUFFIXES = {".parquet", ".pq", ".csv"}
LOAD_COLS = [TIME_COL, *Q_COLS, *QDOT_COLS, *U_COLS, "sample_idx", "cmd_stale"]

N_JOINTS = 3

HIDDEN = [128, 128, 128]
ACTIVATION = "relu"
TARGET_MODE = "delta_velocity"


# ----------------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSpec:
    """History geometry of one training sample.

    Defaults are what measured best on this machine: current position, 0.10 s of
    dense velocity history, and 0.99 s of valve-command history sampled every
    30 ms. Widening either history hurt, and so did sparser command taps.
    """

    dt: float = 0.01            # 100 Hz control period
    hist_q: int = 1             # current position only
    hist_qdot: int = 11         # t ... t-0.10 s at 100 Hz
    hist_u: int = 34            # t ... t-0.99 s at 30 ms taps
    qdot_stride: int = 1
    u_stride: int = 3
    include_q: bool = True

    @property
    def history_samples(self) -> int:
        """Rows of past data one sample needs. THE definition -- do not inline."""
        return max(
            self.hist_q,
            (self.hist_qdot - 1) * self.qdot_stride + 1,
            (self.hist_u - 1) * self.u_stride + 1,
        )

    @property
    def warmup(self) -> int:
        """Index of the first row with a complete measured history."""
        return self.history_samples - 1

    @property
    def in_dim(self) -> int:
        n_taps = (self.hist_q if self.include_q else 0) + self.hist_qdot + self.hist_u
        return N_JOINTS * n_taps

    @property
    def qdot_offset(self) -> int:
        """Column where the current qdot sits inside a feature vector."""
        return N_JOINTS * self.hist_q if self.include_q else 0

    @classmethod
    def from_seconds(
        cls,
        velocity_history_sec: float,
        command_history_sec: float,
        command_stride_sec: float,
        dt: float = 0.01,
    ) -> "WindowSpec":
        if velocity_history_sec < 0.0 or command_history_sec < 0.0:
            raise ValueError("History durations must be non-negative")
        if command_stride_sec <= 0.0:
            raise ValueError("Command stride must be positive")
        u_stride = max(1, int(round(command_stride_sec / dt)))
        return cls(
            dt=dt,
            hist_qdot=int(round(velocity_history_sec / dt)) + 1,
            hist_u=int(round(command_history_sec / (dt * u_stride))) + 1,
            u_stride=u_stride,
        )

    @classmethod
    def from_meta(cls, meta: Dict) -> "WindowSpec":
        """Read a spec back out of a model_meta.json payload."""
        if meta["model"]["activation"] != ACTIVATION or meta["target_mode"] != TARGET_MODE:
            raise ValueError("Model must use ReLU and the delta_velocity target")
        return cls(
            dt=meta["dt"],
            hist_q=meta["hist_q"],
            hist_qdot=meta["hist_qdot"],
            hist_u=meta["hist_u"],
            qdot_stride=meta["qdot_stride"],
            u_stride=meta["u_stride"],
            include_q=meta["include_q"],
        )

    def to_meta(self) -> Dict:
        """Everything in model_meta.json except the ``training`` block.

        This is the ONLY place the artifact contract is written. Every key here
        is read with ``[]`` by actuators/hydraulic_actuator.py, so a rename is an
        instant KeyError on the sim side -- test_contract.py pins the key list.
        """
        return {
            "dt": self.dt,
            "target_mode": TARGET_MODE,
            "hist_q": self.hist_q,
            "hist_qdot": self.hist_qdot,
            "hist_u": self.hist_u,
            "qdot_stride": self.qdot_stride,
            "u_stride": self.u_stride,
            "hist_q_sec": float((self.hist_q - 1) * self.dt),
            "hist_qdot_sec": float((self.hist_qdot - 1) * self.dt * self.qdot_stride),
            "hist_u_sec": float((self.hist_u - 1) * self.dt * self.u_stride),
            "include_q": self.include_q,
            "time_col": TIME_COL,
            "u_cols": U_COLS,
            "q_cols": Q_COLS,
            "qdot_cols": QDOT_COLS,
            "units": {
                "timestamp_raw": "s",
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "command": "normalized_-1_to_1",
            },
            "model": {
                "hidden": HIDDEN,
                "activation": ACTIVATION,
                "in_dim": self.in_dim,
                "out_dim": N_JOINTS,
                "target": TARGET_MODE,
            },
        }


# ----------------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------------

def make_history_matrix(arr: np.ndarray, hist: int, stride: int = 1) -> np.ndarray:
    """Stack history taps (t, t-stride, ..., t-(hist-1)*stride) of ``arr`` [T, D].

    Returns [T, D*hist]. The first rows are padded by repeating row 0, which is
    the convention HydraulicActuatorNet mirrors by priming its position buffer
    from the first observation.
    """
    T, D = arr.shape
    out = np.zeros((T, D * hist), dtype=np.float32)
    for k in range(hist):
        src_idx = np.clip(np.arange(T) - k * stride, 0, T - 1)
        out[:, k * D:(k + 1) * D] = arr[src_idx, :]
    return out


def build_features(q: np.ndarray, qdot: np.ndarray, u: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """THE feature layout: [q taps | qdot taps | u taps], newest tap first.

    Takes plain arrays rather than a DataFrame so the feature path stays free of
    pandas and can be shared with the evaluation side unchanged.
    """
    feats: List[np.ndarray] = []
    if spec.include_q:
        feats.append(make_history_matrix(q, spec.hist_q, 1))
    if spec.hist_qdot > 0:
        feats.append(make_history_matrix(qdot, spec.hist_qdot, spec.qdot_stride))
    feats.append(make_history_matrix(u, spec.hist_u, spec.u_stride))
    return np.concatenate(feats, axis=1)


def build_windows(
    q: np.ndarray, qdot: np.ndarray, u: np.ndarray, spec: WindowSpec
) -> Tuple[np.ndarray, np.ndarray]:
    """Supervised pairs for one continuous chunk.

    Returns X [N, in_dim] and y [N, 3] where y = qdot(t+dt) - qdot(t). Rows
    without a complete measured history, and the final row (no next-step target),
    are dropped.
    """
    X_all = build_features(q, qdot, u, spec)
    y_all = np.roll(qdot, shift=-1, axis=0) - qdot
    return X_all[spec.warmup:-1, :], y_all[spec.warmup:-1, :]


# ----------------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------------

@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        # float64 accumulation: summing ~500k float32 values in float32 loses bits.
        return Normalizer(
            mean=x.mean(axis=0, dtype=np.float64),
            std=np.maximum(x.std(axis=0, dtype=np.float64), eps),
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def apply_inplace(self, x: np.ndarray) -> None:
        """Normalize a large float32 array without allocating a second copy."""
        x -= self.mean.astype(x.dtype)
        x /= self.std.astype(x.dtype)

    def save(self, model_dir: Path, prefix: str) -> None:
        np.save(model_dir / f"{prefix}_mean.npy", self.mean)
        np.save(model_dir / f"{prefix}_std.npy", self.std)

    @staticmethod
    def load(model_dir: Path, prefix: str) -> "Normalizer":
        return Normalizer(
            mean=np.load(model_dir / f"{prefix}_mean.npy"),
            std=np.load(model_dir / f"{prefix}_std.npy"),
        )


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    """Plain ReLU MLP.

    actuators/hydraulic_actuator.py defines an identical network independently.
    The nn.Sequential layout must stay exactly this shape, because it determines
    the state_dict key names (net.0.weight, net.2.weight, ...) that the sim side
    loads with strict=True. Inserting any layer renumbers those keys.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[List[int]] = None):
        super().__init__()
        hidden = HIDDEN if hidden is None else hidden
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------
# Loading clean drive logs
# ----------------------------------------------------------------------------

@dataclass
class Chunk:
    """One stretch of genuinely contiguous, finite, non-stale samples."""

    name: str
    q: np.ndarray
    qdot: np.ndarray
    u: np.ndarray
    # Last timestamp, relative to the start of the source recording. splits.py
    # needs it to work out how long each recording ran, and hence which
    # recordings are consecutive segments of one driving session.
    t_end: float = 0.0

    def __len__(self) -> int:
        return len(self.q)


def _is_drive_log(path: Path) -> bool:
    return (
        path.suffix.lower() in DATA_SUFFIXES
        and not path.name.startswith("imu_raw_")
    )


def resolve_data_files(value: str) -> List[Path]:
    """Accept a Parquet/CSV file, directory, or glob from cwd or this repo."""
    here = Path(__file__).resolve().parent
    for base in (Path(value), here / value):
        if base.is_file() and base.suffix.lower() in DATA_SUFFIXES:
            return [base]
        if base.is_dir():
            return sorted(
                path for path in base.iterdir()
                if path.is_file() and _is_drive_log(path)
            )

    for pattern in (value, str(here / value)):
        matches = sorted(
            Path(path) for path in glob.glob(pattern)
            if os.path.isfile(path) and _is_drive_log(Path(path))
        )
        if matches:
            return matches
    return []


def resolve_csvs(value: str) -> List[Path]:
    """Backward-compatible alias for callers using the old CSV-specific name."""
    return resolve_data_files(value)


def read_log(path: Path) -> pd.DataFrame:
    """Read only model inputs plus optional legacy continuity columns."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(path).names)
        columns = [column for column in LOAD_COLS if column in available]
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=lambda column: column in LOAD_COLS)


def resample_to_fixed_dt(df: pd.DataFrame, dt: float, time_col: str) -> pd.DataFrame:
    """Resample one continuous segment onto a uniform time grid."""
    df = df.sort_values(time_col).drop_duplicates(time_col)
    if len(df) < 2:
        return df.copy()

    t0 = float(df[time_col].iloc[0])
    t1 = float(df[time_col].iloc[-1])
    t_grid = np.arange(t0, t1 + 1e-12, dt)
    out = pd.DataFrame({time_col: t_grid})

    x = df[time_col].to_numpy(np.float64)
    columns = Q_COLS + QDOT_COLS + U_COLS
    if "cmd_stale" in df.columns:
        columns.append("cmd_stale")
    for col in columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            first_valid = next((v for v in df[col] if pd.notna(v)), np.nan)
            out[col] = first_valid
            continue
        y = numeric.to_numpy(np.float64)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            out[col] = y[valid][0] if valid.sum() == 1 else np.nan
            continue
        if col in U_COLS or col == "cmd_stale":
            # Commands are held by the real controller, not ramped between samples.
            idx = np.searchsorted(x[valid], t_grid, side="right") - 1
            out[col] = y[valid][np.clip(idx, 0, valid.sum() - 1)]
        else:
            out[col] = np.interp(t_grid, x[valid], y[valid])
    return out


def split_into_segments(
    df: pd.DataFrame,
    time_col: str,
    dt: float,
    max_gap_factor: float = 3.0,
) -> List[pd.DataFrame]:
    """Split one file wherever its recorded timeline is discontinuous."""
    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    if len(df) < 2:
        return []

    t = df[time_col].to_numpy(np.float64)
    gaps = np.diff(t)
    discontinuity = (gaps <= 0.0) | (gaps > max_gap_factor * dt)

    if "sample_idx" in df.columns:
        sample_idx = pd.to_numeric(df["sample_idx"], errors="coerce").to_numpy()
        index_step = np.diff(sample_idx)
        discontinuity |= ~np.isfinite(index_step) | (index_step != 1)

    split_idx = np.where(discontinuity)[0] + 1
    bounds = [0, *split_idx.tolist(), len(df)]
    parts = [df.iloc[start:end] for start, end in zip(bounds[:-1], bounds[1:])]
    return [p for p in parts if len(p) >= 2]


def _runs_of_true(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal [start, stop) ranges where ``mask`` is True."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [len(mask)]))
    return [
        (int(a), int(b))
        for a, b in zip(bounds[:-1], bounds[1:])
        if mask[a]
    ]


def load_chunks(path: Path, spec: WindowSpec) -> List[Chunk]:
    """Load one clean drive log into contiguous chunks ready for windowing.

    Unusable rows (non-finite values, or a stale command) SPLIT the timeline
    rather than being filtered out of it. Dropping them in place -- as the
    pre-refactor code did -- would let a history window span the resulting hole
    as though it were contiguous.
    """
    df = read_log(path)
    required = [TIME_COL] + Q_COLS + QDOT_COLS + U_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    commands = df[U_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isfinite(commands).any() and np.nanmax(np.abs(commands)) > 1.001:
        raise ValueError(f"{path.name} contains valve commands outside [-1, 1]")

    segments = split_into_segments(df, TIME_COL, spec.dt)
    if not segments:
        raise ValueError(f"{path.name} has no continuous timestamped data")

    out: List[Chunk] = []
    for raw in segments:
        seg = resample_to_fixed_dt(raw, spec.dt, TIME_COL)
        values = seg[Q_COLS + QDOT_COLS + U_COLS].to_numpy(np.float64)
        good = np.isfinite(values).all(axis=1)
        if "cmd_stale" in seg.columns:
            stale = pd.to_numeric(seg["cmd_stale"], errors="coerce").fillna(0.0).to_numpy()
            good &= stale <= 0.5

        for a, b in _runs_of_true(good):
            if b - a <= spec.history_samples:
                continue
            part = seg.iloc[a:b]
            out.append(
                Chunk(
                    name=f"{path.stem}#{len(out)}",
                    q=part[Q_COLS].to_numpy(np.float32),
                    qdot=part[QDOT_COLS].to_numpy(np.float32),
                    u=part[U_COLS].to_numpy(np.float32),
                    t_end=float(part[TIME_COL].iloc[-1]),
                )
            )
    return out


# ----------------------------------------------------------------------------
# Pool assembly
# ----------------------------------------------------------------------------

@dataclass
class Pool:
    """All windows from a set of drive logs, concatenated in file/chunk order.

    ``q``/``qdot``/``u`` are the raw physical state at each window's own
    timestamp, kept unnormalized and aligned row-for-row with ``X``. Free-running
    rollouts need them to seed history buffers and to score against ground truth;
    they cost ~2% of the memory ``X`` already uses.
    """

    X: np.ndarray           # [N, in_dim]
    y: np.ndarray           # [N, 3]
    q: np.ndarray           # [N, 3] position at window time
    qdot: np.ndarray        # [N, 3] velocity at window time
    u: np.ndarray           # [N, 3] command at window time
    counts: List[int]       # windows per chunk; sum(counts) == N
    names: List[str]
    t_ends: List[float]     # per chunk, last timestamp within its source recording
    files: List[str]

    def __len__(self) -> int:
        return len(self.X)


def build_pool(paths: List[Path], spec: WindowSpec, label: str = "", verbose: bool = True) -> Pool:
    """Window every chunk of every drive log into one flat array.

    ``counts`` and ``names`` are the interface to splits.py: they locate every
    chunk in the flat arrays and identify its source recording.
    """
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    qs: List[np.ndarray] = []
    qdots: List[np.ndarray] = []
    us: List[np.ndarray] = []
    counts: List[int] = []
    names: List[str] = []
    t_ends: List[float] = []

    for path in paths:
        chunks = load_chunks(path, spec)
        usable = 0
        for chunk in chunks:
            X, y = build_windows(chunk.q, chunk.qdot, chunk.u, spec)
            if len(X) == 0:
                continue
            xs.append(X)
            ys.append(y)
            # Same slice build_windows applies, so these stay row-aligned with X.
            keep = slice(spec.warmup, -1)
            qs.append(chunk.q[keep])
            qdots.append(chunk.qdot[keep])
            us.append(chunk.u[keep])
            counts.append(len(X))
            names.append(chunk.name)
            t_ends.append(chunk.t_end)
            usable += len(X)
        if verbose:
            print(f"[{label}] {path.name}: chunks={len(chunks)}, windows={usable}")

    if not xs:
        raise RuntimeError(f"No usable {label or 'input'} data after preprocessing")

    pool = Pool(
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        q=np.concatenate(qs, axis=0),
        qdot=np.concatenate(qdots, axis=0),
        u=np.concatenate(us, axis=0),
        counts=counts,
        names=names,
        t_ends=t_ends,
        files=[p.name for p in paths],
    )
    if pool.X.shape[1] != spec.in_dim or pool.y.shape[1] != N_JOINTS:
        raise RuntimeError(
            f"Unexpected dimensions: X={pool.X.shape[1]}, y={pool.y.shape[1]}, "
            f"expected X={spec.in_dim}, y={N_JOINTS}"
        )
    if not (np.isfinite(pool.X).all() and np.isfinite(pool.y).all()):
        raise RuntimeError("Non-finite values remain after preprocessing")
    if verbose:
        print(f"Loaded {label}: files={len(paths)}, chunks={len(counts)}, windows={len(pool)}")
    return pool
