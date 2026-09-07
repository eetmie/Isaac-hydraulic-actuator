"""Shared data and model definitions for the hydraulic actuator model.

This module is the single source of truth for everything the training side and
the evaluation side must agree on:

* ``WindowSpec``   -- the columns and history geometry, and the one definition
  of how long a history is (``history_samples``) and how wide a feature vector
  is (``in_dim``). Its column tuples are what set the model's width, so nothing
  in this module assumes three joints.
* ``ModelSpec``    -- the one definition of the network architecture.
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

Cleaning is not here -- bring your own. This module consumes CSVs that already
have the required columns, and splits the timeline wherever they are unusable
rather than dropping rows in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple
import glob
import os

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:  # pandas is imported lazily -- see _pandas()
    import pandas as pd


def _pandas():
    """Import pandas on first use.

    Only the CSV loaders need it. Keeping it out of module scope means the spec,
    feature and model layer -- and therefore test_contract.py's parity tests --
    import cleanly in the Isaac Sim environment, which ships no pandas.
    """
    import pandas as pd

    return pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
"""Repository root: this file lives in ``training/``, artifacts live above it."""


# ----------------------------------------------------------------------------
# Column contract (matches data_collection/drive_logger.py output)
# ----------------------------------------------------------------------------
#
# These are the excavator's column names, and nothing but WindowSpec's defaults.
# A different machine passes its own via --q-cols / --qdot-cols / --u-cols; the
# widths of those lists are what set the model's input and output size.

TIME_COL = "timestamp"
U_COLS = ("combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop")
Q_COLS = ("joint_pos_boom", "joint_pos_arm", "joint_pos_bucket")
QDOT_COLS = ("joint_vel_boom", "joint_vel_arm", "joint_vel_bucket")

TARGET_MODE = "delta_velocity"
"""Not a knob: the delta reconstruction is wired into four rollout recurrences."""

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}
"""Selectable activations. Both are parameterless, so the nn.Sequential index
numbering -- and hence the state_dict keys the sim side loads with strict=True --
is identical whichever is chosen."""


# ----------------------------------------------------------------------------
# Window geometry
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Network architecture, independent of what the inputs mean.

    Kept separate from :class:`WindowSpec` because the two are chosen for
    different reasons: the window is a property of the machine's dynamics, the
    architecture is a property of the fit.
    """

    hidden: Tuple[int, ...] = (128, 128, 128)
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {self.activation!r}; expected one of {sorted(ACTIVATIONS)}"
            )
        if not self.hidden or any(h <= 0 for h in self.hidden):
            raise ValueError(f"hidden must be a non-empty list of positive widths, got {self.hidden}")

    @classmethod
    def from_meta(cls, meta: Dict) -> "ModelSpec":
        return cls(
            hidden=tuple(meta["model"]["hidden"]),
            activation=meta["model"]["activation"],
        )


@dataclass(frozen=True)
class WindowSpec:
    """Which columns feed one training sample, and over what history.

    The column tuples are what set the model's width: ``in_dim`` and ``out_dim``
    are derived from them, so a one-joint slew model and a three-joint arm model
    differ only in what is passed here. Command width is free of joint width, so
    a three-joint arm driven by four valve channels is expressible.

    History defaults are what measured best on the excavator: current position,
    0.10 s of dense velocity history, and 0.99 s of valve-command history sampled
    every 30 ms. Widening either history hurt, and so did sparser command taps.
    """

    q_cols: Tuple[str, ...] = Q_COLS
    qdot_cols: Tuple[str, ...] = QDOT_COLS
    u_cols: Tuple[str, ...] = U_COLS
    time_col: str = TIME_COL
    dt: float = 0.01            # 100 Hz control period
    hist_q: int = 1             # current position only
    hist_qdot: int = 11         # t ... t-0.10 s at 100 Hz
    hist_u: int = 34            # t ... t-0.99 s at 30 ms taps
    qdot_stride: int = 1
    u_stride: int = 3
    include_q: bool = True

    def __post_init__(self) -> None:
        # Tolerate lists from JSON / argparse so callers do not have to remember.
        for name in ("q_cols", "qdot_cols", "u_cols"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.n_q != self.n_qdot:
            raise ValueError(
                f"q_cols and qdot_cols must describe the same joints: got {self.n_q} "
                f"position columns and {self.n_qdot} velocity columns. Position is "
                "integrated from velocity, so the two widths are structurally tied."
            )
        if self.n_q == 0 or self.n_u == 0:
            raise ValueError("q_cols/qdot_cols and u_cols must each name at least one column")

    @property
    def n_q(self) -> int:
        """Number of joint-position channels."""
        return len(self.q_cols)

    @property
    def n_qdot(self) -> int:
        """Number of joint-velocity channels; also the network's output width."""
        return len(self.qdot_cols)

    @property
    def n_u(self) -> int:
        """Number of command channels, free of the joint count."""
        return len(self.u_cols)

    @property
    def out_dim(self) -> int:
        """One predicted velocity delta per joint."""
        return self.n_qdot

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
        return (
            (self.n_q * self.hist_q if self.include_q else 0)
            + self.n_qdot * self.hist_qdot
            + self.n_u * self.hist_u
        )

    @property
    def qdot_offset(self) -> int:
        """Column where the current qdot sits inside a feature vector."""
        return self.n_q * self.hist_q if self.include_q else 0

    @classmethod
    def from_seconds(
        cls,
        velocity_history_sec: float,
        command_history_sec: float,
        command_stride_sec: float,
        dt: float = 0.01,
        position_history_sec: float = 0.0,
        velocity_stride_sec: Optional[float] = None,
        q_cols: Sequence[str] = Q_COLS,
        qdot_cols: Sequence[str] = QDOT_COLS,
        u_cols: Sequence[str] = U_COLS,
        time_col: str = TIME_COL,
    ) -> "WindowSpec":
        """Build a spec from durations in seconds rather than tap counts.

        Args:
            velocity_history_sec: How far back the velocity history reaches [s].
            command_history_sec: How far back the command history reaches [s].
            command_stride_sec: Spacing between command taps [s].
            dt: Control period [s].
            position_history_sec: How far back the position history reaches [s];
                0 means the current position only.
            velocity_stride_sec: Spacing between velocity taps [s]; defaults to ``dt``.
            q_cols: Joint-position column names.
            qdot_cols: Joint-velocity column names.
            u_cols: Command column names.
            time_col: Timestamp column name.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if velocity_history_sec < 0.0 or command_history_sec < 0.0 or position_history_sec < 0.0:
            raise ValueError("History durations must be non-negative")
        if command_stride_sec <= 0.0:
            raise ValueError("Command stride must be positive")
        if velocity_stride_sec is not None and velocity_stride_sec <= 0.0:
            raise ValueError("Velocity stride must be positive")
        u_stride = max(1, int(round(command_stride_sec / dt)))
        qdot_stride = 1 if velocity_stride_sec is None else max(1, int(round(velocity_stride_sec / dt)))
        return cls(
            q_cols=tuple(q_cols),
            qdot_cols=tuple(qdot_cols),
            u_cols=tuple(u_cols),
            time_col=time_col,
            dt=dt,
            hist_q=int(round(position_history_sec / dt)) + 1,
            hist_qdot=int(round(velocity_history_sec / (dt * qdot_stride))) + 1,
            hist_u=int(round(command_history_sec / (dt * u_stride))) + 1,
            qdot_stride=qdot_stride,
            u_stride=u_stride,
        )

    @classmethod
    def from_meta(cls, meta: Dict) -> "WindowSpec":
        """Read a spec back out of a model_meta.json payload."""
        if meta["target_mode"] != TARGET_MODE:
            raise ValueError(
                f"Model target must be {TARGET_MODE!r}, got {meta['target_mode']!r}"
            )
        return cls(
            q_cols=tuple(meta["q_cols"]),
            qdot_cols=tuple(meta["qdot_cols"]),
            u_cols=tuple(meta["u_cols"]),
            time_col=meta["time_col"],
            dt=meta["dt"],
            hist_q=meta["hist_q"],
            hist_qdot=meta["hist_qdot"],
            hist_u=meta["hist_u"],
            qdot_stride=meta["qdot_stride"],
            u_stride=meta["u_stride"],
            include_q=meta["include_q"],
        )

    def to_meta(self, model_spec: Optional[ModelSpec] = None) -> Dict:
        """Everything in model_meta.json except the ``training`` block.

        This is the ONLY place the artifact contract is written. Every key here
        is read with ``[]`` by actuators/hydraulic_actuator.py, so a rename is an
        instant KeyError on the sim side -- test_contract.py pins the key list.

        Args:
            model_spec: Architecture to record. Defaults to :class:`ModelSpec`'s
                own defaults.
        """
        model_spec = ModelSpec() if model_spec is None else model_spec
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
            "time_col": self.time_col,
            "u_cols": list(self.u_cols),
            "q_cols": list(self.q_cols),
            "qdot_cols": list(self.qdot_cols),
            "units": {
                "timestamp_raw": "s",
                "joint_position": "rad",
                "joint_velocity": "rad/s",
                "command": "normalized_-1_to_1",
            },
            "model": {
                "hidden": list(model_spec.hidden),
                "activation": model_spec.activation,
                "in_dim": self.in_dim,
                "out_dim": self.out_dim,
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
    """Plain feed-forward MLP with a selectable activation.

    actuators/hydraulic_actuator.py defines an identical network independently.
    The nn.Sequential layout must stay exactly this shape, because it determines
    the state_dict key names (net.0.weight, net.2.weight, ...) that the sim side
    loads with strict=True. Inserting any layer renumbers those keys.

    Every activation in :data:`ACTIVATIONS` is parameterless, so switching one
    for another leaves that numbering -- and therefore checkpoint compatibility
    at the key level -- untouched.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Optional[Sequence[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()
        hidden = ModelSpec().hidden if hidden is None else tuple(hidden)
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(ACTIVATIONS)}"
            )
        make_activation = ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), make_activation()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @classmethod
    def from_meta(cls, meta: Dict) -> "MLP":
        """Build the network a model_meta.json describes."""
        return cls(
            meta["model"]["in_dim"],
            meta["model"]["out_dim"],
            meta["model"]["hidden"],
            meta["model"]["activation"],
        )


# ----------------------------------------------------------------------------
# Loading clean CSVs
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


_SEARCH_ROOTS = (Path(__file__).resolve().parent, PROJECT_ROOT)
"""Where a relative path is looked for after the working directory: ``training/``
first, then the repository root. Both matter because these tools get run from
either place -- ``python train.py`` from inside ``training/``, and
``python training/train.py`` from the root."""


def resolve_csvs(value: str) -> List[Path]:
    """Accept a file, a directory, a glob, or any of those relative to the project.

    Every form -- including globs -- is tried against the working directory
    first, then ``training/``, then the repository root, so ``--csv my_logs`` and
    ``--csv "my_logs/*.csv"`` both work from any of the three.
    """
    for base in (Path(value), *(root / value for root in _SEARCH_ROOTS)):
        if base.is_file():
            return [base]
        if base.is_dir():
            return sorted(base.glob("*.csv"))
    # glob.glob, not Path.glob: the latter raises on an absolute pattern, which is
    # exactly what you get from a launcher script that expands paths for you.
    for pattern in (value, *(str(root / value) for root in _SEARCH_ROOTS)):
        matches = sorted(Path(p) for p in glob.glob(pattern) if os.path.isfile(p))
        if matches:
            return matches
    return []


def _resolve_dir(value: str) -> Path:
    """First existing directory among the search roots, else the literal path.

    Falling back to the literal keeps the caller's own error message pointed at
    what the user actually typed, rather than at a guess.
    """
    for base in (Path(value), *(root / value for root in _SEARCH_ROOTS)):
        if base.is_dir():
            return base
    return Path(value)


def resolve_model_dir(value: str) -> Path:
    """Locate a trained model directory the same way :func:`resolve_csvs` does.

    Model directories live at the repository root (``models/arm``), while these
    tools live one level down, so a bare ``--model models/arm`` has to resolve
    upwards as well as against the working directory.
    """
    return _resolve_dir(value)


def resolve_dataset_dir(value: str) -> Path:
    """Locate a dataset root (e.g. a LeRobot tree) the same way."""
    return _resolve_dir(value)


def resample_to_fixed_dt(df: "pd.DataFrame", spec: WindowSpec) -> "pd.DataFrame":
    """Resample one continuous segment onto a uniform time grid.

    Args:
        df: One continuous segment, with the spec's columns present.
        spec: Supplies the time column, the data columns and the period [s].
    """
    pd = _pandas()
    time_col, dt = spec.time_col, spec.dt
    df = df.sort_values(time_col).drop_duplicates(time_col)
    if len(df) < 2:
        return df.copy()

    t0 = float(df[time_col].iloc[0])
    t1 = float(df[time_col].iloc[-1])
    t_grid = np.arange(t0, t1 + 1e-12, dt)
    out = pd.DataFrame({time_col: t_grid})

    x = df[time_col].to_numpy(np.float64)
    columns = list(spec.q_cols + spec.qdot_cols + spec.u_cols)
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
        if col in spec.u_cols or col == "cmd_stale":
            # Commands are held by the real controller, not ramped between samples.
            idx = np.searchsorted(x[valid], t_grid, side="right") - 1
            out[col] = y[valid][np.clip(idx, 0, valid.sum() - 1)]
        else:
            out[col] = np.interp(t_grid, x[valid], y[valid])
    return out


def split_into_segments(
    df: "pd.DataFrame",
    time_col: str,
    dt: float,
    max_gap_factor: float = 3.0,
) -> List["pd.DataFrame"]:
    """Split one file wherever its recorded timeline is discontinuous."""
    pd = _pandas()
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
    """Load one clean CSV into contiguous chunks ready for windowing.

    Unusable rows (non-finite values, or a stale command) SPLIT the timeline
    rather than being filtered out of it. Dropping them in place -- as the
    pre-refactor code did -- would let a history window span the resulting hole
    as though it were contiguous.
    """
    pd = _pandas()
    df = pd.read_csv(path)
    required = [spec.time_col, *spec.q_cols, *spec.qdot_cols, *spec.u_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    commands = df[list(spec.u_cols)].apply(pd.to_numeric, errors="coerce").to_numpy()
    if np.isfinite(commands).any() and np.nanmax(np.abs(commands)) > 1.001:
        raise ValueError(f"{path.name} contains valve commands outside [-1, 1]")

    segments = split_into_segments(df, spec.time_col, spec.dt)
    if not segments:
        raise ValueError(f"{path.name} has no continuous timestamped data")

    out: List[Chunk] = []
    for raw in segments:
        seg = resample_to_fixed_dt(raw, spec)
        values = seg[list(spec.q_cols + spec.qdot_cols + spec.u_cols)].to_numpy(np.float64)
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
                    q=part[list(spec.q_cols)].to_numpy(np.float32),
                    qdot=part[list(spec.qdot_cols)].to_numpy(np.float32),
                    u=part[list(spec.u_cols)].to_numpy(np.float32),
                    t_end=float(part[spec.time_col].iloc[-1]),
                )
            )
    return out


# ----------------------------------------------------------------------------
# Pool assembly
# ----------------------------------------------------------------------------

@dataclass
class Pool:
    """All windows from a set of CSVs, concatenated in file/chunk order.

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


def build_pool(
    paths: List[Path],
    spec: WindowSpec,
    label: str = "",
    verbose: bool = True,
    loader: Optional[Callable[[Path, WindowSpec], List[Chunk]]] = None,
) -> Pool:
    """Window every chunk of every source into one flat array.

    ``counts`` and ``names`` are the interface to splits.py: they locate every
    chunk in the flat arrays and identify its source recording.

    Args:
        paths: CSV files, or dataset roots when ``loader`` reads a directory.
        spec: Channel and window geometry.
        label: Name used in progress output.
        verbose: Whether to print per-source counts.
        loader: How to turn one path into chunks. Defaults to :func:`load_chunks`
            (CSV); pass ``lerobot_source.load_lerobot_chunks`` for a LeRobot
            dataset root. Everything after this call is source-agnostic.
    """
    loader = load_chunks if loader is None else loader
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    qs: List[np.ndarray] = []
    qdots: List[np.ndarray] = []
    us: List[np.ndarray] = []
    counts: List[int] = []
    names: List[str] = []
    t_ends: List[float] = []

    for path in paths:
        chunks = loader(path, spec)
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
    if pool.X.shape[1] != spec.in_dim or pool.y.shape[1] != spec.out_dim:
        raise RuntimeError(
            f"Unexpected dimensions: X={pool.X.shape[1]}, y={pool.y.shape[1]}, "
            f"expected X={spec.in_dim}, y={spec.out_dim}"
        )
    if not (np.isfinite(pool.X).all() and np.isfinite(pool.y).all()):
        raise RuntimeError("Non-finite values remain after preprocessing")
    if verbose:
        print(f"Loaded {label}: files={len(paths)}, chunks={len(counts)}, windows={len(pool)}")
    return pool
