"""Read a LeRobot v3.0 dataset into the same ``Chunk`` list the CSV loader returns.

NOT YET TESTED AGAINST A REAL DATASET. Built from the published v3.0 schema and
exercised end to end on a generated tree, with the field names checked against a
real dataset's ``meta/info.json``. No recorded LeRobot data has been through it.

LeRobot stores a whole vector per column -- ``observation.state`` is one
``fixed_size_list<float>[N]``, not N scalar columns -- and names the elements in
``meta/info.json``. This module resolves the model's channel names against those
element names, so ``--qdot-cols lift tilt scoop`` picks the right three slots out
of whatever vector happens to carry them. Nothing here knows what an excavator
is, and nothing downstream knows the data came from parquet.

**Velocity must already be in the dataset.** This reader will not differentiate
position for you. The model's target is ``qdot(t+dt) - qdot(t)``, so a qdot
synthesised by differencing would make the target a second difference of
position -- quantisation noise at any realistic log rate, and silently wrong
rather than loudly missing. Record velocity, or add it to the dataset first.

The rate itself is read from ``fps`` in ``meta/info.json``; nothing here assumes
one. ``--dt`` defaults to ``1/fps``, and a dataset whose rate disagrees with the
spec is refused rather than resampled.

Only ``codebase_version: "v3.0"`` is read. The v2.x trees put one parquet per
episode next to a ``meta/episodes.jsonl``; v3.0 packs many episodes into chunked
files and keeps episode records in parquet. Rather than guess which is which,
this raises and names the version it found.

Layout (v3.0)::

    meta/info.json                       fps, features, path templates
    meta/episodes/chunk-000/file-000.parquet
    meta/tasks.parquet
    data/chunk-000/file-000.parquet      many episodes per file
    videos/<key>/chunk-000/file-000.mp4  ignored here

The ``lerobot`` package is deliberately NOT a dependency: this reads the parquet
directly with pyarrow, so ``training/`` stays installable next to Isaac Sim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from dataset import Chunk, WindowSpec

SUPPORTED_CODEBASE_VERSION = "v3.0"

# Bookkeeping columns every LeRobot data parquet carries. They are not channels,
# so they are excluded when reporting what a dataset offers.
RESERVED_COLUMNS = frozenset({"timestamp", "frame_index", "episode_index", "index", "task_index"})


def _pyarrow():
    """Import pyarrow on first use, with an actionable message when it is absent."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading LeRobot datasets needs pyarrow. Install it with 'pip install pyarrow', "
            "or train from CSV with --csv instead."
        ) from exc
    return pq


def load_info(root: Path) -> Dict:
    """Read and version-check ``meta/info.json``.

    Args:
        root: Dataset root, the directory containing ``meta/`` and ``data/``.

    Returns:
        The parsed ``info.json`` payload.

    Raises:
        FileNotFoundError: If the path is not a LeRobot dataset root.
        ValueError: If the dataset is not ``codebase_version`` v3.0.
    """
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{root} does not look like a LeRobot dataset: no meta/info.json. "
            "Point --lerobot at the dataset root (the directory holding meta/ and data/)."
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = info.get("codebase_version")
    if version != SUPPORTED_CODEBASE_VERSION:
        raise ValueError(
            f"{root} is codebase_version {version!r}; this reader handles "
            f"{SUPPORTED_CODEBASE_VERSION!r} only. Convert it with lerobot's own "
            "dataset migration tooling, or export to CSV and use --csv."
        )
    return info


def dataset_dt(info: Dict) -> float:
    """Control period implied by the dataset's frame rate [s]."""
    fps = info.get("fps")
    if not fps or float(fps) <= 0.0:
        raise ValueError(f"meta/info.json has no usable fps (got {fps!r})")
    return 1.0 / float(fps)


def _element_names(feature: Dict) -> List[str]:
    """Element names of one vector feature, or [] when it has none.

    ``names`` is a flat list in some datasets and ``{"motors": [...]}`` in others
    (both appear in the wild), so both shapes are flattened here.
    """
    names = feature.get("names")
    if names is None:
        return []
    if isinstance(names, dict):
        flat: List[str] = []
        for value in names.values():
            if isinstance(value, (list, tuple)):
                flat.extend(str(v) for v in value)
        return flat
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def _numeric_features(info: Dict) -> Dict[str, Dict]:
    """Features that carry numbers this reader can pull channels out of."""
    out = {}
    for key, feature in info.get("features", {}).items():
        if key in RESERVED_COLUMNS:
            continue
        dtype = str(feature.get("dtype", ""))
        if dtype.startswith(("float", "int")):
            out[key] = feature
    return out


def available_channels(info: Dict) -> List[str]:
    """Every channel name this dataset offers, qualified as ``feature/channel``."""
    out: List[str] = []
    for key, feature in _numeric_features(info).items():
        element_names = _element_names(feature)
        if element_names:
            out.extend(f"{key}/{name}" for name in element_names)
        else:
            shape = feature.get("shape") or [1]
            width = int(np.prod(shape))
            out.extend(f"{key}[{i}]" for i in range(width)) if width > 1 else out.append(key)
    return out


def resolve_channel(info: Dict, name: str) -> Tuple[str, int]:
    """Locate one requested channel inside the dataset's vector features.

    Accepts three spellings:

    * ``lift`` -- an element name, searched across every numeric feature
    * ``observation.state/lift`` -- qualified, when the bare name is ambiguous
    * ``observation.state[0]`` -- positional, when the feature names nothing

    Args:
        info: Parsed ``meta/info.json``.
        name: The channel name as written in ``--q-cols`` and friends.

    Returns:
        ``(feature_key, index)`` locating the channel within its vector.

    Raises:
        KeyError: If the name matches no channel, or more than one.
    """
    features = _numeric_features(info)

    # Positional: observation.state[2]
    if name.endswith("]") and "[" in name:
        key, _, rest = name.partition("[")
        if key in features:
            return key, int(rest[:-1])

    # Qualified: observation.state/lift
    if "/" in name:
        key, _, channel = name.rpartition("/")
        if key in features:
            element_names = _element_names(features[key])
            if channel in element_names:
                return key, element_names.index(channel)
            raise KeyError(f"feature {key!r} has no channel {channel!r}; it names {element_names}")

    # Bare element name, or the key of a scalar feature.
    matches: List[Tuple[str, int]] = []
    for key, feature in features.items():
        element_names = _element_names(feature)
        if name in element_names:
            matches.append((key, element_names.index(name)))
        elif key == name and int(np.prod(feature.get("shape") or [1])) == 1:
            matches.append((key, 0))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(
            f"no channel named {name!r} in this dataset. Available: "
            f"{', '.join(available_channels(info)) or '(none named)'}"
        )
    owners = ", ".join(f"{key}/{name}" for key, _ in matches)
    raise KeyError(f"channel {name!r} is ambiguous; qualify it as one of: {owners}")


def resolve_channels(info: Dict, names: Sequence[str]) -> List[Tuple[str, int]]:
    """Resolve a whole channel group, reporting every failure at once."""
    resolved: List[Tuple[str, int]] = []
    problems: List[str] = []
    for name in names:
        try:
            resolved.append(resolve_channel(info, name))
        except KeyError as exc:
            problems.append(str(exc).strip('"'))
    if problems:
        raise KeyError("; ".join(problems))
    return resolved


def data_files(root: Path, info: Dict) -> List[Path]:
    """Every data shard, in dataset order.

    ``info.json`` lists them explicitly in some exports; otherwise the chunked
    tree is globbed, which sorts correctly because the indices are zero-padded.
    """
    root = Path(root)
    listed = info.get("data_files")
    if listed:
        return [root / rel for rel in listed]
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root}/data has no chunk-*/file-*.parquet shards")
    return files


def _column_matrix(table, key: str, width_hint: int) -> np.ndarray:
    """One feature column as a float32 [rows, width] array."""
    column = table.column(key)
    if column.null_count:
        raise ValueError(f"column {key!r} contains nulls; this reader expects complete frames")
    combined = column.combine_chunks()
    values = getattr(combined, "flatten", lambda: combined)()
    flat = np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return flat.reshape(table.num_rows, -1) if flat.size != table.num_rows else flat.reshape(-1, 1)


def _gather(table, channels: Sequence[Tuple[str, int]]) -> np.ndarray:
    """Pull an ordered channel group out of a table as [rows, len(channels)]."""
    cache: Dict[str, np.ndarray] = {}
    columns = []
    for key, index in channels:
        if key not in cache:
            cache[key] = _column_matrix(table, key, index + 1)
        matrix = cache[key]
        if index >= matrix.shape[1]:
            raise ValueError(
                f"channel index {index} is out of range for {key!r}, which is "
                f"{matrix.shape[1]} wide in the data files"
            )
        columns.append(matrix[:, index])
    return np.stack(columns, axis=1).astype(np.float32)


def load_lerobot_chunks(root: Path, spec: WindowSpec) -> List[Chunk]:
    """Load a LeRobot v3.0 dataset as contiguous chunks, one per episode.

    Episodes are the natural chunk boundary: history windows never span them, so
    the model is never shown a jump between two unrelated runs. Timeline breaks
    *within* an episode split it further, exactly as the CSV loader does.

    Args:
        root: Dataset root holding ``meta/`` and ``data/``.
        spec: Supplies the channel names to read and the expected period [s].

    Returns:
        One :class:`~dataset.Chunk` per contiguous stretch, named ``episode_<N>``.
        The names carry no wall clock, so ``--split session`` treats each episode
        as its own session.

    Raises:
        KeyError: If any requested channel is missing -- including velocity,
            which this reader never synthesises.
    """
    pq = _pyarrow()
    root = Path(root)
    info = load_info(root)

    q_channels = resolve_channels(info, spec.q_cols)
    qdot_channels = resolve_channels(info, spec.qdot_cols)
    u_channels = resolve_channels(info, spec.u_cols)

    dataset_period = dataset_dt(info)
    if abs(dataset_period - spec.dt) > 1e-6:
        raise ValueError(
            f"dataset runs at {1.0 / dataset_period:g} Hz (dt={dataset_period:g}s) but the "
            f"model spec uses dt={spec.dt:g}s. Pass --dt {dataset_period:g}, or resample the "
            "dataset -- history taps assume samples are dt apart."
        )

    out: List[Chunk] = []
    for path in data_files(root, info):
        table = pq.read_table(path)
        missing = [k for k, _ in q_channels + qdot_channels + u_channels if k not in table.schema.names]
        if missing:
            raise KeyError(f"{path.name} is missing feature columns: {', '.join(sorted(set(missing)))}")

        episodes = np.asarray(table.column("episode_index").to_numpy(zero_copy_only=False))
        timestamps = np.asarray(table.column(spec.time_col).to_numpy(zero_copy_only=False), dtype=np.float64)
        q_all = _gather(table, q_channels)
        qdot_all = _gather(table, qdot_channels)
        u_all = _gather(table, u_channels)

        if np.isfinite(u_all).any() and np.nanmax(np.abs(u_all)) > 1.001:
            raise ValueError(f"{path.name} contains commands outside [-1, 1]; normalize them before training")
        if np.isfinite(q_all).any() and np.nanmax(np.abs(q_all)) > 2.0 * np.pi:
            print(
                f"[WARN] {path.name}: joint positions reach "
                f"{np.nanmax(np.abs(q_all)):.1f}, which looks like degrees. The model "
                "works in radians -- convert the dataset if so."
            )

        # An episode lives in one shard (meta/episodes records a single
        # data/file_index per episode), so splitting within a file is enough. If
        # one ever did span shards it would become two chunks, which only makes
        # the history boundary more conservative -- never a leak.
        for start, stop in _runs(episodes):
            for a, b in _split_on_time_gaps(timestamps[start:stop], spec.dt):
                lo, hi = start + a, start + b
                if hi - lo <= spec.history_samples:
                    continue
                values = np.concatenate([q_all[lo:hi], qdot_all[lo:hi], u_all[lo:hi]], axis=1)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path.name} episode {int(episodes[lo])} has non-finite samples")
                out.append(
                    Chunk(
                        name=f"episode_{int(episodes[lo]):06d}#{len(out)}",
                        q=q_all[lo:hi],
                        qdot=qdot_all[lo:hi],
                        u=u_all[lo:hi],
                        t_end=float(timestamps[hi - 1]),
                    )
                )
    if not out:
        raise RuntimeError(f"{root} yielded no episode long enough for the requested window")
    return out


def _runs(values: np.ndarray) -> List[Tuple[int, int]]:
    """Maximal ``[start, stop)`` ranges over which ``values`` is constant."""
    if len(values) == 0:
        return []
    edges = np.flatnonzero(np.diff(values)) + 1
    bounds = np.concatenate(([0], edges, [len(values)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def _split_on_time_gaps(
    timestamps: np.ndarray, dt: float, max_gap_factor: float = 3.0
) -> List[Tuple[int, int]]:
    """Break one episode wherever its timestamps stop being dt apart.

    LeRobot computes ``timestamp`` as ``frame_index / fps``, so this is normally
    a no-op. It is here for datasets whose timestamps come from a real capture
    clock, where a dropped frame must not be windowed across.
    """
    if len(timestamps) < 2:
        return [(0, len(timestamps))]
    gaps = np.diff(timestamps)
    broken = (gaps <= 0.0) | (gaps > max_gap_factor * dt)
    bounds = np.concatenate(([0], np.flatnonzero(broken) + 1, [len(timestamps)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]


__all__ = [
    "SUPPORTED_CODEBASE_VERSION",
    "available_channels",
    "data_files",
    "dataset_dt",
    "load_info",
    "load_lerobot_chunks",
    "resolve_channel",
    "resolve_channels",
]
