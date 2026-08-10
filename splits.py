"""Train/validation splits for a sliding-window dataset.

Two strategies, for two different jobs:

``train_val_indices``          whole-group split. All chunks of one group (a
                               recording, or a driving session -- see
                               ``session_ids``) stay on the same side. Validation
                               is independent at the group level, so this is the
                               honest generalization measure. It also hands
                               free-running rollouts complete, naturally
                               contiguous ranges. No leakage buffer is needed:
                               windows never cross a chunk boundary.

``snippet_train_val_indices``  contiguous-snippet split within every chunk. Every
                               session contributes to training, which measured
                               stronger on the external benchmark, at the cost of
                               a validation set that is *not* independent at the
                               session level. Needs a leakage buffer.

Splitting a sliding-window dataset per row is wrong under either strategy:
consecutive windows overlap by nearly their whole history, so a random row split
puts near-duplicates on both sides and reports an optimistic validation loss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_PREPARED_SUFFIX = re.compile(r"(?:_seg\d+)?_chunk\d+$")
_WALL_CLOCK = re.compile(r"\d{8}_\d{6}")


def recording_id(name: str) -> str:
    """Return the source recording name for a prepared file or pool chunk."""
    stem = name.split("#", 1)[0]
    return _PREPARED_SUFFIX.sub("", stem)


# ----------------------------------------------------------------------------
# Session grouping
# ----------------------------------------------------------------------------

def _start_time(recording: str) -> Optional[float]:
    """Seconds since the epoch for the logger's ``YYYYMMDD_HHMMSS`` stamp.

    Parsed as UTC. Only differences are ever used, and reading the stamp as
    local time would make two files an hour apart across a DST boundary.
    Returns None when the name carries no wall clock at all.
    """
    match = _WALL_CLOCK.search(recording)
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(0), "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=timezone.utc).timestamp()


def session_ids(
    names: Sequence[str],
    t_ends: Sequence[float],
    max_gap_sec: float = 5.0,
) -> List[str]:
    """Group chunks into driving sessions. Returns one label per chunk.

    The logger rolls to a new file every ~600 s under a fresh wall-clock name,
    so one continuous drive arrives as several "recordings". Grouping on the
    filename alone would scatter one drive across both sides of a split built
    to prevent exactly that.

    Two recordings join the same session when the later one starts no more than
    ``max_gap_sec`` after the earlier one ended -- overlapping spans included,
    since a rollover can re-stamp slightly before the previous file's last
    sample. Distinct drives here are minutes to hours apart, so the threshold
    is not delicate.

    ``t_ends`` is each chunk's last timestamp *relative to the start of its
    source recording*, i.e. how long that recording ran. Chunks of one recording
    are collapsed to their longest.

    A name with no parsable wall clock cannot be placed on the timeline, so it
    becomes its own session rather than being merged into someone else's.
    """
    if len(names) != len(t_ends):
        raise ValueError("names and t_ends must have the same length")

    chunk_recordings = [recording_id(n) for n in names]

    duration: Dict[str, float] = {}
    for recording, t_end in zip(chunk_recordings, t_ends):
        duration[recording] = max(duration.get(recording, 0.0), float(t_end))

    starts = {recording: _start_time(recording) for recording in duration}

    # Undated recordings are their own session, keyed by their own name so they
    # can never collide with a dated session (which is named after a recording).
    session_of: Dict[str, str] = {
        recording: recording for recording, start in starts.items() if start is None
    }

    label: Optional[str] = None
    session_end = 0.0
    for start, recording in sorted(
        (start, recording) for recording, start in starts.items() if start is not None
    ):
        if label is None or start > session_end + max_gap_sec:
            label = recording
            session_end = start + duration[recording]
        else:
            session_end = max(session_end, start + duration[recording])
        session_of[recording] = label

    return [session_of[recording] for recording in chunk_recordings]


# ----------------------------------------------------------------------------
# Whole-group split
# ----------------------------------------------------------------------------

def train_val_indices(
    counts: Sequence[int],
    group_ids: Sequence[str],
    *,
    val_fraction: float,
    seed: int,
    group_key: str = "recordings",
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], dict]:
    """Split pool windows by group, keeping every group whole.

    ``counts`` and ``group_ids`` contain one entry per contiguous pool chunk.
    The returned validation ranges are chunk-local stretches expressed in the
    pool's global index space, ready for free-running rollout scoring.

    ``group_key`` only names things in ``info`` -- pass ``"sessions"`` when the
    ids come from :func:`session_ids`, so ``split_manifest.json`` says what was
    actually held out.
    """
    if len(counts) != len(group_ids):
        raise ValueError("counts and group_ids must have the same length")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if any(n <= 0 for n in counts):
        raise ValueError("every chunk must contain at least one window")

    groups = sorted(set(group_ids))
    if len(groups) < 2:
        raise ValueError(f"at least two source {group_key} are required for a split")

    n_val = int(round(len(groups) * val_fraction))
    n_val = min(max(n_val, 1), len(groups) - 1)
    rng = np.random.default_rng(seed)
    val_groups = set(rng.choice(groups, size=n_val, replace=False).tolist())

    train: List[np.ndarray] = []
    val: List[np.ndarray] = []
    val_ranges: List[Tuple[int, int]] = []
    offset = 0
    for count, source in zip(counts, group_ids):
        stop = offset + int(count)
        indices = np.arange(offset, stop, dtype=np.int64)
        if source in val_groups:
            val.append(indices)
            val_ranges.append((offset, stop))
        else:
            train.append(indices)
        offset = stop

    if not train or not val:
        raise ValueError("split produced an empty train or validation set")

    train_idx = np.concatenate(train)
    val_idx = np.concatenate(val)
    info = {
        "split": f"random_{group_key.rstrip('s')}",
        "val_fraction": float(val_fraction),
        "actual_val_window_fraction": float(len(val_idx) / offset),
        "split_seed": int(seed),
        f"n_{group_key}": len(groups),
        f"n_val_{group_key}": len(val_groups),
        f"train_{group_key}": sorted(set(groups) - val_groups),
        f"validation_{group_key}": sorted(val_groups),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "windows_dropped": 0,
    }
    return train_idx, val_idx, val_ranges, info


# ----------------------------------------------------------------------------
# Snippet split
# ----------------------------------------------------------------------------
#
# Leakage arithmetic. Window ``j`` of a chunk corresponds to resampled row
# ``j + H - 1`` (``H`` = ``spec.history_samples``); its inputs read rows
# ``[j, j+H-1]`` and its target reads row ``j+H``. Two windows therefore share no
# raw sample iff ``|j - k| >= H + 1``. If a train snippet ends at ``a`` and the
# next snippet starts at ``a+1`` with a different label, trimming ``H`` windows
# from that snippet's start puts its first surviving window at ``a+H+1`` --
# exactly disjoint.
#
# Only the second snippet of a differing-label pair is trimmed, and only from its
# start. Trimming both ends of every snippet would cost ~13% of the data instead
# of ~2%. Adjacent same-label snippets need no gap, and neither do chunk
# boundaries (different recordings never share samples).
#
# Rule of thumb for the cost:
#
#     trim_fraction ~= 2 * val_fraction * buffer / snippet_len
#
# At 10 s snippets, 10% validation and H=100 that is ~2%. Do not shorten snippets
# casually -- at 2 s it becomes ~10%.


@dataclass(frozen=True)
class Snippet:
    """A contiguous run of windows in the pool's global index space."""

    start: int              # inclusive
    stop: int               # exclusive
    first_in_chunk: bool    # no predecessor in time -> never needs a buffer

    def __len__(self) -> int:
        return self.stop - self.start


def make_snippets(counts: Sequence[int], snippet_len: int) -> List[Snippet]:
    """Cut each chunk's windows into contiguous snippets.

    Snippets are emitted in chunk order and never span a chunk, so
    ``snippets[k - 1]`` is the time-adjacent predecessor of ``snippets[k]``
    whenever ``snippets[k].first_in_chunk`` is False. The ragged remainder is
    folded into the chunk's last snippet rather than dropped.
    """
    if snippet_len < 1:
        raise ValueError("snippet_len must be at least 1 window")

    out: List[Snippet] = []
    offset = 0
    for n in counts:
        k = max(1, n // snippet_len)
        for i in range(k):
            start = offset + i * snippet_len
            stop = offset + n if i == k - 1 else offset + (i + 1) * snippet_len
            out.append(Snippet(start, stop, first_in_chunk=(i == 0)))
        offset += n
    return out


def label_snippets(n_snippets: int, val_fraction: float, seed: int) -> np.ndarray:
    """Randomly mark ``val_fraction`` of snippets as validation. Returns bool[n]."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    n_val = int(round(n_snippets * val_fraction))
    n_val = min(max(n_val, 1), n_snippets - 1)

    # A local generator, so the split is independent of the weight-init seed.
    rng = np.random.default_rng(seed)
    is_val = np.zeros(n_snippets, dtype=bool)
    is_val[rng.choice(n_snippets, size=n_val, replace=False)] = True
    return is_val


def gather_indices(
    snippets: Sequence[Snippet], is_val: np.ndarray, buffer: int
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Expand labelled snippets into index arrays, trimming across label changes.

    Also returns the validation snippets as contiguous ``[start, stop)`` ranges.
    Free-running rollouts need real consecutive time, which the flat index array
    cannot express.
    """
    train: List[np.ndarray] = []
    val: List[np.ndarray] = []
    val_ranges: List[Tuple[int, int]] = []
    for k, s in enumerate(snippets):
        crosses_label = not s.first_in_chunk and is_val[k] != is_val[k - 1]
        start = s.start + (buffer if crosses_label else 0)
        if start >= s.stop:
            continue                       # snippet shorter than the buffer
        idx = np.arange(start, s.stop)
        if is_val[k]:
            val.append(idx)
            val_ranges.append((start, s.stop))
        else:
            train.append(idx)

    if not train or not val:
        raise ValueError("split produced an empty train or validation set")
    return np.concatenate(train), np.concatenate(val), val_ranges


def snippet_train_val_indices(
    counts: Sequence[int],
    *,
    snippet_len: int,
    val_fraction: float,
    buffer: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], dict]:
    """Split pool windows into train/validation indices by contiguous snippet.

    Returns (train_idx, val_idx, val_ranges, info). ``val_ranges`` are the
    contiguous validation stretches, needed for free-running rollouts;
    ``info`` describes the split for the run's metadata.
    """
    snippets = make_snippets(counts, snippet_len)
    is_val = label_snippets(len(snippets), val_fraction, seed)
    train_idx, val_idx, val_ranges = gather_indices(snippets, is_val, buffer)

    total = int(sum(counts))
    info = {
        "split": "random_snippet",
        "snippet_len": int(snippet_len),
        "val_fraction": float(val_fraction),
        "actual_val_window_fraction": float(len(val_idx) / total),
        "buffer_windows": int(buffer),
        "split_seed": int(seed),
        "n_snippets": len(snippets),
        "n_val_snippets": int(is_val.sum()),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "windows_dropped": int(total - len(train_idx) - len(val_idx)),
        "val_snippet_ids": sorted(int(i) for i in np.flatnonzero(is_val)),
    }
    return train_idx, val_idx, val_ranges, info
