"""Contract tests for the train <-> sim boundary.

The training side (dataset.py, used by train.py and eval.py) and the sim side
(actuators/hydraulic_actuator.py) build the same feature vector from two
independent implementations: one vectorised over a whole log, one from ring
buffers at 100 Hz. Nothing but convention keeps them in agreement, and a
mismatch shows up only as wrong robot behaviour in Isaac Sim.

test_parity is what catches that. Run it after any change to the feature layout,
the meta keys, or the network architecture.

    python test_contract.py                     # split and meta tests only
    python test_contract.py --model models/arm_v4  # adds the two parity tests

Function names start with test_ so `pytest test_contract.py` also works, but
pytest is not required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PROJECT_ROOT, ModelSpec, WindowSpec, build_features, resolve_model_dir
from splits import (
    recording_id,
    session_ids,
    snippet_train_val_indices,
    train_val_indices,
)

# Every key actuators/hydraulic_actuator.py reads with [] -- a rename here is a
# KeyError at sim startup, so this list is the contract.
REQUIRED_META = [
    "dt",
    "hist_q",
    "hist_qdot",
    "hist_u",
    "qdot_stride",
    "u_stride",
    "include_q",
    "target_mode",
    "q_cols",
    "qdot_cols",
    "u_cols",
    "time_col",
]
REQUIRED_META_MODEL = ["in_dim", "out_dim", "hidden", "activation"]


def _load_actuator_class():
    """Import HydraulicActuatorNet without importing the actuators package.

    ``actuators/__init__.py`` also pulls in the two Isaac Lab actuators, so a
    plain ``from actuators import ...`` only works inside Isaac Sim. The net
    itself needs nothing but numpy and torch, so load its module by path and
    keep this test runnable from a bare interpreter.

    The path reaches up out of ``training/``: that the sim side is a sibling
    package this test can only get at by path, never by import, is exactly the
    independence the parity test exists to exploit.
    """
    import importlib.util

    path = PROJECT_ROOT / "actuators" / "hydraulic_actuator.py"
    spec = importlib.util.spec_from_file_location("_hydraulic_actuator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HydraulicActuatorNet


# One-joint slew shape: the geometry a cabin-slew model would be trained with.
SLEW_SPEC = WindowSpec(
    q_cols=("joint_pos_slew",),
    qdot_cols=("joint_vel_slew",),
    u_cols=("combined_cmd_slew",),
)

# Three joints, four command channels: widths are independent by design.
WIDE_U_SPEC = WindowSpec(u_cols=("cmd_a", "cmd_b", "cmd_c", "cmd_d"))


def test_meta_roundtrip() -> None:
    """A spec survives a trip through model_meta.json, and keeps every sim key."""
    specs = (
        WindowSpec(),
        WindowSpec(target_mode="velocity"),
        WindowSpec.from_seconds(0.2, 1.5, 0.03),
        WindowSpec(hist_q=1, hist_qdot=6, hist_u=20, qdot_stride=2, u_stride=5),
        SLEW_SPEC,
        WIDE_U_SPEC,
    )
    archs = (ModelSpec(), ModelSpec(hidden=(64, 64), activation="tanh"), ModelSpec(hidden=(32,)))
    for spec in specs:
        for arch in archs:
            meta = spec.to_meta(arch)
            for key in REQUIRED_META:
                assert key in meta, f"model_meta.json lost required key {key!r}"
            for key in REQUIRED_META_MODEL:
                assert key in meta["model"], f"model_meta.json lost required key model.{key!r}"
            assert WindowSpec.from_meta(meta) == spec, "spec did not survive the meta round trip"
            assert ModelSpec.from_meta(meta) == arch, "arch did not survive the meta round trip"
            assert meta["model"]["in_dim"] == spec.in_dim
            assert meta["model"]["out_dim"] == spec.out_dim


def test_width_derivation() -> None:
    """in_dim/out_dim follow the column lists, and mismatched widths are rejected."""
    assert SLEW_SPEC.out_dim == 1
    assert SLEW_SPEC.in_dim == SLEW_SPEC.hist_q + SLEW_SPEC.hist_qdot + SLEW_SPEC.hist_u

    # Commands are counted separately from joints.
    assert WIDE_U_SPEC.out_dim == 3
    assert WIDE_U_SPEC.in_dim == (3 * WIDE_U_SPEC.hist_q + 3 * WIDE_U_SPEC.hist_qdot + 4 * WIDE_U_SPEC.hist_u)

    # Position is integrated from velocity, so those two widths cannot differ.
    try:
        WindowSpec(q_cols=("a", "b"), qdot_cols=("c",), u_cols=("d",))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched q_cols/qdot_cols widths were accepted")

    try:
        ModelSpec(activation="selu")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown activation was accepted")


def test_activation_keeps_state_dict_layout() -> None:
    """Swapping the activation must not renumber the nn.Sequential keys.

    The sim side loads with strict=True, so a renumbering would break every
    checkpoint. Both activations are parameterless, and this pins that.
    """
    from dataset import MLP

    relu = MLP(8, 2, (4, 4), "relu")
    tanh = MLP(8, 2, (4, 4), "tanh")
    assert relu.state_dict().keys() == tanh.state_dict().keys(), (
        "activation change renumbered the state_dict keys"
    )


# A LeRobot meta/info.json shaped like the real thing: one feature names its
# elements with a flat list, the other with the {"motors": [...]} dict. Both
# spellings appear in published datasets and the reader must accept either.
LEROBOT_INFO = {
    "codebase_version": "v3.0",
    "fps": 100,
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [4],
            "names": ["pos_a", "pos_b", "vel_a", "vel_b"],
        },
        "action": {
            "dtype": "float32",
            "shape": [3],
            "names": {"motors": ["cmd_slew", "cmd_a", "cmd_b"]},
        },
        "load": {"dtype": "float32", "shape": [1], "names": None},
        "observation.images.cam": {"dtype": "video", "shape": [8, 8, 3], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    },
}


def test_lerobot_channel_resolution() -> None:
    """Channel names resolve to the right slot of the right vector feature.

    This is the whole of the LeRobot mapping: everything downstream sees plain
    arrays. Getting an index wrong here silently trains on the wrong joint, so
    the three spellings and both ``names`` encodings are pinned.
    """
    import lerobot_source as ls

    # Flat-list names, dict-of-lists names, and a scalar feature by its own key.
    assert ls.resolve_channel(LEROBOT_INFO, "pos_b") == ("observation.state", 1)
    assert ls.resolve_channel(LEROBOT_INFO, "vel_a") == ("observation.state", 2)
    assert ls.resolve_channel(LEROBOT_INFO, "cmd_b") == ("action", 2)
    assert ls.resolve_channel(LEROBOT_INFO, "load") == ("load", 0)

    # Qualified and positional spellings.
    assert ls.resolve_channel(LEROBOT_INFO, "action/cmd_slew") == ("action", 0)
    assert ls.resolve_channel(LEROBOT_INFO, "observation.state[3]") == ("observation.state", 3)

    # Video features are not channels.
    assert not any(c.startswith("observation.images") for c in ls.available_channels(LEROBOT_INFO))
    # Nor are the reserved bookkeeping columns.
    assert not any(c.startswith(("timestamp", "episode_index")) for c in ls.available_channels(LEROBOT_INFO))

    # A missing channel must raise rather than be invented -- this is what makes
    # "the dataset must carry velocity" a contract instead of a hope.
    try:
        ls.resolve_channels(LEROBOT_INFO, ("pos_a", "vel_missing"))
    except KeyError as exc:
        assert "vel_missing" in str(exc)
    else:
        raise AssertionError("a missing velocity channel was silently accepted")

    # Ambiguity across two features must be reported, not guessed.
    ambiguous = json.loads(json.dumps(LEROBOT_INFO))
    ambiguous["features"]["action"]["names"] = {"motors": ["pos_a", "x", "y"]}
    try:
        ls.resolve_channel(ambiguous, "pos_a")
    except KeyError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("an ambiguous channel name resolved silently")

    assert abs(ls.dataset_dt(LEROBOT_INFO) - 1.0 / 100.0) < 1e-12


def test_lerobot_rejects_other_versions(tmp_root=None) -> None:
    """Only v3.0 is read; anything else names its own version and stops."""
    import tempfile

    import lerobot_source as ls

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": "v2.1", "fps": 100}), encoding="utf-8"
        )
        try:
            ls.load_info(root)
        except ValueError as exc:
            assert "v2.1" in str(exc) and "v3.0" in str(exc)
        else:
            raise AssertionError("a v2.1 dataset was accepted")


def test_split_no_leakage() -> None:
    """Whole source recordings are disjoint and no windows are discarded."""
    counts = [100, 120, 140, 160, 180]
    sources = ["recording_a", "recording_a", "recording_b", "recording_c", "recording_d"]
    train_idx, val_idx, val_ranges, info = train_val_indices(counts, sources, val_fraction=0.25, seed=0)

    offsets = np.cumsum([0, *counts])
    source_by_index = np.empty(sum(counts), dtype=object)
    for i, source in enumerate(sources):
        source_by_index[offsets[i] : offsets[i + 1]] = source
    train_sources = set(source_by_index[train_idx])
    val_sources = set(source_by_index[val_idx])

    assert train_sources.isdisjoint(val_sources), "a recording appears in both splits"
    assert np.array_equal(np.sort(np.concatenate([train_idx, val_idx])), np.arange(sum(counts))), (
        "split lost or duplicated windows"
    )
    assert info["windows_dropped"] == 0
    assert len(val_ranges) == sum(source in val_sources for source in sources)

    # Prepared chunks from one recording must resolve to one stable group.
    assert recording_id("drive_log_123_seg000_chunk004#0") == "drive_log_123"
    assert recording_id("drive_log_123_seg001_chunk000") == "drive_log_123"

    # Determinism: same seed -> same split, different seed -> different split.
    again, _, _, _ = train_val_indices(counts, sources, val_fraction=0.25, seed=0)
    assert np.array_equal(train_idx, again), "split is not reproducible from its seed"
    other, _, _, _ = train_val_indices(counts, sources, val_fraction=0.25, seed=1)
    assert not np.array_equal(train_idx, other), "split ignored the seed"


def test_snippet_split_no_leakage() -> None:
    """No train window shares a raw sample with any validation window.

    A single chunk keeps global indices equal to in-chunk indices, so the
    minimum distance below is exactly the quantity the buffer must protect.
    """
    buffer = 100
    train_idx, val_idx, val_ranges, info = snippet_train_val_indices(
        [10_000], snippet_len=1000, val_fraction=0.2, buffer=buffer, seed=0
    )
    gap = np.abs(train_idx[:, None] - val_idx[None, :]).min()
    assert gap > buffer, f"train/val windows only {gap} apart, need > {buffer}"
    assert info["windows_dropped"] > 0, "buffer trimmed nothing"
    assert all(stop > start for start, stop in val_ranges)

    # Determinism: same seed -> same split, different seed -> different split.
    again, _, _, _ = snippet_train_val_indices(
        [10_000], snippet_len=1000, val_fraction=0.2, buffer=buffer, seed=0
    )
    assert np.array_equal(train_idx, again), "split is not reproducible from its seed"
    other, _, _, _ = snippet_train_val_indices(
        [10_000], snippet_len=1000, val_fraction=0.2, buffer=buffer, seed=1
    )
    assert not np.array_equal(train_idx, other), "split ignored the seed"


def test_session_grouping() -> None:
    """Consecutive logger files from one drive must land in the same session.

    The logger rolls to a new file every ~600 s under a fresh wall-clock name,
    so 185559_seg000 (600.00 s) and 190558_seg001 are one continuous 20-minute
    drive. Grouping on the filename would split them across train/val and leak
    ten minutes of the same drive through a split built to prevent exactly that.
    """
    names = [
        "drive_log_20260804_185559_seg000_chunk000",  # 18:55:59 + 600.00 s
        "drive_log_20260804_190558_seg001_chunk000",  # 19:05:58 -> continuation
        "drive_log_20260805_165209_seg000_chunk000",  # 16:52:09 + 600.01 s
        "drive_log_20260805_165216_seg001_chunk000",  # starts inside the above
        "drive_log_20260805_131325_seg000_chunk000",  # its own drive
    ]
    t_ends = [600.00, 594.92, 600.01, 2.87, 600.01]
    groups = session_ids(names, t_ends)

    assert groups[0] == groups[1], "file rollover split one drive into two sessions"
    assert groups[2] == groups[3], "overlapping spans must be the same session"
    assert groups[4] not in (groups[0], groups[2]), "unrelated drives were merged"
    assert len(set(groups)) == 3, f"expected 3 sessions, got {sorted(set(groups))}"

    # A name with no wall clock can never be merged into someone else's session.
    lone = session_ids(names + ["mystery_chunk000"], t_ends + [10.0])
    assert lone[-1] not in lone[:-1], "undated recording was merged into a session"


def test_parity(model_dir: str, tol: float = 1e-5) -> None:
    """dataset.build_features + eval.Rollout must equal the sim-side actuator.

    Guards, in one assertion: the [q | qdot | u] concat order, both stride
    subsamplings, the newest-tap-first buffer convention, the normalize /
    denormalize direction, the delta_velocity reconstruction, and the
    state_dict key names.
    """
    from eval import Rollout

    HydraulicActuatorNet = _load_actuator_class()

    R = Rollout(resolve_model_dir(model_dir))
    spec = R.spec
    H = spec.history_samples

    rng = np.random.default_rng(0)
    T = H + 200
    q = np.cumsum(rng.normal(0, 0.01, (T, spec.n_q)), axis=0).astype(np.float32)
    qdot = rng.normal(0, 0.2, (T, spec.n_qdot)).astype(np.float32)
    u = np.clip(rng.normal(0, 0.5, (T, spec.n_u)), -1, 1).astype(np.float32)

    reference = R.predict(build_features(q, qdot, u, spec))

    net = HydraulicActuatorNet(resolve_model_dir(model_dir), device="cpu")
    assert net.num_joints == spec.n_qdot, (
        f"sim actuator reports {net.num_joints} joints, training spec says {spec.n_qdot}"
    )
    assert net.num_commands == spec.n_u, (
        f"sim actuator reports {net.num_commands} commands, training spec says {spec.n_u}"
    )
    net.reset()
    actual = np.stack([net.predict_velocity(q[t], qdot[t], u[t]) for t in range(T)])

    # Compare only once the ring buffers are genuinely full. Before that the two
    # sides pad differently by design: make_history_matrix repeats row 0, while
    # reset() zero-fills the velocity/command buffers.
    start = H - 1
    err = np.abs(reference[start:] - actual[start:]).max()
    assert err < tol, f"train/sim feature mismatch: max abs error {err:.3e} (tol {tol:.0e})"
    return err


def test_rollout_matches_eval(model_dir: str, tol: float = 1e-4) -> float:
    """The torch rollout used for selection must match eval.py's numpy one.

    Training selects checkpoints with rollout.RolloutScorer; eval.py reports
    with Rollout.free_run_batch. If they drift, training optimizes a metric the
    reports do not measure.
    """
    from eval import Rollout
    from rollout import RolloutScorer

    R = Rollout(resolve_model_dir(model_dir))
    spec = R.spec

    # Synthetic, not logged: both implementations receive identical inputs, so
    # what is being compared is the recurrence, not the dynamics. Using a real
    # log here would only tie the test to a dataset that does not ship. The
    # frequencies are generated per channel so this works at any model width.
    T = 1600
    t = np.arange(T, dtype=np.float32) * spec.dt

    def waves(n: int, base: float, spread: float, scale: float = 1.0) -> np.ndarray:
        return (
            scale
            * np.stack(
                [np.sin(2 * np.pi * (base + i * spread) * t + 0.7 * i) for i in range(n)],
                axis=1,
            )
        ).astype(np.float32)

    u = waves(spec.n_u, 0.31, 0.22)
    qdot = waves(spec.n_qdot, 0.23, 0.19, scale=0.4)
    q = np.cumsum(qdot * spec.dt, axis=0).astype(np.float32)

    horizon = 200
    starts = np.arange(spec.history_samples, spec.history_samples + 32) * 7
    starts = starts[starts + horizon < len(q) - 1]

    _, oq = R.free_run_batch(q, qdot, u, starts, np.stack([u[s : s + horizon] for s in starts]))
    numpy_error = float(np.abs(oq[:, -1] - q[starts + horizon]).mean())

    scorer = RolloutScorer(
        spec,
        q,
        qdot,
        u,
        starts,
        horizon,
        R.xnorm.mean,
        R.xnorm.std,
        R.ynorm.mean,
        R.ynorm.std,
        "cpu",
    )
    torch_error = scorer.score(R.model)

    diff = abs(numpy_error - torch_error)
    assert diff < tol, (
        f"torch rollout {torch_error:.6f} != numpy rollout {numpy_error:.6f} (diff {diff:.2e}, tol {tol:.0e})"
    )
    return diff


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=None,
        help="Model dir to run the parity tests against, e.g. models/arm_v4",
    )
    p.add_argument("--tol", type=float, default=1e-5)
    args = p.parse_args()

    failures = 0
    for fn in (
        test_meta_roundtrip,
        test_width_derivation,
        test_activation_keeps_state_dict_layout,
        test_lerobot_channel_resolution,
        test_lerobot_rejects_other_versions,
        test_split_no_leakage,
        test_snippet_split_no_leakage,
        test_session_grouping,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL  {fn.__name__}: {exc}")
            failures += 1

    if args.model is None:
        print("SKIP  test_parity, test_rollout_matches_eval (pass --model <dir>)")
    else:
        try:
            err = test_parity(args.model, args.tol)
            print(f"PASS  test_parity (max abs error {err:.3e}, tol {args.tol:.0e})")
        except AssertionError as exc:
            print(f"FAIL  test_parity: {exc}")
            failures += 1
        try:
            diff = test_rollout_matches_eval(args.model)
            print(f"PASS  test_rollout_matches_eval (diff {diff:.3e} rad)")
        except AssertionError as exc:
            print(f"FAIL  test_rollout_matches_eval: {exc}")
            failures += 1

    print("OK" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    raise SystemExit(main())
