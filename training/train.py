"""Train a supervised hydraulic actuator forward model.

One MLP models every joint it is given, so it can learn their coupling. The
input is the current position, a dense velocity history and a sparse command
history; the target is ``delta_qdot = qdot(t + dt) - qdot(t)``, and the next
absolute velocity is reconstructed as ``qdot(t) + delta_qdot``.

Nothing here is fixed to one machine. ``--q-cols`` / ``--qdot-cols`` / ``--u-cols``
set the input and output widths, ``--hidden`` and ``--activation`` set the
architecture, and the ``*-history-sec`` flags set the window. Defaults provide
a three-joint baseline. The shipped V4 and slew models were selected through
separate recurrent fine-tuning experiments; this is the reusable base trainer.

``mlp_state_dict.pt`` is the checkpoint with the best held-out *rollout* position
error, not the last epoch and not the best one-step loss -- see rollout.py. The
final-epoch weights are kept alongside as ``mlp_state_dict_final.pt``.

Train/validation is split either over whole driving sessions or over contiguous
snippets -- ``--split``, see splits.py.

Two input sources, one pipeline. ``--csv`` reads the flat column layout below;
``--lerobot`` reads a LeRobot v3.0 dataset root and resolves the same channel
names against the vectors in its ``meta/info.json``. Both produce the same
chunks, so everything after loading is identical.

Cleaning is not part of this file; feed it data that already has the required
columns, velocity included -- neither loader differentiates position for you.
RPM and oil temperature are not inputs.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lerobot_source
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import (
    ACTIVATIONS,
    MLP,
    Q_COLS,
    QDOT_COLS,
    TARGET_MODE,
    TARGET_MODES,
    TIME_COL,
    U_COLS,
    ModelSpec,
    Normalizer,
    WindowSpec,
    build_pool,
    resolve_csvs,
    resolve_dataset_dir,
)
from rollout import RolloutScorer, make_starts
from splits import (
    recording_id,
    session_ids,
    snippet_train_val_indices,
    train_val_indices,
)


def resolve_out_dir(out_dir: Optional[str]) -> Path:
    """Where a run writes its artifacts.

    A relative ``--out`` is resolved against this script, not the working
    directory, so runs land under ``training/`` whether they were launched from
    here or from the repository root.

    The default is a fresh timestamped run directory. It is deliberately NOT a
    deployed ``models/*`` directory: that made a bare ``python train.py``
    overwrite the shipped artifact in place, with no way back.
    """
    if out_dir is None:
        out_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path = Path(out_dir)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, loss_fn) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(X), y).item())


def train(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    scorer: Optional[RolloutScorer] = None,
    select_on: str = "rollout",
    rollout_every: int = 5,
    checkpoint_every: int = 0,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, float], List[Dict[str, float]]]:
    """Minibatch training with best-checkpoint tracking.

    ``select_on`` picks the selection metric: ``"rollout"`` (free-running
    position error, the thing the model is actually judged by) or ``"val"``
    (one-step MSE). Rollout scoring runs every ``rollout_every`` epochs, and
    only those epochs are selection candidates.

    Always runs the full epoch budget -- no early stopping -- so the complete
    curve stays available. Returns the best weights, the epoch they came from,
    that epoch's metrics, and the per-epoch history.
    """
    if select_on == "rollout" and scorer is None:
        raise ValueError("select_on='rollout' requires a RolloutScorer")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    n_train = X_train.shape[0]
    step = n_train if batch_size <= 0 else batch_size

    best_score = float("inf")
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    try:
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            # Accumulate on-device: loss.item() here would sync once per step.
            running = torch.zeros((), device=device)
            for s in range(0, n_train, step):
                idx = perm[s : s + step]
                loss = loss_fn(model(X_train[idx]), y_train[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.detach() * idx.numel()
            tr_loss = float((running / n_train).item())

            va_loss = evaluate(model, X_val, y_val, loss_fn)
            scored = scorer is not None and (ep % rollout_every == 0 or ep == epochs)
            roll = scorer.score(model) if scored else float("nan")

            if select_on == "rollout":
                # Only scored epochs can be selected, so an unscored epoch is
                # never silently preferred over a measured one.
                candidate = roll if scored else float("inf")
            else:
                candidate = va_loss

            is_best = candidate < best_score
            if is_best:
                best_score, best_epoch = candidate, ep
                best_metrics = {"val": va_loss, "rollout": roll}
                best_state = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}

            if checkpoint_every > 0 and ep % checkpoint_every == 0:
                if checkpoint_dir is None:
                    raise ValueError("checkpoint_dir is required when checkpointing is enabled")
                path = checkpoint_dir / f"mlp_state_dict_epoch_{ep:05d}.pt"
                torch.save(model.state_dict(), path)
                print(f"Saved checkpoint: {path}")

            history.append(
                {
                    "epoch": float(ep),
                    "train": tr_loss,
                    "val": va_loss,
                    "rollout": roll,
                    "is_best": 1.0 if is_best else 0.0,
                    "elapsed_min": (time.perf_counter() - started) / 60.0,
                }
            )
            marker = " *best" if is_best else ""
            roll_text = "" if np.isnan(roll) else f" | roll {roll:.4f}rad"
            print(
                f"epoch {ep:05d} | train {tr_loss:.6f} | val {va_loss:.6f}"
                f"{roll_text} | {history[-1]['elapsed_min']:.1f} min{marker}"
            )
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping training and saving current artifacts.")

    if best_state is None:
        raise RuntimeError("Training stopped before a single scored epoch completed")
    return best_state, best_epoch, best_metrics, history


def main(
    csv_path: Optional[str],
    out_dir: Optional[str] = None,
    *,
    epochs: int = 1800,
    batch_size: int = 1024,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_every: int = 0,
    seed: int = 0,
    split_seed: int = 0,
    val_fraction: float = 0.10,
    split: str = "session",
    snippet_sec: float = 10.0,
    select_on: str = "rollout",
    rollout_every: int = 5,
    rollout_horizon_sec: float = 5.0,
    rollout_starts: int = 256,
    lerobot_path: Optional[str] = None,
    dt: Optional[float] = None,
    position_history_sec: float = 0.0,
    velocity_history_sec: float = 0.10,
    velocity_stride_sec: float = 0.01,
    command_history_sec: float = 0.99,
    command_stride_sec: float = 0.03,
    q_cols: Sequence[str] = Q_COLS,
    qdot_cols: Sequence[str] = QDOT_COLS,
    u_cols: Sequence[str] = U_COLS,
    time_col: str = TIME_COL,
    hidden: Sequence[int] = (128, 128, 128),
    activation: str = "relu",
    target_mode: str = TARGET_MODE,
    device: str = "auto",
):
    out = resolve_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {out}")
    checkpoint_dir = out / "checkpoints"
    if checkpoint_every > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_every > epochs:
            print(
                f"WARNING: --checkpoint-every {checkpoint_every} exceeds --epochs {epochs}; "
                "no checkpoint will be written. Note it now counts EPOCHS, not steps."
            )

    # One source or the other. The loader is the only thing that differs; every
    # stage after build_pool sees the same Chunk list either way.
    if (csv_path is None) == (lerobot_path is None):
        raise ValueError("Pass exactly one of --csv or --lerobot")

    if lerobot_path is not None:
        root = resolve_dataset_dir(lerobot_path)
        info = lerobot_source.load_info(root)
        dataset_period = lerobot_source.dataset_dt(info)
        if dt is None:
            dt = dataset_period
            print(f"Using dt={dt:g}s from the dataset's {info['fps']} fps")
        source_files = [root]
        loader = lerobot_source.load_lerobot_chunks
    else:
        dt = 0.01 if dt is None else dt
        source_files = resolve_csvs(csv_path)
        if not source_files:
            raise FileNotFoundError(f"Could not resolve CSV input: {csv_path}")
        loader = None

    spec = WindowSpec.from_seconds(
        velocity_history_sec,
        command_history_sec,
        command_stride_sec,
        dt=dt,
        position_history_sec=position_history_sec,
        velocity_stride_sec=velocity_stride_sec,
        q_cols=q_cols,
        qdot_cols=qdot_cols,
        u_cols=u_cols,
        time_col=time_col,
    )
    arch = ModelSpec(hidden=tuple(hidden), activation=activation)
    spec = replace(spec, target_mode=target_mode)

    pool = build_pool(source_files, spec, "pool", loader=loader)
    if split == "session":
        # Group by driving session, not by filename: the logger rolls files mid-drive
        # under a fresh timestamp, so filename grouping would straddle the split.
        chunk_sessions = session_ids(pool.names, pool.t_ends)
        train_idx, val_idx, val_ranges, split_info = train_val_indices(
            pool.counts,
            chunk_sessions,
            val_fraction=val_fraction,
            seed=split_seed,
            group_key="sessions",
        )
        session_of = dict(zip((recording_id(n) for n in pool.names), chunk_sessions))
        val_sessions = set(split_info["validation_sessions"])
        # LeRobot episodes carry no wall clock, so each is already its own
        # session and there is no file-level grouping to report.
        in_val = [session_of.get(recording_id(f.stem)) in val_sessions for f in source_files]
        split_info["train_files"] = [f.name for f, v in zip(source_files, in_val) if not v]
        split_info["validation_files"] = [f.name for f, v in zip(source_files, in_val) if v]
    else:
        # Snippet split: every session contributes to training, at the cost of a
        # validation set that is not independent at the session level.
        train_idx, val_idx, val_ranges, split_info = snippet_train_val_indices(
            pool.counts,
            snippet_len=max(1, int(round(snippet_sec / spec.dt))),
            val_fraction=val_fraction,
            buffer=spec.history_samples,
            seed=split_seed,
        )
        split_info["snippet_sec"] = float(snippet_sec)
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Normalizers are fitted on training windows only -- never validation,
    # never benchmark -- then applied to everything.
    xnorm = Normalizer.fit(pool.X[train_idx])
    ynorm = Normalizer.fit(pool.y[train_idx])
    xnorm.save(out, "x")
    ynorm.save(out, "y")

    horizon = max(1, int(round(rollout_horizon_sec / spec.dt)))
    starts = make_starts(val_ranges, spec, horizon, rollout_starts, seed=split_seed)

    # In place: a normalized copy of the pool would be another ~300 MB.
    xnorm.apply_inplace(pool.X)
    ynorm.apply_inplace(pool.y)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested but this torch build has no CUDA support "
            f"(torch {torch.__version__}, cuda build {torch.version.cuda}). "
            "Install a CUDA wheel, or use --device cpu."
        )
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dev.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    n_train = len(train_idx)
    step = n_train if batch_size <= 0 else batch_size
    steps_per_epoch = (n_train + step - 1) // step

    print(
        "\nTraining config:"
        f" dt={spec.dt}, velocity_history={(spec.hist_qdot - 1) * spec.dt:.2f}s"
        f" ({spec.hist_qdot} taps), command_history="
        f"{(spec.hist_u - 1) * spec.dt * spec.u_stride:.2f}s"
        f" ({spec.hist_u} taps at {spec.dt * spec.u_stride:.2f}s), hist_q={spec.hist_q}"
    )
    print(
        f"Split: {split_info['split']}, "
        + (
            f"{split_info['n_sessions']} sessions, {split_info['n_val_sessions']} to validation"
            if split == "session"
            else f"{split_info['n_snippets']} snippets of {snippet_sec:g}s, "
            f"{split_info['n_val_snippets']} to validation"
        )
        + f" (seed {split_seed}); train={split_info['train_windows']}, "
        f"val={split_info['val_windows']} "
        f"({100.0 * split_info['actual_val_window_fraction']:.1f}% of windows), "
        f"dropped={split_info['windows_dropped']}"
    )
    if split == "session":
        print(f"Validation sessions: {', '.join(split_info['validation_sessions'])}")
    print(f"Channels: {spec.n_q} joints {list(spec.qdot_cols)}, {spec.n_u} commands {list(spec.u_cols)}")
    print(
        f"Model: hidden={list(arch.hidden)}, activation={arch.activation}, "
        f"in_dim={spec.in_dim}, out_dim={spec.out_dim}, target={spec.target_mode}, "
        f"lr={lr}, weight_decay={weight_decay}"
    )
    print(
        f"Batching: batch_size={'full' if batch_size <= 0 else batch_size}, "
        f"steps/epoch={steps_per_epoch}, epochs={epochs}, "
        f"total optimizer steps={steps_per_epoch * epochs:,}"
    )
    print(
        f"Selection: on {select_on}; rollout = {len(starts)} starts x "
        f"{rollout_horizon_sec:g}s, scored every {rollout_every} epochs"
    )
    print(f"Device: {dev}")

    def upload(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x)).float().to(dev)

    Xtr, ytr = upload(pool.X[train_idx]), upload(pool.y[train_idx])
    Xva, yva = upload(pool.X[val_idx]), upload(pool.y[val_idx])
    resident = sum(t.numel() * 4 for t in (Xtr, ytr, Xva, yva))
    print(f"Resident on {dev}: {resident / 1e6:.0f} MB\n")

    scorer = RolloutScorer(
        spec,
        pool.q,
        pool.qdot,
        pool.u,
        starts,
        horizon,
        xnorm.mean,
        xnorm.std,
        ynorm.mean,
        ynorm.std,
        dev,
    )

    model = MLP(spec.in_dim, spec.out_dim, arch.hidden, arch.activation)
    best_state, best_epoch, best_metrics, history = train(
        model,
        Xtr,
        ytr,
        Xva,
        yva,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=dev,
        scorer=scorer,
        select_on=select_on,
        rollout_every=rollout_every,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )

    torch.save(model.state_dict(), out / "mlp_state_dict_final.pt")
    torch.save(best_state, out / "mlp_state_dict.pt")
    print(
        f"\nShipping best-on-{select_on} weights from epoch {best_epoch} "
        f"(val {best_metrics['val']:.6f}, rollout {best_metrics['rollout']:.4f} rad); "
        "final-epoch weights kept as mlp_state_dict_final.pt"
    )

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out / "train_history.csv", index=False)
    print(f"Saved training history to {out / 'train_history.csv'} ({len(history)} epochs)")

    meta = spec.to_meta(arch)
    meta["training"] = {
        "epochs": epochs,
        "epochs_completed": len(history),
        "batch_size": int(step),
        "steps_per_epoch": int(steps_per_epoch),
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "checkpoint_every": checkpoint_every,
        "seed": seed,
        "select_on": select_on,
        "rollout_every": int(rollout_every),
        "rollout_horizon_sec": float(rollout_horizon_sec),
        "rollout_starts": int(len(starts)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_metrics["val"]),
        "best_rollout_error_rad": float(best_metrics["rollout"]),
        "final_val_loss": float(hist_df["val"].iloc[-1]),
        "shipped_weights": f"best_on_{select_on}",
        "pool_files": [f.name for f in source_files],
        "source": "lerobot" if lerobot_path is not None else "csv",
        **split_info,
    }
    with open(out / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nDeployable model written to {out}:\n  "
        + "\n  ".join(
            [
                "mlp_state_dict.pt",
                "model_meta.json",
                "split_manifest.json",
                "x_mean.npy",
                "x_std.npy",
                "y_mean.npy",
                "y_std.npy",
            ]
        )
    )


if __name__ == "__main__":
    import argparse
    import sys

    # Windows consoles default to cp1252, which blows up on non-ASCII library output
    # (and block-buffers when redirected to a file, scrambling the log order).
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    p = argparse.ArgumentParser(
        description="Train a hydraulic actuator forward model. Defaults reproduce "
        "the shipped three-joint excavator arm model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io = p.add_argument_group("data and output")
    source = io.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        help="Training CSV file, directory, or glob. Relative paths are tried "
        "against the working directory, then training/, then the repo root.",
    )
    source.add_argument(
        "--lerobot",
        help="Root of a LeRobot v3.0 dataset (the directory holding meta/ and "
        "data/). Channel names in --q-cols/--qdot-cols/--u-cols are matched "
        "against the element names in meta/info.json, so they can name slots "
        "inside observation.state / action. The dataset must already carry "
        "velocity; this does not differentiate position for you.",
    )
    io.add_argument(
        "--out",
        default=None,
        help="Run directory; relative paths are resolved against this script. "
        "Defaults to a fresh runs/<timestamp>, never the deployed model/.",
    )
    io.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")

    opt = p.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=1800, help="Fixed epoch budget; always runs fully")
    opt.add_argument("--batch-size", type=int, default=1024, help="Minibatch size; 0 = full batch")
    opt.add_argument(
        "--lr", type=float, default=1e-4, help="Largest observed effect; lower needs more epochs"
    )
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--seed", type=int, default=0, help="Weight-init seed")
    opt.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save weights every N epochs (not steps); 0 disables. Needed to "
        "compare intermediate epochs after the fact -- the shipped and "
        "final weights alone cannot reconstruct the trajectory.",
    )

    sp = p.add_argument_group("train/validation split")
    sp.add_argument(
        "--split",
        default="session",
        choices=["session", "snippet"],
        help="session: hold out whole drives -- honest generalization measurement. "
        "snippet: hold out contiguous snippets from every drive -- keeps all "
        "sessions in training, which measured stronger on the external benchmark.",
    )
    sp.add_argument("--val-fraction", type=float, default=0.10)
    sp.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Kept separate from --seed so a weight-init sweep does not silently "
        "re-split the data (which would also refit the normalizers)",
    )
    sp.add_argument(
        "--snippet-sec",
        type=float,
        default=10.0,
        help="Snippet length for --split snippet. Shorter costs more to the "
        "leakage buffer: roughly 2*val_fraction*history/snippet_len.",
    )

    ro = p.add_argument_group("checkpoint selection (free-running rollout)")
    ro.add_argument(
        "--select-on",
        default="rollout",
        choices=["rollout", "val"],
        help="Selection metric: free-running position error, or one-step MSE. "
        "At 100 Hz one-step MSE mis-ranks free-running models -- 'val' is "
        "kept mainly so that result stays reproducible.",
    )
    ro.add_argument("--rollout-every", type=int, default=5, help="Score rollouts every N epochs")
    ro.add_argument("--rollout-horizon-sec", type=float, default=5.0)
    ro.add_argument("--rollout-starts", type=int, default=256)

    arch = p.add_argument_group("model architecture")
    arch.add_argument(
        "--target_mode",
        choices=TARGET_MODES,
        default=TARGET_MODE,
        help="Predict a velocity increment (legacy default) or absolute next velocity",
    )
    arch.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 128, 128],
        metavar="WIDTH",
        help="Hidden layer widths; the count of values is the number of layers",
    )
    arch.add_argument(
        "--activation",
        default="relu",
        choices=sorted(ACTIVATIONS),
        help="Hidden-layer activation. Both are parameterless, so this does not "
        "change the state_dict key layout the sim side loads",
    )

    ch = p.add_argument_group("input/output channels")
    ch.add_argument(
        "--q-cols",
        nargs="+",
        default=list(Q_COLS),
        metavar="COL",
        help="Joint-position columns. Their count is the joint count, and hence part of the input width",
    )
    ch.add_argument(
        "--qdot-cols",
        nargs="+",
        default=list(QDOT_COLS),
        metavar="COL",
        help="Joint-velocity columns; their count is also the OUTPUT width. Must "
        "match --q-cols in length -- position is integrated from velocity",
    )
    ch.add_argument(
        "--u-cols",
        nargs="+",
        default=list(U_COLS),
        metavar="COL",
        help="Command columns. Free of the joint count: three joints driven by "
        "four valve channels is expressible",
    )
    ch.add_argument("--time-col", default=TIME_COL, help="Timestamp column")

    win = p.add_argument_group("input window geometry")
    win.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Control period in seconds. Defaults to 0.01 for --csv, and to 1/fps "
        "from meta/info.json for --lerobot",
    )
    win.add_argument(
        "--position-history-sec",
        type=float,
        default=0.0,
        help="Position history depth; 0 means the current position only",
    )
    win.add_argument("--velocity-history-sec", type=float, default=0.10)
    win.add_argument(
        "--velocity-stride-sec",
        type=float,
        default=0.01,
        help="Spacing between velocity history taps",
    )
    win.add_argument("--command-history-sec", type=float, default=0.99)
    win.add_argument(
        "--command-stride-sec",
        type=float,
        default=0.03,
        help="Spacing between command history taps, not a normalization knob",
    )

    args = p.parse_args()

    main(
        args.csv,
        args.out,
        lerobot_path=args.lerobot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_every=max(args.checkpoint_every, 0),
        seed=args.seed,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        split=args.split,
        snippet_sec=args.snippet_sec,
        select_on=args.select_on,
        rollout_every=max(args.rollout_every, 1),
        rollout_horizon_sec=args.rollout_horizon_sec,
        rollout_starts=args.rollout_starts,
        dt=args.dt,
        position_history_sec=args.position_history_sec,
        velocity_history_sec=args.velocity_history_sec,
        velocity_stride_sec=args.velocity_stride_sec,
        command_history_sec=args.command_history_sec,
        command_stride_sec=args.command_stride_sec,
        q_cols=args.q_cols,
        qdot_cols=args.qdot_cols,
        u_cols=args.u_cols,
        time_col=args.time_col,
        hidden=args.hidden,
        activation=args.activation,
        target_mode=args.target_mode,
        device=args.device,
    )
