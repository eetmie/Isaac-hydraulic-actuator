"""
train.py

Supervised hydraulic actuator model:
X(t) = [q(t), qdot(t..t-Hq), u(t..t-Hu), pA(t), pB(t)]  -->  y(t) = qdot(t+1)

- Feed-forward MLP with explicit history stacking (like Egli & Hutter actuator model).
- Exports ONNX for IsaacSim usage.

Assumptions:
- Input logs come from ``data_collection/drive_logger.py``.
- We train on the three hydraulic arm axes only: boom, arm, bucket.
- Commands are the logged combined valve commands actually sent to the valves.
- Optional pressures are the logged extend/retract chamber readings.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Config for data_collection/drive_logger.py
# ----------------------------

TIME_COL = "timestamp"
SEGMENT_COL = "segment_id"

U_COLS = ["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]
Q_COLS = ["joint_pos_boom", "joint_pos_arm", "joint_pos_bucket"]
QDOT_COLS = ["joint_vel_boom", "joint_vel_arm", "joint_vel_bucket"]

@dataclass
class WindowSpec:
    dt: float = 0.02            # 50 Hz
    hist_qdot: int = 5          # qdot history length (0.1 s @ 50 Hz — paper default)
    hist_u: int = 50            # command history length (1.0 s @ 50 Hz — paper default)
    include_q: bool = True


# ----------------------------
# Utilities
# ----------------------------

def resample_to_fixed_dt(df: pd.DataFrame, dt: float, time_col: str) -> pd.DataFrame:
    """Resample one continuous segment using interpolation on a uniform time grid."""
    df = df.sort_values(time_col).drop_duplicates(time_col)
    if len(df) < 2:
        return df.copy()

    t0 = float(df[time_col].iloc[0])
    t1 = float(df[time_col].iloc[-1])
    t_grid = np.arange(t0, t1 + 1e-12, dt)
    out = pd.DataFrame({time_col: t_grid})

    x = df[time_col].to_numpy(np.float64)
    for col in df.columns:
        if col == time_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            first_valid = next((v for v in df[col] if pd.notna(v)), np.nan)
            out[col] = first_valid
            continue
        y = numeric.to_numpy(np.float64)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            if valid.sum() == 1:
                out[col] = y[valid][0]
            else:
                out[col] = np.nan
            continue
        out[col] = np.interp(t_grid, x[valid], y[valid])
    return out


def split_into_segments(
    df: pd.DataFrame,
    time_col: str,
    dt: float,
    max_gap_factor: float = 3.0,
) -> List[pd.DataFrame]:
    """Split dataframe into continuous segments, respecting segment_id if available."""
    if SEGMENT_COL in df.columns:
        segments: List[pd.DataFrame] = []
        for _, seg in df.groupby(SEGMENT_COL, sort=True):
            seg = seg.sort_values(time_col)
            if len(seg) >= 2:
                segments.append(seg)
        return segments

    df = df.sort_values(time_col)
    t = df[time_col].to_numpy(np.float64)
    gaps = np.diff(t)
    split_idx = np.where((gaps <= 0.0) | (gaps > max_gap_factor * dt))[0] + 1
    chunks = np.split(df, split_idx)
    return [c for c in chunks if len(c) >= 2]


def build_dataset_from_segments(
    segments: List[pd.DataFrame],
    spec: WindowSpec,
    resample: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build X/y arrays for each segment independently."""
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for seg in segments:
        seg_df = resample_to_fixed_dt(seg, spec.dt, TIME_COL) if resample else seg.sort_values(TIME_COL).ffill().bfill()

        if "cmd_stale" in seg_df.columns:
            cmd_stale = pd.to_numeric(seg_df["cmd_stale"], errors="coerce").fillna(0.0)
            seg_df = seg_df.loc[cmd_stale <= 0.5]

        required_cols = Q_COLS + QDOT_COLS + U_COLS
        seg_df = seg_df.dropna(subset=required_cols)
        if len(seg_df) <= max(spec.hist_qdot, spec.hist_u) + 1:
            continue
        X, y = build_supervised_xy(seg_df, spec)
        if len(X) > 0:
            out.append((X, y))
    return out


def make_history_matrix(arr: np.ndarray, hist: int) -> np.ndarray:
    """
    Given arr [T, D], return stacked history [T, D*hist]
    using (t, t-1, ..., t-hist+1). Pads by repeating first row.
    """
    T, D = arr.shape
    out = np.zeros((T, D * hist), dtype=np.float32)
    for k in range(hist):
        src_idx = np.clip(np.arange(T) - k, 0, T - 1)
        out[:, k * D:(k + 1) * D] = arr[src_idx, :]
    return out


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float = 1e-8) -> "Normalizer":
        m = x.mean(axis=0)
        s = x.std(axis=0)
        s = np.maximum(s, eps)
        return Normalizer(mean=m, std=s)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


# ----------------------------
# Dataset
# ----------------------------

class ActuatorDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_supervised_xy(
    df: pd.DataFrame,
    spec: WindowSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N, input_dim]
      y: [N, n_joints] = qdot(t+1)

    Notes:
      - drive_logger.py stores joint positions in radians and velocities in rad/s.
      - The model is unit-agnostic as long as training and inference use the same units.
    """
    q    = df[Q_COLS].to_numpy(np.float32)
    qdot = df[QDOT_COLS].to_numpy(np.float32)
    u    = df[U_COLS].to_numpy(np.float32)

    feats: List[np.ndarray] = []
    if spec.include_q:
        feats.append(q)
    feats.append(make_history_matrix(qdot, spec.hist_qdot))
    feats.append(make_history_matrix(u, spec.hist_u))

    X_all = np.concatenate(feats, axis=1)

    # Target is qdot(t+1)
    y_all = np.roll(qdot, shift=-1, axis=0)

    # Drop last row (has invalid y)
    X = X_all[:-1, :]
    y = y_all[:-1, :]
    return X, y


# ----------------------------
# Model
# ----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int] = [256, 256, 256]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Train / Eval
# ----------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 10,
    min_delta: float = 1e-5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    live_plotter: Optional["LivePlotter"] = None,
    plot_every: int = 1,
    checkpoint_every: int = 50,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[nn.Module, List[Dict[str, float]], int, float]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: Optional[Dict] = None
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    try:
        for ep in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item() * Xb.shape[0]
            tr_loss /= len(train_loader.dataset)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = model(Xb)
                    loss = loss_fn(pred, yb)
                    va_loss += loss.item() * Xb.shape[0]
            va_loss /= len(val_loader.dataset)

            best_display = best_val if np.isfinite(best_val) else va_loss
            delta = va_loss - best_display
            history.append(
                {
                    "epoch": float(ep),
                    "train": float(tr_loss),
                    "val": float(va_loss),
                    "best": float(best_display),
                    "delta": float(delta),
                    "no_improve": float(epochs_no_improve),
                }
            )
            print(
                f"epoch {ep:03d} | train {tr_loss:.6f} | val {va_loss:.6f} | "
                f"best {best_display:.6f} | delta {delta:+.6f} | "
                f"no_improve {epochs_no_improve}"
            )
            if live_plotter is not None and (ep % plot_every == 0 or ep == 1 or ep == epochs):
                live_plotter.update(ep, tr_loss, va_loss, best_display)

            if checkpoint_dir is not None and checkpoint_every > 0 and ep % checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"mlp_state_dict_epoch_{ep:04d}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            if va_loss < best_val - min_delta:
                best_val = va_loss
                best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {ep} (best epoch {best_epoch}, val {best_val:.6f})")
                    break
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping training and saving current artifacts.")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


class LivePlotter:
    def __init__(self, title: str = "Training Loss"):
        # Lazy import so matplotlib is only required when plotting is enabled.
        import matplotlib.pyplot as plt  # type: ignore

        self._plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.train_line, = self.ax.plot([], [], label="train")
        self.val_line, = self.ax.plot([], [], label="val")
        self.best_line, = self.ax.plot([], [], label="best", linestyle="--", alpha=0.6)
        self.ax.legend()
        self.epochs: List[int] = []
        self.train: List[float] = []
        self.val: List[float] = []
        self.best: List[float] = []

    def update(self, epoch: int, train_loss: float, val_loss: float, best_loss: float) -> None:
        self.epochs.append(epoch)
        self.train.append(float(train_loss))
        self.val.append(float(val_loss))
        self.best.append(float(best_loss))

        self.train_line.set_data(self.epochs, self.train)
        self.val_line.set_data(self.epochs, self.val)
        self.best_line.set_data(self.epochs, self.best)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self._plt.pause(0.001)

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    in_dim: int,
):
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, in_dim, dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["obs"],
        output_names=["qdot_next"],
        dynamic_axes={"obs": {0: "batch"}, "qdot_next": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {onnx_path}")


def main(
    csv_path: str,
    out_dir: str = "data_collection/nn/actuator_model_out",
    *,
    val_csv_path: Optional[str] = None,
    dt: float = 0.02,
    hist_qdot: Optional[int] = None,
    hist_u: Optional[int] = None,
    hist_qdot_sec: float = 0.1,
    hist_u_sec: float = 1.0,
    include_q: bool = True,
    resample: bool = True,
    hidden: List[int] = [128, 128],
    epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 2048,
    patience: int = 2000,
    min_delta: float = 1e-5,
    do_export_onnx: bool = True,
    save_xy: bool = False,
    live_plot: bool = False,
    plot_every: int = 1,
    plot_save: Optional[str] = None,
    checkpoint_every: int = 50,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path_resolved = Path(csv_path)
    csv_files: List[Path] = []
    if csv_path_resolved.is_file():
        csv_files = [csv_path_resolved]
    elif csv_path_resolved.is_dir():
        csv_files = sorted(csv_path_resolved.glob("*.csv"))
    else:
        candidate = Path(__file__).resolve().parent / csv_path
        if candidate.is_file():
            csv_files = [candidate]
        else:
            csv_files = sorted(Path().glob(csv_path))

    if not csv_files:
        raise FileNotFoundError(f"Could not resolve CSV input: {csv_path}")

    val_csv_files: List[Path] = []
    if val_csv_path is not None:
        val_path_resolved = Path(val_csv_path)
        if val_path_resolved.is_file():
            val_csv_files = [val_path_resolved]
        elif val_path_resolved.is_dir():
            val_csv_files = sorted(val_path_resolved.glob("*.csv"))
        else:
            candidate = Path(__file__).resolve().parent / val_csv_path
            if candidate.is_file():
                val_csv_files = [candidate]
            else:
                val_csv_files = sorted(Path().glob(val_csv_path))
        if not val_csv_files:
            raise FileNotFoundError(f"Could not resolve validation CSV input: {val_csv_path}")

    def _load_parts(files: List[Path], start_seg: int) -> Tuple[pd.DataFrame, int]:
        dfs: List[pd.DataFrame] = []
        global_seg = start_seg
        for i, f in enumerate(files):
            part = pd.read_csv(f)
            # Logger records timestamps in milliseconds — convert to seconds for dt-based windowing
            if TIME_COL in part.columns:
                part[TIME_COL] = pd.to_numeric(part[TIME_COL], errors="coerce") / 1000.0
            if SEGMENT_COL in part.columns:
                seg_ids = sorted(part[SEGMENT_COL].dropna().unique())
                seg_map = {old: idx + global_seg for idx, old in enumerate(seg_ids)}
                part[SEGMENT_COL] = part[SEGMENT_COL].map(seg_map)
                global_seg += len(seg_map)
            else:
                part[SEGMENT_COL] = global_seg + i
            dfs.append(part)
        df_local = pd.concat(dfs, axis=0, ignore_index=True)
        return df_local, global_seg + (0 if any(SEGMENT_COL in pd.read_csv(f, nrows=0).columns for f in files) else len(files))

    df, next_seg = _load_parts(csv_files, 0)
    print(f"Loaded train CSV file(s)={len(csv_files)}, total rows={len(df)}")

    # 1) derive history lengths from time horizons unless explicitly provided
    hist_qdot_steps = int(hist_qdot) if hist_qdot is not None else max(1, int(round(hist_qdot_sec / dt)))
    hist_u_steps = int(hist_u) if hist_u is not None else max(1, int(round(hist_u_sec / dt)))
    spec = WindowSpec(
        dt=dt,
        hist_qdot=hist_qdot_steps,
        hist_u=hist_u_steps,
        include_q=include_q,
    )

    # 2) split into continuous segments and build per-segment datasets
    train_segments = split_into_segments(df, TIME_COL, spec.dt)
    train_pairs = build_dataset_from_segments(train_segments, spec, resample=resample)
    if not train_pairs:
        raise RuntimeError("No usable training data segments after cleaning/resampling. Check your logs and settings.")

    if val_csv_files:
        df_val, _ = _load_parts(val_csv_files, next_seg)
        print(f"Loaded val CSV file(s)={len(val_csv_files)}, total rows={len(df_val)}")
        val_segments = split_into_segments(df_val, TIME_COL, spec.dt)
        val_pairs = build_dataset_from_segments(val_segments, spec, resample=resample)
        if not val_pairs:
            raise RuntimeError("No usable validation data segments after cleaning/resampling. Check your logs and settings.")
    else:
        n_segments = len(train_pairs)
        n_train_segments = max(1, int(0.8 * n_segments))
        if n_train_segments == n_segments and n_segments > 1:
            n_train_segments = n_segments - 1
        val_pairs = train_pairs[n_train_segments:] if n_segments > 1 else train_pairs[-1:]
        train_pairs = train_pairs[:n_train_segments]

    Xtr = np.concatenate([x for x, _ in train_pairs], axis=0)
    ytr = np.concatenate([y for _, y in train_pairs], axis=0)
    Xva = np.concatenate([x for x, _ in val_pairs], axis=0)
    yva = np.concatenate([y for _, y in val_pairs], axis=0)
    X = np.concatenate([Xtr, Xva], axis=0)
    y = np.concatenate([ytr, yva], axis=0)

    if save_xy:
        np.save(out / "X.npy", X)
        np.save(out / "y.npy", y)
        print(f"Saved X/y to {out}")

    # 4) normalize
    xnorm = Normalizer.fit(Xtr)
    ynorm = Normalizer.fit(ytr)

    Xtr_n = xnorm.transform(Xtr)
    Xva_n = xnorm.transform(Xva)
    ytr_n = ynorm.transform(ytr)
    yva_n = ynorm.transform(yva)

    # Save normalizers
    np.save(out / "x_mean.npy", xnorm.mean)
    np.save(out / "x_std.npy", xnorm.std)
    np.save(out / "y_mean.npy", ynorm.mean)
    np.save(out / "y_std.npy", ynorm.std)

    # 5) train
    print(
        "Training config:"
        f" dt={spec.dt}, hist_qdot={spec.hist_qdot}, hist_u={spec.hist_u},"
        f" include_q={spec.include_q}"
    )
    print(
        f"Data shapes: X={X.shape}, y={y.shape}, "
        f"segments={len(train_pairs) + len(val_pairs)}, train_segments={len(train_pairs)}, val_segments={len(val_pairs)}, "
        f"train={Xtr.shape[0]}, val={Xva.shape[0]}"
    )
    print(
        f"Model: hidden={hidden}, epochs={epochs}, lr={lr}, "
        f"weight_decay={weight_decay}, batch_size={batch_size}"
    )
    train_ds = ActuatorDataset(Xtr_n, ytr_n)
    val_ds = ActuatorDataset(Xva_n, yva_n)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    live_plotter = LivePlotter() if live_plot else None
    model = MLP(in_dim=X.shape[1], out_dim=y.shape[1], hidden=hidden)
    model, history, best_epoch, best_val = train(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_delta=min_delta,
        live_plotter=live_plotter,
        plot_every=plot_every,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=out,
    )

    # Save torch weights
    torch.save(model.state_dict(), out / "mlp_state_dict.pt")

    # Save training history for analysis
    hist_path = out / "train_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"Saved training history to {hist_path} (best epoch {best_epoch}, val {best_val:.6f})")
    if live_plotter is not None:
        if plot_save:
            live_plotter.save(Path(plot_save))
            print(f"Saved training plot to {plot_save}")
        else:
            live_plotter.save(out / "train_plot.png")
            print(f"Saved training plot to {out / 'train_plot.png'}")

    # Save model/spec metadata for sim-side inference
    meta = {
        "dt": spec.dt,
        "hist_qdot": spec.hist_qdot,
        "hist_u": spec.hist_u,
        "hist_qdot_sec": float(spec.hist_qdot * spec.dt),
        "hist_u_sec": float(spec.hist_u * spec.dt),
        "include_q": spec.include_q,
        "time_col": TIME_COL,
        "u_cols": U_COLS,
        "q_cols": Q_COLS,
        "qdot_cols": QDOT_COLS,
        "units": {
            "timestamp_raw": "ms",
            "joint_position": "rad",
            "joint_velocity": "rad/s",
            "command": "normalized_-1_to_1",
        },
        "model": {
            "hidden": hidden,
            "in_dim": int(X.shape[1]),
            "out_dim": int(y.shape[1]),
        },
    }
    with open(out / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 6) export ONNX
    if do_export_onnx:
        export_onnx(model, out / "actuator_mlp.onnx", in_dim=X.shape[1])


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV file, directory, or glob pattern")
    p.add_argument("--val-csv", default=None, help="Optional validation CSV file, directory, or glob pattern")
    p.add_argument("--out", default="data_collection/nn/actuator_model_out", help="Output directory")
    p.add_argument("--dt", type=float, default=0.01, help="Resample dt (s) — match your data logging rate")
    p.add_argument("--hist-qdot", type=int, default=None, help="Optional direct history length for qdot")
    p.add_argument("--hist-u", type=int, default=None, help="Optional direct history length for u")
    p.add_argument("--hist-qdot-sec", type=float, default=0.1, help="qdot history horizon in seconds (paper: 0.1 s)")
    p.add_argument("--hist-u-sec",    type=float, default=1.0, help="u history horizon in seconds   (paper: ~0.99 s)")
    p.add_argument("--no-resample", action="store_true", help="Disable resampling to fixed dt")
    p.add_argument("--no-q", action="store_true", help="Exclude current q from input")
    p.add_argument("--hidden", default="128,128", help="MLP hidden sizes, comma-separated")
    p.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    p.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    p.add_argument("--patience", type=int, default=2000, help="Early stopping patience")
    p.add_argument("--min-delta", type=float, default=1e-5, help="Min val improvement for early stop")
    p.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    p.add_argument("--save-xy", action="store_true", help="Save X.npy and y.npy")
    p.add_argument("--plot", action="store_true", help="Show live training plot")
    p.add_argument("--plot-every", type=int, default=1, help="Update plot every N epochs")
    p.add_argument("--plot-save", default=None, help="Optional path to save plot image")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Save model checkpoint every N epochs")
    args = p.parse_args()

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    main(
        args.csv,
        args.out,
        val_csv_path=args.val_csv,
        dt=args.dt,
        hist_qdot=args.hist_qdot,
        hist_u=args.hist_u,
        hist_qdot_sec=args.hist_qdot_sec,
        hist_u_sec=args.hist_u_sec,
        include_q=not args.no_q,
        resample=not args.no_resample,
        hidden=hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        patience=args.patience,
        min_delta=args.min_delta,
        do_export_onnx=not args.no_onnx,
        save_xy=args.save_xy,
        live_plot=args.plot,
        plot_every=max(args.plot_every, 1),
        plot_save=args.plot_save,
        checkpoint_every=max(args.checkpoint_every, 1),
    )
