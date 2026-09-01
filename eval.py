"""
eval.py

Rollout evaluation for the hydraulic actuator model.

One-step validation loss is a misleading score for this model: because
qdot(t+1) ~= qdot(t) at 100 Hz, a net can win on one-step MSE by leaning on its
own past velocity, then drift or limit-cycle the moment it is rolled out
autoregressively in the sim. Every number below is therefore reported
alongside a free-running rollout, which is how sim.py actually drives it.

For the canonical model, the network output is a velocity difference. This file
reconstructs the absolute next velocity before reporting metrics, so metrics
remain in physical rad/s and rad units.

Usage:
    python eval.py --model model --data "held_out/*.parquet"

Metrics:
  1. ONE-STEP     teacher-forced R2 / RMSE, against a persistence baseline
                  (qdot(t+1) = qdot(t)). Beating persistence is the bar for
                  "learned something"; losing to it is not automatically bad
                  if the rollout numbers are good.
  2. ROLLOUT      free-running: the model eats its own predicted qdot while
                  real logged commands drive it. Velocity RMSE and integrated
                  position error vs horizon. This is the metric that matters.
  3. REST         held at zero command from real quiet log points, self-fed.
                  Detects the limit cycle / creep. Goal is ~0.
  4. RESPONSE     mean predicted vs true qdot binned by command magnitude.
                  Checks monotonicity and per-axis gain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import MLP, Normalizer, WindowSpec, build_features, load_chunks, resolve_data_files

JOINTS = ["boom", "arm", "bucket"]


class Rollout:
    """Loads a trained model dir and runs teacher-forced / free-running predictions."""

    def __init__(self, model_dir: Path, weights: Path | None = None):
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.spec = WindowSpec.from_meta(meta)

        # Convenience aliases for callers.
        self.dt = self.spec.dt
        self.hq = self.spec.hist_qdot
        self.hu = self.spec.hist_u
        self.history_samples = self.spec.history_samples
        self.target_mode = meta["target_mode"]

        self.xnorm = Normalizer.load(model_dir, "x")
        self.ynorm = Normalizer.load(model_dir, "y")
        self.weights = weights if weights is not None else model_dir / "mlp_state_dict.pt"
        self.model = MLP(
            meta["model"]["in_dim"], meta["model"]["out_dim"], meta["model"]["hidden"]
        )
        self.model.load_state_dict(
            torch.load(self.weights, map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Absolute next velocity, reconstructed from the predicted delta."""
        with torch.no_grad():
            t = torch.from_numpy(self.xnorm.transform(x).astype(np.float32))
            if t.ndim == 1:
                t = t.unsqueeze(0)
            model_output = self.ynorm.inverse(self.model(t).numpy())

        offset = self.spec.qdot_offset
        current_qdot = np.asarray(x)[..., offset:offset + 3]
        return model_output + current_qdot

    def features(self, q, qdot, u) -> np.ndarray:
        return build_features(q, qdot, u, self.spec)

    def _buffer_sizes(self) -> tuple[int, int, int]:
        """Ring-buffer depths for the free-running rollouts (per signal, not the max)."""
        s = self.spec
        return (
            s.hist_q,
            (s.hist_qdot - 1) * s.qdot_stride + 1,
            (s.hist_u - 1) * s.u_stride + 1,
        )

    def free_run(
        self,
        q,
        qdot,
        u,
        start: int,
        u_seq: np.ndarray,
        velocity_limit: float | None = None,
        position_limits: np.ndarray | None = None,
    ):
        """Free-running rollout: model's own qdot is fed back. Returns (qdot_pred, q_pred)."""
        steps = len(u_seq)
        spec = self.spec
        nq, nv, nu = self._buffer_sizes()
        qbuf = np.zeros((nq, 3), np.float32)
        vbuf = np.zeros((nv, 3), np.float32)
        ubuf = np.zeros((nu, 3), np.float32)
        for k in range(nq - 1, -1, -1):
            qbuf = np.roll(qbuf, 1, 0); qbuf[0] = q[max(start - k, 0)]
        for k in range(nv - 1, -1, -1):
            vbuf = np.roll(vbuf, 1, 0); vbuf[0] = qdot[max(start - k, 0)]
        for k in range(nu - 1, -1, -1):
            ubuf = np.roll(ubuf, 1, 0); ubuf[0] = u[max(start - 1 - k, 0)]

        qc = q[start].copy()
        ov = np.zeros((steps, 3), np.float32)
        oq = np.zeros((steps, 3), np.float32)
        for s in range(steps):
            ubuf = np.roll(ubuf, 1, 0); ubuf[0] = u_seq[s]
            parts = ([qbuf.flatten()] if spec.include_q else []) + \
                    ([vbuf[::spec.qdot_stride].flatten()] if spec.hist_qdot > 0 else []) + \
                    [ubuf[::spec.u_stride].flatten()]
            p = self.predict(np.concatenate(parts))[0].astype(np.float32)
            if velocity_limit is not None:
                p = (np.clip(p, -velocity_limit, velocity_limit)
                     if np.all(np.isfinite(p)) else np.zeros_like(p))
            next_q = qc + self.dt * p
            if position_limits is not None:
                clamped_q = np.clip(next_q, position_limits[:, 0], position_limits[:, 1])
                p = np.where(clamped_q != next_q, 0.0, p)
                next_q = clamped_q
            ov[s] = p
            qc = next_q
            oq[s] = qc
            qbuf = np.roll(qbuf, 1, 0); qbuf[0] = qc
            if self.hq > 0:
                vbuf = np.roll(vbuf, 1, 0); vbuf[0] = p
        return ov, oq


    def free_run_batch(self, q, qdot, u, starts, u_seqs):
        """
        Vectorised free-running rollout over many trajectories at once.

        Rollouts are independent and equal length, so they batch cleanly. This is the
        dominant cost in large evaluations -- one trajectory at a time means thousands of
        single-sample forward passes, which is latency-bound (and actually *worse* on a
        GPU). Returns (qdot_pred, q_pred), each [B, steps, 3].
        """
        B, steps = len(starts), u_seqs.shape[1]
        spec = self.spec
        nq, nv, nu = self._buffer_sizes()
        qbuf = np.zeros((B, nq, 3), np.float32)
        vbuf = np.zeros((B, nv, 3), np.float32)
        ubuf = np.zeros((B, nu, 3), np.float32)
        for b, s0 in enumerate(starts):
            for k in range(nq):
                qbuf[b, k] = q[max(s0 - k, 0)]
            for k in range(nv):
                vbuf[b, k] = qdot[max(s0 - k, 0)]
            for k in range(nu):
                ubuf[b, k] = u[max(s0 - 1 - k, 0)]

        qc = np.stack([q[s] for s in starts]).astype(np.float32)
        ov = np.zeros((B, steps, 3), np.float32)
        oq = np.zeros((B, steps, 3), np.float32)
        for s in range(steps):
            ubuf = np.roll(ubuf, 1, axis=1); ubuf[:, 0] = u_seqs[:, s]
            parts = ([qbuf.reshape(B, -1)] if spec.include_q else []) + \
                    ([vbuf[:, ::spec.qdot_stride].reshape(B, -1)] if spec.hist_qdot > 0 else []) + \
                    [ubuf[:, ::spec.u_stride].reshape(B, -1)]
            x = np.concatenate(parts, axis=1)
            p = self.predict(x).astype(np.float32)
            ov[:, s] = p
            qc = qc + self.dt * p
            oq[:, s] = qc
            qbuf = np.roll(qbuf, 1, axis=1); qbuf[:, 0] = qc
            if self.hq > 0:
                vbuf = np.roll(vbuf, 1, axis=1); vbuf[:, 0] = p
        return ov, oq


def quiet_starts(u, hu, n, margin):
    """Indices where the command has been at rest for a full u-history window."""
    quiet = (np.abs(u) < 0.02).all(axis=1)
    runlen = np.convolve(quiet.astype(int), np.ones(hu, int), mode="full")[:len(quiet)]
    s = np.where(runlen >= hu)[0]
    s = s[(s > hu) & (s < len(u) - margin)]
    return s[:: max(1, len(s) // n)][:n]


def compute_metrics(R: "Rollout", q, qdot, u, n_track=24, n_rest=12, seed=0):
    """Compute the standard offline rollout metrics. Returns a flat dict."""
    T = len(q)
    eval_start = R.history_samples - 1
    X = R.features(q, qdot, u)[eval_start:-1]
    Y = np.roll(qdot, -1, axis=0)[eval_start:-1]
    P = R.predict(X)
    r2 = float((1 - ((P - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)).mean())

    H = int(5.0 / R.dt)
    history_len = R.history_samples
    starts = np.random.default_rng(seed).integers(history_len + 1, T - H - 2, size=n_track)
    ov, oq = R.free_run_batch(q, qdot, u, starts, np.stack([u[s:s + H] for s in starts]))
    tv = np.stack([qdot[s + 1:s + 1 + H] for s in starts])
    tp = np.stack([q[s + 1:s + 1 + H] for s in starts])
    roll_v = float(np.sqrt(((ov - tv) ** 2).mean(axis=1)).mean())
    roll_path = float(np.abs(oq - tp).mean())
    roll_final = float(np.abs(oq[:, -1] - tp[:, -1]).mean())

    qs = quiet_starts(u, history_len, n_rest, int(3.0 / R.dt) + 2)
    rest = float("nan")
    if len(qs):
        steps = int(3.0 / R.dt)
        ov2, _ = R.free_run_batch(q, qdot, u, qs, np.zeros((len(qs), steps, 3), np.float32))
        rest = float(np.abs(ov2[:, int(2.0 / R.dt):]).mean())
    return {"r2_1step": r2, "rollout_vel": roll_v,
            "rollout_pos": roll_path, "rollout_pos_final": roll_final,
            "rest": rest}


def load_log_segments(data_file: Path, dt: float):
    """Load a log as a list of (q, qdot, u) arrays, one per contiguous chunk."""
    spec = WindowSpec(dt=dt)
    return [(c.q, c.qdot, c.u) for c in load_chunks(Path(data_file), spec)]


def load_pool_segments(data_files, dt: float):
    """Every contiguous chunk of every file, pooled into one list.

    Chunks carry their own length, and the summary below weights by duration,
    so pooling files is equivalent to evaluating them separately and combining
    -- which is what a held-out benchmark set actually is.
    """
    spec = WindowSpec(dt=dt)
    out = []
    for path in data_files:
        out.extend((c.q, c.qdot, c.u) for c in load_chunks(Path(path), spec))
    return out


def load_log(data_file: Path, dt: float):
    segments = load_log_segments(data_file, dt)
    if len(segments) != 1:
        raise ValueError(f"{data_file.name} contains {len(segments)} continuous chunks")
    return segments[0]


def main(
    model_dir: str,
    data: str,
    horizons,
    n_track: int,
    n_rest: int,
    seed: int,
    weights: str | None = None,
):
    R = Rollout(Path(model_dir), Path(weights) if weights else None)
    print(f"model : {model_dir}  weights={R.weights.name}  (dt={R.dt}s "
          f"hist_qdot={R.hq} hist_u={R.hu} target={R.target_mode})")

    data_files = resolve_data_files(data)
    if not data_files:
        raise FileNotFoundError(f"Could not resolve data input: {data}")
    logs = load_pool_segments(data_files, R.dt)
    label = data_files[0].name if len(data_files) == 1 else f"{len(data_files)} files ({data})"
    if len(logs) > 1:
        print(f"log   : {label}  {len(logs)} continuous chunks\n")
        rows = []
        minimum = int(5.0 / R.dt) + R.history_samples + 2
        for i, (q, qdot, u) in enumerate(logs):
            if len(q) < minimum:
                print(f"chunk {i:02d}: skipped ({len(q)} rows; too short for 5 s rollout)")
                continue
            metrics = compute_metrics(R, q, qdot, u, n_track=n_track, n_rest=n_rest, seed=seed)
            rows.append((len(q), metrics))
            print(
                f"chunk {i:02d}: rows={len(q):5d} R2={metrics['r2_1step']:.4f} "
                f"vel={metrics['rollout_vel']:.4f} path={metrics['rollout_pos']:.4f} "
                f"final={metrics['rollout_pos_final']:.4f} "
                f"rest={metrics['rest']:.4f}"
            )
        if not rows:
            raise ValueError("No continuous chunk is long enough for evaluation")

        print("\nDuration-weighted summary:")
        for key in ("r2_1step", "rollout_vel", "rollout_pos", "rollout_pos_final", "rest"):
            values = np.array([metrics[key] for _, metrics in rows], dtype=np.float64)
            weights = np.array([length for length, _ in rows], dtype=np.float64)
            valid = np.isfinite(values)
            value = np.average(values[valid], weights=weights[valid]) if valid.any() else float("nan")
            print(f"  {key:16s} {value:.4f}")
        return

    q, qdot, u = logs[0]
    T = len(q)
    print(f"log   : {label}  {T} rows = {T * R.dt:.0f}s\n")

    # ---------------- 1) one-step ----------------
    eval_start = R.history_samples - 1
    X = R.features(q, qdot, u)[eval_start:-1]
    Y = np.roll(qdot, -1, axis=0)[eval_start:-1]
    P = R.predict(X)
    persist = qdot[R.history_samples - 1:-1]

    def r2(p, t):
        return 1.0 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)

    def rmse(p, t):
        return np.sqrt(((p - t) ** 2).mean(0))

    print("=== 1) ONE-STEP (teacher forced) ===")
    print(f"{'joint':8s} {'R2':>8s} {'R2_persist':>11s} {'RMSE':>8s} {'RMSE_persist':>13s}")
    a, b, c, d = r2(P, Y), r2(persist, Y), rmse(P, Y), rmse(persist, Y)
    for i, n in enumerate(JOINTS):
        print(f"{n:8s} {a[i]:8.4f} {b[i]:11.4f} {c[i]:8.4f} {d[i]:13.4f}")
    print(f"{'MEAN':8s} {a.mean():8.4f} {b.mean():11.4f} {c.mean():8.4f} {d.mean():13.4f}")

    # ---------------- 2) free-running rollout ----------------
    print("\n=== 2) ROLLOUT (free-running, real commands) ===")
    print("  velocity RMSE plus trajectory/final position MAE, mean over "
          f"{n_track} random starts")
    print(f"  {'horizon':>8s} | {'vel boom':>9s} {'arm':>7s} {'bucket':>7s} | "
          f"{'path boom':>9s} {'arm':>7s} {'bucket':>7s} | "
          f"{'final boom':>10s} {'arm':>7s} {'bucket':>7s}")
    rng = np.random.default_rng(seed)
    for hsec in horizons:
        H = int(hsec / R.dt)
        history_len = R.history_samples
        if T - H - 2 <= history_len + 1:
            continue
        starts = rng.integers(history_len + 1, T - H - 2, size=n_track)
        ve, path_error, final_error = [], [], []
        for s0 in starts:
            ov, oq = R.free_run(q, qdot, u, int(s0), u[s0:s0 + H])
            true_v = qdot[s0 + 1:s0 + 1 + H]
            true_q = q[s0 + 1:s0 + 1 + H]
            ve.append(np.sqrt(((ov - true_v) ** 2).mean(0)))
            path_error.append(np.abs(oq - true_q).mean(0))
            final_error.append(np.abs(oq[-1] - true_q[-1]))
        ve = np.mean(ve, axis=0)
        path_error = np.mean(path_error, axis=0)
        final_error = np.mean(final_error, axis=0)
        print(f"  {hsec:6.1f}s | {ve[0]:9.3f} {ve[1]:7.3f} {ve[2]:7.3f} | "
              f"{path_error[0]:9.3f} {path_error[1]:7.3f} {path_error[2]:7.3f} | "
              f"{final_error[0]:10.3f} {final_error[1]:7.3f} {final_error[2]:7.3f}")

    # ---------------- 3) rest stability ----------------
    print("\n=== 3) REST (hold u=0 from real quiet points, self-fed) ===")
    quiet = (np.abs(u) < 0.02).all(axis=1)
    history_len = R.history_samples
    runlen = np.convolve(quiet.astype(int), np.ones(history_len, int), mode="full")[:len(quiet)]
    starts = np.where(runlen >= history_len)[0]
    starts = starts[(starts > history_len) & (starts < T - int(3.0 / R.dt) - 2)]
    if len(starts) == 0:
        print("  no sustained-quiet windows in this log")
    else:
        starts = starts[:: max(1, len(starts) // n_rest)][:n_rest]
        steps = int(3.0 / R.dt)
        early, late, drift = [], [], []
        for s0 in starts:
            ov, oq = R.free_run(q, qdot, u, int(s0), np.zeros((steps, 3), np.float32))
            early.append(np.abs(ov[:int(1.0 / R.dt)]).mean(0))
            late.append(np.abs(ov[int(2.0 / R.dt):]).mean(0))
            drift.append(oq[-1] - q[s0])
        early = np.array(early).mean(0); late = np.array(late).mean(0)
        drift = np.array(drift).mean(0)
        print(f"  windows={len(starts)}")
        print(f"  |qdot| 0-1s : " + "  ".join(f"{n}={early[i]:.4f}" for i, n in enumerate(JOINTS)))
        print(f"  |qdot| 2-3s : " + "  ".join(f"{n}={late[i]:.4f}" for i, n in enumerate(JOINTS)))
        grew = late > early * 1.15 + 1e-3
        if grew.any():
            print("  VERDICT     : growing -> limit cycle on "
                  + ", ".join(n for i, n in enumerate(JOINTS) if grew[i])
                  + "   (model requires a rollout-stability change)")
        elif late.max() < 0.01:
            print("  VERDICT     : settles to rest")
        else:
            print(f"  VERDICT     : stable but creeps ({late.max():.3f} rad/s worst axis)")
        print(f"  drift @3s   : " + "  ".join(f"{n}={drift[i]:+.4f}" for i, n in enumerate(JOINTS)))

    # ---------------- 4) command response ----------------
    print("\n=== 4) RESPONSE (binned by command, teacher forced) ===")
    bins = [(-1.01, -0.6), (-0.6, -0.25), (-0.25, -0.02), (-0.02, 0.02),
            (0.02, 0.25), (0.25, 0.6), (0.6, 1.01)]
    for i, n in enumerate(JOINTS):
        parts = []
        for lo, hi in bins:
            m = (u[eval_start:-1, i] >= lo) & (u[eval_start:-1, i] < hi)
            if m.sum() < 50:
                continue
            parts.append(f"[{lo:+.2f},{hi:+.2f}) true={Y[m][:, i].mean():+.3f} "
                         f"pred={P[m][:, i].mean():+.3f}")
        print(f"  {n}:")
        for p in parts:
            print(f"    {p}")


if __name__ == "__main__":
    import sys
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model", help="Trained model directory")
    p.add_argument(
        "--data", "--csv", dest="data", required=True,
        help="Held-out Parquet or legacy CSV log(s): a file, a directory, or a glob. Multiple files are "
             "pooled and summarized duration-weighted.",
    )
    p.add_argument("--horizons", default="0.5,1,2,5,10", help="Rollout horizons in seconds")
    p.add_argument("--n-track", type=int, default=24, help="Random rollout starts per horizon")
    p.add_argument("--n-rest", type=int, default=20, help="Quiet windows for the rest test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights", default=None, help="Optional checkpoint state dict")
    a = p.parse_args()
    main(a.model, a.data, [float(x) for x in a.horizons.split(",")],
         a.n_track, a.n_rest, a.seed, a.weights)
