# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare causal IMU data, run a paired actuator experiment, and freeze a selection.

Run as ``python -m training.gyro_transfer prepare|train|evaluate`` from the repo root.
Preparation reuses accepted recording segments and frozen splits, but reconstructs
inputs from the last complete IMU packet available at each recorded host tick.
Centred velocity labels remain *targets/diagnostics only* in the gyro experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from actuators.hydraulic_actuator import HydraulicActuatorNet
from hydraulic_controller.core import DEFAULT_MODEL, ROOT, model_fingerprint
from hydraulic_controller.sensors import ROLES, SENSOR_CONTRACT, aligned_rates, causal_indices, imu_positions

DEFAULT_OUT = ROOT / "runs/gyro_transfer"
ARCHIVE = ROOT / "local/prototyping"


def unwrap_clock(values):
    """Unwrap the Pico's unsigned microsecond clock without using future samples."""
    raw = np.asarray(values, dtype=np.float64)
    return raw + np.cumsum(np.r_[0, np.diff(raw) < -(2**31)]) * 2**32


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    import pandas as pd

    if args.split == "test" and not (args.out / "selection.json").exists():
        raise ValueError("Freeze selection on development recordings before preparing test inputs")
    manifest = json.loads((args.archive / "data_clean/claude/manifest.json").read_text())
    if abs(manifest["carriage_pitch_zero_rad"] - SENSOR_CONTRACT["carriage_pitch_zero_rad"]) > 1e-8:
        raise ValueError("Carriage zero differs from the deployed sensor contract")
    splits = ("train", "dev") if args.split == "development" else ("test",)
    output = args.out / "data"
    output.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    for split in splits:
        files = sorted((args.archive / "data_clean/claude/new" / split).glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No accepted chunks for {split}")
        drive_groups = {}
        for path in files:
            drive_groups.setdefault(path.stem.split("_native_")[0], []).append(path)
        for stem, paths in drive_groups.items():
            drive_path = args.archive / "dataset/V3" / (stem + ".csv")
            stamp = stem.split("_")[3]
            if manifest["recording_splits"][stamp] != split:
                raise ValueError("Split mismatch")

            def seconds(s):
                return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])

            raw_paths = [
                p
                for p in drive_path.parent.glob("imu_raw_" + stem.split("_")[2] + "_*.csv")
                if 0 <= seconds(p.stem.split("_")[3]) - seconds(stamp) <= 15
            ]
            if len(raw_paths) != 1:
                raise ValueError(f"Ambiguous raw recording for {stem}")
            d = pd.read_csv(
                drive_path,
                usecols=[
                    "timestamp",
                    "state_imu_ts_us",
                    "combined_cmd_lift",
                    "combined_cmd_tilt",
                    "combined_cmd_scoop",
                ],
            )
            r = pd.read_csv(
                raw_paths[0],
                usecols=["device_ts_us"]
                + [f"imu_{role}_q{axis}" for role in ROLES for axis in "wxyz"]
                + [f"imu_{role}_gy_dps" for role in ROLES],
            )
            raw_t, host_clock = unwrap_clock(r.device_ts_us), unwrap_clock(d.state_imu_ts_us)
            host_clock += round((raw_t[0] - host_clock[0]) / 2**32) * 2**32
            quats = np.stack([r[[f"imu_{role}_q{a}" for a in "wxyz"]].to_numpy() for role in ROLES], 1)
            # Historical mounting corrections are pure rotations about Y; gyro Y is invariant.
            positions = imu_positions(quats)
            velocities = aligned_rates(r[[f"imu_{role}_gy_dps" for role in ROLES]].to_numpy())
            commands = d[["combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop"]].to_numpy()
            hashes = {"drive": digest(drive_path), "imu": digest(raw_paths[0])}
            for source in paths:
                with np.load(source) as old:
                    ticks = old["t"].copy()
                    host_ids = causal_indices(d.timestamp.to_numpy(), ticks)
                    packet_ids = causal_indices(raw_t * 1e-6, host_clock[host_ids] * 1e-6)
                    q, v, u = (
                        positions[packet_ids],
                        velocities[packet_ids],
                        commands[host_ids].astype(np.float32),
                    )
                    target_q, target_v = old["q"].copy(), old["v_smooth21"].copy()
                if not np.isfinite(np.c_[q, v, u, target_q, target_v]).all():
                    raise ValueError(f"Nonfinite accepted data in {source}")
                dest = output / split / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, q=q, v=v, u=u, target_q=target_q, target_v=target_v, t=ticks)
                one_s = q[100:] - q[:-100] - 0.01 * (np.cumsum(v, axis=0)[100:] - np.cumsum(v, axis=0)[:-100])
                quiet = (np.max(np.abs(u), 1) < 0.01) & (np.max(np.abs(target_v[:, :3]), 1) < 0.01)
                diagnostics.append(
                    {
                        "chunk": source.stem,
                        "split": split,
                        "rows": len(q),
                        "gyro_angle_closure_rms_deg": np.rad2deg(np.sqrt((one_s**2).mean(0))).tolist(),
                        "gyro_target_rmse_rad_s": np.sqrt(((v - target_v) ** 2).mean(0)).tolist(),
                        "quiet_rows": int(quiet.sum()),
                        "quiet_rate_std_rad_s": v[quiet].std(0).tolist() if quiet.any() else None,
                        "quiet_rate_mean_rad_s": v[quiet].mean(0).tolist() if quiet.any() else None,
                    }
                )
                rows.append(
                    {
                        "path": str(dest.relative_to(args.out)),
                        "split": split,
                        "source": str(source),
                        "source_sha256": digest(source),
                        "raw_sha256": hashes,
                        "rows": len(q),
                    }
                )
            print(f"Prepared {split}: {stem}", flush=True)
    report = {
        "sensors": SENSOR_CONTRACT,
        "dt": 0.01,
        "cohort": "V2/V3 native firmware only",
        "target_policy": "Existing smoothed labels are targets only; no future packets in gyro inputs",
        "chunks": rows,
        "diagnostics": diagnostics,
    }
    (args.out / f"data_{args.split}.json").write_text(json.dumps(report, indent=2) + "\n")


def load_chunks(out, split):
    chunks = []
    for path in sorted((out / "data" / split).glob("*.npz")):
        with np.load(path) as data:
            chunks.append({**{key: data[key] for key in data.files}, "name": path.stem})
    if not chunks:
        raise FileNotFoundError(f"Prepare {split} data first")
    return chunks


class Predictor:
    def __init__(self, path, device):
        src = HydraulicActuatorNet(path, device=device, sim_dt=0.01)
        if (src.hist_q, src.hist_qdot, src.qdot_stride, src.hist_u, src.u_stride) != (1, 41, 1, 21, 3):
            raise ValueError("This paired experiment requires the V5 history contract")
        self.src, self.net = src, src._model

    def next_velocity(self, q, vh, uh):
        src = self.src
        x = torch.cat((q, vh.flatten(1), uh[:, ::3].flatten(1)), 1)
        return vh[:, 0] + self.net((x - src._x_mean) / src._x_std) * src._y_std + src._y_mean


@torch.no_grad()
def evaluate(path, chunks, device, count=24):
    """Common causal-history evaluation for every candidate; no measured state feedback during rollouts."""
    p = Predictor(path, device)
    p.net.eval()
    result = {}
    for seconds in (0.01, 1, 10, 30):
        horizon = max(1, round(seconds * 100))
        candidates = [(c, s) for c in chunks for s in range(100, len(c["q"]) - horizon, max(horizon, 100))]
        ids = np.random.default_rng(731).choice(len(candidates), min(count, len(candidates)), replace=False)
        cases = [candidates[i] for i in ids]
        if not cases:
            raise ValueError(f"No {seconds} s rollout cases")

        def tens(x):
            return torch.tensor(np.stack(x), dtype=torch.float32, device=device)

        q = tens([c["q"][s] for c, s in cases])
        vh = tens([c["v"][s - np.arange(41)] for c, s in cases])
        truth = tens([c["target_q"][s + 1 : s + horizon + 1] for c, s in cases])
        all_u = tens([c["u"][s - 60 : s + horizon] for c, s in cases])
        velocity_target = tens([c["target_v"][s + 1 : s + horizon + 1] for c, s in cases])
        error, rate_error = [], []
        for k in range(horizon):
            uh = all_u[:, k + 60 - torch.arange(61, device=device)]
            v = p.next_velocity(q, vh, uh)
            q = q + 0.01 * v
            vh = torch.cat((v[:, None], vh[:, :-1]), 1)
            error.append((q - truth[:, k]).abs())
            rate_error.append((v - velocity_target[:, k]).square())
        e, ve = torch.stack(error), torch.stack(rate_error)
        if not torch.isfinite(e).all():
            return {"finite": False, "path10_deg": 1e12}
        result[str(seconds)] = {
            "cases": len(cases),
            "path_deg": float(torch.rad2deg(e[:, :, :3].mean())),
            "endpoint_deg": float(torch.rad2deg(e[-1, :, :3].mean())),
            "velocity_rmse_rad_s": ve[:, :, :3].mean().sqrt().item(),
        }
    # Synthetic holds and a commanded stop from measured moving histories.
    cases = [(c, s) for c in chunks for s in range(200, len(c["q"]), 2000)][:count]
    rest = torch.tensor(np.stack([c["q"][s] for c, s in cases]), device=device)
    rest[:, 3] = 0
    q = torch.cat((rest, rest), 0)
    q0 = q.clone()
    moving = torch.tensor(np.stack([c["v"][s - np.arange(41)] for c, s in cases]), device=device)
    vh = torch.cat((torch.zeros_like(moving), moving), 0)
    uh = torch.tensor(np.stack([c["u"][s - np.arange(61)] for c, s in cases]), device=device)
    uh = torch.cat((torch.zeros_like(uh), uh), 0)
    late_speed = []
    for k in range(1000):
        uh = torch.cat((torch.zeros_like(uh[:, :1]), uh[:, :-1]), 1)
        v = p.next_velocity(q, vh, uh)
        q += 0.01 * v
        vh = torch.cat((v[:, None], vh[:, :-1]), 1)
        if k >= 900:
            late_speed.append(v[len(rest) :, :3].square())
    result["finite"] = bool(torch.isfinite(q).all())
    result["rest10_deg"] = float(torch.rad2deg((q[: len(rest), :3] - q0[: len(rest), :3]).abs().mean()))
    result["stop_rate_rad_s"] = float(torch.stack(late_speed).mean().sqrt())
    result["path10_deg"] = result["10"]["path_deg"]
    return result


def save_candidate(predictor, original, out, step, label, metrics):
    out.mkdir(parents=True, exist_ok=True)
    src = predictor.src
    meta = json.loads((original / "model_meta.json").read_text())
    meta["version"] = "gyro-transfer-1"
    meta["training"] = {
        "parent": str(original),
        "input_velocity": label,
        "target_velocity": "smooth21",
        "sensor_contract": SENSOR_CONTRACT,
        "step": step,
        "development": metrics,
        "test_seen": False,
        "note": "Fine-tuned weights; parent's compiled bound is not guaranteed",
    }
    for name, value in (
        ("x_mean", src._x_mean),
        ("x_std", src._x_std),
        ("y_mean", src._y_mean),
        ("y_std", src._y_std),
    ):
        np.save(out / (name + ".npy"), value.detach().cpu().numpy())
    torch.save(
        {k: v.detach().cpu() for k, v in predictor.net.state_dict().items()}, out / "mlp_state_dict.pt"
    )
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def train(args):
    """Equal-budget smooth-history control and causal-gyro experiment, from identical weights."""
    chunks, dev = load_chunks(args.out, "train"), load_chunks(args.out, "dev")
    baseline = evaluate(args.model, dev, args.device, args.cases)
    (args.out / "baseline_dev.json").write_text(json.dumps(baseline, indent=2) + "\n")
    selection = {
        "baseline": str(args.model.resolve()),
        "baseline_fingerprint": model_fingerprint(args.model),
        "baseline_metrics": baseline,
        "budget": vars(args).copy(),
        "candidates": {},
    }
    selection["budget"] = {k: str(v) if isinstance(v, Path) else v for k, v in selection["budget"].items()}
    for label in ("smooth", "gyro"):
        torch.manual_seed(args.seed)
        p = Predictor(args.model, args.device)
        p.net.train().requires_grad_(True)

        def tensor(key):
            return torch.tensor(np.concatenate([c[key] for c in chunks]), device=args.device)

        q, v, target_q, target_v, u = [tensor(k) for k in ("q", "v", "target_q", "target_v", "u")]
        input_v = target_v if label == "smooth" else v
        legal, offset = [], 0
        for c in chunks:
            legal.extend(range(offset + 61, offset + len(c["q"]) - args.horizon - 1))
            offset += len(c["q"])
        legal = torch.tensor(legal, device=args.device)
        scale = target_v.std(0).clamp_min(0.003)
        opt = torch.optim.AdamW(p.net.parameters(), lr=args.lr, weight_decay=0.0)
        taps_v, taps_u = torch.arange(41, device=args.device), torch.arange(61, device=args.device)
        best_score, history = float("inf"), []
        out = args.out / (label + "_model")
        started = time.perf_counter()
        for step in range(1, args.steps + 1):
            ids = legal[torch.randint(len(legal), (args.batch,), device=args.device)]
            qc = q[ids]
            vh = input_v[ids[:, None] - taps_v]
            loss = torch.zeros((), device=args.device)
            for k in range(args.horizon):
                uh = u[ids[:, None] + k - taps_u]
                vn = p.next_velocity(qc, vh, uh)
                qc = qc + 0.01 * vn
                loss = loss + torch.nn.functional.smooth_l1_loss(vn / scale, target_v[ids + k + 1] / scale)
                loss = (
                    loss + 2 * ((qc - target_q[ids + k + 1]) / (scale * 0.01 * args.horizon)).square().mean()
                )
                vh = torch.cat((vn[:, None], vh[:, :-1]), 1)
            # Static rest prior is shared by both experiments.
            rv = p.next_velocity(
                q[ids], torch.zeros_like(vh), torch.zeros(args.batch, 61, 3, device=args.device)
            )
            loss = loss / args.horizon + 0.2 * (rv / scale).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.net.parameters(), 2.0)
            opt.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{label} training diverged at {step}")
            if step % args.eval_every == 0 or step == args.steps:
                # Save a disposable *candidate artifact*, never overwrite the baseline.
                save_candidate(p, args.model, out / "latest", step, label, {})
                metrics = evaluate(out / "latest", dev, args.device, args.cases)
                score = metrics.get("path10_deg", 1e12) + metrics.get("rest10_deg", 1e12)
                row = {
                    "step": step,
                    "loss": loss.item(),
                    "elapsed_s": time.perf_counter() - started,
                    "metrics": metrics,
                }
                history.append(row)
                if metrics["finite"] and score < best_score:
                    best_score = score
                    save_candidate(p, args.model, out, step, label, metrics)
                (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps({"label": label, **row}), flush=True)
                p.net.train()
            elif step % 25 == 0:
                print(json.dumps({"label": label, "step": step, "loss": loss.item()}), flush=True)
        selection["candidates"][label] = json.loads((out / "model_meta.json").read_text())["training"]
    gyro = selection["candidates"]["gyro"]["development"]
    control = selection["candidates"]["smooth"]["development"]
    # Floors avoid rejecting numerical roundoff around the baseline's exact neutral bound.
    accepted = gyro["finite"] and gyro["path10_deg"] < min(baseline["path10_deg"], control["path10_deg"])
    for metric, floor in (("rest10_deg", 0.1), ("stop_rate_rad_s", 0.005)):
        accepted &= gyro[metric] <= max(floor, baseline[metric] * 1.1)
    accepted &= gyro["30"]["endpoint_deg"] <= baseline["30"]["endpoint_deg"] * 1.1
    selection["selected"] = str((args.out / "gyro_model").resolve() if accepted else args.model.resolve())
    selection["gyro_accepted"] = bool(accepted)
    selection["rule"] = (
        "Improve 10s position vs baseline and matched control; <=10% 30s/stop/hold regression (floors .1deg/.005rad/s)"
    )
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"selected": selection["selected"], "gyro_accepted": bool(accepted)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--split", choices=("development", "test"), default="development")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--cases", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "train":
        if (args.out / "selection.json").exists():
            raise FileExistsError("Use a new --out for another experiment; the selection is frozen")
        train(args)
    else:
        selection = json.loads((args.out / "selection.json").read_text())
        split = "test" if args.split == "test" else "dev"
        paths = {
            "baseline": Path(selection["baseline"]),
            "gyro": args.out / "gyro_model",
            "smooth": args.out / "smooth_model",
        }
        results = {
            name: evaluate(path, load_chunks(args.out, split), args.device, args.cases)
            for name, path in paths.items()
        }
        (args.out / f"evaluation_{split}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
