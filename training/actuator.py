# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Train and evaluate the actuator twin on frozen canonical splits.

Prepare collection-group splits with ``python -m training.prepare``;
then run ``python -m training.actuator train|evaluate``. Targets are the
same-packet gyro rates from the complete dataset, without label smoothing.
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
from hydraulic_controller.sensors import SENSOR_CONTRACT

DEFAULT_OUT = ROOT / "runs/actuator"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_chunks(out, split):
    """Read exactly the immutable chunks assigned by the canonical split manifest."""
    manifest = json.loads((out / "split_manifest.json").read_text())
    expected = {
        Path(row["source"]).name: row["sha256"] for row in manifest["chunks"] if row["split"] == split
    }
    paths = {path.name: path for path in (out / "data" / split).glob("*.npz")}
    if not expected:
        raise FileNotFoundError(f"Prepare {split} data with training.prepare first")
    if paths.keys() != expected.keys():
        raise ValueError(f"{split} files differ from the frozen split manifest")
    chunks = []
    for name in sorted(expected):
        if digest(paths[name]) != expected[name]:
            raise ValueError(f"Prepared chunk checksum mismatch: {name}")
        with np.load(paths[name], allow_pickle=False) as data:
            chunks.append({**{key: data[key] for key in data.files}, "name": paths[name].stem})
    return chunks


class Predictor:
    def __init__(self, path, device):
        src = HydraulicActuatorNet(path, device=device, sim_dt=0.01)
        if (src.hist_q, src.hist_qdot, src.qdot_stride, src.hist_u, src.u_stride) != (1, 41, 1, 21, 3):
            raise ValueError("Canonical actuator training requires the actuator history contract")
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


def init_scratch(p, chunks, args):
    """Re-initialise the network and fit its normalisers to the prepared data, then regress one step.

    The historical baseline weights and normalisers were fitted to smoothed labels; a scratch model sees only the
    prepared targets. One-step teacher-forced regression gives the rollout phase a usable start.
    """
    device = args.device
    for module in p.net.modules():
        if isinstance(module, torch.nn.Linear):
            module.reset_parameters()

    def tensor(key):
        return torch.tensor(np.concatenate([c[key] for c in chunks]), device=device)

    q, v, u = tensor("q"), tensor("target_v"), tensor("u")
    legal, offset = [], 0
    for c in chunks:
        legal.extend(range(offset + 61, offset + len(c["q"]) - 1))
        offset += len(c["q"])
    legal = torch.tensor(legal, device=device)
    taps_v, taps_u = torch.arange(41, device=device), torch.arange(61, device=device)

    def batch(ids):
        x = torch.cat(
            (q[ids], v[ids[:, None] - taps_v].flatten(1), u[ids[:, None] - taps_u][:, ::3].flatten(1)), 1
        )
        return x, v[ids + 1] - v[ids]

    sample = legal[torch.randperm(len(legal), device=device)[:200_000]]
    x, y = batch(sample)
    src = p.src
    src._x_mean, src._x_std = x.mean(0), x.std(0).clamp_min(1e-4)
    src._y_mean, src._y_std = y.mean(0), y.std(0).clamp_min(1e-5)
    opt = torch.optim.AdamW(p.net.parameters(), lr=args.pretrain_lr, weight_decay=1e-5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.pretrain_steps, eta_min=args.pretrain_lr * 0.01
    )
    p.net.train()
    for step in range(1, args.pretrain_steps + 1):
        x, y = batch(legal[torch.randint(len(legal), (2048,), device=device)])
        loss = torch.nn.functional.mse_loss(
            p.net((x - src._x_mean) / src._x_std), (y - src._y_mean) / src._y_std
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        if step % 1000 == 0:
            print(json.dumps({"label": "pretrain", "step": step, "loss": loss.item()}), flush=True)


def save_candidate(predictor, original, out, step, label, metrics):
    out.mkdir(parents=True, exist_ok=True)
    src = predictor.src
    meta = json.loads((original / "model_meta.json").read_text())
    meta["version"] = "actuator"
    meta["training"] = {
        "parent": str(original),
        "input_velocity": "gyro",
        "target_velocity": "gyro",
        "sensor_contract": SENSOR_CONTRACT,
        "step": step,
        "development": metrics,
        "test_seen": False,
        "note": "Canonical raw-gyro training; parent's compiled bound is not guaranteed",
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
    """Train a raw-gyro candidate and compare it to the frozen baseline."""
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
    split_path = args.out / "split_manifest.json"
    split = json.loads(split_path.read_text())
    selection["dataset_manifest_sha256"] = split["dataset_manifest_sha256"]
    selection["split_manifest_sha256"] = digest(split_path)
    selection["budget"] = {k: str(v) if isinstance(v, Path) else v for k, v in selection["budget"].items()}
    label = "actuator"
    torch.manual_seed(args.seed)
    p = Predictor(args.model, args.device)
    p.net.train().requires_grad_(True)
    if args.scratch:
        init_scratch(p, chunks, args)

    def tensor(key):
        return torch.tensor(np.concatenate([c[key] for c in chunks]), device=args.device)

    q, v, target_q, target_v, u = [tensor(k) for k in ("q", "v", "target_q", "target_v", "u")]
    legal, offset = [], 0
    for c in chunks:
        legal.extend(range(offset + 61, offset + len(c["q"]) - args.horizon - 1))
        offset += len(c["q"])
    legal = torch.tensor(legal, device=args.device)
    scale = target_v.std(0).clamp_min(0.003)
    opt = torch.optim.AdamW(p.net.parameters(), lr=args.lr, weight_decay=0.0)
    taps_v, taps_u = torch.arange(41, device=args.device), torch.arange(61, device=args.device)
    best_score, history = float("inf"), []
    out = args.out / "model"
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = legal[torch.randint(len(legal), (args.batch,), device=args.device)]
        qc = q[ids]
        vh = v[ids[:, None] - taps_v]
        loss = torch.zeros((), device=args.device)
        for k in range(args.horizon):
            uh = u[ids[:, None] + k - taps_u]
            vn = p.next_velocity(qc, vh, uh)
            qc = qc + 0.01 * vn
            loss = loss + torch.nn.functional.smooth_l1_loss(vn / scale, target_v[ids + k + 1] / scale)
            loss = loss + 2 * ((qc - target_q[ids + k + 1]) / (scale * 0.01 * args.horizon)).square().mean()
            vh = torch.cat((vn[:, None], vh[:, :-1]), 1)
        # Penalize drift at rest alongside the measured motion loss.
        rv = p.next_velocity(q[ids], torch.zeros_like(vh), torch.zeros(args.batch, 61, 3, device=args.device))
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
    candidate = selection["candidates"][label]["development"]
    # Floors avoid rejecting numerical roundoff around the baseline's exact neutral bound.
    accepted = candidate["finite"] and candidate["path10_deg"] < baseline["path10_deg"]
    for metric, floor in (("rest10_deg", 0.1), ("stop_rate_rad_s", 0.005)):
        accepted &= candidate[metric] <= max(floor, baseline[metric] * 1.1)
    accepted &= candidate["30"]["endpoint_deg"] <= baseline["30"]["endpoint_deg"] * 1.1
    selection["selected"] = str(out.resolve() if accepted else args.model.resolve())
    selection["selected_fingerprint"] = model_fingerprint(Path(selection["selected"]))
    selection["candidate_accepted"] = bool(accepted)
    selection["rule"] = (
        "Improve 10s position vs baseline; <=10% 30s/stop/hold regression (floors .1deg/.005rad/s)"
    )
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"selected": selection["selected"], "candidate_accepted": bool(accepted)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
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
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Start from random weights and normalisers fitted to the prepared data, not from --model",
    )
    parser.add_argument(
        "--pretrain_steps", type=int, default=8000, help="One-step regression updates for --scratch"
    )
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        if (args.out / "selection.json").exists():
            raise FileExistsError("Use a new --out for another experiment; the selection is frozen")
        train(args)
    else:
        selection = json.loads((args.out / "selection.json").read_text())
        split = "test" if args.split == "test" else "dev"
        paths = {
            "baseline": Path(selection["baseline"]),
            "candidate": args.out / "model",
        }
        paths = {name: path for name, path in paths.items() if name == "baseline" or path.exists()}
        results = {
            name: evaluate(path, load_chunks(args.out, split), args.device, args.cases)
            for name, path in paths.items()
        }
        (args.out / f"evaluation_{split}.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
