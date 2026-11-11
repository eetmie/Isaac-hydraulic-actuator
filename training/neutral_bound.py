# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compile a closed-valve velocity bound into an actuator MLP ("steady" models).

For each arm joint j, the next velocity is clipped to [-c_j, c_j] with c_j = gain * sum |u_j| over the sampled
valve history the network sees. Once a joint's valve history clears, its velocity is exactly zero, so the model
cannot drift at rest; while valves move the bound is loose. Carriage pitch is unbounded.

The bound is built from ReLU relay neurons, so the result is an ordinary sequential MLP with the same
normalizers and input contract: no runtime changes. Rebuilt from the construction described for the historical
``arm_v5_steady`` (credit: the original ``models/codex/compile_neutral.py``). Output form
``d - relu(v + d - c) + relu(-v - d - c)`` returns ``d`` exactly while the bound is inactive.

    python -m training.neutral_bound SOURCE_MODEL OUT_DIR [--gain 5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

ARM = 3  # bounded joints: boom, arm, bucket


def _layout(meta: dict) -> tuple[int, list[int]]:
    """Input index of the current velocity row and of every sampled valve tap, per arm joint."""
    if meta["target_mode"] != "delta_velocity" or meta["model"]["activation"] != "relu":
        raise ValueError("The bound needs a ReLU delta-velocity actuator")
    joints = len(meta["q_cols"])
    v0 = joints * meta["hist_q"] if meta.get("include_q", True) else 0
    u0 = v0 + joints * len(range(0, meta["hist_qdot"], meta["qdot_stride"]))
    taps = meta["hist_u"]
    if u0 + taps * len(meta["u_cols"]) != meta["model"]["in_dim"]:
        raise ValueError("Unexpected actuator input layout")
    return v0, [[u0 + t * len(meta["u_cols"]) + j for t in range(taps)] for j in range(ARM)]


def compile_bound(state: dict, meta: dict, norm: dict, gain: float) -> tuple[dict, list[int]]:
    """Return the bounded state dict and hidden sizes. ``norm`` holds x_mean/x_std/y_mean/y_std arrays."""
    v0, taps = _layout(meta)
    linears = sorted({int(k.split(".")[1]) for k in state})
    W = [state[f"net.{i}.weight"].double() for i in linears]
    b = [state[f"net.{i}.bias"].double() for i in linears]
    if len(W) < 3:
        raise ValueError("Need at least two hidden layers")
    xm, xs = (torch.as_tensor(norm[k], dtype=torch.float64) for k in ("x_mean", "x_std"))
    ym, ys = (torch.as_tensor(norm[k], dtype=torch.float64) for k in ("y_mean", "y_std"))
    n_in, out = W[0].shape[1], W[-1].shape[0]
    n_taps = len(taps[0])

    # Layer 1: originals, then relu(+-v_j) and relu(+-u_j,t) on the raw (denormalized) inputs.
    rows, bias = [], []
    for j in range(ARM):
        for sign in (1, -1):
            r = torch.zeros(n_in, dtype=torch.float64)
            r[v0 + j] = sign * xs[v0 + j]
            rows.append(r)
            bias.append(sign * xm[v0 + j])
    for j in range(ARM):
        for i in taps[j]:
            for sign in (1, -1):
                r = torch.zeros(n_in, dtype=torch.float64)
                r[i] = sign * xs[i]
                rows.append(r)
                bias.append(sign * xm[i])
    layers = [(torch.cat((W[0], torch.stack(rows))), torch.cat((b[0], torch.stack(bias))))]
    extra = len(rows)

    # Layer 2: originals, relays [v+_j, v-_j] * 3, c_j = gain * sum(u+ + u-).
    relay = 2 * ARM + ARM
    h = W[1].shape[0]
    w = torch.zeros(h + relay, W[1].shape[1] + extra, dtype=torch.float64)
    w[:h, : W[1].shape[1]] = W[1]
    base = W[1].shape[1]
    for k in range(2 * ARM):
        w[h + k, base + k] = 1
    for j in range(ARM):
        start = base + 2 * ARM + 2 * n_taps * j
        w[h + 2 * ARM + j, start : start + 2 * n_taps] = gain
    layers.append((w, torch.cat((b[1], torch.zeros(relay, dtype=torch.float64)))))

    # Remaining hidden layers: originals plus identity relays.
    for Wi, bi in zip(W[2:-1], b[2:-1]):
        h, p = Wi.shape
        w = torch.zeros(h + relay, p + relay, dtype=torch.float64)
        w[:h, :p] = Wi
        w[h:, p:] = torch.eye(relay, dtype=torch.float64)
        layers.append((w, torch.cat((bi, torch.zeros(relay, dtype=torch.float64)))))

    # Clip layer: raw deltas d_k = ys_k (W h + b)_k + ym_k; per arm joint relu(+-d), relu(+-(v+d)-c); pitch relu(+-d).
    Wo, bo = W[-1], b[-1]
    p = Wo.shape[1]
    d_w = torch.cat((ys[:, None] * Wo, torch.zeros(out, relay, dtype=torch.float64)), dim=1)
    d_b = ys * bo + ym

    def v_row(j):
        r = torch.zeros(p + relay, dtype=torch.float64)
        r[p + 2 * j], r[p + 2 * j + 1] = 1, -1
        return r

    def c_row(j):
        r = torch.zeros(p + relay, dtype=torch.float64)
        r[p + 2 * ARM + j] = 1
        return r

    rows, bias, final = [], [], torch.zeros(out, 4 * ARM + 2 * (out - ARM), dtype=torch.float64)
    for k in range(out):
        rows += [d_w[k], -d_w[k]]
        bias += [d_b[k], -d_b[k]]
        col = len(rows) - 2
        final[k, col], final[k, col + 1] = 1, -1
        if k < ARM:
            rows += [d_w[k] + v_row(k) - c_row(k), -d_w[k] - v_row(k) - c_row(k)]
            bias += [d_b[k], -d_b[k]]
            final[k, col + 2], final[k, col + 3] = -1, 1
    layers.append((torch.stack(rows), torch.stack(bias)))
    # Output back in the stored normalization: y = (delta - ym) / ys.
    layers.append((final / ys[:, None], -ym / ys))

    state = {}
    for n, (w, bb) in enumerate(layers):
        state[f"net.{2 * n}.weight"] = w.float()
        state[f"net.{2 * n}.bias"] = bb.float()
    return state, [w.shape[0] for w, _ in layers[:-1]]


def compile_model(source: Path, out: Path, gain: float = 5.0) -> Path:
    """Write a bounded copy of the actuator in ``source`` to ``out``; normalizers are copied unchanged."""
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((source / "model_meta.json").read_text())
    norm = {k: np.load(source / f"{k}.npy") for k in ("x_mean", "x_std", "y_mean", "y_std")}
    state = torch.load(source / "mlp_state_dict.pt", map_location="cpu", weights_only=True)
    bounded, hidden = compile_bound(state, meta, norm, gain)
    out.mkdir(parents=True)
    for k in norm:
        shutil.copy2(source / f"{k}.npy", out / f"{k}.npy")
    torch.save(bounded, out / "mlp_state_dict.pt")
    meta["model"]["hidden"] = hidden
    meta.setdefault("training", {})["neutral_bound"] = {
        "gain": gain,
        "axes": list(range(ARM)),
        "rule": "|v_next[j]| <= gain * sum |u[j]| over the command history",
        "source": str(source),
        "source_state_sha256": hashlib.sha256((source / "mlp_state_dict.pt").read_bytes()).hexdigest(),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--gain", type=float, default=5.0)
    args = parser.parse_args()
    print(compile_model(args.source, args.out, args.gain))


if __name__ == "__main__":
    main()
