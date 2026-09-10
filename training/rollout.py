"""Batched free-running rollout in torch, for model selection during training.

At 100 Hz ``qdot(t+dt) ~= qdot(t)``, so a net can win on one-step MSE by leaning
on its own past velocity and still drift once it eats its own predictions. On
this data the checkpoint with the best one-step R2 was 21% worse on free-running
position error, so training selects on the rollout instead.

This is the torch counterpart of ``eval.Rollout.free_run_batch``: same
recurrence, run against the in-training model with no disk round trip, which is
what makes per-epoch scoring affordable. ``test_contract.py`` asserts the two
agree -- if they drift, training optimizes a metric the reports do not measure.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from dataset import WindowSpec


def make_starts(
    val_ranges: Sequence[Tuple[int, int]],
    spec: WindowSpec,
    horizon: int,
    n_starts: int,
    seed: int = 0,
) -> np.ndarray:
    """Pick fixed rollout start indices inside the validation ranges.

    A start needs ``history_samples`` windows of run-up to seed its buffers and
    ``horizon`` windows of future to score against. Both must stay inside the
    same validation range: reaching back into a neighbouring range would seed a
    validation rollout with training data.

    The starts are drawn once and reused every evaluation, so the metric moves
    only when the model does.
    """
    lead = spec.history_samples
    candidates: List[np.ndarray] = []
    for start, stop in val_ranges:
        first = start + lead
        last = stop - horizon - 1
        if last > first:
            candidates.append(np.arange(first, last))
    if not candidates:
        raise ValueError(
            "No validation range is long enough for a rollout. "
            "Shorten --rollout-horizon-sec or use longer recordings."
        )

    pool = np.concatenate(candidates)
    rng = np.random.default_rng(seed)
    if len(pool) <= n_starts:
        return np.sort(pool)
    return np.sort(rng.choice(pool, size=n_starts, replace=False))


class RolloutScorer:
    """Scores a model by final position error over free-running rollouts.

    Everything the rollout needs is uploaded once at construction, so scoring is
    a pure forward pass with no host traffic.
    """

    def __init__(
        self,
        spec: WindowSpec,
        q: np.ndarray,
        qdot: np.ndarray,
        u: np.ndarray,
        starts: np.ndarray,
        horizon: int,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        device: str,
    ):
        self.spec = spec
        self.horizon = horizon
        self.device = device
        self.n_starts = len(starts)

        nq = spec.hist_q
        nv = (spec.hist_qdot - 1) * spec.qdot_stride + 1
        nu = (spec.hist_u - 1) * spec.u_stride + 1
        self._buf_sizes = (nq, nv, nu)

        def tens(a: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a)).float().to(device)

        # Seed buffers: index k holds the tap k steps into the past.
        self.qbuf0 = tens(np.stack([q[starts - k] for k in range(nq)], axis=1))
        self.vbuf0 = tens(np.stack([qdot[starts - k] for k in range(nv)], axis=1))
        self.ubuf0 = tens(np.stack([u[starts - 1 - k] for k in range(nu)], axis=1))
        self.q0 = tens(q[starts])
        # Real commands drive the rollout; only the state is fed back.
        self.u_seq = tens(np.stack([u[s : s + horizon] for s in starts], axis=0))
        self.q_true = tens(q[starts + horizon])

        self.x_mean, self.x_std = tens(x_mean), tens(x_std)
        self.y_mean, self.y_std = tens(y_mean), tens(y_std)

    @torch.no_grad()
    def score(self, model: torch.nn.Module) -> float:
        """Mean absolute position error [rad] at the rollout horizon."""
        spec = self.spec
        was_training = model.training
        model.eval()

        qbuf = self.qbuf0.clone()
        vbuf = self.vbuf0.clone()
        ubuf = self.ubuf0.clone()
        qc = self.q0.clone()
        B = qc.shape[0]

        for s in range(self.horizon):
            ubuf = torch.roll(ubuf, 1, dims=1)
            ubuf[:, 0] = self.u_seq[:, s]

            parts = [qbuf.reshape(B, -1)] if spec.include_q else []
            if spec.hist_qdot > 0:
                parts.append(vbuf[:, :: spec.qdot_stride].reshape(B, -1))
            parts.append(ubuf[:, :: spec.u_stride].reshape(B, -1))
            x = torch.cat(parts, dim=1)

            delta = model((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean
            qdot_next = vbuf[:, 0] + delta if spec.target_mode == "delta_velocity" else delta
            qc = qc + spec.dt * qdot_next

            qbuf = torch.roll(qbuf, 1, dims=1)
            qbuf[:, 0] = qc
            if spec.hist_qdot > 0:
                vbuf = torch.roll(vbuf, 1, dims=1)
                vbuf[:, 0] = qdot_next

        if was_training:
            model.train()
        return float((qc - self.q_true).abs().mean().item())
