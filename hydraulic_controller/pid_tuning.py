"""Tune the robot's joint PID gains with CMA-ES on the learned plant.

Every candidate runs every scenario in one batch (``closed_loop.rollout``), so a generation of 32 gain sets over
~600 scenarios is a single 700-tick rollout. The search runs in log-gain space inside a box, starting from the
robot's current gains. Only scenarios a perfect controller could follow take part: the reference path stays
inside the limits and the plant can reach its speed (``Prepared.feasible``). Reach edges, where tracking speed
dies, therefore drop out on their own.

The score is a weighted sum of the ``rollout`` cost terms, averaged per plant perturbation and mixed with the
worst plant, so gains that only suit the nominal model lose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .closed_loop import COST_TERMS, rollout
from .pid import JointPIDController, PidGains
from .tasks import PID_JOINTS, Prepared

GAIN_NAMES = tuple(f"{joint}_{term}" for joint in PID_JOINTS for term in ("kp", "ki", "kd"))


class CMAES:
    """Plain (mu/mu_w, lambda)-CMA-ES with rank-one and rank-mu updates (Hansen, "The CMA Evolution Strategy:
    A Tutorial", 2016), minimizing."""

    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        n = len(x0)
        self.n, self.mean, self.sigma = n, np.asarray(x0, dtype=float), float(sigma0)
        self.popsize = popsize or 4 + int(3 * math.log(n))
        self.mu = self.popsize // 2
        w = math.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / (self.weights**2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.generation = 0
        self.rng = np.random.default_rng(seed)

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.popsize, self.n))
        return self.mean + self.sigma * (z * self.D) @ self.B.T

    def tell(self, x: np.ndarray, cost: np.ndarray) -> None:
        order = np.argsort(cost)
        best = x[order[: self.mu]]
        old = self.mean
        self.mean = self.weights @ best
        y = (self.mean - old) / self.sigma
        inv_sqrt_c = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * inv_sqrt_c @ y
        norm_ps = np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
        hsig = norm_ps / self.chi_n < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
        steps = (best - old) / self.sigma
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * (steps.T * self.weights) @ steps
        )
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.C = (self.C + self.C.T) / 2
        eigenvalues, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.clip(eigenvalues, 1e-20, None))
        self.generation += 1


@dataclass(frozen=True)
class TuneConfig:
    """Search space (gain bounds per kp/ki/kd), budget and score weights."""

    bounds: dict = field(default_factory=lambda: {"kp": (0.5, 60.0), "ki": (1e-3, 20.0), "kd": (1e-3, 3.0)})
    popsize: int = 32
    generations: int = 40
    sigma0: float = 0.6  # in log-gain units: one sigma is a factor of ~1.8
    seed: int = 0
    weights: dict = field(
        default_factory=lambda: {
            "track": 1.0,
            "final": 2.0,
            "overshoot": 1.0,
            "tail_p2p": 1.0,
            "valve_travel": 0.3,
            "invalid": 100.0,
        }
    )
    worst_plant_weight: float = 0.5
    out_of_box_penalty: float = 10.0


def bounds_arrays(cfg: TuneConfig) -> tuple[np.ndarray, np.ndarray]:
    """Log-gain box in ``GAIN_NAMES`` order."""
    lo = np.log([cfg.bounds[name.split("_")[1]][0] for name in GAIN_NAMES])
    hi = np.log([cfg.bounds[name.split("_")[1]][1] for name in GAIN_NAMES])
    return lo, hi


def to_vector(gains: PidGains, cfg: TuneConfig) -> np.ndarray:
    """Log-gain vector of ``gains``, clipped into the box (a zero gain lands on its lower bound)."""
    lo, hi = bounds_arrays(cfg)
    raw = np.array([[gains.kp[j], gains.ki[j], gains.kd[j]] for j in range(3)]).ravel()
    return np.clip(np.log(np.maximum(raw, 1e-12)), lo, hi)


def to_gains(vectors: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """kp, ki, kd tensors [C, 3] from log-gain vectors [C, 9]."""
    g = torch.tensor(np.exp(vectors), dtype=torch.float32, device=device).reshape(len(vectors), 3, 3)
    return g[:, :, 0], g[:, :, 1], g[:, :, 2]


def score(prep: Prepared, terms: dict[str, torch.Tensor], cfg: TuneConfig) -> torch.Tensor:
    """Scalar cost per candidate [C]: weighted terms, mean per plant, mixed with the worst plant."""
    per_scenario = sum(cfg.weights[key] * terms[key] for key in COST_TERMS)  # [C, S]
    plants = sorted({s.plant for s in prep.scenarios})
    index = torch.tensor([plants.index(s.plant) for s in prep.scenarios], device=per_scenario.device)
    per_plant = torch.stack([per_scenario[:, index == p].mean(1) for p in range(len(plants))], dim=1)
    return (1 - cfg.worst_plant_weight) * per_plant.mean(1) + cfg.worst_plant_weight * per_plant.amax(1)


def evaluate(prep: Prepared, vectors: np.ndarray, base: PidGains, cfg: TuneConfig, seed: int):
    """Cost [C] and cost terms of log-gain candidates; out-of-box candidates are clipped and penalized."""
    lo, hi = bounds_arrays(cfg)
    clipped = np.clip(vectors, lo, hi)
    controller = JointPIDController(base, *to_gains(clipped, prep.q0.device))
    terms, _ = rollout(prep, controller, copies=len(vectors), seed=seed)
    cost = score(prep, terms, cfg).cpu().numpy()
    cost += cfg.out_of_box_penalty * ((vectors - clipped) ** 2).sum(1)
    return cost, terms


def vector_to_pid_gains(vector: np.ndarray, base: PidGains, digits: int = 3) -> PidGains:
    """Round to ``digits`` significant figures for the robot YAML."""
    g = np.exp(vector).reshape(3, 3)

    def round_sig(value):
        return float(f"{value:.{digits}g}")

    return PidGains(
        kp=[round_sig(v) for v in g[:, 0]],
        ki=[round_sig(v) for v in g[:, 1]],
        kd=[round_sig(v) for v in g[:, 2]],
        deriv_filter_tau=list(base.deriv_filter_tau),
        output_limits=base.output_limits,
        ik_lambda=base.ik_lambda,
    )


def tune(prep: Prepared, base: PidGains, cfg: TuneConfig, log=print) -> dict:
    """Run CMA-ES from ``base``; returns the history and the best (rounded) gains."""
    es = CMAES(to_vector(base, cfg), cfg.sigma0, cfg.popsize, cfg.seed)
    base_cost, _ = evaluate(prep, to_vector(base, cfg)[None], base, cfg, seed=cfg.seed)
    best_vector, best_cost = to_vector(base, cfg), float(base_cost[0])
    history = []
    log(f"[tune] baseline cost {best_cost:.3f}; {len(prep.scenarios)} scenarios x {es.popsize} candidates")
    for generation in range(cfg.generations):
        candidates = es.ask()
        # One noise seed per generation, shared by all candidates (common random numbers).
        cost, _ = evaluate(prep, candidates, base, cfg, seed=cfg.seed + 1 + generation)
        es.tell(candidates, cost)
        i = int(np.argmin(cost))
        if cost[i] < best_cost:
            best_cost, best_vector = float(cost[i]), np.clip(candidates[i], *bounds_arrays(cfg))
        history.append(
            {
                "generation": generation,
                "best": float(cost[i]),
                "median": float(np.median(cost)),
                "best_so_far": best_cost,
                "sigma": es.sigma,
                **dict(zip(GAIN_NAMES, np.exp(candidates[i]).tolist())),
            }
        )
        log(
            f"[tune] gen {generation:3d}  best {cost[i]:.3f}  median {np.median(cost):.3f}  "
            f"best-so-far {best_cost:.3f}  sigma {es.sigma:.3f}"
        )
    tuned = vector_to_pid_gains(best_vector, base)
    return {"history": history, "baseline_cost": float(base_cost[0]), "best_cost": best_cost, "tuned": tuned}


def breakdown(prep: Prepared, gains: PidGains, cfg: TuneConfig, seed: int) -> tuple[float, dict]:
    """Held-out score (fresh noise seed) and mean cost terms per family, nominal vs worst plant."""
    vector = to_vector(gains, cfg)[None]
    cost, terms = evaluate(prep, vector, gains, cfg, seed=seed)
    families = sorted({s.family for s in prep.scenarios})
    plants = sorted({s.plant for s in prep.scenarios})
    table = {}
    for family in families:
        for key in COST_TERMS:
            values = {}
            for plant in plants:
                mask = torch.tensor(
                    [s.family == family and s.plant == plant for s in prep.scenarios], device=prep.q0.device
                )
                values[plant] = float(terms[key][0, mask].mean())
            table[(family, key)] = {"nominal": values.get("nominal", math.nan), "worst": max(values.values())}
    return float(cost[0]), table
