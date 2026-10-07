"""Score several controllers on the same closed-loop tip tasks and report them side by side.

Every controller runs the same feasible tip lines (``tasks``) through ``closed_loop.rollout``. The summary
reports plain units: tip error in mm, bucket angle error in deg, overshoot past the line end in mm, valve travel
per second, and failures. Each figure is given over all plants and for the nominal and worst plant, plus per
commanded speed.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch

from .closed_loop import rollout
from .tasks import Prepared

LABELS = {  # display names for the usual controllers; anything else shows as given
    "pid_robot": "PID (robot gains)",
    "pid_tuned": "PID (sim-tuned)",
    "mpc": "MPC (MPPI)",
    "proto": "MLP (prototype)",
    "mlp": "MLP",
    "v6": "MLP (V6)",
}

METRICS = {  # summary name: (rollout term, scale, how to pool over scenarios)
    "tip_mean_mm": ("tip_mean_mm", 1.0, "mean"),
    "tip_max_mm": ("tip_max_mm", 1.0, "max"),
    "tip_p95_max_mm": ("tip_max_mm", 1.0, "p95"),
    "tip_final_mm": ("tip_final_mm", 1.0, "mean"),
    "pitch_mean_deg": ("pitch_mean_deg", 1.0, "mean"),
    "overshoot_mm": ("overshoot", 10.0, "mean"),  # rollout reports tip overshoot in mm / 10
    "valve_travel_per_s": ("valve_travel", 1.0, "mean"),
    "failures": ("invalid", 1.0, "sum"),
}


def pool(values: torch.Tensor, how: str) -> float:
    values = values[torch.isfinite(values)]
    if len(values) == 0:
        return math.nan
    if how == "mean":
        return float(values.mean())
    if how == "max":
        return float(values.max())
    if how == "p95":
        return float(torch.quantile(values, 0.95))
    return float(values.sum())


def summarize(prep: Prepared, terms: dict[str, torch.Tensor]) -> dict:
    """Summary metrics over all scenarios, per plant (nominal / worst) and per commanded speed."""
    scenarios = prep.scenarios
    out = {"all": {}, "nominal": {}, "worst_plant": {}, "by_speed_mm_s": {}}
    plants = sorted({s.plant for s in scenarios})
    for name, (term, scale, how) in METRICS.items():
        values = terms[term][0].cpu() * scale
        out["all"][name] = pool(values, how)
        per_plant = {plant: pool(values[[s.plant == plant for s in scenarios]], how) for plant in plants}
        out["nominal"][name] = per_plant.get("nominal", math.nan)
        out["worst_plant"][name] = max(per_plant.values())
    for speed in sorted({s.size for s in scenarios}):
        mask = [s.size == speed for s in scenarios]
        out["by_speed_mm_s"][f"{speed * 1000:g}"] = {
            name: pool(terms[term][0].cpu()[mask] * scale, how)
            for name, (term, scale, how) in METRICS.items()
        }
    return out


@torch.no_grad()
def compare(prep: Prepared, controllers: dict, seed: int = 0, log=print) -> tuple[dict, dict, dict]:
    """Run each controller once over ``prep``; returns (summaries, per-scenario terms, traces) by name."""
    import time

    summaries, all_terms, all_traces = {}, {}, {}
    for name, controller in controllers.items():
        start = time.perf_counter()
        terms, traces = rollout(prep, controller, record=True, seed=seed)
        summaries[name] = summarize(prep, terms)
        all_terms[name], all_traces[name] = {k: v[0].cpu() for k, v in terms.items()}, traces
        log(f"[compare] {name:10s} {time.perf_counter() - start:6.1f} s")
    return summaries, all_terms, all_traces


def write_report(out_dir: Path, prep: Prepared, summaries: dict, terms: dict, info: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({"summaries": summaries, **info}, indent=1))
    keys = list(next(iter(terms.values())))
    with open(out_dir / "per_scenario.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["controller", "scenario", "plant", "start", "speed_mm_s", *keys])
        for name, values in terms.items():
            for i, s in enumerate(prep.scenarios):
                writer.writerow(
                    [name, s.name, s.plant, s.start, s.size * 1000, *(float(values[k][i]) for k in keys)]
                )


def plot_report(
    path: Path, prep: Prepared, summaries: dict, traces: dict, showcase_speed: float = 0.04
) -> None:
    """Headline bars, per-speed tip error, and tip-error / valve traces of one line per direction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    palette = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    colors = {n: palette(i % palette.N) for i, n in enumerate(names)}
    fig = plt.figure(figsize=(17, 13), constrained_layout=True)
    grid = fig.add_gridspec(4, 4)

    bars = ("tip_mean_mm", "tip_p95_max_mm", "tip_final_mm", "valve_travel_per_s")
    for i, metric in enumerate(bars):
        ax = fig.add_subplot(grid[0, i])
        x = torch.arange(len(names)).numpy()
        ax.bar(
            x - 0.2,
            [summaries[n]["nominal"][metric] for n in names],
            0.4,
            label="nominal plant",
            color=[colors[n] for n in names],
        )
        ax.bar(
            x + 0.2,
            [summaries[n]["worst_plant"][metric] for n in names],
            0.4,
            label="worst plant",
            color=[colors[n] for n in names],
            alpha=0.45,
        )
        ax.set_xticks(x, names, rotation=20)
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="y")
    fig.axes[0].legend(fontsize=7)

    ax = fig.add_subplot(grid[1, :2])
    for n in names:
        speeds = summaries[n]["by_speed_mm_s"]
        ax.plot(
            [float(v) for v in speeds],
            [speeds[v]["tip_mean_mm"] for v in speeds],
            "o-",
            color=colors[n],
            label=n,
        )
    ax.set(
        xlabel="commanded tip speed [mm/s]",
        ylabel="mean tip error [mm]",
        title="tracking vs speed (all plants)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[1, 2:])
    for n in names:
        speeds = summaries[n]["by_speed_mm_s"]
        ax.plot(
            [float(v) for v in speeds],
            [speeds[v]["valve_travel_per_s"] for v in speeds],
            "o-",
            color=colors[n],
        )
    ax.set(xlabel="commanded tip speed [mm/s]", ylabel="valve travel [1/s]", title="valve activity vs speed")
    ax.grid(alpha=0.3)

    steps = next(iter(traces.values()))["u"].shape[0]
    t = (torch.arange(1, steps + 1) * 0.01 - prep.cfg.lead_s).numpy()
    for col, direction in enumerate(("+X", "-X", "+Z", "-Z")):
        picks = [
            i
            for i, s in enumerate(prep.scenarios)
            if s.joint == direction and s.plant == "nominal" and abs(s.size - showcase_speed) < 1e-9
        ]
        if not picks:
            continue
        i = picks[0]
        ax, vx = fig.add_subplot(grid[2, col]), fig.add_subplot(grid[3, col])
        for n in names:
            err = (traces[n]["tip"][:, i, :2] - traces[n]["tip_target"][:, i, :2]).norm(dim=1) * 1000
            ax.plot(t, err.numpy(), color=colors[n], lw=1, label=n)
            vx.plot(t, traces[n]["u"][:, i].abs().sum(1).numpy(), color=colors[n], lw=0.8)
        ax.set(title=f"{prep.scenarios[i].name} (nominal)", ylabel="tip error [mm]", xlabel="s")
        vx.set(ylabel="sum |valve|", xlabel="s")
        ax.grid(alpha=0.3)
        vx.grid(alpha=0.3)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def circle_deviation(prep: Prepared, traces: dict) -> torch.Tensor:
    """Per-tick radial deviation |dist(tip, center) - radius| [mm] of each circle scenario once it starts, [T', N].

    Timing-free: a tip that lags along the circle but stays on it is not charged (cf. the tip error terms).
    """
    radius = prep.cfg.circle_radius_m
    steps = traces["tip"].shape[0]
    t = torch.arange(1, steps + 1) * 0.01 - prep.cfg.lead_s
    center = traces["tip_target"][0, :, :2].clone()
    center[:, 0] -= radius
    distance = (traces["tip"][t >= 0, :, :2] - center).norm(dim=-1)
    return (distance - radius).abs() * 1000


def summarize_circles(prep: Prepared, traces: dict) -> dict:
    """Mean and worst radial deviation [mm] per sense, nominal plant and worst plant."""
    out = {}
    for sense in ("ccw", "cw"):
        per_plant = {}
        for plant in sorted({s.plant for s in prep.scenarios}):
            rows = [i for i, s in enumerate(prep.scenarios) if s.joint == sense and s.plant == plant]
            if rows:
                dev = circle_deviation(prep, traces)[:, rows]
                per_plant[plant] = {"mean_mm": float(dev.mean()), "worst_mm": float(dev.max())}
        out[sense] = {
            "nominal": per_plant.get("nominal"),
            "worst_plant": max(per_plant.values(), key=lambda v: v["mean_mm"]) if per_plant else None,
        }
    return out


def plot_circles(path: Path, prep: Prepared, traces: dict, speed_mm_s: float) -> None:
    """One column per controller, counterclockwise above clockwise; every start overlaid about the circle center."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    names = list(traces)
    radius_mm = prep.cfg.circle_radius_m * 1000
    starts = len({s.start for s in prep.scenarios})
    fig, axes = plt.subplots(2, len(names), figsize=(4.3 * len(names), 10.6), squeeze=False)
    fig.suptitle(f"{2 * radius_mm:g} mm circles at {speed_mm_s:g} mm/s, nominal plant", fontsize=15)
    fig.text(
        0.5,
        0.925,
        f"{starts} start poses per direction overlaid about the circle center; dashed = reference, dot = start",
        ha="center",
        fontsize=10,
    )
    angle = torch.linspace(0, 2 * torch.pi, 361)
    for col, name in enumerate(names):
        deviation = circle_deviation(prep, traces[name])
        for row, (sense, label, color) in enumerate(
            (("ccw", "Counterclockwise", "tab:blue"), ("cw", "Clockwise", "tab:red"))
        ):
            ax = axes[row, col]
            rows = [i for i, s in enumerate(prep.scenarios) if s.joint == sense and s.plant == "nominal"]
            center = traces[name]["tip_target"][0, rows, :2].clone()
            center[:, 0] -= prep.cfg.circle_radius_m
            for k, i in enumerate(rows):
                xy = (traces[name]["tip"][:, i, :2] - center[k]) * 1000
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.1, alpha=0.75)
            ax.plot(radius_mm * angle.cos(), radius_mm * angle.sin(), "k--", lw=1.2)
            ax.plot([radius_mm], [0], "ko", ms=5)
            sweep = 1 if sense == "ccw" else -1
            ax.add_patch(
                FancyArrowPatch(
                    (0.33 * radius_mm, -0.22 * radius_mm * sweep),
                    (0.33 * radius_mm, 0.22 * radius_mm * sweep),
                    connectionstyle=f"arc3,rad={0.35 * sweep}",
                    arrowstyle="-|>",
                    mutation_scale=14,
                    color="gray",
                    lw=1.5,
                )
            )
            dev = deviation[:, rows]
            title = f"{LABELS.get(name, name)}: {label}"
            ax.set_title(
                f"{title}\nmean deviation {dev.mean():.1f} mm, worst {dev.max():.1f} mm", fontsize=10
            )
            lim = radius_mm * 1.25
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
            ax.set_xlabel("X from circle center [mm]", fontsize=8)
            ax.set_ylabel("Z from circle center [mm]", fontsize=8)
            ax.grid(alpha=0.3)
    fig.text(
        0.5,
        0.01,
        "Simulation on the learned actuator model; not yet validated on hardware.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.91), h_pad=3.0)
    fig.savefig(path, dpi=130)
    plt.close(fig)
