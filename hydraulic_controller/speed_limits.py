"""Achievable joint and tip speeds of the learned hydraulic plant, independent of any controller.

At each pose the joints are pinned and constant valve commands run until the velocity settles, so every number
includes the deadband, flow saturation and flow sharing between joints. A tip direction counts as achieved when
the steady tip twist points that way within ``max_cross`` and turns the bucket by at most ``max_pitch_per_m``
per metre travelled.

The valves for a requested tip speed are solved, not guessed. J^-1 turns the twist into joint speeds. Each spool
is commanded through the inverse of its own steady curve at that pose, which absorbs the deadband and the steep
rise behind it. The speed fed to that inverse is corrected by ``gain * (wanted - actual)`` until it settles on the
steady command a perfect integral controller would hold, flow sharing included. The fastest speed on a geometric ladder that still
meets the tolerances is the achievable speed: a lower bound on what any controller can reach. A broad sample of
fixed valve combinations backs this up. The single-joint bound (every joint at its own full-valve speed at once)
is reported beside it, and the gap between the two is what flow sharing costs.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .benchmark import PLANTS
from .core import HOME, HydraulicPlant, sha256

PID_JOINTS = ("boom", "arm", "bucket")


@dataclass(frozen=True)
class SpeedMapConfig:
    """Workspace grid [m], held bucket angle [rad], directions [deg from +X toward +Z] and settling [steps]."""

    x_range: tuple[float, float] = (0.25, 0.95)
    z_range: tuple[float, float] = (-0.40, 0.30)
    spacing: float = 0.05
    pitch: float | None = None  # held bucket angle; None = the HOME angle
    directions_deg: tuple[float, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
    valve_levels: tuple[float, ...] = (-1.0, -0.7, -0.5, -0.4, 0.0, 0.4, 0.5, 0.7, 1.0)
    random_valves: int = 1024
    settle_steps: int = 60
    average_steps: int = 20
    max_cross: float = 0.10  # sideways tip speed / speed along the direction
    max_pitch_per_m: float = 0.5  # bucket rotation [rad] per metre of tip travel
    curve_points: int = 401  # dense: speed jumps tenfold within 0.1 of opening just past the deadband
    speed_ladder_mm_s: tuple[float, float, int] = (
        5.0,
        400.0,
        40,
    )  # geometric, ~12% per rung: first, last, count
    solve_iterations: int = 12
    solve_gain: float = 0.6
    plant: str = "nominal"
    batch: int = 131072
    seed: int = 0


def valve_samples(cfg: SpeedMapConfig, device: str) -> torch.Tensor:
    """Grid of valve levels for all three spools, plus uniform random combinations, shape [S, 3]."""
    grid = torch.tensor(list(itertools.product(cfg.valve_levels, repeat=3)), dtype=torch.float32)
    generator = torch.Generator().manual_seed(cfg.seed)
    random = torch.rand(cfg.random_valves, 3, generator=generator) * 2 - 1
    return torch.cat((grid, random)).to(device)


@torch.no_grad()
def steady_joint_speeds(
    model: Path, kin, poses: torch.Tensor, valves: torch.Tensor, cfg: SpeedMapConfig
) -> torch.Tensor:
    """Steady joint velocity [rad/s] for every (pose, valve) pair with the pose pinned, shape [P, S, 3]."""
    p, s = len(poses), len(valves)
    flat_q = poses[:, None, :].expand(p, s, 4).reshape(-1, 4)
    flat_u = valves[None].expand(p, s, 3).reshape(-1, 3)
    return paired_steady_speeds(model, kin, flat_q, flat_u, cfg).reshape(p, s, 3)


@torch.no_grad()
def paired_steady_speeds(model: Path, kin, flat_q: torch.Tensor, flat_u: torch.Tensor, cfg) -> torch.Tensor:
    """Steady joint velocity [rad/s] of row i's valves held at row i's pinned pose, shape [N, 3]."""
    device = flat_q.device
    out = torch.empty(len(flat_q), 3, device=device)
    perturbation = PLANTS[cfg.plant]
    for start in range(0, len(flat_q), cfg.batch):
        q0, u = flat_q[start : start + cfg.batch], flat_u[start : start + cfg.batch]
        plant = HydraulicPlant(model, kin, len(q0), str(device))
        plant.valve_gain.fill_(perturbation["gain"])
        plant.valve_offset.fill_(perturbation["offset"])
        plant.action_delay.fill_(perturbation["delay"])
        plant.speed_scale[:, :3] = perturbation["speed"]
        plant.reset(torch.arange(len(q0), device=device), q0)
        total = torch.zeros(len(q0), 3, device=device)
        for k in range(cfg.settle_steps):
            plant.u_cmd.copy_(u)
            plant.step()
            if k >= cfg.settle_steps - cfg.average_steps:
                total += plant.v[:, :3]
            plant.q.copy_(q0)  # pinned: the speed belongs to this pose, not to wherever the joint drifted
        out[start : start + cfg.batch] = total / cfg.average_steps
    return out


def direction_vectors(degrees, device) -> torch.Tensor:
    angles = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32, device=device))
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def on_direction(twist: torch.Tensor, directions: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Speed along each direction [m/s] and whether the twist [..., 3] counts as going that way, shape [..., D]."""
    along = twist[..., None, :2].mul(directions).sum(-1)
    cross = (twist[..., None, :2] - along[..., None] * directions).norm(dim=-1)
    ok = (along > 0) & (cross <= cfg.max_cross * along)
    ok &= twist[..., 2:3].abs() <= cfg.max_pitch_per_m * along
    return along, ok


def sampled_tip_speeds(
    kin, poses: torch.Tensor, joint_speeds: torch.Tensor, valves: torch.Tensor, directions: torch.Tensor, cfg
):
    """Best tip speed [m/s] among fixed valve samples per pose and direction, and its valves.

    Returns ``(speed [P, D], valves [P, D, 3])``; speed is 0 where no sample goes that way.
    """
    _, jacobian = kin.pose_jacobian(poses)
    twist = torch.einsum("pij,psj->psi", jacobian[:, :, :3], joint_speeds)  # [P, S, 3]
    along, ok = on_direction(twist, directions, cfg)  # [P, S, D]
    best, index = torch.where(ok, along, 0.0).max(dim=1)  # [P, D]
    return best, valves[index]


def spool_curves(model: Path, kin, poses: torch.Tensor, cfg) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-spool steady speed [rad/s] against opening at each pose: (u [C], speed [P, 3, C])."""
    u = torch.linspace(-1, 1, cfg.curve_points, device=poses.device)
    valves = torch.zeros(3, len(u), 3, device=poses.device)
    for j in range(3):
        valves[j, :, j] = u
    speeds = steady_joint_speeds(model, kin, poses, valves.reshape(-1, 3), cfg).reshape(
        len(poses), 3, len(u), 3
    )
    return u, torch.stack([speeds[:, j, :, j] for j in range(3)], dim=1)


def invert_curves(u: torch.Tensor, curves: torch.Tensor, wanted: torch.Tensor) -> torch.Tensor:
    """Opening whose single-spool speed is closest to ``wanted`` [N, 3], given curves [N, 3, C] on grid u [C]."""
    return u[(curves - wanted[..., None]).abs().argmin(dim=-1)]


@torch.no_grad()
def solved_tip_speeds(model: Path, kin, poses: torch.Tensor, directions: torch.Tensor, cfg):
    """Fastest ladder speed [m/s] whose solved steady valves meet the tolerances, with those valves.

    Returns ``(speed [P, D], valves [P, D, 3])``; speed is 0 if not even the slowest rung works.
    """
    device = poses.device
    first, last, count = cfg.speed_ladder_mm_s
    ladder = torch.logspace(math.log10(first), math.log10(last), count, device=device) / 1000
    p, d, k = len(poses), len(directions), len(ladder)
    _, jacobian = kin.pose_jacobian(poses)
    j = jacobian[:, :, :3]
    twist = torch.zeros(d, k, 3, device=device)
    twist[:, :, :2] = directions[:, None] * ladder[None, :, None]
    wanted = torch.linalg.solve(j[:, None, None], twist[None, :, :, :, None].expand(p, d, k, 3, 1)).squeeze(
        -1
    )

    u_grid, curves = spool_curves(model, kin, poses, cfg)  # curves [P, 3, C]
    full = curves.abs().amax(dim=-1)  # [P, 3]
    flat_q = poses[:, None, None].expand(p, d, k, 4).reshape(-1, 4)
    flat_wanted = wanted.reshape(-1, 3)
    flat_curves = curves[:, None, None].expand(p, d, k, 3, -1).reshape(-1, 3, curves.shape[-1])
    flat_full = full[:, None, None].expand(p, d, k, 3).reshape(-1, 3)
    request = flat_wanted.clone()
    u = invert_curves(u_grid, flat_curves, request)
    for _ in range(cfg.solve_iterations):
        actual = paired_steady_speeds(model, kin, flat_q, u, cfg)
        request = (request + cfg.solve_gain * (flat_wanted - actual)).clamp(-flat_full, flat_full)
        u = invert_curves(u_grid, flat_curves, request)
    actual = paired_steady_speeds(model, kin, flat_q, u, cfg).reshape(p, d, k, 3)
    reached = torch.einsum("pij,pdkj->pdki", j, actual)
    along, ok = on_direction(reached, directions, cfg)  # [P, D, K, D]
    ok = torch.diagonal(ok, dim1=1, dim2=3).permute(0, 2, 1)  # [P, D, K]: each row against its own direction
    along = torch.diagonal(along, dim1=1, dim2=3).permute(0, 2, 1)
    ok &= along >= 0.9 * ladder  # must actually deliver the rung, not merely point the right way
    rung = torch.where(ok, torch.arange(k, device=device), -1).amax(dim=-1)  # [P, D]
    speed = torch.where(rung >= 0, along.gather(-1, rung.clamp_min(0)[..., None]).squeeze(-1), 0.0)
    valves = u.reshape(p, d, k, 3).gather(2, rung.clamp_min(0)[..., None, None].expand(p, d, 1, 3)).squeeze(2)
    return speed, valves


def achievable_tip_speeds(model: Path, kin, poses: torch.Tensor, directions: torch.Tensor, cfg):
    """Best of the solved ladder and the fixed valve samples: ``(speed [P, D] m/s, valves [P, D, 3])``."""
    valves = valve_samples(cfg, poses.device)
    sampled, sampled_valves = sampled_tip_speeds(
        kin, poses, steady_joint_speeds(model, kin, poses, valves, cfg), valves, directions, cfg
    )
    solved, solved_valves = solved_tip_speeds(model, kin, poses, directions, cfg)
    use_solved = solved >= sampled
    return torch.maximum(solved, sampled), torch.where(use_solved[..., None], solved_valves, sampled_valves)


def single_joint_bound(
    kin, poses: torch.Tensor, capacity: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Tip speed [m/s] if every joint could run at its own full-valve speed at once, shape [P, D].

    ``capacity`` is [P, 3, 2] = (negative, positive) speed magnitude per joint [rad/s].
    """
    _, jacobian = kin.pose_jacobian(poses)
    j = jacobian[:, :, :3]
    twist = torch.cat(
        (directions, torch.zeros(len(directions), 1, device=poses.device)), dim=1
    )  # unit, no pitch
    rates = torch.linalg.solve(j[:, None], twist[None, :, :, None].expand(len(poses), -1, -1, -1)).squeeze(-1)
    limit = torch.where(rates < 0, capacity[:, None, :, 0], capacity[:, None, :, 1])
    return (limit / rates.abs().clamp_min(1e-9)).amin(dim=-1)


def workspace_poses(kin, cfg: SpeedMapConfig, device: str):
    """Grid targets [G, 3] (x, z, pitch), the IK solutions [G, 4] and which of them are valid."""
    xs = torch.arange(cfg.x_range[0], cfg.x_range[1] + 1e-9, cfg.spacing)
    zs = torch.arange(cfg.z_range[0], cfg.z_range[1] + 1e-9, cfg.spacing)
    xx, zz = torch.meshgrid(xs, zs, indexing="ij")
    home = torch.tensor([HOME], device=device)
    pitch = float(kin.pose_jacobian(home)[0][0, 2]) if cfg.pitch is None else cfg.pitch
    targets = torch.stack((xx.flatten(), zz.flatten(), torch.full((xx.numel(),), pitch)), dim=1).to(device)
    q, valid = kin.inverse(targets, home)
    return xs, zs, targets, q, valid


def curve_summary(u: torch.Tensor, speed: torch.Tensor, fraction: float = 0.1) -> dict:
    """Full-valve speed [deg/s] and deadband edge (smallest |u| reaching ``fraction`` of it) per joint and sign."""
    out = {}
    for j, joint in enumerate(PID_JOINTS):
        for sign, name in ((-1, "neg"), (1, "pos")):
            side = (u * sign) > 0
            us, vs = (u[side] * sign), (speed[j, side] * sign)
            full = float(vs[us.argmax()])
            moving = us[vs >= fraction * full]
            out[f"{joint}_{name}"] = {
                "full_valve_deg_s": math.degrees(full),
                "deadband_u": float(moving.min()) if len(moving) else math.nan,
            }
    return out


@torch.no_grad()
def measure_speed_map(model: Path, kin, cfg: SpeedMapConfig, device: str) -> dict:
    """Run the whole measurement; tensors in the result live on the CPU."""
    xs, zs, targets, q, valid = workspace_poses(kin, cfg, device)
    home = torch.tensor([HOME], device=device)
    poses = torch.cat((q[valid], home))  # HOME rides along as the last row
    directions = direction_vectors(cfg.directions_deg, device)
    best, best_valves = achievable_tip_speeds(model, kin, poses, directions, cfg)

    u, curves = spool_curves(model, kin, poses, cfg)
    singles = torch.stack((-curves[..., 0], curves[..., -1]), dim=-1).clamp_min(0)  # full valve, [P, 3, 2]
    bound = single_joint_bound(kin, poses, singles, directions)
    home_speed, home_curves = best[-1], curves[-1]
    best, best_valves, bound = best[:-1], best_valves[:-1], bound[:-1]

    shape = (len(xs), len(zs), len(directions))
    grid_best = torch.full((len(targets), len(directions)), math.nan, device=device)
    grid_bound = grid_best.clone()
    grid_best[valid], grid_bound[valid] = best, bound
    grid_valves = torch.full((len(targets), len(directions), 3), math.nan, device=device)
    grid_valves[valid] = best_valves

    return {
        "x": xs,
        "z": zs,
        "valid": valid.reshape(shape[:2]).cpu(),
        "tip_speed": grid_best.reshape(shape).cpu(),
        "single_joint_bound": grid_bound.reshape(shape).cpu(),
        "valves": grid_valves.reshape(*shape, 3).cpu(),
        "directions_deg": list(cfg.directions_deg),
        "home_tip_speed": home_speed.cpu(),
        "curve_u": u.cpu(),
        "curve_speed": home_curves.cpu(),
        "curves": curve_summary(u.cpu(), home_curves.cpu()),
    }


def direction_name(degrees: float) -> str:
    names = {0: "+X", 45: "+X+Z", 90: "+Z", 135: "-X+Z", 180: "-X", 225: "-X-Z", 270: "-Z", 315: "+X-Z"}
    return names.get(round(degrees) % 360, f"{degrees:g}deg")


def write_report(out_dir: Path, result: dict, cfg: SpeedMapConfig, model: Path) -> None:
    """speed_map.json (mm/s, null outside the workspace), speed_map.pt and the two figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_dir / "speed_map.pt")

    def mm(t):
        return [[None if math.isnan(v) else round(v * 1000, 1) for v in row] for row in t.tolist()]

    summary = {
        "config": asdict(cfg),
        "model": str(model),
        "model_sha256": sha256(Path(model) / "mlp_state_dict.pt"),
        "x_m": result["x"].tolist(),
        "z_m": result["z"].tolist(),
        "valve_curves_at_home": result["curves"],
        "home_tip_speed_mm_s": {
            direction_name(d): round(float(v) * 1000, 1)
            for d, v in zip(cfg.directions_deg, result["home_tip_speed"])
        },
        "tip_speed_mm_s": {
            direction_name(d): mm(result["tip_speed"][:, :, i]) for i, d in enumerate(cfg.directions_deg)
        },
        "single_joint_bound_mm_s": {
            direction_name(d): mm(result["single_joint_bound"][:, :, i])
            for i, d in enumerate(cfg.directions_deg)
        },
    }
    (out_dir / "speed_map.json").write_text(json.dumps(summary, indent=1))
    plot_report(out_dir, result, cfg)


def plot_report(out_dir: Path, result: dict, cfg: SpeedMapConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, z = result["x"].numpy(), result["z"].numpy()
    extent = (
        x[0] - cfg.spacing / 2,
        x[-1] + cfg.spacing / 2,
        z[0] - cfg.spacing / 2,
        z[-1] + cfg.spacing / 2,
    )
    count = len(cfg.directions_deg)
    fig, axes = plt.subplots(2, count, figsize=(3.1 * count, 6.4), constrained_layout=True)
    top = float(torch.nan_to_num(result["tip_speed"], nan=0).max()) * 1000
    for i, degrees in enumerate(cfg.directions_deg):
        speed = result["tip_speed"][:, :, i].numpy() * 1000
        loss = result["tip_speed"][:, :, i] / result["single_joint_bound"][:, :, i]
        image = axes[0, i].imshow(speed.T, origin="lower", extent=extent, vmin=0, vmax=top, cmap="viridis")
        axes[0, i].set_title(f"{direction_name(degrees)} mm/s")
        ratio = axes[1, i].imshow(loss.numpy().T, origin="lower", extent=extent, vmin=0, vmax=1, cmap="magma")
        axes[1, i].set_title("/ single-joint bound")
        for row in (0, 1):
            axes[row, i].set_xlabel("x [m]")
            arrow = direction_vectors([degrees], "cpu")[0].numpy() * 0.08
            axes[row, i].annotate(
                "",
                xy=(x[-1] - 0.1 + arrow[0], z[0] + 0.1 + arrow[1]),
                xytext=(x[-1] - 0.1, z[0] + 0.1),
                arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5},
            )
        for row, label in ((0, "z [m]"), (1, "z [m]")):
            axes[row, 0].set_ylabel(label)
    fig.colorbar(image, ax=axes[0, :], shrink=0.8, label="achievable tip speed [mm/s]")
    fig.colorbar(ratio, ax=axes[1, :], shrink=0.8, label="achieved / single-joint bound")
    fig.savefig(out_dir / "speed_map.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    u = result["curve_u"].numpy()
    for j, joint in enumerate(PID_JOINTS):
        axes[j].plot(u, torch.rad2deg(result["curve_speed"][j]).numpy(), "o-", ms=3)
        for sign, name in ((-1, "neg"), (1, "pos")):
            edge = result["curves"][f"{joint}_{name}"]["deadband_u"]
            if not math.isnan(edge):
                axes[j].axvline(sign * edge, color="gray", ls=":", lw=1)
        axes[j].set(title=f"{joint} at HOME, single spool", xlabel="valve u", ylabel="steady speed [deg/s]")
        axes[j].grid(alpha=0.3)
    fig.savefig(out_dir / "valve_curves.png", dpi=110)
    plt.close(fig)
