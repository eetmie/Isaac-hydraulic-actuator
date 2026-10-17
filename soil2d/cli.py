"""Run a scripted scene, replay common bucket poses, or time batched physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import warp as wp

from .config import Scene
from .solver import Soil
from .trajectory import POSE_COLUMNS, Trajectory, digging_path


def statistics(values) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(data.max()),
    }


def source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob("*.py"))}


def provenance(solver: Soil, trajectory: Trajectory) -> dict:
    buffers = [v for v in vars(solver).values() if isinstance(v, wp.array)]
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "scene": solver.scene.as_dict(),
        "device": str(solver.device),
        "device_name": solver.device.name,
        "warp": wp.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "envs": solver.envs,
        "particles_per_env": solver.particles,
        "graph": solver.graph is not None,
        "substeps": solver.scene.substeps,
        "solver_buffer_bytes": sum(a.capacity for a in buffers),
        "collision_radius_m": solver.scene.collision_radius,
        "source_sha256": source_hashes(),
        "trajectory": {
            "times_s": trajectory.times.tolist(),
            "poses_x_z_ccw": trajectory.poses.tolist(),
            "phases": trajectory.phases,
            "smooth_quintic": trajectory.smooth,
        },
        "wrench_convention": "soil on bucket; Fx,Fz,torque CCW in x/z (negative world Y); about bucket origin",
        "model": "MLS-MPM/APIC, Hencky Drucker-Prager; prescribed capsule walls; no compaction memory",
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def timed_step(solver: Soil, trajectory: Trajectory, *, enabled=True) -> float:
    pose, twist = trajectory.command(solver.time, solver.scene.frame_dt)
    wp.synchronize_device(solver.device)
    started = time.perf_counter()
    solver.step(pose, twist, enabled=enabled)
    wp.synchronize_device(solver.device)
    return 1000 * (time.perf_counter() - started)


def run_demo(scene: Scene, trajectory: Trajectory, args) -> int:
    from . import render

    duration = args.seconds if args.seconds is not None else trajectory.duration
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    frames = round(duration / scene.frame_dt)
    if frames < 1 or not math.isclose(frames * scene.frame_dt, duration, abs_tol=1e-8):
        raise ValueError("Duration must be a positive integer multiple of frame_dt")
    output = args.out or Path("runs/soil2d") / datetime.now().strftime("demo-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    solver = Soil(scene, device=args.device, graph=not args.no_graph)
    # Build/capture and warm-up are deliberately outside the timed interval.
    for _ in range(3):
        solver.step(enabled=False)
    wp.synchronize_device(solver.device)
    solver.reset()
    metadata = provenance(solver, trajectory)
    rows, poses, images, timings = [], [], [], []
    poses.append(dict(zip(POSE_COLUMNS, [0.0, *trajectory.sample(0.0)])))
    next_image = 0.0
    worst_cfl, max_projection = 0.0, 0.0
    loop_started = time.perf_counter()
    try:
        for frame in range(frames):
            wall_ms = timed_step(solver, trajectory, enabled=not args.no_bucket)
            timings.append(wall_ms)
            state = solver.snapshot()
            report = solver.check()
            worst_cfl = max(worst_cfl, report["cfl"])
            max_projection = max(max_projection, report["max_projection_m"])
            force = state["wrench"][0]
            pos = state["position"][0]
            vel = state["velocity"][0]
            pose = state["bucket_pose"][0]
            twist = solver.twist.numpy()[0]
            phase = trajectory.phase(solver.time - scene.frame_dt / 2)
            rows.append(
                {
                    "time_s": solver.time,
                    "phase": phase,
                    "bucket_x_m": float(pose[0]),
                    "bucket_z_m": float(pose[1]),
                    "bucket_angle_rad": float(pose[2]),
                    "bucket_vx_m_s": float(twist[0]),
                    "bucket_vz_m_s": float(twist[1]),
                    "bucket_omega_ccw_rad_s": float(twist[2]),
                    "force_x_n": float(force[0]),
                    "force_z_n": float(force[1]),
                    "torque_ccw_nm": float(force[2]),
                    "physics_wall_ms": wall_ms,
                    "particle_translational_ke_j": float(
                        0.5 * solver.particle_mass * np.sum(vel.astype(float) ** 2)
                    ),
                    "soil_com_x_m": float(pos[:, 0].mean()),
                    "soil_com_z_m": float(pos[:, 1].mean()),
                    "max_speed_m_s": report["max_speed_m_s"],
                    "max_projection_m": report["max_projection_m"],
                    "cfl": report["cfl"],
                }
            )
            poses.append(dict(zip(POSE_COLUMNS, [solver.time, *pose.tolist()])))
            if args.gif and solver.time >= next_image - 1e-9:
                images.append(
                    render.image(
                        scene,
                        pos,
                        solver.initial[0],
                        pose,
                        time_s=solver.time,
                        phase=phase,
                        force=force,
                        bucket_enabled=not args.no_bucket,
                    )
                )
                next_image += 0.05
            if frame % max(1, frames // 8) == 0:
                print(
                    f"{solver.time:5.2f}/{duration:.2f} s  {phase:12s}  F=({force[0]:+.1f},{force[1]:+.1f}) N",
                    flush=True,
                )
        loop_wall_s = time.perf_counter() - loop_started
        if source_hashes() != metadata["source_sha256"]:
            raise ValueError("Source files changed during the run; repeat with stable sources")
        final = solver.snapshot()
        summary = {
            "status": "ok",
            **metadata,
            "bucket_enabled": not args.no_bucket,
            "duration_s": solver.time,
            "physics_frame_wall_ms": statistics(timings),
            "physics_wall_s": sum(timings) / 1000,
            "real_time_factor": solver.time / (sum(timings) / 1000),
            "validated_preview_loop_wall_s": loop_wall_s,
            "validated_preview_real_time_factor": solver.time / loop_wall_s,
            "preview_loop_scope": "physics, diagnostics, logging in memory and optional GIF frame drawing; excludes setup and file encoding/export",
            "timing_scope": "host command upload, graph/kernel launches, device synchronization; excludes diagnostics, logging and rendering",
            "max_force_n": max(math.hypot(r["force_x_n"], r["force_z_n"]) for r in rows),
            "worst_cfl": worst_cfl,
            "max_projection_m": max_projection,
            "validation": solver.check(),
        }
        render.image(
            scene,
            final["position"][0],
            solver.initial[0],
            final["bucket_pose"][0],
            time_s=solver.time,
            phase="final",
            force=final["wrench"][0],
            bucket_enabled=not args.no_bucket,
        ).save(output / "final.png")
        render.force_plot(rows, output / "forces.png")
        np.savez_compressed(output / "final_state.npz", **final)
        if images:
            images[0].save(
                output / "preview.gif",
                save_all=True,
                append_images=images[1:],
                duration=50,
                loop=0,
                optimize=False,
            )
        for name, values in (("tracking.csv", rows), ("bucket_poses.csv", poses)):
            with (output / name).open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=values[0].keys())
                writer.writeheader()
                writer.writerows(values)
        write_json(output / "summary.json", summary)
        print(
            f"Saved {output}; physics {summary['real_time_factor']:.2f}x real time, peak force {summary['max_force_n']:.1f} N"
        )
    except Exception as exc:
        write_json(
            output / "failure.json",
            {"status": "failed", "time_s": solver.time, "error": str(exc), **metadata},
        )
        raise
    return 0


def benchmark(scene: Scene, trajectory: Trajectory, args) -> int:
    if args.seconds is None or not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Benchmark seconds must be positive")
    frames = round(args.seconds / scene.frame_dt)
    if frames < 1 or not math.isclose(frames * scene.frame_dt, args.seconds, abs_tol=1e-8):
        raise ValueError("Benchmark duration must be an integer multiple of frame_dt")
    if args.out and args.out.exists():
        raise FileExistsError(args.out)
    results = []
    for count in args.envs:
        solver = Soil(scene, envs=count, device=args.device, graph=not args.no_graph)
        for _ in range(5):
            solver.step(enabled=False)
        wp.synchronize_device(solver.device)
        solver.reset()
        metadata = provenance(solver, trajectory)
        timings = []
        worst_cfl, max_projection = 0.0, 0.0
        loop_started = time.perf_counter()
        for _ in range(frames):
            timings.append(timed_step(solver, trajectory))
            validation = solver.check()
            worst_cfl = max(worst_cfl, validation["cfl"])
            max_projection = max(max_projection, validation["max_projection_m"])
        loop_wall_s = time.perf_counter() - loop_started
        if source_hashes() != metadata["source_sha256"]:
            raise ValueError("Source files changed during the benchmark; repeat with stable sources")
        elapsed = sum(timings) / 1000
        result = {
            **metadata,
            "validation": validation,
            "duration_s_per_env": solver.time,
            "physics_frame_wall_ms": statistics(timings),
            "physics_wall_s": elapsed,
            "real_time_factor_per_env": solver.time / elapsed,
            "aggregate_simulated_seconds_per_wall_second": count * solver.time / elapsed,
            "validated_loop_wall_s": loop_wall_s,
            "validated_real_time_factor_per_env": solver.time / loop_wall_s,
            "worst_cfl": worst_cfl,
            "max_projection_m": max_projection,
            "timing_scope": "host command upload, graph/kernel launches and device synchronization; excludes snapshot/render/log I/O",
        }
        results.append(result)
        print(
            f"{count:4d} envs: {result['physics_frame_wall_ms']['p50']:.3f} ms/frame; {result['real_time_factor_per_env']:.2f}x per env; {result['aggregate_simulated_seconds_per_wall_second']:.1f} aggregate",
            flush=True,
        )
    output = args.out or Path("runs/soil2d") / datetime.now().strftime("benchmark-%Y%m%d-%H%M%S-%f.json")
    write_json(output, {"status": "ok", "results": results})
    print(f"Saved {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("demo", "view", "benchmark"):
        p = commands.add_parser(name)
        p.add_argument("--device", help="cpu, cuda:0, ...; default: CUDA when available")
        p.add_argument("--no-graph", action="store_true", help="disable CUDA graph replay")
        p.add_argument("--dx", type=float, default=0.01)
        p.add_argument("--dt", type=float, default=0.0005)
        p.add_argument("--frame-dt", type=float, default=0.02)
        p.add_argument("--young-modulus", type=float, default=20_000)
        p.add_argument("--friction-angle", type=float, default=32)
        p.add_argument("--cohesion", type=float, default=0)
        p.add_argument("--effective-width", type=float, default=0.11)
        p.add_argument("--particles-per-axis", type=int, default=2)
        p.add_argument("--trajectory", type=Path, help="common bucket-origin pose CSV; linear interpolation")
        if name != "view":
            p.add_argument("--seconds", type=float, default=8.0 if name == "benchmark" else None)
            p.add_argument(
                "--out", type=Path, help="new output directory for demo; new JSON file for benchmark"
            )
        if name == "demo":
            p.add_argument("--gif", action="store_true", help="also write a 20 fps preview")
            p.add_argument("--no-bucket", action="store_true", help="soil-only settling baseline")
        if name == "benchmark":
            p.add_argument("--envs", type=int, nargs="+", default=[1, 8, 32])
    return root


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        material = replace(
            Scene().material,
            young_modulus=args.young_modulus,
            friction_angle_deg=args.friction_angle,
            cohesion_pa=args.cohesion,
        )
        scene = replace(
            Scene(),
            dx=args.dx,
            dt=args.dt,
            frame_dt=args.frame_dt,
            material=material,
            effective_width=args.effective_width,
            particles_per_axis=args.particles_per_axis,
        )
        scene.validate()
        trajectory = Trajectory.from_csv(args.trajectory) if args.trajectory else digging_path()
        if args.command == "view":
            from .render import view

            view(scene, trajectory, device=args.device, graph=not args.no_graph)
            return 0
        if args.command == "benchmark":
            return benchmark(scene, trajectory, args)
        return run_demo(scene, trajectory, args)
    except (ValueError, OSError, FloatingPointError) as exc:
        p.exit(1, f"soil2d: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
