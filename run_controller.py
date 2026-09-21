"""Run, draw with, or benchmark the learned hydraulic valve controller.

Examples from the Isaac Lab root::

    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/run_controller.py --mode draw
    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/run_controller.py --mode circle
    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/run_controller.py --mode gamepad
    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/run_controller.py --mode benchmark --viz none

The policy runs at its trained rate (20 Hz by default) and the frozen hydraulic
recurrence remains the authoritative state transition at 100 Hz, matching training.
Draw and circle modes close a proportional position loop around the policy
(Egli & Hutter, RA-L 2022, eq. 2); gamepad mode commands tip velocity directly.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import queue
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# The prototype controller shipped with the repository, so the demo runs on a fresh clone.
DEFAULT_CHECKPOINT = ROOT / "models/controller_proto/model_1798.pt"


def resolve_checkpoint(value: str | None) -> Path:
    """Resolve an explicit checkpoint, else the shipped prototype controller."""
    path = Path(value).expanduser().resolve() if value else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


DEFAULT_SPEED_MM_S = 30.0


def selected_speed(checkpoint: Path, requested: float | None, result_dir: Path) -> float:
    """Return the cruise speed [m/s]: an override, else 30 mm/s.

    A benchmark result tied to this exact checkpoint is only reported, not applied. Results are looked up in
    ``result_dir`` and in ``benchmark_<checkpoint stem>`` next to the checkpoint.
    """
    if requested is not None:
        if not math.isfinite(requested) or not 1 <= requested <= 120:
            raise ValueError("--speed-mm-s must be between 1 and 120")
        return requested / 1000.0
    from hydraulic_controller.core import sha256

    for directory in (result_dir, checkpoint.parent / f"benchmark_{checkpoint.stem}"):
        result = directory / "best_speed.json"
        if not result.is_file():
            continue
        metadata = json.loads(result.read_text())
        if metadata.get("checkpoint_sha256") == sha256(checkpoint):
            recommended = float(metadata["recommended_speed_mm_s"])
            print(f"[INFO] Benchmark recommends {recommended:g} mm/s (--speed-mm-s {recommended:g})")
            break
    return DEFAULT_SPEED_MM_S / 1000.0


def circle_path(box: np.ndarray, points: int = 97) -> np.ndarray:
    """Create a closed circle inset from the validated drawing rectangle."""
    center = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
    radius = 0.32 * min(box[1] - box[0], box[3] - box[2])
    phase = np.linspace(0, 2 * np.pi, points)
    return center + radius * np.stack((np.cos(phase), np.sin(phase)), axis=1)


def drain_ui(incoming, outgoing, kin, plant, box, session, validate_stroke, sketch):
    """Apply all pending UI requests at a 20 Hz policy boundary."""
    closing = False
    while incoming is not None:
        try:
            message = incoming.get_nowait()
        except queue.Empty:
            break
        pose, _ = kin.pose_jacobian(plant.q)
        position = pose[0, :2].detach().cpu().numpy()
        if message["kind"] == "close":
            closing = True
        elif message["kind"] == "speed":
            session.set_speed(message["mm_s"] / 1000.0)
        elif message["kind"] == "cancel":
            session.cancel(position)
            if message.get("clear"):
                sketch.clear()
        elif message["kind"] == "stroke":
            try:
                path = validate_stroke(kin, plant.q, np.asarray(message["points"]), box, session.angle)
                session.submit(path, position)
            except ValueError as exc:
                print(f"[WARN] Stroke rejected: {exc}")
                bad = getattr(exc, "points", np.zeros((0, 2)))
                # A couple of dozen marks are enough to show where the stroke leaves the reachable area.
                bad = bad[:: max(1, len(bad) // 20)]
                outgoing.put({"kind": "error", "text": str(exc), "points": bad.tolist()})
            else:
                sketch.show_path(path)
                outgoing.put({"kind": "path", "points": path.tolist()})
    return closing


def run_interactive(args_cli, checkpoint: Path, simulation_app) -> None:
    """Create one Isaac scene and run the custom deterministic playback loop."""
    # Isaac/Kit imports must happen after AppLauncher starts the application.
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import SPHERE_MARKER_CFG
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab_physx.physics import PhysxCfg

    from hydraulic_controller.core import HOME, HydraulicPlant
    from hydraulic_controller.draw_ui import open_drawing_window
    from hydraulic_controller.gamepad import GamepadVelocity
    from hydraulic_controller.kinematics import ExcavatorKinematics
    from hydraulic_controller.policy import ControllerPolicy
    from hydraulic_controller.scene import HydraulicSceneBridge, ViewportSketch, robot_config, setup_scene
    from hydraulic_controller.trajectory import DrawingSession, validate_stroke

    torch.set_num_threads(args_cli.torch_threads)
    output_dir = Path(args_cli.benchmark_out).resolve()
    speed = selected_speed(checkpoint, args_cli.speed_mm_s, output_dir)
    policy = ControllerPolicy(checkpoint, args_cli.device)
    kin = ExcavatorKinematics(policy.asset_path, args_cli.device)
    settings = policy.settings
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Requested cruise: {1000 * speed:g} mm/s")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=settings.dt,
            render_interval=1,
            device=args_cli.device,
            physics=PhysxCfg(enable_external_forces_every_iteration=True),
        )
    )
    sim.set_camera_view([2.2, 2.2, 1.5], [0.3, 0.0, 0.25])
    scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True))
    robot = setup_scene(scene, sim.stage, robot_config(str(policy.asset_path)))
    sim.reset()
    print(f"[INFO] Fixed base: {robot.is_fixed_base}")

    target_marker = None
    if args_cli.mode in ("draw", "circle"):
        target_marker_cfg = SPHERE_MARKER_CFG.copy()
        target_marker_cfg.markers["sphere"].radius = 0.006
        target_marker = VisualizationMarkers(
            target_marker_cfg.replace(prim_path="/Visuals/HydraulicController/target")
        )

    plant = HydraulicPlant(policy.model_path, kin, 1, args_cli.device, settings)
    bridge = HydraulicSceneBridge(sim, robot, plant)
    governor = policy.governor(kin, args_cli.governor_margin)
    home_ids = torch.zeros(1, dtype=torch.long, device=args_cli.device)
    requested = torch.zeros(1, 3, device=args_cli.device)
    pose, _ = kin.pose_jacobian(plant.q)
    session = DrawingSession(
        pose[0].cpu().numpy(),
        speed,
        kp=args_cli.kp,
        speed_limit=settings.speed_max,
        pitch_rate_limit=settings.pitch_rate_max,
    )
    gamepad_request = np.zeros(3, dtype=np.float32)

    ui_process = ui_in = ui_out = None
    gamepad = None
    sketch = None
    box = kin.drawing_box(plant.q)
    if args_cli.mode == "draw":
        # Shade what the arm cannot reach at the held bucket angle, so strokes are not silently rejected.
        reachable = kin.reachable_cells(plant.q, box, session.angle)
        sketch = ViewportSketch(robot.data.root_pos_w.torch[0], box)
        context = mp.get_context("spawn")
        ui_in, ui_out = context.Queue(), context.Queue()
        ui_process = context.Process(
            target=open_drawing_window,
            args=(ui_in, ui_out, box.tolist(), 1000 * speed, reachable.tolist(), 1000 * settings.speed_max),
            daemon=True,
        )
        ui_process.start()
        print(
            f"[INFO] Drawing box [m]: {box.tolist()}; {reachable.mean():.0%} reachable at the held bucket angle"
        )
    elif args_cli.mode == "circle":
        circle_box = kin.local_reachable_box(plant.q)
        sketch = ViewportSketch(robot.data.root_pos_w.torch[0], circle_box)
        path = validate_stroke(kin, plant.q, circle_path(circle_box), circle_box)
        session.submit(path, pose[0, :2].cpu().numpy())
        sketch.show_path(path)
    else:
        gamepad = GamepadVelocity(speed, settings.pitch_rate_max)

    step = 0
    closing = False
    drawing_errors = []
    try:
        while (
            simulation_app.is_running()
            and not closing
            and (not args_cli.max_steps or step < args_cli.max_steps)
        ):
            if step % settings.decimation == 0:
                pose, jacobian = kin.pose_jacobian(plant.q)
                twist = torch.einsum("nij,nj->ni", jacobian, plant.v)
                if args_cli.mode == "draw":
                    closing = drain_ui(ui_in, ui_out, kin, plant, box, session, validate_stroke, sketch)
                    command_np = session.command(
                        pose[0].detach().cpu().numpy(),
                        twist[0].detach().cpu().numpy(),
                        settings.dt * settings.decimation,
                    )
                elif args_cli.mode == "circle":
                    command_np = session.command(
                        pose[0].detach().cpu().numpy(),
                        twist[0].detach().cpu().numpy(),
                        settings.dt * settings.decimation,
                    )
                else:
                    if gamepad.wants_reset():
                        plant.reset(home_ids, torch.tensor([HOME], device=args_cli.device))
                        bridge.sync()
                    # Slew toward the stick command at the training command acceleration.
                    policy_dt = settings.dt * settings.decimation
                    delta = gamepad.command() - gamepad_request
                    norm = float(np.linalg.norm(delta[:2]))
                    delta[:2] *= min(1.0, 0.25 * policy_dt / norm) if norm > 1e-9 else 1.0
                    delta[2] = np.clip(delta[2], -policy_dt, policy_dt)
                    gamepad_request += delta
                    command_np = gamepad_request
                if sketch is not None:
                    sketch.trace(pose[0, :2].detach().cpu().numpy(), session.phase)
                if args_cli.mode != "gamepad" and session.phase == "drawing":
                    drawing_errors.append(
                        float(np.linalg.norm(pose[0, :2].detach().cpu().numpy() - session.reference))
                    )
                requested[0] = torch.as_tensor(command_np, device=args_cli.device)
                admitted, _ = governor(plant.q, plant.v, requested)
                plant.begin_action(policy(plant.observe(admitted)))
                if target_marker is not None:
                    target_xz = torch.as_tensor(
                        session.display_target, device=args_cli.device, dtype=pose.dtype
                    ).reshape(1, 2)
                    target_position = robot.data.root_pos_w.torch + torch.stack(
                        (target_xz[:, 0], torch.zeros_like(target_xz[:, 0]), target_xz[:, 1]), dim=1
                    )
                    target_marker.visualize(translations=target_position)
                if ui_out is not None:
                    error = float(np.linalg.norm(pose[0, :2].detach().cpu().numpy() - session.reference))
                    ui_out.put(
                        {
                            "kind": "state",
                            "position": pose[0, :2].detach().cpu().tolist(),
                            "target": session.display_target.tolist(),
                            "phase": session.phase,
                            "error_mm": 1000 * error,
                            "speed_mm_s": 1000 * float(twist[0, :2].norm()),
                            "valves": plant.u[0].detach().cpu().tolist(),
                        }
                    )
            plant.step()
            bridge.sync()
            scene.write_data_to_sim()
            sim.step()
            scene.update(settings.dt)
            step += 1
            if bool(plant.invalid[0] or plant.limit_hit[0] or kin.colliding(plant.q)[0]):
                raise RuntimeError(
                    "Controller stopped after invalid state, a joint limit, or a geometry collision"
                )
    finally:
        # Report before closing the app: SimulationApp.close() may end the process.
        print(f"[INFO] Completed {step} hydraulic steps; PhysX post-step syncs={bridge.post_steps}")
        if drawing_errors:
            print(
                f"[INFO] Drawing position error: mean {1000 * np.mean(drawing_errors):.2f} mm, "
                f"max {1000 * np.max(drawing_errors):.2f} mm over {len(drawing_errors)} policy steps"
            )
        if gamepad is not None:
            gamepad.close()
        if ui_out is not None:
            ui_out.put({"kind": "close"})
        if ui_process is not None:
            ui_process.join(timeout=2)
            if ui_process.is_alive():
                ui_process.terminate()
                ui_process.join(timeout=1)
        bridge.close()
        simulation_app.close()


def entrypoint() -> int:
    """Parse once in the parent process, then launch benchmark or Isaac playback."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Hydraulic end-effector velocity controller")
    parser.add_argument(
        "--checkpoint", help="RSL-RL model_*.pt; default is the shipped prototype in models/controller_proto"
    )
    parser.add_argument("--mode", choices=("draw", "circle", "gamepad", "benchmark"), default="draw")
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        help=f"Cruise speed in mm/s; default {DEFAULT_SPEED_MM_S:g}. In draw mode, the initial value of the window's slider",
    )
    parser.add_argument("--kp", type=float, default=3.0, help="Position loop gain for draw/circle [1/s]")
    parser.add_argument(
        "--governor_margin",
        type=float,
        help="Joint-speed governor margin for draw/circle/gamepad: 0 disables, e.g. 0.8 enables; default as trained",
    )
    parser.add_argument("--benchmark-out", default=str(ROOT / "runs/controller_benchmark"))
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many 100 Hz steps; 0 is continuous"
    )
    parser.add_argument("--torch_threads", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.max_steps < 0:
        parser.error("--max_steps must be non-negative")
    if args_cli.torch_threads < 1:
        parser.error("--torch_threads must be positive")
    try:
        checkpoint = resolve_checkpoint(args_cli.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # Benchmark the shared plant directly; no stage or rendering is involved.
    if args_cli.mode == "benchmark":
        import torch

        from hydraulic_controller.benchmark import BenchmarkConfig, run_benchmark

        torch.set_num_threads(args_cli.torch_threads)
        try:
            output = run_benchmark(
                checkpoint, args_cli.benchmark_out, args_cli.device, BenchmarkConfig(kp=args_cli.kp)
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            parser.error(str(exc))
        print(f"[INFO] Benchmark complete: {output}")
        return 0

    # Default to the Kit viewer, except when headless (a Kit visualizer cannot be configured without a display).
    if not getattr(args_cli, "visualizer_explicit", False) and not args_cli.headless:
        args_cli.visualizer = ["kit"]
    app_launcher = AppLauncher(args_cli)
    run_interactive(args_cli, checkpoint, app_launcher.app)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
