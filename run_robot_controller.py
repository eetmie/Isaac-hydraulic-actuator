# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Export a portable controller or run it on the Jetson with kaivuriprokkis.

Shadow is sensor-only: the PCA9685 is not initialized. Rotate requires a passed
simulation report for this exact bundle/config plus a connected gamepad held
Left Bumper throughout motion. Run locally on the robot, not through UDP.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from hydraulic_controller.bundle import PolicyBundle, export_bundle
from hydraulic_controller.hardware import ImuReader, OutputGate
from hydraulic_controller.robot_geometry import policy_twist
from hydraulic_controller.rotation import FixedTipRotation


def validate_report(bundle, report_path):
    """Require completed simulation evidence for the exact actor and physical tool geometry."""
    report = json.loads(Path(report_path).read_text())
    if report["checkpoint_sha256"] != bundle.contract["checkpoint_sha256"]:
        raise ValueError("Rotation report is for another checkpoint")
    expected = bundle.contract["files"]["bucket_control_config.yaml"]
    if report.get("robot_profile_sha256") != expected:
        raise ValueError("Rotation report must use this exact robot geometry/profile")
    if not report["passed"] or report["cycles"] < 3 or not bundle.contract["sensor_trained"]:
        raise ValueError(
            "Live motion needs a sensor-trained policy and a passing three-cycle simulation report"
        )
    rows = report.get("rows", [])
    if not rows or report.get("rate_deg_s") != 3:
        raise ValueError("Rotation report has no cases or an unsupported reference rate")
    if {(r["amplitude_deg"], r["sensor_mode"]) for r in rows} != {
        (5.0, "ideal"),
        (5.0, "gyro"),
        (10.0, "ideal"),
        (10.0, "gyro"),
    } or any(r["failed"] or not np.isfinite(r["tip_max_mm"]) or r["tip_max_mm"] > 20 for r in rows):
        raise ValueError("Rotation report is incomplete or exceeds the simulated drift envelope")


def robot_run(args):
    FixedTipRotation(np.zeros(3), args.amplitude_deg, args.rate_deg_s)
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("--seconds must be finite and positive")
    bundle = PolicyBundle(args.bundle)
    repo = args.robot_repo.resolve()
    profile_dir = repo / "configuration_files/profiles" / args.robot
    bundle.check_profile(profile_dir)
    motion = args.mode == "rotate"
    if motion:
        if args.validation_report is None:
            raise ValueError("--validation_report is required for motion")
        validate_report(bundle, args.validation_report)
    sys.path.insert(0, str(repo))
    from modules.board import resolve_profile
    from modules.bringup import wait_for_hardware_ready
    from modules.hardware_interface import HardwareInterface

    profile = resolve_profile(args.robot)
    if (repo / profile["servo_config_file"]).resolve() != (profile_dir / "servo_config.yaml").resolve():
        raise ValueError("Board profile must resolve to the checksum-pinned servo configuration")
    configuration = bundle.robot_kin.profile
    if configuration["rates"]["control_hz"] != 100:
        raise ValueError("The hardware valve/history loop must run at 100 Hz")
    # Confirm alignment, not just an arbitrary full 3D mounting calibration.
    for mount in configuration["imu"]["mounting_offsets_quat"].values():
        if abs(mount[1]) + abs(mount[3]) > 1e-6:
            raise ValueError("This adapter requires pure Y mounting corrections")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    hardware = HardwareInterface(
        config_file=str((repo / profile["servo_config_file"]).resolve()),
        control_config_file=str(bundle.directory / "bucket_control_config.yaml"),
        enable_pwm=motion,
        enable_imu=True,
        enable_adc=False,
        start_adc_reader=False,
        pump_auto_mode=False,
        toggle_channels=True,
        stale_timeout_s=0.15,
        cleanup_disable_osc=False,
        pwm_i2c_bus=profile["pwm_i2c_bus"],
        pwm_i2c_addr=profile["pwm_i2c_addr"],
    )
    controller = pad = gate = None
    monitor_stop = threading.Event()
    monitor = None
    try:
        if motion:
            if not hardware.set_pump_enabled(False):
                raise RuntimeError("Hardware rejected pump disable")
            hardware.reset(reset_pump=True)
        wait_for_hardware_ready(hardware)
        if motion:
            from modules.excavator_controller import ExcavatorController
            from modules.gamepad import XboxController

            pad = XboxController()

            def deadman():
                return pad.is_connected() and bool(pad.LeftBumper) and not bool(pad.B)

            gate = OutputGate(hardware, deadman)
            hardware.send_named_pwm_commands = gate.write
            controller = ExcavatorController(
                hardware, control_config_file=str(bundle.directory / "bucket_control_config.yaml")
            )
            controller.enter_direct_command_mode(hold_timeout_s=0.15, decay_s=0.0, blend_s=0.0)
            controller.start()

            def supervise():
                while not monitor_stop.wait(0.01):
                    try:
                        gate.check()
                    except Exception as exc:
                        gate.stop(exc)

            monitor = threading.Thread(target=supervise, daemon=True)
            monitor.start()
        # Controller/gamepad construction may take seconds. Establish the device
        # clock baseline only afterwards; setup is not a sensor dropout.
        reader = ImuReader(hardware)
        q, _, _ = reader.read()
        bundle.history.reset(torch.tensor([0]), torch.tensor(q[None]))
        with torch.inference_mode():
            # Warm TorchScript before enforcing the live inference deadline.
            for _ in range(3):
                bundle.valves(bundle.history.observe(bundle.kin, torch.zeros(1, 3)))
        # Validate the full path before entering the timed loop, while pump and
        # valves remain off. In-loop IK would interrupt the 100 Hz sampling.
        q, _, _ = ImuReader(hardware).read()
        qt = torch.tensor(q[None])
        candidate = FixedTipRotation(
            bundle.robot_kin.pose_jacobian(qt)[0][0].numpy(), args.amplitude_deg, args.rate_deg_s, cycles=3
        )
        candidate.validate(bundle.robot_kin, qt)
        bundle.history.reset(torch.tensor([0]), qt)
        reader = ImuReader(hardware)
        print("Warming up measured histories with neutral commands for 1 second.", flush=True)
        step = 0
        started = next_tick = time.monotonic()
        sweep = None
        motion_start = None
        inference_ms = 0.0
        pending_command = np.zeros(3)
        with args.log.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "t_s",
                    "device_ts_us",
                    "q_boom",
                    "q_arm",
                    "q_bucket",
                    "q_pitch",
                    "v_boom",
                    "v_arm",
                    "v_bucket",
                    "v_pitch",
                    "tip_x_m",
                    "tip_z_m",
                    "angle_rad",
                    "tip_error_m",
                    "boom_u",
                    "arm_u",
                    "bucket_u",
                    "inference_ms",
                    "armed",
                ]
            )
            while True:
                tick = time.monotonic()
                if tick - next_tick > 0.03:
                    raise RuntimeError("Control observation loop missed its deadline by >30 ms")
                q, v, stamp = reader.read(tick)
                if gate is not None:
                    with gate.lock:
                        gate.sensor_time = reader.fresh_time
                        preceding = gate.last_command.copy()
                    if gate.fault is not None:
                        raise RuntimeError(gate.fault)
                else:
                    preceding = np.zeros(3)
                qt, vt = torch.tensor(q[None]), torch.tensor(v[None])
                if np.any(np.abs(v) > bundle.settings.velocity_limit):
                    raise ValueError("Measured angular velocity exceeds the controller envelope")
                if not bool(bundle.robot_kin.valid(qt)[0]):
                    raise ValueError("Measured pose is outside joint/collision margins")
                if abs(q[3]) > 0.015:
                    raise ValueError("Carriage pitch exceeds the level-ground demo envelope")
                bundle.history.push(qt, vt, torch.tensor(preceding[None], dtype=torch.float32))
                pose = bundle.robot_kin.pose_jacobian(qt)[0][0].numpy()
                error = 0.0 if sweep is None else float(np.linalg.norm(pose[:2] - sweep.initial[:2]))
                if motion and error > 0.02:
                    raise RuntimeError("Measured tip drift exceeded the 20 mm acceptance envelope")
                if step % bundle.settings.decimation == 0:
                    elapsed = tick - started
                    if sweep is None and elapsed >= 1.0:
                        if not motion or (pad.is_connected() and pad.LeftBumper and not pad.B):
                            angle_change = np.arctan2(
                                np.sin(pose[2] - candidate.initial[2]), np.cos(pose[2] - candidate.initial[2])
                            )
                            if np.linalg.norm(pose[:2] - candidate.initial[:2]) > 0.005 or abs(
                                angle_change
                            ) > np.deg2rad(1):
                                raise RuntimeError(
                                    "Robot moved since path preflight; restart while stationary"
                                )
                            sweep = candidate
                            motion_start = tick
                    t = 0.0 if motion_start is None else tick - motion_start
                    request = np.zeros(3) if sweep is None else sweep.command(t, pose)
                    before = time.monotonic()
                    with torch.inference_mode():
                        desired = policy_twist(
                            qt, torch.tensor(request[None], dtype=torch.float32), bundle.robot_kin, bundle.kin
                        )
                        admitted, _ = bundle.governor(qt, vt, desired)
                        valves = bundle.valves(bundle.history.observe(bundle.kin, admitted))[0].numpy()
                    inference_ms = (time.monotonic() - before) * 1000
                    if inference_ms > 40:
                        raise RuntimeError("Policy/geometry inference exceeded 40 ms")
                    pending_command = valves
                    if gate is not None:
                        with gate.lock:
                            gate.policy_time = time.monotonic()
                            if sweep is not None and not gate.armed:
                                gate.arm()
                                if not hardware.set_pump_enabled(True):
                                    raise RuntimeError("Hardware rejected pump enable")
                        controller.give_direct_commands(
                            dict(zip(("boom", "arm", "bucket"), valves.tolist())) if gate.armed else {}
                        )
                error = 0.0 if sweep is None else float(np.linalg.norm(pose[:2] - sweep.initial[:2]))
                writer.writerow(
                    [
                        tick - started,
                        stamp,
                        *q,
                        *v,
                        *pose,
                        error,
                        *pending_command,
                        inference_ms,
                        bool(gate and gate.armed),
                    ]
                )
                if step % 100 == 0:
                    stream.flush()
                    print(
                        f"t={tick - started:.1f}s tip_error={error * 1000:.1f}mm inference={inference_ms:.1f}ms "
                        f"{'ARMED' if gate and gate.armed else 'outputs disabled'}",
                        flush=True,
                    )
                if sweep is not None and tick - motion_start >= sweep.duration:
                    break
                if not motion and tick - started >= args.seconds:
                    break
                step += 1
                next_tick += 0.01
                time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        if gate is not None:
            gate.stop("run ended")
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=1.0)
        if controller is not None:
            controller.stop()
        if pad is not None:
            pad.stop_monitoring()
        hardware.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "shadow", "rotate"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--robot_repo", type=Path, required=True)
    parser.add_argument("--robot", default="jetson")
    parser.add_argument("--validation_report", type=Path)
    parser.add_argument("--amplitude_deg", type=float, choices=(5.0, 10.0), default=5.0)
    parser.add_argument("--rate_deg_s", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--log", type=Path, default=Path("runs/robot_rotation.csv"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "export":
        if args.checkpoint is None:
            parser.error("export requires --checkpoint")
        from hydraulic_controller.policy import ControllerPolicy

        export_bundle(
            ControllerPolicy(args.checkpoint),
            args.bundle,
            args.robot_repo / "configuration_files/profiles" / args.robot,
        )
        print(args.bundle.resolve())
    else:
        robot_run(args)


if __name__ == "__main__":
    main()
