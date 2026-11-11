# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Replay archived IMUs with the pinned, audited robot Fusion implementation.

Run ``python -m training.offline_fusion audit`` or ``replay INPUT.npz OUTPUT.npz``.
Gyro and accelerometer inputs are already calibrated, in dps and g respectively.
No gyro-bias estimator, velocity smoothing or fabricated 416 Hz samples are added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name("fusion")


def _settings() -> dict:
    text = (SOURCE / "settings.c").read_text()
    result = {
        key: float(re.search(r"\." + key + r"\s*=\s*([\d.]+)", text)[1])
        for key in ("ahrsGain", "ahrsAccelRejection", "ahrsRecoveryPeriodS", "gyroRangeDps", "accelRangeG")
    }
    table = (SOURCE / "sensor_config.h").read_text().split("#define ISM330_GYRO_FULL_SCALES", 1)[1]
    table = table.split("#define ISM330_ACCEL_FULL_SCALES", 1)[0]
    ranges = re.findall(r"\{\s*([\d.]+)f,\s*0x[\dA-Fa-f]+,\s*([\d.]+)f", table)
    sensitivity = next(float(value) for scale, value in ranges if float(scale) >= result["gyroRangeDps"])
    result["gyro_saturation_dps"] = 32768 * sensitivity
    return result


def _backend(name: str) -> tuple[Path, dict]:
    paths = sorted(p for p in SOURCE.rglob("*") if p.suffix in (".c", ".h", ".json"))
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()[:16]
    binary = ROOT / "local/fusion_cache" / fingerprint / name
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        fusion = SOURCE / "Fusion"
        subprocess.run(
            [
                "cc",
                "-O2",
                f"-I{fusion}",
                f"-I{SOURCE}",
                str(SOURCE / f"{name}.c"),
                str(SOURCE / "settings.c"),
                str(fusion / "InitFusion.c"),
                str(fusion / "FusionAhrs.c"),
                str(fusion / "FusionOffset.c"),
                "-lm",
                "-o",
                str(binary),
            ],
            check=True,
        )
    return binary, _settings()


def _run(name: str, device_us, gyro, accel, seed, *, cold=False):
    binary, settings = _backend(name)
    dt = np.r_[0.0, np.diff(device_us) * 1e-6]
    period = float(np.median(dt[1:])) if len(dt) > 1 else 0.005
    recovery = round(settings["ahrsRecoveryPeriodS"] / period)
    quats, flags = [], []
    for role in range(4):
        records = np.c_[dt, gyro[:, role], accel[:, role]].astype("<f4")
        header = struct.pack(
            "<i8f",
            len(dt),
            settings["ahrsGain"],
            settings["ahrsAccelRejection"],
            recovery,
            settings["gyro_saturation_dps"],
            *seed[role],
        )
        command = [str(binary), *(["--cold"] if cold else [])]
        data = subprocess.run(
            command, input=header + records.tobytes(), capture_output=True, check=True
        ).stdout
        if len(data) != len(dt) * 17:
            raise ValueError("Truncated Fusion replay output")
        quats.append(np.frombuffer(data[: len(dt) * 16], "<f4").reshape(-1, 4))
        flags.append(np.frombuffer(data[len(dt) * 16 :], np.uint8))
    return np.stack(quats, axis=1), np.stack(flags, axis=1)


def replay(device_us, gyro_dps, accel_g, recorded_quats, *, settling_seconds=30.0):
    """Return replayed quaternions, recovery flags and settled mask for four IMUs.

    Inputs use base/boom/arm/bucket order. Each gap over 10 ms starts a new segment
    seeded from that segment's recorded quaternion. Invalid sensor samples split
    segments; isolated nonzero low-g samples remain valid, as in the live firmware.
    """
    t = np.asarray(device_us, dtype=np.float64)
    g, a, seeds = map(np.asarray, (gyro_dps, accel_g, recorded_quats))
    n = len(t)
    if n == 0 or g.shape != (n, 4, 3) or a.shape != g.shape or seeds.shape != (n, 4, 4):
        raise ValueError("Expected device times, four gyro/accel vectors and four wxyz quaternions per frame")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Unwrapped device timestamps must be finite and strictly increasing")
    valid = np.isfinite(g).all(axis=(1, 2)) & np.isfinite(a).all(axis=(1, 2))
    valid &= np.isfinite(seeds).all(axis=(1, 2)) & (np.linalg.norm(seeds, axis=2) > 0.5).all(axis=1)
    valid &= (np.linalg.norm(a, axis=2) > 0).all(axis=1)
    quats = np.full((n, 4, 4), np.nan, dtype=np.float32)
    flags = np.full((n, 4), 255, dtype=np.uint8)
    settled = np.zeros(n, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        cuts = [start, *(start + 1 + np.flatnonzero(np.diff(t[start:stop]) > 10000)), stop]
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            quats[lo:hi], flags[lo:hi] = _run("replay", t[lo:hi], g[lo:hi], a[lo:hi], seeds[lo])
            settled[lo:hi] = t[lo:hi] - t[lo] >= settling_seconds * 1e6
    return quats, flags, settled


def audit() -> dict:
    """Compare the host wrapper to the actual firmware update block, plus physical yaw behavior."""
    results = []
    for rate in (200, 416):
        n = rate * 12
        t = np.rint(np.arange(n) * 1e6 / rate)
        seed = np.tile([1.0, 0, 0, 0], (4, 1))
        for case in ("still", "slow_yaw", "acceleration_recovery", "gyro_saturation", "cold_start"):
            gyro = np.zeros((n, 4, 3))
            accel = np.tile([0.0, 0, 1], (n, 4, 1))
            if case == "slow_yaw":
                gyro[:, :, 2] = 0.5
            if case == "acceleration_recovery":
                accel[rate : rate * 5] = [-np.sin(0.35), 0, np.cos(0.35)]
            if case == "gyro_saturation":
                gyro[:, :, 2] = 30
                gyro[rate * 2, :, 2] = 600
            if case == "cold_start":
                accel[:] = [-np.sin(np.deg2rad(20)), 0, np.cos(np.deg2rad(20))]
            q, flags = _run("replay", t, gyro, accel, seed, cold=case == "cold_start")
            reference, ref_flags = _run("firmware_reference", t, gyro, accel, seed, cold=case == "cold_start")
            difference = float(np.max(np.abs(q - reference)))
            if difference > 2e-6 or not np.array_equal(flags, ref_flags):
                raise ValueError(f"Firmware parity failed at {rate} Hz for {case}: {difference}")
            if case == "slow_yaw":
                yaw = np.unwrap(2 * np.arctan2(q[:, 0, 3], q[:, 0, 0]))
                if abs(np.degrees(yaw[-1]) - (n - 1) * 0.5 / rate) > 0.01:
                    raise ValueError("Slow yaw was incorrectly absorbed into bias")
            results.append(dict(rate_hz=rate, case=case, max_quaternion_component_difference=difference))
    return dict(settings=_settings(), comparisons=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("audit")
    play = commands.add_parser("replay")
    play.add_argument("input", type=Path)
    play.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "audit":
        print(json.dumps(audit(), indent=2))
        return
    if args.output.exists():
        raise FileExistsError("Use a new output path; archived data is never replaced in place")
    with np.load(args.input, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    q, flags, settled = replay(
        data["device_ts_us_unwrapped"],
        data["canonical_gyro_dps"],
        data["canonical_accel_g"],
        data["canonical_quat_wxyz"],
    )
    data.update(canonical_quat_wxyz=q, fusion_flags=flags, fusion_settled=settled)
    data["valid"] = data.get("valid", np.ones(len(q), dtype=bool)) & settled
    np.savez_compressed(args.output, **data)


if __name__ == "__main__":
    main()
