# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared causal IMU conversion for recorded and live, axis-aligned excavator IMUs."""

from __future__ import annotations

import numpy as np

ROLES = ("base", "boom", "arm", "bucket")
# Fixed historical calibration, also used by the robot's Jetson profile. Y rotations
# change pitch zero but leave the physical gyro Y axis unchanged.
MOUNT_PITCH = 2 * np.arctan2([0, 0.12057, 0.0087265, 0], [1, 0.9927048, 0.9999619, 1])
CARRIAGE_PITCH_ZERO = -0.004821687005460262
SENSOR_CONTRACT = {
    "version": 1,
    "roles": list(ROLES),
    "velocity_source": "aligned_y_child_minus_parent",
    "software_filter": "none",
    "sampling": "latest_complete_packet_not_after_tick",
    "bias": "firmware_startup_correction_only",
    "carriage_pitch_zero_rad": CARRIAGE_PITCH_ZERO,
    "mount_pitch_rad": MOUNT_PITCH.tolist(),
}


def policy_joint_offset(imu: dict, contract: dict = SENSOR_CONTRACT) -> np.ndarray:
    """Map runtime joint angles to training calibration by a fixed offset [rad].

    With parallel Y axes, mounting rotations do not affect gyro Y rates.
    A corrected link pitch is raw pitch minus its mounting angle. Therefore
    training-relative positions equal runtime positions plus adjacent
    differences of (runtime mounting - training mounting).
    """
    runtime = []
    for role in contract["roles"]:
        quat = np.asarray(imu["mounting_offsets_quat"][role], dtype=float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or abs(quat[1]) + abs(quat[3]) > 1e-6:
            raise ValueError("Calibration transfer requires finite pure-Y mounting quaternions")
        if np.linalg.norm(quat) < 0.5:
            raise ValueError("Mounting quaternion must be nonzero")
        runtime.append(2 * np.arctan2(quat[2], quat[0]))
    delta = np.asarray(runtime) - np.asarray(contract["mount_pitch_rad"])
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    return np.r_[np.diff(delta), delta[0]].astype(np.float32)


def aligned_rates(gyro_y_dps: np.ndarray) -> np.ndarray:
    """Relative boom/arm/bucket rates and base pitch rate [rad/s].

    Input [..., 4] is same-packet, bias-corrected gyro Y [deg/s], in ROLES order.
    All four positive Y axes must point in the same physical direction. The
    fourth rate describes the planar base pitch; roll/slew motion is out of scope.
    """
    g = np.asarray(gyro_y_dps, dtype=np.float64)
    if g.shape[-1] != 4 or not np.isfinite(g).all():
        raise ValueError("Expected four finite, synchronized gyro Y channels")
    return np.deg2rad(np.concatenate((np.diff(g, axis=-1), g[..., :1]), axis=-1)).astype(np.float32)


def imu_positions(quats: np.ndarray, *, corrected: bool = False) -> np.ndarray:
    """Relative joint angles and fixed-reference carriage pitch [rad].

    Input [..., 4, 4] contains wxyz quaternions in ROLES order. Hardware already
    removes mounting offsets; raw recording quaternions do not.
    """
    q = np.asarray(quats, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if q.shape[-2:] != (4, 4) or not np.isfinite(q).all() or np.any(norm < 0.5):
        raise ValueError("Expected four finite, nonzero IMU quaternions")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    pitch = np.arctan2(-2 * (x * z - w * y), 1 - 2 * (x * x + y * y))
    if not corrected:
        pitch -= MOUNT_PITCH
    relative = np.diff(pitch, axis=-1)
    relative = np.arctan2(np.sin(relative), np.cos(relative))
    return np.concatenate((relative, pitch[..., :1] - CARRIAGE_PITCH_ZERO), axis=-1).astype(np.float32)


def causal_indices(sample_times: np.ndarray, ticks: np.ndarray, max_age: float = 0.03) -> np.ndarray:
    """Index the most recent available packet at each tick [s]; reject gaps."""
    times, ticks = np.asarray(sample_times), np.asarray(ticks)
    if not np.isfinite(times).all() or not np.isfinite(ticks).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Packet clock must be finite and strictly increasing")
    ids = np.searchsorted(times, ticks, side="right") - 1
    if np.any(ids < 0) or np.any(ticks - times[ids] > max_age):
        raise ValueError("No sufficiently recent causal packet for tick")
    return ids
