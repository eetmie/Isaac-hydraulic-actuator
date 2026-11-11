# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Firmware parity and observable yaw/gap behavior of the offline reader."""

import numpy as np

from training.offline_fusion import audit, replay


def test_replay_matches_the_actual_firmware_update_and_cold_start():
    report = audit()
    assert max(row["max_quaternion_component_difference"] for row in report["comparisons"]) < 2e-6


def test_yaw_continues_and_low_gravity_is_not_mistaken_for_sensor_loss():
    n = 800
    t = np.arange(n) * 5000.0
    t[200:] += 100000
    yaw = np.deg2rad(0.5 * t * 1e-6)
    seeds = np.zeros((n, 4, 4))
    seeds[:, :, 0] = np.cos(yaw[:, None] / 2)
    seeds[:, :, 3] = np.sin(yaw[:, None] / 2)
    gyro = np.zeros((n, 4, 3))
    gyro[:, :, 2] = 0.5
    accel = np.tile([0.0, 0, 1], (n, 4, 1))
    accel[400] = [0.0, 0, 0.05]  # A real transient, permitted by the live algorithm.
    accel[600] = 0  # Missing information splits the stream.
    q, flags, settled = replay(t, gyro, accel, seeds, settling_seconds=0.1)
    actual = np.unwrap(2 * np.arctan2(q[:600, 0, 3], q[:600, 0, 0]))
    np.testing.assert_allclose(actual, yaw[:600], atol=2e-6)
    assert np.isfinite(q[400]).all() and settled[400]
    assert not settled[200] and settled[221]
    assert np.isnan(q[600]).all() and np.all(flags[600] == 255)
    assert not settled[601] and settled[622]
