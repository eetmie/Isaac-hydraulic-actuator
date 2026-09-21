# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hardware-independent parts of the robot adapter, including a latched output gate."""

from __future__ import annotations

import threading
import time

import numpy as np

from .sensors import ROLES, aligned_rates, imu_positions


class ImuReader:
    """Read one immutable same-packet snapshot from kaivuriprokkis.

    Its current public quaternion and gyro getters lock separately. This adapter
    deliberately reads its already-published immutable snapshot once, avoiding a
    torn pair. Unsupported snapshot layouts fail closed instead of falling back
    to independently sampled public getters.
    """

    def __init__(self, hardware, max_age_s=0.05):
        self.hardware = hardware
        self.max_age = max_age_s
        self.previous_device_us = None
        self.fresh_time = None
        self.clock_lag = 0.0

    def read(self, now=None):
        """Return q [rad], qdot [rad/s], and the source device clock [us]."""
        now = time.monotonic() if now is None else now
        if not self.hardware.is_hardware_ready():
            raise RuntimeError("IMU hardware is not ready")
        snapshot = getattr(self.hardware, "_imu_snapshot", None)
        if snapshot is None or snapshot.device_ts is None:
            raise RuntimeError("Atomic timestamped IMU snapshot is required")
        stamp = int(snapshot.device_ts)
        if self.previous_device_us is None:
            self.fresh_time = now
        elif stamp != self.previous_device_us:
            delta = (stamp - self.previous_device_us) % 2**32
            if not 0 < delta <= 100000:
                raise RuntimeError("IMU clock reset, reversed, or skipped more than 100 ms")
            self.clock_lag = max(0.0, self.clock_lag + now - self.fresh_time - delta * 1e-6)
            self.fresh_time = now
        self.previous_device_us = stamp
        if now - self.fresh_time > self.max_age or self.clock_lag > self.max_age:
            raise RuntimeError("IMU packets are stale or falling behind the host clock")
        roles = self.hardware._imu_joint_roles
        gyros = {role: gyro for role, gyro in zip(roles, snapshot.imu_gyro, strict=True)}
        gyros["base"] = snapshot.base_imu_gyro
        quats = np.stack([snapshot.imu_by_role[role] for role in ROLES])
        gyro_y = np.array([gyros[role][1] for role in ROLES])
        return imu_positions(quats, corrected=True), aligned_rates(gyro_y), stamp


class OutputGate:
    """Serialize every valve write against an independent, latched pump-off stop.

    The robot controller still owns the 100 Hz output thread. This gate checks
    freshness and deadman state on every write, and a monitor calls ``check``
    even if the controller/inference thread stalls. Re-arming requires a new run.
    """

    def __init__(self, hardware, deadman, clock=time.monotonic):
        self.hardware, self.deadman, self.clock = hardware, deadman, clock
        self.lock = threading.RLock()
        self.armed = False
        self.fault = None
        self.sensor_time = None
        self.policy_time = None
        self.last_command = np.zeros(3, dtype=np.float32)
        self.original_write = hardware.send_named_pwm_commands

    def arm(self):
        """Enable output after valid sensor/policy heartbeats and operator deadman."""
        with self.lock:
            if self.fault is not None:
                raise RuntimeError(f"Stop is latched: {self.fault}")
            self.armed = True
            self.check()
            if not self.armed:
                raise RuntimeError(self.fault)

    def stop(self, reason):
        """Latch neutral valves and pump off, serialized with all output writes."""
        with self.lock:
            self.fault = self.fault or str(reason)
            self.armed = False
            self.last_command[:] = 0
            self.hardware.set_pump_enabled(False)
            self.hardware.reset(reset_pump=True)

    def check(self):
        """Stop on released deadman, >50 ms stale sensors, or >150 ms stale policy."""
        with self.lock:
            if not self.armed:
                return
            now = self.clock()
            if not self.deadman():
                self.stop("operator enable released/disconnected")
            elif self.sensor_time is None or now - self.sensor_time > 0.05:
                self.stop("sensor timeout")
            elif self.policy_time is None or now - self.policy_time > 0.15:
                self.stop("policy timeout")

    def write(self, commands, **kwargs):
        """Gate normalized valve output; slew/tracks/other channels always receive zero."""
        with self.lock:
            self.check()
            values = np.array([commands.get(name, 0.0) for name in ("boom", "arm", "bucket")], dtype=float)
            if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
                self.stop("invalid valve command")
            if not self.armed:
                values[:] = 0
            safe = dict(zip(("boom", "arm", "bucket"), values.tolist()))
            safe.update(slew=0.0, trackL=0.0, trackR=0.0)
            okay = self.original_write(safe, **kwargs)
            if not okay:
                self.stop("hardware rejected PWM write")
                return False
            self.last_command[:] = values
            return True
