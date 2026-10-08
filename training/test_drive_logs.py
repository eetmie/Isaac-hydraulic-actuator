# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""drive_log/imu_raw pairs to training chunks: causal sampling, splits, labels."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hydraulic_controller.sensors import MOUNT_PITCH, ROLES, aligned_rates
from training import drive_logs

HZ = 2.0  # test motion frequency [Hz]


def link_pitch(t):
    """Absolute link pitches in ROLES order (base, boom, arm, bucket) [rad]."""
    w = 2 * np.pi * HZ
    return np.stack([0.002 * np.sin(w * t), 0.3 * np.sin(w * t), 0.2 * np.cos(w * t) - 0.5, 0.1 * t], -1)


def link_rate(t):
    w = 2 * np.pi * HZ
    return np.stack(
        [0.002 * w * np.cos(w * t), 0.3 * w * np.cos(w * t), -0.2 * w * np.sin(w * t), 0.1 + 0 * t], -1
    )


def write_pair(tmp_path, stale=(), drop=()):
    """A 20 s recording in kaivuriprokkis' schema; the Pico clock wraps mid-run."""
    rng = np.random.default_rng(3)
    frame_t = np.arange(0, 20, 0.005)
    device_us = (2**32 - 7_000_000 + np.round(frame_t * 1e6).astype(np.int64)) % 2**32
    pitch = link_pitch(frame_t) + MOUNT_PITCH  # raw quaternions still carry the mounting
    raw = {"device_ts_us": device_us}
    for i, role in enumerate(ROLES):
        raw.update(
            {
                f"imu_{role}_qw": np.cos(pitch[:, i] / 2),
                f"imu_{role}_qx": 0 * frame_t,
                f"imu_{role}_qy": np.sin(pitch[:, i] / 2),
                f"imu_{role}_qz": 0 * frame_t,
                f"imu_{role}_gy_dps": np.rad2deg(link_rate(frame_t)[:, i]),
            }
        )
    rows_t = np.arange(0, 20, 0.01) + rng.uniform(0, 0.002, 2000)
    frame = np.searchsorted(frame_t, rows_t, side="right") - 1
    u = np.repeat(rng.uniform(-1, 1, (200, 3)), 10, 0)
    log = {
        "timestamp": rows_t,
        "sample_idx": np.arange(2000),
        "excitation_mode": np.full(2000, "circle_mlp"),
        "state_imu_ts_us": device_us[frame],
        "cmd_stale": np.isin(np.arange(2000), stale).astype(int),
        "cmd_age_s": np.full(2000, 0.0005),
        **{c: u[:, i] for i, c in enumerate(drive_logs.COMMANDS)},
    }
    keep = ~np.isin(np.arange(2000), drop)
    for name, table in (("drive_log", log), ("imu_raw", raw)):
        with (tmp_path / f"{name}_20261008_120000_circle_mlp_ccw.csv").open("w") as stream:
            stream.write(",".join(table) + "\n")
            rows = zip(*(np.asarray(v)[keep] if name == "drive_log" else v for v in table.values()))
            stream.writelines(",".join(map(str, r)) + "\n" for r in rows)
    return tmp_path / "drive_log_20261008_120000_circle_mlp_ccw.csv", rows_t, u


def test_chunks_sample_the_last_frame_and_hold_the_command(tmp_path):
    path, rows_t, u = write_pair(tmp_path)
    (chunk,), report = drive_logs.load_pair(path)
    assert report["live_rows"] == 2000
    np.testing.assert_allclose(np.diff(chunk["t"]), 0.01, atol=1e-9)
    # Each tick holds the last row's command and that row's pose frame; never a later one.
    row = np.searchsorted(rows_t, chunk["t"], side="right") - 1
    np.testing.assert_array_equal(chunk["u"], u[row].astype(np.float32))
    frame_t = np.floor(rows_t[row] / 0.005 + 1e-9) * 0.005
    truth = link_pitch(frame_t)
    np.testing.assert_allclose(chunk["q"][:, :3], np.diff(truth, axis=1), atol=1e-5)
    np.testing.assert_allclose(chunk["v"], aligned_rates(np.rad2deg(link_rate(frame_t))), atol=1e-5)
    np.testing.assert_array_equal(chunk["target_v"], chunk["v"])
    np.testing.assert_array_equal(chunk["target_q"], chunk["q"])


def test_stale_rows_and_dropped_samples_split_instead_of_bridging(tmp_path):
    path, _, _ = write_pair(tmp_path, stale=range(600, 650), drop=[1500])
    chunks, report = drive_logs.load_pair(path)
    assert report["live_rows"] == 1949 and len(chunks) == 3
    for chunk in chunks:
        assert not ((chunk["t"] > 5.99) & (chunk["t"] < 6.5)).any()
    assert all(np.diff(c["t"]).max() < 0.0101 for c in chunks)


def test_smooth21_label_follows_slow_motion(tmp_path):
    path, _, _ = write_pair(tmp_path)
    (smooth,), _ = drive_logs.load_pair(path, target_v="smooth21")
    (gyro,), _ = drive_logs.load_pair(path)
    np.testing.assert_array_equal(smooth["q"], gyro["q"])
    # At 2 Hz the smoothed derivative of q matches the gyro rate, up to the
    # 0-10 ms frame jitter of the positions it differentiates.
    error = smooth["target_v"][:, :3] - gyro["v"][:, :3]
    assert np.sqrt((error**2).mean()) < 0.1 * np.sqrt((gyro["v"][:, :3] ** 2).mean())


def test_prepare_writes_training_chunks_and_refuses_repeats(tmp_path):
    path, _, _ = write_pair(tmp_path)
    args = SimpleNamespace(logs=[path], split="dev", out=tmp_path / "run", min_seconds=3.0, target_v="gyro")
    drive_logs.prepare(args)
    manifest = json.loads((tmp_path / "run/drive_logs_dev.json").read_text())
    (entry,) = manifest["recordings"]
    assert manifest["target_v"] == "gyro" and entry["chunks"] == 1
    with np.load(tmp_path / "run" / entry["paths"][0]) as chunk:
        assert set(chunk.files) == {"t", "q", "v", "u", "target_q", "target_v"}
    with pytest.raises(FileExistsError):
        drive_logs.prepare(args)
    with pytest.raises(ValueError, match="one label per split"):
        drive_logs.prepare(SimpleNamespace(**{**vars(args), "target_v": "smooth21"}))
