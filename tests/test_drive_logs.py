# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""drive_log/imu_raw pairs to training chunks: causal sampling, splits, labels."""

import csv
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hydraulic_controller.sensors import MOUNT_PITCH, ROLES, SENSOR_CONTRACT, aligned_rates
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


def test_prepare_writes_training_chunks_and_refuses_repeats(tmp_path):
    path, _, _ = write_pair(tmp_path)
    args = SimpleNamespace(logs=[path], split="dev", out=tmp_path / "run", min_seconds=3.0)
    drive_logs.prepare(args)
    manifest = json.loads((tmp_path / "run/drive_logs_dev.json").read_text())
    (entry,) = manifest["recordings"]
    assert manifest["target_v"] == "gyro" and entry["chunks"] == 1
    with np.load(tmp_path / "run" / entry["paths"][0]) as chunk:
        assert set(chunk.files) == {"t", "q", "v", "u", "target_q", "target_v"}
    with pytest.raises(FileExistsError):
        drive_logs.prepare(args)
    manifest["target_v"] = "legacy_smoothed"
    (tmp_path / "run/drive_logs_dev.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different velocity targets"):
        drive_logs.prepare(args)


def native_pair(tmp_path, **overrides):
    path, times, commands = write_pair(tmp_path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    defaults = dict(pump_on=1, write_ok=1, imu_valid=1, input_valid=1, state_age_s=0.005, dry_pwm=0)
    defaults.update(overrides)
    for i, row in enumerate(rows):
        for key, value in defaults.items():
            row[key] = value(i) if callable(value) else value
        row["timestamp"] = times[i] + 1000
        row["post_write_ns"] = int((times[i] + 1000) * 1e9)
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    excitation = path.with_name(path.name.replace("drive_log_", "excitation_")).with_suffix(".json")
    excitation.write_text('{"version":3,"blocks":[]}\n')
    manifest = {
        "format": "kaivuri-native-mlp-v1",
        "complete": True,
        "dropped": 0,
        "sensor_contract": {**SENSOR_CONTRACT, "axis_aligned_y": True},
        "outputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (path, drive_logs.raw_path_for(path), excitation)
        },
    }
    session = path.with_name(path.name.replace("drive_log_", "session_")).with_suffix(".json")
    session.write_text(json.dumps(manifest))
    return path, times + 1000, commands, session


def test_native_data_quality_flags_split_the_recording(tmp_path):
    path, times, _, _ = native_pair(
        tmp_path,
        pump_on=lambda i: int(not 200 <= i < 250),
        imu_valid=lambda i: int(not 600 <= i < 650),
        state_age_s=lambda i: 0.1 if 1000 <= i < 1050 else 0.005,
        write_ok=lambda i: int(not 1400 <= i < 1450),
        input_valid=lambda i: int(not 1800 <= i < 1850),
    )
    chunks, report = drive_logs.load_pair(path, min_seconds=0.2)
    assert report["live_rows"] == 1750
    for a, b in ((200, 250), (600, 650), (1000, 1050), (1400, 1450), (1800, 1850)):
        assert all(not ((chunk["t"] >= times[a]) & (chunk["t"] <= times[b - 1])).any() for chunk in chunks)


def test_native_commit_marker_and_hashes_are_required(tmp_path):
    path, _, _, session = native_pair(tmp_path)
    manifest = json.loads(session.read_text())
    manifest["complete"] = False
    session.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete native"):
        drive_logs.load_pair(path)
    manifest["complete"] = True
    session.write_text(json.dumps(manifest))
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        drive_logs.load_pair(path)
    session.unlink()
    with pytest.raises(ValueError, match="manifest missing"):
        drive_logs.load_pair(path)


def test_native_profile_calibration_is_applied_exactly_once(tmp_path):
    path, _, _, session = native_pair(tmp_path)
    (baseline,), _ = drive_logs.load_pair(path)
    manifest = json.loads(session.read_text())
    manifest["sensor_contract"]["runtime_mount_pitch_rad"] = [0, 0.2, 0.1, 0]
    session.write_text(json.dumps(manifest))
    (calibrated,), report = drive_logs.load_pair(path)
    np.testing.assert_array_equal(calibrated["q"], baseline["q"])
    np.testing.assert_array_equal(calibrated["v"], baseline["v"])
    assert report["sensor_contract"]["mount_pitch_rad"] == MOUNT_PITCH.tolist()
    manifest["sensor_contract"]["mount_pitch_rad"][2] += 0.1
    session.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixed training calibration"):
        drive_logs.load_pair(path)


def test_native_jittered_timestamps_hold_only_prior_commands(tmp_path):
    path, times, commands, _ = native_pair(tmp_path)
    (chunk,), _ = drive_logs.load_pair(path)
    rows = np.searchsorted(times, chunk["t"], side="right") - 1
    assert (times[rows] <= chunk["t"]).all()
    np.testing.assert_array_equal(chunk["u"], commands[rows].astype(np.float32))
    np.testing.assert_allclose(np.diff(chunk["t"]), 0.01, atol=1e-9)


def test_native_session_cannot_leak_between_splits(tmp_path):
    path, _, _, session = native_pair(tmp_path)
    metadata = json.loads(session.read_text())
    metadata.update(collection_id="one-collection", source_revision="test-revision", suffix="chirp_lift")
    session.write_text(json.dumps(metadata))
    args = SimpleNamespace(
        logs=[path], split="dev", out=tmp_path / "experiment", min_seconds=3.0, target_v="gyro"
    )
    drive_logs.prepare(args)
    entry = json.loads((args.out / "drive_logs_dev.json").read_text())["recordings"][0]
    assert entry["source"] == "native_mlp" and entry["source_session_id"] == "one-collection"
    assert entry["operator_suffix"] == "chirp_lift"
    assert entry["excitation"]["sha256"]
    with pytest.raises(ValueError, match="another train/dev/test split"):
        drive_logs.prepare(SimpleNamespace(**{**vars(args), "split": "train"}))


def test_native_future_raw_availability_is_rejected(tmp_path):
    path, _, _, session = native_pair(tmp_path)
    raw = drive_logs.raw_path_for(path)
    with raw.open() as stream:
        rows = list(csv.DictReader(stream))
    for i, row in enumerate(rows):
        # The middle second was not available until after its named drive rows.
        row["host_read_ns"] = int((1000 + i * 0.005 + (0.05 if 1000 <= i < 1200 else 0)) * 1e9)
    with raw.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata = json.loads(session.read_text())
    metadata["outputs"][raw.name] = hashlib.sha256(raw.read_bytes()).hexdigest()
    session.write_text(json.dumps(metadata))
    chunks, report = drive_logs.load_pair(path)
    assert 1898 <= report["live_rows"] <= 1902
    assert len(chunks) == 2
    assert all(not ((chunk["t"] >= 1005) & (chunk["t"] < 1006)).any() for chunk in chunks)


def test_native_dry_data_never_becomes_training_samples(tmp_path):
    path, _, _, session = native_pair(tmp_path, dry_pwm=1)
    chunks, report = drive_logs.load_pair(path)
    assert chunks == [] and report["live_rows"] == 0
    # Manifest-level dry status also excludes an inconsistent CSV claiming live PWM.
    metadata = json.loads(session.read_text())
    metadata["dry_pwm"] = True
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["dry_pwm"] = "0"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata["outputs"][path.name] = drive_logs.digest(path)
    session.write_text(json.dumps(metadata))
    chunks, report = drive_logs.load_pair(path)
    assert chunks == [] and report["live_rows"] == 0


def pid_pair(tmp_path):
    path, times, commands, session = native_pair(
        tmp_path, control_source_code=1, controller_valid=1, sine_enabled=0
    )
    metadata = json.loads(session.read_text())
    # A distinct take name allows both applications in one training split.
    renamed = {}
    for name in metadata["outputs"]:
        old = tmp_path / name
        new = old.with_name(old.stem + "_pid" + old.suffix)
        old.rename(new)
        renamed[new.name] = drive_logs.digest(new)
        if old == path:
            path = new
    metadata["outputs"] = renamed
    new_session = drive_logs.companion_path(path, "session", ".json")
    session.unlink()
    session = new_session
    metadata["collector"] = "kaivuri-pid-collect"
    metadata["collection_id"] = "pid-session"
    control = drive_logs.companion_path(path, "pid_control", ".csv")
    protocol = drive_logs.companion_path(path, "pid_protocol", ".json")
    control.write_text("post_write_ns,active\n1000,1\n")
    protocol.write_text('{"version":1,"segments":[]}\n')
    metadata["attachments"] = {p.name: drive_logs.digest(p) for p in (control, protocol)}
    session.write_text(json.dumps(metadata))
    return path, session, control, protocol


def test_pid_and_manual_collectors_combine_with_same_unsmoothed_targets(tmp_path):
    manual = tmp_path / "manual"
    manual.mkdir()
    pid = tmp_path / "pid"
    pid.mkdir()
    manual_path, _, _, _ = native_pair(manual)
    pid_path, _, _, _ = pid_pair(pid)
    chunks, report = drive_logs.load_pair(pid_path)
    assert report["collector"] == "kaivuri-pid-collect" and report["pid_rows"] == 2000
    assert report["manual_rows"] == 0 and report["idle_rows"] == 0
    assert report["attachments"]
    for chunk in chunks:
        np.testing.assert_array_equal(chunk["target_v"], chunk["v"])
    out = tmp_path / "combined"
    drive_logs.prepare(
        SimpleNamespace(
            logs=[manual_path, pid_path], split="train", out=out, min_seconds=3.0, target_v="gyro"
        )
    )
    manifest = json.loads((out / "drive_logs_train.json").read_text())
    assert len(manifest["recordings"]) == 2 and manifest["target_v"] == "gyro"
    assert {r["collector"] for r in manifest["recordings"]} == {"kaivuri-mlp-collect", "kaivuri-pid-collect"}


def test_pid_missing_or_changed_sidecar_is_rejected(tmp_path):
    path, session, control, _ = pid_pair(tmp_path)
    metadata = json.loads(session.read_text())
    metadata["attachments"].pop(control.name)
    session.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete PID recording attachments"):
        drive_logs.load_pair(path)
    metadata["attachments"][control.name] = drive_logs.digest(control)
    session.write_text(json.dumps(metadata))
    with control.open("a") as stream:
        stream.write("2000,0\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        drive_logs.load_pair(path)
