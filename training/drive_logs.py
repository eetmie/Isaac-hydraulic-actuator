# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Import the robot's drive_log/imu_raw pairs as raw-gyro staging chunks.

kaivuriprokkis writes the same pair from operator recordings (simple_drive.py)
and controller runs (learned_control/run_circle.py). Run as
``python -m training.drive_logs prepare --split train --out runs/<name> --logs drive_log_*.csv``.
Each continuous usable stretch becomes ``<out>/data/<split>/<log stem>_NNN.npz``.
Review and add these recordings to the canonical complete corpus before preparing
frozen training splits with ``training.prepare``. Staging manifests are
not accepted directly by the trainer. Arrays follow its sensor contract:

    t          uniform 100 Hz ticks on the drive log's clock [s]
    q          joint angles and carriage pitch from the last IMU frame at each tick [rad]
    v          same-frame aligned gyro rates [rad/s] (the model's velocity input)
    u          valve command held at each tick, lift/tilt/scoop [-1, 1]
    target_q   q, the rollout position target
    target_v   v itself: same-packet relative gyro rates, without smoothing.

A tick is usable when its drive-log row has a live command (cmd_stale=0, command
age <= 30 ms), the rows are consecutive, and the IMU frame its pose came from is
in the raw strip. Anything else splits the timeline instead of being bridged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hydraulic_controller.sensors import ROLES, SENSOR_CONTRACT, aligned_rates, imu_positions

DT = 0.01
MAX_AGE = 0.03  # same freshness bound as causal_indices
EDGE_TRIM = 10  # Discard edge samples before exposing a continuous training chunk.
COMMANDS = ("combined_cmd_lift", "combined_cmd_tilt", "combined_cmd_scoop")


def unwrap_clock(values):
    """Unwrap the Pico's unsigned 32-bit microsecond clock."""
    raw = np.asarray(values, dtype=np.float64)
    return raw + np.cumsum(np.r_[0, np.diff(raw) < -(2**31)]) * 2**32


def read_columns(path, names, optional=()):
    """Numeric CSV columns by name; empty fields read as NaN. Missing optional columns are None."""
    header = Path(path).open().readline().strip().split(",")
    missing = [n for n in names if n not in header]
    if missing:
        raise ValueError(f"{Path(path).name} lacks {', '.join(missing)}")
    wanted = [n for n in (*names, *optional) if n in header]
    data = np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=[header.index(n) for n in wanted], dtype=np.float64
    )
    data = data.reshape(-1, len(wanted))
    out = dict(zip(wanted, data.T, strict=True))
    return {n: out.get(n) for n in (*names, *optional)}


def companion_path(drive_log, prefix, suffix):
    drive_log = Path(drive_log)
    stem = drive_log.stem.removeprefix("drive_log_")
    return drive_log.with_name(f"{prefix}_{stem}{suffix}")


def raw_path_for(drive_log):
    """The imu_raw strip written with a drive log: same name, other prefix."""
    drive_log = Path(drive_log)
    if not drive_log.name.startswith("drive_log_"):
        raise ValueError(f"Expected a drive_log_*.csv file, got {drive_log.name}")
    raw = companion_path(drive_log, "imu_raw", ".csv")
    if not raw.exists():
        raise FileNotFoundError(f"No imu_raw strip next to {drive_log.name}")
    return raw


def runs(mask):
    """[start, stop) ranges where mask is True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def native_manifest(drive_log, required=False):
    """The final session manifest commits a native CSV/IMU/excitation trio."""
    path = companion_path(drive_log, "session", ".json")
    if not path.exists():
        if required or path.with_suffix(".json.partial").exists():
            raise ValueError(f"{Path(drive_log).name}: incomplete native recording; session manifest missing")
        return None
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "kaivuri-native-mlp-v1":
        if required:
            raise ValueError(f"{path.name}: unsupported native recording format")
        return None
    if manifest.get("complete") is not True or manifest.get("dropped", 0):
        raise ValueError(f"{path.name}: incomplete native recording or dropped samples")
    expected = {
        Path(drive_log).name,
        raw_path_for(drive_log).name,
        companion_path(drive_log, "excitation", ".json").name,
    }
    if set(manifest.get("outputs", {})) != expected:
        raise ValueError(f"{path.name}: native output manifest does not describe the complete trio")
    attachments = manifest.get("attachments", {})
    if not isinstance(attachments, dict) or set(attachments) & expected:
        raise ValueError(f"{path.name}: invalid native recording attachments")
    if manifest.get("collector") == "kaivuri-pid-collect":
        required = {
            companion_path(drive_log, "pid_control", ".csv").name,
            companion_path(drive_log, "pid_protocol", ".json").name,
        }
        if set(attachments) != required:
            raise ValueError(f"{path.name}: incomplete PID recording attachments")
    for name, checksum in {**manifest["outputs"], **attachments}.items():
        if Path(name).name != name or digest(path.parent / name) != checksum:
            raise ValueError(f"{path.name}: output checksum mismatch for {name}")
    return manifest


def load_pair(drive_log, min_seconds=3.0):
    """Usable chunks of one recording, and a report of what was kept."""
    raw_log = raw_path_for(drive_log)
    d = read_columns(
        drive_log,
        ("timestamp", "state_imu_ts_us", "cmd_stale", *COMMANDS),
        (
            "sample_idx",
            "cmd_age_s",
            "pump_on",
            "write_ok",
            "imu_valid",
            "input_valid",
            "state_age_s",
            "post_write_ns",
            "dry_pwm",
            "sine_enabled",
            "control_source_code",
            "controller_valid",
        ),
    )
    r = read_columns(
        raw_log,
        ("device_ts_us", *(f"imu_{role}_q{a}" for role in ROLES for a in "wxyz"))
        + tuple(f"imu_{role}_gy_dps" for role in ROLES),
        ("host_read_ns",),
    )
    native = native_manifest(drive_log, required=d["post_write_ns"] is not None and d["dry_pwm"] is not None)
    contract = dict(SENSOR_CONTRACT)
    if native is not None:
        sensor = native.get("sensor_contract", {})
        if sensor.get("roles") != list(ROLES) or sensor.get("axis_aligned_y") is not True:
            raise ValueError("Unsupported native sensor mapping or mounting axes")
        mount = np.asarray(sensor.get("mount_pitch_rad"), dtype=np.float64)
        zero = sensor.get("carriage_pitch_zero_rad")
        if mount.shape != (4,) or not np.isfinite(mount).all() or zero is None or not np.isfinite(zero):
            raise ValueError("Native sensor calibration is missing or nonfinite")
        if (
            not np.allclose(mount, contract["mount_pitch_rad"], rtol=0, atol=1e-8)
            or abs(zero - contract["carriage_pitch_zero_rad"]) > 1e-8
        ):
            raise ValueError("Native sensor contract differs from the fixed training calibration")
    stamp = d["timestamp"]
    if not np.isfinite(stamp).all() or np.any(np.diff(stamp) <= 0):
        raise ValueError(f"{Path(drive_log).name}: host timestamps must increase strictly")
    raw_t = unwrap_clock(r["device_ts_us"])
    if np.any(np.diff(raw_t) <= 0):
        raise ValueError(f"{raw_log.name}: raw IMU clock is not strictly increasing")
    quats = np.stack([np.stack([r[f"imu_{role}_q{a}"] for a in "wxyz"], -1) for role in ROLES], 1)
    gyro = np.stack([r[f"imu_{role}_gy_dps"] for role in ROLES], 1)
    commands = np.stack([d[c] for c in COMMANDS], 1)

    # Rows a tick may hold: live, finite command and a reported pose frame.
    live = (d["cmd_stale"] == 0) & np.isfinite(commands).all(1) & (d["state_imu_ts_us"] >= 0)
    if d["cmd_age_s"] is not None:
        live &= (d["cmd_age_s"] >= 0) & (d["cmd_age_s"] <= MAX_AGE)
    for key in ("pump_on", "write_ok", "imu_valid", "input_valid", "controller_valid"):
        if d[key] is not None:
            live &= d[key] == 1
    if d["dry_pwm"] is not None:
        live &= d["dry_pwm"] == 0
    if native is not None and native.get("dry_pwm") is True:
        live[:] = False
    if d["state_age_s"] is not None:
        live &= (d["state_age_s"] >= 0) & (d["state_age_s"] <= MAX_AGE)
    # A pose frame is usable only if the raw strip holds it.
    host_clock = unwrap_clock(np.where(d["state_imu_ts_us"] >= 0, d["state_imu_ts_us"], np.nan))
    first = np.flatnonzero(np.isfinite(host_clock))
    if not len(first):
        return [], {"rows": len(stamp), "usable_s": 0.0}
    host_clock += round((raw_t[0] - host_clock[first[0]]) / 2**32) * 2**32
    frame = np.clip(np.searchsorted(raw_t, host_clock, side="right") - 1, 0, len(raw_t) - 1)
    complete = np.isfinite(quats).all((1, 2)) & np.isfinite(gyro).all(1)
    live &= np.isfinite(host_clock) & (raw_t[frame] == host_clock) & complete[frame]
    if r["host_read_ns"] is not None and d["post_write_ns"] is not None:
        age = (d["post_write_ns"] - r["host_read_ns"][frame]) / 1e9
        live &= (age >= 0) & (age <= MAX_AGE)
    # Consecutive rows only: a dropped or reordered sample splits the timeline.
    joined = np.diff(stamp) <= MAX_AGE
    if d["sample_idx"] is not None:
        joined &= np.diff(d["sample_idx"]) == 1
    chunks = []
    for a, b in runs(live):
        breaks = a + 1 + np.flatnonzero(~joined[a : b - 1])
        for lo, hi in zip((a, *breaks), (*breaks, b)):
            ticks = np.arange(stamp[lo], stamp[hi - 1] + 1e-9, DT)
            rows = lo + np.searchsorted(stamp[lo:hi], ticks, side="right") - 1
            if len(ticks) < 2 * EDGE_TRIM + 1:
                continue
            ids = frame[rows]
            q = imu_positions(quats[ids])
            v = aligned_rates(gyro[ids])
            keep = slice(EDGE_TRIM, -EDGE_TRIM)
            chunk = {
                "t": ticks[keep],
                "q": q[keep],
                "v": v[keep],
                "u": commands[rows][keep].astype(np.float32),
                "target_q": q[keep],
                "target_v": v[keep],
            }
            if len(chunk["t"]) >= min_seconds / DT:
                chunks.append(chunk)
    report = {
        "rows": len(stamp),
        "live_rows": int(live.sum()),
        "chunks": len(chunks),
        "usable_s": round(sum(len(c["t"]) for c in chunks) * DT, 2),
        "sensor_contract": contract,
        "source": "native_mlp" if native is not None else "legacy_drive_log",
    }
    if d["sine_enabled"] is not None:
        report.update(
            manual_rows=int((live & (d["sine_enabled"] == 0)).sum()),
            excited_rows=int((live & (d["sine_enabled"] == 1)).sum()),
        )
    if d["control_source_code"] is not None:
        report.update(
            manual_rows=int((live & (d["control_source_code"] == 0)).sum()),
            pid_rows=int((live & (d["control_source_code"] == 1)).sum()),
            idle_rows=int((live & (d["control_source_code"] == 2)).sum()),
        )
    if native is not None:
        sidecar = companion_path(drive_log, "excitation", ".json")
        excitation = json.loads(sidecar.read_text())
        report.update(
            collector=native.get("collector", "kaivuri-mlp-collect"),
            attachments=native.get("attachments", {}),
            source_session_id=native.get("collection_id") or Path(drive_log).stem,
            source_revision=native.get("source_revision"),
            executable_sha256=native.get("executable_sha256"),
            operator_suffix=native.get("suffix", ""),
            excitation={"file": sidecar.name, "sha256": digest(sidecar), "blocks": excitation["blocks"]},
        )
    return chunks, report


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    dest = args.out / "data" / args.split
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f"drive_logs_{args.split}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"recordings": []}
    if manifest.setdefault("target_v", "gyro") != "gyro":
        raise ValueError("Existing split uses different velocity targets; start a fresh output directory")
    known = {entry["drive_log"] for entry in manifest["recordings"]}
    other_splits = []
    for path in args.out.glob("drive_logs_*.json"):
        if path != manifest_path:
            other_splits.extend(json.loads(path.read_text()).get("recordings", []))
    for drive_log in args.logs:
        if drive_log.name in known:
            raise FileExistsError(f"{drive_log.name} is already in {manifest_path.name}")
        chunks, report = load_pair(drive_log, args.min_seconds)
        checksum = digest(drive_log)
        for entry in other_splits:
            same_session = (
                report.get("source_session_id")
                and entry.get("source_session_id") == report["source_session_id"]
            )
            if (
                same_session
                or entry.get("drive_log_sha256") == checksum
                or entry["drive_log"] == drive_log.name
            ):
                raise ValueError("Recording session is already assigned to another train/dev/test split")
        contract = report.pop("sensor_contract", dict(SENSOR_CONTRACT))
        if "sensors" in manifest and manifest["sensors"] != contract:
            raise ValueError("Recording calibration differs from this split; use a separate experiment")
        manifest["sensors"] = contract
        paths = []
        for index, chunk in enumerate(chunks):
            path = dest / f"{drive_log.stem}_{index:03d}.npz"
            np.savez_compressed(path, **chunk)
            paths.append(str(path.relative_to(args.out)))
        manifest["recordings"].append(
            {
                "drive_log": drive_log.name,
                "drive_log_sha256": checksum,
                "imu_raw_sha256": digest(raw_path_for(drive_log)),
                **report,
                "paths": paths,
            }
        )
        known.add(drive_log.name)
        summary = {
            key: report[key] for key in ("rows", "live_rows", "chunks", "usable_s", "source") if key in report
        }
        print(f"{drive_log.name}: {summary}", flush=True)
    manifest.update(dt=DT)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("prepare",))
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="drive_log_*.csv files")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="Staging directory for imported chunks")
    parser.add_argument("--min_seconds", type=float, default=3.0, help="Shortest chunk kept")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
