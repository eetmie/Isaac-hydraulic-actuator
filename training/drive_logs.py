# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Turn the robot's drive_log/imu_raw pairs into gyro_transfer training chunks.

kaivuriprokkis writes the same pair from operator recordings (simple_drive.py)
and controller runs (learned_control/run_circle.py). Run as
``python -m training.drive_logs prepare --split train --out runs/<name> --logs drive_log_*.csv``.
Each continuous usable stretch becomes ``<out>/data/<split>/<log stem>_NNN.npz``
with the arrays training.gyro_transfer trains and evaluates on:

    t          uniform 100 Hz ticks on the drive log's clock [s]
    q          joint angles and carriage pitch from the last IMU frame at each tick [rad]
    v          same-frame aligned gyro rates [rad/s] (the model's velocity input)
    u          valve command held at each tick, lift/tilt/scoop [-1, 1]
    target_q   q, the rollout position target
    target_v   the rollout velocity target, by --target_v:
                 gyro      v itself (default)
                 smooth21  21-sample Savitzky-Golay derivative of q, the label the
                           V2-V5 models were trained on. It passes 5 Hz at 0.84,
                           7 Hz at 0.54 and 10 Hz at 0.03 of the true rate, so it
                           hides the fast valve response; kept for comparisons.

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
from scipy.signal import savgol_filter

from hydraulic_controller.sensors import ROLES, SENSOR_CONTRACT, aligned_rates, imu_positions

DT = 0.01
MAX_AGE = 0.03  # same freshness bound as causal_indices
SMOOTH = 21  # smooth21 window; both targets trim it from chunk ends, where the filter is one-sided
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


def raw_path_for(drive_log):
    """The imu_raw strip written with a drive log: same name, other prefix."""
    drive_log = Path(drive_log)
    if not drive_log.name.startswith("drive_log_"):
        raise ValueError(f"Expected a drive_log_*.csv file, got {drive_log.name}")
    raw = drive_log.with_name("imu_raw_" + drive_log.name.removeprefix("drive_log_"))
    if not raw.exists():
        raise FileNotFoundError(f"No imu_raw strip next to {drive_log.name}")
    return raw


def runs(mask):
    """[start, stop) ranges where mask is True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def load_pair(drive_log, min_seconds=3.0, target_v="gyro"):
    """Usable chunks of one recording, and a report of what was kept."""
    raw_log = raw_path_for(drive_log)
    d = read_columns(
        drive_log,
        ("timestamp", "state_imu_ts_us", "cmd_stale", *COMMANDS),
        ("sample_idx", "cmd_age_s"),
    )
    r = read_columns(
        raw_log,
        ("device_ts_us", *(f"imu_{role}_q{a}" for role in ROLES for a in "wxyz"))
        + tuple(f"imu_{role}_gy_dps" for role in ROLES),
    )
    stamp = d["timestamp"]
    raw_t = unwrap_clock(r["device_ts_us"])
    if np.any(np.diff(raw_t) <= 0):
        raise ValueError(f"{raw_log.name}: raw IMU clock is not strictly increasing")
    quats = np.stack([np.stack([r[f"imu_{role}_q{a}"] for a in "wxyz"], -1) for role in ROLES], 1)
    gyro = np.stack([r[f"imu_{role}_gy_dps"] for role in ROLES], 1)
    commands = np.stack([d[c] for c in COMMANDS], 1)

    # Rows a tick may hold: live, finite command and a reported pose frame.
    live = (d["cmd_stale"] == 0) & np.isfinite(commands).all(1) & (d["state_imu_ts_us"] >= 0)
    if d["cmd_age_s"] is not None:
        live &= d["cmd_age_s"] <= MAX_AGE
    # A pose frame is usable only if the raw strip holds it.
    host_clock = unwrap_clock(np.where(d["state_imu_ts_us"] >= 0, d["state_imu_ts_us"], np.nan))
    first = np.flatnonzero(np.isfinite(host_clock))
    if not len(first):
        return [], {"rows": len(stamp), "usable_s": 0.0}
    host_clock += round((raw_t[0] - host_clock[first[0]]) / 2**32) * 2**32
    frame = np.clip(np.searchsorted(raw_t, host_clock, side="right") - 1, 0, len(raw_t) - 1)
    complete = np.isfinite(quats).all((1, 2)) & np.isfinite(gyro).all(1)
    live &= np.isfinite(host_clock) & (raw_t[frame] == host_clock) & complete[frame]
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
            if len(ticks) < SMOOTH + 1:
                continue
            ids = frame[rows]
            q, v = imu_positions(quats[ids]), aligned_rates(gyro[ids])
            if target_v == "smooth21":
                target = savgol_filter(q, SMOOTH, 3, deriv=1, delta=DT, axis=0).astype(np.float32)
            else:
                target = v
            keep = slice(SMOOTH // 2, -(SMOOTH // 2))
            chunk = {
                "t": ticks[keep],
                "q": q[keep],
                "v": v[keep],
                "u": commands[rows][keep].astype(np.float32),
                "target_q": q[keep],
                "target_v": target[keep],
            }
            if len(chunk["t"]) >= min_seconds / DT:
                chunks.append(chunk)
    report = {
        "rows": len(stamp),
        "live_rows": int(live.sum()),
        "chunks": len(chunks),
        "usable_s": round(sum(len(c["t"]) for c in chunks) * DT, 2),
    }
    return chunks, report


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    dest = args.out / "data" / args.split
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / f"drive_logs_{args.split}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"recordings": []}
    if manifest.setdefault("target_v", args.target_v) != args.target_v:
        raise ValueError(
            f"{manifest_path.name} holds target_v={manifest['target_v']}; use one label per split"
        )
    known = {entry["drive_log"] for entry in manifest["recordings"]}
    for drive_log in args.logs:
        if drive_log.name in known:
            raise FileExistsError(f"{drive_log.name} is already in {manifest_path.name}")
        chunks, report = load_pair(drive_log, args.min_seconds, args.target_v)
        paths = []
        for index, chunk in enumerate(chunks):
            path = dest / f"{drive_log.stem}_{index:03d}.npz"
            np.savez_compressed(path, **chunk)
            paths.append(str(path.relative_to(args.out)))
        manifest["recordings"].append(
            {
                "drive_log": drive_log.name,
                "drive_log_sha256": digest(drive_log),
                "imu_raw_sha256": digest(raw_path_for(drive_log)),
                **report,
                "paths": paths,
            }
        )
        print(f"{drive_log.name}: {report}", flush=True)
    manifest.update(sensors=SENSOR_CONTRACT, dt=DT)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("prepare",))
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="drive_log_*.csv files")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="gyro_transfer experiment directory")
    parser.add_argument("--min_seconds", type=float, default=3.0, help="Shortest chunk kept")
    parser.add_argument("--target_v", choices=("gyro", "smooth21"), default="gyro", help="Velocity target")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
