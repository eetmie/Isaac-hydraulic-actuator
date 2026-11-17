# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Motion smoothness at a fixed 100 Hz, independently of policy update frequency."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import wrap_angle

DT = 0.01
PROTOCOL = (
    "Physical tip acceleration, jerk and angular acceleration use 100 Hz finite differences. "
    "The motion window includes 100 ms either side of reference motion. Valve step statistics use "
    "actual policy updates, excluding zero-order holds. Holding ranges use the final tail window. "
    "These are learned-plant estimates; they do not establish robot vibration or comfort."
)


def metrics(trace, policy_hz, tail_s=1.0):
    """Per-scenario vectors from true tip poses and commanded valves, both sampled at 100 Hz."""
    pose, valves = trace["tip"].double(), trace["u"].double()
    if len(pose) < 4 or policy_hz <= 0 or 100 % policy_hz:
        raise ValueError("Need at least four 100 Hz samples and a policy rate dividing 100")
    reference = trace.get("tip_target")
    if reference is None:
        active = torch.ones(pose.shape[:2], dtype=torch.bool)
    else:
        diff = reference[1:] - reference[:-1]
        moving = (diff[:, :, :2].norm(dim=-1) > 1e-8) | (wrap_angle(diff[:, :, 2]).abs() > 1e-8)
        active = torch.cat((moving[:1], moving), dim=0)
        active = F.max_pool1d(active.T[:, None].float(), 21, stride=1, padding=10)[:, 0].T.bool()

    velocity = torch.diff(pose[:, :, :2], dim=0) / DT
    acceleration = torch.diff(velocity, dim=0) / DT
    jerk = torch.diff(acceleration, dim=0) / DT
    pitch_rate = wrap_angle(torch.diff(pose[:, :, 2], dim=0)) / DT
    pitch_accel = torch.diff(pitch_rate, dim=0) / DT

    def rms(values, mask):
        count = mask.sum(0)
        result = (torch.where(mask, values.square(), 0).sum(0) / count.clamp_min(1)).sqrt()
        return torch.where(count > 0, result, math.nan)

    acceleration_mask = active[2:] & active[1:-1] & active[:-2]
    jerk_mask = acceleration_mask[1:] & active[:-3]
    sampled = valves[:: int(100 // policy_hz)]
    # Count the first policy command as a step from neutral, as the rollout does.
    changes = torch.diff(sampled, dim=0, prepend=torch.zeros_like(sampled[:1])).abs()
    changes = changes.permute(1, 0, 2).flatten(1)
    tail = pose[-max(1, round(tail_s / DT)) :]
    relative_pitch = wrap_angle(tail[:, :, 2] - tail[:1, :, 2])
    return dict(
        motion_accel_rms_m_s2=rms(acceleration.norm(dim=-1), acceleration_mask),
        motion_jerk_rms_m_s3=rms(jerk.norm(dim=-1), jerk_mask),
        motion_pitch_accel_rms_deg_s2=torch.rad2deg(rms(pitch_accel, acceleration_mask)),
        valve_step_p99=torch.quantile(changes, 0.99, dim=1),
        valve_step_max=changes.amax(1),
        hold_tip_p2p_mm=(tail[:, :, :2].amax(0) - tail[:, :, :2].amin(0)).norm(dim=-1) * 1000,
        hold_pitch_p2p_deg=torch.rad2deg(relative_pitch.amax(0) - relative_pitch.amin(0)),
    )


def summarize(values, mask):
    result = {}
    for name, vector in values.items():
        selected = vector[mask & torch.isfinite(vector)]
        result[name] = float(selected.mean()) if len(selected) else math.nan
        result[name + "_worst"] = float(selected.max()) if len(selected) else math.nan
    return result


def write_report(out: Path, prep, traces, policy_rates):
    """Retain every case, plus all-case and capacity-feasible summaries at each controller's native rate."""
    summaries, rows = {}, []
    feasible = (prep.path_valid & prep.feasible).cpu()
    nominal = torch.tensor([scenario.plant == "nominal" for scenario in prep.scenarios])
    for name, trace in traces.items():
        hz = policy_rates.get(name, 100)
        values = metrics(trace, hz, prep.cfg.tail_s)
        summaries[name] = dict(
            policy_hz=hz,
            all=summarize(values, torch.ones(len(feasible), dtype=torch.bool)),
            capacity_feasible=summarize(values, feasible),
            nominal=summarize(values, nominal),
        )
        for index, scenario in enumerate(prep.scenarios):
            rows.append(
                dict(
                    controller=name,
                    scenario=scenario.name,
                    plant=scenario.plant,
                    policy_hz=hz,
                    speed_mm_s=scenario.size * 1000,
                    capacity_feasible=bool(feasible[index]),
                    **{key: float(value[index]) for key, value in values.items()},
                )
            )
    with (out / "smoothness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "smoothness.json").write_text(
        json.dumps(dict(protocol=PROTOCOL, summaries=summaries), indent=2) + "\n"
    )
    torch.save(dict(dt_s=DT, policy_rates=policy_rates, traces=traces), out / "traces.pt")
    return summaries
