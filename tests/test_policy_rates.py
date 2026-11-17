# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rate comparisons preserve physical-time settings and measure smoothness without hold-sample bias."""

from pathlib import Path

import pytest
import torch
import yaml

from hydraulic_controller.smoothness import metrics
from training.policy_rate_comparison import RATES, matched_iterations, rate_settings


def test_rate_recipes_preserve_physical_time_and_integrated_valve_rate_cost():
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "models/controller_pivot/params/env.yaml").read_text()
    )
    for hz in RATES:
        config = rate_settings(source, hz)
        assert config["lookahead_steps"] / hz == pytest.approx(0.2)
        assert config["curriculum_steps"] / hz == pytest.approx(360)
        assert config["motion_pool_refresh_steps"] / hz == pytest.approx(6)
        assert config["action_rate_l2"] / hz == pytest.approx(0.05)
        assert config["action_rate_l1"] == source["action_rate_l1"]
        assert config["gyro_observations"] and config["motion_start_prob"] == 0.3
    rates = {10: 1.0, 20: 1.3, 50: 1.7, 100: 2.5}
    count = matched_iterations(12 * 3600, rates, [42, 43, 44])
    assert count >= 500 and count % 100 == 0
    assert count * sum(rates.values()) * 3 <= 12 * 3600 * 0.9
    with pytest.raises(ValueError, match="Not enough time"):
        matched_iterations(60, rates, [42, 43, 44])


def test_smoothness_measures_physical_acceleration_and_actual_valve_steps():
    t = torch.arange(201, dtype=torch.float64) * 0.01
    pose = torch.zeros(201, 1, 3, dtype=torch.float64)
    pose[:, 0, 0] = t.square()  # independently known acceleration: 2 m/s², jerk: 0
    pose[:, 0, 2] = 0.05 * t.square()  # angular acceleration: 0.1 rad/s²
    for hz in (10, 100):
        interval = 100 // hz
        valves = torch.zeros_like(pose)
        valves[:, 0, 0] = (torch.arange(201) // interval) * interval * 0.001
        result = metrics(dict(tip=pose, tip_target=pose, u=valves), hz)
        assert result["motion_accel_rms_m_s2"].item() == pytest.approx(2.0)
        assert result["motion_jerk_rms_m_s3"].item() < 1e-7
        assert result["motion_pitch_accel_rms_deg_s2"].item() == pytest.approx(5.72957795)
        assert result["valve_step_p99"].item() == pytest.approx(0.001 * interval, abs=1e-7)


def test_stationary_reference_reports_no_motion_score_and_retains_holding_hunt():
    pose = torch.zeros(201, 1, 3)
    pose[-100:, 0, 0] = torch.arange(100) % 2 * 0.002
    result = metrics(dict(tip=pose, tip_target=pose * 0, u=pose * 0), 20)
    assert torch.isnan(result["motion_accel_rms_m_s2"]).all()
    assert result["hold_tip_p2p_mm"].item() == pytest.approx(2.0)
