"""Speed comparisons complete paths and do not let endpoint holds dominate motion scores."""

import csv
import json
import math

import pytest
import torch

from hydraulic_controller.speed_audit import line_duration, motion_metrics, write_speed_report
from hydraulic_controller.tasks import trapezoid


def test_every_reference_finishes_before_the_tail_window():
    speeds = [v / 1000 for v in (5, 10, 20, 40, 70, 100, 120, 1000)]
    duration = line_duration(speeds, 0.1, 0.5, 1.0, 1.0)
    assert duration == 23
    for speed in speeds:
        end = trapezoid(0.1, speed, 0.5, torch.tensor(duration - 2.0))
        assert float(end) == pytest.approx(0.1)


def test_motion_scores_exclude_large_hold_errors():
    target = torch.zeros(6, 1, 3)
    target[:, 0, 0] = torch.tensor([0, 0, 0.01, 0.02, 0.02, 0.02])
    actual = target.clone()
    actual[:, :, 0] += 0.5
    actual[2:4, :, 0] = target[2:4, :, 0] + 0.005
    actual[2:4, :, 2] = 0.1
    result = motion_metrics(
        dict(tip=actual, tip_target=target, u=torch.zeros(6, 1, 3)), 1.0, torch.tensor([0.01])
    )
    assert float(result["moving_seconds"][0]) == 2
    assert float(result["moving_tip_mean_mm"][0]) == pytest.approx(5)
    assert float(result["cruise_tip_mean_mm"][0]) == pytest.approx(5)
    assert float(result["moving_pitch_mean_deg"][0]) == pytest.approx(math.degrees(0.1))


def test_empty_motion_window_is_not_reported_as_perfect_tracking():
    trace = dict(tip=torch.zeros(3, 1, 3), tip_target=torch.zeros(3, 1, 3), u=torch.zeros(3, 1, 3))
    result = motion_metrics(trace, 0.01, torch.tensor([0.01]))
    assert result["moving_seconds"].item() == 0
    assert torch.isnan(result["moving_tip_mean_mm"]).all()


def test_exact_cruise_has_unit_along_speed_and_zero_relative_error():
    target = torch.zeros(5, 1, 3)
    target[:, 0, 0] = torch.tensor([0, 0, 0.01, 0.02, 0.02])
    result = motion_metrics(
        dict(tip=target.clone(), tip_target=target, u=torch.zeros(5, 1, 3)), 1.0, torch.tensor([0.01])
    )
    assert result["cruise_along_speed_ratio"].item() == pytest.approx(1.0)
    assert result["cruise_relative_velocity_error"].item() == pytest.approx(0.0)


def test_speed_report_separates_feasibility_and_averages_seed_means(tmp_path):
    for seed, good, bad in ((42, 1, 100), (43, 3, 200)):
        directory = tmp_path / f"2026-11-13_20-00-00_standard_new_lines_seed{seed}"
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(dict(task_config=dict(families=["tip_line"], feasible_speed_ratio=0.9)))
        )
        rows = [
            dict(
                controller="tip_control",
                speed_mm_s=speed,
                path_valid=valid,
                speed_ratio=ratio,
                tip_mean_mm=error,
                pitch_mean_deg=2,
                valve_travel=0.5,
                invalid=0,
            )
            for speed, valid, ratio, error in ((20, 1, 0.5, good), (20, 1, 1.2, bad), (40, 0, 0.5, 500))
        ]
        with (directory / "per_scenario.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    rows = write_speed_report(tmp_path, tmp_path / "out")
    feasible = next(r for r in rows if r["subset"] == "capacity_feasible")
    assert feasible["tip_mean_mm"] == 2
    assert feasible["tip_mean_mm_std"] == pytest.approx(math.sqrt(2))
    assert feasible["seeds"] == 2 and feasible["cases_per_seed"] == [1, 1]
    assert not any(r["subset"] == "capacity_feasible" and r["speed_mm_s"] == 40 for r in rows)
    all_twenty = next(r for r in rows if r["subset"] == "all" and r["speed_mm_s"] == 20)
    assert all_twenty["tip_mean_mm"] == 76
