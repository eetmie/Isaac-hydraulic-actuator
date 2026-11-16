"""Long comparisons retain matched budgets, intentional factors and fair checkpoint indices."""

import json
from collections import Counter

from training.weekend_comparison import SEEDS, budgets, jobs, rotation_summary, selected_checkpoints


def test_allocated_updates_fit_the_training_window():
    rates = dict(tip_control=1.3, tip_rot4x=1.4, tip_rot4x_linear=1.5, pivot_control=1.4)
    available = 50 * 3600
    counts = budgets(available, rates, SEEDS)
    fixed = counts["main"] * sum(rates.values()) * 3
    deploy = counts["deploy"] * rates["tip_rot4x_linear"] * 3
    conditioned = counts["priority"] * sum(rates.values()) / 4 * 6
    assert fixed + deploy + conditioned <= available * 0.92
    assert all(value >= 500 and value % 500 == 0 for value in counts.values())


def test_conditions_have_three_seeds_and_supported_fixed_tip_export_recipe():
    planned = jobs(dict(seeds=list(SEEDS), smoke=False), {}, dict(main=6000, deploy=4000, priority=5000))
    assert Counter(j["group"] for j in planned) == dict(main=12, deploy=3, priority=6)
    for job in planned:
        cfg = job["config"]
        assert cfg["gyro_observations"] and cfg["motion_start_prob"] == 0.3
        if job["group"] == "deploy":
            assert cfg["tracked_point"] == "tip" and cfg["lookahead_steps"] == 0 and not cfg["pose_priority"]
        if job["group"] == "priority":
            assert cfg["pose_priority"] and cfg["pose_angular_error_weight_max"] == 0.5


def test_time_limited_group_is_evaluated_at_a_shared_saved_iteration(tmp_path):
    planned = [dict(name="a", group="main"), dict(name="b", group="main")]
    for name, end in (("a", 299), ("b", 250)):
        directory = tmp_path / "controllers" / name
        directory.mkdir(parents=True)
        (directory / "training_result.json").write_text(json.dumps(dict(iteration=end)))
        (directory / "model_200.pt").write_bytes(b"saved checkpoint")
    chosen = selected_checkpoints(tmp_path, planned)
    assert all(value.endswith("model_200.pt") for value in chosen.values())


def test_rotation_report_preserves_worst_case_before_combining_seeds(tmp_path):
    for seed, error in ((42, 10), (43, 20)):
        directory = tmp_path / "rotation" / f"seed{seed}" / "tip_control"
        directory.mkdir(parents=True)
        rows = [
            dict(
                sensor_mode="gyro",
                amplitude_deg=10,
                tip_max_mm=value,
                angle_max_deg=value / 2,
                failed=value > 10,
            )
            for value in (1, error)
        ]
        (directory / "rotation_results.json").write_text(json.dumps(dict(rows=rows)))
    result = rotation_summary(tmp_path)["gyro_10deg"]["tip_control"]
    assert result["tip_max_mm"]["mean"] == 15
    assert result["tip_max_mm"]["worst"] == 20
    assert result["angle_max_deg"]["mean"] == 7.5
    assert result["failures"]["mean"] == 0.5
