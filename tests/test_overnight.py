"""Monitoring failures must not interrupt training, and recovery keeps finished phase budgets spent."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from training.overnight import Run, make_schedule, write_json


def test_json_write_retries_windows_sharing_violation(tmp_path):
    destination = tmp_path / "status.json"
    original = Path.replace
    attempts = []

    def replace(source, target):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError("Windows file sharing violation")
        return original(source, target)

    with patch.object(Path, "replace", replace), patch("training.overnight.time.sleep"):
        write_json(destination, {"status": "running"})
    assert len(attempts) == 3
    assert json.loads(destination.read_text()) == {"status": "running"}
    assert list(tmp_path.iterdir()) == [destination]


def test_persistent_monitoring_failure_does_not_abort_child_or_completion(tmp_path):
    run = Run(tmp_path, {})
    (tmp_path / "completed").mkdir()

    def status_locked(path, value):
        if path.name == "status.json":
            raise PermissionError("locked")
        return write_json(path, value)

    with patch("training.overnight.write_json", side_effect=status_locked):
        run.step("child", ["-c", "print('child completed')"])
    assert "child completed" in (tmp_path / "child.log").read_text()
    assert (tmp_path / "completed/child.json").exists()


def test_persistent_status_failure_is_best_effort(tmp_path):
    run = Run(tmp_path, {})
    with patch("training.overnight.write_json", side_effect=PermissionError("locked")):
        run.status(status="running", stage="phase_b")
        run.status(status="complete")
    assert run.state["status"] == "complete"


def test_completed_phase_a_is_excluded_from_new_recovery_schedule(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "selection.json").write_text("{}")
    plan = dict(
        hours=12,
        extra_seed_hours=2,
        actuator_fraction=0.65,
        controller_fraction=0.30,
        evaluation_fraction=0.05,
        phases=[dict(name="a", weight=0.45), dict(name="b", weight=0.30), dict(name="c", weight=0.25)],
    )
    schedule = make_schedule(plan, tmp_path, 100)
    assert schedule["actuator_end"] - 100 == pytest.approx(12 * 3600 * 0.65 * 0.55)
    assert schedule["initial_controller_end"] - schedule["actuator_end"] == pytest.approx(12 * 3600 * 0.30)
    assert schedule["controller_end"] - schedule["initial_controller_end"] == 2 * 3600
