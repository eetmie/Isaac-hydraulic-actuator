"""Configuration, pose replay and a complete small output run."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("warp")

from soil2d import Bucket, Material, Scene
from soil2d.cli import main
from soil2d.solver import Soil
from soil2d.trajectory import POSE_COLUMNS, Trajectory, digging_path


@pytest.mark.parametrize(
    "scene",
    [
        replace(Scene(), dt=0),
        replace(Scene(), dt=0.003),
        replace(Scene(), effective_width=-1),
        replace(Scene(), frame_dt=0.0201),
        replace(Scene(), dx=0.011),
        replace(Scene(), material=Material(density=0)),
        replace(Scene(), material=Material(poisson_ratio=0.5)),
        replace(Scene(), material=Material(cohesion_pa=-1)),
        replace(Scene(), bucket=Bucket(vertices=((0, 0), (0, 0)))),
        replace(Scene(), gravity=float("nan")),
        replace(Scene(), soil_min=(0.4, 0.0)),
        replace(Scene(), soil_min=(0.4,)),
    ],
)
def test_invalid_physical_configuration_is_rejected(scene):
    with pytest.raises(ValueError):
        scene.validate()


def test_invalid_commands_are_rejected_without_advancing():
    solver = Soil(device="cpu")
    for pose, twist in [((0, 0, float("nan")), (0, 0, 0)), ((0, 0, 0), (100, 0, 0)), ((0, 0), (0, 0, 0))]:
        with pytest.raises(ValueError):
            solver.step(pose, twist)
        assert solver.time == 0


def test_trajectory_command_reaches_next_pose_without_time_or_angle_guessing(tmp_path):
    path = digging_path()
    for t in np.linspace(0, path.duration, 41):
        pose, velocity = path.command(float(t), 0.02)
        np.testing.assert_allclose(pose + 0.02 * velocity, path.sample(float(t) + 0.02), atol=1e-12)
    csv_path = tmp_path / "poses.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(POSE_COLUMNS)
        writer.writerows([[0, 0.2, 0.4, 0], [1, 0.4, 0.3, 0.5], [2, 0.5, 0.4, 1]])
    loaded = Trajectory.from_csv(csv_path)
    np.testing.assert_allclose(loaded.sample(0.5), [0.3, 0.35, 0.25])
    csv_path.write_text("time_s,bucket_x_m,bucket_z_m,bucket_angle_rad\n0,0,0,0\n0,1,1,1\n")
    with pytest.raises(ValueError, match="increase strictly"):
        Trajectory.from_csv(csv_path)


def test_demo_writes_finite_results_and_replayable_pose_csv(tmp_path):
    output = tmp_path / "run"
    assert main(["demo", "--device", "cpu", "--seconds", "0.04", "--out", str(output), "--gif"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "ok" and summary["validation"]["valid"]
    assert summary["duration_s"] == 0.04
    assert summary["physics_frame_wall_ms"]["mean"] > 0
    assert summary["source_sha256"]["kernels.py"]
    assert summary["trajectory"]["smooth_quintic"]
    assert summary["validated_preview_loop_wall_s"] > 0
    for name in ("preview.gif", "final.png", "forces.png", "tracking.csv", "final_state.npz"):
        assert (output / name).stat().st_size > 0
    trajectory = Trajectory.from_csv(output / "bucket_poses.csv")
    assert trajectory.duration == 0.04
    with pytest.raises(SystemExit):
        main(["demo", "--device", "cpu", "--seconds", "0.04", "--out", str(output)])
