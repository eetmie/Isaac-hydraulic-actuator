# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check the distributed models and demo defaults without local training data."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_common import ARM_JOINT_NAMES, DEFAULT_ARM_MODEL_DIR, DEFAULT_SLEW_MODEL_DIR, resolve_demo_defaults


def test_neural_model_import_does_not_require_isaac():
    code = """
import importlib.abc
import sys
class NoIsaac(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"isaaclab", "isaacsim", "omni", "carb", "pxr"}:
            raise ImportError("Neural inference must not import " + fullname)
sys.meta_path.insert(0, NoIsaac())
from actuators import HydraulicActuatorNet
assert HydraulicActuatorNet.__name__ == "HydraulicActuatorNet"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_default_release_selects_v4_direct_pitch():
    assert Path(DEFAULT_ARM_MODEL_DIR) == ROOT / "models/arm_v4"
    assert Path(DEFAULT_SLEW_MODEL_DIR) == ROOT / "models/slew"
    args = SimpleNamespace(
        model=DEFAULT_ARM_MODEL_DIR,
        integration=None,
        physics_substeps=None,
        carriage_rocking=False,
        arm_stiffness=None,
        arm_damping=None,
    )
    resolve_demo_defaults(args)
    assert args.model_joint_names == ARM_JOINT_NAMES + ["revolute_carriage_pitch"]
    assert args.learned_carriage and args.carriage_rocking
    assert args.integration == "direct" and args.physics_substeps == 1


@pytest.mark.parametrize("name", ["arm_v4", "slew"])
def test_release_hashes(name):
    folder = ROOT / "models" / name
    manifest = json.loads((folder / "release_manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected


def test_slew_runtime_is_heading_invariant_and_stationary_at_rest():
    from actuators import HydraulicActuatorNet

    nets = [HydraulicActuatorNet(ROOT / "models/slew", sim_dt=0.01) for _ in range(3)]
    headings = np.array([0.0, 8 * np.pi, -7.3], dtype=np.float32)
    velocities = np.zeros(3, dtype=np.float32)
    for step in range(300):
        command = np.array([0.6 if 50 <= step < 200 else 0.0], dtype=np.float32)
        for j, net in enumerate(nets):
            velocities[j] = net.predict_velocity(headings[j : j + 1], velocities[j : j + 1], command)[0]
        np.testing.assert_array_equal(velocities, np.repeat(velocities[0], 3))
        if step < 50:
            np.testing.assert_allclose(velocities, 0.0, atol=1e-7)
        headings += 0.01 * velocities


@pytest.mark.parametrize("tool", ["bucket", "gripper"])
def test_packaged_assets_are_self_contained_with_continuous_slew(tool):
    from pxr import Usd, UsdPhysics, UsdUtils

    for suffix in ("", "_rocking"):
        path = ROOT / f"assets/excavator_{tool}{suffix}.usd"
        stage = Usd.Stage.Open(str(path))
        assert str(stage.GetDefaultPrim().GetPath()) == "/excavator"
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        assert not unresolved
        joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/excavator/Joints/revolute_carriage"))
        assert joint.GetLowerLimitAttr().Get() == -np.inf
        assert joint.GetUpperLimitAttr().Get() == np.inf
        for name in ["revolute_carriage", *ARM_JOINT_NAMES]:
            drive = UsdPhysics.DriveAPI(stage.GetPrimAtPath("/excavator/Joints/" + name), "angular")
            gains = np.array([drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get()]) * 180 / np.pi
            expected = [600.0, 40.0] if name == "revolute_carriage" else [2400.0, 120.0]
            np.testing.assert_allclose(gains, expected, rtol=1e-6)
        if suffix:
            for name in ("revolute_carriage_roll", "revolute_carriage_pitch"):
                assert stage.GetPrimAtPath("/excavator/Joints/" + name)
