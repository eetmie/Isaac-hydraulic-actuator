# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check actual demo configuration expressions without starting Kit."""

import ast
from pathlib import Path
from types import SimpleNamespace


def test_physics_substeps_do_not_change_control_period():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "sim.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "sim_cfg" for t in node.targets)
    )
    scope = {
        "sim_utils": SimpleNamespace(SimulationCfg=lambda **kw: SimpleNamespace(**kw)),
        "SIM_HZ": 100,
        "args_cli": SimpleNamespace(physics_substeps=5, device="cpu"),
        "PhysxCfg": lambda **kw: kw,
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "sim_config", "exec"), scope)
    assert scope["sim_cfg"].dt == 0.002
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "sim_dt" for t in node.targets)
    )
    scope["physics_dt"] = 0.002
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "control_period", "exec"), scope)
    assert scope["sim_dt"] == 0.01
