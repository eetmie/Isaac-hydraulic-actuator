# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The compiled closed-valve bound equals clipping the source model's next velocity."""

import shutil

import pytest
import torch

from hydraulic_controller.core import DEFAULT_MODEL

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="actuator model missing")


def test_compiled_bound_clips_the_source_and_stops_with_clear_valves(tmp_path):
    from training.actuator import Predictor
    from training.neutral_bound import compile_model

    # The packaged model is already bounded; an output offset makes a copy that drifts with clear valves.
    drifting = tmp_path / "drifting"
    shutil.copytree(DEFAULT_MODEL, drifting)
    state = torch.load(drifting / "mlp_state_dict.pt", map_location="cpu", weights_only=True)
    last = max((k for k in state if k.endswith(".bias")), key=lambda k: int(k.split(".")[1]))
    state[last] = state[last] + 1.0
    torch.save(state, drifting / "mlp_state_dict.pt")
    compiled = Predictor(compile_model(drifting, tmp_path / "bounded", gain=5.0), "cpu")
    source = Predictor(drifting, "cpu")
    g = torch.Generator().manual_seed(0)
    q = torch.randn(256, 4, generator=g) * 0.3
    vh = torch.randn(256, 41, 4, generator=g) * 0.3
    uh = (torch.rand(256, 61, 3, generator=g) * 2 - 1) * 0.3
    uh[::4] = 0  # every valve clear
    uh[1::4, :, 2] = 0  # only the bucket clear
    with torch.no_grad():
        raw, bounded = source.next_velocity(q, vh, uh), compiled.next_velocity(q, vh, uh)
    cap = 5 * uh[:, ::3].abs().sum(1)
    expected = raw.clone()
    expected[:, :3] = torch.maximum(torch.minimum(raw[:, :3], cap), -cap)
    torch.testing.assert_close(bounded, expected, atol=2e-6, rtol=0)
    assert bounded[::4, :3].abs().max() < 1e-6 and bounded[1::4, 2].abs().max() < 1e-6
    assert (raw[:, :3] - expected[:, :3]).abs().max() > 1e-3  # the bound was active somewhere
