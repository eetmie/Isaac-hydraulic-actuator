# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The current raw-gyro trainer must obey the deployed model's history contract."""

import numpy as np
import torch

from actuators import HydraulicActuatorNet
from hydraulic_controller.core import DEFAULT_MODEL
from training.actuator import Predictor


@torch.no_grad()
def test_training_predictor_matches_the_deployed_history_reader():
    runtime = HydraulicActuatorNet(DEFAULT_MODEL, device="cpu", sim_dt=0.01)
    training = Predictor(DEFAULT_MODEL, "cpu")
    rng = np.random.default_rng(17)
    q = rng.normal(0, 0.1, (110, 4)).astype(np.float32)
    v = rng.normal(0, 0.2, (110, 4)).astype(np.float32)
    u = rng.uniform(-0.8, 0.8, (110, 3)).astype(np.float32)
    for i in range(len(q)):
        actual = runtime.predict_velocity(q[i], v[i], u[i])
        if i < 61:
            continue
        predicted = training.next_velocity(
            torch.from_numpy(q[i : i + 1]),
            torch.from_numpy(v[i - np.arange(41)][None]),
            torch.from_numpy(u[i - np.arange(61)][None]),
        ).numpy()[0]
        np.testing.assert_allclose(predicted, actual, atol=2e-6, rtol=2e-6)
