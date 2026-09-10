# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Supply artifact parameters for the standalone contract tests under pytest."""

from pathlib import Path

import pytest


@pytest.fixture(params=("arm_v4", "slew"))
def model_dir(request) -> str:
    """Check both models distributed with the demo."""
    path = Path(__file__).resolve().parents[1] / "models" / request.param
    return str(path)
