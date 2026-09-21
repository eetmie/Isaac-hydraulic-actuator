"""Supply artifact parameters for the standalone contract tests under pytest."""

from pathlib import Path

import pytest


@pytest.fixture(params=("arm_v4", "arm_v5", "slew"))
def model_dir(request) -> str:
    """Check every model distributed with the demo."""
    path = Path(__file__).resolve().parents[1] / "models" / request.param
    return str(path)
