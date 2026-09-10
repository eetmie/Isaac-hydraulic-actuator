# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Deterministic simulation throughput reporting, without launching Kit."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_common import RealtimeReporter


def test_recent_and_average_exclude_startup():
    reporter = RealtimeReporter(2.0, wall_s=100.0, sim_s=5.0)
    assert reporter.update(wall_s=100.0, sim_s=5.0) is None
    assert reporter.update(wall_s=101.0, sim_s=5.5) is None
    message = reporter.update(wall_s=102.0, sim_s=6.0)
    assert "real-time=0.50x" in message
    assert "2.00x slower" in message
    assert "average=0.50x" in message
    message = reporter.update(wall_s=104.0, sim_s=10.0)
    assert "real-time=2.00x" in message
    assert "2.00x faster" in message
    assert "average=1.25x" in message


def test_pause_counts_wall_time_but_not_simulated_time():
    reporter = RealtimeReporter(2.0, wall_s=0.0, sim_s=0.0)
    message = reporter.update(wall_s=3.0, sim_s=0.0)
    assert "real-time=0.00x" in message
    assert "no simulated time advanced" in message
    assert "inf" not in message


def test_disabled():
    reporter = RealtimeReporter(0.0, wall_s=0.0, sim_s=0.0)
    assert reporter.update(wall_s=100.0, sim_s=1.0) is None


@pytest.mark.parametrize("interval", [-1.0, float("nan"), float("inf")])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        RealtimeReporter(interval, wall_s=0.0, sim_s=0.0)
