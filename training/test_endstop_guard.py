# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""A held valve cannot induce learned rebound from a software end stop."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from endstop_guard import EndStopGuard


def guard():
    return EndStopGuard(np.array([[-1, 1]] * 3), 0.01)


def test_unconstrained_motion_is_bit_exact():
    g = guard()
    v = np.array([0.2, -0.1, 0.05], np.float32)
    np.testing.assert_array_equal(g.apply(np.zeros(3), v, np.ones(3)), v)


def test_arrival_then_opposite_prediction_is_held():
    g = guard()
    # Preserve the arrival increment; actuator still does the final hard clamp.
    np.testing.assert_array_equal(g.apply([0.999, 0, 0], [0.3, 0, 0], [1, 0, 0]), [0.3, 0, 0])
    np.testing.assert_array_equal(g.apply([1, 0, 0], [-0.4, 0.2, 0], [1, 0, 0]), [0, 0.2, 0])


def test_lower_stop_and_independent_axes():
    g = guard()
    g.apply([-1, 0, 1], [-0.3, 0, 0.2], [-1, 0, 1])
    np.testing.assert_array_equal(g.apply([-1, 0, 1], [0.4, 0.15, -0.3], [-1, 0, 1]), [0, 0.15, 0])


def test_reversal_releases_immediately():
    g = guard()
    g.apply([1, 0, 0], [0.3, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(g.apply([1, 0, 0], [-0.4, 0, 0], [-1, 0, 0]), [-0.4, 0, 0])


def test_neutral_release_preserves_ringdown():
    g = guard()
    g.apply([1, 0, 0], [0.3, 0, 0], [1, 0, 0])
    np.testing.assert_array_equal(g.apply([1, 0, 0], [-0.04, 0, 0], [0, 0, 0]), [-0.04, 0, 0])


def test_neutral_arrival_does_not_latch():
    g = guard()
    g.apply([1, 0, 0], [0.3, 0, 0], [0, 0, 0])
    assert not g.latched.any()


def test_reset_clears_latches():
    g = guard()
    g.apply([1, 0, 0], [0.3, 0, 0], [1, 0, 0])
    g.reset()
    assert not g.latched.any()
