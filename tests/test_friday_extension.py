"""Extension experiment counts must match and fit their measured runtime budget."""

import pytest

from training.friday_extension import matched_iterations


def test_matched_updates_fit_combined_measured_cost_with_runtime_margin():
    rates = [1.0, 2.0]
    count = matched_iterations(36600, rates, 100)
    assert count == 11300
    assert (count - 100) * sum(rates) <= (36600 - 600) * 0.94
    assert matched_iterations(36600, [2.0, 4.0], 100) < count


def test_no_empty_experiment_when_deadline_is_too_close():
    with pytest.raises(ValueError, match="Not enough time"):
        matched_iterations(600, [1.0, 1.5], 100)
