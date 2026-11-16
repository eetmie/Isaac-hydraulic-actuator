"""Benchmark sensor disturbances are causal, shared across policies and separate from physics."""

from types import SimpleNamespace

import torch

from hydraulic_controller.observations import SensorController, SensorSettings


class Recorder:
    def reset(self, task):
        pass

    def __call__(self, k, q, v):
        self.seen = q.clone(), v.clone()
        return q[:, :3]


def test_zero_sensor_errors_preserve_measurements():
    target = Recorder()
    sensor = SensorController(target, SensorSettings(0, 0, 0, 0))
    q, v = torch.randn(5, 4), torch.randn(5, 4)
    sensor.reset(SimpleNamespace(q0=q))
    sensor(0, q, v)
    torch.testing.assert_close(target.seen[0], q, rtol=0, atol=0)
    torch.testing.assert_close(target.seen[1], v, rtol=0, atol=0)


def test_delayed_state_contains_only_past_samples_and_preserves_inputs():
    target = Recorder()
    sensor = SensorController(target, SensorSettings(0, 0, 0, 2))
    sensor.reset(SimpleNamespace(q0=torch.zeros(2, 4)))
    sensor.delay.fill_(2)
    for k in range(5):
        q, v = torch.full((2, 4), float(k + 1)), torch.full((2, 4), float(10 * (k + 1)))
        sensor(k, q, v)
        expected = max(0, k - 1)
        torch.testing.assert_close(target.seen[0], torch.full_like(q, expected))
        torch.testing.assert_close(target.seen[1], torch.full_like(v, 10 * expected))
        assert (q == k + 1).all() and (v == 10 * (k + 1)).all()


def test_sensor_draws_match_across_policies_without_global_rng_changes():
    a, b = Recorder(), Recorder()
    wrappers = [SensorController(target, SensorSettings(), seed=738) for target in (a, b)]
    q, v = torch.zeros(8, 4), torch.zeros(8, 4)
    before = torch.get_rng_state().clone()
    for sensor in wrappers:
        sensor.reset(SimpleNamespace(q0=q))
        for k in range(10):
            sensor(k, q, v)
    torch.testing.assert_close(a.seen[0], b.seen[0], rtol=0, atol=0)
    torch.testing.assert_close(a.seen[1], b.seen[1], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), before)
    assert a.seen[0].abs().sum() > 0 and a.seen[1].abs().sum() > 0
