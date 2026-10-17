"""Physical invariants and analytical contacts, independent of a rendered demo."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("warp")

import warp as wp

from soil2d import Bucket, Scene
from soil2d import kernels as k
from soil2d.solver import Soil


@wp.kernel
def strain_probe(values: wp.array(dtype=wp.vec2), output: wp.array(dtype=wp.vec2), cfg: k.Parameters):
    p = wp.tid()
    output[p] = k.project_strain(values[p], cfg.mu, cfg.lam, cfg.alpha, cfg.cohesion)


@wp.kernel
def contact_probe(values: wp.array(dtype=wp.vec2), output: wp.array(dtype=wp.vec2)):
    p = wp.tid()
    output[p] = k.contact_velocity(values[p], wp.vec2(0.0), wp.vec2(0.0, 1.0), 0.5)


def small_scene(**changes):
    return replace(Scene(), dx=0.02, soil_min=(0.35, 0.08), soil_max=(0.55, 0.24), **changes)


def test_drucker_prager_elastic_compression_tensile_apex_and_shear_yield():
    solver = Soil(small_scene(), device="cpu")
    values = np.array([[-0.02, -0.02], [0.02, 0.03], [-0.05, 0.01]], dtype=np.float32)
    incoming = wp.array(values, dtype=wp.vec2, device="cpu")
    output = wp.empty(3, dtype=wp.vec2, device="cpu")
    wp.launch(strain_probe, 3, [incoming, output, solver.params], device="cpu")
    projected = output.numpy()
    np.testing.assert_allclose(projected[0], values[0], atol=1e-8)
    np.testing.assert_allclose(projected[1], 0, atol=1e-8)
    original, returned = values[2], projected[2]
    np.testing.assert_allclose(returned.sum(), original.sum(), atol=1e-7)
    assert np.linalg.norm(returned - returned.mean()) < np.linalg.norm(original - original.mean())
    cfg = solver.params
    yield_value = (
        np.linalg.norm(returned - returned.mean())
        + cfg.alpha * (cfg.lam + cfg.mu) / cfg.mu * returned.sum()
        - cfg.cohesion
    )
    assert abs(yield_value) < 1e-7


def test_contact_removes_inward_velocity_and_coulomb_friction_cannot_reverse_slip():
    values = np.array([[2, -1], [0.1, -1], [2, 1], [0, -1]], dtype=np.float32)
    incoming = wp.array(values, dtype=wp.vec2, device="cpu")
    output = wp.empty(4, dtype=wp.vec2, device="cpu")
    wp.launch(contact_probe, 4, [incoming, output], device="cpu")
    np.testing.assert_allclose(output.numpy(), [[1.5, 0], [0, 0], [2, 1], [0, 0]], atol=1e-7)


def test_force_free_translation_preserves_velocity_and_momentum():
    solver = Soil(small_scene(gravity=0.0), device="cpu")
    velocity = np.zeros((1, solver.particles, 2), dtype=np.float32)
    velocity[..., 0] = 0.1
    solver.v.assign(velocity)
    for _ in range(3):
        solver.step(enabled=False)
    solver.check()
    np.testing.assert_allclose(solver.v.numpy(), velocity, atol=2e-5)
    np.testing.assert_allclose(solver.x.numpy(), solver.initial + solver.time * velocity, atol=3e-6)
    np.testing.assert_array_equal(solver.wrench(), 0)


def test_rotation_has_no_elastic_force_at_rest():
    solver = Soil(small_scene(gravity=0.0), device="cpu")
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    solver.f.assign(np.broadcast_to(rotation, (1, solver.particles, 2, 2)).copy())
    solver.step(enabled=False)
    np.testing.assert_allclose(solver.x.numpy(), solver.initial, atol=1e-6)
    np.testing.assert_allclose(solver.v.numpy(), 0, atol=2e-4)
    solver.check()


def test_settling_retains_mass_and_particles_and_dissipates_motion():
    solver = Soil(small_scene(), device="cpu")
    energies = []
    for _ in range(50):
        solver.step(enabled=False)
        solver.check()
        energies.append(float(np.sum(solver.v.numpy().astype(float) ** 2)))
    state = solver.snapshot()
    assert state["position"][0, :, 1].mean() < solver.initial[0, :, 1].mean() - 0.025
    assert energies[-1] < max(energies) * 0.2
    assert state["position"].min() >= 2 * solver.scene.dx - 1e-6
    np.testing.assert_allclose(
        solver.grid_mass.numpy().sum(dtype=float), solver.particle_mass * solver.particles, rtol=2e-6
    )


def test_measured_bucket_impulse_balances_actual_soil_momentum_change():
    bucket = Bucket(vertices=((0.0, -0.10), (0.0, 0.10)), wall_thickness=0.02)
    scene = replace(small_scene(gravity=0.0), bucket=bucket, soil_min=(0.32, 0.12), soil_max=(0.44, 0.24))
    solver = Soil(scene, device="cpu")
    initial_v = np.zeros((1, solver.particles, 2), dtype=np.float32)
    initial_v[..., 0] = 0.3
    solver.v.assign(initial_v)
    impulse = np.zeros(2)
    boundary_impulse = np.zeros(2)
    for _ in range(12):
        solver.step((0.47, 0.19, 0.0))
        solver.check()
        impulse += solver.impulse.numpy()[0, :2]
        boundary_impulse += solver.boundary_impulse.numpy()[0, :2]
    momentum_change = solver.particle_mass * (solver.v.numpy() - initial_v).sum(axis=(0, 1), dtype=float)
    assert impulse[0] > 0.05
    np.testing.assert_allclose(momentum_change + impulse + boundary_impulse, 0, atol=2e-4)


def test_effective_width_scales_mass_and_reaction_without_changing_planar_motion():
    bucket = Bucket(vertices=((0.0, -0.10), (0.0, 0.10)), wall_thickness=0.02)
    scene = replace(small_scene(gravity=0.0), bucket=bucket, soil_min=(0.32, 0.12), soil_max=(0.44, 0.24))
    solvers = [Soil(replace(scene, effective_width=width), device="cpu") for width in (0.11, 0.22)]
    for solver in solvers:
        velocity = np.zeros((1, solver.particles, 2), dtype=np.float32)
        velocity[..., 0] = 0.3
        solver.v.assign(velocity)
    impulses = np.zeros((2, 3))
    for _ in range(12):
        for i, solver in enumerate(solvers):
            solver.step((0.47, 0.19, 0.0))
            solver.check()
            impulses[i] += solver.impulse.numpy()[0]
    assert impulses[0, 0] > 0.05
    np.testing.assert_allclose(impulses[1], 2 * impulses[0], atol=1e-6)
    np.testing.assert_allclose(solvers[1].x.numpy(), solvers[0].x.numpy(), atol=1e-6)
    assert solvers[1].particle_mass == 2 * solvers[0].particle_mass


def test_batch_environments_do_not_exchange_mass_or_momentum():
    scene = small_scene(gravity=0.0)
    batched = Soil(scene, envs=2, device="cpu")
    velocities = np.zeros((2, batched.particles, 2), dtype=np.float32)
    velocities[0, :, 0] = 0.1
    velocities[1, :, 0] = -0.2
    batched.v.assign(velocities)
    batched.step(enabled=False)
    for env in range(2):
        single = Soil(scene, device="cpu")
        single.v.assign(velocities[env : env + 1])
        single.step(enabled=False)
        np.testing.assert_allclose(batched.x.numpy()[env], single.x.numpy()[0], atol=1e-7)
        np.testing.assert_allclose(batched.v.numpy()[env], single.v.numpy()[0], atol=1e-7)


def test_reset_restores_particles_elastic_state_and_measurements():
    solver = Soil(small_scene(), device="cpu")
    for _ in range(4):
        solver.step(enabled=False)
    solver.reset()
    assert solver.time == 0
    np.testing.assert_array_equal(solver.x.numpy(), solver.initial)
    np.testing.assert_array_equal(solver.v.numpy(), 0)
    np.testing.assert_array_equal(solver.f.numpy(), np.broadcast_to(np.eye(2), (1, solver.particles, 2, 2)))
    np.testing.assert_array_equal(solver.wrench(), 0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable")
def test_cpu_cuda_and_graph_replay_agree_and_refresh_commands():
    scene = replace(
        small_scene(gravity=0.0),
        bucket=Bucket(vertices=((0.0, -0.10), (0.0, 0.10)), wall_thickness=0.02),
        soil_min=(0.32, 0.12),
        soil_max=(0.44, 0.24),
    )
    solvers = [
        Soil(scene, envs=2, device=device, graph=graph)
        for device, graph in [("cpu", False), ("cuda:0", False), ("cuda:0", True)]
    ]
    for solver in solvers:
        velocity = np.zeros((2, solver.particles, 2), dtype=np.float32)
        velocity[..., 0] = 0.3
        solver.v.assign(velocity)
    impulses = np.zeros((3, 2, 3))
    for frame in range(5):
        pose = np.array([[0.47 + frame * 0.0002, 0.19, 0.02], [0.47 - frame * 0.0001, 0.19, -0.02]])
        twist = np.array([[0.01, 0.0, 0.1], [-0.005, 0.0, -0.1]])
        for i, solver in enumerate(solvers):
            solver.step(pose, twist)
            solver.check()
            impulses[i] += solver.impulse.numpy()
            np.testing.assert_allclose(solver.pose.numpy(), pose + twist * scene.frame_dt, atol=5e-6)
    assert impulses[0, :, 0].min() > 0.05
    np.testing.assert_allclose(impulses[1:], np.broadcast_to(impulses[0], (2, 2, 3)), atol=2e-4)
    for solver in solvers[1:]:
        np.testing.assert_allclose(solver.x.numpy(), solvers[0].x.numpy(), atol=3e-5)
        np.testing.assert_allclose(solver.v.numpy(), solvers[0].v.numpy(), atol=5e-4)
