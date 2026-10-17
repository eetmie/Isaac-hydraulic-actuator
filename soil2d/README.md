# Lightweight 2D soil prototype

Standalone soil development environment for the hydraulic model: a hollow bucket
cuts, curls, lifts, transports and dumps a small pile, while the solver records
the reaction wrench. Runs on CPU or CUDA without Isaac Sim. **Experimental:
numerical checks pass, but material properties and digging forces are not
calibrated against the real machine.**

The first stage uses prescribed bucket motion. The hydraulic MLP, robot joint
limits, RL policy and physical load feedback are not connected yet.

## Run

From the repository root, use a separate environment to keep Isaac's packages
independent:

```bash
python3 -m venv .venv-soil2d
.venv-soil2d/bin/python -m pip install -r soil2d/requirements.txt
.venv-soil2d/bin/python -m soil2d view
```

The viewer has friction and cohesion sliders. **Apply and reset** rebuilds the
scene with those settings; Space pauses and R resets. The scripted eight-second
cycle repeats. A desktop Matplotlib backend is required for this command; the
commands below work headlessly. On this Linux machine, install `python3-tk` if
your Python installation has no GUI backend.

```bash
# Headless full cycle, plots, state, CSV and optional GIF.
.venv-soil2d/bin/python -m soil2d demo --device cuda:0 --gif

# CPU fallback; leave off --device to select CUDA when available.
.venv-soil2d/bin/python -m soil2d demo --device cpu --seconds 1

# Soil-only settling control and a separate material experiment.
.venv-soil2d/bin/python -m soil2d demo --no-bucket --seconds 2
.venv-soil2d/bin/python -m soil2d demo --friction-angle 38 --cohesion 100 --gif

# Warmed physics timings, with validation outside each timed frame.
.venv-soil2d/bin/python -m soil2d benchmark --device cuda:0 --envs 1 8 32 --seconds 8
```

Results go to new paths under `runs/soil2d/`. `--out` accepts a new directory for
`demo`, or a new JSON file for `benchmark`; existing results are never overwritten.
`requirements-tested.txt` records the exact environment used for development on
Python 3.12 / Linux aarch64; the main requirements allow compatible versions.

## Outputs and replay

Each completed demo writes:

| File | Contents |
| --- | --- |
| `tracking.csv` | End-of-frame bucket pose, commanded twist, mean reaction, translational particle kinetic energy, soil centroid, CFL, projection and physics timing |
| `bucket_poses.csv` | Initial pose plus every frame endpoint, for a common replay path |
| `summary.json` | Material and geometry, device/versions, source hashes, trajectory, validation and timing scope |
| `final_state.npz` | Particle position, velocity, elastic deformation and final measurements |
| `final.png`, `forces.png` | Final scene and reaction trace |
| `preview.gif` | Optional 20 fps visualization, requested with `--gif` |

Replay an exported path with `--trajectory runs/soil2d/<run>/bucket_poses.csv`.
The CSV contract is:

```csv
time_s,bucket_x_m,bucket_z_m,bucket_angle_rad
0.0,0.22,0.38,0.0
1.0,0.22,0.38,0.0
1.8,0.32,0.25,-0.15
```

Times must start at zero and increase strictly; angles must be explicitly
unwrapped. CSV poses interpolate linearly. The built-in path uses quintic
interpolation, then commands a constant twist over each control frame so the
bucket reaches the next sampled pose exactly. Use the same control period and
exported frame samples for comparisons. After the last sample the bucket holds.

## Model and conventions

The implementation uses quadratic MLS-MPM/APIC transfers, a logarithmic Hencky
elastic stress and Drucker-Prager strain projection with a tensile apex. The
bucket is an open chain of capsule walls, not a filled solid, so soil can enter
and leave. Grid contact removes inward relative velocity and clamps Coulomb
friction to the available slip. Residual particle penetration is projected out;
its velocity correction also contributes to the recorded reaction impulse.

Default scene:

| Setting | Value |
| --- | ---: |
| Domain | 1.0 × 0.6 m, contained boundaries with a two-cell margin |
| Grid spacing / particle spacing | 10 mm / 5 mm |
| Particle count | 2,888 per environment |
| Density / effective section width | 1,500 kg/m³ / 0.11 m |
| Initial represented mass | 11.913 kg |
| Young's modulus / Poisson ratio | 20 kPa / 0.2 |
| Friction angle / cohesion | 32° / 0 Pa |
| Physics / command period | 0.5 ms / 20 ms, 40 substeps |
| Bucket wall / numerical padding | 12 mm / half a grid cell |

The default mass is an illustrative soil section, not a measured load for this
excavator. Tune geometry and soil depth to the real bucket before force fitting.

- Coordinates are `x` horizontal and `z` upward, in metres. Positive angle/torque
  is counterclockwise in the x/z plot, equivalent to **negative world-Y** rotation.
- `Soil.wrench()` returns **soil on bucket** `[Fx, Fz, torque_ccw]` in N/N·m,
  averaged over the last command frame. Each impulse moment is about the
  instantaneous bucket origin. Boundary reactions are logged separately.
- Effective width multiplies particle volume, mass and force; it does not add
  lateral soil flow. Doubling width leaves planar motion unchanged and doubles
  reaction, which is covered by a test.
- The pale bucket outline in previews shows contact padding. Particle projection
  is a numerical correction, not a measured collision law; its maximum distance
  is logged. Translational kinetic energy excludes APIC affine energy and elastic
  energy, so it is not a total-energy conservation measurement.

`Scene`, `Material` and `Bucket` are immutable dataclasses in `config.py`.
`Soil(scene, envs=N, device="cuda:0")` batches independent scenes and captures a
whole command frame as a CUDA graph. `step(pose, twist)` accepts shared `(3,)` or
per-environment `(N, 3)` commands. `reset()` restores the seeded particles;
`snapshot()` copies state to the host; `check()` rejects invalid deformation,
escaped stencils, nonfinite values, lost transfer mass and excessive CFL.

The fixed seed and prescribed path make experiments repeatable. CUDA scatter
atomics can change floating-point summation order; bitwise determinism is not
guaranteed.

Stiffness, resolution and timestep must be chosen together. An initial elastic
CFL above 0.35 is rejected; frame checks reject a dynamic CFL above 0.5. The tool
may move at most a quarter cell per substep. Start a convergence study by halving
`--dt` and then `--dx` with a suitable timestep. These checks catch common
failures; they do not establish physical accuracy or guarantee every parameter
combination is stable.

## Research basis and remaining work

Werner et al., [Size Doesn't Matter: Material-State Reinforcement Learning for
Excavator Transferable Soil Manipulation](https://arxiv.org/abs/2609.12677)
(2026), Appendix A, motivates the small batched 2D MPM environment. Their solver
includes compaction memory, material hardening, a modified corotated stress and
a separate shovel grid. This prototype uses a simpler Hencky constitutive law
and analytic walls. Compaction memory, hardening, changing reference volume and
the paper's friction cap remain unimplemented. It is not a reproduction of their
results. No public author implementation was verified during this work.

The numerical starting points also include Jiang et al.'s
[APIC paper](https://doi.org/10.1145/2766996) and Klár et al.'s
[Drucker-Prager sand model](https://doi.org/10.1145/2897824.2925906).
GPU kernels use [NVIDIA Warp](https://nvidia.github.io/warp/), including its
two-dimensional SVD. This repository's Apache-2.0 license applies to the new code.

Initial refinement runs show material force sensitivity to timestep and grid
spacing; see the [measurements](benchmark-notes.md). Next checks are further
convergence work, repose and penetration calibration, then compaction memory.
Extend manual data collection with known contact/load
cases before conditioning the hydraulic model on load. The current
`DirectIntegrationActuator` overwrites joint state, so attaching soil geometry
alone cannot make its MLP slow down. A future adapter must transform the bucket
wrench to joint loads using a consistent Jacobian/reference point and connect
it to a calibrated load-aware plant. RL follows that connection.

## Newton comparison

The existing interactive reference is
[AI-MaSi/IsaacSim, sand_teleop.py](https://github.com/AI-MaSi/IsaacSim/blob/main/IsaacLab/sand_teleop/sand_teleop.py),
with companion `sand_excavator.py` and `sim_common.py`. It uses Newton's 3D
`SolverImplicitMPM` and a PD-driven rigid excavator. It does not use this MLP.
Its existing headless controls support, from a prepared demo checkout with assets:

```bash
python sand_teleop.py --viewer null --autoplay --num_frames 800 \
  --fps 100 --substeps 4 --soil dry_sand --log /tmp/newton-soil.csv
python sand_teleop.py --viewer null --autoplay --num_frames 800 \
  --fps 100 --substeps 4 --no_sand --log /tmp/newton-rigid.csv
```

These commands do not yet replay the common bucket-origin CSV. The Newton side
still needs that adapter plus timing at synchronized frame boundaries. Its
existing target-following path may change under load; comparing those two runs
directly to this prescribed path would mix motion and solver effects.

Use the same machine, physical duration, prescribed path and comparable bucket,
density and soil depth. Warm up compilation, time headless physics separately
from rendering/diagnostics, and record p50/p95 frame time, particle count,
resolution, material settings, force/displacement traces and memory scope. A
no-soil Newton run estimates rigid solver overhead; subtraction is approximate.
Report the different 2D/3D physical scope explicitly. **No current Newton run or
measured speedup against Newton is claimed.**

The benchmark JSON includes physics-only throughput and a separate loop time
with per-frame validation. `solver_buffer_bytes` counts owned Warp arrays; it
excludes CUDA context, graphs, allocator caches and renderer memory. Run GPU
benchmarks alone. A measured baseline for this machine is recorded in
[`benchmark-notes.md`](benchmark-notes.md).

## Verify

```bash
.venv-soil2d/bin/python -m pip install -r soil2d/requirements-dev.txt
.venv-soil2d/bin/python -m pytest soil2d/tests -q
.venv-soil2d/bin/ruff check soil2d
.venv-soil2d/bin/ruff format --check soil2d
```

Tests cover analytical strain/contact responses, translation and rotation,
settling/mass, actual soil momentum versus measured bucket impulse, width
scaling, batch isolation, reset, CPU/CUDA/graph agreement and CSV/CLI behavior.
CUDA checks skip when CUDA is unavailable. The package is optional for existing
training workflows; its tests skip if Warp is absent.
