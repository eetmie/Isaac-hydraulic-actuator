# Initial GB10 measurements — 17 October 2026

This is a development baseline, not a comparison with Newton or a calibrated
excavation-force model. Full settings, versions, source hashes and per-run
validation are in [`benchmarks/gb10-20261017.json`](benchmarks/gb10-20261017.json).

Hardware: NVIDIA GB10, sm_121, driver 580.178.04; Linux aarch64, Python 3.12.3,
Warp 1.17.0, NumPy 2.5.4. CUDA graph replay enabled. No other soil/test/rendering
job ran concurrently with this benchmark. Desktop GPU services remained active.

Every environment contains 2,888 particles and a 101 × 61 node grid. Each
20 ms command frame advances 40 physics substeps of 0.5 ms. All environments
execute the complete eight-second prescribed dig, including contact, lift and
dump. They use identical initial conditions; this measures throughput, not a
randomized stability campaign. Five warm-up frames precede reset and measurement.

```bash
.venv-soil2d/bin/python -m soil2d benchmark --device cuda:0 \
  --envs 1 8 32 --seconds 8 --out runs/soil2d/gb10-repeat.json
```

| Environments | Frame p50 / p95, ms | Physics real-time factor per environment | Aggregate simulated s / wall s | Real-time factor with validation, per environment | Owned solver arrays, MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.443 / 2.071 | 13.26 | 13.26 | 9.19 | 0.25 |
| 8 | 5.471 / 6.736 | 3.56 | 28.46 | 2.28 | 2.00 |
| 32 | 17.187 / 20.562 | 1.15 | 36.72 | 0.68 | 7.99 |

Physics timing includes host command upload, graph launches and device
synchronization. It excludes per-frame state validation and all rendering/file
I/O. The separate validated loop includes host state copies and checks. The
memory column counts owned Warp array capacities only, not total GPU memory.

Every frame passed finite-state, positive elastic determinant, grid mass and CFL
checks. Worst measured dynamic CFL was 0.3361; the largest residual bucket
projection was 0.129 mm. Initial represented soil mass was 11.913 kg per scene.

## Timestep and grid sensitivity

The full default demo was run at two timesteps and two spatial resolutions.
[`benchmarks/sensitivity-20261017.json`](benchmarks/sensitivity-20261017.json)
records the settings, diagnostics and tracking CSV hashes. The physical bucket
wall and the complete 11 mm contact radius were held fixed during spatial
refinement; half-sized cells therefore used one cell of padding instead of half.
The command period remained 20 ms, and particle count increased with resolution.

| Grid / timestep | Particles | Peak reaction, N | Integrated force magnitude, N·s | Integrated bucket Fx / Fz, N·s |
| --- | ---: | ---: | ---: | ---: |
| 10 mm / 0.5 ms | 2,888 | 232.31 | 193.06 | −93.33 / 118.16 |
| 10 mm / 0.25 ms | 2,888 | 263.15 | 211.34 | −100.00 / 138.12 |
| 5 mm / 0.25 ms | 11,552 | 174.19 | 150.92 | −69.86 / 97.08 |

All three completed the eight-second cycle and passed the frame checks. However,
halving the timestep changed peak reaction by +13.3%; halving the grid spacing
and timestep changed it by −25.0% relative to the default. This is material
resolution sensitivity. These samples do **not** demonstrate force convergence.
Reaction traces are suitable for developing interfaces and experiments; they
are not yet targets for fitting the real hydraulic plant or judging controller
performance. Compare penetrated volume, repose, load and contact padding during
further refinement/calibration before using loads quantitatively.

The timestep check is reproducible with:

```bash
.venv-soil2d/bin/python -m soil2d demo --device cuda:0 --dt 0.00025
```

For the fixed-envelope spatial check, construct the scene through Python:

```python
from dataclasses import replace
from soil2d import Scene
from soil2d.solver import Soil

base = Scene()
scene = replace(base, dx=0.005, dt=0.00025,
                bucket=replace(base.bucket, contact_padding_cells=1.0))
solver = Soil(scene, device="cuda:0")
```

Replay `digging_path().command(solver.time, scene.frame_dt)` for 400 frames,
calling `solver.check()` after each. `soil2d.cli.run_demo` was used to save the
same outputs as the CLI. On this workstation the raw runs and visual preview
are under `runs/soil2d/mvp-dig`, `dt-half` and `grid-half`; generated media is not
versioned. A full additional run at friction angle 38° and cohesion 100 Pa also
completed without numerical faults.

The Newton benchmark still needs the prepared scene/assets and a shared
prescribed-pose adapter. No speed ratio against its 3D solver has been measured.
