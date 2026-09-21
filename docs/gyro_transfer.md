# Gyro-history controller and real-bucket deployment

## Result (2026-09-16)

The deployment profile uses the training USD's **bucket cutting tip**, not the
legacy robot tool offset. The real-robot repository was not modified. Export
copies its IMU/PWM configuration and replaces only the kinematic geometry in a
separate `bucket_control_config.yaml`. The runner loads that generated profile.

| Effective zero-angle geometry | Existing robot profile | USD-derived profile |
| --- | ---: | ---: |
| Boom length | 468 mm | 467.713 mm |
| Arm length | 250 mm | 250.012 mm |
| Arm vector Z component | 0 mm | -2.500 mm |
| Bucket pivot to tip X | +31 mm | -12.196 mm |
| Bucket pivot to tip Z | -142 mm | -132.757 mm |

The significant mismatch was the **tool offset**, about 44 mm, not the main
link lengths. The generated origin is the USD lower-carriage frame; the old
robot profile used a different height origin (56 mm difference). Do not mix
absolute tip coordinates from those two profiles. The export also carries the
bucket orientation offset and verifies FK, Jacobians and tool angle at 128 poses.

The USD skill's read-only inspection confirmed a meter-scaled, Z-up asset with
`/excavator` as default/articulation root. No USD hierarchy or asset was edited
for deployment. Existing user asset changes are preserved.

### Plant experiment

Both candidates started from `models/arm_v5_steady`, with identical recording
splits, seed 731, 400 updates, batch 64, 100-step training rollouts and learning
rate 1e-5. One used smoothed velocity history; the other used causal relative
gyro history. Both retained current joint positions. Selection used development
recordings before preparing the held-out test data.

| Model | Development 10 s path error | Test 10 s path error | Test 30 s path error |
| --- | ---: | ---: | ---: |
| Existing steady plant | 3.39° | 1.96° | 2.98° |
| Smooth-history fine-tune | 4.40° | 3.17° | 6.55° |
| Gyro-history fine-tune | 4.69° | 3.81° | 8.74° |

These are mean absolute joint-angle errors over unrolled trajectories, not tip
errors. All evaluations start from the same causal gyro histories. Fine-tuning
also regressed neutral-hold behavior (~0.9–1.0° mean drift over 10 seconds versus
approximately zero for the original). **Neither candidate was promoted.** This
small experiment does not rule out a better gyro-input plant trained with a
larger budget and preserved neutral/stability constraints. Position-history
models were not trained in this experiment.

The selected plant stays frozen. The controller was separately fine-tuned for
300 PPO iterations with gyro-style observation noise, persistent per-link bias
and 0–20 ms sensor delay. Physical plant state and reward remain noise-free;
sensor errors persist in the observation history rather than being redrawn
independently for old samples.

### Fixed-tip simulation

Checkpoint: `logs/rsl_rl/hydraulic_controller/2026-09-16_20-03-52_gyro_transfer/model_1798.pt`,
shipped as `models/controller_proto/model_1798.pt` with repository-relative paths.

100 cases: five starting poses × five plant variations × two sensor modes ×
two amplitudes. Each performed three smooth `0 → +amplitude → -amplitude → 0`
cycles, reference rate at most 3°/s, followed by settling.

| Sweep | Ideal sensors: worst tip drift | Gyro-style sensors: worst tip drift |
| --- | ---: | ---: |
| ±5° | 2.84 mm | 4.55 mm |
| ±10° | 2.99 mm | 4.89 mm |

No simulated limit, collision or invalid-state failures. The previous controller
had a worst gyro-perturbed drift of 5.19 mm in the same rotation suite using its
equivalent USD kinematics. This is a small simulated improvement, not proof of
real-machine improvement. The plant and controller still share a learned plant;
unmodeled load, oil temperature, stiction and calibration errors remain risks.

Artifacts (local, ignored by Git):

- `runs/gyro_transfer/selection.json`: frozen plant selection and budgets.
- `runs/gyro_transfer/evaluation_test.json`: held-out plant metrics.
- `runs/gyro_transfer/data_development.json`: source hashes and gyro diagnostics.
- `runs/gyro_transfer/deployment_bucket/`: final portable bundle (schema 2).
- `runs/gyro_transfer/rotation_deployment/rotation_results.json`: simulation acceptance report.
- `runs/gyro_transfer/rotation_deployment/rotation_traces.npz`: pose/valve traces.

The older `deployment_bundle/` directory is an intermediate schema-1 export;
use `deployment_bucket/` with the current runner.

## Sensor and runtime contract

Joint order is boom, arm, bucket, carriage pitch. Angles are radians and rates
are radians/second. With parallel IMU Y axes, joint rates are
`[boomY-baseY, armY-boomY, bucketY-armY, baseY]`, converted from degrees/second.
Mounting corrections are applied once to quaternion-derived angles. The
firmware's startup gyro bias correction is retained, without a second software
bias subtraction or noncausal filter.

Offline preparation uses the latest complete device packet available at each
recorded host tick. Centered smooth velocities are targets/diagnostics only in
the gyro experiment. Hardware reads one immutable `_imu_snapshot` for both
orientation and rates; incompatible robot snapshot layouts fail closed.

Measured histories run at 100 Hz; policy inference at 20 Hz. The actor includes
its trained normalization; `tanh` is applied exactly once. The robot's existing
100 Hz direct-command thread retains valve dither; PID joint control is bypassed.
Slew and tracks are forced to zero. No UDP control path is introduced.

The portable bundle contains the actor, tensor geometry, conservative collision
lookup grid, generated control profile and checksums. It needs NumPy/PyTorch
for inference, not Isaac Sim, USD, SciPy or RSL-RL. The hardware adapter also
needs the existing `kaivuriprokkis` dependencies. TorchScript uses the same
export route as the current controller; newer PyTorch reports deprecation
warnings, so verify compatibility with the Jetson's installed PyTorch version.

Local CPU smoke tests matched exported actor output exactly. A warmed
geometry/observation/actor loop measured about 1.8 ms median and 2.9 ms maximum
over 100 samples on the development PC. **Jetson timing is not yet measured.**

## First robot session: shadow only

Copy this project's Python source and `deployment_bucket/` bundle to the Jetson.
The MLP plant, training logs and USD assets are not needed for bundle inference.
Keep a copy of the simulation report for the later motion step. The original
Jetson profile's `control_config.yaml`, `servo_config.yaml` and `profile.yaml` must match the
exported source checksums; calibration changes require a new export and checks.

Run from this project's root on the Jetson, with the actual repository path:

```bash
python run_robot_controller.py shadow \
  --bundle runs/gyro_transfer/deployment_bucket \
  --robot_repo /absolute/path/to/kaivuriprokkis \
  --seconds 60 --log runs/shadow_bucket.csv
```

Shadow does **not initialize PWM**. Keep the machine stationary through firmware
calibration and history warmup. Review measured joint signs, gyro residuals,
tip geometry, timestamps and inference times. Log paths must be new (existing
logs are never overwritten). The sweep preflight requires a clear, reachable
pose even in shadow mode; start near the simulated home pose when practical.

## Motion only after physical checks

Check the actual bucket dimensions/IMU mounting against the selected USD, and
verify valve directions with the robot's established procedures. Clear and
restrict the work area, use level stable ground, no payload or contact, and
have an independent physical emergency stop/operator. Run no other process
that writes PWM. Measure drift externally: IMU-based FK alone cannot establish
physical cutting-tip accuracy.

Start with ±5°, using the matching report:

```bash
python run_robot_controller.py rotate \
  --bundle runs/gyro_transfer/deployment_bucket \
  --robot_repo /absolute/path/to/kaivuriprokkis \
  --validation_report runs/gyro_transfer/rotation_deployment/rotation_results.json \
  --amplitude_deg 5 --log runs/rotation_bucket_5deg.csv
```

After neutral warmup, hold **Left Bumper** on the connected local Xbox controller
to enable motion. Releasing it, disconnecting the controller, or pressing **B**
latches a stop. Restart the process to re-arm. The full sweep is validated with
the pump off before the timed loop; movement since that preflight is rejected.

Additional stops include stale IMU (>50 ms), stale policy (>150 ms), loop
lateness (>30 ms), inference (>40 ms), invalid state/valves, configured angular
velocity bound, joint/collision margins, carriage pitch >0.015 rad, and measured
tip drift >20 mm. Stops request neutral valves and pump off. A separate monitor
checks freshness even without new output writes. These are **software guards,
not a safety-rated stop**: OS/GIL/I2C stalls or power faults can defeat them.
Physical emergency stop and hydraulic load-holding provisions remain necessary.

Only attempt ±10° after a successful externally measured ±5° run. Acceptance
is three cycles with no faults and externally measured maximum tip drift within
10–20 mm. No physical acceptance test has been performed yet.

## Reproduce locally

Use the Isaac Lab Python environment, from this project directory:

```powershell
..\..\isaaclab.bat -p -m training.gyro_transfer prepare --out runs/gyro_transfer_new
..\..\isaaclab.bat -p -m training.gyro_transfer train --out runs/gyro_transfer_new
..\..\isaaclab.bat -p -m training.gyro_transfer prepare --split test --out runs/gyro_transfer_new
..\..\isaaclab.bat -p -m training.gyro_transfer evaluate --split test --out runs/gyro_transfer_new
```

The paired experiment needs the local raw-recording archive under
`local/prototyping`; it is not bundled with source. Controller fine-tuning used:

```powershell
..\..\isaaclab.bat -p train_controller.py --run_name gyro_transfer --resume logs/rsl_rl/hydraulic_controller/2026-09-15_19-54-48_v8_ent01_nogov_s3/model_1499.pt --max_iterations 300 --num_envs 2048 --save_interval 100 --entropy_coef 0.01 --curriculum_iterations 0 --set gyro_observations=true governor_joint_speed_margin=0.0 joint_speed_scale_range=[0.75,1.3]
..\..\isaaclab.bat -p run_robot_controller.py export --checkpoint logs/rsl_rl/hydraulic_controller/2026-09-16_20-03-52_gyro_transfer/model_1798.pt --bundle runs/new_bucket_bundle --robot_repo C:/Users/sh23937/Documents/GitHub/kaivuriprokkis
..\..\isaaclab.bat -p -m hydraulic_controller.rotation_benchmark --checkpoint logs/rsl_rl/hydraulic_controller/2026-09-16_20-03-52_gyro_transfer/model_1798.pt --out runs/new_rotation_report --robot_profile runs/new_bucket_bundle/bucket_control_config.yaml
..\..\isaaclab.bat -p -m pytest training/test_gyro_transfer.py training/test_hydraulic_controller.py -q
```
