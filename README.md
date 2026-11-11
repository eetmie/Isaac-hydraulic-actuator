# Learned hydraulic actuators for Isaac Lab

A small excavator demo driven by learned hydraulic dynamics: **an actuator MLP for boom, arm,
bucket and carriage pitch**, plus an independent **slew MLP**. Both predict velocity
increments at 100 Hz; the simulator integrates the resulting velocities into joint
positions.

Idea from Egli, P. and Hutter, M. (2020) 'Towards RL-Based Hydraulic Excavator Automation'. Great paper!

## Run the demo

Tested with **Isaac Lab 3.0.0 / Isaac Sim 6.2** on Linux x86_64 and aarch64
(DGX Spark). Isaac Lab 3.0 is installed and run with [uv](https://docs.astral.sh/uv/);
there is no `isaaclab.sh` and no separate Isaac Sim install. Clone this repository
into `IsaacLab/scripts/Isaac-hydraulic-actuator`. Demo and controller commands run
**from the Isaac Lab root**; the dataset workflow below uses the project directory:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git --branch v3.0.0
cd IsaacLab
git clone https://github.com/eetmie/Isaac-hydraulic-actuator.git scripts/Isaac-hydraulic-actuator

uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/sim.py \
    --model scripts/Isaac-hydraulic-actuator/models/actuator \
    --slew-model scripts/Isaac-hydraulic-actuator/models/slew
```

The first `uv run --extra isaacsim` builds the environment and downloads Isaac Sim
6.2 from PyPI (about 20 GB with PyTorch); the first launch also compiles shaders and
takes a few minutes. Platform notes:

- **EULA:** the first Isaac Sim launch asks you to accept the NVIDIA Omniverse EULA.
  Setting `OMNI_KIT_ACCEPT_EULA=YES` accepts it non-interactively, so only set it
  once you have read and accepted the license.
- **aarch64 (DGX Spark, Jetson):** direct Python commands that import Isaac Sim need
  `export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`.
- **Headless:** pass `--viz none`. Without `--viz`, the demos open the Kit window,
  which the gamepad needs.
- **Windows:** the same `uv run` commands work in PowerShell; use `` ` `` instead of
  `\` to continue a line.

The Kit window opens in NN mode. Connect an Xbox-compatible controller:

| Control | Action |
| --- | --- |
| Right stick Y | Boom valve |
| Left stick Y | Arm valve |
| Right stick X | Bucket/tool valve |
| Left stick X | Slew valve |
| A | Toggle learned hydraulics / manual joint velocities |
| B | Reset to the starting pose |

## Train the end-effector controller

Step two of the Egli & Hutter recipe: a PPO policy receives a desired bucket-tip
twist `(X, Z, bucket pitch rate)` and outputs the three valve commands directly.
The selected actuator network is the plant, so training runs as batched
Torch on the GPU with no simulator (~75k steps/s with 4096 environments; a
1,500-iteration run takes about half an hour). From the Isaac Lab root:

```bash
uv run python scripts/Isaac-hydraulic-actuator/train_controller.py --actuator-run scripts/Isaac-hydraulic-actuator/runs/actuator --run_name my_run
```

Design choices, and the measurement behind each:

- **Carriage rocking is not tracked.** The tracked twist comes from the boom,
  arm and bucket joints only. Rocking cannot be driven by the valves, and the
  learned pitch channel rings far longer than the real carriage when started
  outside the recorded ±0.3° pitch range. Resets therefore start at pitch ≈ 0.
- **Valve chatter costs reward.** An L1 + L2 penalty on valve change per policy
  step keeps commands in the range seen in the recordings. Otherwise the policy
  learns to dither around the deadband and runs the actuator model far outside
  its data.
- **Commands match the data.** Recorded tip speeds are mostly 30–150 mm/s, and
  the valves barely move the arm below |u| ≈ 0.2–0.3. Commands span 10–120 mm/s,
  are held 1–3 s and slew at 0.25 m/s². A governor keeps them away from joint
  limits and self-collision (a cached lookup grid).
- **Hidden plant perturbations.** Each episode draws a valve gain (±10 %),
  offset (±0.02) and a 0–20 ms transport delay, plus observation noise. The
  critic sees them; the policy does not. Pass `--no_randomize` for a
  deterministic baseline.
- **Position loop outside the policy.** Draw and circle modes command
  `v_ref + kp·(x_ref − x)` with `kp = 3`, as in the RA-L paper.

### Run the draw demo

The baseline controller ships in `models/controller` and is the default,
so these commands work on a fresh clone without training. They need RSL-RL,
which the Isaac Lab environment already includes. From the Isaac Lab root:

```bash
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/run_controller.py --mode draw
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/run_controller.py --mode circle --speed-mm-s 40
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/run_controller.py --mode gamepad --speed-mm-s 60
```

Playback cruises at 30 mm/s unless `--speed-mm-s` is given. Pass
`--checkpoint path/to/model_N.pt` to run your own training run instead; the
console reports the benchmark's recommended speed when a benchmark matches the
checkpoint.

The benchmark runs held commands with reversals and stops, plus quintic-timed
circles and lines, under nominal and perturbed valves. Results go to CSV/JSON
and a figure under `runs/`:

```bash
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/run_controller.py --mode benchmark --viz none
```

Draw mode holds the starting bucket angle, so only part of the drawing box is
reachable. The window shades the rest dark and turns a stroke red when it enters
that area. A rejected stroke keeps its reason on screen and in the console. Draw
and circle modes also show the planned path and the executed trace in the 3D
viewport.

Gamepad mode maps the left stick to tip X/Z velocity, right stick X to bucket
pitch rate, and B to reset. Everything here is free-space motion in simulation;
digging, contact and the real machine are not validated.

## Experiment with 2D soil

The standalone [`soil2d` prototype](soil2d/README.md) runs a prescribed bucket
through a pile using CPU or CUDA MPM, with a viewer, reaction-force logs, CSV
trajectory replay and batched benchmarks. It needs no Isaac Sim. The soil is
experimental and is not coupled to the learned hydraulic actuator yet; see its
README for setup, model assumptions and the Newton comparison plan.

## Tune the robot's joint PID

The same plant can tune the classic controller on the real machine: the
boom/arm/bucket PIDs of [kaivuriprokkis](https://github.com/eetmie/kaivuriprokkis)
(`modules/pid.py`, ported 1:1 to batched Torch). All three tools read the gains
from a sibling `kaivuriprokkis` checkout (`--robot_repo`) and need no simulator:

```bash
uv run python scripts/Isaac-hydraulic-actuator/measure_speeds.py   # what the plant can reach at all
uv run python scripts/Isaac-hydraulic-actuator/replay_pid.py       # the robot's current gains
uv run python scripts/Isaac-hydraulic-actuator/tune_pid.py         # CMA-ES, writes a pid: YAML block
```

- `measure_speeds.py` maps the achievable tip speed per direction across the
  workspace, with deadband and flow sharing between joints. That is the ceiling
  for any controller.
- The replay runs the robot's own loop (DLS IK step, then PID, then valves) over
  joint steps, ramps and tip lines under the benchmark's valve perturbations.
  Lines the plant cannot follow are flagged rather than blamed on the gains.
- The tuner scores only those feasible lines, from HOME and three central poses,
  with measured IMU jitter, and mixes the mean with the worst plant. It writes
  before/after replays next to the tuned gains.

## Compare PID, MPC and the learned controller

```bash
uv run python scripts/Isaac-hydraulic-actuator/compare_controllers.py --pid_gains scripts/Isaac-hydraulic-actuator/runs/pid_tune/<run>/pid_gains.yaml
```

Runs the robot's PID, the tuned PID, the PPO valve policy and a sampling MPC
(MPPI on the actuator network, deadband-compensated, 20 Hz) on the same
feasible tip lines under every valve/speed perturbation. Each controller sees
only measured angles, rates and its own past commands, as on the robot. In
simulation the MLP and the MPC have an advantage the PID lacks: they were trained
on, or plan with, this very network.


## Deploy to the real robot

The robot runs controllers with kaivuriprokkis `learned_control` (see its
README); it needs no copy of this repository. `export_robot_bundle.py` writes a
self-contained bundle: TorchScript actor, geometry, collision grid, contract and
reference cases the robot replays at load. First generate a robot profile whose
arm geometry is the controller USD's (once per calibration), then export:

```bash
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/export_robot_bundle.py profile \
    --source_profile ../kaivuriprokkis/configuration_files/profiles/jetson --robot jetson_bucket \
    --out ../kaivuriprokkis/configuration_files/profiles/jetson_bucket
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/export_robot_bundle.py export \
    --checkpoint models/controller/model_1798.pt --robot jetson_bucket \
    --out ../kaivuriprokkis/learned_control/bundles/controller_bucket
```

`tests/test_robot_parity.py` checks the robot copy (sibling checkout) against
the training code: kinematics, governor, observations, sensors, circle and PID.
Real-robot sessions are kept under `drive_logs/`.


## Included models

`models/actuator` is the frozen historical actuator baseline; `models/slew` is
an independent slew network. `models/controller` contains the historical PPO
baseline and its plant/geometry contract. Weights and normalization arrays are
unchanged by this cleanup. The baseline actuator has **not** been retrained on
the complete dataset. New MLP and PPO runs write their own artifacts and contracts.

A controller checkpoint pins the exact actuator and USD it was trained with.
If either changes, train a new controller or use the original matching artifacts;
the loader checks those hashes. See [robot deployment](docs/robot_deployment.md).

## How integration works

At each 0.01 s step, the network predicts a velocity increment:

```text
velocity_next = velocity + model(position, velocity_history, valve_history)
position_next = position + 0.01 * velocity_next
```

The actuator selects `direct` integration automatically. The learned joints are written
directly into simulation; their physical drives and carriage springs are disabled
so they do not fight the predicted motion. This demo models free motion and does
not provide a validated contact or digging-force response. Arm limits still apply,
and a held valve at an end stop cannot induce a learned rebound.

The assets include the tested target-drive gains: arm 2400 N·m/rad and
120 N·m·s/rad; slew 600 N·m/rad and 40 N·m·s/rad. The alternative `target`
integration route remains available for compatible three-joint models; the four-channel actuator requires
`direct`.

For direct Python use, only NumPy and PyTorch are needed. From the Isaac Lab root:

```python
import sys
import numpy as np

sys.path.insert(0, "scripts/Isaac-hydraulic-actuator")
from actuators import HydraulicActuatorNet

model = HydraulicActuatorNet("scripts/Isaac-hydraulic-actuator/models/actuator", sim_dt=0.01)
model.reset()
q = np.array([-0.5498, 1.2549, -0.7540, 0.0], dtype=np.float32)  # rad: boom, arm, bucket, pitch
v = np.zeros(4, dtype=np.float32)                              # rad/s
u = np.array([0.2, 0.0, 0.0], dtype=np.float32)                # valves: boom, arm, bucket
for _ in range(100):
    v = model.predict_velocity(q, v, u)
    q = q + model.dt * v
```

This minimal loop starts with zero command/velocity history and omits the demo's
joint-limit handling. Slew uses another `HydraulicActuatorNet` instance with
one position, one velocity and one valve channel.

## Dataset → actuator MLP → PPO

There is one local corpus: `dataset/complete_dataset`. It contains **33 unique
recordings**, 317.7 recorded minutes and **292.9 usable training minutes** in 168
contiguous 100 Hz chunks. September orientations are rebuilt with the robot's
current fusion settings (gain 0.5, rejection 5°, recovery 2 seconds); matching
October orientations are retained. Velocity targets remain unsmoothed raw gyro.
Historical recording labels live in provenance, not separate training pipelines.
See [the fusion audit](docs/offline_fusion_audit.md).

The corpus is public on Hugging Face as
[eetmie/excavator-hydraulic-actuator](https://huggingface.co/datasets/eetmie/excavator-hydraulic-actuator).
Training needs only the manifest and chunks (about 120 MB):

```bash
uvx --from huggingface_hub hf download eetmie/excavator-hydraulic-actuator --type dataset \
    --include dataset.json --include "training/*" --local-dir dataset/complete_dataset
```

To train overnight on another machine, see [overnight training](docs/overnight_training.md).

From the actuator project directory, use the Isaac Lab environment:

```bash
cd scripts/Isaac-hydraulic-actuator
uv run --project ../.. python -m training.prepare --out runs/actuator
uv run --project ../.. python -m training.actuator train --out runs/actuator --scratch
uv run --project ../.. python -m training.actuator evaluate --out runs/actuator --split development
uv run --project ../.. python train_controller.py --actuator-run runs/actuator --run_name controller --set gyro_observations=true
```

Preparation validates hashes and keeps each collection group in one split. The
held-out development group defaults to the September 9 collection.
Use `--dev-groups` and `--test-groups` to reserve other complete collection groups
before training; the default has no independent test set. Each run freezes its
split, so use a fresh output directory to change it.

MLP training predicts velocity increments from position, measured gyro history
and valve history. `--scratch` fits fresh normalization statistics and starts
from random weights; omit it to fine-tune `models/actuator`. Development rollouts,
closed-valve drift and stopping checks select between the candidate and the
baseline. `runs/actuator/selection.json` records the selected artifact, its
fingerprint and the dataset/split hashes. The historical baseline may have seen
these recordings, so its metrics are not an independent held-out result.

PPO's `--actuator-run` reads that selection and verifies the selected model's
fingerprint before training. Its saved contract records the actuator, current
USD, sensor layout and MLP dataset/split provenance. It uses the selected model,
which can be the baseline if the candidate fails acceptance. For an explicit
artifact instead, use `train_controller.py --model PATH`.
PPO checkpoints are saved under `logs/rsl_rl/hydraulic_controller/<time>_<run>/`.
Pass a resulting `model_<iteration>.pt` to `run_controller.py --checkpoint` for
playback. Training a new model does not overwrite the packaged baselines.

The corpus is already processed. To check or rerun fusion from canonical raw IMUs:

```bash
uv run --project ../.. python -m training.offline_fusion audit
uv run --project ../.. python -m training.offline_fusion replay INPUT_IMU.npz OUTPUT_IMU.npz
```

The tool needs a host C compiler. See the audit for input units and the limits of
replaying 200 Hz recordings from the 416 Hz firmware. Raw inputs, recording
provenance, corpus validation and checksums stay with the dataset. Dataset files,
run outputs and retired release archives are local and ignored by Git.

## Repository layout

For USD-matched deployment profiles and Jetson instructions, see
[the hardware transfer guide](docs/robot_deployment.md).

- `sim.py`, `sim_common.py`, `endstop_guard.py`: excavator demo and joint configuration.
- `actuators/`: reusable MLP inference and direct/target integration.
- `hydraulic_controller/`, `train_controller.py`, `run_controller.py`: learned valve controller training, benchmark and playback.
- `measure_speeds.py`, `replay_pid.py`, `tune_pid.py`: robot PID replay and tuning. `hydraulic_controller/tasks.py` and `closed_loop.py` hold the controller-neutral scenarios and loop; `pid.py`, `pid_tuning.py` and `speed_limits.py` the PID side; `mpc.py`, the `PolicyController` in `policy.py` and `comparison.py` / `compare_controllers.py` the controller comparison.
- `assets/`: self-contained bucket/gripper USDs; the actuator uses the pitch-capable `_rocking` variants.
- `models/`: the actuator, slew and controller baselines, each with metadata and hashes.
- `training/`: canonical dataset preparation, raw-gyro training/evaluation, native log ingestion, and audited offline fusion.
- `tests/`: actuator, dataset, controller, and deployment regression checks; `soil2d/tests/` stays with its owner.
- `media/`: the README rollout figure.

Run regression tests from this project directory with the Isaac Lab environment:

```bash
uv run --project ../.. --with pytest python -m pytest tests -q
```

Development formatting and checks use `pre-commit run --all-files` from this
repository's root. License: [BSD-3-Clause](LICENSE).
