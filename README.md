# Learned hydraulic actuators for Isaac Lab

A small excavator demo driven by learned hydraulic dynamics: **V5 for boom, arm,
bucket and carriage pitch**, plus an independent **slew MLP**. Both predict velocity
increments at 100 Hz; the simulator integrates the resulting velocities into joint
positions.

Idea from Egli, P. and Hutter, M. (2020) 'Towards RL-Based Hydraulic Excavator Automation'. Great paper!

## Run the demo

Tested with **Isaac Lab 3.0 / Isaac Sim 6.0.1**. Clone this repository into
`IsaacLab/scripts/Isaac-hydraulic-actuator`, activate your Isaac environment, and
run this command **from the Isaac Lab root**:

```bat
isaaclab.bat -p scripts/isaac-hydraulic-actuator/sim.py --model scripts\Isaac-hydraulic-actuator\models\arm_v5 --slew-model scripts\Isaac-hydraulic-actuator\models\slew
```

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
The frozen V5 steady actuator network is the plant, so training runs as batched
Torch on the GPU with no simulator (~75k steps/s with 4096 environments; a
1,500-iteration run takes about half an hour). From the Isaac Lab root:

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\train_controller.py --run_name my_run
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

The prototype controller ships in `models/controller_proto` and is the default,
so these commands work on a fresh clone without training. They need RSL-RL,
which `isaaclab.bat --install` includes. From the Isaac Lab root:

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\run_controller.py --mode draw
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\run_controller.py --mode circle --speed-mm-s 40
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\run_controller.py --mode gamepad --speed-mm-s 60
```

Playback cruises at 30 mm/s unless `--speed-mm-s` is given. Pass
`--checkpoint path\to\model_N.pt` to run your own training run instead; the
console reports the benchmark's recommended speed when a benchmark matches the
checkpoint.

The benchmark runs held commands with reversals and stops, plus quintic-timed
circles and lines, under nominal and perturbed valves. Results go to CSV/JSON
and a figure under `runs/`:

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\run_controller.py --mode benchmark --viz none
```

Draw mode holds the starting bucket angle, so only part of the drawing box is
reachable. The window shades the rest dark and turns a stroke red when it enters
that area. A rejected stroke keeps its reason on screen and in the console. Draw
and circle modes also show the planned path and the executed trace in the 3D
viewport.

Gamepad mode maps the left stick to tip X/Z velocity, right stick X to bucket
pitch rate, and B to reset. Everything here is free-space motion in simulation;
digging, contact and the real machine are not validated.

## Tune the robot's joint PID

The same plant can tune the classic controller on the real machine: the
boom/arm/bucket PIDs of [kaivuriprokkis](https://github.com/eetmie/kaivuriprokkis)
(`modules/pid.py`, ported 1:1 to batched Torch). All three tools read the gains
from a sibling `kaivuriprokkis` checkout (`--robot_repo`) and need no simulator:

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\measure_speeds.py   # what the plant can reach at all
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\replay_pid.py       # the robot's current gains
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\tune_pid.py         # CMA-ES, writes a pid: YAML block
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

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\compare_controllers.py --pid_gains runs\pid_tune\<run>\pid_gains.yaml
```

Runs the robot's PID, the tuned PID, the PPO valve policy and a sampling MPC
(MPPI on the actuator network, deadband-compensated, 20 Hz) on the same
feasible tip lines under every valve/speed perturbation. Each controller sees
only measured angles, rates and its own past commands, as on the robot. In
simulation the MLP and the MPC have an advantage the PID lacks: they were trained
on, or plan with, this very network.


## Bucket circle on the real robot

`run_robot_circle.py` runs the same 100 mm diameter, 20 mm/s X/Z circle with
the learned controller or the simulation-tuned DLS/joint PID. Both feed the
robot's existing 100 Hz direct-command output thread; the PID uses the same
calculation as `tune_pid.py`, bypassing the production pose smoother. Slew,
tracks and auxiliary outputs stay neutral. No shadow run is required.

Use a separate `jetson_bucket` profile. Its origin is the slew bearing,
boom-pivot height is 78.5 mm (confirmed by production CAD), and link vectors
and the bucket cutting-tip offset come from the training USD. FK consumes
relative joint angles, not authored USD body rotations. The blade's fixed
orientation offset is handled by the runner; legacy robot FK reports the
final joint-frame orientation. The arm IMU correction is +0.612 degrees;
boom remains +13.850 degrees with the CAD magnitude 13.832 recorded pending
sign verification. The frozen actor sees its original training joint zeros
through an explicit calibration transform; physical FK and PID use the new
mounting correction.

Prepare on the development PC using Isaac Lab Python and a current copy of
the robot's Jetson profile (the output directory must be new):

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\run_robot_circle.py prepare --bundle scripts\Isaac-hydraulic-actuator\runs\gyro_transfer\deployment_bucket --source_profile scripts\Isaac-hydraulic-actuator\runs\hardware_circle\source_profile --out scripts\Isaac-hydraulic-actuator\runs\hardware_circle\bucket
```

Install `bucket/profile/jetson_bucket` under the robot repository's
`configuration_files/profiles/`. Copy this project's `run_robot_circle.py`,
`hydraulic_controller/`, `actuators/` and `bucket/bundle` to a sibling runtime
directory on the robot. The runner needs NumPy, PyTorch, PyYAML and the existing
robot dependencies; running it needs no USD, Isaac Sim or RSL-RL.

From the installed runtime directory on the robot:

```bash
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py check
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py run --controller pid_tuned --pid_gains runs/hardware_circle/pid_gains.yaml --log runs/circle_pid_ccw_01.csv
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py run --controller mlp --log runs/circle_mlp_ccw_01.csv
```

`check` is an offline geometry/actor check and opens no devices. `run` preflights
the complete circle from the measured pose with the pump off. Release then hold
**Left Bumper** to start; **B**, release or disconnect latches a stop. A one-second
neutral history warmup is retained for inference. Each default run contains
one lap, a one-second lead hold and two seconds of settling. Use `--direction cw`
for the opposite direction. Run each controller/direction three times, alternate
controller order and return to the same starting pose between runs.

Logs are exclusive CSV plus JSON files, with reference/measured tip positions,
joint angles/rates, requested/emitted valves, timing, tracking and radial errors,
and a completion/fault result. PID gains are loaded per run; `pid_robot` uses
the profile gains and `pid_tuned` requires an explicit gains file. Software stops
include stale sensors/controller, loop overruns, position/angle error, and
joint/collision margins. Use the physical emergency stop and free-space motion.
IMU-based FK metrics describe internal tracking; externally measured tip motion
is needed to establish physical accuracy. Hardware acceptance remains pending.

After recording runs, compare matching profiles/speeds without opening hardware:

```bash
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py compare --logs runs/circle_pid_ccw_01.csv runs/circle_mlp_ccw_01.csv --out runs/circle_comparison --plot
```

This writes a comparison table and overlays the circles and timed errors.
Stopped trials remain in the report. Plotting optionally needs matplotlib.

For repeated warm-up/testing passes, use `--continuous`. Release then press LB
once to start; it may then be released. **A stops motion and the pump**;
**B starts logging** for the rest of the session. Gamepad disconnect and the
existing fault limits still stop the robot. Passes reuse the original center,
including the lead/settling holds. Warm-up samples are discarded until B;
recorded samples stay in a compact preallocated RAM buffer. B starts a
five-minute recording (`--record_seconds 300`); the pump shuts off at the end,
then all passes are saved together with `pass_index`. A can stop and save early.
The JSON `passes` list marks partially recorded passes. Continuous sessions
end on A; their session result is separate from individual pass scores.

```bash
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py check
../kaivuriprokkis/.venv-lerobot/bin/python run_robot_circle.py run --continuous --log runs/circle_mlp_20hz_session_01.csv
```

The actor runs at its trained 20 Hz by default. `--policy_hz 100` is an
experiment with the same frozen actor: it sees fresh history at 100 Hz, while
the twist projection and governor stay at 20 Hz. On 2026-10-07 it chattered at
6–7 Hz on the machine (valve travel ~20/s against the PID's 1.7/s; see
`drive_logs/20261007_circle_mlp_vs_pid`). The history holds the controller's own
command for each interval, as in training, not the delayed emitted value.
The run loop writes the valves in the tick it computes them, as `simple_drive.py`
did when the actuator data was recorded (`--valve_writes loop`, default). The
2026-10-07 sessions went through the robot controller's direct-command thread,
about 20 ms later; `--valve_writes thread` keeps that path for comparison. If
the loop stalls, the 150 ms PWM watchdog and the gate's policy timeout still
stop the valves. For continuous PID use
`--continuous --controller pid_tuned --pid_gains runs/hardware_circle/pid_gains.yaml`.
Passive carriage rocking allows +/-3 degrees (`--max_carriage_pitch_deg`),
matching the model geometry. The configurable driven-joint velocity guard
(`--max_joint_velocity_rad_s`, default 2) applies only to boom/arm/bucket.
Passive carriage rate is logged but does not trip that driven-joint limit.
To approach the same start as an earlier successful trial, add
`--start_from_log runs/hardware_circle/circle_start_pose.csv` and a `--pid_gains`
file. LB authorizes a quintic joint approach at <=0.05 rad/s reference speed,
with valves capped to +/-0.25 by default (`--approach_output_limit 0.5` allows
the stronger tested approach), followed by the selected circle controller.
`--joint_margin_deg 0` removes the extra software margin within the pinned
joint bounds; it does not expand those bounds. The approach and circle are
checked independently before the pump starts.
Recorded CSVs include sensor-frame acceleration XYZ [g] and gyro XYZ [deg/s]
for every physical IMU (`imu0_...` through `imu3_...`), plus the role-corrected
quaternions and gyro XYZ. These share the exact packet used by the controller;
firmware startup gyro-bias removal precedes this capture. The JSON records the
sensor-to-role mapping. CSV encoding, disk writes and per-pass scoring occur
only after pump/output shutdown. No saving occurs during circle transitions.
For timing diagnostics, `--allow_timing_overruns` records and reschedules missed
loop deadlines instead of aborting on 30 ms lateness / 40 ms computation.
The independent sensor/controller freshness gate and operator stops remain
active. The requested actor rate and measured timing are stored in the log;
use actual timestamps when evaluating the high-rate experiment.

## Example rollouts

![V5 free-running rollouts against recorded IMU angles](media/example_rollout.png)

Three fixed 10-second windows from the new held-out recording. Each rollout uses
measured initial history and recorded valve commands, then feeds back its own
predictions with **no later measured-state correction**. These are offline neural
rollouts; Isaac's direct integration was checked against the same calculation.
The middle window deliberately retains a substantial bucket tracking error.

Note that I'm using hobby grade, uncalibrated ISM330 IMU's. Industry grade 3D systems (Leica, Novatron, and such) will prob lead to _much_ better results.

## Included models

| Model directory | Motion outputs | Hidden layers | Parameters |
| --- | --- | --- | ---: |
| `models/arm_v5` | Boom, arm, bucket, carriage pitch | 512 / 512 / 384, ReLU | 579,972 |
| `models/arm_v4` | Boom, arm, bucket, carriage pitch | 512 / 512 / 384, ReLU | 579,972 |
| `models/slew` | Slew | 32 / 32, tanh | 2,145 |


| Held-out endpoint MAE | 10 seconds | 30 seconds |
| --- | ---: | ---: |
| V5 arm, new IMUs | 2.73° | 5.27° |
| V5 arm, older IMUs corrected offline | 3.11° | 5.70° |
| V4 arm, new IMUs | 3.49° | 6.78° |
| V4 arm, older IMUs corrected offline | 3.71° | 6.17° |
| Slew (gyro only, tiny dataset!) | 9.23° | 21.11° |

Arm errors average the three hydraulic joint endpoints across disjoint windows;
pitch is excluded from that metric. Each arm row uses one held-out recording.
Slew uses a separate recording containing partial turns, so full-turn hardware
accuracy is still unmeasured. These errors are relative to IMU measurements,
not external ground truth. Bucket drift and imperfect pitch oscillations remain.

Each model directory contains its weights, normalization arrays, channel/architecture
metadata, and a release manifest with SHA-256 hashes. No dataset download or
training step is needed to run the demo.

## How integration works

At each 0.01 s step, the network predicts a velocity increment:

```text
velocity_next = velocity + model(position, velocity_history, valve_history)
position_next = position + 0.01 * velocity_next
```

V4 and V5 select `direct` integration automatically. The learned joints are written
directly into simulation; their physical drives and carriage springs are disabled
so they do not fight the predicted motion. This demo models free motion and does
not provide a validated contact or digging-force response. Arm limits still apply,
and a held valve at an end stop cannot induce a learned rebound.

The assets include the tested target-drive gains: arm 2400 N·m/rad and
120 N·m·s/rad; slew 600 N·m/rad and 40 N·m·s/rad. The alternative `target`
integration route remains available for compatible three-joint models; V4 and V5 require
`direct`.

For direct Python use, only NumPy and PyTorch are needed. From the Isaac Lab root:

```python
import sys
import numpy as np

sys.path.insert(0, "scripts/Isaac-hydraulic-actuator")
from actuators import HydraulicActuatorNet

model = HydraulicActuatorNet("scripts/Isaac-hydraulic-actuator/models/arm_v5", sim_dt=0.01)
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

## Repository layout

For the gyro-history controller, USD-matched bucket deployment profile, fixed-tip
rotation results and Jetson shadow/robot instructions, see
[the hardware transfer guide](docs/gyro_transfer.md). Simulation acceptance is
complete; physical robot acceptance is still pending.

- `sim.py`, `sim_common.py`, `endstop_guard.py`: excavator demo and joint configuration.
- `actuators/`: reusable MLP inference and direct/target integration.
- `hydraulic_controller/`, `train_controller.py`, `run_controller.py`: learned valve controller training, benchmark and playback.
- `measure_speeds.py`, `replay_pid.py`, `tune_pid.py`: robot PID replay and tuning. `hydraulic_controller/tasks.py` and `closed_loop.py` hold the controller-neutral scenarios and loop; `pid.py`, `pid_tuning.py` and `speed_limits.py` the PID side; `mpc.py`, the `PolicyController` in `policy.py` and `comparison.py` / `compare_controllers.py` the controller comparison.
- `assets/`: self-contained bucket/gripper USDs; V4 and V5 select the pitch-capable `_rocking` variants.
- `models/`: the selected V4, V5 and slew releases, plus `arm_v5_steady`, the controller's plant, and `controller_proto`, the prototype controller.
- `training/`: reusable CSV/LeRobot training, evaluation, and regression checks.
- `media/`: the README rollout figure.

The generic trainer is a baseline training tool, not the
complete V4/V5 fine-tuning recipe. To inspect its options or check the release:

```powershell
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\training\train.py --help
.\isaaclab.bat -p scripts\Isaac-hydraulic-actuator\training\eval.py --help
.\isaaclab.bat -p -m pytest scripts\Isaac-hydraulic-actuator\training -q
```

Development formatting and checks use `pre-commit run --all-files` from this
repository's root. License: [BSD-3-Clause](LICENSE).
