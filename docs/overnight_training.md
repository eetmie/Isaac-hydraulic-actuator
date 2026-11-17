# Overnight training on a Windows laptop

The launchers use `uv run --project <IsaacLabRoot> python`; the actuator repository
can live anywhere. It trains on the GPU without launching Isaac Sim. Keep the laptop on
mains power. The runner prevents Windows from sleeping while it is alive and
releases that request when it exits; it does not change the system power plan.

## Setup

Set `-IsaacLabRoot` if Isaac Lab is not at `$HOME/IsaacLab`. Download the manifest
and the 100 Hz chunks (about 120 MB) from the actuator repository directory:

```powershell
uvx --from huggingface_hub hf download eetmie/excavator-hydraulic-actuator --type dataset `
    --include dataset.json --include "training/*" --local-dir dataset/complete_dataset
```

`kaivuriprokkis` must be beside this checkout for the final robot-PID comparisons.
The Python environment needs the normal training dependencies (Torch, NumPy,
RSL-RL, TensorDict, PyYAML, TensorBoard, filelock, SciPy, matplotlib and OpenUSD).
uv handles environment synchronization before starting each launcher stage. Once
running, the Python orchestrator uses that interpreter for its child processes.

## Benchmark first

The November 11 laptop measurements, on an 8 GB RTX 2000 Ada, were:

| Work | Measured time |
| --- | ---: |
| Actuator: 500 updates, batch 256, horizon 100, two 96-case evaluations | 87.8 s |
| PPO: 4096 environments, deployed plant | about 0.9 s/iteration |
| PPO: 4096 environments, two plants, lookahead and acceleration | about 1.0 s/iteration |

Use two CPU threads (`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`). Allow for sustained
load and evaluation overhead; the launcher defaults to 0.18 s/actuator update and
1.05 s/PPO iteration. These are estimates; actual phase lengths are controlled by
wall time and checkpoint saves. Benchmark logs are under `runs/overnight_benchmark`.

## Run

From this repository, first exercise the complete chain with tiny budgets:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs/overnight_training.ps1 -Smoke
```

Then run the initial experiments in about 12 hours, with an optional two-hour
second-seed follow-on:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs/overnight_training.ps1 `
    -Hours 12 -ExtraSeedHours 2 -DatasetRevision ab28bc223f19ff2e536ebdd93ca4f02fcf2a8a52
```

Other options: `-RunRoot runs/my_run`, `-IsaacLabRoot C:/path/IsaacLab`,
`-ActuatorStepSeconds 0.18`, `-PpoIterationSeconds 1.05`, and `-PlanOnly`.
The entry point, from this repository in PowerShell, is
`uv run --project "$HOME/IsaacLab" python -m training.overnight --help`.

For long runs, use `-RunRoot C:/Users/<user>/IsaacTraining/runs/<name>` outside
OneDrive. The first night's queue hit a Windows sharing violation while replacing
its monitoring JSON. Monitoring writes now retry and remain best effort, so a
locked status file cannot stop training. Important plan and completion records
still report persistent write failures.

The initial budget assigns 65% to actuator training, 30% to three PPO variations,
and 5% to evaluation and timing overhead. Time limits save checkpoints and finish
normally. PPO checks the clock when saving, every 100 iterations, so a phase can
run roughly another checkpoint interval. The time budget is a target, not a hard
process termination.

1. Verify and freeze collection-group splits. Dev is `v2_20260909` and robot
   `...-71723`; the four October 22 robot groups are test only.
2. Actuator A starts from scratch with 20k one-step pretraining updates, then
   rollout training at lr 1e-4. It receives 45% of the actuator budget. B and C
   continue the previous phase's latest weights at 3e-5 and 1e-5, receiving 30%
   and 25%. The measured 12-hour plan estimates about 156k rollout updates total.
3. Compile closed-valve bounds into each phase's best checkpoint. Compare on dev,
   choose the best including the deployed model, then report on held-out test.
4. Train three PPO policies at 4096 environments each: baseline at fixed lr 3e-4;
   lookahead=4 plus acceleration reward=0.4 at fixed lr 3e-4; and the same at fixed
   lr 1e-4. Each is allocated about 4,100 iterations at the measured rate. All use
   both the new and deployed plants when the new plant wins dev selection.
5. Compare the three policies, packaged baseline and robot PID on lines and
   circles, with nominal and perturbed plants, on both actuator models.
6. If requested, repeat the lr 1e-4 lookahead variant with seed 43 for the same
   iteration count, within the extra time budget, then write final comparisons.
   Initial comparisons are available before this extra run.

## Monitoring and recovery

`runs/overnight/plan.json` freezes the experiment settings, source commit and
dataset manifest hash. `source.patch` and `overnight_source.py` preserve local
implementation changes. `schedule.json` records the timing targets, and
`status.json` reports the active child process, stage and log. For example:

```powershell
Get-Content runs/overnight/status.json
Get-Content runs/overnight/a_scratch.log -Tail 20 -Wait
```

Run the same command after a crash to reuse its frozen plan and schedule.
Completed stages are skipped. An interrupted actuator phase restarts from that
phase's original parent (A repeats scratch pretraining); an interrupted PPO phase
loads its last checkpoint and optimizer. Use a new `-RunRoot` for a new budget or
experiment; changing CLI settings does not overwrite an existing plan.

To relocate a stopped run, copy its finished actuator phase directories and
checkpoint archives to a fresh run root, then create a new plan there. The new
schedule excludes completed actuator phases and retains the original allocation
for each unfinished phase. Keep the original plan, status and logs as provenance;
do not reuse an expired schedule when allocating recovery time.

Outputs:

- `results.md`: tables of actuator and controller metrics, updated after each benchmark round.
- `compare_dev.log`, `compare_test.log`: actuator metrics, selected using dev only.
- `*/model/history.json`, `*/model/latest`: actuator dev curves, best and last weights.
- `controllers/<variation>/`: PPO checkpoints, optimizer state, contracts, settings,
  TensorBoard events and `training_result.json` with the final checkpoint path.
- `benchmarks/`: controller comparison CSVs, JSON metrics and figures. Initial
  reports have `initial_` in their names; reports including the extra seed use `final_`.
- `completed/`: successful stage records, including commands and wall times.

The packaged controller pins an earlier USD asset. The runner recovers that exact
asset from Git and stages a reference bundle under `reference_controller`, keeping
the original weights and asset hash. Its relocated contract and provenance record
the historical training asset. Benchmarks evaluate that historical policy as a
transfer reference on the current geometry, alongside the newly trained policies.

The run leaves packaged models unchanged. Multi-day continuation can use a prior
actuator's `model/latest` with `training.actuator --model`, or a PPO checkpoint
with `train_controller.py --resume`, in a fresh output directory. Compare the first
night's curves and cross-plant benchmarks before committing days to a recipe.

## Optional Friday extension

The extension waits for the original queue to finish and then fine-tunes a matched
pair from the same `lookahead_accel_lr1e-4` weights and optimizer:

- Both use gyro observations with link noise 0.005 rad/s, bias up to 0.003 rad/s,
  angle noise 0.003 rad and sensor delay of 0-20 ms.
- One keeps stationary resets; the other resets 30% of episodes into states driven
  for 1.2 seconds by a position controller, preserving motion/command histories.
  Ordinary episodes already train starts, stops, reversals and moving command changes.

It benchmarks 100 PPO iterations per condition, then allocates the same number of
additional updates to both conditions using the measured costs. Learning rate is
fixed at 1e-4. Checkpoints and optimizer state are retained every 100 iterations.
The original queue and its budget remain unchanged.

```powershell
uv run --project "$HOME/IsaacLab" python -u -m training.friday_extension `
    --base C:/Users/<user>/IsaacTraining/runs/overnight_20261112 `
    --finish_at 2026-11-13T10:00:00+02:00
```

Use `--smoke` with a completed smoke run to exercise checkpoint loading, both
conditions, and the gyro benchmark. Outputs live in `<base>/friday_extension`.
The deadline is frozen in its own plan. Training ends 90 minutes before the target
by default, reserving time for lines/circles on both actuator models under all
valve/speed perturbations and three measurement profiles:

- `standard`: the original comparison, with 0.02-degree arm-angle noise.
- `gyro`: the training-level gyro, angle, bias and 0-20 ms delay profile above.
- `stress`: double those noise/bias levels and up to 50 ms delay.

Each policy and the robot PID see matching random sensor disturbances by scenario
and tick. Scores use the physical plant state; adding sensor error does not change
the measured ground truth. These profiles are also available through
`compare_controllers.py --sensor_profile standard|gyro|stress`.

## Weekend rotation comparison (13-16 November)

The active isolated run root is `C:/Users/sh23937/IsaacTraining/runs/weekend_20261113`.
`quick/` runs four equal 1500-update conditions from scratch: original tip reward,
4x angular reward, 4x angular reward plus a 0.5 linear angular-error penalty, and
pivot tracking with the original angular reward. Every controller still accepts EE
pose/twist; rigid-body twist conversion happens inside pivot controllers.
These reward/penalty terms concern angular-velocity tracking in the inner policy;
the outer pose controller corrects angle, and benchmarks score actual bucket-angle error.

`study/` waits for that queue and benchmarks four fresh 100-update pilots before
allocating a larger matched study ending training at Monday 03:00 Helsinki time.
Six hours are reserved for evaluation, targeting Monday 16 November at 09:00.
The time is a planning target; an unexpectedly slow evaluation can finish later.

There are 21 independent runs at fixed learning rate 3e-4 and 4096 environments:

- Four reward/geometry conditions across seeds 42, 43 and 44 (12 runs, 60% of training time).
- Fixed-tip, no-lookahead candidates with the stronger reward and penalty (3 runs, 10%).
- Tip and pivot policies conditioned on a preference scalar, each across three seeds
  (6 runs, 30%). Evaluate the slider at 0, 0.25, 0.5, 0.75 and 1.

All weekend policies use noisy gyro observations and 30% moving resets. Updates are
matched within each group; if a time cap truncates a run, evaluation uses a common
saved iteration across its group. Do not compare group learning efficiency without
accounting for their different update budgets. Pilot weights are discarded.
Checkpoints, optimizer states and evaluation traces are retained. Relaunching the
same driver reuses the plan, skips completed stages and resumes saved PPO states.

Preference 0 removes the angular tracking reward/error penalty; preference 1 uses
angular reward 2.0 and angular-error penalty 0.5. Position reward and common costs
remain active throughout. This is relative orientation priority, not pure rotation
at one endpoint. It uses the existing preference-conditioning path.

Evaluate both actuator plants, lines/circles, common perturbations and three sensor
profiles. Standard/gyro sweep all five preferences; stress uses the midpoint.
`study/results.md` reports position, orientation, valve travel and failures with
training-seed variation. `preference_tradeoff.png` plots position/angle tradeoffs.
Fixed-tip rotation adds +/-5 and +/-10 degree sweeps, ideal/gyro observations and
plant perturbations; worst errors across cases are retained before seed aggregation.

The Monday deployment target is **native_stack**. Its current GR00T/OpenPI loaders
do not implement this PPO contract. `native_handoff/` pins the latest fetched native
source and lists the sensor/frame/action mapping and remaining adapter/parity work.
The three fixed-tip bundles are also checked against a frozen legacy Python runtime
with a generated USD profile; that check does not validate native integration or
the actual robot calibration. No policy is automatically deployed or promoted.

```powershell
uv run --project "$HOME/IsaacLab" python -u -m training.weekend_comparison `
    --base C:/Users/sh23937/IsaacTraining/runs/weekend_20261113/weekend_frozen/base `
    --quick C:/Users/sh23937/IsaacTraining/runs/weekend_20261113/quick `
    --out C:/Users/sh23937/IsaacTraining/runs/weekend_20261113/study `
    --finish_at 2026-11-16T09:00:00+02:00
```

Run that command from the `weekend_frozen/Isaac-hydraulic-actuator` snapshot.
Read `study/status.json` and the named log for the current stage. The runner holds
a Windows system-awake request while active and runs outside OneDrive.

After a reboot or transient CUDA failure, resume **both** queues from this checkout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File docs/resume_weekend_training.ps1
```

This launcher checks CUDA, finishes the quick queue using its frozen snapshot,
then resumes the study using its separate snapshot and saved deadline. It skips
completed stages and retains checkpoints and logs. `-PlanOnly` checks the paths
and prints the queues without starting work; `-RunRoot` and `-IsaacLabRoot`
override the defaults. The study-only command above requires a completed quick queue.

The weekend line evaluations additionally use `--speed_sweep`: 5, 10, 20, 40,
70, 100 and 120 mm/s, completing every 100 mm path. They write `speed_audit.csv`
with motion-only and cruise-only position, angle, velocity and valve metrics.
`speed_analysis/results.md` separates all cases from model-estimated feasible
cases and aggregates training seeds. Headline summaries still include endpoint
holds; use the motion-window report to compare performance between speeds.
Circle/rotation speed settings are unchanged. The configured training target
magnitudes are uniform from 10-120 mm/s with 10% zero, before slew/governor;
actual admitted commands need not have that uniform distribution.
