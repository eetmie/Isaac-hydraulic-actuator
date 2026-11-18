# Controller playback and robot deployment

New policies use internal bucket-pivot tracking. The default demo loads
`models/controller_pivot/model_3499.pt` and its packaged actuator from
`models/actuator_pivot`. External requests and reported path errors remain at
the bucket cutting edge. The conversion is applied before the pivot governor;
observations and rewards then use the pivot's pose and arm-driven twist, with
the bucket's absolute angle unchanged.

Use **20 Hz training / 20 Hz policy execution**, with hydraulic and sensor
histories at 100 Hz; see the [rate-study decision](policy_rate_comparison.md).

The native robot stack runs these policies with `kaivuriprokkis/native_stack/tools/ppo_live.py`
(circle and gamepad modes through `kaivuri-control`). Export the bundle it loads, from the Isaac Lab
root:

```bash
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/export_robot_bundle.py native \
    --out ../kaivuriprokkis/native_stack/ppo_bundles/controller_pivot
```

The bundle holds the actor weights, the kinematic chain, the collision grid and reference cases
computed with the training code (edge-to-pivot conversion, governor, observation with lookahead,
history rollover, actor). The robot's NumPy runtime replays them at every load and refuses a
mismatch. No generated profile is needed: the training USD is retargeted to the robot profile and
angles use the fixed sensor contract. The weekend default first ran with powered hydraulics on
2026-11-18: a 50 mm circle at 20 mm/s tracked with 2.5 mm RMS error (max 7.1 mm), and gamepad edge
control worked. Speed surges at about 1 Hz on the machine and, about half as large, in simulation. The legacy bundle instructions
below apply to tip policies on the frozen Python stack.

Train an actuator with `training.prepare` and `training.actuator`, then pass its
run directory to `train_controller.py --actuator-run`. The saved PPO checkpoint
contains a contract for the selected actuator, USD geometry, observation layout
and timing. The loader checks the actuator and USD hashes before inference.
Use a checkpoint from the new run when either artifact has changed; the packaged
historical controller baseline retains its original training contract.
Resuming training preserves the saved checkpoint's tracking point, including
old contracts that omit the field and mean tip tracking. A conflicting
`--set tracked_point=...` is rejected before writing training outputs.

From the Isaac Lab root, play back a checkpoint:

```bash
uv run --extra isaacsim python scripts/Isaac-hydraulic-actuator/run_controller.py \
    --checkpoint scripts/Isaac-hydraulic-actuator/logs/rsl_rl/hydraulic_controller/<run>/model_<iteration>.pt \
    --mode draw
```

For a checkout outside Isaac Lab, run from the actuator repository in PowerShell:

```powershell
uv run --project "$HOME/IsaacLab" --extra isaacsim python .\run_controller.py --mode draw --viz kit
```

This selects Isaac Lab's managed environment while retaining the actuator
repository as the working directory. Replace the project path if Isaac Lab is
installed elsewhere.

The command frame is lower-carriage XZ with pitch about positive Y. Joint order
is boom, arm, bucket, carriage pitch, in radians and radians/second. The measured
joint rates are `[boomY-baseY, armY-boomY, bucketY-armY, baseY]`, converted from
degrees/second. Mounting correction is applied once; gyro history has no second
bias estimator or noncausal software filter.

The robot uses `kaivuriprokkis/learned_control`, without a copy of this project.
Generate a profile using the controller's USD geometry and the robot's calibrated
IMU/PWM configuration, then export the checkpoint. From this project directory:

```bash
uv run --project ../.. python export_robot_bundle.py profile \
    --source_profile /path/to/kaivuriprokkis/configuration_files/profiles/jetson \
    --robot jetson_bucket --out /path/to/kaivuriprokkis/configuration_files/profiles/jetson_bucket
uv run --project ../.. python export_robot_bundle.py export \
    --checkpoint logs/rsl_rl/hydraulic_controller/<run>/model_<iteration>.pt \
    --robot_repo /path/to/kaivuriprokkis --robot jetson_bucket --out runs/robot_bundle
```

The generated profile sits in the robot configuration location expected by
`learned_control`. The exported bundle includes the actor,
geometry, collision grid, sensor contract and numerical reference cases. Refer to
the robot project's runner instructions for its profile and bundle paths.

The training geometry uses the bucket cutting tip. Do not mix it with the old
robot tool offset or absolute coordinates from a different origin. Export checks
forward kinematics, Jacobians and tool angle against the USD geometry. Kept tests
cover measurements, actuator histories, geometry export and robot parity; the
robot-parity tests require a sibling `kaivuriprokkis` checkout.

Hardware validation starts with neutral valves and shadow mode, followed by
small, externally measured movements. Simulated tracking metrics alone do not
establish real-machine accuracy or hydraulic load-holding behavior.
