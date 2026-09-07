# Isaac Hydraulic Actuator

This is a data-driven forward model for the three hydraulic arm joints (boom, arm, bucket) of an excavator, hooked up as a demo to Isaac Lab with gamepad teleop.

The idea is pretty straightforward: train a small MLP on real drive logs to predict what joint velocity a given valve command will actually produce. Feed that predicted velocity into the sim either via PhysX joint control or directly, and the sim starts behaving like the real machine, hydraulic lag and all that goodness. Joint positions are just integrated forward each step from the predicted velocity.

Thanks to Egli & Hutter (IROS 2020). Their paper on RL-based hydraulic excavator automation is what gave me the idea for this approach.

An excavator boom / arm / bucket is included as a worked example, with a trained
model you can run immediately. Current model is pretty good except the initial jerks with large commands. This is the datasets fault, not the models (excvacator hull moved, its fixed in the sim!).

Targets **Isaac Lab 3.0 / Isaac Sim 6.0.1**.

```text
sim.py          the demo: joint names, gamepad map, tool assets, model wiring
sim_common.py   the demo's robot config: joint groups, drive gains, asset paths
actuators/      reusable and robot-agnostic; knows nothing about excavators
models/arm/     shipped three-joint arm model
models/slew/    cabin slew model -- not trained yet, sim.py falls back
training/       train.py, eval.py, dataset.py, rollout.py, splits.py,
                lerobot_source.py, test_contract.py
assets/         excavator_bucket.usd, excavator_gripper.usd
```

`actuators/` and `training/` are the reusable halves and the excavator is the
worked example. The two trees never import each other — that is what makes the
parity test in `training/test_contract.py` mean something. Only `sim.py` and
`actuators/` need Isaac Lab; `training/` is plain numpy / torch / pandas.

![Command replay through the simulator](docs/example_rollout.png)

Ten seconds of recorded valve commands replayed through the sim. Blue is the real
machine, crimson is `direct` (model drives the joints), yellow is `target` (PhysX
PD chasing the model's setpoint). The dashed line is the model on its own with no
sim at all, and it lands right on top of crimson, which is how you know the sim
loop isn't adding anything of its own.

## How it works

At 100 Hz one MLP receives current joint position, 0.1 s of joint-velocity
history and 0.99 s of valve-command history, and predicts the velocity change:

```text
delta_qdot = qdot(t + 0.01) - qdot(t)
qdot_next  = qdot(t) + delta_qdot
q_next     = q(t) + dt * qdot_next
```

Position is integrated outside the network. One network models all three joints
together so it can learn the coupling between them. On a real machine the axes
share a pump, and moving two at once slows both.

## Quick start

Requires Isaac Lab 3.0 (Isaac Sim 6.0.1). From your Isaac Lab installation:

```bash
isaaclab.bat -p /path/to/Isaac-hydraulic-actuator/sim.py
```

Gamepad: right stick Y lifts the boom, left stick Y the arm, right stick X the
bucket, left stick X slews the cabin. `A` toggles between the learned model and
direct velocity control, `B` resets.

### Two joint groups, one model each

The arm (`[lift, tilt, tool]`) and the cabin slew (`[carriage]`) are separate
learned models, both instances of the same `HydraulicActuatorNet` — the joint
count comes from the model's own metadata, so one class serves both.

| group | model dir | flag |
|---|---|---|
| arm | `models/arm` (ships with the repo) | `--model` |
| slew | `models/slew` (**not trained yet**) | `--slew-model` |

When `models/slew` is absent, `sim.py` says so and drives the slew joint straight
from the stick, exactly as it always has. That fallback is decided in `sim.py`
and nowhere else: `HydraulicActuatorNet` has no degraded mode, and nothing under
`actuators/` knows a model might be missing. Drop a trained model into
`models/slew` and the slew joint starts using it with no code change.

With a slew model present the slew joint follows the same `--integration` route
as the arm. Without one it stays on the PD-target route whatever `--integration`
says, because there is no learned state to make authoritative.

Isaac Lab 3.0 runs headless unless a visualizer is asked for, and the gamepad only
works when the Omniverse app window is up, so `sim.py` requests `--viz kit` on your
behalf when you do not pass `--viz` yourself. Pass `--viz none` (or the deprecated
`--headless`) for a windowless run: the sim then free-runs on zero commands, which
is what you want for a smoke test.

### End effectors

Two assets ship in `assets/`, selected with `--tool`:

| `--tool` | asset | extra DOFs |
|---|---|---|
| `bucket` (default) | `assets/excavator_bucket.usd` | none, the bucket is welded to `tool_body` |
| `gripper` | `assets/excavator_gripper.usd` | `revolute_gripper`, `revolute_claw_1`, `revolute_claw_2` |

Both share the same arm and carriage joints, so the learned model applies unchanged;
the gripper DOFs are left on their PD drives and are not teleoperated.

## The two integration routes

`--integration`
selects which actuator drives the joints, and they are genuinely different
physics:

**`direct`** — [`DirectIntegrationActuator`](actuators/direct_integration_actuator.py).
The learned velocity is integrated into joint state and written straight to the
simulator. The model *is* the dynamics; PhysX does not get a say. Use it when you
want the learned hydraulics to be ground truth. These joints will not react to
contact.

**`target`** — [`VelocityIntegratedActuator`](actuators/velocity_integrated_actuator.py).
The same velocity is integrated into a *position target* and the articulation's
PD drive chases it. Contact and load response are preserved (but not tested, can be awful), at the cost of a
second dynamic system the real machine does not have, with its own lag.

> **If you use `direct`, you must zero the drive gains for those joints.**

## Training on your own machine
It's pretty easy to convert this to accept basically any kind of dataset, I had multiple 10min runs available.

```bash
python training/train.py --csv /path/to/your/logs --out runs/my_run --device cuda
```

Two input formats. `--csv` reads flat columns; `--lerobot` reads a LeRobot v3.0
dataset root. They share everything after loading, so the flags below apply to
both.

### CSV

CSV columns, one row per 100 Hz sample. These are the *defaults*; pass your own
with `--q-cols` / `--qdot-cols` / `--u-cols` rather than renaming your data:

| column | unit |
|---|---|
| `timestamp` | seconds |
| `joint_pos_boom`, `joint_pos_arm`, `joint_pos_bucket` | rad |
| `joint_vel_boom`, `joint_vel_arm`, `joint_vel_bucket` | rad/s |
| `combined_cmd_lift`, `combined_cmd_tilt`, `combined_cmd_scoop` | normalized [-1, 1] |

Optional `sample_idx` and `cmd_stale` columns are used to split the timeline
where the log is discontinuous or the command was not fresh.

### LeRobot datasets

> **Not tested against a real dataset yet.** The reader was built against the
> published v3.0 schema and verified end to end on a generated tree plus a real
> dataset's `meta/info.json`, but no recorded LeRobot data has been through it.
> Expect to fix something the first time you point it at one.

```bash
python training/train.py --lerobot /path/to/dataset --out runs/my_run \
    --q-cols pos_lift pos_tilt pos_scoop \
    --qdot-cols vel_lift vel_tilt vel_scoop \
    --u-cols cmd_lift cmd_tilt cmd_scoop
```

Point `--lerobot` at the dataset root — the directory holding `meta/` and
`data/`. LeRobot stores a whole vector per column (`observation.state` is one
`fixed_size_list<float>[N]`, not N columns) and names the elements in
`meta/info.json`, so the same `--*-cols` flags name *channels* here and the
reader finds which vector and which slot each one lives in. An action vector
that is wider than the joint count, or in a different order, needs no special
handling — `--u-cols cmd_lift cmd_tilt cmd_scoop` picks those three slots and
ignores the rest.

Three spellings, in order of preference:

| written as | means |
|---|---|
| `cmd_tilt` | that element name, searched across every numeric feature |
| `action/cmd_tilt` | qualified, when the bare name appears in two features |
| `action[2]` | positional, for a feature that names nothing |

`--dt` defaults to `1/fps` from `meta/info.json` rather than the CSV default of
0.01, and the reader refuses a dataset whose rate disagrees with the spec — the
history taps assume samples are `dt` apart. Episodes are the chunk boundary, so
no history window ever spans two runs, and each episode is its own session for
`--split session`.

Some things this deliberately does **not** do:

- **It will not differentiate position to get velocity.** The dataset must carry
  velocity channels; if the names in `--qdot-cols` are not there you get an error
  listing what the dataset does have. The training target is
  `qdot(t+dt) − qdot(t)`, so a differenced qdot would make the target a *second*
  difference of position — quantisation noise, and quietly wrong rather than
  loudly missing. Record velocity.
- **Only `codebase_version: "v3.0"`.** A v2.x tree raises and names its own
  version. Convert it with lerobot's migration tooling.
- **No dependency on the `lerobot` package.** The parquet is read directly with
  pyarrow, so `training/` stays installable next to Isaac Sim.

### Shaping the model

Nothing about the network is fixed. The column lists set the widths, so a
one-joint slew model is the same command with single-entry lists:

```bash
python training/train.py --csv slew_logs \
    --q-cols joint_pos_slew --qdot-cols joint_vel_slew --u-cols combined_cmd_slew \
    --hidden 64 64 --activation tanh --out runs/slew
```

| flag | default | what it controls |
|---|---|---|
| `--hidden` | `128 128 128` | hidden widths; how many values you pass is how many layers you get |
| `--activation` | `relu` | `relu` or `tanh` |
| `--q-cols`, `--qdot-cols` | boom/arm/bucket | the joints, and so the input and **output** width. Must be the same length — position is integrated from velocity |
| `--u-cols` | lift/tilt/scoop | the command channels. Counted separately from joints, so three joints driven by four valve channels is fine |
| `--dt` | `0.01` | control period |
| `--position-history-sec` | `0.0` | position history depth; 0 is the current position only |
| `--velocity-history-sec`, `--velocity-stride-sec` | `0.10`, `0.01` | velocity history depth and tap spacing |
| `--command-history-sec`, `--command-stride-sec` | `0.99`, `0.03` | command history depth and tap spacing |

Both activations are parameterless, so switching between them does not renumber
the `state_dict` keys the sim side loads with `strict=True`. All of this is
recorded in `model_meta.json`, and the sim reads its geometry from there — there
is no second place to update when you change shape.

### Checkpoint selection is on rollout error, not validation loss

At 100 Hz `qdot(t+dt) ≈ qdot(t)`, so a network can score well on one-step MSE by
leaning on its own past velocity, then drift the moment it eats its own
predictions. On this data the two genuinely pull against each other: over one
training run, the checkpoint with the best one-step R2 was 21% worse on
free-running position error than the one rollout selection picked.

So every `--rollout-every` epochs, [`rollout.py`](training/rollout.py) free-runs
trajectories inside the held-out recordings and scores mean absolute position
error. `mlp_state_dict.pt` is the best of those, not the last epoch; final
weights are kept alongside as `mlp_state_dict_final.pt`. `--select-on val`
restores one-step selection, mainly so you can reproduce that comparison
yourself.

### Splitting

`--split session` holds out whole driving sessions. Recordings are grouped by wall-clock continuity, not filename, because
loggers roll to a new file mid-drive and filename grouping would put ten minutes
of the same drive on both sides of the split.

`--split snippet` holds out contiguous snippets from every drive instead, with a
leakage buffer. Every session then contributes to training, which measured
stronger on a held-out benchmark here at the cost of a validation set that is not
independent at the session level.

`--split-seed` is separate from `--seed` so a weight-init sweep does not silently
re-split the data and refit the normalizers.

## Evaluating

```bash
python training/eval.py --model runs/my_run --csv "held_out/*.csv"
python training/eval.py --model runs/my_run --lerobot /path/to/held_out_dataset
```
Reports one-step R2, free-running velocity and position error at several
horizons, and drift at rest. Multiple files are pooled and summarized
duration-weighted. Use free-running position error as the primary number.

## Testing

```bash
python training/test_contract.py --model models/arm
```

[`actuators/hydraulic_actuator.py`](actuators/hydraulic_actuator.py) deliberately
does **not** import `dataset.py` — it cannot, they are separate trees now. It is
an independent second implementation of
the same feature layout, written against ring buffers for real-time single-step
inference. That independence is what gives the parity test something real to
check — if it shared the feature builder, the test would only prove numpy equals
numpy. Run it after touching the feature layout, the metadata keys, or the
architecture.

It needs no pandas, so it runs in the Isaac Sim environment as-is. Alongside the
parity tests it pins the width derivation, the meta round trip for one-joint and
wide-command shapes, and that swapping the activation leaves the `state_dict`
keys alone.

## Porting to your own robot

Honest about what this costs today:

- **The joint count is not hardcoded any more.** It comes from `--q-cols` /
  `--qdot-cols` at training time, lands in `model_meta.json`, and both the
  training side and `HydraulicActuatorNet` read it from there. Command channels
  are counted separately, so they need not match the joint count.
- **`sim.py` is an example, not a template.** Joint names, the scene, the fixed
  base, and the gamepad mapping are all specific to this excavator. The reusable
  part is `actuators/`.
- **Channel order is the model's, and `sim.py` owns the mapping onto it.** The
  arm model's order is `[lift, tilt, tool]`; the actuators resolve joints with
  `preserve_order=True` because a silent permutation would scramble the model,
  and `sim.py` checks the model's width against the group it is driving at
  startup rather than letting a mismatch show up as wrong motion.
- **Adding a joint group is adding a model directory.** A group is a joint name
  list in `sim_common.py`, a model directory, and the two lines in `sim.py` that
  build its actuator and controller. No new actuator class.
- **Control rate must match the model's `dt`.** History taps assume samples are
  `dt` apart, so running at a different rate stretches every horizon by that
  ratio. `HydraulicActuatorNet` raises rather than let this pass silently.
- **The actuators are written against the Isaac Lab 3.0 API.** They read state as
  `robot.data.<field>.torch` (3.0 returns `ProxyArray`, not `torch.Tensor`) and
  write through the `*_index` methods (`write_joint_position_to_sim_index`,
  `set_joint_position_target_index`, …). The 2.x spellings still work behind a
  deprecation shim scheduled for removal in Isaac Lab 4.0.
- **The base is welded in `sim.py`, not in the USD.** The assets put
  `PhysicsArticulationRootAPI` on the `/excavator` Xform rather than on a rigid
  body, which is exactly the case Isaac Lab's `fix_root_link` refuses, so
  `fix_carriage_to_world()` authors the world fixed joint per env instead.

## The included model

`models/arm`, trained on ~90 minutes of logged excavator driving. Default
configuration, best checkpoint at epoch 1470 of 1800, selected on a 5 s
free-running rollout error of 0.032 rad over held-out sessions.

| | |
|---|---|
| inputs | 138 — 3 positions, 11×3 velocity taps, 34×3 command taps |
| hidden | 3 × 128, ReLU |
| target | `delta_velocity` |
| rate | 100 Hz |

Not modelled: cabin slew (`models/slew` is empty — the joint runs on the raw
stick until you train one), oil temperature, engine RPM, and **end stops**. Joint
limits are enforced by the simulator. Pretty easy to update to match your robot though.

The current dataset wasn't the best possible as I was just testing the idea, so the model can still be improved a lot.

## Credits

The learned-forward-model approach, predicting a velocity delta from position,
velocity history and command history, then integrating outside the network, and
treating that integration as the joint dynamics, follows the hydraulic
excavator work of **Egli and Hutter** (Robotic Systems Lab, ETH Zürich).
This repository is an **independent implementation** written from the published
method. It is not affiliated with, derived from, or endorsed by those authors,
and it contains none of their code. Any errors here are mine.

Trained on data recorded from instrumented excavator at Savonia
University of Applied Sciences.
Also thanks Claude for the help.

## License

Apache License 2.0, see [LICENSE](LICENSE). You may use, modify and
redistribute this commercially, including in closed-source work, with no
obligation to contribute changes back.

The included model weights and the `assets/excavator_*.usd` files are released
under the same terms.
