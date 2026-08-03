# Hydraulic Actuator Net

WIP - Missing dataset, one will be added soon. Also the dataset format is tad borked, will update at some point....

This is a data-driven forward model for the three hydraulic arm joints (boom, arm, bucket) of excavator, hooked up to an Isaac Lab sim with gamepad teleop.

The idea is pretty straightforward: train a small MLP on real drive logs to predict what joint velocity a given valve command will actually produce. Feed that predicted velocity into the sim instead of letting PhysX handle it, and the sim starts behaving like the real machine, hydraulic lag and all that goodness. Joint positions are just integrated forward each step from the predicted velocity.

Thanks to Egli & Hutter (IROS 2020). Their paper on RL-based hydraulic excavator automation is what gave me the idea for this approach.

---

## Structure

```
scripts/nn/
├── sim.py                          # gamepad teleop in Isaac Lab
├── train.py                        # train the forward model on your drive logs
├── actuators/
│   ├── hydraulic_actuator.py       # loads the model, runs inference, maintains history buffers
│   └── velocity_integrated_actuator.py  # integrates predicted qdot into position targets
├── model/                          # trained model files, loaded by sim.py at startup
└── assets/
    └── excavator.usd               # robot asset (1:14 scale miniature excavator)
```


---

## Data format

| Column | Type | Notes |
|---|---|---|
| `timestamp` | float, seconds | monotonically increasing |
| `joint_pos_boom` | float, radians | |
| `joint_pos_arm` | float, radians | |
| `joint_pos_bucket` | float, radians | |
| `combined_cmd_lift` | float, [-1, 1] | valve command sent to hardware |
| `combined_cmd_tilt` | float, [-1, 1] | |
| `combined_cmd_scoop` | float, [-1, 1] | |
| `cmd_stale` | 0 or 1 | optional |

Joint velocities are **computed from positions** using causal backward differences. I will compare these to IMU gyro velocities and update if necessary...

Commands should be the actual values sent to the valves, normalized to [-1, 1] (i.e. `int8_value / 127`). Log them at a consistent rate matching your intended control frequency (`--dt`).

---

## Training

Point it at a folder of drive log CSVs and go:

```bash
python train.py --csv path/to/segments --out my_run --plot
```

The defaults match the paper's time windows (0.1 s of velocity history, 1.0 s of command history). Set `--dt` to match your data's logging rate. It controls the resampling and determines how many timesteps each window covers. Play around with `--hist-qdot-sec` / `--hist-u-sec` to tune to your hardware.

Once training finishes, copy `mlp_state_dict.pt`, `model_meta.json`, and the four `.npy` normalizer files from your run folder into `model/`.

---

## Running the sim

Sim just demoes the usage in IsaacLab. Drop files into isaacLab dir and run similarly to other examples, e.g.

```bash
isaaclab.bat -p scripts/nn/sim.py
```

Use XBox gamepad to control the robot.


- **A** — toggle between NN mode (valve commands → model → velocity) and direct mode (sticks → velocity directly)
- **B** — reset robot to default pose

---

