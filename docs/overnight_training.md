# Overnight training on a Windows laptop

`overnight_training.ps1` runs the whole chain unattended on one CUDA GPU (tested budgets assume
an 8 GB laptop GPU, about 7–9 hours):

1. Prepare the split. The Oct 22 robot sessions are the held-out test set; dev is
   `v2_20260909` and robot `…-71723`.
2. Actuator A: from scratch, 30k rollout steps at lr 1e-4. The desktop run's loss was
   still falling at 6k steps.
3. Actuator B and C: continue from the previous phase's last weights at lr 3e-5, then 1e-5,
   20k steps each.
4. Add the closed-valve bound (`training.neutral_bound`) to each phase's best checkpoint,
   pick the best on dev, and report everything on test.
5. Two PPO controllers on two plants (best new actuator + deployed V5): the original
   recipe, and lookahead + acceleration reward (`lookahead_steps=4 accel_weight=0.4`).

## Setup (once)

From PowerShell, with [uv](https://docs.astral.sh/uv/) and git installed:

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git --branch v3.0.0
cd IsaacLab
git clone -b dev https://github.com/eetmie/Isaac-hydraulic-actuator.git scripts/Isaac-hydraulic-actuator
cd scripts/Isaac-hydraulic-actuator
# Training needs only the manifest and the 100 Hz chunks (about 120 MB of the 1.4 GB dataset).
uvx --from huggingface_hub hf download eetmie/excavator-hydraulic-actuator --type dataset `
    --include dataset.json --include "training/*" --local-dir dataset/complete_dataset
```

Keep the laptop on mains power and stop it sleeping (`powercfg /change standby-timeout-ac 0`).

## Run

From `scripts/Isaac-hydraulic-actuator`:

```powershell
# A few minutes: every step with tiny budgets. Check that it ends with "done".
powershell -ExecutionPolicy Bypass -File docs/overnight_training.ps1 -Smoke
# The real run.
powershell -ExecutionPolicy Bypass -File docs/overnight_training.ps1
```

Each step prints its log file; follow one with `Get-Content runs/overnight/<...>.log -Wait`.
If the run stops (crash, reboot), start it again: finished phases are skipped and only the
interrupted one restarts.

## Results

- `runs/overnight/compare_dev.log`, `compare_test.log`: all actuators against the deployed one.
- `runs/overnight/*/model/history.json`: dev metrics every 1000 steps; `train.log`: loss every 25.
- PPO checkpoints: `logs/rsl_rl/hydraulic_controller/<time>_overnight_*`.

Bring back `runs/overnight` and the two `logs/rsl_rl/hydraulic_controller/*overnight*` folders
to compare the controllers with `compare_controllers.py` (it needs the kaivuriprokkis checkout).
