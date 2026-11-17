# Policy-rate results: use 20/20 Hz

Decision (17 November 2026): train and execute the pivot policy at **20 Hz**.
The packaged `models/controller_pivot/model_3499.pt` already uses this timing;
keep it as the default. Hydraulic dynamics and measured histories run at 100 Hz.
Valve activity takes priority over small improvements in tracking error.

## Results

Both studies used three training seeds, two actuator plants, common perturbations,
5-120 mm/s lines, circles and fixed-tip rotations. Tables show seed means for gyro
line tests on the new plant, including infeasible requests and endpoint holds.
Lower is better; feasible motion windows are reported separately.

Matched training used 1,500 PPO updates per policy, 1.2-second rollouts, 200 ms
preview and fixed learning rate 3e-4. L1 valve cost stays fixed; L2 scales with
Hz/20. Curriculum and moving-state refresh keep their physical durations.

| Train/run Hz | Tip MAE mm | Valve travel/s | Valve step p99 | Motion jerk m/s^3 |
| --- | ---: | ---: | ---: | ---: |
| 10/10 | 6.30 | 1.00 | 0.135 | 12.05 |
| **20/20** | **5.47** | **1.61** | **0.106** | **12.55** |
| 50/50 | 5.69 | 4.02 | 0.107 | 18.04 |
| 100/100 | 26.51 | 7.88 | 0.110 | 36.17 |
| Packaged 20/20 reference | 4.65 | 1.78 | 0.108 | 10.03 |

The reference had 3,500 updates and is outside the matched group. Fresh 20/20
policies had no invalid/limit/collision cases in these gyro line tests.

Faster execution reused the exact 10 Hz checkpoints without retraining or filtering.
Histories stay at 100 Hz and preview slots at +100/+200 ms at every execution rate.
This explores the slow-training/fast-execution approach in
[Egli & Hutter, Sections II-B/C](https://doi.org/10.3929/ethz-b-000487440).

| Train/run Hz | Tip MAE mm | Valve travel/s | Valve step p99 | Motion jerk m/s^3 |
| --- | ---: | ---: | ---: | ---: |
| 10/20 | 6.11 | 1.84 | 0.116 | 12.17 |
| 10/50 | 6.05 | 4.49 | 0.112 | 12.48 |
| 10/100 | 5.95 | 8.91 | 0.113 | 13.09 |

100 Hz reduces individual jumps and holding drift, but uses about nine times the
valve travel of 10/10 for little line-tracking gain. Rotation results vary by seed;
some exceed the 20 mm tip-hold target. Keep 20/20; physical jitter and accuracy
still need machine measurements. Lines/circles use planned preview; rotations
use hold-current-command preview.

## Reproduce and inspect

From a `training.freeze_comparison` snapshot, use Isaac Lab's uv environment.
These commands resume saved plans and skip completed stages:

```powershell
$runs = Join-Path $HOME 'IsaacTraining/runs'
uv run --project "$HOME/IsaacLab" python -u -m training.policy_rate_comparison `
    --base "$runs/policy_rates_20261116/frozen/base" `
    --out "$runs/policy_rates_20261116/study" --finish_at 2026-11-17T10:00:00+02:00
uv run --project "$HOME/IsaacLab" python -u -m training.execution_rate_comparison `
    --source_study "$runs/policy_rates_20261116/study" `
    --base "$runs/train10_exec_rates_20261117/frozen/base" `
    --out "$runs/train10_exec_rates_20261117/study"
```

Each study retains `status.json`, `results.md`, `aggregate.csv`,
`speed_analysis/results.md`, `rotation_summary.json` and raw 100 Hz traces locally.
Use `--smoke` with a separate output directory for a small end-to-end check.
No policy is automatically promoted. See [robot deployment](robot_deployment.md)
for native integration requirements.
