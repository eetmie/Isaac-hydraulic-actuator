# Pivot controller trained with 0-80 ms sensor delay

`model_3499.pt` is `delay8_s43` from the 2026-11-19 jitter study (`IsaacTraining/funky_20261119`). The recipe is
the `controller_pivot` recipe (3,500 PPO updates, 4,096 envs, fixed lr 3e-4, both actuator plants), except for
`sensor_delay_max_steps: 8` (0-80 ms random sensor delay instead of 0-20 ms) and seed 43.

On benchmark circles at 20 mm/s with a fixed sensor delay and real-IMU-level noise, its speed surge (rms
speed error divided by speed) compared with `controller_pivot`:

| sensor delay | 0 ms | 30 ms | 50 ms | 80 ms |
|---|---|---|---|---|
| controller_pivot | 0.26 | 0.43 | 0.58 | 0.84 |
| controller_pivot_delay8 | 0.26 | 0.32 | 0.46 | 0.68 |

Tracking error without delay is unchanged (median max 6.3 mm). The same recipe with seed 42 was worse, so this is
a selected seed, not a proven recipe. `robot_bundle/` is the `export_robot_bundle.py native` export for
`ppo_live.py --bundle`. Not hardware validated.
