# gen2 raw-target plant: status 2026-10-21

## Done
- Pulled `dev` (fast-forward to 15daedb).
- Fetched gen2 recordings from the robot (`pikkumasi2tb:~/actuator-data/2026-10-22`) to `D:\gen2_data` (6 recordings, ~56 min usable).
- Finding: the raw CSVs (V3 and gen2) are NOT smoothed. The Savitzky-Golay smoothing was in the label code
  (`local/prototyping/training/prepare_*.py`, `v_smooth21`/`v_smooth9`). Old `data_clean/v3*` outputs must not be reused;
  V3 raw files can be, via `python -m training.drive_logs prepare` (raw gyro target by default).
  V3 imu_raw filenames are a few seconds off from drive_log names: pair by order (copies in `D:\v3_paired`).
- gen2 has ~5-10x more power above 5 Hz in gyro and valve commands than V3.
- Added to `training/gyro_transfer.py` (UNCOMMITTED): `--gyro_only`, `--scratch`, `--pretrain_steps`, `--pretrain_lr`.

## Run: `runs/gen2_raw_v2` (pure raw model, random init)
- Train: warmup chirp, 202706 chirp, 203822 (user's manual driving), 204756 sine, 211003 PID (~46 min). Dev: 205913 stroke. No test split.
- Command: `python -m training.gyro_transfer train --out runs/gen2_raw_v2 --model models/arm_v5_steady --gyro_only --scratch --pretrain_steps 8000 --steps 2000 --lr 1e-4 --eval_every 250`
- Best = step 1500, in `runs/gen2_raw_v2/gyro_model/`. Dev vs V5: 10 s path 4.19 vs 4.90 deg, 30 s path 8.55 vs 10.1 deg, 1 s path 1.12 vs 1.34 deg.
- Problem: rest drift 4.8 deg/10 s (V5 ~0) and stop rate 0.013 rad/s (V5 ~0) -> rejected by selection rule, NOT deployable.
  Likely cause: noisy raw gyro near zero + rest-prior weight 0.2 too weak.

## Next (tomorrow)
1. User adds new data incl. STATIC recordings: fully static (pump on, 5-10 min), abrupt stop then 30-60 s hold,
   single-axis stops (boom/arm/bucket) with long holds, several arm poses.
2. Prepare with `training.drive_logs` (raw target), keep a held-out test recording this time.
3. Stronger rest prior, or rest loss from real static segments; retrain scratch; check drift and stop rate with tracking.
4. If rest is stable: train a PPO controller on the new plant, then the real-machine circle test.

Memory entry: `gen2-raw-target-actuator.md`. Robot password is not stored anywhere; ask.
