# First real-robot circle logs: MLP vs PID (2026-10-07)

Five-minute continuous sessions of the 100 mm, 20 mm/s bucket circle, recorded
with `run_robot_circle.py run --continuous` on the Jetson. Each CSV holds about
30 000 rows at about 100 Hz, from approach and settling to a final partial pass.
Use the JSON `passes` list, or the `pass_index` column, to keep only complete
circles.

| Stem | Controller | Profile | Input deadzone |
|---|---|---|---|
| `circle_mlp_100hz_live_20261007_10` | MLP, actor at 100 Hz | `jetson_bucket` | 2 % |
| `circle_pid_dz0_live_20261007_03` | tuned PID (`pid_gains.yaml`) | `jetson_bucket_dz0` | 0 % |
| `circle_mlp_dz0_live_20261007_03` | MLP, actor at 100 Hz | `jetson_bucket_dz0` | 0 % |

`successful_circle_comparison.json` gives the 15 complete circles per session.
The MLP tracks better than the PID: 3.7 mm vs 6.1 mm RMSE, and 1.8 mm vs 2.3 mm
mean radial error. It also moves the valves about 12x more (19.5 vs 1.7 travel
per second) and rocks the carriage more (3.7 vs 0.3 deg/s RMS pitch rate). Tip
positions come from the joint angles, not from an independent position
measurement. The sessions ran one after another, and the oil temperature was
not controlled. See `recordings_manifest.json` for the notes and units.

## Files

- `*.csv.gz`: the full logs, gzipped. The sha256 values in
  `recordings_manifest.json` are for the decompressed CSVs.
  `pandas.read_csv` reads `.csv.gz` directly, and `gunzip -k` restores the
  originals.
- `*.json`: the run report. It holds settings, per-pass scores, the IMU
  mapping, and the result.
- `*_rocking.json`: the carriage-rocking analysis for each session.
- `handoff_probe_20261007.csv.gz`: a short probe recording from the same day,
  in the same schema.
- `circle_start_pose.csv` and `pid_gains.yaml`: the shared start pose and the
  PID gains.
- `bundles/bucket` and `bundles/bucket_dz0`: the deployed actor bundles. The
  actor is the same in both. The dz0 bundle carries the dz0 profile config.
- `compare_dz0.py`: regenerates `successful_circle_comparison.{json,png}` from
  the gzipped logs (needs matplotlib).
- `figures/`: all-IMU and rocking plots for each session, plus the comparison
  figure.
