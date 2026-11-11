# Offline fusion audit against current robot firmware

The first export used the repository's ready-made replay driver, linked against
the Pico's vendored Fusion implementation and the September V2 settings. It did
not introduce a new AHRS. Its wrapper had already been changed from
`FusionAhrsUpdateNoMagnetometer` to the firmware's direct
`FusionAhrsUpdate(..., FUSION_VECTOR_ZERO, dt)` call. This audit led to a more
faithful wrapper for the expanded dataset; the Fusion mathematics is unchanged.

## Firmware evidence

Sources were retrieved from pikkumasi2tb's current `kaivuriprokkis` checkout and
its retained November 2 `pico-build/dropout/pico_imu_reader` build tree. The
settings, `InitFusion.c`, sensor driver/configuration, `FusionAhrs.c` and
`FusionMath.h` have identical SHA-256 hashes in both trees. The October 21
retained build uses the same AHRS implementation and configuration values.
Its settings-file comments differ, which changes that file's hash.

The later main-loop build includes sensor-loss/stuck-sensor guards. The audited
replayer uses the November 2 main-loop source and preserves its numerical
timing and quaternion-continuity helpers. `robot_firmware_hash_comparison.json`
records the complete hashes, including retained UF2 build images. This identifies
the checked source/build evidence; no running Pico flash readback, USB reset or
firmware flashing was performed, so it is not independent proof of the exact
bytes currently flashed on the board.

## Comparison and corrections

| Component | Current firmware | Expanded-dataset replay |
| --- | --- | --- |
| AHRS implementation | Vendored x-io Fusion | Same source, compiled on the host |
| Settings | NWU, gain 0.5, rejection 5°, recovery 2 s | Same |
| Update call | Direct update with zero magnetometer | Same; no forced yaw zero |
| Initialization | Actual `initialize_algos` and settings | Actual firmware function, not a hand-copied initializer |
| Mid-recording state | Already running after startup | Seed from first recorded quaternion, normal running gain; first sample is not integrated twice |
| Timestamp clamp | Nominal loop interval × 0.5–2 | Same source helper, with the recorded stream's nominal rate |
| Quaternion publication | Choose continuous hemisphere; reset nonfinite AHRS | Same source helpers/guard |
| Gyro saturation passed to Fusion | 573.44 dps; internal trip at 98% | Same |
| Gyro bias | Firmware startup correction; online offset tracker inactive | Recorded correction retained; no second bias estimator |
| Sensor calibration | Applied before streamed gyro/accel | Already-calibrated recorded pair is consumed directly |
| Missing/invalid input | Firmware read/loss guards | Input validation, frozen/saturation flags, gap splits and training masks |

The old ready-made `FusionAhrsUpdateNoMagnetometer` call can force yaw to zero
during AHRS initialization, including initialization triggered by gyro-range
recovery. A deliberately over-range synthetic input exposed a **152.7°**
orientation difference from the firmware path. That test exercises the original
ready-made driver, not the first dataset export's already-corrected direct call.
The expanded export uses the audited current wrapper throughout.

## Numerical tests

The reference executable compiles the actual firmware `InitFusion.c` and the
calibrated update block extracted directly from `main.c`. The replay executable
uses the same Fusion source with the audited host wrapper.

Eight comparisons tested static attitude, slow yaw, prolonged acceleration
rejection/recovery and deliberate gyro saturation, each at 200 Hz and 416 Hz.
Two additional comparisons retained the firmware's cold-start gain ramp and
tested convergence from a tilted gravity vector. All **10 tests produced
identical quaternion components and recovery flags** between the two host
executables. These are tests of identical inputs/timing on the host, not a claim
that an ARM Pico and host have bit-identical floating-point library behavior.

`audited_gen2_fusion/audit.json` records the numerical results. Its source and
reference executable files are retained alongside the dataset.

## Limits of real-recording replay

The Pico updates each sensor around 416 Hz, but the recordings retain only the
200 Hz USB frames and one output timestamp, not every internal sensor update or
per-sensor integration interval. Exact hardware-history replay is impossible.
Recovery counts are therefore adapted to the replay rate: about 400 updates at
200 Hz versus 832 at 416 Hz, keeping the configured duration at 2 seconds.
The 416 Hz timestamp clamp must not be applied directly to a 200 Hz interval;
doing so would systematically under-integrate motion.

Old hardware filtering, prior aliasing, clipped acceleration and historical
startup calibration cannot be reconstructed from these logs. Original raw
values, source generation and quality provenance remain available. All
September orientations are rebuilt with the 5° settings and 30-second settling
exclusions; the October robot gen2 quaternions are retained. Gyro/velocity
targets stay unsmoothed.

Each robot recording also receives a same-motion offline-versus-recorded joint
angle comparison, excluding its first 30 seconds. These results quantify the
200 Hz replay approximation against the firmware's recorded output; they are
not errors against independent physical ground truth. See the per-recording
`offline_vs_recorded_p99_deg` values and `validation.json` in the final bundle.
