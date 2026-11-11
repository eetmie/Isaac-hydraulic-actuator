# Complete actuator dataset — gen2 processing

This corpus includes all unique September Desktop recordings and the October
robot recordings downloaded from `pikkumasi2tb:~/actuator-data/2026-10-22`.
All orientations use **current gen2 processing: NWU, gain 0.5, acceleration
rejection 5°, recovery 2 seconds**. Gyro/velocity targets are **unsmoothed**.

| Source collection | Recordings | Recorded minutes | Training minutes |
| --- | ---: | ---: | ---: |
| September 7 V1 | 11 | 110.00 | 98.84 |
| September V2, including three V3 extras | 11 | 98.31 | 88.19 |
| October 21–22 robot gen2 | 11 | 109.36 | 105.85 |
| Total | **33** | **317.67** | **292.88** |

The eight September recordings duplicated between `hydraulic_data_2` and `V3`
are counted once. The corpus retains **3,811,747 raw frames** and exports
**168 continuous 100 Hz training chunks, 1,757,300 rows**. Canonical format v1
is retained for compatibility with the existing complete-dataset training adapter;
`normalized_generation: gen2` identifies the processing generation.

## Fusion and data quality

All September quaternions were rebuilt from their original raw inputs using the
robot's current Fusion sources and settings. V1 gyro scaling was corrected by
**1.14688**; September V2 gyro scaling was already correct. The October robot
quaternions were preserved because they already match current gen2 processing.
Source-generation labels remain in the manifest for provenance.

The offline wrapper uses the actual Pico `initialize_algos` and vendored Fusion
code, with the current firmware's direct zero-magnetometer update, timestamp
clamp and quaternion-continuity behavior. It was audited against a reference
compiled from the actual firmware update block: **10 host comparisons produced
identical quaternions and recovery flags**, including cold startup and saturation.
See [the audit](provenance/FUSION_AUDIT.md) and `provenance/audited_gen2_fusion/`.

The replay uses the first recorded quaternion as a mid-stream seed and excludes
30 seconds at the start and after gaps over 10 ms. Six V1 near-saturation frames
are flagged and excluded from training. Seventeen isolated, nonzero low-g frames
in the robot collection are retained and diagnosed, consistent with the live
firmware; its loss guard requires repeated failures. No frozen or zeroed sensor
streams were accepted for training.

Robot session manifests and all referenced CSV/excitation/PID attachments were
hash-verified with the project's native importer. Pump state, write success,
controller/input validity, exact packet identity and causal host-read availability
are enforced for robot training rows. September logs lack pump/write metadata,
which remains an explicit provenance limitation.

## Layout and schema

- `dataset.json`: source/output checksums, settings, recording index, quality
  counts, collection groups and training paths. Every recording has
  `processing_generation: gen2` and the same `fusion_settings`.
- `recordings/<id>/imu.npz`: original raw columns plus `canonical_quat_wxyz`,
  `canonical_gyro_dps`, `canonical_accel_g`, unwrapped device clock and validity
  flags. Canonical IMU arrays use **base, boom, arm, bucket** role order.
- `recordings/<id>/control.npz`: original drive columns plus `canonical_q`,
  `canonical_v`, exact source packet row, state availability and validity.
- `training/unsplit/*.npz`: `q`, `v`, `target_q`, `target_v`, `u`, `t`, source row
  indexes and timing. State order is **boom, arm, bucket, base pitch**;
  positions are radians and rates rad/s. Action order is **lift, tilt, scoop**,
  normalized to [-1, 1]. `u_rotate` separately retains the slew command.
  `target_q == q` and `target_v == v`; no Savitzky–Golay smoothing is used.
- `validation.json`, `SHA256SUMS`: numerical/integrity checks and package hashes.
- `provenance/`: audited firmware/replay source, builder and validator code,
  earlier source inventory and firmware/build hash comparison.

Original column names retain original values, including V1's original gyro
scale and fused angles. Use the **canonical** fields or training chunks for
normalized gen2 data. Replay flags bit 0 means acceleration ignored, bit 1
acceleration recovery. Preserved native robot flags are 255 = not recorded.

Training clocks are per recording. Never concatenate them into one continuous
timeline or bridge chunk boundaries. Splits remain unassigned; keep each
`collection_group` whole when assigning train/dev/test. The original validation
recording retains `validation_named: true`. Robot collection IDs keep related
strips together even when a process created multiple recordings.

## Practical limits

The Pico integrates around 416 Hz but the logs contain only 200 Hz output and
one timestamp per frame. Offline reconstruction cannot reproduce every internal
update or per-sensor time interval. Across the 11 robot recordings, replay versus
recorded joint angles stayed below **0.31° p99** after settling; these are
comparisons to firmware output, not independent physical ground truth.

Normalization does not recreate newer hardware filtering before sampling, undo
historical aliasing/clipping or recover the original startup calibration
measurements. The audit compared retained November 2 build sources and the robot
repository; no live flash readback or firmware change was performed.

Check integrity from this directory with `sha256sum --quiet -c SHA256SUMS`.
The dataset is self-contained for training; original Desktop CSV paths and
temporary audit-directory paths are provenance references, not runtime inputs.
