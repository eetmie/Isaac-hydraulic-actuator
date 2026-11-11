# Pinned robot fusion sources

These are the sources audited against `kaivuriprokkis/pico_imu_reader` and the
robot's retained November 2, 2026 build. `ORIGIN.json` records the source hashes.
`Fusion/InitFusion.c`, `settings.c` and `sensor_config.h` retain the firmware's
initializer, settings and sensor range table. The vendored x-io Fusion license
is retained in `Fusion/LICENSE.md`.

`replay.c` adapts the recorded sample rate and resumes from recorded attitude.
`firmware_reference.c` preserves the calibrated update block extracted from the
firmware main loop for comparison. Both compile the actual Fusion implementation.
The local `pico/time.h` and `i2c_helpers.h` only supply host type declarations;
they do not emulate sensor reads or firmware timing.

Run `python -m training.offline_fusion audit` from the project root. Generated
binaries are cached under ignored `local/fusion_cache`, keyed by source contents.
See [the audit](../../docs/offline_fusion_audit.md) for evidence, input units,
settings and the limits of replaying 200 Hz recordings from a 416 Hz reader.
