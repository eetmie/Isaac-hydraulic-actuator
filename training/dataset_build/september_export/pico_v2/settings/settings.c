#include "settings.h"

// AHRS settings are hardcoded — no host handshake needed.
// Pico self-calibrates on power-on, then streams at sampleRate.
//
// Sensor full scales are requests, not promises: initialize_sensors() snaps each
// to the nearest range the ISM330DLC supports and writes the result back here.
// Ask for the smallest range that still clears the motion actually seen on the
// machine — the stream carries raw gyro and accel, so a recorded strip shows
// directly how much headroom is left before clipping.
//
// Values below are the reviewed proposal, not the previously flashed set.
// Previously flashed:  gyro 250 dps, accel 2 g, gain 4.5, rejection 20 deg,
//                      recovery 0.5 s.
// Why each changed — measured over the 11 raw recordings in
// dataset/hydraulic_data (1,319,834 frames), rescaled to true 8.75 mdps/LSB:
//
//  gyroRangeDps 500      Bucket peaks at 224 dps. At the 250 dps setting the
//                        output saturates at 286.7 dps and Fusion trips its
//                        angular-rate recovery at 281 dps — under 8% headroom
//                        on a link that is already the fastest on the machine.
//                        The extra quantisation (17.5 vs 8.75 mdps/LSB, about
//                        5 mdps rms) sits well under the part's own noise.
//  accelRangeG  4        The bucket accelerometer already clips at +/-2 g in
//                        the existing recordings (6 samples at 1.9998 g, 9
//                        above 1.9 g). Quantisation cost is ~0.035 mg rms,
//                        far below the ~0.5 mg rms of sensor noise in band.
//  ahrsGain     0.5      Gain is 1/tau for accelerometer feedback, so 4.5 put
//                        the correction corner at 0.72 Hz — inside the 1-2 Hz
//                        band the machine is deliberately excited in. Offline
//                        replay of the vendored Fusion over these recordings
//                        cut the gyro/attitude inconsistency from 0.47 to
//                        0.11 deg and the attitude lag from 25 ms to 2.5 ms.
//                        Use 1.0f instead if stationary drift matters more
//                        than lag; do not exceed 2.0f on this machine.
//  ahrsAccelRejection 10 Tighter than 20 deg, more conservative than 5 deg.
//  ahrsRecoveryPeriodS 2 Now counted in AHRS updates at ahrsRateHz. The old
//                        0.5 s was multiplied by sampleRate (200) and applied
//                        at 416 Hz, so it was really 0.24 s.
imu_reader_settings_t imu_reader_settings = {
    .sampleRate          = 200,
    .ahrsRateHz          = 200,     // overwritten by main.c with the real AHRS rate
    .gyroRangeDps        = 500.0f,
    .accelRangeG         = 4.0f,
    .gyroSaturationDps   = 0.0f,    // resolved by initialize_sensors()
    .ahrsGain            = 0.5f,
    .ahrsAccelRejection  = 10.0f,
    .ahrsRecoveryPeriodS = 2.0f,
};
