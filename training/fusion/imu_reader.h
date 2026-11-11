#ifndef IMU_READER_H
#define IMU_READER_H

typedef struct imu_reader_settings_t {
    // USB output frame rate. Purely a transport rate: the AHRS runs faster.
    int   sampleRate;
    // Rate the AHRS is actually stepped at, in Hz. Fusion counts recovery
    // periods in updates, not in output frames, so this — not sampleRate —
    // is what a recovery period in seconds must be multiplied by. main.c
    // writes SENSOR_INTERNAL_ODR_HZ here before initialize_algos().
    int   ahrsRateHz;
    // Requested sensor full scales. initialize_sensors() rewrites both to the
    // nearest range the part actually supports, so everything downstream (the
    // raw-count conversion, Fusion's gyroscopeRange, the descriptor frame the
    // host logs) reads one already-normalized value.
    float gyroRangeDps;
    float accelRangeG;
    // Rate at which the gyro output code saturates, 32768 * sensitivity.
    // Written by initialize_sensors(); do not set it by hand. This is NOT
    // gyroRangeDps: at the +/-250 dps setting the part reads 8.75 mdps/LSB,
    // so the output saturates at 286.7 dps, not 250. Fusion trips its
    // angular-rate recovery at 0.98 * this, so feeding it the nominal range
    // resets the estimator ~15% below real saturation.
    float gyroSaturationDps;
    float ahrsGain;
    float ahrsAccelRejection;
    float ahrsRecoveryPeriodS;
} imu_reader_settings_t;

// Global settings (defined in settings.c, consumed by InitFusion.c)
extern imu_reader_settings_t imu_reader_settings;

#endif
