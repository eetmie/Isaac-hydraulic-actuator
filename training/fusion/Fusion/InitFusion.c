#include "InitFusion.h"
#include "pico/time.h"

extern imu_reader_settings_t imu_reader_settings;

void initialize_sensors_values(Sensor* sensors, int count) {
    for (int i = 0; i < count; i++) {
        sensors[i].gyroscope.axis.x = 0.0f;
        sensors[i].gyroscope.axis.y = 0.0f;
        sensors[i].gyroscope.axis.z = 0.0f;

        sensors[i].accelerometer.axis.x = 0.0f;
        sensors[i].accelerometer.axis.y = 0.0f;
        sensors[i].accelerometer.axis.z = 0.0f;

        sensors[i].mountingQuaternion = FUSION_IDENTITY_QUATERNION;
        sensors[i].pitchDeg = 0.0f;
        sensors[i].rollDeg = 0.0f;

        // Initialize timestamps to avoid huge deltaTime on first iteration
        sensors[i].timestamp = time_us_64();
        sensors[i].previousTimestamp = sensors[i].timestamp;

        // Initialize previous quaternion for continuity tracking
        sensors[i].previousQuaternion = FUSION_IDENTITY_QUATERNION;
    }
}

void initialize_calibrations(Sensor* sensors, int count) {
    for (int i = 0; i < count; i++) {
        // Identity calibration (no correction)
        sensors[i].calibration.gyroscopeMisalignment = (FusionMatrix){
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
        sensors[i].calibration.gyroscopeSensitivity = (FusionVector){1.0f, 1.0f, 1.0f};
        sensors[i].calibration.gyroscopeOffset = (FusionVector){0.0f, 0.0f, 0.0f};

        sensors[i].calibration.accelerometerMisalignment = (FusionMatrix){
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
        sensors[i].calibration.accelerometerSensitivity = (FusionVector){1.0f, 1.0f, 1.0f};
        sensors[i].calibration.accelerometerOffset = (FusionVector){0.0f, 0.0f, 0.0f};
    }
}

void initialize_algos(Sensor* sensors, int count) {
    // Fusion counts recovery periods in AHRS updates. main.c steps the AHRS at
    // SENSOR_INTERNAL_ODR_HZ and only publishes every other update over USB, so
    // multiplying by the output rate would make every configured period short
    // by that ratio. Fall back to the output rate only if nothing set this.
    int ahrs_rate_hz = imu_reader_settings.ahrsRateHz;
    if (ahrs_rate_hz <= 0) ahrs_rate_hz = imu_reader_settings.sampleRate;

    // Fusion trips angular-rate recovery at 0.98 * gyroscopeRange, so it needs
    // the rate at which the output code actually saturates. That is 32768 *
    // sensitivity (286.7 dps at the "250 dps" setting), not the nominal name.
    float gyro_range = imu_reader_settings.gyroSaturationDps;
    if (gyro_range <= 0.0f) gyro_range = imu_reader_settings.gyroRangeDps;

    for (int i = 0; i < count; i++) {
        // Initialize gyro offset tracker. Nothing in the streaming loop calls
        // FusionOffsetUpdate, so this stays inert: the only bias correction is
        // the startup one applied in the driver. Enabling the tracker would
        // absorb genuine slow rotation into the bias (its stationary test is a
        // 3 dps threshold held for 5 s) and needs a motion/valve interlock
        // first — see docs/IMU_STARTUP_BIAS.md.
        FusionOffsetInitialise(&sensors[i].offset, ahrs_rate_hz);

        // Initialize AHRS
        FusionAhrsInitialise(&sensors[i].ahrs);

        // Recovery period in samples
        float recovery_s = imu_reader_settings.ahrsRecoveryPeriodS;
        if (recovery_s <= 0.0f) recovery_s = 1.0f;

        sensors[i].settings = (FusionAhrsSettings){
            .convention = FusionConventionNwu,
            .gain = imu_reader_settings.ahrsGain,
            .gyroscopeRange = gyro_range,
            .accelerationRejection = imu_reader_settings.ahrsAccelRejection,
            .magneticRejection = 0.0f,  // No magnetometer
            .recoveryTriggerPeriod = (unsigned int)(recovery_s * (float)ahrs_rate_hz),
        };
        FusionAhrsSetSettings(&sensors[i].ahrs, &sensors[i].settings);
    }
}
