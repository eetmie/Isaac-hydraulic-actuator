#include "ism330dlc.h"
#include "i2c_helpers.h"
#include "output.h"
#include "FusionMath.h"
#include "cdc_console.h"
#include "pico/time.h"
#include <math.h>

#define I2C_REG_TIMEOUT_US 2000u

typedef struct ActiveImu {
    i2c_inst_t *i2c_port;
    uint8_t bus_index;
    uint8_t device_addr;
} ActiveImu;

static ActiveImu active_imus[MAX_SENSORS];
static FusionVector active_gyro_biases[MAX_SENSORS];
static uint8_t active_sensor_count = 0;

static const imu_full_scale_t gyro_full_scales[] = ISM330_GYRO_FULL_SCALES;
static const imu_full_scale_t accel_full_scales[] = ISM330_ACCEL_FULL_SCALES;

// Register bits chosen for the ranges in imu_reader_settings, resolved once by
// normalize_full_scales() before any sensor is configured.
static uint8_t active_gyro_range_mask = 0x00;
static uint8_t active_accel_range_mask = 0x00;
static float active_gyro_sensitivity = 0.008750f;
static float active_accel_sensitivity = 0.000061f;

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

// Pick the smallest supported full scale that still covers the request, and
// return the whole table row. Register bits and conversion scale then always
// come from the same row: a request the part cannot honour exactly (300 dps,
// say) lands on the next range up, and dividing raw counts by the *requested*
// value instead of the selected one would skew every sample by that ratio.
static imu_full_scale_t select_full_scale(const imu_full_scale_t *table, size_t count, float requested) {
    for (size_t i = 0; i < count; i++) {
        if (requested <= table[i].scale) {
            return table[i];
        }
    }
    return table[count - 1];
}

// Resolve both requested ranges against the hardware and write the results back
// into imu_reader_settings, so the conversion below, Fusion's gyroscopeRange and
// the descriptor frame all read the range the sensor is genuinely running at.
static void normalize_full_scales(void) {
    const float gyro_request = (imu_reader_settings.gyroRangeDps > 0.0f)
        ? imu_reader_settings.gyroRangeDps : 250.0f;
    const float accel_request = (imu_reader_settings.accelRangeG > 0.0f)
        ? imu_reader_settings.accelRangeG : 2.0f;

    const imu_full_scale_t gyro_fs =
        select_full_scale(gyro_full_scales, ARRAY_COUNT(gyro_full_scales), gyro_request);
    const imu_full_scale_t accel_fs =
        select_full_scale(accel_full_scales, ARRAY_COUNT(accel_full_scales), accel_request);

    imu_reader_settings.gyroRangeDps = gyro_fs.scale;
    imu_reader_settings.accelRangeG = accel_fs.scale;
    active_gyro_range_mask = gyro_fs.mask;
    active_accel_range_mask = accel_fs.mask;
    active_gyro_sensitivity = gyro_fs.sensitivity;
    active_accel_sensitivity = accel_fs.sensitivity;
    // The rate at which the output code runs out, not the nominal range name.
    // InitFusion.c hands this to Fusion so its angular-rate recovery trips at
    // real saturation instead of 14.7% below it.
    imu_reader_settings.gyroSaturationDps = 32768.0f * gyro_fs.sensitivity;
}

static bool ism330dhcx_probe(i2c_inst_t *i2c_port, uint8_t device_addr) {
    uint8_t who_am_i = 0;
    if (!ism330dhcx_read_reg(i2c_port, device_addr, WHO_AM_I, &who_am_i, 1)) {
        return false;
    }
    return who_am_i == ISM330DHCX_ID;
}

// Function to write to ISM330DHCX register
bool ism330dhcx_write_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t value) {
    uint8_t buf[2] = {reg, value};
    int result = i2c_write_timeout_us(i2c_port, device_addr, buf, 2, false, I2C_REG_TIMEOUT_US);
    return result == 2;
}

// Function to read from ISM330DHCX register
bool ism330dhcx_read_reg(i2c_inst_t *i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value, uint8_t read_count) {
    int result = i2c_write_timeout_us(i2c_port, device_addr, &reg, 1, true, I2C_REG_TIMEOUT_US);
    if (result != 1) return false;
    result = i2c_read_timeout_us(i2c_port, device_addr, value, read_count, false, I2C_REG_TIMEOUT_US);
    return result == read_count;
}


void print_list(uint8_t list[], int size){
    cdc_write_str("Raw gyro values: ");
    for (int i = 0; i < size; i++){
        cdc_writef("%d ", list[i]);
    }
    cdc_write_str("\n");
}

bool ism330dhcx_read_gyro(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector) {
    uint8_t raw_gyro_values[6];
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_G, raw_gyro_values, 6)) {
        return false;
    }
    int16_t raw_gyro_x = combine_8_bits(raw_gyro_values[0], raw_gyro_values[1]);
    int16_t raw_gyro_y = combine_8_bits(raw_gyro_values[2], raw_gyro_values[3]);
    int16_t raw_gyro_z = combine_8_bits(raw_gyro_values[4], raw_gyro_values[5]);

    fusion_vector->axis.x = (float)raw_gyro_x * active_gyro_sensitivity;
    fusion_vector->axis.y = (float)raw_gyro_y * active_gyro_sensitivity;
    fusion_vector->axis.z = (float)raw_gyro_z * active_gyro_sensitivity;
    return true;
}

bool ism330dhcx_read(i2c_inst_t* i2c_port, uint8_t device_addr, uint8_t reg, uint8_t* value) {
	// ism330dhcx_read_accelerometer();
	// ism330dhcx_read_gyro();
	return 0;
}

// Choose ODR field for requested sample rate; returns CTRL ODR bits in [7:4]
static inline uint8_t select_odr_bits(int requested_hz) {
    if (requested_hz <= 104) return 0x40;   // 104 Hz
    if (requested_hz <= 208) return 0x50;   // 208 Hz
    if (requested_hz <= 416) return 0x60;   // 416 Hz
    if (requested_hz <= 833) return 0x70;   // 833 Hz
    // Fallback to 1.66 kHz for anything higher
    return 0x80;                             // 1660 Hz
}

// Bring one sensor to a known reset state. A Pico reset does not reset the
// IMUs, so without this the part is configured on top of whatever the previous
// firmware left behind — including filter bits this version never writes.
static bool ism330dhcx_soft_reset(i2c_inst_t *i2c_port, uint8_t device_addr) {
    const uint8_t CTRL3_C_SW_RESET = 0x01;  // bit0, self-clearing
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL3_C, CTRL3_C_SW_RESET)) {
        return false;
    }
    // The datasheet allows ~50 us; poll with a wide margin instead of a blind wait.
    for (int attempt = 0; attempt < 100; attempt++) {
        sleep_us(100);
        uint8_t ctrl3 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL3_C, &ctrl3, 1)) {
            continue;
        }
        if ((ctrl3 & CTRL3_C_SW_RESET) == 0u) {
            return true;
        }
    }
    return false;
}

// Initialize ISM330DHCX
bool ism330dhcx_init(i2c_inst_t *i2c_port, uint8_t device_addr, uint internal_odr_hz) {
    if (!ism330dhcx_soft_reset(i2c_port, device_addr)) {
        cdc_writef("Software reset did not complete (addr=0x%02x)\n", device_addr);
        return false;
    }

    // ST AN5398: set the device-configuration bit (CTRL9_XL bit 1) before the
    // rest of the configuration, not after it.
    {
        uint8_t ctrl9 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL9_XL, &ctrl9, 1)) return false;
        ctrl9 |= 0x02;
        if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL9_XL, ctrl9)) return false;
    }

    // Enable auto-increment + BDU for clean multi-byte reads
    const uint8_t CTRL3_C_IF_INC = 0x04;   // bit2
    const uint8_t CTRL3_C_BDU    = 0x40;   // bit6
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL3_C, CTRL3_C_IF_INC | CTRL3_C_BDU)) {
        return false;
    }

    // Map requested internal ODR to nearest supported hardware ODR at or above request
    const uint8_t odr_bits = select_odr_bits(internal_odr_hz);

    // Configure accelerometer and gyroscope: ODR + the resolved full scales.
    // CTRL1_XL bit 1 selects LPF2 (not CTRL8_XL bit 7).
    const uint8_t xl_cntrl1_val = odr_bits | active_accel_range_mask | 0x02;
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL1_XL, xl_cntrl1_val)) {
        cdc_writef("Failed to configure accelerometer (addr=0x%02x)\n", device_addr);
        return false;
    }

    const uint8_t g_cntrl_val = odr_bits | active_gyro_range_mask;
    if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL2_G, g_cntrl_val)) {
        cdc_writef("Failed to configure gyroscope (addr=0x%02x)\n", device_addr);
        return false;
    }
    // ISM330DHCX DS13012: HPCF_XL[2:0] are CTRL8_XL bits [7:5].
    // Low-pass (not high-pass), ODR/10 = 41.6 Hz at ODR 416 Hz.
    {
        uint8_t ctrl8 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL8_XL, &ctrl8, 1)) return false;
        ctrl8 = (uint8_t)((ctrl8 & (uint8_t)~0xF7) | 0x20);
        if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL8_XL, ctrl8)) return false;
    }

    // FTYPE is bits [2:0], not [3:0]. Preserve accelerometer offset weight.
    {
        uint8_t ctrl6 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL6_C, &ctrl6, 1)) return false;
        ctrl6 = (uint8_t)((ctrl6 & (uint8_t)~0x07) | 0x05);
        if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL6_C, ctrl6)) return false;
    }
    // Choosing FTYPE alone does not enable LPF1. CTRL4_C bit 1 does.
    {
        uint8_t ctrl4 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL4_C, &ctrl4, 1)) return false;
        ctrl4 |= 0x02;
        if (!ism330dhcx_write_reg(i2c_port, device_addr, CTRL4_C, ctrl4)) return false;
    }

    // Read the filter registers back. These four bits are the whole reason the
    // previous configuration was silently inert, so confirm them rather than
    // trusting that the writes landed.
    {
        uint8_t ctrl1 = 0, ctrl4 = 0, ctrl6 = 0, ctrl8 = 0;
        if (!ism330dhcx_read_reg(i2c_port, device_addr, CTRL1_XL, &ctrl1, 1) ||
            !ism330dhcx_read_reg(i2c_port, device_addr, CTRL4_C, &ctrl4, 1) ||
            !ism330dhcx_read_reg(i2c_port, device_addr, CTRL6_C, &ctrl6, 1) ||
            !ism330dhcx_read_reg(i2c_port, device_addr, CTRL8_XL, &ctrl8, 1)) {
            return false;
        }
        const bool filters_ok = ((ctrl1 & 0x02u) != 0u) && ((ctrl4 & 0x02u) != 0u) &&
                                ((ctrl6 & 0x07u) == 0x05u) && ((ctrl8 & 0xE0u) == 0x20u);
        cdc_writef("ISM330 0x%02x filters: CTRL1_XL=0x%02x CTRL4_C=0x%02x CTRL6_C=0x%02x CTRL8_XL=0x%02x %s\n",
                   device_addr, ctrl1, ctrl4, ctrl6, ctrl8, filters_ok ? "OK" : "MISMATCH");
        if (!filters_ok) {
            return false;
        }
    }

    cdc_write_line("ISM330: accel LPF2 on, HPCF=001 (ODR/10); gyro LPF1 on, FTYPE=5");

    return true;
}

int initialize_sensors(uint internal_odr_hz) {
    static const struct {
        i2c_inst_t *i2c_port;
        uint8_t bus_index;
        uint8_t device_addr;
    } candidates[MAX_SENSORS] = {
        // Must match pico_imu_simulator and control_config.yaml imu_mapping:
        // [0] base=I2C1:0x6B, [1] boom=I2C0:0x6A,
        // [2] bucket=I2C0:0x6B, [3] arm=I2C1:0x6A.
        {I2C_PORT_1, 1, ISM330DHCX_ADDR_DO_HIGH},
        {I2C_PORT_0, 0, ISM330DHCX_ADDR_DO_LOW},
        {I2C_PORT_0, 0, ISM330DHCX_ADDR_DO_HIGH},
        {I2C_PORT_1, 1, ISM330DHCX_ADDR_DO_LOW},
    };

    active_sensor_count = 0;
    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        active_gyro_biases[i] = FUSION_VECTOR_ZERO;
    }

    // Resolve the requested ranges before any sensor is touched: the register
    // writes below and every conversion afterwards read the normalized values.
    normalize_full_scales();
    // Every supported full scale is a whole number, so print as int — the
    // console's vsnprintf has no float support linked in.
    cdc_writef("Full scale: gyro +/-%d dps (saturates at %d dps), accel +/-%d g\n",
               (int)imu_reader_settings.gyroRangeDps,
               (int)imu_reader_settings.gyroSaturationDps,
               (int)imu_reader_settings.accelRangeG);
    cdc_write_line("Probing IMUs on I2C0/I2C1 @ 0x6A/0x6B...");

    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        const uint8_t addr = candidates[i].device_addr;
        const uint8_t bus = candidates[i].bus_index;

        if (!ism330dhcx_probe(candidates[i].i2c_port, addr)) {
            cdc_writef("No IMU at I2C%d @ 0x%02x\n", bus, addr);
            continue;
        }

        if (!ism330dhcx_init(candidates[i].i2c_port, addr, internal_odr_hz)) {
            cdc_writef("Probe OK but init failed: I2C%d @ 0x%02x\n", bus, addr);
            continue;
        }

        active_imus[active_sensor_count].i2c_port = candidates[i].i2c_port;
        active_imus[active_sensor_count].bus_index = bus;
        active_imus[active_sensor_count].device_addr = addr;
        active_sensor_count++;
        cdc_writef("Active IMU %d: I2C%d @ 0x%02x\n", active_sensor_count - 1, bus, addr);
    }

    cdc_writef("Detected %d IMU(s)\n", active_sensor_count);
    return active_sensor_count > 0 ? 1 : 0;
}

uint8_t get_active_sensor_count(void) {
    return active_sensor_count;
}

uint8_t get_active_sensor_bus(uint8_t sensor_index) {
    if (sensor_index >= active_sensor_count) {
        return 0xFF;
    }
    return active_imus[sensor_index].bus_index;
}

uint8_t get_active_sensor_addr(uint8_t sensor_index) {
    if (sensor_index >= active_sensor_count) {
        return 0xFF;
    }
    return active_imus[sensor_index].device_addr;
}

// Wait for new data to be ready from the sensor
// Bit 0: XLDA (accelerometer data available)
// Bit 1: GDA (gyroscope data available)
bool ism330dhcx_wait_for_data(i2c_inst_t* i2c_port, uint8_t device_addr, uint32_t timeout_us) {
    uint64_t start = time_us_64();
    uint8_t status;

    while ((time_us_64() - start) < timeout_us) {
        if (ism330dhcx_read_reg(i2c_port, device_addr, STATUS_REG, &status, 1)) {
            // Check if both accelerometer (bit 0) and gyroscope (bit 1) data are ready
            if ((status & 0x03) == 0x03) {
                return true;
            }
        }
        // Small delay to avoid hammering the I2C bus
        sleep_us(10);
    }
    return false; // Timeout
}

bool ism330dhcx_read_accelerometer(i2c_inst_t* i2c_port, uint8_t device_addr, FusionVector* fusion_vector) {
    uint8_t raw_acc_values[6];
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_XL, raw_acc_values, 6)) {
        return false;
    }
    int16_t raw_acc_x = combine_8_bits(raw_acc_values[0], raw_acc_values[1]);
    int16_t raw_acc_y = combine_8_bits(raw_acc_values[2], raw_acc_values[3]);
    int16_t raw_acc_z = combine_8_bits(raw_acc_values[4], raw_acc_values[5]);

    fusion_vector->axis.x = (float)raw_acc_x * active_accel_sensitivity;
    fusion_vector->axis.y = (float)raw_acc_y * active_accel_sensitivity;
    fusion_vector->axis.z = (float)raw_acc_z * active_accel_sensitivity;
    return true;
}

static bool ism330dhcx_read_motion(i2c_inst_t* i2c_port, uint8_t device_addr,
                                   FusionVector* accelerometer, FusionVector* gyroscope) {
    uint8_t raw_values[12];
    if (!ism330dhcx_read_reg(i2c_port, device_addr, OUTX_L_G, raw_values, sizeof(raw_values))) {
        return false;
    }

    const int16_t raw_gyro_x = combine_8_bits(raw_values[0], raw_values[1]);
    const int16_t raw_gyro_y = combine_8_bits(raw_values[2], raw_values[3]);
    const int16_t raw_gyro_z = combine_8_bits(raw_values[4], raw_values[5]);
    const int16_t raw_acc_x = combine_8_bits(raw_values[6], raw_values[7]);
    const int16_t raw_acc_y = combine_8_bits(raw_values[8], raw_values[9]);
    const int16_t raw_acc_z = combine_8_bits(raw_values[10], raw_values[11]);

    gyroscope->axis.x = (float)raw_gyro_x * active_gyro_sensitivity;
    gyroscope->axis.y = (float)raw_gyro_y * active_gyro_sensitivity;
    gyroscope->axis.z = (float)raw_gyro_z * active_gyro_sensitivity;

    accelerometer->axis.x = (float)raw_acc_x * active_accel_sensitivity;
    accelerometer->axis.y = (float)raw_acc_y * active_accel_sensitivity;
    accelerometer->axis.z = (float)raw_acc_z * active_accel_sensitivity;
    return true;
}

bool read_active_sensor_motion_unbiased(uint8_t sensor_index, FusionVector* accelerometer, FusionVector* gyroscope) {
    if ((sensor_index >= active_sensor_count) || (accelerometer == NULL) || (gyroscope == NULL)) {
        return false;
    }

    return ism330dhcx_read_motion(
        active_imus[sensor_index].i2c_port,
        active_imus[sensor_index].device_addr,
        accelerometer,
        gyroscope);
}

void set_active_sensor_gyro_bias(uint8_t sensor_index, FusionVector gyro_bias) {
    if (sensor_index >= active_sensor_count) {
        return;
    }
    active_gyro_biases[sensor_index] = gyro_bias;
}

void read_all_sensors(Sensor* sensors, bool sensor_valid[]) {
    for (uint8_t i = 0; i < active_sensor_count; i++) {
        FusionVector accelerometer;
        FusionVector gyroscope;
        const bool ok = ism330dhcx_read_motion(
            active_imus[i].i2c_port,
            active_imus[i].device_addr,
            &accelerometer,
            &gyroscope);

        if (ok) {
            sensors[i].accelerometer = accelerometer;
            sensors[i].gyroscope = FusionVectorSubtract(gyroscope, active_gyro_biases[i]);
        }
        if (sensor_valid != NULL) {
            sensor_valid[i] = ok;
        }
        sensors[i].timestamp = time_us_64();
    }
}
 
