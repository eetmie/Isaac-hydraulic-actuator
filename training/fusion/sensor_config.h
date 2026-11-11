#ifndef ISM330DLC_CONFIG_H
#define ISM330DLC_CONFIG_H

#include <stdint.h>

// What full scales this part supports, and how each is encoded in its CTRL
// register. This header describes the sensor's capabilities only — the ranges
// actually requested live next to every other tunable in settings.c, so gyro
// and accel are configured the same way instead of one being a #define here and
// the other a runtime setting.
//
// ODR is deliberately absent: it is derived from the requested rate by
// select_odr_bits() in ism330dlc.c rather than being fixed per sensor.
//
// Both tables must stay sorted ascending by scale — selection walks them in
// order and takes the first entry at or above the requested full scale.

typedef struct imu_full_scale_t {
    float   scale;       // nominal full-scale magnitude (dps for gyro, g for accel)
    uint8_t mask;        // CTRL register bits selecting that full scale
    float   sensitivity; // dps/LSB for gyro, g/LSB for accel (ST DS13012 Table 2)
} imu_full_scale_t;

// CTRL2_G: FS_125 is bit 1, FS_G[1:0] is bits [3:2] (00=250, 01=500, 10=1000,
// 11=2000 dps).
//
// The gyro's nominal full scale is NOT scale/32768. ST specifies sensitivity
// directly (4.375 mdps/LSB at +/-125 dps, doubling per range), which puts the
// output code's true saturation 14.688% above the nominal name — 286.7 dps at
// the "250 dps" setting. Converting with scale/32768 under-reports every
// angular rate by that factor, which is exactly what the previously flashed
// firmware did.
#define ISM330_GYRO_FULL_SCALES { \
    {  125.0f, 0x02, 0.004375f }, \
    {  250.0f, 0x00, 0.008750f }, \
    {  500.0f, 0x04, 0.017500f }, \
    { 1000.0f, 0x08, 0.035000f }, \
    { 2000.0f, 0x0C, 0.070000f }, \
}

// CTRL1_XL: FS_XL[1:0] is bits [3:2]. This encoding is not monotonic —
// 00=2 g, 01=16 g, 10=4 g, 11=8 g — so 16 g is 0x04 and 8 g is 0x0C.
//
// Unlike the gyro, the accelerometer's sensitivity IS scale/32768 exactly;
// ST's 0.061/0.122/0.244/0.488 mg/LSB are those values rounded for print.
// The exact constants are used here so the conversion stays bit-for-bit what
// it has always been and no host-side rescaling of accel data is implied.
#define ISM330_ACCEL_FULL_SCALES {          \
    {  2.0f, 0x00, 0.00006103515625f },     \
    {  4.0f, 0x08, 0.0001220703125f  },     \
    {  8.0f, 0x0C, 0.000244140625f   },     \
    { 16.0f, 0x04, 0.00048828125f    },     \
}

#endif //ISM330DLC_CONFIG_H
