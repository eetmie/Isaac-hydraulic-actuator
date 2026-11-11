// Replay the vendored x-io Fusion AHRS over recorded gyro (dps) / accel (g).
// stdin: int32 n; float gain, rejection_deg, recovery_updates, gyro_range; float q0[4]; n x float[7] (dt, gx, gy, gz, ax, ay, az)
// stdout: n x float[4] quaternion wxyz, then n x uint8 flags (bit0 accel ignored, bit1 accel recovery)
#include <stdio.h>
#include <stdlib.h>
#include "Fusion.h"
int main(void) {
    int n; float p[4], q0[4];
    if (fread(&n, 4, 1, stdin) != 1 || fread(p, 4, 4, stdin) != 4 || fread(q0, 4, 4, stdin) != 4) return 1;
    float *rec = malloc(sizeof(float) * 7 * n); if (fread(rec, 4, 7 * n, stdin) != (size_t)(7 * n)) return 2;
    FusionAhrs ahrs; FusionAhrsInitialise(&ahrs);
    FusionAhrsSettings s = {.convention = FusionConventionNwu, .gain = p[0], .gyroscopeRange = p[3],
                            .accelerationRejection = p[1], .magneticRejection = 0, .recoveryTriggerPeriod = (unsigned)p[2]};
    FusionAhrsSetSettings(&ahrs, &s);
    FusionAhrsSetQuaternion(&ahrs, (FusionQuaternion){.element = {q0[0], q0[1], q0[2], q0[3]}});
    ahrs.initialising = false; ahrs.rampedGain = p[0];
    float *out = malloc(sizeof(float) * 4 * n); unsigned char *fl = malloc(n);
    for (int i = 0; i < n; i++) {
        float *r = rec + 7 * i;
        FusionAhrsUpdateNoMagnetometer(&ahrs, (FusionVector){.axis = {r[1], r[2], r[3]}}, (FusionVector){.axis = {r[4], r[5], r[6]}}, r[0]);
        FusionQuaternion q = FusionAhrsGetQuaternion(&ahrs);
        for (int k = 0; k < 4; k++) out[4 * i + k] = q.array[k];
        FusionAhrsInternalStates st = FusionAhrsGetInternalStates(&ahrs); FusionAhrsFlags f = FusionAhrsGetFlags(&ahrs);
        fl[i] = (st.accelerometerIgnored ? 1 : 0) | (f.accelerationRecovery ? 2 : 0);
    }
    fwrite(out, 4, 4 * n, stdout); fwrite(fl, 1, n, stdout);
    return 0;
}
