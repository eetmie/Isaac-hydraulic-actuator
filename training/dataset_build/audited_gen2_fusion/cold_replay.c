#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "InitFusion.h"
#include "settings.h"
static inline FusionQuaternion enforce_quaternion_continuity(FusionQuaternion q, FusionQuaternion q_ref) {
    // Choose hemisphere (q or -q) closest to reference quaternion
    float dot = q.element.w * q_ref.element.w +
                q.element.x * q_ref.element.x +
                q.element.y * q_ref.element.y +
                q.element.z * q_ref.element.z;

    if (dot < 0.0f) {
        q.element.w = -q.element.w;
        q.element.x = -q.element.x;
        q.element.y = -q.element.y;
        q.element.z = -q.element.z;
    }
    return q;
}
static inline float clamp_delta_time(float delta_time_s, float nominal_s) {
    if (!isfinite(delta_time_s) || (delta_time_s <= 0.0f)) {
        return nominal_s;
    }
    if (delta_time_s < nominal_s * 0.5f) {
        return nominal_s * 0.5f;
    }
    if (delta_time_s > nominal_s * 2.0f) {
        return nominal_s * 2.0f;
    }
    return delta_time_s;
}
static int fusion_vector_is_finite(FusionVector v) { return isfinite(v.axis.x)&&isfinite(v.axis.y)&&isfinite(v.axis.z); }
static int fusion_quaternion_is_finite(FusionQuaternion q) { return isfinite(q.element.w)&&isfinite(q.element.x)&&isfinite(q.element.y)&&isfinite(q.element.z); }
static float tilt_pitch_deg_from_gravity(FusionVector g) { return atan2f(-g.axis.x,g.axis.z)*180.0f/3.14159265358979323846f; }
static float tilt_roll_deg_from_gravity(FusionVector g) { return atan2f(g.axis.y,g.axis.z)*180.0f/3.14159265358979323846f; }
static void write_sensor_output(float *out, FusionQuaternion q, FusionVector g, FusionVector a) { (void)out; (void)q; (void)g; (void)a; }

static FusionQuaternion step(Sensor *sensor,float *r,float nominal) {
    FusionVector g={.axis={r[1],r[2],r[3]}},a={.axis={r[4],r[5],r[6]}};
    if (!fusion_vector_is_finite(g)||!fusion_vector_is_finite(a)) return sensor->previousQuaternion;
    FusionAhrsUpdate(&sensor->ahrs,g,a,FUSION_VECTOR_ZERO,clamp_delta_time(r[0],nominal));
    FusionQuaternion q=FusionAhrsGetQuaternion(&sensor->ahrs);
    if (!fusion_quaternion_is_finite(q)) { FusionAhrsReset(&sensor->ahrs); q=FUSION_IDENTITY_QUATERNION; }
    q=enforce_quaternion_continuity(q,sensor->previousQuaternion);
    sensor->previousQuaternion=q;
    return q;
}

int main(void) {
    int n;float p[4],q0[4];
    if(fread(&n,4,1,stdin)!=1||n<=0||fread(p,4,4,stdin)!=4||fread(q0,4,4,stdin)!=4) return 1;
    float *rec=malloc(sizeof(float)*7*n),*out=malloc(sizeof(float)*4*n);uint8_t *flags=malloc(n);
    if(!rec||!out||!flags||fread(rec,4,7*n,stdin)!=(size_t)(7*n)) return 2;
    Sensor sensor;initialize_sensors_values(&sensor,1);initialize_calibrations(&sensor,1);
    imu_reader_settings.ahrsGain=p[0];imu_reader_settings.ahrsAccelRejection=p[1];
    imu_reader_settings.gyroSaturationDps=p[3];
    imu_reader_settings.ahrsRateHz=(int)lroundf(p[2]/imu_reader_settings.ahrsRecoveryPeriodS);
    initialize_algos(&sensor,1);
    FusionQuaternion seed={.element={q0[0],q0[1],q0[2],q0[3]}};
    FusionAhrsSetQuaternion(&sensor.ahrs,seed);sensor.previousQuaternion=seed;
    /* retain firmware cold-start gain ramp */
    float nominal=1.0f/imu_reader_settings.ahrsRateHz;
    for(int i=0;i<n;i++) {
        FusionQuaternion q=step(&sensor,rec+7*i,nominal);
        for(int k=0;k<4;k++)out[4*i+k]=q.array[k];
        FusionAhrsInternalStates st=FusionAhrsGetInternalStates(&sensor.ahrs);
        FusionAhrsFlags fl=FusionAhrsGetFlags(&sensor.ahrs);
        flags[i]=(st.accelerometerIgnored?1:0)|(fl.accelerationRecovery?2:0);
    }
    fwrite(out,4,4*n,stdout);fwrite(flags,1,n,stdout);return 0;
}
