"""Host replay wrapper around the current robot's actual Pico Fusion sources.

No AHRS math is implemented here. The reference executable compiles the actual
firmware initialization plus the calibrated update block extracted from main.c.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function(text, name):
    start=text.index('static inline ',text.index(name)-40)
    brace=text.index('{',start);depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False)
    fusion=out/'Fusion';fusion.mkdir()
    src=source/'src/imu_reader'
    for p in (src/'Fusion/Fusion').glob('*'):
        if p.suffix in ('.c','.h'):shutil.copy2(p,fusion/p.name)
    for name in ('settings.c','settings.h'):
        shutil.copy2(src/'settings'/name,out/name)
    shutil.copy2(src/'imu_reader.h',out/'imu_reader.h')
    shutil.copy2(source/'main.c',out/'firmware_main.c')
    shutil.copy2(src/'ism330dlc/ism330dlc.c',out/'sensor_driver.c')
    shutil.copy2(src/'ism330dlc/ism330dlc_config.h',out/'sensor_config.h')
    # Headers stub platform declarations only. No device is accessed or emulated.
    (out/'pico').mkdir()
    (out/'pico/time.h').write_text('#include <stdint.h>\nstatic inline uint64_t time_us_64(void) { return 0; }\n')
    (out/'i2c_helpers.h').write_text('#include <stdint.h>\n')
    main=(out/'firmware_main.c').read_text()
    helpers=function(main,'enforce_quaternion_continuity(')+'\n'+function(main,'clamp_delta_time(')
    begin=main.index('// Apply calibration (identity by default)',main.index('static void update_loop(float period_ms, float sensors_data[][FLOATS_PER_SENSOR], Sensor* sensors, uint8_t sensor_count) {'))
    end=main.index('// Publish the calibrated, bias-corrected pair',begin)
    firmware_block=main[begin:end]
    common='''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "InitFusion.h"
#include "settings.h"
'''+helpers+'''
static int fusion_vector_is_finite(FusionVector v) { return isfinite(v.axis.x)&&isfinite(v.axis.y)&&isfinite(v.axis.z); }
static int fusion_quaternion_is_finite(FusionQuaternion q) { return isfinite(q.element.w)&&isfinite(q.element.x)&&isfinite(q.element.y)&&isfinite(q.element.z); }
static float tilt_pitch_deg_from_gravity(FusionVector g) { return atan2f(-g.axis.x,g.axis.z)*180.0f/3.14159265358979323846f; }
static float tilt_roll_deg_from_gravity(FusionVector g) { return atan2f(g.axis.y,g.axis.z)*180.0f/3.14159265358979323846f; }
static void write_sensor_output(float *out, FusionQuaternion q, FusionVector g, FusionVector a) { (void)out; (void)q; (void)g; (void)a; }
'''
    reference='''
static FusionQuaternion step(Sensor *sensors,float *r,float nominal_delta_time_s) {
    const int i=0; float sensors_data[1][10]; FusionQuaternion output=sensors[0].previousQuaternion;
    sensors[0].gyroscope=(FusionVector){.axis={r[1],r[2],r[3]}};
    sensors[0].accelerometer=(FusionVector){.axis={r[4],r[5],r[6]}};
    sensors[0].timestamp+=(uint64_t)llroundf(r[0]*1e6f);
    do {
'''+firmware_block+'''
    output=quat;
    } while (0);
    return output;
}
'''
    wrapper='''
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
'''
    loop='''
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
    sensor.ahrs.initialising=false;sensor.ahrs.rampedGain=p[0];
    float nominal=1.0f/imu_reader_settings.ahrsRateHz;
    for(int i=0;i<n;i++) {
        FusionQuaternion q=i==0?seed:step(&sensor,rec+7*i,nominal);
        for(int k=0;k<4;k++)out[4*i+k]=q.array[k];
        FusionAhrsInternalStates st=FusionAhrsGetInternalStates(&sensor.ahrs);
        FusionAhrsFlags fl=FusionAhrsGetFlags(&sensor.ahrs);
        flags[i]=(st.accelerometerIgnored?1:0)|(fl.accelerationRecovery?2:0);
    }
    fwrite(out,4,4*n,stdout);fwrite(flags,1,n,stdout);return 0;
}
'''
    for name,body in [('replay',wrapper),('firmware_reference',reference)]:
        path=out/(name+'.c');path.write_text(common+body+loop)
        subprocess.run(['cc','-O2',f'-I{fusion}',f'-I{out}',str(path),str(out/'settings.c'),str(fusion/'InitFusion.c'),str(fusion/'FusionAhrs.c'),str(fusion/'FusionOffset.c'),'-lm','-o',str(out/name)],check=True)
    txt=(out/'settings.c').read_text()
    settings={k:float(re.search(r'\.'+k+r'\s*=\s*([\d.]+)',txt)[1]) for k in ('ahrsGain','ahrsAccelRejection','ahrsRecoveryPeriodS','gyroRangeDps','accelRangeG')}
    settings.update(gyro_saturation_dps=573.44,convention='NWU',source='robot Nov-2 dropout build source; same AHRS/settings as current robot repository')
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    return out/'replay',settings


def run(binary,settings,t,gyro,accel,seed):
    period=np.median(np.diff(t))*1e-6 if len(t)>1 else .005
    count=round(settings['ahrsRecoveryPeriodS']/period)
    dt=np.r_[0.,np.diff(t)*1e-6]
    outputs=[];flags=[]
    for role in range(4):
        record=np.c_[dt,gyro[:,role],accel[:,role]].astype('<f4')
        header=struct.pack('<i8f',len(t),settings['ahrsGain'],settings['ahrsAccelRejection'],count,settings['gyro_saturation_dps'],*seed[role])
        result=subprocess.run([str(binary)],input=header+record.tobytes(),capture_output=True,check=True).stdout
        assert len(result)==len(t)*17
        outputs.append(np.frombuffer(result[:len(t)*16],'<f4').reshape(-1,4));flags.append(np.frombuffer(result[len(t)*16:],np.uint8))
    return np.stack(outputs,1),np.stack(flags,1)


def audit(out,settings):
    report=[]
    for rate in (200,416):
        n=rate*12;t=np.rint(np.arange(n)*1e6/rate)
        for case in ('still','slow_yaw','shake_recovery','gyro_saturation'):
            g=np.zeros((n,4,3));a=np.tile([0.,0.,1.],(n,4,1));seed=np.tile([1.,0.,0.,0.],(4,1))
            if case=='slow_yaw':g[:,:,2]=.5
            if case=='shake_recovery':a[rate:rate*5]=[-np.sin(.35),0,np.cos(.35)]
            if case=='gyro_saturation':
                g[:,:,2]=30.;g[rate*2,:,2]=600.
            q,flags=run(out/'replay',settings,t,g,a,seed)
            reference,rf=run(out/'firmware_reference',settings,t,g,a,seed)
            error=float(np.max(np.abs(q-reference)))
            assert error<2e-6,(rate,case,error)
            assert np.array_equal(flags,rf),(rate,case,'recovery flags differ')
            if case=='slow_yaw':
                yaw=np.unwrap(2*np.arctan2(q[:,0,3],q[:,0,0]));assert abs(np.degrees(yaw[-1])-(n-1)*.5/rate)<.01
            report.append(dict(rate_hz=rate,case=case,max_quaternion_component_difference=error,identical_recovery_flags=True))
    return report


if __name__=='__main__':
    base=Path(__file__).parent
    out=base/'audited_gen2_fusion'
    binary,settings=prepare(base/'robot_current_source/pico_imu_reader',out)
    result=dict(settings=settings,synthetic_comparisons=audit(out,settings),source_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
                hardware_identity_limit='Retained Nov-2 source/build evidence; no Pico flash readback or USB reset was performed.')
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['synthetic_comparisons'],indent=2))
