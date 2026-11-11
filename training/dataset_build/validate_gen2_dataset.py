"""Validate the built corpus against source-indexed controls and raw IMU data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def validate(root):
    manifest=json.loads((root/'dataset.json').read_text())
    assert manifest['summary']['unique_recordings']==33
    assert manifest['summary']['duplicate_recordings_not_repeated']==8
    assert manifest['fusion_target']['ahrsAccelRejection']==5
    assert manifest['normalized_generation']=='gen2'
    for name,expected in manifest['output_sha256'].items():
        assert digest(root/name)==expected,name
    count=rows=flagged=0;minimum_age=float('inf');max_age=0.;peak_gyro=0.
    per_generation={}
    for rec in manifest['recordings']:
        assert rec['processing_generation']=='gen2'
        assert rec['fusion_settings']=={'gain':.5,'rejection_deg':5.,'recovery_s':2.}
        raw=np.load(root/rec['imu'],allow_pickle=False)
        control=np.load(root/rec['control'],allow_pickle=False)
        assert len(raw['device_ts_us'])==rec['raw_rows']
        assert len(control['timestamp'])==rec['control_rows']
        assert np.all(np.diff(raw['device_ts_us_unwrapped'])>0)
        quat=raw['canonical_quat_wxyz'];gyro=raw['canonical_gyro_dps'];accel=raw['canonical_accel_g']
        assert quat.shape==(rec['raw_rows'],4,4)
        assert gyro.shape==accel.shape==(rec['raw_rows'],4,3)
        assert all(np.isfinite(a).all() for a in (quat,gyro,accel))
        peak_gyro=max(peak_gyro,float(np.abs(gyro).max()))
        assert np.max(np.abs(np.linalg.norm(quat,axis=-1)-1))<.002
        original_q=np.stack([np.stack([raw[f'imu_{r}_q{x}'] for x in 'wxyz'],-1) for r in manifest['roles']],1)
        original_g=np.stack([np.stack([raw[f'imu_{r}_g{x}_dps'] for x in 'xyz'],-1) for r in manifest['roles']],1)
        assert np.allclose(gyro,original_g*rec['gyro_scale_applied'],atol=2e-5)
        if rec['fusion']=='recorded_gen2_preserved':
            assert np.array_equal(quat,original_q.astype(np.float32))
            assert np.all(raw['fusion_flags']==255)
        else:
            assert np.all(raw['fusion_flags']<4)
        assert not np.any(raw['valid']&raw['clipping'])
        flagged+=int(raw['clipping'].sum())
        used=0
        for name in rec['training_paths']:
            chunk=np.load(root/name,allow_pickle=False)
            t=chunk['t'];n=len(t);ci=chunk['source_control_row'];ri=chunk['source_raw_row']
            assert n>=300
            assert np.allclose(np.diff(t),.01,atol=1e-9)
            assert chunk['q'].shape==chunk['v'].shape==(n,4)
            assert chunk['u'].shape==(n,3)
            assert all(np.isfinite(chunk[k]).all() for k in ['q','v','u','target_q','target_v'])
            assert np.array_equal(chunk['q'],chunk['target_q'])
            assert np.array_equal(chunk['v'],chunk['target_v'])
            assert np.all(control['valid'][ci]) and np.all(raw['valid'][ri])
            assert not np.any(raw['frozen'][ri])
            if rec['generation']=='robot_gen2':
                assert np.all(control['pump_on'][ci]==1)
                assert np.all(control['write_ok'][ci]==1)
                assert np.all(control['dry_pwm'][ci]==0)
                availability=(control['post_write_ns'][ci]-raw['host_read_ns'][ri])/1e9
                assert np.all(availability>=0)
                assert np.all(availability+t-control['timestamp'][ci]<=.030000001)
            assert np.array_equal(control['canonical_imu_row'][ci],ri)
            assert np.array_equal(control['canonical_q'][ci],chunk['q'])
            assert np.array_equal(control['canonical_v'][ci],chunk['v'])
            assert np.allclose(chunk['u'],np.stack([control[f'combined_cmd_{k}'][ci] for k in ('lift','tilt','scoop')],-1),atol=1e-7)
            assert np.allclose(chunk['u_rotate'],control['combined_cmd_rotate'][ci],atol=1e-7)
            assert np.array_equal(chunk['command_timestamp_s'],control['timestamp'][ci])
            command_age=t-chunk['command_timestamp_s']
            assert np.all(command_age>=-1e-12)
            assert np.all(command_age+control['cmd_age_s'][ci]<=.030000001)
            assert np.all(chunk['state_age_s']<=.030000001)
            assert np.all(np.abs(chunk['u'])<=1.000001)
            expected_v=np.deg2rad(np.c_[np.diff(gyro[ri,:,1],axis=1),gyro[ri,0,1]])
            assert np.allclose(expected_v,chunk['v'],atol=1e-6)
            minimum_age=min(minimum_age,float(command_age.min()))
            max_age=max(max_age,float(command_age.max()))
            used+=n
            count+=1
        assert abs(used*.01-rec['training_usable_s'])<.00001
        rows+=used
        per_generation.setdefault(rec['generation'],dict(recordings=0,training_rows=0))
        per_generation[rec['generation']]['recordings']+=1
        per_generation[rec['generation']]['training_rows']+=used
        raw.close();control.close()
        print('VERIFIED',rec['id'],used,'training rows',flush=True)
    assert flagged==6,'Expected six conservative V1 near-saturation flags'
    assert abs(rows*.01/60-manifest['summary']['training_minutes'])<1e-8
    report=dict(passed=True,recordings=len(manifest['recordings']),chunks=count,training_rows=rows,
                training_minutes=rows*.01/60,near_saturation_frames=flagged,
                earliest_command_hold_age_s=minimum_age,latest_command_hold_age_s=max_age,
                output_files_hash_verified=len(manifest['output_sha256']),per_generation=per_generation,
                peak_canonical_gyro_dps=peak_gyro,
                checks=['Every output SHA-256','Complete finite synchronized raw channels',
                        'V1 gyro unit correction','V2 fused quaternions preserved',
                        '100 Hz contiguous training clocks','No future command hold',
                        'Fresh state and command availability','Recorded-packet identity',
                        'No unsettled or saturation-flagged training samples',
                        'Unsmooth same-packet relative gyro velocity targets'])
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    report=validate(args.root)
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
