"""Rebuild the September bundle with audited current Fusion, then add robot gen2."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

import build_complete_dataset as b
import gen2_fusion as gf

HERE=Path(__file__).parent
ROOT=Path('/home/joel/IsaacLab/scripts/Isaac-hydraulic-actuator/dataset/complete_dataset')
STAGE=ROOT.with_name('.complete_dataset.gen2.partial')
NATIVE=HERE/'robot_gen2_source'
PENDING=STAGE/'dataset.pending.json'


def frozen_mask(t,g,a):
    frozen=np.zeros(len(t),bool)
    for role in range(4):
        same=np.r_[False,(np.diff(np.c_[g[:,role],a[:,role]],axis=0)==0).all(axis=1)]
        for lo,hi in b.runs(same):
            lo=max(0,lo-1)
            if t[hi-1]-t[lo]>=100000:frozen[lo:hi]=True
    return frozen


def process(manifest,d,r,info,source_dir,native=None):
    rec=copy.deepcopy(info);rec_id=rec['id'];folder=STAGE/'recordings'/rec_id
    folder.mkdir(parents=True,exist_ok=False)
    original_q=np.stack([r[[f'imu_{role}_q{x}' for x in 'wxyz']].to_numpy() for role in b.ROLES],1)
    g=np.stack([r[[f'imu_{role}_g{x}_dps' for x in 'xyz']].to_numpy() for role in b.ROLES],1)
    a=np.stack([r[[f'imu_{role}_a{x}_g' for x in 'xyz']].to_numpy() for role in b.ROLES],1)
    assert all(np.isfinite(v).all() for v in (original_q,g,a))
    assert np.all(np.linalg.norm(original_q,axis=-1)>.5)
    # Live firmware consumes isolated nonzero low-g readings. Its 0.1 g guard
    # declares loss only after repeated failures, not on an individual sample.
    accel_norm=np.linalg.norm(a,axis=-1)
    missing=(accel_norm==0).any(axis=1)
    low_g=(accel_norm<=.1).any(axis=1)
    t=b.unwrap(r.device_ts_us.to_numpy())
    assert np.all(np.diff(t)>0)
    scale=info['gyro_scale_applied'];gyro=g*scale
    latest=manifest['fusion_target'];replayed=native is None
    cuts=np.r_[0,np.flatnonzero(np.diff(t)>b.GAP_S*1e6)+1,len(t)]
    q=original_q.astype(np.float32);settled=np.ones(len(t),bool);flags=np.full((len(t),4),255,np.uint8)
    executable=STAGE/'provenance/audited_gen2_fusion/replay'
    if replayed:
        settled[:]=False
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            q[lo:hi],flags[lo:hi]=gf.run(executable,latest,t[lo:hi],gyro[lo:hi],a[lo:hi],original_q[lo])
            settled[lo:hi]=(t[lo:hi]-t[lo])>=b.WARMUP_S*1e6
    accel_range=2. if scale!=1. else 4.
    clipping=(np.abs(a)>=accel_range-4*accel_range/32768).any(axis=(1,2))
    clipping|=(np.abs(gyro)>=(286.72 if scale!=1. else 573.44)*(1-2/32768)).any(axis=(1,2))
    frozen=frozen_mask(t,gyro,a);raw_valid=settled&~clipping&~frozen&~missing
    raw=b.column_arrays(r)
    raw.update(device_ts_us_unwrapped=t.astype(np.uint64),canonical_quat_wxyz=q,
               canonical_gyro_dps=gyro.astype(np.float32),canonical_accel_g=a.astype(np.float32),
               fusion_flags=flags,fusion_settled=settled,clipping=clipping,frozen=frozen,
               low_gravity=low_g,missing_sensor_data=missing,valid=raw_valid)
    np.savez_compressed(folder/'imu.npz',**raw)
    ids,matched,valid,cq,cv,commands=b.aligned_drive(d,t,q,gyro,raw_valid)
    training_d=d.copy();availability=None
    if native is not None:
        assert native['complete'] and not native['dry_pwm'] and not native['dropped']
        sensor=native['sensor_contract']
        assert sensor['roles']==list(b.ROLES) and sensor['axis_aligned_y']
        assert np.allclose(sensor['mount_pitch_rad'],b.CONTRACT['mount_pitch_rad'],atol=1e-8,rtol=0)
        assert abs(sensor['carriage_pitch_zero_rad']-b.CONTRACT['carriage_pitch_zero_rad'])<1e-8
        valid&=d.dry_pwm.to_numpy()==0
        availability=(d.post_write_ns.to_numpy()-r.host_read_ns.to_numpy()[ids])/1e9
        valid&=np.isfinite(availability)&(availability>=0)&(availability<=b.MAX_AGE)
        training_d['state_age_s']=np.maximum(d.state_age_s.to_numpy(),availability)
    control=b.column_arrays(d)
    control.update(canonical_q=cq,canonical_v=cv,canonical_imu_row=np.where(matched,ids,-1),
                   canonical_state_age_s=training_d.state_age_s.to_numpy(),valid=valid)
    if availability is not None:control['canonical_availability_age_s']=availability
    np.savez_compressed(folder/'control.npz',**control)
    paths=[];used=0
    for i,chunk in enumerate(b.training_chunks(training_d,valid,cq,cv,commands,ids)):
        dst=STAGE/'training/unsplit'/f'{rec_id}_{i:03d}.npz';dst.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dst,**chunk);paths.append(str(dst.relative_to(STAGE)));used+=len(chunk['t'])
    sidecars=[]
    if native:
        stem=rec_id
        names=[f'session_{stem}.json',f'excitation_{stem}.json',*native.get('attachments',{})]
        for name in names:
            shutil.copy2(source_dir/name,folder/name);sidecars.append(name)
    else:
        for name in info.get('sidecars',[]):
            shutil.copy2(source_dir/name,folder/name);sidecars.append(name)
    rec.update(processing_generation='gen2',fusion='recorded_gen2_preserved' if native else 'offline_current_gen2',
               fusion_settings=dict(gain=latest['ahrsGain'],rejection_deg=latest['ahrsAccelRejection'],recovery_s=latest['ahrsRecoveryPeriodS']),
               raw_rows=len(r),control_rows=len(d),duration_s=float(d.timestamp.iloc[-1]-d.timestamp.iloc[0]),
               raw_gap_count=len(cuts)-2,settling_excluded_raw_rows=int((~settled).sum()),
               clipped_raw_rows=int(clipping.sum()),frozen_raw_rows=int(frozen.sum()),
               transient_low_gravity_rows=int(low_g.sum()),missing_sensor_rows=int(missing.sum()),
               matched_control_rows=int(matched.sum()),valid_control_rows=int(valid.sum()),
               training_usable_s=used*.01,training_paths=paths,sidecars=sidecars,
               imu=str((folder/'imu.npz').relative_to(STAGE)),control=str((folder/'control.npz').relative_to(STAGE)))
    if native:
        rec.update(source_session_id=native['collection_id'],collector=native['collector'],native_manifest_complete=True,
                   collection_group='robot:'+native['collection_id'],pump_state_recorded=True)
        # Real-recording comparison uses the same samples and settings as the build.
        replay=[]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            replay.append(gf.run(executable,latest,t[lo:hi],gyro[lo:hi],a[lo:hi],original_q[lo])[0])
        compared=b.positions(np.concatenate(replay))-b.positions(original_q)
        keep=np.ones(len(t),bool)
        for lo,hi in zip(cuts[:-1],cuts[1:]):keep[lo:hi]=(t[lo:hi]-t[lo])>=30e6
        error=np.degrees(np.abs(np.arctan2(np.sin(compared),np.cos(compared))))[keep]
        rec['offline_vs_recorded_p99_deg']=np.percentile(error,99,axis=0).tolist()
    b.write_json(folder/'recording.json',rec);manifest['recordings'].append(rec)
    b.write_json(PENDING,manifest)
    print(rec_id,rec['fusion'],'training min',round(used*.01/60,3),flush=True)


def september():
    assert not STAGE.exists()
    old=json.loads((ROOT/'dataset.json').read_text())
    for name,expected in old['output_sha256'].items():assert b.sha(ROOT/name)==expected,name
    STAGE.mkdir();provenance=STAGE/'provenance';provenance.mkdir()
    shutil.copytree(ROOT/'provenance',provenance/'september_export')
    shutil.copytree(HERE/'audited_gen2_fusion',provenance/'audited_gen2_fusion')
    for name in ['expand_gen2_dataset.py','gen2_fusion.py','build_complete_dataset.py']:
        shutil.copy2(HERE/name,provenance/name)
    shutil.copy2(ROOT/'dataset.json',provenance/'september_source_manifest.json')
    latest=json.loads((provenance/'audited_gen2_fusion/settings.json').read_text());assert latest['ahrsAccelRejection']==5
    manifest=copy.deepcopy(old)
    manifest.update(format='kaivuri-complete-actuator-v1',normalized_generation='gen2',fusion_target=latest,
                    previous_export_summary=old['summary'],recordings=[],output_sha256={},source_code_sha256={})
    catalog=json.loads((ROOT/'provenance/input_inventory.json').read_text())
    metadata={entry['drive']:entry for entry in catalog['records'] if 'duplicate_of' not in entry}
    for rec in old['recordings']:
        meta=metadata[rec['input_drive']]
        with np.load(ROOT/rec['imu'],allow_pickle=False) as archive:
            r=pd.DataFrame({name:archive[name] for name in meta['raw_columns']})
        with np.load(ROOT/rec['control'],allow_pickle=False) as archive:
            d=pd.DataFrame({name:archive[name] for name in meta['drive_columns']})
        process(manifest,d,r,rec,(ROOT/rec['imu']).parent)


def finish():
    manifest=json.loads(PENDING.read_text());assert 22<=len(manifest['recordings'])<=33
    sys.path.insert(0,str(ROOT.parents[1]))
    from training.drive_logs import native_manifest
    seen={(r['input_sha256']['drive'],r['input_sha256']['raw']) for r in manifest['recordings']}
    sessions=sorted(NATIVE.glob('session_*.json'));assert len(sessions)==11
    for session in sessions:
        stem=session.stem.removeprefix('session_');drive=NATIVE/f'drive_log_{stem}.csv';raw=NATIVE/f'imu_raw_{stem}.csv'
        native=native_manifest(drive,required=True)
        dh,rh=b.sha(drive),b.sha(raw)
        if (dh,rh) in seen:
            assert any(rec['id']==stem for rec in manifest['recordings'])
            continue
        seen.add((dh,rh))
        d,r=pd.read_csv(drive),pd.read_csv(raw)
        rec=dict(id=stem,generation='robot_gen2',source_folder='robot_2026-10-22',input_drive='pikkumasi2tb:'+str(drive).replace(str(NATIVE),'/home/joel/actuator-data/2026-10-22'),
                 input_raw='pikkumasi2tb:/home/joel/actuator-data/2026-10-22/'+raw.name,
                 input_sha256={'drive':dh,'raw':rh},gyro_scale_applied=1.,validation_named=False)
        process(manifest,d,r,rec,NATIVE,native)
    manifest['summary']=dict(unique_recordings=len(manifest['recordings']),duplicate_recordings_not_repeated=len(manifest['duplicate_aliases']),
                              recorded_minutes=sum(r['duration_s'] for r in manifest['recordings'])/60,
                              training_minutes=sum(r['training_usable_s'] for r in manifest['recordings'])/60,
                              raw_frames=sum(r['raw_rows'] for r in manifest['recordings']))
    manifest['limitations']=[*manifest['limitations'],'All September attitudes were replayed to 5-degree current gen2 settings; robot gen2 native attitudes retained.']
    PENDING.unlink()
    for path in STAGE.rglob('*'):
        if path.is_file():manifest['output_sha256'][str(path.relative_to(STAGE))]=b.sha(path)
    b.write_json(STAGE/'dataset.json',manifest)
    print(json.dumps(manifest['summary']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['september','finish']);a=p.parse_args()
    september() if a.phase=='september' else finish()
