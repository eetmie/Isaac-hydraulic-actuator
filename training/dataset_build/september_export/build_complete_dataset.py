"""Build a deduplicated actuator corpus from the audited Desktop CSVs.

V1 gyro scale is corrected and attitudes replayed with the pinned V2 Pico
settings. V2 attitudes are preserved. All timing joins are causal; no smoothing
or interpolation is used for training labels or commands. Originals are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import numpy as np
import pandas as pd

ROLES = ('base', 'boom', 'arm', 'bucket')
DT = .01
MAX_AGE = .03
WARMUP_S = 30.
GAP_S = .01
V1_SCALE = .00875 / (250 / 32768)
V2_REV = '995e3018bbe772f782c7e4c7dfaf0210a3629d6e'
MOUNT = 2 * np.arctan2([0, .12057, .0087265, 0], [1, .9927048, .9999619, 1])
CONTRACT = dict(version=1, roles=list(ROLES), velocity_source='aligned_y_child_minus_parent',
                software_filter='none', sampling='latest_complete_packet_not_after_tick',
                bias='firmware_startup_correction_only', carriage_pitch_zero_rad=-.004821687005460262,
                mount_pitch_rad=MOUNT.tolist())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unwrap(values):
    v = np.asarray(values, dtype=np.float64)
    if not np.isfinite(v).all():
        raise ValueError('Nonfinite device clock')
    return v + np.cumsum(np.r_[0, np.diff(v) < -2**31]) * 2**32


def runs(mask):
    edge = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edge[::2], edge[1::2]))


def positions(quats):
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w,x,y,z = np.moveaxis(q, -1, 0)
    pitch = np.arctan2(-2*(x*z-w*y), 1-2*(x*x+y*y)) - MOUNT
    delta = np.diff(pitch, axis=-1)
    relative = np.arctan2(np.sin(delta), np.cos(delta))
    return np.concatenate([relative, pitch[...,:1]-CONTRACT['carriage_pitch_zero_rad']], axis=-1).astype(np.float32)


def rates(gyro):
    g = np.asarray(gyro)[...,1]
    return np.deg2rad(np.concatenate([np.diff(g,axis=-1),g[...,:1]],axis=-1)).astype(np.float32)


def column_arrays(frame):
    arrays = {}
    for col in frame:
        series = frame[col]
        arrays[col] = series.to_numpy() if pd.api.types.is_numeric_dtype(series) else series.fillna('').to_numpy(dtype=str)
        if arrays[col].dtype.kind == 'O':
            raise ValueError(f'Object array in numeric input: {col}')
    return arrays


def make_replayer(pico, provenance):
    """Pin both the V2 settings and the same vendored Fusion implementation."""
    prefix = 'pico_imu_reader/src/imu_reader/'
    files = ['settings/settings.c','ism330dlc/ism330dlc_config.h',
             *[f'Fusion/Fusion/{name}' for name in ('FusionAhrs.c','FusionAhrs.h','FusionAxes.h','FusionMath.h','FusionConvention.h','Fusion.h','FusionOffset.c','FusionOffset.h','FusionCalibration.h','FusionCompass.h')]]
    source = provenance/'pico_v2'
    source.mkdir()
    snapshots = {}
    for name in files:
        result = subprocess.run(['git','-C',str(pico),'show',f'{V2_REV}:{prefix}{name}'],capture_output=True,check=True)
        dst = source/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(result.stdout)
        snapshots[str(dst.relative_to(provenance.parent))] = sha(dst)
    settings_text = (source/'settings/settings.c').read_text()
    settings = {k:float(re.search(r'\.'+k+r'\s*=\s*([\d.]+)',settings_text)[1])
                for k in ('ahrsGain','ahrsAccelRejection','ahrsRecoveryPeriodS','gyroRangeDps','accelRangeG')}
    # The existing driver uses normal running gain and accepts a recorded q seed.
    # It has no FusionOffsetUpdate, matching main.c's already corrected gyro input.
    driver = pico/'native_stack/tools/fusion_replay/replay.c'
    driver_text = driver.read_text()
    driver_text = driver_text.replace('FusionAhrsUpdateNoMagnetometer(&ahrs,', 'FusionAhrsUpdate(&ahrs,')
    driver_text = driver_text.replace('(FusionVector){.axis = {r[4], r[5], r[6]}}, r[0]);',
                                      '(FusionVector){.axis = {r[4], r[5], r[6]}}, FUSION_VECTOR_ZERO, r[0]);')
    (source/'replay.c').write_text(driver_text)
    snapshots[str((source/'replay.c').relative_to(provenance.parent))] = sha(source/'replay.c')
    binary = source/'replay'
    fusion = source/'Fusion/Fusion'
    subprocess.run(['cc','-O2',f'-I{fusion}',str(source/'replay.c'),str(fusion/'FusionAhrs.c'),str(fusion/'FusionOffset.c'),'-lm','-o',str(binary)],check=True)
    settings.update(gyro_saturation_dps=573.44, convention='NWU', source_revision=V2_REV)
    return binary, settings, snapshots


def replay_segment(binary, settings, t, gyro, accel, seed):
    dt = np.r_[0.,np.diff(t)*1e-6]
    if np.any(dt[1:]<=0) or np.any(dt>GAP_S):
        raise ValueError('Replay segment is not contiguous')
    period = np.median(dt[1:]) if len(dt)>1 else .005
    quats, flags = [], []
    for role in range(4):
        rec = np.c_[dt,gyro[:,role],accel[:,role]].astype('<f4')
        header = struct.pack('<i8f',len(t),settings['ahrsGain'],settings['ahrsAccelRejection'],
                             settings['ahrsRecoveryPeriodS']/period,settings['gyro_saturation_dps'],*seed[role])
        result = subprocess.run([str(binary)],input=header+rec.tobytes(),capture_output=True,check=True)
        if len(result.stdout)!=len(t)*17:
            raise ValueError('Truncated Fusion replay output')
        quats.append(np.frombuffer(result.stdout[:len(t)*16],'<f4').reshape(-1,4))
        flags.append(np.frombuffer(result.stdout[len(t)*16:],np.uint8))
    return np.stack(quats,1),np.stack(flags,1)


def raw_columns(r):
    q = np.stack([r[[f'imu_{role}_q{x}' for x in 'wxyz']].to_numpy() for role in ROLES],1)
    g = np.stack([r[[f'imu_{role}_g{x}_dps' for x in 'xyz']].to_numpy() for role in ROLES],1)
    a = np.stack([r[[f'imu_{role}_a{x}_g' for x in 'xyz']].to_numpy() for role in ROLES],1)
    if not all(np.isfinite(v).all() for v in (q,g,a)):
        raise ValueError('Nonfinite raw IMU input')
    if np.any(np.linalg.norm(a,axis=-1)<=.1) or np.any(np.linalg.norm(q,axis=-1)<.5):
        raise ValueError('Missing/invalid sensor stream')
    return q,g,a


def aligned_drive(d, clock, raw_q, gyro, raw_valid):
    """Use the packet explicitly named by the logged pose, never a future packet."""
    state = d.state_imu_ts_us.to_numpy(dtype=float)
    nonnegative = np.isfinite(state)&(state>=0)
    if not nonnegative.any():
        raise ValueError('No valid recorded pose timestamps')
    state_unwrapped = np.full(len(state), np.nan)
    state_unwrapped[nonnegative] = unwrap(state[nonnegative])
    first = np.flatnonzero(nonnegative)[0]
    state_unwrapped += round((clock[0]-state_unwrapped[first])/2**32)*2**32
    ids = np.clip(np.searchsorted(clock,state_unwrapped,side='right')-1,0,len(clock)-1)
    matched = nonnegative&(clock[ids]==state_unwrapped)
    valid = matched&raw_valid[ids]
    for col in ('cmd_age_s','state_age_s'):
        v = d[col].to_numpy()
        valid &= np.isfinite(v)&(v>=0)&(v<=MAX_AGE)
    valid &= d.cmd_stale.to_numpy()==0
    commands = d[[f'combined_cmd_{n}' for n in ('rotate','lift','tilt','scoop')]].to_numpy()
    valid &= np.isfinite(commands).all(axis=1)&(np.abs(commands)<=1.000001).all(axis=1)
    for col in ('pump_on','pump_enabled','write_ok','imu_valid','input_valid','controller_valid'):
        if col in d:
            valid &= d[col].to_numpy()==1
    q = positions(raw_q[ids])
    v = rates(gyro[ids])
    q[~matched] = np.nan
    v[~matched] = np.nan
    return ids,matched,valid,q,v,commands


def training_chunks(d,valid,q,v,commands,raw_ids):
    stamp = d.timestamp.to_numpy()
    if np.any(np.diff(stamp)<=0):
        raise ValueError('Non-increasing control clock')
    connected = (np.diff(stamp)<=MAX_AGE)&(np.diff(d.sample_idx.to_numpy())==1)
    for lo,hi in runs(valid):
        cuts = [lo,*list(lo+1+np.flatnonzero(~connected[lo:hi-1])),hi]
        for start,stop in zip(cuts[:-1],cuts[1:]):
            ticks = np.arange(np.ceil(stamp[start]/DT)*DT,stamp[stop-1]+1e-9,DT)
            if len(ticks)<321:
                continue
            rows = np.searchsorted(stamp,ticks,side='right')-1
            age = ticks-stamp[rows]
            fresh = (d.cmd_age_s.to_numpy()[rows]+age<=MAX_AGE)&(d.state_age_s.to_numpy()[rows]+age<=MAX_AGE)
            for a,b in runs(fresh):
                # Same 10-sample edge trim as training.drive_logs, without SG smoothing.
                a,b = a+10,b-10
                if b-a<300:
                    continue
                take = rows[a:b]
                yield dict(t=ticks[a:b],q=q[take],v=v[take],u=commands[take,1:].astype(np.float32),
                           target_q=q[take],target_v=v[take],u_rotate=commands[take,0].astype(np.float32),
                           source_control_row=take,source_raw_row=raw_ids[take],
                           command_timestamp_s=stamp[take],state_age_s=(d.state_age_s.to_numpy()[take]+age[a:b]))


def build(args):
    catalog = json.loads(args.inventory.read_text())
    output = args.out.resolve()
    partial = output.with_name(output.name+'.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(f'Refusing to replace {output} or {partial}')
    partial.mkdir(parents=True)
    provenance = partial/'provenance'
    provenance.mkdir()
    shutil.copy2(__file__,provenance/'build_complete_dataset.py')
    shutil.copy2(args.inventory,provenance/'input_inventory.json')
    binary,settings,source_hashes = make_replayer(args.pico_source,provenance)
    manifest = dict(format='kaivuri-complete-actuator-v1',created_utc=datetime.now(timezone.utc).isoformat(),
                    sensor_contract=CONTRACT,fusion_target=settings,source_code_sha256=source_hashes,
                    dependencies={name:version(name) for name in ('numpy','pandas','scipy')},
                    roles=list(ROLES),plant_state_names=['boom','arm','bucket','base_pitch'],
                    plant_action_names=['lift','tilt','scoop'],training_hz=100,
                    clipping_policy='Conservative near-saturation flag: accel within four output codes of recorded range; gyro within two codes.',
                    split='unassigned; keep collection_group together when assigning splits',
                    recordings=[],duplicate_aliases=[],output_sha256={},limitations=[
                        'V1 hardware filtering, clipping and original startup calibration cannot be reconstructed exactly.',
                        'Legacy CSVs do not record pump state; no claim that all rows had pump power.',
                        'USB 200 Hz raw stream omits internal 416 Hz AHRS updates; offline replay is approximate.',
                        'Slew is retained separately; planar training q/v/u describes three cylinders and carriage pitch.',
                    ])
    for entry in catalog['records']:
        if 'duplicate_of' in entry:
            manifest['duplicate_aliases'].append({k:entry[k] for k in ('drive','raw','duplicate_of','drive_sha256','raw_sha256')})
            continue
        drive,raw = Path(entry['drive']),Path(entry['raw'])
        if entry.get('error') or entry.get('missing_raw_columns'):
            raise ValueError(f'Incomplete input {drive}')
        if sha(drive)!=entry['drive_sha256'] or sha(raw)!=entry['raw_sha256']:
            raise ValueError(f'Input changed since audit: {drive}')
        v1 = entry['folder']=='hydraulic_data'
        generation = 'v1' if v1 else 'v2'
        rec_id = drive.stem.removeprefix('drive_log_')
        folder = partial/'recordings'/rec_id
        folder.mkdir(parents=True)
        d,r = pd.read_csv(drive),pd.read_csv(raw)
        original_q,original_g,accel = raw_columns(r)
        t = unwrap(r.device_ts_us.to_numpy())
        if np.any(np.diff(t)<=0):
            raise ValueError('Repeated or reset device clock within a raw strip')
        gyro = original_g*(V1_SCALE if v1 else 1.)
        boundary = np.r_[0,np.flatnonzero(np.diff(t)*1e-6>GAP_S)+1,len(t)]
        q = original_q.astype(np.float32)
        flags = np.full((len(t),4),255,np.uint8)  # V2 flags were not recorded.
        settled = np.ones(len(t),bool)
        if v1:
            settled[:] = False
            for lo,hi in zip(boundary[:-1],boundary[1:]):
                q[lo:hi],flags[lo:hi] = replay_segment(binary,settings,t[lo:hi],gyro[lo:hi],accel[lo:hi],original_q[lo])
                settled[lo:hi] = t[lo:hi]-t[lo]>=WARMUP_S*1e6
        accel_limit = 2. if v1 else 4.
        clipping = (np.abs(accel)>=accel_limit-4*accel_limit/32768).any(axis=(1,2))
        gyro_limit = 286.72 if v1 else 573.44
        clipping |= (np.abs(gyro)>=gyro_limit*(1-2/32768)).any(axis=(1,2))
        raw_valid = settled&~clipping
        raw_arrays = column_arrays(r)
        raw_arrays.update(device_ts_us_unwrapped=t.astype(np.uint64),canonical_quat_wxyz=q,
                          canonical_gyro_dps=gyro.astype(np.float32),canonical_accel_g=accel.astype(np.float32),
                          fusion_flags=flags,fusion_settled=settled,clipping=clipping,valid=raw_valid)
        np.savez_compressed(folder/'imu.npz',**raw_arrays)
        ids,matched,valid,cq,cv,commands = aligned_drive(d,t,q,gyro,raw_valid)
        drive_arrays = column_arrays(d)
        drive_arrays.update(canonical_q=cq,canonical_v=cv,canonical_imu_row=np.where(matched,ids,-1),valid=valid)
        np.savez_compressed(folder/'control.npz',**drive_arrays)
        paths=[];usable=0
        for i,chunk in enumerate(training_chunks(d,valid,cq,cv,commands,ids)):
            out=partial/'training/unsplit'/f'{rec_id}_{i:03d}.npz'
            out.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(out,**chunk)
            paths.append(str(out.relative_to(partial)))
            usable+=len(chunk['t'])*DT
        sidecars=[]
        for candidate in [drive.parent/'note.md',drive.with_name(drive.name.replace('drive_log_','excitation_')).with_suffix('.json')]:
            if candidate.exists():
                shutil.copy2(candidate,folder/candidate.name)
                sidecars.append(candidate.name)
        rec=dict(id=rec_id,generation=generation,source_folder=entry['folder'],
                 collection_group=f'{generation}_{rec_id[:8]}',validation_named='validation' in rec_id,
                 input_drive=str(drive),input_raw=str(raw),input_sha256={'drive':entry['drive_sha256'],'raw':entry['raw_sha256']},
                 gyro_scale_applied=V1_SCALE if v1 else 1.,fusion='offline_v2' if v1 else 'recorded_v2_preserved',
                 control_rows=len(d),raw_rows=len(r),duration_s=float(d.timestamp.iloc[-1]-d.timestamp.iloc[0]),
                 raw_gap_count=len(boundary)-2,settling_excluded_raw_rows=int((~settled).sum()),
                 clipped_raw_rows=int(clipping.sum()),matched_control_rows=int(matched.sum()),
                 valid_control_rows=int(valid.sum()),training_usable_s=round(usable,2),training_paths=paths,
                 imu=str((folder/'imu.npz').relative_to(partial)),control=str((folder/'control.npz').relative_to(partial)),
                 sidecars=sidecars)
        write_json(folder/'recording.json',rec)
        manifest['recordings'].append(rec)
        print(f'{rec_id}: {generation}, {len(r)} raw frames, {len(paths)} training chunks, {usable/60:.2f} usable min',flush=True)
    for path in sorted(partial.rglob('*')):
        if path.is_file():
            manifest['output_sha256'][str(path.relative_to(partial))]=sha(path)
    manifest['summary']=dict(unique_recordings=len(manifest['recordings']),duplicate_recordings_not_repeated=len(manifest['duplicate_aliases']),
                             recorded_minutes=sum(r['duration_s'] for r in manifest['recordings'])/60,
                             training_minutes=sum(r['training_usable_s'] for r in manifest['recordings'])/60,
                             raw_frames=sum(r['raw_rows'] for r in manifest['recordings']))
    write_json(partial/'dataset.json',manifest)
    partial.rename(output)
    print(json.dumps(manifest['summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--pico-source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args())
