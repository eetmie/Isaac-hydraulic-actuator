"""Inventory original Desktop actuator recordings without modifying them."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path('/home/joel/Desktop/dataset')
OUT = Path(__file__).with_name('desktop_input_inventory.json')
ROLES = ('base', 'boom', 'arm', 'bucket')
records, hashes, cache = [], {}, {}


def digest(path):
    before = path.stat()
    with path.open('rb') as f:
        result = hashlib.file_digest(f, 'sha256').hexdigest()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'File is changing: {path}'
    hashes[str(path)] = result
    return result


def clock(values):
    values = np.asarray(values, np.float64)
    return values + np.cumsum(np.r_[0, np.diff(values) < -2**31]) * 2**32


def stamp(path):
    value = re.search(r'(\d{8}_\d{6})', path.name)[1]
    return datetime.strptime(value, '%Y%m%d_%H%M%S')


def spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


for folder in ('hydraulic_data', 'hydraulic_data_2', 'V3'):
    used_raw = set()
    raws = sorted((ROOT / folder).glob('imu_raw*.csv'))
    for drive in sorted((ROOT / folder).glob('drive_log*.csv')):
        candidates = sorted(raws, key=lambda p: abs((stamp(p) - stamp(drive)).total_seconds()))
        raw = candidates[0] if candidates and abs((stamp(candidates[0]) - stamp(drive)).total_seconds()) <= 15 else None
        rec = {'folder': folder, 'drive': str(drive), 'drive_sha256': digest(drive)}
        if raw is None:
            rec['error'] = 'No raw filename within 15 seconds'
            records.append(rec)
            continue
        assert raw not in used_raw, f'Raw file paired twice: {raw}'
        used_raw.add(raw)
        rec.update(raw=str(raw), raw_sha256=digest(raw), filename_offset_s=(stamp(raw)-stamp(drive)).total_seconds())
        key = (rec['drive_sha256'], rec['raw_sha256'])
        if key in cache:
            rec.update(duplicate_of=cache[key])
            records.append(rec)
            print('DUPLICATE', drive.name, 'of', rec['duplicate_of'], flush=True)
            continue
        cache[key] = str(drive)
        d, r = pd.read_csv(drive), pd.read_csv(raw)
        required = [f'imu_{role}_{name}' for role in ROLES for name in ('qw','qx','qy','qz','gx_dps','gy_dps','gz_dps','ax_g','ay_g','az_g')]
        missing = sorted(set(required)-set(r.columns))
        rec.update(drive_rows=len(d), raw_rows=len(r), missing_raw_columns=missing,
                   drive_duration_s=float(d.timestamp.iloc[-1]-d.timestamp.iloc[0]),
                   drive_columns=list(d.columns), raw_columns=list(r.columns), sensors={})
        if missing:
            records.append(rec)
            print('MISSING',drive.name,missing,flush=True)
            continue
        t = clock(r.device_ts_us)
        dt = np.diff(t)
        rec.update(raw_duration_s=float((t[-1]-t[0])*1e-6), nonforward_raw_intervals=int((dt<=0).sum()),
                   raw_intervals_over_10ms=int((dt>10000).sum()), max_raw_interval_ms=float(dt.max()/1000))
        complete = np.ones(len(r), bool)
        for role in ROLES:
            q = r[[f'imu_{role}_q{x}' for x in 'wxyz']].to_numpy()
            g = r[[f'imu_{role}_g{x}_dps' for x in 'xyz']].to_numpy()
            a = r[[f'imu_{role}_a{x}_g' for x in 'xyz']].to_numpy()
            finite = np.isfinite(np.c_[q,g,a]).all(1)
            norm = np.linalg.norm(a, axis=1)
            qnorm = np.linalg.norm(q, axis=1)
            basic = finite & (norm>.1) & (qnorm>.5) & (qnorm<1.5)
            identical = np.r_[False, (np.diff(np.c_[g,a], axis=0)==0).all(1)]
            frozen = np.zeros(len(r), bool)
            longest = 0.
            for lo,hi in spans(identical):
                start=max(0,lo-1)
                duration=(t[hi-1]-t[start])*1e-6
                longest=max(longest,duration)
                if duration>=.1:frozen[start:hi]=True
            changes = np.diff(g,axis=0).reshape(-1)
            changes=changes[np.isfinite(changes)&(np.abs(changes)>.04)]
            changes=changes[::max(1,len(changes)//100000)]
            lattice = {str(step):float(np.median(np.abs(changes/step-np.round(changes/step)))) for step in (.00762939453125,.0152587890625,.00875,.0175)} if len(changes) else {}
            complete &= basic & ~frozen
            info={'nonfinite_rows':int((~finite).sum()), 'accel_norm_under_0_1g_rows':int((norm<=.1).sum()),
                  'bad_quaternion_rows':int(((qnorm<=.5)|(qnorm>=1.5)).sum()),
                  'frozen_rows_100ms':int(frozen.sum()),'longest_identical_gyro_accel_s':longest,
                  'usable_raw_rows':int((basic&~frozen).sum()),
                  'accel_abs_peak_g':float(np.nanmax(np.abs(a))), 'gyro_abs_peak_dps':float(np.nanmax(np.abs(g))),
                  'gyro_lattice_median_residual':lattice,
                  'accel_at_2g_rows':int((np.abs(a)>=1.9997).any(1).sum()),
                  'accel_at_4g_rows':int((np.abs(a)>=3.9995).any(1).sum())}
            rec['sensors'][role]=info
        state=clock(d.state_imu_ts_us)
        valid=np.isfinite(state)&(d.state_imu_ts_us.to_numpy()>=0)
        state+=round((t[0]-state[np.flatnonzero(valid)[0]])/2**32)*2**32
        ids=np.clip(np.searchsorted(t,state,side='right')-1,0,len(t)-1)
        joined=(t[ids]==state)&valid
        live=joined&complete[ids]
        for col in ['cmd_age_s','state_age_s']:
            if col in d:live&=np.isfinite(d[col])&(d[col]>=0)&(d[col]<=.03)
        if 'cmd_stale' in d:live&=d.cmd_stale==0
        live&=np.isfinite(d[[f'combined_cmd_{n}' for n in ('rotate','lift','tilt','scoop')]]).all(1)
        rec.update(raw_complete_rows=int(complete.sum()),state_packets_found=int(joined.sum()),
                   valid_control_rows_before_warmup=int(live.sum()),
                   missing_pump_state='pump_on' not in d and 'pump_enabled' not in d,
                   max_control_interval_ms=float(np.diff(d.timestamp).max()*1000),
                   sample_index_gaps=int((np.diff(d.sample_idx)!=1).sum()))
        records.append(rec)
        print(json.dumps({'file':drive.name,'folder':folder,'raw_rows':len(r),'complete_raw_rows':rec['raw_complete_rows'],
                         'matched_control_rows':rec['state_packets_found'],'valid_control_rows':rec['valid_control_rows_before_warmup'],
                         'sensors':{role:{k:v for k,v in info.items() if k in ['nonfinite_rows','frozen_rows_100ms','accel_norm_under_0_1g_rows']} for role,info in rec['sensors'].items()}}),flush=True)
    if len(used_raw)!=len(raws):
        print('UNPAIRED_RAW',folder,[str(p) for p in raws if p not in used_raw],flush=True)
OUT.write_text(json.dumps({'root':str(ROOT),'records':records,'file_sha256':hashes},indent=2)+'\n')
print('SAVED',OUT,flush=True)
