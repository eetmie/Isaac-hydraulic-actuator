"""Compare completed recorded circles; retain partial sessions separately."""
import csv
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
sources = {
    'PID, deadzone 0%': 'circle_pid_dz0_live_20261007_03',
    'MLP 100 Hz, deadzone 0%': 'circle_mlp_dz0_live_20261007_03',
    'MLP 100 Hz, deadzone 2% (earlier)': 'circle_mlp_100hz_live_20261007_10',
}
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
results = {}
for label, stem in sources.items():
    report = json.loads((root / (stem+'.json')).read_text())
    passes = [p for p in report['passes'] if p['completed'] and not p['partial']]
    ids = {p['pass_index'] for p in passes}
    move_end = 1 + 2*np.pi*(report['radius_mm']/1000)/(report['speed_mm_s']/1000) + (report['speed_mm_s']/1000)/0.5
    with gzip.open(root / (stem+'.csv.gz'), 'rt', newline='') as f:
        rows = [r for r in csv.DictReader(f) if int(float(r['pass_index'])) in ids and 1 <= float(r['motion_t_s']) < move_end]
    get = lambda k: np.array([float(r[k]) for r in rows])
    t = get('t_s')
    error = 1000*get('error_m')
    rate = np.rad2deg(get('v_pitch'))
    pitch = np.rad2deg(get('q_pitch'))
    metrics = {
        'complete_recorded_circles': len(passes),
        'tracking_rmse_mm': float(np.sqrt(np.mean(error**2))),
        'radial_mean_abs_mm': float(np.mean(np.abs(1000*get('radial_error_m')))),
        'pitch_rate_rms_deg_s': float(np.sqrt(np.mean(rate**2))),
        'pitch_peak_to_peak_deg': float(np.ptp(pitch)),
        'mean_pass_valve_travel_per_s': float(np.mean([p['valve_travel_per_s'] for p in passes])),
        'session_result': report['result'],
    }
    results[label] = metrics
    axes[0,0].plot([p['pass_index']+1 for p in passes], [p['tracking_rmse_mm'] for p in passes], 'o-', label=label)
    axes[0,1].plot([p['pass_index']+1 for p in passes], [p['valve_travel_per_s'] for p in passes], 'o-', label=label)
    axes[1,0].plot(t-t[0], pitch, lw=.6, label=label, alpha=.75)
    axes[1,1].plot(t-t[0], rate, lw=.6, label=label, alpha=.65)
axes[0,0].set(xlabel='Recorded circle number', ylabel='Tracking RMSE [mm]', title='Joint-derived cutting-tip tracking')
axes[0,1].set(xlabel='Recorded circle number', ylabel='Total normalized valve travel / s', title='Command variation')
axes[1,0].set(xlabel='Time from first included sample [s]', ylabel='Base pitch [degrees]', title='Base IMU orientation')
axes[1,1].set(xlabel='Time from first included sample [s]', ylabel='Base gyro Y [degrees/s]', title='Base pitch rate')
for ax in axes.flat:
    ax.grid(alpha=.25)
    ax.legend(fontsize=7)
fig.suptitle('100 mm circles at 20 mm/s — complete recorded circles only\nSequential hardware sessions; oil temperature uncontrolled')
fig.savefig(root/'figures'/'successful_circle_comparison.png', dpi=160)
(root/'successful_circle_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
