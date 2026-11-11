"""Check physical coordinates, clock wrap, missing packets and causal hold behavior."""
import numpy as np
import pandas as pd
import build_complete_dataset as b


def about_y(degrees):
    angle=np.deg2rad(degrees)/2
    return np.stack([np.cos(angle),np.zeros_like(angle),np.sin(angle),np.zeros_like(angle)],axis=-1)


def test_sensor_contract_and_scale():
    # Sensor mounting changes zero, while relative gyro rates remain child-parent.
    raw=about_y(np.array([0.,30.,70.,80.])+np.degrees(b.MOUNT))
    q=b.positions(raw)
    assert np.allclose(q[:3],np.deg2rad([30,40,10]),atol=1e-6)
    assert abs(q[3]+b.CONTRACT['carriage_pitch_zero_rad'])<1e-8
    gyro=np.zeros((4,3));gyro[:,1]=[1,3,6,10]
    assert np.allclose(b.rates(gyro),np.deg2rad([2,3,4,1]))
    assert abs(b.V1_SCALE-1.14688)<1e-12
    assert abs((250/32768)*b.V1_SCALE-.00875)<1e-12


def frame(stamp,state):
    n=len(stamp)
    return pd.DataFrame(dict(timestamp=stamp,sample_idx=np.arange(n),state_imu_ts_us=state,
                             cmd_age_s=np.full(n,.001),state_age_s=np.full(n,.002),cmd_stale=np.zeros(n),
                             **{f'combined_cmd_{k}':np.zeros(n) for k in ('rotate','lift','tilt','scoop')}))


def test_wrap_and_missing_state_do_not_select_future_packet():
    wrap=2**32
    raw=b.unwrap(np.array([wrap-10000,wrap-5000,0,5000,10000]))
    # A missing state before a real wrap must not fabricate a second wrap.
    d=frame(np.arange(6)*.01,[wrap-10000,-1,wrap-5000,0,7500,10000])
    quat=np.tile([1.,0,0,0],(5,4,1));gyro=np.zeros((5,4,3))
    ids,matched,valid,*_=b.aligned_drive(d,raw,quat,gyro,np.ones(5,bool))
    assert matched.tolist()==[True,False,True,True,False,True]
    assert valid.tolist()==matched.tolist()
    assert ids[4]==3  # 7500 has no exact recorded packet; never use the 10000 packet.


def test_training_holds_only_prior_commands_and_splits_bad_rows():
    n=1100;stamp=.001+np.arange(n)*.01
    d=frame(stamp,np.arange(n)*10000)
    valid=np.ones(n,bool);valid[550]=False
    q=np.zeros((n,4),np.float32);v=np.zeros_like(q)
    commands=np.zeros((n,4));commands[:,1]=np.arange(n)/n
    chunks=list(b.training_chunks(d,valid,q,v,commands,np.arange(n)))
    assert len(chunks)==2
    for c in chunks:
        ids=c['source_control_row']
        assert np.all(c['command_timestamp_s']<=c['t'])
        assert np.all(c['t']-c['command_timestamp_s']<.010001)
        assert np.allclose(c['u'][:,0],ids/n)
        assert 550 not in ids
        assert np.allclose(np.diff(c['t']),.01)
    assert chunks[0]['source_control_row'][-1]<550<chunks[1]['source_control_row'][0]


def test_stale_held_state_is_not_bridged():
    n=1500;stamp=np.arange(n)*.01
    d=frame(stamp,np.arange(n)*10000)
    d.loc[500:900,'state_age_s']=.04
    zeros=np.zeros((n,4),np.float32)
    chunks=list(b.training_chunks(d,np.ones(n,bool),zeros,zeros,zeros,np.arange(n)))
    assert len(chunks)==2
    assert all(not np.any((c['source_control_row']>=500)&(c['source_control_row']<=900)) for c in chunks)


def test_replay_physics(binary):
    settings=dict(ahrsGain=.5,ahrsAccelRejection=10.,ahrsRecoveryPeriodS=2.,gyro_saturation_dps=573.44)
    n=2000;t=np.arange(n)*5000.;gyro=np.zeros((n,4,3));accel=np.tile([0.,0.,1.],(n,4,1))
    seed=np.tile([1.,0.,0.,0.],(4,1))
    q,_=b.replay_segment(binary,settings,t,gyro,accel,seed)
    assert np.max(np.abs(q[:,:,1:]))<1e-6
    # Real yaw stays integrated without a magnetometer or forced heading zero.
    gyro[:,:,2]=30.
    q,_=b.replay_segment(binary,settings,t,gyro,accel,seed)
    yaw=np.unwrap(2*np.arctan2(q[:,0,3],q[:,0,0]))
    assert abs(np.degrees(yaw[-1])-(n-1)*.005*30)<.02
    # Small gravity tilt converges with the configured two-second time constant.
    gyro[:]=0.;tilt=np.deg2rad(3)
    accel[:]=[-np.sin(tilt),0,np.cos(tilt)]
    q,_=b.replay_segment(binary,settings,t,gyro,accel,seed)
    pitch=np.arctan2(2*(q[:,0,0]*q[:,0,2]-q[:,0,1]*q[:,0,3]),1-2*(q[:,0,1]**2+q[:,0,2]**2))
    assert abs(np.degrees(pitch[-1])-3)<.05
    assert .5<np.degrees(pitch[400])/3<.75


if __name__=='__main__':
    for test in [test_sensor_contract_and_scale,test_wrap_and_missing_state_do_not_select_future_packet,
                 test_training_holds_only_prior_commands_and_splits_bad_rows,test_stale_held_state_is_not_bridged]:
        test()
        print(test.__name__,'passed')
