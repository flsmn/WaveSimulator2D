import numpy as np
import pytest
import torch
from wave_sim2d.config import Grid,Source
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.measurements import Detector,Measurements
from wave_sim2d.wave_simulation import WaveSimulator2D


def test_finite_window_fractional_end_and_detector_reconfiguration():
    g=Grid(32,32,.25,.125,0)
    m=Measurements(g,torch.device('cpu'))
    a=Detector('a',(2,2),settling=.18,window=.31)
    b=Detector('b',(3,3),settling=0,window=.5)
    m.configure([a,b],0)
    field=torch.full(g.shape,2.)
    for step in range(1,9): m.update(field,step*g.dt,g.dt)
    result=m.results()['a']
    assert result['status']=='complete'
    assert result['elapsed']==pytest.approx(.31)
    assert result['values']==pytest.approx([4])
    old=m.detectors['b']
    m.configure([Detector('a',(4,4),settling=0,window=.2,revision=1),b],1)
    assert m.detectors['b'] is old
    assert m.results()['a']['activated']==1
    assert m.results()['a']['elapsed']==0


def test_measurements_independent_of_snapshot_rate_and_reset():
    torch.set_num_threads(1)
    g=Grid(40,40,.25,.125,1)
    scene=prepare_scene(g,sources=[Source((5,5))])
    values=[]
    for snapshot_every in (1,100):
        sim=WaveSimulator2D.from_scene(scene,device='cpu')
        sim.measurements.configure([Detector('d',(6,3),(6,7),17,.3,4.7)],0)
        for i in range(80):
            sim.step()
            if i%snapshot_every==0: sim.snapshot()
        result=sim.measurements.results()['d']
        values.append([result['values'],result['field_values']])
        sim.reset()
        assert sim.measurements.results()['d']['sample_count']==0
        assert sim.measurements.results()['d']['field_values']==[0]*17
        assert not sim.measurements.intensity.any()
    np.testing.assert_array_equal(*values)


@pytest.mark.parametrize('end,samples',[(None,1),((4,2),5)])
def test_signed_field_stays_live_after_intensity_window_completes(end,samples):
    grid=Grid(32,32,.25,.125,0)
    measurements=Measurements(grid,'cpu')
    measurements.configure([Detector('d',(2,2),end,samples,0,grid.dt)],0)
    measurements.update(torch.full(grid.shape,-2.),grid.dt,grid.dt)
    first=measurements.results()['d']
    assert first['status']=='complete' and first['values']==pytest.approx([4]*samples)
    assert first['field_values']==pytest.approx([-2]*samples)
    measurements.update(torch.full(grid.shape,3.),2*grid.dt,grid.dt)
    second=measurements.results()['d']
    assert second['values']==first['values'] and second['sample_count']==first['sample_count']
    assert second['field_values']==pytest.approx([3]*samples)
    assert second['field_time']==2*grid.dt


def test_signed_interpolation_cancels_opposite_signs_without_cancelling_intensity():
    grid=Grid(32,32,.25,.125,0)
    measurements=Measurements(grid,'cpu')
    measurements.configure([Detector('d',(2,2),settling=0,window=grid.dt)],0)
    field=torch.ones(grid.shape)*2;field[:,::2]=-2
    measurements.update(field,grid.dt,grid.dt)
    result=measurements.results()['d']
    assert result['field_values']==pytest.approx([0])
    assert result['values']==pytest.approx([4])


def test_paused_detector_reconfiguration_samples_field_without_advancing_window():
    grid=Grid(32,32,.25,.125,0)
    measurements=Measurements(grid,'cpu');spec=Detector('d',(2,2),settling=0,window=1)
    measurements.configure([spec],0)
    measurements.update(torch.ones(grid.shape),grid.dt,grid.dt)
    before=measurements.results()['d']
    measurements.configure([spec,Detector('new',(3,3))],grid.dt,field=torch.full(grid.shape,-4.))
    after=measurements.results()
    assert after['d']['values']==before['values'] and after['d']['sample_count']==before['sample_count']
    assert after['d']['field_values']==[-4] and after['new']['field_values']==[-4]
    assert after['new']['elapsed']==0 and after['new']['field_time']==grid.dt
