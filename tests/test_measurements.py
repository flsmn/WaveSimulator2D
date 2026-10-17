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
        values.append(sim.measurements.results()['d']['values'])
        sim.reset()
        assert sim.measurements.results()['d']['sample_count']==0
        assert not sim.measurements.intensity.any()
    np.testing.assert_array_equal(*values)
