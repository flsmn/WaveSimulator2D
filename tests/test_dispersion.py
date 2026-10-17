"""Measured homogeneous recurrence versus the stencil preflight symbol."""
import math
import numpy as np
import pytest
import torch
from wave_sim2d.config import Grid,Material,rectangle
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D
from wave_sim2d.dispersion import temporal_frequency,phase_drift

@pytest.mark.parametrize('dx,n,modes',[(.25,1,(8,1)),(.125,1,(6,6)),(.125,1.5,(8,1)),(.0625,1.5,(6,6))])
def test_eigenmode_matches_preflight_symbol(dx,n,modes):
    torch.set_num_threads(1)
    length=20;g=Grid(round(length/dx)+1,round(length/dx)+1,dx,.5*dx,0,0)
    q=np.arange(g.nx)*dx;mx,my=modes;kx,ky=mx*math.pi/length,my*math.pi/length
    u=(np.sin(kx*q)[None,:]*np.sin(ky*q)[:,None]).astype('float32')
    sim=WaveSimulator2D.from_scene(prepare_scene(g,[Material(rectangle(0,0,g.width,g.height),n)]),device='cpu',initial_field=u)
    history=[]
    for _ in range(180):
        sim.step();history.append(float(np.sum(sim.snapshot()*u,dtype=np.float64)))
    a=np.asarray(history);cosine=np.dot(a[1:-1],a[:-2]+a[2:])/(2*np.dot(a[1:-1],a[1:-1]))
    measured=math.acos(float(cosine))/g.dt
    expected=temporal_frequency(kx,ky,dx,g.dt,n)
    assert abs(measured/expected-1)<=.001
    assert abs(measured/(math.hypot(kx,ky)/n)-1)<=.02


def test_phase_drift_scales_with_path_and_decreases_with_resolution():
    assert phase_drift(2,1.5,.125,.5,128)==pytest.approx(2*phase_drift(2,1.5,.125,.5,64))
    assert phase_drift(2,1.5,.0625,.5,128)<phase_drift(2,1.5,.125,.5,128)
