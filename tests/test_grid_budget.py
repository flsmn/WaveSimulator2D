from dataclasses import replace
import numpy as np
import pytest
from wave_sim2d.config import Grid,ConfigurationError
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D


def test_explicit_axis_budget_reaches_solver_constructor():
    with pytest.raises(ConfigurationError):Grid(5000,8,1,.5,0)
    g=Grid(5000,8,1,.5,0,max_axis_cells=8192)
    sim=WaveSimulator2D.from_scene(prepare_scene(g),device='cpu');sim.step(2)
    assert sim.grid==g and sim.snapshot().shape==(8,5000)
    with pytest.raises(ConfigurationError):replace(g,max_axis_cells=True)


def test_admission_budget_does_not_change_numerics():
    field=np.zeros((16,16),dtype=np.float32);field[8,8]=1
    a=Grid(16,16,.25,.125,0)
    sims=[WaveSimulator2D.from_scene(prepare_scene(g),device='cpu',initial_field=field) for g in (a,replace(a,max_axis_cells=8192))]
    for sim in sims:sim.step(20)
    np.testing.assert_array_equal(sims[0].snapshot(),sims[1].snapshot())
