import math
import numpy as np
import pytest
import torch
from wave_sim2d import __version__
from wave_sim2d.config import ConfigurationError, BackendError, Grid, Source, Material, Wall, rectangle, MAX_COURANT
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D, resolve_device


@pytest.fixture(autouse=True, scope='session')
def threads():
    torch.set_num_threads(1)


def test_import_does_not_load_optional_visualizer():
    import sys
    assert 'wave_sim2d.wave_visualizer' not in sys.modules
    assert __version__ == '0.1.0.dev2'


def test_invalid_grid_initial_state_and_device():
    with pytest.raises(ConfigurationError): Grid(dt=MAX_COURANT * .125)
    with pytest.raises(ConfigurationError): Grid(dx=float('nan'))
    with pytest.raises(ConfigurationError): WaveSimulator2D(16, 24, initial_field=np.zeros((16,24)), device='cpu')
    with pytest.raises(ConfigurationError): WaveSimulator2D(16, 24, initial_field=np.full((24,16), np.inf), device='cpu')
    with pytest.raises(ConfigurationError): resolve_device('bogus')
    if not torch.cuda.is_available():
        with pytest.raises(BackendError): resolve_device('cuda')


def test_steps_snapshot_and_complete_reset():
    g = Grid(40, 32, .2, .1, 1)
    scene = prepare_scene(g, sources=[Source((3.1,3.1), wavelength=3)])
    sim = WaveSimulator2D.from_scene(scene, device='cpu')
    sim.step(37)
    first = sim.snapshot()
    assert not first.flags.writeable
    assert sim.steps == 37 and sim.t == 3.7
    sim.step()
    assert not np.array_equal(first, sim.snapshot())
    sim.reset()
    assert sim.steps == sim.t == 0
    assert not sim.u.any() and not sim.u_prev.any()
    sim.step(37)
    np.testing.assert_array_equal(first, sim.snapshot())
    fresh = WaveSimulator2D.from_scene(scene, device='cpu')
    fresh.step(37)
    np.testing.assert_array_equal(first, fresh.snapshot())
    with pytest.raises(ConfigurationError): sim.step(0)
    from wave_sim2d.config import NumericalError
    sim.u.fill_(float('nan'))
    with pytest.raises(NumericalError):sim.step()


def test_material_order_and_fresh_raster():
    g = Grid(32,32,.25,.125,0)
    bottom = Material(rectangle(2,2,3,3), 2, .7)
    top = Material(rectangle(3,3,2,2), 1.5, 0)
    scene = prepare_scene(g, [bottom,top])
    assert scene.refractive_index[14,14] == 1.5
    assert scene.material_damping[14,14] == 0
    assert scene.material_damping[9,9] == pytest.approx(.7)
    moved = prepare_scene(g, [Material(rectangle(5,5,2,2), 2)])
    assert moved.refractive_index[9,9] == 1
    assert scene.refractive_index[9,9] == 2
    with pytest.raises(ConfigurationError): Material(((1,1),(4,4),(1,4),(4,1)))


def test_wall_both_time_levels_and_source_rejection():
    g = Grid(48,32,.2,.1,0)
    wall = Wall(4.6,0,.4,6.4)
    scene = prepare_scene(g, walls=[wall], sources=[Source((2.1,3.1), wavelength=3)])
    sim = WaveSimulator2D.from_scene(scene, device='cpu')
    sim.step(100)
    assert not sim.u[:,25:].any()  # full-height wall also blocks diagonal stencil links
    assert not sim.u[sim.walls].any() and not sim.u_prev[sim.walls].any()
    with pytest.raises(ConfigurationError):
        prepare_scene(g,walls=[wall],sources=[Source((4.7,3.1))])


def test_additive_phase_and_incoming_wave():
    g = Grid(48,48,.2,.1,0)
    a = Source((4.1,4.1),wavelength=3,phase=.3)
    def run(sources, initial=None):
        sim = WaveSimulator2D.from_scene(prepare_scene(g,sources=sources),device='cpu',initial_field=initial)
        sim.step(40)
        return sim.snapshot()
    one = run([a])
    np.testing.assert_allclose(run([a,a]), 2*one, atol=1e-6)
    anti = Source(a.start,wavelength=a.wavelength,phase=a.phase+math.pi)
    np.testing.assert_allclose(run([a,anti]),0,atol=2e-6)
    initial = np.exp(-((np.arange(48)[None,:]-10)**2+(np.arange(48)[:,None]-20)**2)/12).astype('float32')
    np.testing.assert_allclose(run([a],initial),run([],initial)+one,atol=2e-6)


def test_spatial_source_measure_and_separate_border():
    for dx in (.25,.125):
        g = Grid(round(8/dx),round(8/dx),dx,.5*dx,1,2)
        s = prepare_scene(g,[Material(rectangle(0,0,8,8),1,0)],sources=[Source((3,3)),Source((2,4),(6,4))])
        assert s.sources[0].weights.sum()*dx**2 == pytest.approx(1)
        assert s.sources[1].weights.sum()*dx**2 == pytest.approx(4)
        assert s.border_damping[0,0] > 0 and s.material_damping.max() == 0
