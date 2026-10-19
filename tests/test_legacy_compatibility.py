"""Optional legacy compatibility; these checks do not certify the static model."""
import importlib.util
import subprocess
import sys
import textwrap

import pytest


def run_legacy(code, *, visualizer=False):
    # Keep optional imports out of the core-import isolation test's interpreter.
    for dependency in ('cv2', 'matplotlib') if visualizer else ('cv2',):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f'Install the examples extra for legacy checks ({dependency}).')
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(code)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_sources_materials_and_reset_on_explicit_cpu():
    run_legacy('''
        import numpy as np
        import torch
        from wave_sim2d.wave_simulation import WaveSimulator2D
        from wave_sim2d.scene_objects.source import PointSource, LineSource, ModulatorSmoothSquare
        from wave_sim2d.scene_objects.static_dampening import StaticDampening
        from wave_sim2d.scene_objects.static_refractive_index import StaticRefractiveIndex, StaticRefractiveIndexBox
        from wave_sim2d.scene_objects.strain_refractive_index import StrainRefractiveIndex
        torch.set_num_threads(1)
        sim = WaveSimulator2D(64, 64, [
            StaticDampening(np.ones((64, 64), dtype='float32'), 4),
            StaticRefractiveIndex(np.full((64, 64), 1.2, dtype='float32')),
            StaticRefractiveIndexBox((40, 32), (8, 20), .2, 1.5),
            StrainRefractiveIndex(1.2, .01),
            PointSource(20, 30, .4, amp_modulator=ModulatorSmoothSquare(.1, phase=0.0)),
            LineSource((24, 24), (24, 40), .3, amplitude=.2),
        ], device='cpu')
        sim.step(40)
        first = sim.snapshot()
        assert np.isfinite(first).all() and np.max(np.abs(first)) > 0
        assert sim.device.type == 'cpu'
        sim.reset_time()
        assert sim.steps == sim.t == 0 and not sim.u.any()
        sim.step(40)
        np.testing.assert_array_equal(first, sim.snapshot())
    ''')


def test_legacy_image_scene_remains_usable():
    run_legacy('''
        import numpy as np
        import torch
        from wave_sim2d.wave_simulation import WaveSimulator2D
        from wave_sim2d.scene_objects.static_image_scene import StaticImageScene
        torch.set_num_threads(1)
        image = np.zeros((128, 128, 3), dtype='uint8')
        image[:, :, 0] = 100
        image[64, 64, 1] = 100
        sim = WaveSimulator2D(128, 128, [StaticImageScene(image)], device='cpu')
        sim.step(30)
        assert np.isfinite(sim.snapshot()).all() and sim.u.abs().max() > 0
    ''')


def test_legacy_visualizer_renders_field_and_intensity_without_windows():
    run_legacy('''
        import numpy as np
        import torch
        from wave_sim2d.wave_simulation import WaveSimulator2D
        from wave_sim2d.scene_objects.source import PointSource
        from wave_sim2d.wave_visualizer import WaveVisualizer, get_colormap_lut
        torch.set_num_threads(1)
        sim = WaveSimulator2D(32, 32, [PointSource(16, 16, .4)], device='cpu')
        view = WaveVisualizer(
            get_colormap_lut('colormap_wave1', invert=False),
            get_colormap_lut('afmhot', invert=False),
        )
        sim.step(20)
        view.update(sim)
        for frame in (view.render_field(), view.render_intensity()):
            assert frame.shape == (32, 32, 3) and frame.dtype == np.uint8
    ''', visualizer=True)
