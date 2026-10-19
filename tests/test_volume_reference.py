"""Reference-operator correctness, independent of the time-domain solver."""
import numpy as np
import pytest
import importlib.util
from pathlib import Path

pytest.importorskip('scipy')
from scipy.integrate import quad
spec = importlib.util.spec_from_file_location('helmholtz_volume', Path(__file__).resolve().parents[1]/'validation/helmholtz_volume.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VolumePotential, green = module.VolumePotential, module.green
self_cell, disk_plane_reference = module.self_cell, module.disk_plane_reference


def test_integrated_singular_cell_matches_independent_duffy_quadrature():
    h, k = .13, 2.1
    # Independent double integration, not the analytic radial antiderivative.
    def integral(component):
        def outer(x):
            return quad(lambda t: component(green(k, x*np.sqrt(1+t*t)))*x,
                        0., 1., epsabs=1e-12)[0]
        return 8*quad(outer, 0., h/2, epsabs=1e-12)[0]
    expected = integral(np.real)+1j*integral(np.imag)
    np.testing.assert_allclose(self_cell(k, h), expected, rtol=1e-10, atol=1e-12)


def test_fft_convolution_and_gmres_match_direct_small_matrix():
    shape, h, k = (5, 4), .2, 3.
    potential = VolumePotential(shape, h, k)
    indices = np.indices(shape).reshape(2, -1).T
    matrix = np.array([[potential.kernel[a[0]-b[0]+shape[0]-1, a[1]-b[1]+shape[1]-1]
                        for b in indices] for a in indices])
    rng = np.random.default_rng(45)
    contrast = rng.uniform(.1, 1., shape)
    incident = rng.normal(size=shape)+1j*rng.normal(size=shape)
    np.testing.assert_allclose(potential.apply(incident).ravel(), matrix@incident.ravel(), atol=1e-15)
    solved, report = potential.solve(contrast, incident)
    expected = np.linalg.solve(np.eye(incident.size)-matrix*contrast.ravel()[None, :], incident.ravel())
    np.testing.assert_allclose(solved.ravel(), expected, atol=1e-8)
    assert report['relative_residual'] < 1e-8


def test_cylinder_reference_becomes_incident_wave_without_contrast():
    probes = [[2., 0.], [2., 1.], [-3., -1.]]
    actual = disk_plane_reference(probes, 1., 1., 4.)
    np.testing.assert_allclose(actual, np.exp(-4j*np.asarray(probes)[:, 0]), atol=1e-14)
