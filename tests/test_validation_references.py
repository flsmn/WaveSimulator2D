"""Qualify the independent reference and catch phase/normalization mistakes."""
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def validation_module(name):
    # Validation scripts are deliberately not installed with the runtime wheel.
    spec = importlib.util.spec_from_file_location(name, ROOT/'validation'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_policy = validation_module('reference_policy').load_policy
snell_diagnostics = validation_module('reference_policy').snell_diagnostics


def test_historical_mac_policy_retains_original_thresholds():
    base = json.loads((ROOT/'validation/tolerances.json').read_text())
    mac = load_policy(ROOT/'validation/mac-presets-v1.json')
    for group in ('T01', 'T02', 'T03'):
        assert mac[group] == base[group]
    assert mac['T04']['focus_position_relative_error'] == base['T04']['focus_position_relative_error']
    assert mac['T04']['minimum_focus_gain'] == base['T04']['minimum_focus_gain']
    assert mac['T04']['grid_focus_shift_relative_error'] == .05


def test_measured_incidence_exposes_previously_misleading_one_degree_claim():
    saved = json.loads((ROOT/'docs/physical-results-dev4-review-input.json').read_text())['results']['T02']
    result = snell_diagnostics(saved['incidence_degrees'], saved['transmission_degrees'])
    assert result['nominal_angle_error_degrees'] == pytest.approx(saved['angle_error_degrees'])
    assert result['nominal_angle_error_degrees'] < 1
    assert result['measured_incidence_angle_error_degrees'] > 1
    assert not result['quantitative_angle_accepted']
    with pytest.raises(ValueError):
        snell_diagnostics(70., 30., n1=1.5, n2=1.)


def test_source_reference_has_correct_delta_forcing_normalization_and_phase():
    pytest.importorskip('scipy')
    outgoing_reference = validation_module('quantitative_sources').outgoing_reference
    from wave_sim2d.config import Source
    source = Source((0., 0.), amplitude=1.7, phase=.41)
    radius, step = 1e-5, 1e-8
    u = outgoing_reference(source, [[radius-step, 0.], [radius+step, 0.]])
    flux = 2*np.pi*radius*(u[1]-u[0])/(2*step)
    # Integrate Delta U=-F over a shrinking circle, F=-i*A*exp(i*phase).
    np.testing.assert_allclose(flux, 1j*source.amplitude*np.exp(1j*source.phase), rtol=1e-6)


def test_finite_line_gauss_reference_matches_independent_adaptive_integration():
    scipy = pytest.importorskip('scipy')
    from scipy.integrate import quad
    from scipy.special import hankel2
    outgoing_reference = validation_module('quantitative_sources').outgoing_reference
    from wave_sim2d.config import Source
    source = Source((0., -8.), (0., 8.), amplitude=.7, phase=.23)
    probes = np.array([[8., 0.], [9., 4.], [7., -3.]])
    expected = []
    for x, y in probes:
        def field(s):
            return -source.amplitude*np.exp(1j*source.phase)*hankel2(0, source.omega*np.hypot(x, y-s))/4
        expected.append(quad(lambda s: field(s).real, -8., 8., epsabs=1e-12)[0] +
                        1j*quad(lambda s: field(s).imag, -8., 8., epsabs=1e-12)[0])
    np.testing.assert_allclose(outgoing_reference(source, probes, 128), expected, rtol=1e-11, atol=1e-12)


def test_phasor_fit_retains_absolute_phase_and_does_not_hide_error():
    pytest.importorskip('scipy')
    module = validation_module('quantitative_sources')
    fit_phasor, compare = module.fit_phasor, module.compare
    times = np.linspace(13.2, 40.1, 407)
    omega = 1.4
    reference = np.array([.2+.3j, -.8+.1j])
    values = np.real(np.exp(1j*omega*times[:, None])*reference) + np.array([2., -.4])
    measured = fit_phasor(times, values, omega)
    np.testing.assert_allclose(measured, reference, atol=2e-15)
    assert compare(measured*np.exp(.04j), reference)['maximum_complex_relative_error'] > .039
    assert compare(measured*1.03, reference)['maximum_amplitude_relative_error'] > .029
    with pytest.raises(ValueError):
        compare([0j], [0j])
