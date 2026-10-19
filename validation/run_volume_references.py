"""Calibrate an independent scalar lens reference against an exact cylinder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from helmholtz_volume import (VolumePotential, grid_cells, area_fraction,
                             evaluate_scattered, disk_plane_reference, lens_fraction)
from quantitative_sources import outgoing_reference, serial_phasor
from wave_sim2d.config import Source

ROOT = Path(__file__).resolve().parent


def field_difference(actual, reference):
    return {'complex_relative_l2': float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
            'maximum_absolute_error': float(np.max(np.abs(actual-reference)))}


def disk_case(configuration, spacing):
    c = configuration
    k = 2*np.pi/c['wavelength']
    r = c['radius']
    xx, yy = grid_cells((-r, r, -r, r), spacing)
    fraction = area_fraction(xx, yy, spacing, lambda x, y: x*x+y*y < r*r)
    contrast = k*k*(c['index']**2-1)*fraction
    incident = np.exp(-1j*k*xx)
    potential = VolumePotential(xx.shape, spacing, k)
    solution, solve_report = potential.solve(contrast, incident)
    angles = np.arange(c['angles'])*2*np.pi/c['angles']
    probes = np.concatenate([np.column_stack((radius*np.cos(angles), radius*np.sin(angles)))
                             for radius in c['probe_radii']])
    measured = np.exp(-1j*k*probes[:, 0])+evaluate_scattered(probes, xx, yy, contrast*solution, spacing, k)
    exact, check = [disk_plane_reference(probes, r, c['index'], k, modes)
                    for modes in c['series_modes']]
    return {'spacing': spacing, 'shape': list(xx.shape), **solve_report,
            'metrics': field_difference(measured, check),
            'series_check': field_difference(exact, check),
            'field': serial_phasor(measured), 'exact_field': serial_phasor(check)}


def continuous_line(source, positions, order):
    return np.concatenate([outgoing_reference(source, positions[start:start+256], order)
                           for start in range(0, len(positions), 256)])


def peak_location(x, intensity, points=3):
    peak = int(np.argmax(intensity))
    radius = points//2
    if not radius <= peak < len(x)-radius:
        raise ValueError('Reference maximum is at the edge of the prescribed search interval.')
    local = slice(peak-radius, peak+radius+1)
    coefficients = np.polyfit(x[local]-x[peak], intensity[local], 2)
    if coefficients[0] >= 0:
        raise ValueError('Reference peak fit is not concave.')
    vertex = -coefficients[1]/(2*coefficients[0])
    if abs(vertex) > x[peak+radius]-x[peak]:
        raise ValueError('Reference peak fit leaves the fitted neighbourhood.')
    return float(x[peak]+vertex)


def lens_case(configuration, spacing, subcells=8, quadrature=512):
    c = configuration
    k = 2*np.pi/c['wavelength']
    cx, cy = c['centre']
    radius, aperture = c['radius'], c['aperture']
    thickness = 2*(radius-np.sqrt(radius*radius-(aperture/2)**2))
    bfl = (1-((c['index']-1)*thickness)/(c['index']*radius)) / (
        (c['index']-1)*(2/radius-(c['index']-1)*thickness/(c['index']*radius*radius)))
    xx, yy = grid_cells((cx-thickness/2, cx+thickness/2, cy-aperture/2, cy+aperture/2), spacing)
    if xx.size > c.get('max_cells', 100000):
        raise ValueError('Reference grid exceeds the explicit validation cell budget.')
    fraction = lens_fraction(xx, yy, spacing, radius=radius, aperture=aperture,
                             cx=cx, cy=cy, subcells=subcells)
    contrast = k*k*(c['index']**2-1)*fraction
    source = Source(tuple(c['source_start']), tuple(c['source_end']), c['wavelength'], phase=c['source_phase'])
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    incident = continuous_line(source, positions, quadrature).reshape(xx.shape)
    potential = VolumePotential(xx.shape, spacing, k)
    field, solve_report = potential.solve(contrast, incident, maxiter=c.get('linear_maxiter', 40))
    x = np.linspace(*c['profile_x'], c['profile_count'])
    probes = np.column_stack((x, np.full_like(x, cy)))
    background = continuous_line(source, probes, quadrature)
    total = background+evaluate_scattered(probes, xx, yy, contrast*field, spacing, k)
    intensity = np.abs(total)**2
    peak, wider_peak = peak_location(x, intensity), peak_location(x, intensity, 5)
    rear = cx+thickness/2
    row = {'spacing': spacing, 'shape': list(xx.shape), 'subcells': subcells,
           'quadrature': quadrature, 'linear_maxiter': c.get('linear_maxiter', 40), **solve_report, 'focus_x': peak,
           'wider_fit_focus_x': wider_peak, 'back_focal_distance': peak-rear,
           'paraxial_back_focal_distance': float(bfl),
           'paraxial_relative_difference': abs((peak-rear)/bfl-1),
           'fit_difference_over_paraxial_bfl': abs(peak-wider_peak)/bfl,
           'peak_gain': float(intensity.max()/np.abs(background[np.argmax(intensity)])**2),
           'x': x.tolist(), 'field': serial_phasor(total)}
    return row, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT/'volume-reference-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path, help='Reuse completed grids with identical physical inputs and reference operator.')
    args = parser.parse_args()
    data = args.policy.read_bytes()
    policy = json.loads(data)
    report = {'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__,
              'policy': policy, 'policy_sha256': hashlib.sha256(data).hexdigest(),
              'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in ['helmholtz_volume.py', 'run_volume_references.py']},
              'disk': [], 'lens': [], 'sensitivity': {}, 'candidate_reference_passed': False,
              'time_domain_engine_compared': False, 'mac01_complete': False}
    cached = None
    if args.resume_from:
        cached_data = args.resume_from.read_bytes()
        cached = json.loads(cached_data)
        if cached['implementation_sha256']['helmholtz_volume.py'] != report['implementation_sha256']['helmholtz_volume.py']:
            raise ValueError('Cached reference operator differs from the current implementation.')
        if cached['policy']['disk'] != policy['disk'] or cached['policy']['limits'] != policy['limits']:
            raise ValueError('Cached disk configuration or acceptance limits differ.')
        def physical_inputs(configuration):
            return {k: v for k, v in configuration.items() if k not in ('linear_maxiter', 'max_cells')}
        if physical_inputs(cached['policy']['lens']) != physical_inputs(policy['lens']):
            raise ValueError('Cached lens geometry, sampling or illumination differs.')
        report['resumed_from'] = {'path': str(args.resume_from), 'sha256': hashlib.sha256(cached_data).hexdigest(),
                                  'note': 'Completed grids reused; only linear iteration/cell admission budgets may differ. Previous numerical residuals retain their recorded values.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    began = time.monotonic()
    try:
        for spacing in policy['disk']['spacings']:
            row = next((row for row in cached.get('disk', []) if row['spacing'] == spacing), None) if cached else None
            row = row or disk_case(policy['disk'], spacing)
            report['disk'].append(row)
            save()
            print('disk', spacing, row['metrics'], flush=True)
        previous = None
        for spacing in policy['lens']['spacings']:
            row = next((row for row in cached.get('lens', []) if row['spacing'] == spacing), None) if cached else None
            if row:
                field = np.asarray([z['real']+1j*z['imag'] for z in row['field']])
            else:
                row, field = lens_case(policy['lens'], spacing)
            if previous is not None:
                row['change_from_previous_grid'] = field_difference(field, previous)
                row['focus_shift_from_previous_over_paraxial_bfl'] = abs(row['focus_x']-report['lens'][-1]['focus_x'])/row['paraxial_back_focal_distance']
            report['lens'].append(row)
            previous = field
            save()
            print('lens', spacing, row['back_focal_distance'], row.get('change_from_previous_grid'), flush=True)
        baseline = previous
        for name, kwargs in [('geometry', {'subcells': policy['lens']['geometry_subcells'][1]}),
                             ('source_quadrature', {'quadrature': policy['lens']['source_quadrature_orders'][0]})]:
            row, field = lens_case(policy['lens'], policy['lens']['spacings'][-1], **kwargs)
            row['change'] = field_difference(field, baseline)
            report['sensitivity'][name] = row
            save()
            print(name, row['change'], flush=True)
        limits = policy['limits']
        disk, lens = report['disk'][-1], report['lens'][-1]
        checks = {
            'disk_l2': disk['metrics']['complex_relative_l2'] <= limits['disk_finest_complex_relative_l2'],
            'disk_maximum': disk['metrics']['maximum_absolute_error'] <= limits['disk_finest_maximum_absolute_error'],
            'disk_series': disk['series_check']['complex_relative_l2'] <= limits['disk_series_relative_change'],
            'lens_grid': lens['change_from_previous_grid']['complex_relative_l2'] <= limits['lens_last_grid_complex_relative_l2'],
            'lens_geometry': report['sensitivity']['geometry']['change']['complex_relative_l2'] <= limits['lens_geometry_complex_relative_l2'],
            'lens_quadrature': report['sensitivity']['source_quadrature']['change']['complex_relative_l2'] <= limits['lens_quadrature_complex_relative_l2'],
            'lens_focus_grid': lens['focus_shift_from_previous_over_paraxial_bfl'] <= limits['lens_last_grid_focus_shift_over_paraxial_bfl'],
            'lens_peak_fit': lens['fit_difference_over_paraxial_bfl'] <= limits['lens_peak_fit_difference_over_paraxial_bfl'],
        }
        report['checks'] = {name: bool(value) for name, value in checks.items()}
        report['candidate_reference_passed'] = all(report['checks'].values())
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-began
        save()
    return 0 if report['candidate_reference_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
