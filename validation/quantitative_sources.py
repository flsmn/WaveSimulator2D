"""Independent outgoing-wave source reference; never a full MAC-01 certificate.

Run with validation extras installed. Neither the reference nor the phasor fit
uses the engine's stencil, forcing quadrature, or source-normalization code.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
from scipy.special import hankel2
import torch

from wave_sim2d import __version__
from wave_sim2d.config import Grid, Source
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D

ROOT = Path(__file__).resolve().parent


def outgoing_reference(source, probes, quadrature_order=256):
    """Solve (Delta+omega²)U=i*A*exp(i*phase)*delta in infinite R².

    With exp(+i*omega*t), G=-i*H0^(2)/4 solves (Delta+k²)G=-delta.
    Multiplication by the forcing phasor -i*A*exp(i*phase) gives -A*H/4.
    The finite line is integrated in continuous arc length by Gauss-Legendre.
    """
    probes = np.asarray(probes, dtype=float)
    if source.end is None:
        distances = np.linalg.norm(probes-np.asarray(source.start), axis=1)
        if np.any(distances <= 0):
            raise ValueError('The outgoing point reference is singular at the source.')
        integral = hankel2(0, source.omega*distances)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        start, end = np.asarray(source.start), np.asarray(source.end)
        positions = start+(nodes[:, None]+1)*(end-start)/2
        distances = np.linalg.norm(probes[:, None]-positions[None, :], axis=2)
        if np.any(distances <= 0):
            raise ValueError('Probe must be outside the line source.')
        integral = hankel2(0, source.omega*distances) @ weights * np.linalg.norm(end-start)/2
    return -source.amplitude*np.exp(1j*source.phase)*integral/4


def fit_phasor(times, values, omega):
    """Fit real samples to Re(U exp(i*omega*t))+DC, without phase alignment."""
    times, values = np.asarray(times), np.asarray(values)
    basis = np.column_stack((np.cos(omega*times), np.sin(omega*times), np.ones(len(times))))
    coefficients, _, rank, _ = np.linalg.lstsq(basis, values, rcond=None)
    if rank != 3:
        raise ValueError('Time samples do not resolve the excitation frequency.')
    return coefficients[0]-1j*coefficients[1]


def compare(measured, reference):
    measured, reference = np.asarray(measured), np.asarray(reference)
    if np.any(np.abs(reference) < 1e-14):
        raise ValueError('Relative source metrics require nonzero reference fields.')
    if not np.isfinite(measured).all() or not np.isfinite(reference).all():
        raise ValueError('Source phasors must be finite.')
    return {
        'maximum_complex_relative_error': float(np.max(np.abs((measured-reference)/reference))),
        'maximum_amplitude_relative_error': float(np.max(np.abs(np.abs(measured)/np.abs(reference)-1))),
        'complex_relative_l2': float(np.linalg.norm(measured-reference)/np.linalg.norm(reference)),
        'maximum_phase_error_radians': float(np.max(np.abs(np.angle(measured/reference)))),
    }


def serial_phasor(values):
    return [{'real': float(z.real), 'imag': float(z.imag)} for z in values]


def sample_windows(configuration, kind, dx, *, courant=None, width=None, height=None,
                   translation=(0., 0.), border_width=None, device='cpu'):
    c = configuration
    shift = np.asarray(translation)
    spec = c[kind]
    source = Source(tuple(np.asarray(spec['start'])+shift),
                    tuple(np.asarray(spec['end'])+shift) if 'end' in spec else None,
                    c['wavelength'], c['amplitude'], c['phase'])
    probes = np.asarray(spec['probes'])+shift
    width, height = width or c['width'], height or c['height']
    g = Grid(round(width/dx), round(height/dx), dx, dx*(courant or c['courant']),
             border_width or c['border_width'], c['border_rate'])
    if g.estimated_bytes > 256*1024**2:
        raise ValueError('This source subprotocol is limited to 256 MiB estimated buffers.')
    scene = prepare_scene(g, sources=[source])
    sim = WaveSimulator2D.from_scene(scene, device=device)
    xy = probes/dx-.5
    base = np.floor(xy).astype(int)
    fraction = xy-base
    indices, weights = [], []
    for dy in (0, 1):
        for ix in (0, 1):
            indices.append((base[:, 1]+dy)*g.nx+base[:, 0]+ix)
            weights.append((fraction[:, 0] if ix else 1-fraction[:, 0]) *
                           (fraction[:, 1] if dy else 1-fraction[:, 1]))
    indices = torch.tensor(np.asarray(indices), device=device)
    weights = torch.tensor(np.asarray(weights), dtype=torch.float32, device=device)
    settle_steps = round(c['settle']/g.dt)
    while sim.steps < settle_steps:
        sim.step(min(128, settle_steps-sim.steps))
    stride = max(1, int(c['wavelength']/(c['samples_per_cycle']*g.dt)))
    windows = []
    all_times, all_values = [], []
    for window in range(2):
        stop = round((c['settle']+(window+1)*c['window'])/g.dt)
        times, values = [], []
        while sim.steps < stop:
            sim.step(min(stride, stop-sim.steps))
            times.append(sim.t)
            values.append((sim.u.view(-1)[indices]*weights).sum(dim=0).cpu().numpy())
        windows.append(fit_phasor(times, values, source.omega))
        all_times.extend(times)
        all_values.extend(values)
    combined = fit_phasor(all_times, all_values, source.omega)
    return {
        'grid': asdict(g), 'source': asdict(source), 'probes': probes.tolist(),
        'sampling_stride_steps': stride, 'estimated_buffer_bytes': g.estimated_bytes,
        'first_window': serial_phasor(windows[0]), 'later_window': serial_phasor(windows[1]),
        'double_window': serial_phasor(combined),
    }, windows[0], windows[1], combined


def run_kind(policy, kind, device, save):
    c, limits = policy['source'], policy['source']['limits']
    source = Source(tuple(c[kind]['start']), tuple(c[kind]['end']) if 'end' in c[kind] else None,
                    c['wavelength'], c['amplitude'], c['phase'])
    low, high = (outgoing_reference(source, c[kind]['probes'], n) for n in c['quadrature_orders'])
    reference_change = compare(low, high)['maximum_complex_relative_error']
    result = {'reference': serial_phasor(high), 'quadrature_relative_change': reference_change,
              'grids': [], 'sensitivity': {}, 'passed': False, 'status': 'running'}
    save(result)
    baseline = None
    for dx in c['dx_values']:
        began = time.monotonic()
        row, first, later, longer = sample_windows(c, kind, dx, device=device)
        row.update(metrics=compare(first, high), later_metrics=compare(later, high),
                   temporal_window_relative_change=compare(later, first)['maximum_complex_relative_error'],
                   double_window_relative_change=compare(longer, first)['maximum_complex_relative_error'],
                   elapsed_seconds=time.monotonic()-began)
        result['grids'].append(row)
        if dx == c['qualified_dx']:
            baseline = first
            result['sensitivity']['later_window'] = row['temporal_window_relative_change']
            result['sensitivity']['double_window'] = row['double_window_relative_change']
        save(result)
        print(kind, dx, row['metrics'], flush=True)
    if baseline is None:
        raise ValueError('qualified_dx must occur in dx_values.')
    sensitivity = c['sensitivity']
    variants = {
        'half_time_step': {'courant': sensitivity['courant']},
        'expanded_domain': {'width': sensitivity['expanded_width'], 'height': sensitivity['expanded_height'],
                            'translation': sensitivity['expanded_translation']},
        'wider_border': {'border_width': sensitivity['wider_border']},
    }
    result['sensitivity_runs'] = []
    for name, kwargs in variants.items():
        row, first, later, longer = sample_windows(c, kind, c['qualified_dx'], device=device, **kwargs)
        row.update(name=name, metrics=compare(first, high),
                   relative_change_from_baseline=compare(first, baseline)['maximum_complex_relative_error'])
        result['sensitivity_runs'].append(row)
        result['sensitivity'][name] = row['relative_change_from_baseline']
        save(result)
        print(kind, name, row['relative_change_from_baseline'], flush=True)
    qualified = next(row for row in result['grids'] if row['grid']['dx'] == c['qualified_dx'])
    result['qualified_dx'] = c['qualified_dx']
    result['passed'] = (
        qualified['metrics']['maximum_complex_relative_error'] <= limits['maximum_complex_relative_error']
        and qualified['metrics']['maximum_amplitude_relative_error'] <= limits['maximum_amplitude_relative_error']
        and max(result['sensitivity'].values()) <= limits['maximum_nuisance_relative_change']
        and reference_change <= limits['maximum_reference_quadrature_relative_change']
    )
    result['status'] = 'passed_specified_source_scope' if result['passed'] else 'failed_specified_source_scope'
    save(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT/'quantitative-v2.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['point', 'line', 'all'], default='all')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    policy_bytes = args.policy.read_bytes()
    policy = json.loads(policy_bytes)
    torch.set_num_threads(1)
    report = {'engine': __version__, 'python': platform.python_version(), 'platform': platform.platform(),
              'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
              'device': args.device, 'dtype': 'float32', 'policy': policy,
              'policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'engine_commit': subprocess.check_output(['git', '-C', str(ROOT.parent), 'rev-parse', 'HEAD'], text=True).strip(),
              'selected_scope_passed': False, 'mac01_complete': False, 'results': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(kind, result):
        report['results'][kind] = result
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for kind in ['point', 'line'] if args.kind == 'all' else [args.kind]:
        try:
            run_kind(policy, kind, args.device, lambda result: save(kind, result))
        except Exception as error:
            save(kind, {'passed': False, 'status': 'error', 'error': str(error)})
            raise
    report['selected_scope_passed'] = all(r['passed'] for r in report['results'].values())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['selected_scope_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
