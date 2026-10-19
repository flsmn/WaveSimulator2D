"""Reproduce limited CPU comparisons with the first PyTorch fork.

No source, damping or internal-wall/lens acceptance is inferred from this probe.
It executes the historical update loaded from this repository's fixed Git ref.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import types

import numpy as np
import torch
from wave_sim2d import __version__
from wave_sim2d.config import Grid, Material, rectangle
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.wave_simulation import WaveSimulator2D

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '40f6c78'
STEPS = 180
SCALE = .316


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def measured_speed(history, dt, wave_number):
    a = np.asarray(history)
    cosine = np.dot(a[1:-1], a[:-2] + a[2:]) / (2 * np.dot(a[1:-1], a[1:-1]))
    return math.acos(float(cosine)) / (dt * wave_number)


def compare(legacy, index, modes, current_dt, compare_fields):
    mx, my = modes
    q = np.arange(81)
    initial = (np.sin(mx*np.pi*q/80)[None, :] * np.sin(my*np.pi*q/80)[:, None]).astype('float32')
    # Equivalent interior: old zero-padded ghost cells; new fixed zero edge.
    old = legacy.WaveSimulator2D(79, 79, [])
    old.c.fill_(1/index)
    # Bypass the unrelated historical initial_field shape assertion, without
    # changing its update. Constant coefficients need no scene rendering.
    old.u.copy_(torch.tensor(initial[1:-1, 1:-1]))
    old.u_prev.copy_(old.u)
    g = Grid(81, 81, 1., current_dt, 0, 0)
    scene = prepare_scene(g, [Material(rectangle(0, 0, g.width, g.height), index)])
    current = WaveSimulator2D.from_scene(scene, device='cpu', initial_field=initial)
    old_history, new_history = [], []
    for _ in range(STEPS):
        old.update_field()
        current.step()
        old_history.append(float(np.sum(old.u.numpy()*initial[1:-1, 1:-1], dtype=np.float64)))
        new_history.append(float(np.sum(current.snapshot()*initial, dtype=np.float64)))
    row = {'index': index, 'mode': list(modes), 'reference_speed': 1/index}
    if compare_fields:
        a = old.u.numpy().astype('float64')
        b = current.snapshot()[1:-1, 1:-1].astype('float64')
        row.update(relative_L2=float(np.linalg.norm(b-a)/np.linalg.norm(a)),
                   maximum_absolute_field_difference=float(np.max(abs(b-a))))
    else:
        k = math.pi*math.hypot(mx, my)/80
        a = measured_speed(old_history, old.dt, k)
        b = measured_speed(new_history, current.dt, k)
        row.update(fork_speed=a, current_speed=b,
                   fork_relative_speed_error=abs(a*index-1),
                   current_relative_speed_error=abs(b*index-1))
    if not all(math.isfinite(value) for value in row.values() if isinstance(value, float)):
        raise RuntimeError('Non-finite comparison result')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('local-results/original-comparison-results.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = git('show', f'{BASELINE}:wave_sim2d/wave_simulation.py')
    legacy = types.ModuleType('original_fork')
    exec(compile(source, f'{BASELINE}/wave_simulation.py', 'exec'), legacy.__dict__)
    legacy._get_device = lambda: torch.device('cpu')
    report = {
        'engine': __version__, 'baseline_commit': git('rev-parse', BASELINE),
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'current_source_sha256': hashlib.sha256((ROOT/'wave_sim2d/wave_simulation.py').read_bytes()).hexdigest(),
        'pre_pytorch_commit': git('rev-parse', f'{BASELINE}^'),
        'pre_pytorch_cupy_runtime_executed': False,
        'platform': platform.platform(), 'torch': torch.__version__,
        'backend': 'cpu', 'dtype': 'float32', 'threads': 1, 'steps': STEPS,
        'domain_length': 80, 'spacing': 1,
        'method': 'Homogeneous sine modes; matched interior, no sources, damping or internal walls. Old u/u_prev initialized directly to bypass its unrelated shape assertion. Historical update itself unchanged; force CPU and retain constant c=1/n.',
        'declared_units': {'old_dt': 1., 'current_dt': .5, 'results': []},
        'calibrated_units': {'old_dt': 1., 'current_dt': math.sqrt(SCALE),
                             'mapping': 't_current=sqrt(0.316)*t_fork at dx=1', 'results': []},
        'interpretation': 'Declared-unit errors test the claimed speed scale. L2 after analytic unit calibration tests recurrence agreement, not absolute physical accuracy. No full T01-T04 old/new or native CuPy/CUDA runtime comparison is claimed. This is a measured comparison, not a new tolerance policy or release-acceptance PASS.'
    }
    for index in (1., 1.5):
        for modes in ((8, 1), (6, 6)):
            report['declared_units']['results'].append(compare(legacy, index, modes, .5, False))
            report['calibrated_units']['results'].append(compare(legacy, index, modes, math.sqrt(SCALE), True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
