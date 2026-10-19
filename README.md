# WaveSimulator2D

**A PyTorch engine for scalar two-dimensional wave experiments.** Simulate propagation, interference, diffraction, refraction and damping with programmable scenes and passive point or line detectors.

WaveSimulator2D is the numerical engine behind [Wave Studio](https://github.com/flsmn/WaveStudio), the companion desktop editor. It also runs independently as a Python library, without Qt, OpenCV or a graphical environment. Scenes are prepared with NumPy and advanced with PyTorch on a selected compute device.

The current package is **`wave-simulator-2d==0.1.0.dev5`**, imported as **`wave_sim2d`**, and requires **Python 3.12**. This repository is an actively developed fork of [0x23/WaveSimulator2D](https://github.com/0x23/WaveSimulator2D), retaining its history and legacy examples. The current calibrated model changes several historical numerical conventions; old screenshots, benchmarks and compatibility reports are not current accuracy or performance guarantees.

## Features

- **Scalar finite-difference solver:** a calibrated nine-point spatial stencil, centred time stepping, float32 state and physical-rate damping.
- **Static scenes:** rectangular or polygonal refractive materials, rectangular opaque walls, a biconvex lens contour helper and a damped outer border.
- **Coherent excitation:** additive sinusoidal point and finite line sources with explicit wavelength, amplitude and phase, using spatially normalized forcing.
- **Passive measurements:** finite-window mean field-square profiles and live signed field samples, with physical coordinates and sampling metadata.
- **Explicit lifecycle:** deterministic scene preparation, exact step counts, complete reset and owned read-only CPU snapshots.
- **Backend selection:** CPU and Apple MPS paths, plus an explicit engine-level CUDA option subject to hardware validation.
- **Validation tools:** regression tests, frozen physical demonstration protocols, dispersion estimates, independent source references and a frequency-domain volume-reference study.

For an interactive scene editor, project files, plots and PNG/CSV exports, use [Wave Studio](https://github.com/flsmn/WaveStudio). This repository provides the solver and numerical tools.

## Comparison with the original repository

The original [0x23/WaveSimulator2D](https://github.com/0x23/WaveSimulator2D) provides the foundation: a GPU wave solver, extensible scene objects, image-based scenes and example visualizations. This fork builds on that work with a PyTorch backend, an explicit numerical contract, passive measurements and a separately packaged desktop workflow.

The table compares the original CuPy implementation preserved at [`16de78f`](https://github.com/flsmn/WaveSimulator2D/tree/16de78f7dd308f93fbd681fbafcbf7d2fc90b03a) with this fork's current supported prepared-scene API. It describes that fixed baseline rather than making claims about future upstream changes.

| Area | Original CuPy implementation | Current fork (`0.1.0.dev5`) |
| --- | --- | --- |
| Compute runtime | CuPy and CUDA-oriented GPU execution | PyTorch with explicit CPU, MPS and CUDA selection; CUDA hardware acceptance remains pending |
| Scene API | Mutable scene objects render coefficients and modify the field during updates | Validated static configuration and read-only prepared scenes, with solver-owned device state; legacy objects retained separately |
| Units and spatial operator | Pixel/time-step convention, `dt=1`, unnormalized nine-point kernel | Explicit `dx`/`dt`, reference speed `c0=1`, kernel divided by `0.316*dx²`, and a checked stability limit |
| Excitation | Point/line objects overwrite selected field cells with their sinusoidal displacement | Additive forcing with bilinear point weights and arc-length line quadrature; amplitude has an explicit integrated-forcing meaning |
| Damping | Per-step factors applied to the displacement increment | Nonnegative decay rates per project time, centred damping update and a physical-width border profile |
| Reset and field access | `reset_time()` resets the clock; `get_field()` returns the live array | `reset()` restores initial state and measurements; `snapshot()` returns an owned read-only CPU copy |
| Measurements | Visualizer field/intensity output | Passive point/line detectors, finite settling/windows, frozen mean field-square profiles, live signed samples and sampling metadata |
| Application and packaging | Example scripts with visualization and image-based scene input | Installable engine package plus the separate [Wave Studio](https://github.com/flsmn/WaveStudio) editor, project persistence and exports |
| Validation | Example-driven workflow | Regression suite, documented model, frozen demonstration criteria and independent reference studies with retained failures |

The original's dynamic scene objects and image-driven experiments remain useful historical examples. They are outside the current static prepared-scene validation scope. Porting an experiment requires revisiting its units, source amplitudes and damping; reproducing the same input numbers does not guarantee the same physical experiment.

### What the numerical comparison establishes

The repository also contains a reproducible comparison against the **first PyTorch port**, [`40f6c78`](https://github.com/flsmn/WaveSimulator2D/commit/40f6c78480583e465d59c56032849e6b1efe4480), which retained the original unnormalized update. This CPU comparison is distinct from executing the original CuPy solver on NVIDIA hardware.

For four homogeneous sine-mode cases (`n=1` and `n=1.5`, nearly axial and diagonal propagation), the retained [comparison report](docs/original-comparison-results.json) records:

| Quantity | Historical PyTorch baseline | Calibrated engine |
| --- | --- | --- |
| Vacuum phase speed at each version's default time step, `dx=1` | About 0.560 against the stated reference speed 1 | About 0.996–0.997 against reference speed 1 |
| Relative phase-speed error against `1/n`, across the four cases | About 44% | About 0.32–0.47% |

These values identify a correction to the declared speed scale. They do not mean every historical field had a 44% error. After the analytic time mapping `t_current = sqrt(0.316) * t_baseline`, the relative L2 field differences after 180 steps are only **6.38e-7 to 1.27e-6**, with maximum absolute difference **1.85e-6** for unit initial amplitude. This shows that the homogeneous recurrence remains closely related after unit calibration.

The retained report was generated with engine `0.1.0.dev4`; dev5 adds signed probe samples without changing the solver stencil. Reproduce the comparison with the current checkout using `python validation/compare_original.py --output /tmp/wave-original-comparison.json`. The test uses CPU/float32, one Torch thread, matched interiors and no sources, damping or internal walls. It bypasses the historical initial-field shape assertion while preserving the historical update. It establishes neither a GPU speedup nor a complete old/new comparison of slit, lens, source or boundary accuracy. See [methods and error definitions](docs/ERROR_METRICS.md) and the [open quantitative findings](docs/QUANTITATIVE_PROGRESS.md).

## Installation

Clone the repository and create a Python 3.12 environment:

```sh
git clone https://github.com/flsmn/WaveSimulator2D.git
cd WaveSimulator2D
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1` in PowerShell.

The core dependencies are **NumPy `>=1.26,<3`** and **PyTorch `>=2.5,<3`**. Install a PyTorch build suitable for your platform and intended backend. CuPy is not a dependency of the current engine.

Optional extras:

| Extra | Purpose | Install from this checkout |
| --- | --- | --- |
| `dev` | pytest and wheel build tooling | `python -m pip install -e ".[dev]"` |
| `validation` | SciPy for independent numerical references | `python -m pip install -e ".[validation]"` |
| `examples` | Historical OpenCV/Matplotlib visualizer dependencies | `python -m pip install -e ".[examples]"` |

For development and all reference tests, use `python -m pip install -e ".[dev,validation]"`. The `examples` extra currently selects headless OpenCV; see [legacy examples](#legacy-examples-and-compatibility) before running scripts that open windows.

## Quick start

The supported API is **configuration → `prepare_scene()` → `WaveSimulator2D.from_scene()` → `step()` / `snapshot()` / `reset()`**.

This example creates a 32 × 32 domain, a point source, a rectangular material and a detector line. It uses CPU explicitly and requires only the core package:

```python
import numpy as np
import torch

from wave_sim2d.config import Grid, Material, Source, rectangle
from wave_sim2d.geometry import prepare_scene
from wave_sim2d.measurements import Detector
from wave_sim2d.wave_simulation import WaveSimulator2D

# A reproducible CPU setting for this small example.
torch.set_num_threads(1)

grid = Grid(
    nx=256, ny=256,
    dx=0.125, dt=0.0625,
    border_width=4.0, border_rate=2.0,
)
scene = prepare_scene(
    grid,
    materials=[Material(rectangle(20.0, 8.0, 4.0, 16.0),
                        refractive_index=1.5)],
    sources=[Source((8.0, 16.0), wavelength=4.0, amplitude=1.0)],
)
for warning in scene.warnings:
    print(warning)

sim = WaveSimulator2D.from_scene(scene, device="cpu")
sim.measurements.configure(
    [Detector(
        id="screen",
        start=(26.0, 8.0), end=(26.0, 24.0), samples=65,
        settling=48.0, window=16.0,
    )],
    t=sim.t,
    field=sim.u,
)

sim.step(1280)  # Exactly 80 project time units.
field = sim.snapshot()  # Owned, read-only NumPy array with shape (ny, nx).
profile = sim.measurements.results()["screen"]

print("time:", sim.t, "steps:", sim.steps)
print("field shape:", field.shape, "measurement:", profile["status"])
print("maximum mean field square:", max(profile["values"]))
np.save("field.npy", field)

sim.reset()  # Restore initial displacement, time, step count and measurements.
```

The settling time and window above illustrate the API; they do not establish convergence or steady state. Configure suitable times and compare refinements for the experiment you intend to measure.

### Scene and lifecycle essentials

| API | Meaning |
| --- | --- |
| `Grid(nx, ny, dx, dt, ...)` | Square spatial cells, explicit time step and physical border settings |
| `Source(start, end=None, wavelength=..., amplitude=..., phase=...)` | Point source, or finite line when `end` is supplied; phase in radians |
| `Material(vertices, refractive_index=..., damping_rate=...)` | Simple polygon; `rectangle()` constructs rectangular vertices |
| `Wall(x, y, width, height)` | Axis-aligned rectangle constrained to zero displacement |
| `lens_vertices(x, y, radius, aperture, angle=0)` | Biconvex polygon helper in `wave_sim2d.geometry`; angle in radians |
| `prepare_scene(grid, materials=(), walls=(), sources=())` | Validate and rasterize a static scene into read-only CPU arrays |
| `WaveSimulator2D.from_scene(scene, device=..., initial_field=None)` | Create solver-owned device state |
| `sim.step(count=1)` | Advance exactly 1–10000 steps and check that the resulting field is finite |
| `sim.snapshot()` | Copy the current signed field to an owned, read-only CPU array |
| `sim.reset()` | Restore the original initial displacement, zero discrete initial velocity, time, step count and detector accumulators |

Later materials replace both index and material damping in overlapping cells. Walls take precedence over materials. Sources must remain in the usable interior and their interpolation support must avoid walls and fixed boundary cells. Rebuild the prepared scene and solver when physical geometry or coefficients change.

`sim.u` and the legacy `get_field()` expose borrowed device state. Keep solver and measurement operations under one owner and use `snapshot()` for independent field consumers. Invalid configurations, unavailable backends and non-finite states raise `ConfigurationError`, `BackendError` and `NumericalError`, respectively, from `wave_sim2d.config`.

## Model, units and resolution

The supported static scalar model is

```text
u_tt + gamma * u_t = (1 / n²) * Laplacian(u) + f
```

Reference speed is `c0 = 1`; lengths use fixed project reference units and time uses reference-length/c0. Arrays are indexed **`[y, x]`**, with x pointing right and y up. Cell centres are `((ix + 0.5) * dx, (iy + 0.5) * dx)`. The domain dimensions are `nx * dx` and `ny * dx`.

- **Materials:** `1 <= n <= 4`, with nonnegative damping rate `gamma` per unit project time. The model uses a variable mass coefficient, rather than a variable spatial flux coefficient; displacement and its normal derivative are continuous at ideal interfaces. Cell-centre rasterization approximates curved and oblique interfaces with staircase boundaries.
- **Sources:** wavelength is specified in the reference medium, with angular frequency `2*pi/wavelength`. Point amplitude is integrated delta-forcing strength; line forcing integrates along its physical length. Amplitude is not optical power or prescribed displacement. Finite lines radiate on both sides and diffract at their endpoints.
- **Boundaries:** wall cells and the outermost rows/columns impose zero displacement. A cubic damping profile can reduce returns near the outer boundary, but it is not a PML or a universal reflection-free boundary.
- **Time stepping:** the calibrated stencil requires `dt/dx < 0.92666...` for the admitted index range. `Grid()` defaults to `dx=0.125`, `dt=0.0625`. Stability is necessary but does not establish accuracy.
- **Resolution:** scene preparation warns below 12 cells per shortest material wavelength. Thin structures, long propagation paths, interfaces and finite domains can require substantially finer grids even above that threshold.

This model does not include vector electromagnetic fields, polarization, nonlinear or dispersive materials, absolute power calibration or full 3D optics. A visually plausible pattern is not a quantitative validation. See the [numerical contract](docs/MODEL.md) for the update equation, source normalization and compatibility details.

### Devices and memory

| `device` | Behavior and current scope |
| --- | --- |
| `"cpu"` | Explicit CPU execution; used for core numerical reference reports |
| `"mps"` | Apple Metal execution when available; companion-app checks exist on the local Apple Silicon Mac |
| `"cuda"` | Explicit CUDA execution when available; engine-level option without current native hardware acceptance |
| `"auto"` | Selects MPS when available, otherwise CPU; it does not automatically select CUDA |

An explicitly unavailable backend raises an error; runtime failures do not silently switch devices. No universal GPU speedup or cross-backend accuracy bound is claimed.

`Grid.max_axis_cells` defaults to **4096** and can be set explicitly. It is an admission limit, not a memory reservation. `Grid.estimated_bytes` provides a conservative numerical-buffer estimate, excluding PyTorch runtime and backend allocator overhead. Independent library users must budget their own memory; Wave Studio adds its own project-level buffer limit before raster allocation. Float32 is the supported solver state dtype.

## Passive detector measurements

Configure detectors through `sim.measurements.configure(specs, t=sim.t, field=sim.u)` and read them with `sim.measurements.results()`. At most **16 distinct detectors** are supported, with up to **2048 spatial samples** each. A point has one sample; a line includes both endpoints.

Each detector waits its `settling` time, then integrates over a finite `window`, measured from activation. Every numerical step contributes its endpoint sample weighted by overlap with that window, including fractional first and last intervals. Defaults of 32 and 16 project time units are starting suggestions; travel times, transients and beat periods may require longer observations.

Result fields include:

- `values`: the finite-window mean of **bilinearly interpolated field square**. During an incomplete window these are provisional; once complete, the intensity values remain frozen.
- `field_values` and `field_time`: instantaneous **signed, bilinearly interpolated field** and its snapshot time. These continue updating after intensity completion.
- `positions`, `axis`, status, detector revision, activation/window timing, temporal sample count, interval and method metadata.

Interpolating field square differs from squaring an interpolated signed field: adjacent opposite signs can cancel in the latter. The full-field `sim.measurements.intensity` is a separate exponential live preview with time constant 4, not a completed detector measurement. Display cadence does not affect detector integration. No full field history is retained.

Unchanged detector specifications retain their accumulators when reconfigured; changed detectors restart independently. Reset clears the measurement history. A completed window alone does not certify steady state or numerical accuracy.

## Tests and physical validation

Install the development and validation extras, then run the regression suite:

```sh
python -m pytest -q
```

Tests cover configuration, scene rasterization, source behavior, reset, dispersion, detector sampling, grid budgets and independent reference operators. The older top-level `test_comprehensive.py` and `test_pytorch_port.py` scripts are historical compatibility tools; the configured pytest suite lives in `tests/`.

### Historical physical demonstrations

```sh
python validation/run_references.py --output /tmp/wave-physical-results.json
# Run one reference only:
python validation/run_references.py --only T02 --output /tmp/wave-t02.json
```

| Protocol | Scope |
| --- | --- |
| T01 | Phase speed, driven wavelength and source normalization across grids |
| T02 | Refraction, physical damping and a measured border-return comparison |
| T03 | Double-slit minimum position, symmetry and zero field in wall cells |
| T04 | A fixed scalar lens demonstration against a paraxial back-focal estimate |

These tests retain their frozen **v1 demonstration tolerances**, including broad 15%/20% slit/lens limits. Passing them does not certify general quantitative accuracy. The initial failed border case and subsequent reports remain available. T02 now reports both nominal-incidence and measured-incidence Snell diagnostics. Read the [validation report](docs/VALIDATION.md), [protocol](validation/README.md) and [tolerance review](docs/TOLERANCE_REVIEW.md) when interpreting results.

### Independent quantitative follow-up

Validation-only SciPy tools compare source fields with outgoing Hankel references and study a frequency-domain volume-potential lens reference:

```sh
python validation/quantitative_sources.py --policy validation/quantitative-v2-wide.json --output /tmp/wave-sources.json
python validation/run_volume_references.py --policy validation/volume-reference-v1-followup.json --output /tmp/wave-volume.json
```

The point-source subcase passes its defined protocol. The line-source case fails window/border sensitivity criteria, and the complete lens-reference qualification fails its complex-profile refinement criterion. Both commands currently return a nonzero status for their incomplete overall scopes. They can require substantial CPU time and memory. Full quantitative qualification of arbitrary scenes and the application's slit/lens presets remains open. The [quantitative progress report](docs/QUANTITATIVE_PROGRESS.md) records parameters, results, retained failures and remaining work.

For a controlled comparison with the historical PyTorch solver:

```sh
python validation/compare_original.py --output /tmp/wave-original-comparison.json
```

This reads historical code from the local Git history, so it requires a checkout containing the referenced commits. See [error metrics and original-engine comparison](docs/ERROR_METRICS.md). Agreement with a related implementation is distinct from agreement with an independent physical reference.

## Legacy examples and compatibility

Historical scripts remain in [`wave_sim2d/examples/`](wave_sim2d/examples/), along with scene objects and an OpenCV/Matplotlib visualizer. They use the older `WaveSimulator2D(w, h, scene_objects)` interface. The calibrated stencil, default time step, additive source behavior and complete `reset_time()` semantics differ from their original implementation. Legacy frequency arguments remain angular frequencies, whereas the prepared-scene `Source` accepts wavelength.

Image-based scenes, dynamic strain objects and legacy per-step damping are outside the supported static prepared-scene contract and its physical acceptance scope. Historical images illustrate earlier experiments rather than validated current results.

Some example scripts use `cv2.imshow`, which requires a GUI-capable OpenCV build. The `examples` extra installs `opencv-python-headless`, so it does not provide window display by itself. If using those scripts, choose a suitable GUI-enabled OpenCV environment and avoid installing multiple OpenCV distributions together, especially alongside Qt. Use Wave Studio for the maintained desktop workflow.

The retained [PyTorch migration notes](PYTORCH_MIGRATION.md), [quick-start notes](QUICKSTART.md) and [completion report](COMPLETION_REPORT.md) describe earlier development stages and may contain superseded installation, speed or compatibility claims. The current API and model documentation take precedence.

## Repository layout and development

```text
WaveSimulator2D/
├── wave_sim2d/
│   ├── config.py             # Configuration, limits and errors
│   ├── geometry.py           # Static scene preparation and lens contours
│   ├── wave_simulation.py    # Solver, backend selection and lifecycle
│   ├── measurements.py       # Passive detector windows and field samples
│   ├── dispersion.py         # Homogeneous stencil/phase estimates
│   ├── scene_objects/        # Legacy example scene objects
│   └── examples/             # Historical visualizer scripts
├── tests/                    # Current regression suite
├── validation/               # Versioned protocols and reference runners
├── docs/                     # Model, studies and retained raw reports
└── images/                   # Historical experiment images
```

Build an installable wheel with:

```sh
python -m build --wheel
```

The wheel contains `wave_sim2d`; repository-level validation scripts and reports are separate development resources. Wave Studio pins the engine version and builds its desktop bundle from a matching engine wheel. Numerical solver work belongs here; editor, orchestration, project persistence and UI exports belong in [Wave Studio](https://github.com/flsmn/WaveStudio).

For issues or proposed changes, include engine version, Python/PyTorch versions, hardware, backend, grid/time-step settings and a minimal scene. For numerical discrepancies, include raw measurements, reference definitions and refinement/window/boundary checks. Run the relevant regression tests and preserve the scope and thresholds of historical reports.

## License

No top-level project license is currently included in this fork. The repository retains upstream contributions and history; dependency licenses apply separately. No new license or permission grant is asserted by this README.
