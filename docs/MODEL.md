# Numerical contract v1 – 17 October 2026

Coordinates use fixed project reference lengths, reference speed c0 = 1 and time in reference-length/c0. Arrays are `[y, x]`; x points right and y up. Cell centres are `((ix+.5) dx, (iy+.5) dx)`. The usable interior excludes a damped border whose physical width stays fixed when dx changes. float32 is the supported state dtype. An explicit unavailable backend fails; no runtime fallback. Auto currently selects MPS when available, otherwise CPU. CUDA is engine-only until hardware validation.

The existing nine-point stencil K has axial weights .184, diagonal .066, centre -1. Its long-wave coefficient is `.184 + 2*.066 = .316`; the calibrated Laplacian is `K / (.316 dx²)`. Its negative Fourier symbol is bounded by `1.472/.316`. Thus `cmax dt/dx < 2/sqrt(1.472/.316) = 0.92666…`. This corrects the historical uncalibrated speed claim. We require n in [1,4], making cmax ≤ 1, and reject unstable steps before allocating the solver. The initial default Courant number is .5; this is not a final quality preset.

The scalar model is `u_tt + gamma u_t = (1/n²) Δu + f`. This is the mass-density form `n² u_tt = Δu` without a variable flux coefficient; displacement and normal derivative are continuous at material interfaces. Cell-centre sampling assigns n to each cell and evaluates 1/n² there; no arbitrary alpha blending of n, c or c² is used. This staircase approximation has grid error, especially for curved and oblique interfaces. No subcell accuracy is claimed. Topmost material replaces both n and gamma. Rectangular wall cells and the outermost row/column impose u=0 on both time levels, independently of material order. Scene preparation is rebuilt on every physical change; there is no geometry cache keyed only by dimensions.

Damping gamma is a nonnegative rate per project time, not a factor per step. The centred update is

`u_next = [2u - (1-gamma dt/2)u_prev + (dt/n)² Δu + dt² f(t)] / (1+gamma dt/2)`.

Border gamma is `border_rate * max(0,1-distance_to_edge/border_width)^3`, added to material gamma. This is a damped border, not a PML. Excessively large rates are stable but can reflect and can underresolve temporal decay. Reflection and resolution equivalence require E07 measurements.

Point sources approximate `A sin(2πt/λ + phase) δ(x-x0) δ(y-y0)`. Bilinear weights sum to `1/dx²`, so A is integrated forcing strength, not a prescribed displacement. Lines approximate an arc-length integral of the same delta forcing with trapezoidal quadrature at spacing ≤ dx/2; their weights integrate to line length. Contributions to repeated cells are summed before device transfer. Excitation is additive at time t in the centred update. Wavelength λ is measured in the reference medium, phase in radians. A line radiates on both sides and has finite-end diffraction. Sources whose interpolation support touches a wall, fixed boundary or the damped border are rejected. Source-free incoming waves therefore pass through the source location without being overwritten.

Reset restores the original initial displacement and identical previous displacement (zero discrete initial velocity), steps=0 and t=0. With the default zero initial field, reset removes all wave history. Nonzero initial fields are for numerical development; their first step has the usual discrete initialization phase error. `step(count)` performs exactly count steps, then checks finiteness. `snapshot()` is an owned read-only CPU copy. Borrowed device tensors belong exclusively to the solver's owner.

Fewer than 12 cells per shortest material wavelength produces a warning, not a promise of accuracy above that threshold. Finite positive wavelengths, finite amplitudes ≥ 0, positive cell/time spacing and simple nondegenerate polygons are required. Structure size and long propagation paths may require finer grids even above 12 cells/wavelength. No physical acceptance is inferred from stability or visually plausible images.

## Compatibility

Historical `WaveSimulator2D(w,h,scene_objects)` examples keep their imports, but now use the calibrated stencil, dt=.5 dx by default, one simulator-owned device, additive point/line excitation and complete `reset_time()`. Frequency in those legacy objects remains angular frequency. Old images, dynamic strain objects, per-step damping and legacy point/line discretization are outside the Studio contract and T01–T04 acceptance. Studio calls `prepare_scene()` and `WaveSimulator2D.from_scene()`; both paths share this one solver. Existing example results will differ and are not accuracy references. OpenCV and Matplotlib are optional example dependencies.

## Measurement contract (0.1.0.dev2 / E08)

At most sixteen passive detectors sample the bilinearly interpolated **field square** at fixed project coordinates after every numerical step. This avoids cancellation from interpolating oppositely signed adjacent field values before squaring. Line samples include both endpoints; their physical positions and distance axis are stored. Detector interpolation indices, weights and sums reside on the same device as the solver.

Each detector has settling time and a finite window measured from its acknowledged activation time. Defaults are 32 and 16 project time units, both editable. These are starting suggestions, not an automatic claim of steady state; propagation and beat periods may require longer windows. Each step contributes its endpoint sample weighted by the overlap of that step interval with the measurement window, including fractional first/last intervals. Completed values remain frozen. Sample count and numerical dt describe the actual temporal sampling. Only the changed detector's accumulator is replaced at a block boundary; unchanged detectors retain their state. Reset clears all accumulators.

The full-field intensity view is a bounded exponential live preview (physical time constant 4). It is explicitly separate from finite-window detector measurements and is never used for quantitative CSV or held references. Display cadence, color scale, contrast and discarded frames have no influence on either numerical sampling or raw results. No field history is stored.

## dev3 Mac quality preflight

`dispersion.py` exposes the calibrated nine-point Fourier symbol, its centred-time eigenfrequency and a driven-wave phase-drift estimate sampled at nine directions from axial to diagonal. The estimate assumes a homogeneous medium at the supplied index; it cannot bound interface staircasing, arbitrary concave necks, edge diffraction or finite-domain reflections. Studio combines this estimate with wavelength, geometry and resource guards. Measured eigenmodes at two directions, two spacings and indices 1/1.5 agree within 0.1% and remain within the original 2% speed limit (`tests/test_dispersion.py`).

Polygon rasterization now evaluates only its padded bounding box. The coordinates, cell-centre inclusion rule and resulting mask remain identical to the previous full-domain implementation; regression cases include rotated thin, concave and lens polygons. This makes high-resolution lens compilation practical without changing the continuous contour or the numerical solver.
