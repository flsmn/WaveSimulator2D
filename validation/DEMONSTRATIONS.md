# Historical demonstration measurements – 17 October 2026

**Review, 18 October 2026:** the PASS statuses below belong to historical v1 demonstration criteria. The 15/20-percent limits are not a quantitative-release accuracy certification; their numerical/model/reference uncertainties were not separated. See [TOLERANCE_REVIEW.md](TOLERANCE_REVIEW.md) and the retained dev4 inputs. Quantitative slit/lens accuracy remains open. No historical threshold or result was rewritten.

Engine 0.1.0.dev2; CPU float32, one Torch CPU thread, Python 3.12.12, Torch 2.5.1, macOS arm64. Raw measurements are in [results/physical-results.json](results/physical-results.json); the initial failed border configuration is retained in [results/physical-results-initial.json](results/physical-results-initial.json). Thresholds were frozen in commit `d06d04b` before measurements. Invoke `python validation/run_references.py` from the engine repository with its installed environment. A failed/pending reference exits nonzero.

| Reference | Measured | Frozen limit | Status |
| --- | --- | --- | --- |
| T01 axial/diagonal speed, two grids | 0.079–0.405% relative error | 2% | passed |
| T01 driven wavelength | 0.606–0.861% error | 5% | passed |
| T01 point-/line-source strength across grids | 0.448% / 2.880% difference | 15% | passed |
| T01 coherent sources / return wave | additive doubling, antiphase cancellation, incident-field superposition | absolute field tolerance 2e-6 in core checks | passed |
| T02 refraction | 15.718° vs 16.364° Snell reference; incident 25.623° vs 25° | 3° | passed |
| T02 physical damping rate, two grids | 0.20000027 / 0.20000077 vs 0.2 | 5% | passed |
| T02 reflected border amplitude, two grids | 0.482% / 0.904% (8-unit border) | 8% | passed |
| T03 finite-distance first minimum | 3.5 / 3.625 vs 3.752 project units | 15% | passed |
| T03 wall / symmetry | zero wall field; symmetry error < 2.2e-7 | 1e-7 wall field; 3% asymmetry | passed |
| T04 back focal distance | 19.180 / 20.827 vs paraxial 23.534 | 20% | passed for this coarse demonstration |
| T04 refinement / focus gain | 7% focus shift; 13.32× / 10.39× intensity gain | ≤10% shift; ≥1.5× gain | passed |

Nine core tests cover parameter errors, reset reproducibility, finite initial fields, explicit CPU, material overlap, moved geometry, wall transmission, coherent sources, incoming-wave transparency and spatial source normalization. The historical comprehensive script passes all nine groups; these are compatibility checks, not additional physics acceptance.

## Investigated deviations and limits

The first T02 line endpoint lay within the damped border and was correctly rejected. Its source centre was moved from x=15 to x=16 without changing the target angle. A six-unit damped border then returned 8.56% on dx=.25 (limit 8%) and 6.90% on dx=.125. The retained initial report records that failure. Widening the border to eight project units, with source/probe in the new usable interior, reduced measured return below 1%; the threshold was unchanged. This supports this particular wider border, not all border widths/rates. The prototype's smaller default border has no equivalent reflection promise.

T04 intentionally tests a scalar 2D lens against a paraxial thick-lens estimate. The coarse dx=.10/.08 grids resolve the shortest wavelength in the lens with only 5.3/6.7 cells. The measured 18.5%/11.5% focus errors pass the frozen broad 20% demonstration threshold, but do not support a precision optics or final quality claim. The warning below twelve cells per material wavelength remains active. The 7% refinement shift is a known limitation, not evidence of convergence to arbitrary accuracy. No more accurate lens preset is released from this result.

The full reference report covers only these static scalar scenes, finite line illumination and the documented sampling windows. CPU/MPS and native-platform checks belong to T08/T09 separately. No CUDA support is offered in Studio.

The four references were rerun after E08 with version 0.1.0.dev2. Passive reductions leave the numerical wave field unchanged; all frozen thresholds still pass. The additional line-source measurement uses fixed arc-length forcing and the same physical observation point across both grids.
