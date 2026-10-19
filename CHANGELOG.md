# Changelog

## Unreleased

- Consolidated setup and original-engine comparisons in the README.
- Removed superseded porting reports and ad hoc test scripts. Legacy scene/visualizer compatibility is covered by optional pytest smoke checks.
- Separated selected immutable numerical evidence under `validation/results/` from small regression fixtures under `tests/fixtures/`.
- Local numerical runs default to ignored `local-results/`; historical protocols and tolerances are unchanged.

## 0.1.0.dev5

- Passive detectors expose signed bilinear `field_values` and `field_time`, including after intensity completion.
- Configure/reset can initialize signed samples from the current field. Intensity accumulation and solver stencil are unchanged.

## 0.1.0.dev4

- Added explicit grid-axis admission budgets without changing the numerical method.
- Documented tolerance limitations and a reproducible comparison with the historical PyTorch port.

## 0.1.0.dev3

- Added homogeneous dispersion estimates and bounded polygon rasterization.

## 0.1.0.dev2

- Established the installable calibrated engine API, additive normalized sources, physical damping, complete reset and passive finite-window detectors.
- Added frozen physical demonstration protocols; these are not general quantitative certification.

## Earlier history

The fork ports the original CuPy solver to PyTorch. Historical migration/completion documents and superseded numerical reports remain in Git history. The README and numerical contract define the supported behavior.
