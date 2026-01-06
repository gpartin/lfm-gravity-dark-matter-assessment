# Changelog

All notable changes to the LFM Paper 037 Reproduction Package.

## [1.3.0] - 2026-01-05

### Added

- **orbital/**: Paper 040 orbital motion emergence experiments
  - Exp 1: Trajectory curvature in χ-field
  - Exp 2: Mass independence (amplitude-invariant trajectories)
  - Exp 3: Orbital stability (bound orbits in χ-potential)
  - Exp 4: χ removal test (curvature vanishes when χ→0)
  - Governing equation: ∂²E/∂t² = c²∇²E - χ(x,y)²E (LOCKED)
  - Uses Klein-Gordon wave equation with Störmer-Verlet integration

- **run_all.py**: Added `--orbital` flag for Paper 040 experiments

## [1.2.0] - 2026-01-05

### Added

- **btfr_predicted_test.py**: Critical BTFR test using PREDICTED velocities
  - Test ID: BTFR-V13-ALL-SPARC
  - Computes BTFR from log(v_flat_PREDICTED) vs log(M_b), NOT observed velocities
  - Validates v1.3 PASSES (slope 3.596) while v1.2 FAILS (slope 3.075)
  - Target range: [3.3, 4.0] (observed BTFR slope)

- **run_all.py**: Added `--btfr` flag for BTFR predicted velocity test

### Changed

- **Methodology clarification**: BTFR must be computed from model predictions to test
  whether the coupling law produces correct scaling. Previously computed from observed
  velocities, which tests nothing about the model.

## [1.1.0] - 2026-01-05

### Added

- **galaxy_v13_full_sparc/**: New v1.3 coupling law validation module
  - Validates on all 175 SPARC galaxies with rotation curve decompositions
  - Uses LOCKED constants: A = 10.0, a0 = 50.0 (km/s)²/kpc, c_eff = 300 km/s
  - Supports LFM-PAPER-039 reproduction
  
- **run_all.py**: Added `--galaxy-v13` flag for v1.3 full-SPARC validation

### Changed

- **run_all.py**: Updated docstring to document new LOCKED constants for v1.3

## [1.0.1] - 2025-01-XX

### Fixed

- **A6 closure bug in galaxy/reproduce.py**: The exponential decay extrapolation was using
  `r_train.max()` (the maximum radius in the training data, typically ~15-20 kpc) instead of
  the galaxy's actual extent `r_max_galaxy` (typically ~30-50 kpc). This caused the decay
  length `L = 2.0 * r` to be too short, leading to over-suppressed chi predictions and 99%+
  MAPE errors.

  **Root Cause**: Line `L = 2.0 * r.max()` in `extrapolate_chi_A6()` used training array
  limits rather than physical galaxy extent.

  **Fix**: Changed to `L = 2.0 * r_max_galaxy` where `r_max_galaxy` is passed as an explicit
  parameter from the full galaxy radial profile.

  **Result**: Galaxy Test A now passes 91/91 galaxies with Median MAPE = 10.62%, exactly
  matching Paper 034 Table 3.

### Changed

- **README.md**: Corrected Paper 003 (GEFF-*) test coverage to accurately reflect that
  these tests are NOT implemented in the current package. Previous documentation incorrectly
  claimed GEFF-ROT, GEFF-LENS, GEFF-TIME, GEFF-CROSS were included.

- **README.md**: Added explicit statement that only claim-carrying experiments are included
  and exploratory/diagnostic tests are intentionally excluded.

### Known Limitations

- Paper 003 (LFM-PAPER-003) GEFF-* tests are not currently implemented. These require:
  - DES Y3 weak lensing catalog data
  - H0LiCOW strong lensing time delay measurements
  - Cross-catalog matching infrastructure
  - Original experiment results are archived in the research workspace

## [1.0.0] - 2025-01-XX (Initial Public Release)

### Added

- Initial public reproduction package for LFM Gravitational Series Papers 1-5
- Galaxy domain: chi-reconstruction and rotation curve validation
- Merger domain: cluster offset bound testing
- Cosmology domain: GW speed and pulsar timing tests
- Combined reporting with JSON and Markdown outputs
