# v1.3 Full-SPARC Validation

## Overview

This module validates the v1.3 coupling law against all 175 SPARC galaxies with
rotation curve decompositions.

## v1.3 Coupling Law (LOCKED)

```
G_eff = 1 + A * chi * (a0/g_N)^0.5
```

| Parameter | Value | Units | Status |
|-----------|-------|-------|--------|
| A | 10.0 | dimensionless | LOCKED |
| a0 | 50.0 | (km/s)²/kpc | LOCKED |
| c_eff | 300 | km/s | LOCKED |

## Data Requirements

This module requires SPARC rotation curve data with decomposed velocity components:

- **galaxy_id**: Galaxy identifier
- **radius_kpc**: Radius in kpc
- **V_obs_kms**: Observed rotation velocity
- **V_gas_kms**: Gas component velocity
- **V_disk_kms**: Stellar disk component velocity
- **V_bulge_kms**: Bulge component velocity

Data source: [SPARC Database](http://astroweb.cwru.edu/SPARC/)

## Usage

```bash
# Place sparc_profiles.csv in data/ directory first
python reproduce.py
```

## Outputs

- `outputs/summary.json` - Validation summary
- `outputs/galaxy_v13_results.json` - Per-galaxy metrics

## Validation Criteria

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Mean ratio | [0.5, 3.0] | v_pred/v_obs mean |
| Universality | < 0.50 | σ/μ scatter |
| BTFR slope | [3.3, 4.0] | d(log M)/d(log v) |

## Reference

LFM-PAPER-039: "Resolving the BTFR Constraint in χ-Mediated Gravity via Regime-Locked Coupling (v1.3)"
