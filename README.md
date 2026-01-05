# LFM Gravitational Series: Reproducibility Package

**Papers 1-5 Experiment Reproduction**

This repository allows independent scientists to **reproduce the core experiments**
from the LFM (Lattice Field Medium) gravitational series. It is NOT a verification
harness that checks pre-computed results—it runs the actual models on real data.

## What This Package Does

1. **Galaxy Domain (Papers 1-2)**: Reconstructs χ-field from SPARC rotation curves,
   extrapolates to unseen regions, and predicts velocities
2. **Merger Domain (Papers 3-4)**: Computes DM-gas offset bounds for 12 merging
   clusters using the LFM nonlocal kernel
3. **Cosmology Domain (Paper 5)**: Evaluates GW speed, binary pulsar timing, and
   cosmological scale separation

All computations use **c_eff = 300 km/s** as the single locked constant.
No tunable parameters. No synthetic data.

## Quick Start

```bash
# Install (no external dependencies beyond Python 3.8+ standard library)
pip install -r requirements.txt

# Run all domains
python run_all.py

# Run specific domain
python run_all.py --galaxy
python run_all.py --merger
python run_all.py --cosmology
```

## Repository Structure

```
├── run_all.py              # Master script - runs all domains
├── requirements.txt        # Python dependencies (minimal)
├── README.md               # This file
│
├── galaxy/
│   ├── model.py            # Chi-field reconstruction equations
│   ├── reproduce.py        # Tests A, B, C from Papers 1-2
│   └── sparc_data/         # Real SPARC rotation curve data
│       └── galaxies.csv    # 91 galaxies, ~1700 data points
│
├── merger/
│   ├── model.py            # Offset prediction equations
│   └── reproduce.py        # 12-cluster offset analysis
│
├── cosmology/
│   └── reproduce.py        # GW speed, pulsar timing, BAO checks
│
└── outputs/
    ├── galaxy/             # Regenerated galaxy results
    ├── merger/             # Regenerated merger results
    ├── cosmology/          # Regenerated cosmology results
    ├── combined_report.json
    └── comparison_report.md
```

## Core Equations (LOCKED)

### Galaxy Domain

Chi-field reconstruction from rotation curve:
```
χ(r) = χ₀ exp[-2/c² ∫₀ʳ (v²/r') dr']
```

Velocity prediction from chi-field:
```
v²(r) = -(r c²/2) d(ln χ)/dr
```

### Merger Domain

Maximum DM-gas offset:
```
Δx_max = v_merger × r_c / σ
```

Quality ratio (must be ≤ 1):
```
Q = Δx_obs / Δx_max
```

### Cosmology Domain

LFM predicts:
- GW speed = c (chi doesn't couple to tensor modes)
- Binary pulsar timing = GR (chi doesn't modify geodesics)
- Coherence scale << BAO scale (no cosmological conflict)

## Data Sources

### SPARC Database (Galaxy Rotation Curves)

The galaxy analysis uses processed rotation curves from the **Spitzer Photometry
and Accurate Rotation Curves (SPARC)** database.

**Required Citation:**
> Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).
> "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves."
> *The Astronomical Journal*, 152(6), 157.
> DOI: [10.3847/0004-6256/152/6/157](https://doi.org/10.3847/0004-6256/152/6/157)

### Cluster Merger Data

Merger observations from published literature:
- Bullet Cluster: Clowe et al. (2006), Markevitch et al. (2004)
- MACS J0025: Bradac et al. (2008)
- Additional clusters: See reproduce.py for per-cluster citations

### Cosmological Constraints

- GW170817: Abbott et al. (2017)
- Hulse-Taylor pulsar: Taylor & Weisberg (1989)
- BAO scale: Eisenstein et al. (2005)

## Expected Output

After running `python run_all.py`, you should see:

```
================================================================================
FINAL SUMMARY
================================================================================
GALAXY       PASS
MERGER       PASS
COSMOLOGY    PASS

OVERALL: PASS
```

With detailed outputs in `outputs/`:
- Per-galaxy MAPE values (Test A)
- Per-cluster Q ratios (Merger)
- GW speed deviation (Cosmology)

## Falsification Criteria

**The LFM framework would be FALSIFIED if:**

1. **Galaxy**: Mean MAPE > 20% on out-of-sample predictions
2. **Merger**: Any cluster has Q > 1.5 (clear violation of bound)
3. **GW speed**: |c_g - c|/c > 10⁻¹⁴
4. **Pulsar timing**: Orbital decay deviates from GR by > 1%

**What would NOT falsify LFM:**
- Individual galaxies with high MAPE (some outliers expected)
- Q values between 0.8-1.0 (consistent with model as upper bound)
- Toy model limitations in EXP-1 (BBN) and EXP-5 (BAO)

## Limitations

- **EXP-1 (BBN)**: Toy model only—not a full nucleosynthesis calculation
- **EXP-4 (CMB)**: Not implemented (requires full Boltzmann code)
- **EXP-5 (BAO)**: Scale separation check only—not a full power spectrum

These partial tests establish *consistency*, not prediction.

## Differences from Paper

This reproduction may show small numerical differences from published values due to:
- Floating-point precision
- Integration method (trapezoid vs. other)
- Spline interpolation details

Differences should be < 1% for all metrics. Larger deviations indicate a bug.

## License

**Code**: MIT License

**Data**: SPARC data subject to original terms (http://astroweb.cwru.edu/SPARC/).
Merger cluster data from published literature (fair use for scientific reproduction).

## Contact

For questions about this reproduction package, open an issue on GitHub.
