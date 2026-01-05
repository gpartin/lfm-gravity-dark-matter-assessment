# LFM Gravitational Series: Reproduction Report

**Generated**: 2026-01-05T06:07:09.506834

## Summary

| Domain | Status | Key Metric |
|--------|--------|------------|
| Galaxy | FAIL | Test A: 0% pass |
| Merger | PASS | Q_max = 0.47 |
| Cosmology | PASS | GW + Pulsar tests |

## Domain Details

### Galaxy Domain (Papers 1-2)

**Test A: Inner-fit -> Outer-predict**
- Train on r <= 0.70 r_max, predict r > 0.70 r_max
- PASS threshold: MAPE <= 15%
- Galaxies tested: 91
- Pass rate: 0.0%
- Mean MAPE: 22.72%

### Merger Domain (Papers 3-4)

**Offset Bound Test**
- Model: dx_max = v_merger x r_c / sigma
- PASS criterion: Q = dx_obs / dx_max <= 1 for all clusters
- Clusters tested: 12
- All Q <= 1: True
- Q range: [0.24, 0.47]

### Cosmology Domain (Paper 5)

**Strong-field consistency tests**
- EXP-1: BBN Consistency: PARTIAL
- EXP-2: GW Speed: PASS
- EXP-3: Binary Pulsars: PASS
- EXP-5: BAO Scale: PASS

## Falsification Criteria

The LFM framework would be falsified if:

1. **Galaxy domain**: Mean MAPE > 20% on holdout galaxies
2. **Merger domain**: Any cluster has Q > 1.5 (unambiguous violation)
3. **GW speed**: |c_g - c|/c > 10^-14 (order of magnitude above GW170817)
4. **Pulsar timing**: Orbital decay deviates from GR by > 1%

## Reproducibility Notes

- All computations use c_eff = 300 km/s (LOCKED, not tunable)
- No per-galaxy or per-cluster parameters
- Galaxy chi-reconstruction is parameter-free inversion
- Merger offset formula has zero free parameters
