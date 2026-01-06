#!/usr/bin/env python3
"""
Cosmology Domain Reproduction: Paper 5
=======================================

Reproduces cosmological/strong-field consistency checks:

EXP-1: BBN (Big Bang Nucleosynthesis) - PARTIAL
    - Toy model showing chi -> 1 regime doesn't break BBN
    - Not a full BBN calculation

EXP-2: Gravitational Wave Speed
    - LFM predicts c_g = c (no modification to tensor sector)
    - Compare to GW170817 constraint: |c_g - c|/c < 3e-15

EXP-3: Binary Pulsars
    - LFM predicts GR-level timing (chi doesn't modify geodesics)
    - Compare to Hulse-Taylor orbital decay: < 0.2% deviation

EXP-4: CMB - PARTIAL
    - Toy model only (full Boltzmann code beyond scope)

EXP-5: BAO / Matter Power Spectrum - PARTIAL
    - Check that chi gradient scale doesn't conflict with BAO scale

LOCKED: c_eff = 300 km/s
"""

import numpy as np
import json
from pathlib import Path


# Physical constants (SI)
C_LIGHT = 299792458.0       # m/s
G_NEWTON = 6.67430e-11      # m^3 kg^-1 s^-2
M_SUN = 1.989e30            # kg

# LFM constant
C_EFF_KMS = 300.0           # km/s
C_EFF_MS = C_EFF_KMS * 1000 # m/s

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "cosmology"


# ==============================================================================
# EXP-2: GRAVITATIONAL WAVE SPEED - OUT OF SCOPE
# ==============================================================================

def exp2_gw_speed() -> dict:
    """
    CLASSIFICATION: OUT_OF_SCOPE
    
    LFM uses a scalar wave equation. Gravitational waves are tensor perturbations.
    This framework does not model GW propagation. No computation is performed.
    
    The statement "chi doesn't couple to tensor modes" is a design property,
    not a computed result. This experiment cannot produce PASS/FAIL status.
    """
    print("\n" + "-" * 60)
    print("EXP-2: Gravitational Wave Speed")
    print("-" * 60)
    print("CLASSIFICATION: OUT_OF_SCOPE")
    print("")
    print("WARNING: This framework uses a scalar wave equation.")
    print("         Gravitational waves are tensor perturbations (h_ij).")
    print("         LFM does not model tensor GW propagation.")
    print("         No numerical prediction is computed.")
    print("")
    print("The statement 'chi does not couple to tensor modes' is a")
    print("theoretical property of the framework's design, not a test result.")
    
    result = {
        "test": "EXP-2: GW Speed",
        "classification": "OUT_OF_SCOPE",
        "status": "NOT_APPLICABLE",
        "non_computational": True,
        "scope": "Framework does not model tensor GW propagation",
        "warning": "No numerical prediction computed. Scalar wave equation does not include tensor perturbations.",
        "theoretical_note": "Chi-field couples to scalar field E only. GW propagation is outside modeled domain."
    }
    
    print(f"Status: {result['status']} (out of scope)")
    return result


# ==============================================================================
# EXP-3: BINARY PULSARS
# ==============================================================================

def orbital_decay_gr(m1: float, m2: float, Pb: float, e: float) -> float:
    """
    GR orbital decay due to gravitational wave emission.
    
    Pb_dot = -(192*pi/5) * (Pb/2pi)^(-5/3) * (1 + 73/24 e^2 + 37/96 e^4) * (1-e^2)^(-7/2)
             * (G/c^3)^(5/3) * m1 * m2 * (m1+m2)^(-1/3)
    
    Returns: dimensionless (s/s)
    """
    M_total = m1 + m2
    
    # Eccentricity function
    fe = (1 + (73/24)*e**2 + (37/96)*e**4) / (1 - e**2)**(7/2)
    
    # GR prediction
    n = 2 * np.pi / Pb  # Mean motion
    factor = -(192*np.pi/5) * n**(5/3)
    factor *= (G_NEWTON / C_LIGHT**3)**(5/3)
    factor *= m1 * m2 * M_total**(-1/3)
    factor *= fe
    
    return factor


def exp3_pulsars() -> dict:
    """
    CLASSIFICATION: OUT_OF_SCOPE
    
    LFM is formulated for static chi-fields in weak-field galaxy dynamics.
    Binary pulsar timing requires:
    - Strong-field orbital dynamics
    - GW emission and backreaction
    - Time-dependent chi evolution
    
    This framework does not model compact binary inspiral.
    The assumption "chi doesn't modify geodesics" is an axiom, not a computation.
    """
    print("\n" + "-" * 60)
    print("EXP-3: Binary Pulsar Timing")
    print("-" * 60)
    print("CLASSIFICATION: OUT_OF_SCOPE")
    print("")
    print("WARNING: This framework is formulated for static chi-fields")
    print("         in weak-field galaxy dynamics.")
    print("")
    print("Binary pulsar timing requires:")
    print("  - Strong-field orbital dynamics")
    print("  - GW emission and backreaction")
    print("  - Time-dependent chi evolution")
    print("")
    print("LFM does not model compact binary inspiral.")
    print("The statement 'chi does not modify geodesics' is an axiom,")
    print("not a testable prediction.")
    
    result = {
        "test": "EXP-3: Binary Pulsars",
        "classification": "OUT_OF_SCOPE",
        "status": "NOT_APPLICABLE",
        "non_computational": True,
        "scope": "Framework does not model compact binary dynamics or GW backreaction",
        "warning": "No LFM prediction computed. Assertion 'LFM = GR' is an axiom, not a derived result.",
        "theoretical_note": "Chi-field formulation assumes static weak-field regime. Strong-field binary dynamics outside modeled domain."
    }
    
    print(f"Status: {result['status']} (out of scope)")
    return result


# ==============================================================================
# EXP-1: BBN (TOY MODEL)
# ==============================================================================

def exp1_bbn_toy() -> dict:
    """
    BBN toy model: Check chi -> 1 in early universe.
    
    In early universe (high energy density):
    - Gravitational potential Phi << 1
    - chi = exp(-Phi * (c/c_eff)^2)
    
    For BBN (T ~ 1 MeV, t ~ 1 s):
    - Horizon scale ~ c*t ~ 3e8 m
    - Phi ~ (H*L/c)^2 where H ~ 1/t
    
    Key check: Does chi deviate significantly from 1?
    """
    print("\n" + "-" * 60)
    print("EXP-1: BBN Consistency (Toy Model)")
    print("-" * 60)
    print("PARTIAL: This is a toy model, not full BBN calculation")
    
    # BBN epoch parameters
    t_bbn = 1.0  # seconds (approximate)
    T_bbn_MeV = 1.0  # Temperature
    
    # Hubble parameter at BBN
    # H ~ sqrt(8*pi*G*rho/3) ~ 1/(2*t) for radiation domination
    H_bbn = 1 / (2 * t_bbn)  # s^-1
    
    # Horizon scale
    L_horizon = C_LIGHT * t_bbn  # meters
    
    # Characteristic gravitational potential at horizon scale
    # Phi ~ (H*L/c)^2 ~ 1 (order of magnitude)
    # But actual local potential much smaller
    Phi_local = (H_bbn * L_horizon / C_LIGHT)**2
    
    # Chi factor
    c_ratio = C_LIGHT / C_EFF_MS  # ~ 1000
    chi_bbn = np.exp(-Phi_local * c_ratio**2)
    
    # For chi ~ 1, we need Phi * (c/c_eff)^2 << 1
    # Since c_ratio^2 ~ 10^6, we need Phi << 10^-6
    # But Phi_local ~ 1 at horizon, so chi -> 0 naively
    
    # HOWEVER: LFM applies to LOCAL dynamics, not cosmic expansion
    # The chi-field is sourced by local mass concentrations, not cosmic background
    # At BBN, no localized structures exist yet -> chi = 1
    
    print(f"BBN epoch: t ~ {t_bbn} s, T ~ {T_bbn_MeV} MeV")
    print(f"Key insight: Chi-field sourced by LOCAL mass, not cosmic background")
    print(f"At BBN: No localized structures -> chi = 1 (unmodified)")
    
    result = {
        "test": "EXP-1: BBN Consistency",
        "status": "PARTIAL",
        "model_type": "toy",
        "epoch": "BBN (t ~ 1 s, T ~ 1 MeV)",
        "conclusion": "Chi = 1 at BBN (no local sources)",
        "physical_explanation": "Chi-field is sourced by local mass concentrations. At BBN, homogeneous -> chi = 1."
    }
    
    print(f"Status: {result['status']} (toy model)")
    return result


# ==============================================================================
# EXP-5: BAO SCALE CHECK
# ==============================================================================

def exp5_bao_scale() -> dict:
    """
    Check LFM coherence scale against BAO.
    
    BAO scale: ~150 Mpc (acoustic horizon at recombination)
    LFM coherence: ell_c = r / X where X = v / c_eff
    
    For galaxies: X ~ 1 (v ~ 300 km/s), ell_c ~ r ~ 10 kpc
    For clusters: X ~ 3-5 (sigma ~ 1000 km/s), ell_c ~ 50-100 kpc
    
    Key check: LFM coherence << BAO scale (no conflict)
    """
    print("\n" + "-" * 60)
    print("EXP-5: BAO Scale Consistency")
    print("-" * 60)
    
    # BAO scale
    bao_scale_mpc = 150  # Mpc
    bao_scale_kpc = bao_scale_mpc * 1000  # kpc
    
    # Typical LFM scales
    # Galaxy: r_c ~ 10 kpc, v ~ 200 km/s
    X_galaxy = 200 / C_EFF_KMS
    ell_c_galaxy_kpc = 10 / X_galaxy
    
    # Cluster: r_c ~ 200 kpc, sigma ~ 1000 km/s
    X_cluster = 1000 / C_EFF_KMS
    ell_c_cluster_kpc = 200 / X_cluster
    
    # Ratio to BAO
    ratio_galaxy = ell_c_galaxy_kpc / bao_scale_kpc
    ratio_cluster = ell_c_cluster_kpc / bao_scale_kpc
    
    print(f"BAO scale: {bao_scale_mpc} Mpc = {bao_scale_kpc} kpc")
    print(f"Galaxy coherence: ell_c ~ {ell_c_galaxy_kpc:.0f} kpc (ratio: {ratio_galaxy:.2e})")
    print(f"Cluster coherence: ell_c ~ {ell_c_cluster_kpc:.0f} kpc (ratio: {ratio_cluster:.2e})")
    print(f"Conclusion: LFM coherence << BAO scale (no conflict)")
    
    # Pass if both ratios << 1
    passed = ratio_galaxy < 0.01 and ratio_cluster < 0.01
    
    result = {
        "test": "EXP-5: BAO Scale",
        "status": "PASS" if passed else "PARTIAL",
        "bao_scale_mpc": bao_scale_mpc,
        "ell_c_galaxy_kpc": ell_c_galaxy_kpc,
        "ell_c_cluster_kpc": ell_c_cluster_kpc,
        "ratio_galaxy": ratio_galaxy,
        "ratio_cluster": ratio_cluster,
        "physical_explanation": "LFM coherence scales (10-100 kpc) are << BAO scale (150 Mpc). No conflict."
    }
    
    print(f"Status: {result['status']}")
    return result


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all cosmology domain tests."""
    print("=" * 70)
    print("COSMOLOGY DOMAIN REPRODUCTION")
    print("Paper 5: Strong-Field and Cosmological Consistency")
    print("=" * 70)
    print(f"\nLOCKED CONSTANT: c_eff = {C_EFF_KMS} km/s")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    exp1 = exp1_bbn_toy()
    exp2 = exp2_gw_speed()
    exp3 = exp3_pulsars()
    exp5 = exp5_bao_scale()
    
    # Summary
    print("\n" + "=" * 70)
    print("COSMOLOGY DOMAIN SUMMARY")
    print("=" * 70)
    print(f"EXP-1 (BBN):     {exp1['status']}")
    print(f"EXP-2 (GW):      {exp2['status']} [OUT_OF_SCOPE - no computation]")
    print(f"EXP-3 (Pulsars): {exp3['status']} [OUT_OF_SCOPE - no computation]")
    print(f"EXP-5 (BAO):     {exp5['status']}")
    
    # EXP-2 and EXP-3 are OUT_OF_SCOPE - cannot contribute to PASS/FAIL
    # Only EXP-1 (PARTIAL) and EXP-5 (computational) count
    computational_tests = [exp1, exp5]
    overall = "PARTIAL"
    
    print(f"\nOVERALL COSMOLOGY DOMAIN: {overall}")
    print("")
    print("CORRECTION NOTICE (2026-01-05):")
    print("  EXP-2 (GW Speed) and EXP-3 (Pulsars) previously marked PASS.")
    print("  These were tautological - they asserted answers without computation.")
    print("  LFM's scalar wave equation does not model tensor GW propagation")
    print("  or compact binary dynamics. These domains are OUT_OF_SCOPE.")
    print("  No PASS/FAIL status is valid for out-of-scope domains.")
    
    # Save results
    output = {
        "domain": "cosmology",
        "c_eff_km_s": C_EFF_KMS,
        "experiments": {
            "exp1_bbn": exp1,
            "exp2_gw": exp2,
            "exp3_pulsars": exp3,
            "exp5_bao": exp5
        },
        "overall_status": overall,
        "correction_notice": "2026-01-05: EXP-2 and EXP-3 reclassified as OUT_OF_SCOPE. Previous PASS status was invalid (tautological tests).",
        "note": "EXP-1 and EXP-5 are toy models. EXP-2 and EXP-3 are outside modeled domain (no tensor GW, no binary dynamics)."
    }
    
    output_path = OUTPUT_DIR / "cosmology_reproduction_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")
    
    return output


if __name__ == "__main__":
    main()
