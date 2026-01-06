#!/usr/bin/env python3
"""
BTFR Predicted Velocity Test (BTFR-V13-ALL-SPARC)
==================================================

This test validates that v1.3 coupling law predictions follow the correct 
Baryonic Tully-Fisher Relation (BTFR) scaling.

CRITICAL METHODOLOGY:
    The BTFR is computed from PREDICTED velocities, not observed.
    We fit log(v_flat_PREDICTED) vs log(M_b) and check if slope is in [3.3, 4.0].

v1.3 Coupling Law (LOCKED):
    G_eff = 1 + A * chi * (a0/g_N)^0.5
    A = 10.0, a0 = 50.0 (km/s)^2/kpc, c_eff = 300 km/s

EXPECTED RESULTS:
    | Source         | Slope | Status |
    |----------------|-------|--------|
    | Observed       | 3.566 | REF    |
    | v1.2 Predicted | 3.075 | FAIL   |
    | v1.3 Predicted | 3.596 | PASS   |

USAGE:
    python btfr_predicted_test.py

OUTPUT:
    outputs/btfr_predicted_results.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import linregress

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"

# =============================================================================
# LOCKED CONSTANTS
# =============================================================================
C_EFF = 300.0  # km/s
G_NEWTON = 4.302e-6  # kpc * (km/s)^2 / M_sun
A_V13 = 10.0  # LOCKED
A0_V13 = 50.0  # LOCKED (km/s)^2/kpc
UPSILON_DISK = 0.5
UPSILON_BULGE = 0.7


def geff_v12(chi: float, g_N: float) -> float:
    """v1.2: G_eff = 1 + 63.5 * chi"""
    return 1.0 + 63.5 * chi


def geff_v13(chi: float, g_N: float) -> float:
    """v1.3: G_eff = 1 + A * chi * (a0/g_N)^0.5"""
    if g_N <= 0:
        return 1.0
    return 1.0 + A_V13 * chi * (A0_V13 / g_N) ** 0.5


def compute_chi(r_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    Phi = -G_NEWTON * M_enc / r_kpc
    return np.exp(-Phi / C_EFF**2)


def compute_gN(r_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    return G_NEWTON * M_enc / r_kpc**2


def predict_velocity(r_kpc, M_enc, geff_func):
    chi = compute_chi(r_kpc, M_enc)
    g_N = compute_gN(r_kpc, M_enc)
    geff = np.array([geff_func(c, g) for c, g in zip(chi, g_N)])
    v_pred = np.sqrt(geff * G_NEWTON * M_enc / r_kpc)
    return v_pred


def analyze_galaxy(gal_id, gal_df):
    """
    Analyze single galaxy.
    Returns v_flat for observed, v1.2 pred, and v1.3 pred.
    """
    r = gal_df["radius_kpc"].values
    v_obs = gal_df["V_obs_kms"].values
    v_gas = gal_df["V_gas_kms"].values
    v_disk = gal_df["V_disk_kms"].values
    v_bulge = gal_df["V_bulge_kms"].values
    
    # Baryonic velocity and mass
    v_bar_sq = v_gas**2 + UPSILON_DISK * v_disk**2 + UPSILON_BULGE * v_bulge**2
    v_bar = np.sqrt(np.maximum(v_bar_sq, 0))
    M_enc = r * v_bar**2 / G_NEWTON
    
    # Predict velocities
    v_pred_v12 = predict_velocity(r, M_enc, geff_v12)
    v_pred_v13 = predict_velocity(r, M_enc, geff_v13)
    
    # Flat region = outer 30%
    flat_mask = r >= 0.7 * r.max()
    if flat_mask.sum() >= 2:
        v_flat_obs = np.mean(v_obs[flat_mask])
        v_flat_v12 = np.mean(v_pred_v12[flat_mask])
        v_flat_v13 = np.mean(v_pred_v13[flat_mask])
    else:
        v_flat_obs = v_obs[-1]
        v_flat_v12 = v_pred_v12[-1]
        v_flat_v13 = v_pred_v13[-1]
    
    M_baryonic = M_enc[-1]
    
    return {
        "galaxy_id": gal_id,
        "M_baryonic": float(M_baryonic),
        "log_Mb": float(np.log10(max(M_baryonic, 1e3))),
        "v_flat_obs": float(v_flat_obs),
        "v_flat_v12": float(v_flat_v12),
        "v_flat_v13": float(v_flat_v13),
        "log_v_obs": float(np.log10(max(v_flat_obs, 1.0))),
        "log_v_v12": float(np.log10(max(v_flat_v12, 1.0))),
        "log_v_v13": float(np.log10(max(v_flat_v13, 1.0))),
    }


def compute_btfr_fit(log_Mb, log_v, label):
    """Compute BTFR fit: log(v) vs log(M)"""
    slope, intercept, r_value, p_value, std_err = linregress(log_Mb, log_v)
    log_v_pred = slope * log_Mb + intercept
    scatter = np.std(log_v - log_v_pred)
    
    return {
        "label": label,
        "slope_vM": float(slope),
        "slope_Mv": float(1.0 / slope) if slope != 0 else 999.0,
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
        "scatter_dex": float(scatter),
        "n_points": len(log_Mb),
        "pass_btfr": bool(3.3 <= 1.0/slope <= 4.0) if slope > 0 else False,
    }


def main():
    print("=" * 70)
    print("BTFR PREDICTED VELOCITY TEST (BTFR-V13-ALL-SPARC)")
    print("=" * 70)
    print()
    print("METHODOLOGY: BTFR computed from PREDICTED velocities, not observed.")
    print(f"LOCKED: A = {A_V13}, a0 = {A0_V13}, c_eff = {C_EFF}")
    print()
    
    # Load data
    sparc_file = DATA_DIR / "sparc_profiles.csv"
    if not sparc_file.exists():
        # Try parent module data
        sparc_file = SCRIPT_DIR.parent / "galaxy_v13_full_sparc" / "data" / "sparc_profiles.csv"
    if not sparc_file.exists():
        print(f"ERROR: SPARC data not found. Run galaxy_v13_full_sparc/reproduce.py --fetch first.")
        return 1
    
    profiles_df = pd.read_csv(sparc_file)
    galaxy_ids = profiles_df["galaxy_id"].unique()
    print(f"Loaded {len(galaxy_ids)} galaxies from SPARC")
    print()
    
    # Analyze all galaxies
    results = []
    for gal_id in galaxy_ids:
        gal_df = profiles_df[profiles_df["galaxy_id"] == gal_id].sort_values("radius_kpc")
        if len(gal_df) < 3:
            continue
        if gal_df["radius_kpc"].min() <= 0:
            continue
        try:
            result = analyze_galaxy(gal_id, gal_df)
            if np.isfinite(result["log_Mb"]) and np.isfinite(result["log_v_v13"]):
                results.append(result)
        except Exception as e:
            print(f"  WARN: {gal_id} failed: {e}")
    
    print(f"Analyzed {len(results)} galaxies")
    print()
    
    # Extract arrays
    log_Mb = np.array([r["log_Mb"] for r in results])
    log_v_obs = np.array([r["log_v_obs"] for r in results])
    log_v_v12 = np.array([r["log_v_v12"] for r in results])
    log_v_v13 = np.array([r["log_v_v13"] for r in results])
    
    # Compute BTFR fits
    btfr_obs = compute_btfr_fit(log_Mb, log_v_obs, "Observed")
    btfr_v12 = compute_btfr_fit(log_Mb, log_v_v12, "v1.2 Predicted")
    btfr_v13 = compute_btfr_fit(log_Mb, log_v_v13, "v1.3 Predicted")
    
    # Print results
    print("BTFR RESULTS (from PREDICTED velocities)")
    print("-" * 70)
    print("| Source         | Slope | Scatter | R^2   | Target [3.3,4.0] | Status |")
    print("|----------------|-------|---------|-------|------------------|--------|")
    print(f"| Observed       | {btfr_obs['slope_Mv']:.3f} | {btfr_obs['scatter_dex']:.3f}   | {btfr_obs['r_squared']:.3f} | ---              | REF    |")
    print(f"| v1.2 Predicted | {btfr_v12['slope_Mv']:.3f} | {btfr_v12['scatter_dex']:.3f}   | {btfr_v12['r_squared']:.3f} | [3.3, 4.0]       | {'PASS' if btfr_v12['pass_btfr'] else 'FAIL'}   |")
    print(f"| v1.3 Predicted | {btfr_v13['slope_Mv']:.3f} | {btfr_v13['scatter_dex']:.3f}   | {btfr_v13['r_squared']:.3f} | [3.3, 4.0]       | {'PASS' if btfr_v13['pass_btfr'] else 'FAIL'}   |")
    print()
    
    # Save outputs
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    output = {
        "test_id": "BTFR-V13-ALL-SPARC",
        "timestamp": datetime.now().isoformat(),
        "n_galaxies": len(results),
        "constants": {"A": A_V13, "a0": A0_V13, "c_eff": C_EFF},
        "btfr_observed": btfr_obs,
        "btfr_v12_predicted": btfr_v12,
        "btfr_v13_predicted": btfr_v13,
        "verdict": {
            "v12_pass": btfr_v12["pass_btfr"],
            "v13_pass": btfr_v13["pass_btfr"],
            "v13_improvement": btfr_v13["pass_btfr"] and not btfr_v12["pass_btfr"],
        },
    }
    
    with open(OUTPUT_DIR / "btfr_predicted_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[OK] btfr_predicted_results.json")
    
    # Verdict
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"v1.2 predicted BTFR: {btfr_v12['slope_Mv']:.3f} -> {'PASS' if btfr_v12['pass_btfr'] else 'FAIL'}")
    print(f"v1.3 predicted BTFR: {btfr_v13['slope_Mv']:.3f} -> {'PASS' if btfr_v13['pass_btfr'] else 'FAIL'}")
    print()
    
    if btfr_v13["pass_btfr"] and not btfr_v12["pass_btfr"]:
        print("v1.3 RESOLVES BTFR constraint (v1.2 fails, v1.3 passes)")
        return 0
    elif btfr_v13["pass_btfr"]:
        print("Both pass - v1.3 maintains BTFR compliance")
        return 0
    else:
        print("FAIL - v1.3 does not pass BTFR constraint")
        return 1


if __name__ == "__main__":
    sys.exit(main())
