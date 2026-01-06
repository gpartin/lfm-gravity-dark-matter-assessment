#!/usr/bin/env python3
"""
v1.3 Full-SPARC Validation Module
==================================

Validates the v1.3 coupling law on all 175 SPARC galaxies.

v1.3 Coupling Law (LOCKED):
    G_eff = 1 + A * chi * (a0/g_N)^0.5
    
    where:
    - A = 10.0 (LOCKED)
    - a0 = 50.0 (km/s)^2/kpc (LOCKED)
    - c_eff = 300 km/s (LOCKED)

This module requires SPARC data with rotation curve decompositions.
Data source: sparc_profiles.csv with V_gas, V_disk, V_bulge columns.

USAGE:
    python reproduce.py --fetch    # Fetch SPARC data first
    python reproduce.py            # Run validation

OUTPUT:
    outputs/galaxy_v13_results.json
    outputs/btfr_fit.json
    outputs/summary.json
"""

import sys
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATA_DIR = SCRIPT_DIR / "data"

# SPARC data URL (public source)
SPARC_URL = "http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt"

# =============================================================================
# LOCKED CONSTANTS
# =============================================================================
C_EFF = 300.0  # km/s
G_NEWTON = 4.302e-6  # kpc * (km/s)^2 / M_sun
A_V13 = 10.0  # dimensionless amplitude (LOCKED)
A0_V13 = 50.0  # (km/s)^2/kpc acceleration scale (LOCKED)
EXP_V13 = 0.5  # exponent (LOCKED)
UPSILON_DISK = 0.5  # stellar disk M/L
UPSILON_BULGE = 0.7  # bulge M/L


def geff_v12(chi: float, g_N: float) -> float:
    """v1.2 baseline: G_eff = 1 + 63.5 * chi"""
    return 1.0 + 63.5 * chi


def geff_v13(chi: float, g_N: float) -> float:
    """v1.3 coupling: G_eff = 1 + A * chi * (a0/g_N)^0.5"""
    if g_N <= 0:
        return 1.0
    return 1.0 + A_V13 * chi * (A0_V13 / g_N) ** EXP_V13


def compute_chi(r_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    """chi = exp(-Phi / c_eff^2), Phi = -G*M/r"""
    Phi = -G_NEWTON * M_enc / r_kpc
    return np.exp(-Phi / C_EFF**2)


def compute_gN(r_kpc: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
    """g_N = G * M_enc / r^2"""
    return G_NEWTON * M_enc / r_kpc**2


def predict_velocity(r_kpc, M_enc, geff_func):
    """Predict v from G_eff coupling law."""
    chi = compute_chi(r_kpc, M_enc)
    g_N = compute_gN(r_kpc, M_enc)
    geff = np.array([geff_func(c, g) for c, g in zip(chi, g_N)])
    v_pred = np.sqrt(geff * G_NEWTON * M_enc / r_kpc)
    return v_pred, geff, chi, g_N


def analyze_galaxy(gal_id, gal_df, geff_func, version):
    """Analyze single galaxy."""
    r = gal_df["radius_kpc"].values
    v_obs = gal_df["V_obs_kms"].values
    v_gas = gal_df["V_gas_kms"].values
    v_disk = gal_df["V_disk_kms"].values
    v_bulge = gal_df["V_bulge_kms"].values
    
    # Baryonic velocity and mass
    v_bar_sq = v_gas**2 + UPSILON_DISK * v_disk**2 + UPSILON_BULGE * v_bulge**2
    v_bar = np.sqrt(np.maximum(v_bar_sq, 0))
    M_enc = r * v_bar**2 / G_NEWTON
    
    # Predict
    v_pred, geff, chi, g_N = predict_velocity(r, M_enc, geff_func)
    
    # Metrics
    ratio = v_pred / v_obs
    mean_ratio = np.mean(ratio)
    
    # Flat region (outer 30%)
    flat_mask = r >= 0.7 * r.max()
    v_flat_obs = np.mean(v_obs[flat_mask]) if flat_mask.sum() >= 2 else v_obs[-1]
    
    return {
        "galaxy_id": gal_id,
        "version": version,
        "n_points": len(r),
        "mean_ratio": float(mean_ratio),
        "v_flat_obs": float(v_flat_obs),
        "M_baryonic": float(M_enc[-1]),
        "log_Mb": float(np.log10(max(M_enc[-1], 1e3))),
        "log_v_flat": float(np.log10(max(v_flat_obs, 1.0))),
    }


def compute_btfr(results):
    """Compute BTFR slope from galaxy results."""
    from scipy.stats import linregress
    log_Mb = np.array([r["log_Mb"] for r in results])
    log_v = np.array([r["log_v_flat"] for r in results])
    slope, intercept, r_value, p_value, std_err = linregress(log_Mb, log_v)
    return {
        "slope_vM": float(slope),
        "slope_Mv": float(1.0/slope) if slope != 0 else 999.0,
        "r_value": float(r_value),
        "r_squared": float(r_value**2),
        "n_galaxies": len(results),
        "pass_btfr": bool(3.3 <= 1.0/slope <= 4.0) if slope != 0 else False,
    }


def run_validation(profiles_df):
    """Run full validation on SPARC data."""
    results_v12 = []
    results_v13 = []
    
    for gal_id in profiles_df["galaxy_id"].unique():
        gal_df = profiles_df[profiles_df["galaxy_id"] == gal_id].sort_values("radius_kpc")
        if len(gal_df) < 3:
            continue
        if gal_df["radius_kpc"].min() <= 0:
            continue
        
        try:
            results_v12.append(analyze_galaxy(gal_id, gal_df, geff_v12, "v1.2"))
            results_v13.append(analyze_galaxy(gal_id, gal_df, geff_v13, "v1.3"))
        except Exception:
            continue
    
    # Summary stats
    def summarize(results):
        ratios = np.array([r["mean_ratio"] for r in results])
        return {
            "n_galaxies": len(results),
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "std_ratio": float(np.std(ratios)),
            "universality": float(np.std(ratios) / np.mean(ratios)),
            "fraction_within_2x": float(np.sum((ratios >= 0.5) & (ratios <= 2.0)) / len(ratios)),
            "pass_rotation": bool(0.5 <= np.mean(ratios) <= 3.0),
            "pass_universality": bool(np.std(ratios) / np.mean(ratios) < 0.50),
        }
    
    stats_v12 = summarize(results_v12)
    stats_v13 = summarize(results_v13)
    btfr_v12 = compute_btfr(results_v12)
    btfr_v13 = compute_btfr(results_v13)
    
    return {
        "v12": {"stats": stats_v12, "btfr": btfr_v12},
        "v13": {"stats": stats_v13, "btfr": btfr_v13},
        "per_galaxy_v12": results_v12,
        "per_galaxy_v13": results_v13,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="v1.3 Full-SPARC Validation")
    parser.add_argument("--fetch", action="store_true", help="Fetch SPARC data")
    args = parser.parse_args()
    
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    sparc_path = DATA_DIR / "sparc_profiles.csv"
    
    if args.fetch or not sparc_path.exists():
        print("NOTE: This module requires sparc_profiles.csv with V_gas, V_disk, V_bulge")
        print("Data available from http://astroweb.cwru.edu/SPARC/")
        print("Place sparc_profiles.csv in:", DATA_DIR)
        return {"overall_status": "DATA_REQUIRED", "message": "SPARC data not found"}
    
    print("=" * 80)
    print("v1.3 FULL-SPARC VALIDATION")
    print("=" * 80)
    print(f"LOCKED: A = {A_V13}, a0 = {A0_V13}, c_eff = {C_EFF}")
    print()
    
    profiles_df = pd.read_csv(sparc_path)
    print(f"Loaded {len(profiles_df['galaxy_id'].unique())} galaxies")
    
    results = run_validation(profiles_df)
    
    v13_stats = results["v13"]["stats"]
    v13_btfr = results["v13"]["btfr"]
    
    overall_pass = (
        v13_stats["pass_rotation"] and
        v13_stats["pass_universality"] and
        v13_btfr["pass_btfr"]
    )
    
    summary = {
        "test_id": "GRAV-39",
        "timestamp": datetime.now().isoformat(),
        "constants": {"c_eff": C_EFF, "A": A_V13, "a0": A0_V13},
        "v13_stats": v13_stats,
        "v13_btfr": v13_btfr,
        "overall_status": "PASS" if overall_pass else "FAIL",
    }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(OUTPUT_DIR / "galaxy_v13_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print()
    print("Results:")
    print(f"  N galaxies: {v13_stats['n_galaxies']}")
    print(f"  Mean ratio: {v13_stats['mean_ratio']:.3f}")
    print(f"  Universality: {v13_stats['universality']:.3f}")
    print(f"  BTFR slope: {v13_btfr['slope_Mv']:.3f}")
    print(f"  OVERALL: {summary['overall_status']}")
    
    return summary


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("overall_status") == "PASS" else 1)
