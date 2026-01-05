#!/usr/bin/env python3
"""
Galaxy Domain Reproduction: Test A (Claim-Carrying Only)
=========================================================

This module reproduces ONLY the claim-carrying test from Paper 034.

Tests B and C have been EXCLUDED because they:
- Have no published paper anchor
- Were exploratory/diagnostic experiments
- Are not referenced in any figure, table, or claim

CLAIM-CARRYING TEST (Paper 034, Table 3):

TEST A: Inner-fit -> Outer-predict
    - Train on r <= 0.70 * r_max
    - Reconstruct chi from training data using c = 299,792 km/s
    - Extrapolate chi to test region
    - Predict v(r) for r > 0.70 * r_max using c_eff = 300 km/s
    - Expected result: chi^2_nu = 1.024
    - PASS if MAPE <= 15%

CRITICAL METHODOLOGY (Paper 034, Section II.D):
    - Chi RECONSTRUCTION uses c = 299,792 km/s (physical speed of light)
    - Chi->v PREDICTION uses c_eff = 300 km/s (emergent scale parameter)

Quote: "chi is reconstructed using the physical speed of light c...
The parameter c_eff enters solely as an emergent, scale-dependent
coefficient in the chi -> v conversion."
"""

import numpy as np
import pandas as pd
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List

# Load model from this directory explicitly to avoid module name conflicts
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("galaxy_model", SCRIPT_DIR / "model.py")
galaxy_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(galaxy_model)

ChiReconstructor = galaxy_model.ChiReconstructor
compute_prediction_metrics = galaxy_model.compute_prediction_metrics
C_EFF_KMS = galaxy_model.C_EFF_KMS
C_LIGHT_KMS = galaxy_model.C_LIGHT_KMS


# Paths
DATA_DIR = SCRIPT_DIR / "sparc_data"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "galaxy"


def load_sparc_data() -> pd.DataFrame:
    """Load processed SPARC rotation curve data."""
    csv_path = DATA_DIR / "galaxies.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SPARC data not found at {csv_path}\n"
            "Please ensure sparc_data/galaxies.csv exists."
        )
    return pd.read_csv(csv_path)


def get_galaxy_names(df: pd.DataFrame) -> List[str]:
    """Get unique galaxy names."""
    return df["galaxy_name"].unique().tolist()


# ==============================================================================
# TEST A: Inner-fit -> Outer-predict
# ==============================================================================

def run_test_A_single(
    galaxy_name: str,
    galaxy_df: pd.DataFrame,
    train_fraction: float = 0.70
) -> Dict:
    """
    Run Test A on a single galaxy using A6 closure (Paper 034).
    
    Args:
        galaxy_name: Galaxy identifier
        galaxy_df: DataFrame with radius_kpc, velocity_km_s, velocity_error_km_s
        train_fraction: Split point (default 0.70)
    
    Returns:
        dict with test results or None if insufficient data
    """
    r = galaxy_df["radius_kpc"].values
    v = galaxy_df["velocity_km_s"].values
    v_err = galaxy_df["velocity_error_km_s"].values
    
    r_max = r.max()
    r_split = train_fraction * r_max
    
    # Split data
    train_mask = r <= r_split
    test_mask = r > r_split
    
    r_train, v_train = r[train_mask], v[train_mask]
    r_test, v_test = r[test_mask], v[test_mask]
    v_test_err = v_err[test_mask]
    
    # Need sufficient data
    if len(r_train) < 5 or len(r_test) < 3:
        return None
    
    # Reconstruct chi from training data
    # CRITICAL (Paper 034 Section II.D): Use physical c for reconstruction, c_eff for prediction
    reconstructor = ChiReconstructor(c_recon_km_s=C_LIGHT_KMS, c_pred_km_s=C_EFF_KMS)
    
    try:
        chi_train = reconstructor.reconstruct_chi(r_train, v_train)
    except Exception as e:
        return {"galaxy_name": galaxy_name, "error": str(e), "passed": False}
    
    # Extrapolate chi to test region using A6 closure (Paper 034 Section III.D)
    # A6: chi(r) = chi(r_0) * exp(-(r - r_0) / L) with L = 2.0 * r_max (FULL galaxy)
    chi_test = reconstructor.extrapolate_chi_A6(r_train, chi_train, r_test, r_max_galaxy=r_max, k=2.0)
    
    # Predict velocities
    v_pred = reconstructor.chi_to_velocity(r_test, chi_test)
    
    # Compute metrics
    metrics = compute_prediction_metrics(v_test, v_pred, v_test_err)
    
    # Pass threshold: MAPE <= 15%
    passed = metrics["MAPE_percent"] <= 15.0
    
    return {
        "galaxy_name": galaxy_name,
        "r_max_kpc": float(r_max),
        "r_split_kpc": float(r_split),
        "n_train": len(r_train),
        "n_test": len(r_test),
        "MAPE_percent": metrics["MAPE_percent"],
        "RMSE_km_s": metrics["RMSE_km_s"],
        "max_error_km_s": metrics["max_abs_error_km_s"],
        "passed": passed
    }


def run_test_A(df: pd.DataFrame) -> Dict:
    """
    Run Test A on all galaxies.
    
    Returns:
        dict with per-galaxy results and summary
    """
    print("\n" + "=" * 70)
    print("TEST A: Inner-fit -> Outer-predict")
    print("=" * 70)
    print(f"Using c_eff = {C_EFF_KMS} km/s (LOCKED)")
    print("Train on r <= 0.70 * r_max, predict r > 0.70 * r_max")
    print("PASS threshold: MAPE <= 15%")
    print("-" * 70)
    
    galaxy_names = get_galaxy_names(df)
    results = []
    
    for gname in galaxy_names:
        gdf = df[df["galaxy_name"] == gname].sort_values("radius_kpc")
        result = run_test_A_single(gname, gdf)
        if result is not None:
            results.append(result)
    
    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    if len(valid_results) == 0:
        return {"status": "FAIL", "reason": "No valid results"}
    
    mape_values = [r["MAPE_percent"] for r in valid_results]
    passed_count = sum(1 for r in valid_results if r["passed"])
    
    summary = {
        "n_galaxies_tested": len(valid_results),
        "n_passed": passed_count,
        "pass_rate_percent": 100 * passed_count / len(valid_results),
        "mean_MAPE_percent": float(np.mean(mape_values)),
        "median_MAPE_percent": float(np.median(mape_values)),
        "std_MAPE_percent": float(np.std(mape_values)),
        "min_MAPE_percent": float(np.min(mape_values)),
        "max_MAPE_percent": float(np.max(mape_values)),
    }
    
    # Overall pass: >= 80% of galaxies pass
    overall_pass = summary["pass_rate_percent"] >= 80.0
    
    print(f"\nResults: {passed_count}/{len(valid_results)} galaxies passed ({summary['pass_rate_percent']:.1f}%)")
    print(f"Mean MAPE: {summary['mean_MAPE_percent']:.2f}%")
    print(f"Median MAPE: {summary['median_MAPE_percent']:.2f}%")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    return {
        "test_name": "Test A: Inner-fit -> Outer-predict (Paper 034, Table 3)",
        "status": "PASS" if overall_pass else "FAIL",
        "summary": summary,
        "per_galaxy": valid_results
    }


# ==============================================================================
# EXCLUDED TESTS (No paper anchor - exploratory/diagnostic only)
# ==============================================================================
# TEST B: Chi compression fidelity - EXCLUDED (exploratory)
# TEST C: Holdout galaxy prediction - EXCLUDED (diagnostic)
# 
# These tests were part of development but are NOT referenced in any
# published paper figure, table, or claim. They have been removed from
# this public reproduction package.
# ==============================================================================


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run claim-carrying galaxy domain test (Test A only)."""
    print("=" * 70)
    print("GALAXY DOMAIN REPRODUCTION (CLAIM-CARRYING TESTS ONLY)")
    print("Paper 034: C_eff Closure - Table 3 Validation")
    print("=" * 70)
    print(f"\nMethodology (Paper 034, Section II.D + III.D):")
    print(f"  - Chi reconstruction: c = {C_LIGHT_KMS} km/s (physical)")
    print(f"  - Chi->v prediction:  c_eff = {C_EFF_KMS} km/s (emergent)")
    print(f"  - Closure: A6 exponential tail with k=2.0, L = 2 * r_max")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading SPARC rotation curve data...")
    df = load_sparc_data()
    n_galaxies = len(get_galaxy_names(df))
    n_points = len(df)
    print(f"Loaded {n_galaxies} galaxies, {n_points} total data points")
    
    # Run ONLY the claim-carrying test
    test_a_results = run_test_A(df)
    
    # Aggregate
    all_results = {
        "domain": "galaxy",
        "paper_reference": "Paper 034, Table 3",
        "c_recon_km_s": C_LIGHT_KMS,
        "c_pred_km_s": C_EFF_KMS,
        "n_galaxies": n_galaxies,
        "claim_carrying_tests": {
            "test_A": test_a_results,
        },
        "excluded_tests": {
            "test_B": "Chi compression - EXCLUDED (exploratory, no paper anchor)",
            "test_C": "Holdout prediction - EXCLUDED (diagnostic, no paper anchor)",
        },
        "overall_status": test_a_results["status"]
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    # Write results
    output_path = OUTPUT_DIR / "galaxy_reproduction_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("GALAXY DOMAIN SUMMARY (Paper 034 Claims)")
    print("=" * 70)
    print(f"Test A (Out-of-sample prediction, Table 3): {test_a_results['status']}")
    print(f"\nExcluded (no paper anchor):")
    print(f"  Test B: Chi compression (exploratory)")
    print(f"  Test C: Holdout prediction (diagnostic)")
    print(f"\nOVERALL GALAXY DOMAIN: {all_results['overall_status']}")
    
    return all_results


if __name__ == "__main__":
    main()
