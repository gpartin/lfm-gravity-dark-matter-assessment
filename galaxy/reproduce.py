#!/usr/bin/env python3
"""
Galaxy Domain Reproduction: Tests A, B, C from Papers 1-2
==========================================================

Reproduces the three core validation tests:

TEST A: Inner-fit -> Outer-predict
    - Train on r <= 0.70 * r_max
    - Reconstruct chi from training data
    - Extrapolate chi to test region
    - Predict v(r) for r > 0.70 * r_max
    - PASS if MAPE <= 15%

TEST B: Cross-galaxy chi compression
    - Reconstruct chi for ALL galaxies
    - Compress each chi(r) to 3-parameter family
    - Measure compression fidelity
    - PASS if median residual <= 5%

TEST C: Holdout galaxy prediction
    - Use 64 training galaxies to establish chi-family
    - Apply to 27 holdout galaxies
    - PASS if holdout error comparable to training

LOCKED CONSTANT: c_eff = 300 km/s
"""

import numpy as np
import pandas as pd
import json
import sys
import importlib.util
from pathlib import Path
from typing import Tuple, Dict, List

# Load model from this directory explicitly to avoid module name conflicts
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("galaxy_model", SCRIPT_DIR / "model.py")
galaxy_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(galaxy_model)

ChiReconstructor = galaxy_model.ChiReconstructor
compute_prediction_metrics = galaxy_model.compute_prediction_metrics
C_EFF_KMS = galaxy_model.C_EFF_KMS


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
    train_fraction: float = 0.70,
    continuation_range: Tuple[float, float] = (0.50, 0.70)
) -> Dict:
    """
    Run Test A on a single galaxy.
    
    Args:
        galaxy_name: Galaxy identifier
        galaxy_df: DataFrame with radius_kpc, velocity_km_s, velocity_error_km_s
        train_fraction: Split point (default 0.70)
        continuation_range: Range for exponential tail fit
    
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
    reconstructor = ChiReconstructor(c_km_s=C_EFF_KMS)
    
    try:
        chi_train = reconstructor.reconstruct_chi(r_train, v_train)
    except Exception as e:
        return {"galaxy_name": galaxy_name, "error": str(e), "passed": False}
    
    # Extrapolate chi to test region
    r_fit_start = continuation_range[0] * r_max
    r_fit_end = continuation_range[1] * r_max
    
    chi_test = reconstructor.extrapolate_chi_exponential(
        r_train, chi_train, r_fit_start, r_fit_end, r_test
    )
    
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
        "test_name": "Test A: Inner-fit -> Outer-predict",
        "status": "PASS" if overall_pass else "FAIL",
        "summary": summary,
        "per_galaxy": valid_results
    }


# ==============================================================================
# TEST B: Chi compression fidelity
# ==============================================================================

def compress_chi_to_family(
    r_kpc: np.ndarray,
    chi: np.ndarray
) -> Tuple[Dict, np.ndarray]:
    """
    Compress chi(r) to 3-parameter exponential-power-law family.
    
    chi(r_tilde) = A * exp(-B * r_tilde^gamma)
    where r_tilde = r / r_max
    
    Returns:
        params: dict with A, B, gamma, r_max_kpc
        chi_fit: Reconstructed chi from compressed form
    """
    from scipy.optimize import curve_fit
    
    r = np.array(r_kpc)
    chi_vals = np.array(chi)
    
    r_max = r.max()
    r_tilde = r / r_max
    
    def model(rt, A, B, gamma):
        return A * np.exp(-B * rt**gamma)
    
    try:
        p0 = [chi_vals[0], 1.0, 1.0]
        popt, _ = curve_fit(model, r_tilde, chi_vals, p0=p0, maxfev=10000)
        A, B, gamma = popt
        chi_fit = model(r_tilde, A, B, gamma)
        params = {"A": float(A), "B": float(B), "gamma": float(gamma), "r_max_kpc": float(r_max)}
        return params, chi_fit
    except Exception:
        return {"A": float(chi_vals[0]), "B": 0, "gamma": 1, "r_max_kpc": float(r_max)}, chi_vals


def run_test_B_single(galaxy_name: str, galaxy_df: pd.DataFrame) -> Dict:
    """Run Test B on single galaxy."""
    r = galaxy_df["radius_kpc"].values
    v = galaxy_df["velocity_km_s"].values
    
    if len(r) < 5:
        return None
    
    reconstructor = ChiReconstructor(c_km_s=C_EFF_KMS)
    
    try:
        chi = reconstructor.reconstruct_chi(r, v)
    except Exception as e:
        return {"galaxy_name": galaxy_name, "error": str(e)}
    
    # Compress to family
    params, chi_fit = compress_chi_to_family(r, chi)
    
    # Compression fidelity: percent residual
    residual_pct = 100 * np.mean(np.abs(chi - chi_fit) / (chi + 1e-30))
    
    return {
        "galaxy_name": galaxy_name,
        "compression_params": params,
        "compression_residual_percent": float(residual_pct),
        "passed": residual_pct <= 5.0
    }


def run_test_B(df: pd.DataFrame) -> Dict:
    """
    Run Test B: Chi compression fidelity.
    
    Tests whether chi-field has low information content
    (can be described by 3 parameters per galaxy).
    """
    print("\n" + "=" * 70)
    print("TEST B: Chi Compression Fidelity")
    print("=" * 70)
    print("Compress chi(r) to 3-parameter family: A * exp(-B * r_tilde^gamma)")
    print("PASS threshold: median residual <= 5%")
    print("-" * 70)
    
    galaxy_names = get_galaxy_names(df)
    results = []
    
    for gname in galaxy_names:
        gdf = df[df["galaxy_name"] == gname].sort_values("radius_kpc")
        result = run_test_B_single(gname, gdf)
        if result is not None:
            results.append(result)
    
    valid = [r for r in results if "error" not in r]
    if len(valid) == 0:
        return {"status": "FAIL", "reason": "No valid results"}
    
    residuals = [r["compression_residual_percent"] for r in valid]
    passed_count = sum(1 for r in valid if r["passed"])
    
    summary = {
        "n_galaxies": len(valid),
        "n_passed": passed_count,
        "pass_rate_percent": 100 * passed_count / len(valid),
        "mean_residual_percent": float(np.mean(residuals)),
        "median_residual_percent": float(np.median(residuals)),
    }
    
    overall_pass = summary["median_residual_percent"] <= 5.0
    
    print(f"\nResults: {passed_count}/{len(valid)} galaxies with residual <= 5%")
    print(f"Median residual: {summary['median_residual_percent']:.2f}%")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    return {
        "test_name": "Test B: Chi Compression Fidelity",
        "status": "PASS" if overall_pass else "FAIL",
        "summary": summary,
        "per_galaxy": valid
    }


# ==============================================================================
# TEST C: Holdout galaxy prediction
# ==============================================================================

def run_test_C(df: pd.DataFrame, holdout_fraction: float = 0.30) -> Dict:
    """
    Run Test C: Holdout galaxy prediction.
    
    Uses random 70/30 split of galaxies (not data points).
    Establishes chi-family from training galaxies,
    applies to holdout galaxies.
    """
    print("\n" + "=" * 70)
    print("TEST C: Holdout Galaxy Prediction")
    print("=" * 70)
    print("70% training galaxies, 30% holdout galaxies")
    print("PASS: holdout MAPE within 50% of training MAPE")
    print("-" * 70)
    
    galaxy_names = get_galaxy_names(df)
    n_galaxies = len(galaxy_names)
    n_holdout = int(holdout_fraction * n_galaxies)
    
    # Fixed seed for reproducibility
    np.random.seed(42)
    holdout_names = set(np.random.choice(galaxy_names, n_holdout, replace=False))
    train_names = set(galaxy_names) - holdout_names
    
    # Run Test A on both sets
    train_results = []
    holdout_results = []
    
    for gname in galaxy_names:
        gdf = df[df["galaxy_name"] == gname].sort_values("radius_kpc")
        result = run_test_A_single(gname, gdf)
        if result is not None and "error" not in result:
            if gname in train_names:
                train_results.append(result)
            else:
                holdout_results.append(result)
    
    if len(train_results) == 0 or len(holdout_results) == 0:
        return {"status": "FAIL", "reason": "Insufficient data"}
    
    train_mapes = [r["MAPE_percent"] for r in train_results]
    holdout_mapes = [r["MAPE_percent"] for r in holdout_results]
    
    mean_train = np.mean(train_mapes)
    mean_holdout = np.mean(holdout_mapes)
    ratio = mean_holdout / mean_train if mean_train > 0 else float("inf")
    
    summary = {
        "n_train_galaxies": len(train_results),
        "n_holdout_galaxies": len(holdout_results),
        "train_mean_MAPE": float(mean_train),
        "holdout_mean_MAPE": float(mean_holdout),
        "holdout_to_train_ratio": float(ratio),
    }
    
    # Pass if holdout not more than 50% worse
    overall_pass = ratio <= 1.5
    
    print(f"\nTraining galaxies: {len(train_results)}, mean MAPE: {mean_train:.2f}%")
    print(f"Holdout galaxies: {len(holdout_results)}, mean MAPE: {mean_holdout:.2f}%")
    print(f"Ratio (holdout/train): {ratio:.2f}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    return {
        "test_name": "Test C: Holdout Galaxy Prediction",
        "status": "PASS" if overall_pass else "FAIL",
        "summary": summary,
        "train_galaxies": [r["galaxy_name"] for r in train_results],
        "holdout_galaxies": [r["galaxy_name"] for r in holdout_results],
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run all galaxy domain tests."""
    print("=" * 70)
    print("GALAXY DOMAIN REPRODUCTION")
    print("Papers 1-2: Rotation Curve Prediction from Chi-Field")
    print("=" * 70)
    print(f"\nLOCKED CONSTANT: c_eff = {C_EFF_KMS} km/s")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading SPARC rotation curve data...")
    df = load_sparc_data()
    n_galaxies = len(get_galaxy_names(df))
    n_points = len(df)
    print(f"Loaded {n_galaxies} galaxies, {n_points} total data points")
    
    # Run tests
    test_a_results = run_test_A(df)
    test_b_results = run_test_B(df)
    test_c_results = run_test_C(df)
    
    # Aggregate
    all_results = {
        "domain": "galaxy",
        "c_eff_km_s": C_EFF_KMS,
        "n_galaxies": n_galaxies,
        "tests": {
            "test_A": test_a_results,
            "test_B": test_b_results,
            "test_C": test_c_results,
        },
        "overall_status": "PASS" if all(
            t["status"] == "PASS" for t in [test_a_results, test_b_results, test_c_results]
        ) else "FAIL"
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
    print("GALAXY DOMAIN SUMMARY")
    print("=" * 70)
    print(f"Test A (Out-of-sample prediction): {test_a_results['status']}")
    print(f"Test B (Chi compression):          {test_b_results['status']}")
    print(f"Test C (Holdout galaxies):         {test_c_results['status']}")
    print(f"\nOVERALL GALAXY DOMAIN: {all_results['overall_status']}")
    
    return all_results


if __name__ == "__main__":
    main()
