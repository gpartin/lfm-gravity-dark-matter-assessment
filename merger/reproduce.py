#!/usr/bin/env python3
"""
Merger Domain Reproduction: Papers 3-4
======================================

Reproduces merger offset predictions for 12 well-studied clusters.

For each cluster:
    1. Load observational parameters (sigma, r_c, v_merger, dx_obs)
    2. Compute LFM prediction: dx_max = v_merger * r_c / sigma
    3. Compute Q = dx_obs / dx_max
    4. Check Q <= 1 (LFM bound satisfied)

NO TUNING. NO NEW CONSTANTS.
LOCKED: c_eff = 300 km/s (appears in derived scales, not offset formula)

Sources for observational data:
    - Bullet Cluster: Clowe+2006, Markevitch+2004
    - MACS J0025: Bradac+2008
    - Abell clusters: Various (see per-cluster citations)
"""

import numpy as np
import json
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict

# Load model from this directory explicitly to avoid module name conflicts
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("merger_model", SCRIPT_DIR / "model.py")
merger_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merger_model)

predict_max_offset = merger_model.predict_max_offset
compute_Q_ratio = merger_model.compute_Q_ratio
compute_derived_scales = merger_model.compute_derived_scales
C_EFF_KMS = merger_model.C_EFF_KMS


# Paths
DATA_DIR = SCRIPT_DIR / "cluster_inputs"
OUTPUT_DIR = SCRIPT_DIR.parent / "outputs" / "merger"


# ==============================================================================
# MERGING CLUSTER SAMPLE
# ==============================================================================
# Real observational data from published literature.
# v_merger = None if no published shock velocity estimate.

MERGING_CLUSTERS = [
    # === CONTROL SYSTEMS (well-studied) ===
    {
        "name": "Bullet Cluster (1E 0657-56)",
        "sigma_km_s": 1100,
        "r_c_kpc": 150,
        "v_merger_km_s": 4500,  # Shock velocity: Markevitch+2002, 2004
        "v_merger_source": "Markevitch+2004",
        "dx_obs_kpc": 150,      # Dark matter - gas offset
        "dx_obs_source": "Clowe+2006",
        "directionality": "correct"
    },
    {
        "name": "MACS J0025.4-1222",
        "sigma_km_s": 950,
        "r_c_kpc": 120,
        "v_merger_km_s": 2000,  # Bradac+2008 dynamical estimate
        "v_merger_source": "Bradac+2008",
        "dx_obs_kpc": 100,
        "dx_obs_source": "Bradac+2008",
        "directionality": "correct"
    },
    
    # === ADDITIONAL MERGERS ===
    {
        "name": "Abell 520",
        "sigma_km_s": 1000,
        "r_c_kpc": 200,
        "v_merger_km_s": 2300,  # Markevitch+2005
        "v_merger_source": "Markevitch+2005",
        "dx_obs_kpc": 150,      # Controversial "dark core"
        "dx_obs_source": "Mahdavi+2007",
        "directionality": "ambiguous"
    },
    {
        "name": "Abell 2146",
        "sigma_km_s": 900,
        "r_c_kpc": 100,
        "v_merger_km_s": 2200,  # Russell+2010 shock analysis
        "v_merger_source": "Russell+2010",
        "dx_obs_kpc": 90,
        "dx_obs_source": "Russell+2012",
        "directionality": "correct"
    },
    {
        "name": "Abell 754",
        "sigma_km_s": 850,
        "r_c_kpc": 180,
        "v_merger_km_s": 2000,  # Henry+2004 temperature jump
        "v_merger_source": "Henry+2004",
        "dx_obs_kpc": 200,
        "dx_obs_source": "Okabe+2008",
        "directionality": "correct"
    },
    {
        "name": "Abell 2744 (Pandora)",
        "sigma_km_s": 1500,
        "r_c_kpc": 250,
        "v_merger_km_s": 3000,  # Merten+2011 dynamical modeling
        "v_merger_source": "Merten+2011",
        "dx_obs_kpc": 200,
        "dx_obs_source": "Merten+2011",
        "directionality": "correct"
    },
    {
        "name": "MACS J0717.5+3745",
        "sigma_km_s": 1600,
        "r_c_kpc": 300,
        "v_merger_km_s": 3500,  # Ma+2009 shock velocity
        "v_merger_source": "Ma+2009",
        "dx_obs_kpc": 250,
        "dx_obs_source": "Limousin+2012",
        "directionality": "correct"
    },
    {
        "name": "CIZA J2242.8+5301 (Sausage)",
        "sigma_km_s": 1000,
        "r_c_kpc": 200,
        "v_merger_km_s": 2500,  # van Weeren+2010 radio relic
        "v_merger_source": "vanWeeren+2010",
        "dx_obs_kpc": 180,
        "dx_obs_source": "Jee+2015",
        "directionality": "correct"
    },
    {
        "name": "ZwCl 0008.8+5215",
        "sigma_km_s": 800,
        "r_c_kpc": 120,
        "v_merger_km_s": 2000,  # van Weeren+2011
        "v_merger_source": "vanWeeren+2011",
        "dx_obs_kpc": 100,
        "dx_obs_source": "Golovich+2017",
        "directionality": "correct"
    },
    {
        "name": "Abell 1758",
        "sigma_km_s": 1100,
        "r_c_kpc": 180,
        "v_merger_km_s": 2200,  # David+2004 X-ray morphology
        "v_merger_source": "David+2004",
        "dx_obs_kpc": 120,
        "dx_obs_source": "Monteiro-Oliveira+2017",
        "directionality": "correct"
    },
    {
        "name": "Abell 3667",
        "sigma_km_s": 950,
        "r_c_kpc": 250,
        "v_merger_km_s": 1500,  # Finoguenov+2010 shock
        "v_merger_source": "Finoguenov+2010",
        "dx_obs_kpc": 150,
        "dx_obs_source": "Joffre+2000",
        "directionality": "correct"
    },
    {
        "name": "Abell 2163",
        "sigma_km_s": 1300,
        "r_c_kpc": 220,
        "v_merger_km_s": 2500,  # Markevitch+1996 shock
        "v_merger_source": "Markevitch+1996",
        "dx_obs_kpc": 180,
        "dx_obs_source": "Radovich+2008",
        "directionality": "correct"
    },
]


def run_merger_offset_analysis() -> Dict:
    """
    Run LFM offset predictions for all clusters.
    
    Returns:
        dict with per-cluster results and summary
    """
    print("\n" + "=" * 70)
    print("MERGER OFFSET ANALYSIS")
    print("=" * 70)
    print(f"Model: dx_max = v_merger * r_c / sigma")
    print(f"Pass criterion: Q = dx_obs / dx_max <= 1")
    print(f"c_eff = {C_EFF_KMS} km/s (used in derived scales)")
    print("-" * 70)
    
    print(f"\n{'Cluster':<30} {'sigma':>6} {'r_c':>6} {'v_mrg':>6} "
          f"{'dx_obs':>7} {'dx_max':>7} {'Q':>6} {'Status':>8}")
    print(f"{'':30} {'km/s':>6} {'kpc':>6} {'km/s':>6} "
          f"{'kpc':>7} {'kpc':>7} {'':>6} {'':>8}")
    print("-" * 70)
    
    results = []
    
    for cluster in MERGING_CLUSTERS:
        name = cluster["name"]
        sigma = cluster["sigma_km_s"]
        r_c = cluster["r_c_kpc"]
        v_merger = cluster["v_merger_km_s"]
        dx_obs = cluster["dx_obs_kpc"]
        
        # Predict max offset
        dx_max = predict_max_offset(sigma, r_c, v_merger)
        
        # Compute Q
        Q = compute_Q_ratio(dx_obs, dx_max)
        
        # Derived scales
        scales = compute_derived_scales(sigma, r_c)
        
        # Status
        if Q is not None:
            status = "PASS" if Q <= 1.0 else "FAIL"
        else:
            status = "N/A"
        
        result = {
            "cluster": name,
            "sigma_km_s": sigma,
            "r_c_kpc": r_c,
            "v_merger_km_s": v_merger,
            "v_merger_source": cluster["v_merger_source"],
            "dx_obs_kpc": dx_obs,
            "dx_obs_source": cluster["dx_obs_source"],
            "dx_max_kpc": dx_max,
            "Q": Q,
            "directionality": cluster["directionality"],
            "X": scales["X"],
            "ell_c_kpc": scales["ell_c_kpc"],
            "tau_Myr": scales["tau_Myr"],
            "status": status
        }
        results.append(result)
        
        # Print
        v_str = f"{v_merger:>6.0f}" if v_merger else "   ---"
        dx_max_str = f"{dx_max:>7.0f}" if dx_max else "    ---"
        Q_str = f"{Q:>6.2f}" if Q else "   ---"
        
        print(f"{name[:30]:<30} {sigma:>6.0f} {r_c:>6.0f} {v_str} "
              f"{dx_obs:>7.0f} {dx_max_str} {Q_str} {status:>8}")
    
    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics for merger analysis."""
    valid = [r for r in results if r["Q"] is not None]
    
    if len(valid) == 0:
        return {"status": "FAIL", "reason": "No valid predictions"}
    
    Q_values = [r["Q"] for r in valid]
    n_pass = sum(1 for r in valid if r["status"] == "PASS")
    n_fail = sum(1 for r in valid if r["status"] == "FAIL")
    
    # Directionality
    correct = sum(1 for r in valid if r["directionality"] == "correct")
    wrong = sum(1 for r in valid if r["directionality"] == "wrong")
    ambiguous = sum(1 for r in valid if r["directionality"] == "ambiguous")
    
    return {
        "n_clusters_tested": len(valid),
        "n_passed_Q_bound": n_pass,
        "n_failed_Q_bound": n_fail,
        "pass_rate_percent": 100 * n_pass / len(valid),
        "Q_mean": float(np.mean(Q_values)),
        "Q_median": float(np.median(Q_values)),
        "Q_std": float(np.std(Q_values)),
        "Q_min": float(np.min(Q_values)),
        "Q_max": float(np.max(Q_values)),
        "directionality_correct": correct,
        "directionality_wrong": wrong,
        "directionality_ambiguous": ambiguous,
    }


def main():
    """Run merger domain reproduction."""
    print("=" * 70)
    print("MERGER DOMAIN REPRODUCTION")
    print("Papers 3-4: Cluster Merger Offsets from LFM")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    results = run_merger_offset_analysis()
    
    # Statistics
    stats = compute_statistics(results)
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Clusters with valid predictions: {stats['n_clusters_tested']}")
    print(f"Passed Q <= 1: {stats['n_passed_Q_bound']}")
    print(f"Failed Q > 1: {stats['n_failed_Q_bound']}")
    print(f"Pass rate: {stats['pass_rate_percent']:.1f}%")
    print(f"\nQ statistics:")
    print(f"  Mean:   {stats['Q_mean']:.2f}")
    print(f"  Median: {stats['Q_median']:.2f}")
    print(f"  Std:    {stats['Q_std']:.2f}")
    print(f"  Range:  [{stats['Q_min']:.2f}, {stats['Q_max']:.2f}]")
    print(f"\nDirectionality:")
    print(f"  Correct:   {stats['directionality_correct']}")
    print(f"  Wrong:     {stats['directionality_wrong']}")
    print(f"  Ambiguous: {stats['directionality_ambiguous']}")
    
    # Overall status
    # PASS if 100% of clusters have Q <= 1
    overall_pass = stats["n_failed_Q_bound"] == 0
    
    print(f"\nOVERALL MERGER DOMAIN: {'PASS' if overall_pass else 'FAIL'}")
    
    # Save results
    output = {
        "domain": "merger",
        "model": "dx_max = v_merger * r_c / sigma",
        "c_eff_km_s": C_EFF_KMS,
        "per_cluster": results,
        "summary": stats,
        "overall_status": "PASS" if overall_pass else "FAIL"
    }
    
    output_path = OUTPUT_DIR / "merger_reproduction_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to: {output_path}")
    
    # Also write CSV for easy inspection
    csv_path = OUTPUT_DIR / "merger_reproduction_table.csv"
    with open(csv_path, "w") as f:
        f.write("cluster,sigma_km_s,r_c_kpc,v_merger_km_s,dx_obs_kpc,dx_max_kpc,Q,status\n")
        for r in results:
            dx_max = r["dx_max_kpc"] if r["dx_max_kpc"] else ""
            Q = f"{r['Q']:.3f}" if r["Q"] else ""
            f.write(f"{r['cluster']},{r['sigma_km_s']},{r['r_c_kpc']},"
                    f"{r['v_merger_km_s'] or ''},{r['dx_obs_kpc']},"
                    f"{dx_max},{Q},{r['status']}\n")
    print(f"Table written to: {csv_path}")
    
    return output


if __name__ == "__main__":
    main()
