#!/usr/bin/env python3
"""
LFM Gravitational Series Reproducibility
==========================================

Master script to reproduce all experiments from Papers 1-5.

USAGE:
    python run_all.py                  # Run all domains
    python run_all.py --galaxy         # Run galaxy domain only
    python run_all.py --merger         # Run merger domain only
    python run_all.py --cosmology      # Run cosmology domain only
    python run_all.py --galaxy-v13     # Run v1.3 full-SPARC validation

OUTPUT:
    outputs/galaxy/galaxy_reproduction_results.json
    outputs/merger/merger_reproduction_results.json
    outputs/cosmology/cosmology_reproduction_results.json
    outputs/galaxy_v13/summary.json
    outputs/combined_report.json
    outputs/comparison_report.md

LOCKED CONSTANTS:
    c_eff = 300 km/s (appears throughout all domains)
    A = 10.0, a0 = 50.0 (km/s)^2/kpc (v1.3 coupling law)

NO SYNTHETIC DATA. NO PLACEHOLDERS. NO SHORTCUTS.
All computations run the actual LFM equations on real input data.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def run_galaxy_domain():
    """Run galaxy domain reproduction (Papers 1-2)."""
    print("\n" + "=" * 80)
    print("RUNNING GALAXY DOMAIN")
    print("=" * 80)
    
    # Import from galaxy directory explicitly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "galaxy_reproduce",
        SCRIPT_DIR / "galaxy" / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def run_merger_domain():
    """Run merger domain reproduction (Papers 3-4)."""
    print("\n" + "=" * 80)
    print("RUNNING MERGER DOMAIN")
    print("=" * 80)
    
    # Import from merger directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "merger_reproduce",
        SCRIPT_DIR / "merger" / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def run_cosmology_domain():
    """Run cosmology domain reproduction (Paper 5)."""
    print("\n" + "=" * 80)
    print("RUNNING COSMOLOGY DOMAIN")
    print("=" * 80)
    
    # Import from cosmology directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cosmology_reproduce",
        SCRIPT_DIR / "cosmology" / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def run_galaxy_v13_domain():
    """Run v1.3 full-SPARC validation (Paper 039)."""
    print("\n" + "=" * 80)
    print("RUNNING v1.3 FULL-SPARC VALIDATION")
    print("=" * 80)
    
    # Import from galaxy_v13_full_sparc directory
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "galaxy_v13_reproduce",
        SCRIPT_DIR / "galaxy_v13_full_sparc" / "reproduce.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def generate_comparison_report(results: dict) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# LFM Gravitational Series: Reproduction Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Domain | Status | Key Metric |",
        "|--------|--------|------------|",
    ]
    
    if "galaxy" in results:
        g = results["galaxy"]
        status = g.get("overall_status", "N/A")
        if "tests" in g and "test_A" in g["tests"]:
            metric = f"Test A: {g['tests']['test_A']['summary']['pass_rate_percent']:.0f}% pass"
        else:
            metric = "N/A"
        lines.append(f"| Galaxy | {status} | {metric} |")
    
    if "merger" in results:
        m = results["merger"]
        status = m.get("overall_status", "N/A")
        if "summary" in m:
            metric = f"Q_max = {m['summary']['Q_max']:.2f}"
        else:
            metric = "N/A"
        lines.append(f"| Merger | {status} | {metric} |")
    
    if "cosmology" in results:
        c = results["cosmology"]
        status = c.get("overall_status", "N/A")
        metric = "GW + Pulsar tests"
        lines.append(f"| Cosmology | {status} | {metric} |")
    
    lines.extend([
        "",
        "## Domain Details",
        "",
    ])
    
    # Galaxy
    if "galaxy" in results:
        g = results["galaxy"]
        lines.extend([
            "### Galaxy Domain (Papers 1-2)",
            "",
            "**Test A: Inner-fit -> Outer-predict**",
            "- Train on r <= 0.70 r_max, predict r > 0.70 r_max",
            "- PASS threshold: MAPE <= 15%",
        ])
        if "tests" in g and "test_A" in g["tests"]:
            s = g["tests"]["test_A"]["summary"]
            lines.extend([
                f"- Galaxies tested: {s['n_galaxies_tested']}",
                f"- Pass rate: {s['pass_rate_percent']:.1f}%",
                f"- Mean MAPE: {s['mean_MAPE_percent']:.2f}%",
            ])
        lines.append("")
    
    # Merger
    if "merger" in results:
        m = results["merger"]
        lines.extend([
            "### Merger Domain (Papers 3-4)",
            "",
            "**Offset Bound Test**",
            "- Model: dx_max = v_merger x r_c / sigma",
            "- PASS criterion: Q = dx_obs / dx_max <= 1 for all clusters",
        ])
        if "summary" in m:
            s = m["summary"]
            lines.extend([
                f"- Clusters tested: {s['n_clusters_tested']}",
                f"- All Q <= 1: {s['n_passed_Q_bound'] == s['n_clusters_tested']}",
                f"- Q range: [{s['Q_min']:.2f}, {s['Q_max']:.2f}]",
            ])
        lines.append("")
    
    # Cosmology
    if "cosmology" in results:
        c = results["cosmology"]
        lines.extend([
            "### Cosmology Domain (Paper 5)",
            "",
            "**Strong-field consistency tests**",
        ])
        if "experiments" in c:
            for exp_name, exp in c["experiments"].items():
                lines.append(f"- {exp['test']}: {exp['status']}")
        lines.append("")
    
    # Falsification criteria
    lines.extend([
        "## Falsification Criteria",
        "",
        "The LFM framework would be falsified if:",
        "",
        "1. **Galaxy domain**: Mean MAPE > 20% on holdout galaxies",
        "2. **Merger domain**: Any cluster has Q > 1.5 (unambiguous violation)",
        "3. **GW speed**: |c_g - c|/c > 10^-14 (order of magnitude above GW170817)",
        "4. **Pulsar timing**: Orbital decay deviates from GR by > 1%",
        "",
        "## Reproducibility Notes",
        "",
        "- All computations use c_eff = 300 km/s (LOCKED, not tunable)",
        "- No per-galaxy or per-cluster parameters",
        "- Galaxy chi-reconstruction is parameter-free inversion",
        "- Merger offset formula has zero free parameters",
        "",
    ])
    
    return "\n".join(lines)


def main():
    """Run all domain reproductions."""
    parser = argparse.ArgumentParser(
        description="LFM Gravitational Series Reproducibility"
    )
    parser.add_argument("--galaxy", action="store_true", help="Run galaxy domain only")
    parser.add_argument("--merger", action="store_true", help="Run merger domain only")
    parser.add_argument("--cosmology", action="store_true", help="Run cosmology domain only")
    parser.add_argument("--galaxy-v13", action="store_true", help="Run v1.3 full-SPARC validation")
    args = parser.parse_args()
    
    # If no specific domain, run all
    run_all = not (args.galaxy or args.merger or args.cosmology or getattr(args, 'galaxy_v13', False))
    
    print("=" * 80)
    print("LFM GRAVITATIONAL SERIES REPRODUCIBILITY")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"LOCKED CONSTANT: c_eff = 300 km/s")
    print("")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Galaxy
    if run_all or args.galaxy:
        try:
            results["galaxy"] = run_galaxy_domain()
        except Exception as e:
            print(f"ERROR in galaxy domain: {e}")
            results["galaxy"] = {"overall_status": "ERROR", "error": str(e)}
    
    # Merger
    if run_all or args.merger:
        try:
            results["merger"] = run_merger_domain()
        except Exception as e:
            print(f"ERROR in merger domain: {e}")
            results["merger"] = {"overall_status": "ERROR", "error": str(e)}
    
    # Cosmology
    if run_all or args.cosmology:
        try:
            results["cosmology"] = run_cosmology_domain()
        except Exception as e:
            print(f"ERROR in cosmology domain: {e}")
            results["cosmology"] = {"overall_status": "ERROR", "error": str(e)}
    
    # Galaxy v1.3 Full-SPARC
    if run_all or getattr(args, 'galaxy_v13', False):
        try:
            results["galaxy_v13"] = run_galaxy_v13_domain()
        except Exception as e:
            print(f"ERROR in galaxy v1.3 domain: {e}")
            results["galaxy_v13"] = {"overall_status": "ERROR", "error": str(e)}
    
    # Combined report
    combined = {
        "timestamp": datetime.now().isoformat(),
        "c_eff_km_s": 300.0,
        "domains": results,
        "overall_status": "PASS" if all(
            r.get("overall_status") == "PASS" for r in results.values()
        ) else "FAIL"
    }
    
    # Write JSON
    json_path = OUTPUT_DIR / "combined_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    
    # Write markdown
    md_report = generate_comparison_report(results)
    md_path = OUTPUT_DIR / "comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for domain, result in results.items():
        status = result.get("overall_status", "UNKNOWN")
        print(f"{domain.upper():12} {status}")
    
    print(f"\nOVERALL: {combined['overall_status']}")
    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print(f"  - combined_report.json")
    print(f"  - comparison_report.md")
    
    return 0 if combined["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
