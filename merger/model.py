#!/usr/bin/env python3
"""
LFM Merger Offset Model: Papers 3-4
====================================

Core equations (LOCKED):
    Kernel: W(r; ell_c) = exp(-r/ell_c) / (8 * pi * ell_c^3)
    Coherence length: ell_c = r_c / X, where X = sigma / c_eff
    Timescale: tau = r_c / sigma
    Max offset: dx_max = v_merger * r_c / sigma

NO NEW CONSTANTS. NO PER-CLUSTER TUNING.
c_eff = 300 km/s is the ONLY LFM parameter.
"""

# LOCKED CONSTANT
C_EFF_KMS = 300.0  # km/s - LFM characteristic velocity scale


def predict_max_offset(
    sigma_km_s: float,
    r_c_kpc: float,
    v_merger_km_s: float
) -> float:
    """
    Predict maximum dark matter-gas offset from LFM nonlocal dynamics.
    
    dx_max = v_merger * tau = v_merger * r_c / sigma
    
    Args:
        sigma_km_s: Cluster velocity dispersion (km/s)
        r_c_kpc: Core radius (kpc)
        v_merger_km_s: Merger velocity (km/s)
    
    Returns:
        dx_max_kpc: Maximum predicted offset (kpc)
    """
    if v_merger_km_s is None or v_merger_km_s <= 0:
        return None
    
    # tau = r_c / sigma (timescale)
    # dx_max = v_merger * tau
    dx_max_kpc = v_merger_km_s * r_c_kpc / sigma_km_s
    
    return dx_max_kpc


def compute_Q_ratio(dx_obs_kpc: float, dx_max_kpc: float) -> float:
    """
    Compute Q = dx_obs / dx_max.
    
    If Q <= 1: LFM permits the observation (PASS)
    If Q > 1: Observation exceeds LFM bound (FAIL)
    
    Args:
        dx_obs_kpc: Observed offset (kpc)
        dx_max_kpc: LFM predicted maximum (kpc)
    
    Returns:
        Q ratio
    """
    if dx_max_kpc is None or dx_max_kpc <= 0:
        return None
    return dx_obs_kpc / dx_max_kpc


def compute_derived_scales(sigma_km_s: float, r_c_kpc: float) -> dict:
    """
    Compute LFM-derived scales for a cluster.
    
    Args:
        sigma_km_s: Velocity dispersion
        r_c_kpc: Core radius
    
    Returns:
        dict with X, ell_c_kpc, tau_Myr
    """
    X = sigma_km_s / C_EFF_KMS
    ell_c_kpc = r_c_kpc / X
    
    # tau in Myr: tau = r_c / (sigma * conversion)
    # 1 km/s = 0.001022 kpc/Myr
    tau_Myr = r_c_kpc / (sigma_km_s * 0.001022)
    
    return {
        "X": X,
        "ell_c_kpc": ell_c_kpc,
        "tau_Myr": tau_Myr
    }
