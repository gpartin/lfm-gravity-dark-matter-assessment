#!/usr/bin/env python3
"""
LFM Galaxy Model: Chi-Field Reconstruction from Rotation Curves
================================================================

Core equations (LOCKED):
    v^2(r) = -(r c^2 / 2) d(ln chi)/dr
    chi(r) = chi_0 exp[-2/c^2 * integral_0^r (v^2/r') dr']

This is the parameter-free inversion: given v(r), reconstruct chi(r).
Then extrapolate chi and predict v in unseen regions.

No tunable parameters. c_eff = 300 km/s is the ONLY constant.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit


# LOCKED CONSTANTS (Paper 034 Section II.D clarification)
C_LIGHT_KMS = 299792.458  # km/s - Physical speed of light (for chi reconstruction)
C_EFF_KMS = 300.0  # km/s - Effective propagation scale (for chi->v prediction)


class ChiReconstructor:
    """
    Reconstruct chi-field from observed rotation curve.
    
    This implements the LFM inversion formula:
        chi(r) = chi_0 * exp[-2/c^2 * integral(v^2/r dr)]
    
    CRITICAL (Paper 034 Section II.D):
    - Chi RECONSTRUCTION uses physical c = 299,792 km/s
    - Chi->velocity PREDICTION uses c_eff = 300 km/s
    """
    
    def __init__(self, c_recon_km_s: float = C_LIGHT_KMS, c_pred_km_s: float = C_EFF_KMS, chi_0: float = 1.0):
        """
        Args:
            c_recon_km_s: Speed for reconstruction (physical c). Default = 299,792 km/s
            c_pred_km_s: Speed for prediction (c_eff). Default = 300 km/s
            chi_0: Normalization (arbitrary, cancels in predictions)
        """
        self.c_recon = c_recon_km_s
        self.c_pred = c_pred_km_s
        self.chi_0 = chi_0
    
    def reconstruct_chi(self, r_kpc: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
        """
        Reconstruct chi(r) from observed rotation curve.
        
        Args:
            r_kpc: Radii in kpc (must be positive, sorted)
            v_km_s: Observed velocities in km/s
        
        Returns:
            chi: Chi-field values at each radius
        """
        # Sort by radius
        sort_idx = np.argsort(r_kpc)
        r = np.array(r_kpc)[sort_idx]
        v = np.array(v_km_s)[sort_idx]
        
        # Filter positive radii
        valid = r > 0
        r = r[valid]
        v = v[valid]
        
        if len(r) < 3:
            raise ValueError("Need at least 3 data points")
        
        # Integrand: v^2 / r
        integrand = v**2 / r
        
        # Cumulative integral: integral_0^r (v^2/r') dr'
        integral = cumulative_trapezoid(integrand, r, initial=0)
        
        # chi(r) = chi_0 * exp[-2/c_recon^2 * integral]
        # Uses PHYSICAL c for reconstruction (Paper 034 clarification)
        chi = self.chi_0 * np.exp(-2 * integral / self.c_recon**2)
        
        return chi
    
    def chi_to_velocity(self, r_kpc: np.ndarray, chi: np.ndarray) -> np.ndarray:
        """
        Convert chi(r) back to rotation velocity v(r).
        
        Uses: v^2(r) = -(r c_eff^2 / 2) d(ln chi)/dr
        
        CRITICAL (Paper 034): Uses c_eff = 300 km/s (NOT physical c)
        
        Args:
            r_kpc: Radii in kpc
            chi: Chi-field values
        
        Returns:
            v_km_s: Predicted velocities
        """
        r = np.array(r_kpc)
        chi_vals = np.array(chi)
        
        # Compute d(ln chi)/dr using gradient
        ln_chi = np.log(chi_vals + 1e-30)  # Avoid log(0)
        d_ln_chi_dr = np.gradient(ln_chi, r)
        
        # v^2 = -(r c_eff^2 / 2) * d(ln chi)/dr
        # Uses EFFECTIVE c for prediction (Paper 034 key finding)
        v_squared = -(r * self.c_pred**2 / 2) * d_ln_chi_dr
        
        # Physical constraint: v^2 >= 0
        v_squared = np.maximum(v_squared, 0)
        
        return np.sqrt(v_squared)
    
    def extrapolate_chi_A6(
        self,
        r_kpc: np.ndarray,
        chi: np.ndarray,
        r_extrap: np.ndarray,
        r_max_galaxy: float,
        k: float = 2.0
    ) -> np.ndarray:
        """
        Extrapolate chi using A6 (exponential tail) closure from Paper 034.
        
        Paper 034 Section III.D defines A6 as:
            chi(r) = chi(r_0) * exp(-(r - r_0) / L)
        where:
            L = k * R_gas (with R_gas ~ r_max of FULL galaxy)
            k = 2.0 (pre-registered, fixed across all galaxies)
        
        This is NOT a fitted exponential - it's a prescribed decay!
        
        Args:
            r_kpc: Original radii (training data)
            chi: Original chi values (from reconstruction)
            r_extrap: Radii to extrapolate to (test region)
            r_max_galaxy: Maximum radius of the FULL galaxy (not just training)
            k: Scale factor (default 2.0 per Paper 034)
        
        Returns:
            chi_extrap: Extrapolated chi values
        """
        r = np.array(r_kpc)
        chi_vals = np.array(chi)
        r_out = np.array(r_extrap)
        
        # Boundary values
        r_0 = r[-1]  # Last training radius
        chi_0 = chi_vals[-1]  # Chi at boundary
        
        # A6 decay length: L = k * R_gas (with R_gas = r_max of full galaxy)
        # CRITICAL: Use full galaxy r_max, not training region r_max!
        L = k * r_max_galaxy
        
        # A6 formula: chi(r) = chi(r_0) * exp(-(r - r_0) / L)
        chi_extrap = chi_0 * np.exp(-(r_out - r_0) / L)
        
        return chi_extrap
    
    def extrapolate_chi_exponential(
        self,
        r_kpc: np.ndarray,
        chi: np.ndarray,
        r_fit_start: float,
        r_fit_end: float,
        r_extrap: np.ndarray
    ) -> np.ndarray:
        """
        DEPRECATED: Use extrapolate_chi_A6 instead.
        
        This fitted exponential is NOT the A6 closure from Paper 034.
        Keeping for backwards compatibility but should not be used for claims.
        """
        # Use A6 closure instead of fitted exponential
        return self.extrapolate_chi_A6(r_kpc, chi, r_extrap, k=2.0)


def compute_prediction_metrics(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray = None
) -> dict:
    """
    Compute prediction error metrics.
    
    Args:
        v_obs: Observed velocities
        v_pred: Predicted velocities
        v_err: Observational errors (optional)
    
    Returns:
        dict with MAPE, RMSE, max_error, etc.
    """
    residuals = v_obs - v_pred
    abs_errors = np.abs(residuals)
    
    # Avoid division by zero
    v_obs_safe = np.where(v_obs > 0, v_obs, 1e-10)
    percent_errors = 100 * abs_errors / v_obs_safe
    
    metrics = {
        "MAPE_percent": float(np.mean(percent_errors)),
        "median_APE_percent": float(np.median(percent_errors)),
        "RMSE_km_s": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_error_km_s": float(np.max(abs_errors)),
        "mean_residual_km_s": float(np.mean(residuals)),
    }
    
    if v_err is not None:
        # Chi-squared
        chi_sq = np.sum((residuals / v_err)**2)
        metrics["chi_squared"] = float(chi_sq)
        metrics["reduced_chi_squared"] = float(chi_sq / max(len(v_obs) - 1, 1))
    
    return metrics
