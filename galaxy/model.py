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


# LOCKED CONSTANT
C_EFF_KMS = 300.0  # km/s - LFM characteristic velocity scale


class ChiReconstructor:
    """
    Reconstruct chi-field from observed rotation curve.
    
    This implements the LFM inversion formula:
        chi(r) = chi_0 * exp[-2/c^2 * integral(v^2/r dr)]
    """
    
    def __init__(self, c_km_s: float = C_EFF_KMS, chi_0: float = 1.0):
        """
        Args:
            c_km_s: Speed scale (km/s). Default = 300 km/s (LFM locked value)
            chi_0: Normalization (arbitrary, cancels in predictions)
        """
        self.c = c_km_s
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
        
        # chi(r) = chi_0 * exp[-2/c^2 * integral]
        chi = self.chi_0 * np.exp(-2 * integral / self.c**2)
        
        return chi
    
    def chi_to_velocity(self, r_kpc: np.ndarray, chi: np.ndarray) -> np.ndarray:
        """
        Convert chi(r) back to rotation velocity v(r).
        
        Uses: v^2(r) = -(r c^2 / 2) d(ln chi)/dr
        
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
        
        # v^2 = -(r c^2 / 2) * d(ln chi)/dr
        v_squared = -(r * self.c**2 / 2) * d_ln_chi_dr
        
        # Physical constraint: v^2 >= 0
        v_squared = np.maximum(v_squared, 0)
        
        return np.sqrt(v_squared)
    
    def extrapolate_chi_exponential(
        self,
        r_kpc: np.ndarray,
        chi: np.ndarray,
        r_fit_start: float,
        r_fit_end: float,
        r_extrap: np.ndarray
    ) -> np.ndarray:
        """
        Extrapolate chi beyond observed range using exponential tail.
        
        Fits chi ~ A * exp(-B * r) over [r_fit_start, r_fit_end],
        then extends to r_extrap.
        
        Args:
            r_kpc: Original radii
            chi: Original chi values
            r_fit_start: Start of fitting region
            r_fit_end: End of fitting region
            r_extrap: Radii to extrapolate to
        
        Returns:
            chi_extrap: Extrapolated chi values
        """
        r = np.array(r_kpc)
        chi_vals = np.array(chi)
        
        # Select fitting region
        fit_mask = (r >= r_fit_start) & (r <= r_fit_end)
        r_fit = r[fit_mask]
        chi_fit = chi_vals[fit_mask]
        
        if len(r_fit) < 3:
            # Fallback: constant
            return np.full_like(r_extrap, chi_vals[-1], dtype=float)
        
        # Fit exponential
        def exp_model(r, A, B):
            return A * np.exp(-B * r)
        
        try:
            p0 = [chi_fit[0], 0.1]
            popt, _ = curve_fit(exp_model, r_fit, chi_fit, p0=p0, maxfev=5000)
            A, B = popt
            
            chi_extrap = exp_model(np.array(r_extrap), A, B)
            
            # Floor at 1% of boundary value
            chi_extrap = np.maximum(chi_extrap, chi_vals[-1] * 0.01)
            
            return chi_extrap
        except Exception:
            # Fallback: constant
            return np.full_like(r_extrap, chi_vals[-1], dtype=float)


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
