"""
LFM-PAPER-040 Orbital Motion Emergence Experiments
===================================================

Reproduces all experiments from:
"Orbital Motion Emergence from Propagation Structure Without Mass"
by Greg D. Partin (2025)

Governing equation:
    d^2E/dt^2 = c^2 nabla^2 E - chi(x,y)^2 E

Run: python reproduce.py
Output: ../outputs/orbital/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Simulation Parameters (LOCKED - match paper exactly)
N = 512                    # Grid size
L = 40.0                   # Domain size
dx = L / N                 # Spatial step
c = 1.0                    # Wave speed
CFL = 0.3                  # CFL condition
dt = CFL * dx / (c * np.sqrt(2))  # Time step (2D stability factor)
chi_amplitude = 0.5        # chi field strength
chi_scale = 5.0            # chi field scale
chi_center = (L / 2, L / 2)  # chi field center


def create_chi_field():
    """Create radially symmetric chi field centered in domain."""
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - chi_center[0])**2 + (Y - chi_center[1])**2)
    chi = chi_amplitude * np.exp(-r / chi_scale)
    return chi, X, Y


def create_wavepacket(X, Y, x0, y0, kx, ky, sigma=1.0):
    """Create Gaussian wavepacket at (x0, y0) with momentum (kx, ky)."""
    r2 = (X - x0)**2 + (Y - y0)**2
    envelope = np.exp(-r2 / (2 * sigma**2))
    phase = np.exp(1j * (kx * X + ky * Y))
    return np.real(envelope * phase)


def leapfrog_step(E_prev, E_curr, chi):
    """Single leapfrog integration step."""
    laplacian = (
        np.roll(E_curr, 1, axis=0) + np.roll(E_curr, -1, axis=0) +
        np.roll(E_curr, 1, axis=1) + np.roll(E_curr, -1, axis=1) -
        4 * E_curr
    ) / dx**2
    
    E_next = 2 * E_curr - E_prev + dt**2 * (c**2 * laplacian - chi**2 * E_curr)
    return E_next


def track_centroid(E, X, Y):
    """Track energy-weighted centroid of wavepacket."""
    E2 = E**2
    total = np.sum(E2) + 1e-12
    cx = np.sum(X * E2) / total
    cy = np.sum(Y * E2) / total
    return cx, cy


def run_experiment_1(chi, X, Y, output_dir):
    """Exp 1: Trajectory curvature with tangential injection."""
    print("  Running Exp 1: Trajectory Curvature...")
    
    # Initialize wavepacket at edge, moving tangentially
    x0, y0 = L * 0.15, L * 0.5
    kx, ky = 0, 2.0  # Tangential motion
    E_prev = create_wavepacket(X, Y, x0, y0, kx, ky)
    E_curr = E_prev.copy()
    
    trajectory = []
    n_steps = 3000
    
    for step in range(n_steps):
        E_next = leapfrog_step(E_prev, E_curr, chi)
        E_prev, E_curr = E_curr, E_next
        if step % 50 == 0:
            cx, cy = track_centroid(E_curr, X, Y)
            trajectory.append((cx, cy))
    
    trajectory = np.array(trajectory)
    
    # Compute curvature
    dx_traj = np.gradient(trajectory[:, 0])
    dy_traj = np.gradient(trajectory[:, 1])
    d2x = np.gradient(dx_traj)
    d2y = np.gradient(dy_traj)
    speed = np.sqrt(dx_traj**2 + dy_traj**2) + 1e-12
    curvature = np.abs(dx_traj * d2y - dy_traj * d2x) / speed**3
    
    # Compare to straight line
    straight_line = np.column_stack([
        np.linspace(trajectory[0, 0], trajectory[-1, 0], len(trajectory)),
        np.linspace(trajectory[0, 1], trajectory[-1, 1], len(trajectory))
    ])
    deviation = np.sqrt(np.sum((trajectory - straight_line)**2, axis=1))
    max_deviation = np.max(deviation)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, chi, levels=20, cmap='viridis', alpha=0.5)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='Wave trajectory')
    ax.plot(straight_line[:, 0], straight_line[:, 1], 'w--', linewidth=1, label='Straight line')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exp 1: Trajectory Curvature in chi Field')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig(output_dir / 'exp1_trajectory_curvature.png', dpi=150)
    plt.close()
    
    result = {
        'mean_curvature': float(np.mean(curvature[10:-10])),
        'max_deviation': float(max_deviation),
        'curvature_ratio': float(max_deviation / (dx * len(trajectory)))
    }
    print(f"    Mean curvature: {result['mean_curvature']:.4f}")
    print(f"    Max deviation from straight: {result['max_deviation']:.2f}")
    return result


def run_experiment_2(chi, X, Y, output_dir):
    """Exp 2: Mass independence - compare two wavepackets with different amplitudes."""
    print("  Running Exp 2: Mass Independence...")
    
    # Two wavepackets with different "masses" (amplitudes)
    x0, y0 = L * 0.15, L * 0.5
    kx, ky = 0, 2.0
    
    E1_prev = create_wavepacket(X, Y, x0, y0, kx, ky) * 1.0
    E1_curr = E1_prev.copy()
    E2_prev = create_wavepacket(X, Y, x0, y0, kx, ky) * 3.0  # 3x amplitude
    E2_curr = E2_prev.copy()
    
    traj1, traj2 = [], []
    n_steps = 2000
    
    for step in range(n_steps):
        E1_next = leapfrog_step(E1_prev, E1_curr, chi)
        E2_next = leapfrog_step(E2_prev, E2_curr, chi)
        E1_prev, E1_curr = E1_curr, E1_next
        E2_prev, E2_curr = E2_curr, E2_next
        if step % 50 == 0:
            traj1.append(track_centroid(E1_curr, X, Y))
            traj2.append(track_centroid(E2_curr, X, Y))
    
    traj1, traj2 = np.array(traj1), np.array(traj2)
    
    # Compute trajectory difference
    diff = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
    max_diff = np.max(diff)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, chi, levels=20, cmap='viridis', alpha=0.5)
    ax.plot(traj1[:, 0], traj1[:, 1], 'r-', linewidth=2, label='Amplitude 1.0')
    ax.plot(traj2[:, 0], traj2[:, 1], 'b--', linewidth=2, label='Amplitude 3.0')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exp 2: Mass Independence Test')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig(output_dir / 'exp2_mass_independence.png', dpi=150)
    plt.close()
    
    result = {
        'max_trajectory_difference': float(max_diff),
        'mean_difference': float(np.mean(diff)),
        'is_mass_independent': bool(max_diff < 0.1)
    }
    print(f"    Max trajectory difference: {result['max_trajectory_difference']:.6f}")
    print(f"    Mass independent: {result['is_mass_independent']}")
    return result


def run_experiment_3(chi, X, Y, output_dir):
    """Exp 3: Orbital stability - long-term bound orbit test."""
    print("  Running Exp 3: Orbital Stability...")
    
    # Initialize with careful parameters for bound orbit
    x0, y0 = L * 0.25, L * 0.5
    kx, ky = 0, 1.5
    E_prev = create_wavepacket(X, Y, x0, y0, kx, ky, sigma=1.2)
    E_curr = E_prev.copy()
    
    trajectory = []
    n_steps = 6000
    
    for step in range(n_steps):
        E_next = leapfrog_step(E_prev, E_curr, chi)
        E_prev, E_curr = E_curr, E_next
        if step % 50 == 0:
            cx, cy = track_centroid(E_curr, X, Y)
            trajectory.append((cx, cy))
    
    trajectory = np.array(trajectory)
    
    # Check if trajectory stays bound (within some radius of center)
    distances = np.sqrt((trajectory[:, 0] - chi_center[0])**2 + 
                        (trajectory[:, 1] - chi_center[1])**2)
    max_distance = np.max(distances)
    min_distance = np.min(distances)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, chi, levels=20, cmap='viridis', alpha=0.5)
    colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory)))
    for i in range(len(trajectory) - 1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=colors[i], linewidth=1.5)
    ax.scatter(*chi_center, c='white', s=100, marker='x', label='chi center')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exp 3: Long-term Orbital Stability')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig(output_dir / 'exp3_orbital_stability.png', dpi=150)
    plt.close()
    
    result = {
        'max_distance': float(max_distance),
        'min_distance': float(min_distance),
        'orbital_eccentricity': float((max_distance - min_distance) / (max_distance + min_distance)),
        'bound': bool(max_distance < L * 0.4)
    }
    print(f"    Orbital range: {result['min_distance']:.2f} to {result['max_distance']:.2f}")
    print(f"    Bound orbit: {result['bound']}")
    return result


def run_experiment_4(chi, X, Y, output_dir):
    """Exp 4: chi removal test - remove chi mid-simulation, curvature should vanish."""
    print("  Running Exp 4: chi Removal Test...")
    
    # Initialize at orbital position
    x0, y0 = L * 0.75, L * 0.5  # Offset from center
    kx, ky = 0, 3.0  # Tangential velocity
    
    E_prev = create_wavepacket(X, Y, x0, y0, kx, ky)
    E_curr = E_prev.copy()
    
    n_steps = 4000
    half_steps = n_steps // 2
    
    # Phase 1: With structured chi
    traj_phase1 = []
    chi_active = chi.copy()
    
    for step in range(half_steps):
        E_next = leapfrog_step(E_prev, E_curr, chi_active)
        E_prev, E_curr = E_curr, E_next
        if step % 50 == 0:
            cx, cy = track_centroid(E_curr, X, Y)
            traj_phase1.append((cx, cy))
    
    # Phase 2: Remove chi (set to zero)
    traj_phase2 = []
    chi_removed = np.zeros_like(chi)
    
    for step in range(half_steps):
        E_next = leapfrog_step(E_prev, E_curr, chi_removed)
        E_prev, E_curr = E_curr, E_next
        if step % 50 == 0:
            cx, cy = track_centroid(E_curr, X, Y)
            traj_phase2.append((cx, cy))
    
    traj_phase1 = np.array(traj_phase1)
    traj_phase2 = np.array(traj_phase2)
    
    # Compute curvature for each phase
    def compute_curvature(traj):
        if len(traj) < 5:
            return 0.0
        dx = np.gradient(traj[:, 0])
        dy = np.gradient(traj[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        kappa = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-10)**1.5
        return float(np.nanmean(kappa[2:-2]))
    
    curv_with = compute_curvature(traj_phase1)
    curv_without = compute_curvature(traj_phase2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(X, Y, chi, levels=20, cmap='viridis', alpha=0.3)
    ax.plot(traj_phase1[:, 0], traj_phase1[:, 1], 'b-', linewidth=2, label='Phase 1: With chi')
    ax.plot(traj_phase2[:, 0], traj_phase2[:, 1], 'r-', linewidth=2, label='Phase 2: chi removed')
    ax.axvline(traj_phase1[-1, 0], color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Exp 4: chi Removal - Curved to Straight Transition')
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig(output_dir / 'exp4_chi_removal.png', dpi=150)
    plt.close()
    
    reduction = (curv_with - curv_without) / max(curv_with, 1e-10) * 100
    
    result = {
        'curvature_with_chi': float(curv_with),
        'curvature_without_chi': float(curv_without),
        'curvature_reduction': float(reduction)
    }
    print(f"    Curvature with chi: {result['curvature_with_chi']:.6f}")
    print(f"    Curvature without chi: {result['curvature_without_chi']:.6f}")
    print(f"    Curvature reduction: {result['curvature_reduction']:.1f}%")
    return result


def main():
    print("=" * 60)
    print("LFM-PAPER-040: Orbital Motion Emergence Experiments")
    print("=" * 60)
    print(f"\nSimulation Parameters (LOCKED):")
    print(f"  Grid: {N}x{N}, Domain: {L}x{L}")
    print(f"  dx={dx:.4f}, dt={dt:.6f}, c={c}")
    print(f"  chi amplitude={chi_amplitude}, scale={chi_scale}")
    print()
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / 'outputs' / 'orbital'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create chi field
    chi, X, Y = create_chi_field()
    
    # Run all experiments
    results = {}
    results['exp1'] = run_experiment_1(chi, X, Y, output_dir)
    results['exp2'] = run_experiment_2(chi, X, Y, output_dir)
    results['exp3'] = run_experiment_3(chi, X, Y, output_dir)
    results['exp4'] = run_experiment_4(chi, X, Y, output_dir)
    
    # Add metadata
    results['parameters'] = {
        'N': N, 'L': L, 'dx': dx, 'dt': dt, 'c': c,
        'chi_amplitude': chi_amplitude, 'chi_scale': chi_scale
    }
    
    # Determine pass/fail
    results['exp1']['pass'] = results['exp1']['curvature_ratio'] > 0.01
    results['exp2']['pass'] = results['exp2']['is_mass_independent']
    results['exp3']['pass'] = results['exp3']['bound']
    results['exp4']['pass'] = results['exp4']['curvature_reduction'] > 90
    
    all_pass = all(results[f'exp{i}']['pass'] for i in range(1, 5))
    results['all_pass'] = all_pass
    
    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for i in range(1, 5):
        status = "PASS" if results[f'exp{i}']['pass'] else "FAIL"
        print(f"  Exp {i}: {status}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
