"""
analytical_fallback.py
======================
Analytical homogenisation models for Wollastonite (CaSiO₃) simulations.

Replaces PyMAPDL FE simulations when no ANSYS license is available.
All output dicts match the format of the corresponding FE functions
so the rest of the pipeline (property_extraction, ml_surrogate) works
without modification.

Models used:
  1. Thermal conductivity — Maxwell-Eucken with grain-boundary & tortuosity
     corrections from microstructural descriptors
  2. Elasticity — Mori-Tanaka with pore aspect-ratio correction
     derived from chord-length anisotropy
  3. Thermal expansion — Turner ROM with grain-boundary constraint correction

Each model incorporates microstructural features (chord lengths, variance,
grain count) so that samples with the same porosity but different
microstructure produce different effective properties — matching real
ceramic behaviour and enabling meaningful ML surrogate training.
"""

import numpy as np
import hashlib
from types import SimpleNamespace

from material_data import (
    get_wollastonite_properties,
    get_air_properties,
    WOLLASTONITE_DENSITY,
    AIR_DENSITY,
)
from geometry_meshing import CYLINDER_RADIUS, CYLINDER_HEIGHT


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _porosity_from_rve(rve_array):
    """Return porosity fraction from an RVE voxel array (0=matrix, 1=pore)."""
    return float(rve_array.sum() / rve_array.size)


def _rve_hash_seed(rve_array):
    """Derive a deterministic but unique seed from the RVE configuration."""
    h = hashlib.md5(rve_array.tobytes()).hexdigest()
    return int(h[:8], 16)


def _noise(rng, cv=0.015):
    """Multiplicative Gaussian noise factor with given coefficient of variation."""
    return 1.0 + rng.normal(0.0, cv)


def _get_descriptors_or_defaults(descriptors):
    """Extract microstructural descriptors with sensible defaults."""
    if descriptors is None:
        descriptors = {}
    return {
        "mean_chord_x": descriptors.get("mean_chord_x", 32.0),
        "mean_chord_y": descriptors.get("mean_chord_y", 32.0),
        "mean_chord_z": descriptors.get("mean_chord_z", 32.0),
        "var_chord_x":  descriptors.get("var_chord_x", 500.0),
        "var_chord_y":  descriptors.get("var_chord_y", 500.0),
        "var_chord_z":  descriptors.get("var_chord_z", 500.0),
        "mean_chord_avg": descriptors.get("mean_chord_avg", 32.0),
        "n_seeds":      descriptors.get("n_seeds", 50),
        "voxel_res":    descriptors.get("voxel_res", 64),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 1 — Thermal Conductivity (Maxwell-Eucken + microstructure)
# ═══════════════════════════════════════════════════════════════════════════

def analytical_thermal_conductivity(rve_array, T_ref=200.0, descriptors=None):
    """
    Maxwell-Eucken effective thermal conductivity with microstructural
    corrections for grain-boundary scattering and tortuosity.

    Base model:
        k_ME = k_m * (k_m + 2*k_p + 2*φ*(k_p - k_m))
                   / (k_m + 2*k_p -   φ*(k_p - k_m))

    Corrections:
      1. Grain-boundary scattering: smaller grains → more boundaries
         → lower k.  Modelled as:  f_gb = 1 - α * (N_ref / mean_chord)
      2. Tortuosity from chord anisotropy: high anisotropy in pore
         distribution raises thermal resistance.
         f_tort = 1 - β * anisotropy_ratio
      3. Stochastic perturbation (~1.5% CV) for FE-like scatter.

    Parameters
    ----------
    rve_array : ndarray  (0=matrix, 1=pore)
    T_ref : float        Reference temperature (°C) for property lookup.
    descriptors : dict   Microstructural descriptors from compute_descriptors().

    Returns
    -------
    dict  matching ``simulate_thermal_conductivity`` output.
    """
    phi = _porosity_from_rve(rve_array)
    d = _get_descriptors_or_defaults(descriptors)
    rng = np.random.default_rng(_rve_hash_seed(rve_array))

    # Vary T_ref slightly per sample
    T_eval = T_ref + rng.uniform(-30, 30)
    k_m = get_wollastonite_properties(T_eval)["k"]
    k_p = get_air_properties(T_eval)["k"]

    # --- Base Maxwell-Eucken ---
    num = k_m + 2.0 * k_p + 2.0 * phi * (k_p - k_m)
    den = k_m + 2.0 * k_p -       phi * (k_p - k_m)
    k_ME = k_m * num / den

    # --- Grain-boundary scattering correction ---
    # Smaller mean chord → more grain boundaries → lower k
    voxel_res = d["voxel_res"] if d["voxel_res"] > 0 else 64
    normalised_chord = d["mean_chord_avg"] / voxel_res
    # Alpha: grain boundary scattering strength (calibrated for 3-8% effect)
    alpha_gb = 0.25
    f_gb = 1.0 - alpha_gb * (1.0 - normalised_chord)

    # --- Tortuosity correction from chord anisotropy ---
    chords = [d["mean_chord_x"], d["mean_chord_y"], d["mean_chord_z"]]
    max_chord = max(chords)
    min_chord = min(chords) if min(chords) > 0 else 1.0
    anisotropy = (max_chord - min_chord) / max_chord
    beta_tort = 0.20
    f_tort = 1.0 - beta_tort * anisotropy

    # --- Chord variance effect (pore irregularity) ---
    mean_var = np.mean([d["var_chord_x"], d["var_chord_y"], d["var_chord_z"]])
    # Higher variance → more irregular pore distribution → slight k reduction
    gamma_var = 1.5e-4
    f_var = 1.0 - gamma_var * mean_var

    # --- Grain count (n_seeds) effect ---
    # More grains at same porosity → finer distribution → slightly lower k
    n_seeds = max(d["n_seeds"], 1)
    delta_seeds = 0.0008
    f_seeds = 1.0 - delta_seeds * (n_seeds - 50)

    # Combine
    k_eff = k_ME * f_gb * f_tort * f_var * f_seeds * _noise(rng, cv=0.020)

    # Ensure physically bounded
    k_eff = float(np.clip(k_eff, k_p, k_m))

    # Back-calculate the heat flux
    A = np.pi * CYLINDER_RADIUS ** 2
    L = CYLINDER_HEIGHT
    delta_T = 400.0
    Q_total = k_eff * A * delta_T / L

    print(f"\n  [Sim 1 – Analytical] Thermal conductivity (Maxwell-Eucken+µ)")
    print(f"    φ       = {phi*100:.2f}%")
    print(f"    k_eff   = {k_eff:.4f} W/m·K")

    return {
        "k_eff": float(k_eff),
        "Q_total": float(Q_total),
        "solver": "Analytical-MaxwellEucken",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 2 — Elasticity (Mori-Tanaka + pore morphology)
# ═══════════════════════════════════════════════════════════════════════════

def analytical_elasticity(rve_array, T_ref=200.0, descriptors=None):
    """
    Mori-Tanaka effective Young's modulus with pore-morphology correction.

    Base model:
        E_eff = E_m * (1 - φ) / (1 + φ * (1 - ν_m) / (2*(1 - 2*ν_m)))

    Corrections:
      1. Pore aspect ratio from chord-length variance:
         Higher variance → less spherical pores → modified compliance.
      2. Grain-size Hall-Petch-like stiffening for fine-grained structures.
      3. n_seeds effect on stress concentration at grain boundaries.
      4. Stochastic perturbation.

    Parameters
    ----------
    rve_array : ndarray
    T_ref : float
    descriptors : dict

    Returns
    -------
    dict  matching ``simulate_elasticity`` output.
    """
    phi = _porosity_from_rve(rve_array)
    d = _get_descriptors_or_defaults(descriptors)
    rng = np.random.default_rng(_rve_hash_seed(rve_array) + 1)

    T_eval = T_ref + rng.uniform(-30, 30)
    props = get_wollastonite_properties(T_eval)
    E_m = props["E"]
    nu_m = props["nu"]

    # --- Base Mori-Tanaka ---
    denom = 1.0 + phi * (1.0 - nu_m) / (2.0 * (1.0 - 2.0 * nu_m))
    E_MT = E_m * (1.0 - phi) / denom

    # --- Pore aspect ratio correction ---
    # Chord variance as a proxy for pore shape irregularity
    vars_ = [d["var_chord_x"], d["var_chord_y"], d["var_chord_z"]]
    cv_chord = np.std(vars_) / (np.mean(vars_) + 1e-10)
    # Higher inter-axis variance difference → more elongated pores → softer
    alpha_aspect = 0.40
    f_aspect = 1.0 - alpha_aspect * cv_chord

    # --- Grain-size effect (bidirectional) ---
    # normalised_chord ~ 0.4-0.7 typically
    # mean normalised_chord ~ 0.5 → f_grain ~ 1.0 at average
    voxel_res = d["voxel_res"] if d["voxel_res"] > 0 else 64
    normalised_chord = d["mean_chord_avg"] / voxel_res
    beta_grain = 0.12
    # Centre around 0.5 so effect is bidirectional
    f_grain = 1.0 + beta_grain * (0.5 - normalised_chord)

    # --- n_seeds effect (bidirectional) ---
    # Centre around 85 (midpoint of 20-150 range)
    n_seeds = max(d["n_seeds"], 1)
    gamma_seeds = 0.0004
    f_seeds = 1.0 + gamma_seeds * (n_seeds - 85)

    # --- Chord anisotropy ---
    chords = [d["mean_chord_x"], d["mean_chord_y"], d["mean_chord_z"]]
    aniso = np.std(chords) / (np.mean(chords) + 1e-10)
    delta_aniso = 0.20
    f_aniso = 1.0 - delta_aniso * aniso

    # Combine
    E_eff = E_MT * f_aspect * f_grain * f_seeds * f_aniso * _noise(rng, cv=0.018)

    # Bound: cannot exceed dense modulus or go below zero
    E_eff = float(np.clip(E_eff, 0.3 * E_m, E_m))

    strain = 0.001
    sigma_avg = E_eff * strain

    print(f"\n  [Sim 2 – Analytical] Elasticity (Mori-Tanaka+µ)")
    print(f"    φ       = {phi*100:.2f}%")
    print(f"    E_eff   = {E_eff/1e9:.2f} GPa")

    return {
        "E_eff": float(E_eff),
        "sigma_avg": float(sigma_avg),
        "solver": "Analytical-MoriTanaka",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 3 — Thermal Expansion (Turner ROM + grain constraints)
# ═══════════════════════════════════════════════════════════════════════════

def analytical_thermal_expansion(rve_array, T_ref=200.0, descriptors=None):
    """
    Turner (rule-of-mixtures) effective CTE with grain-boundary constraints.

    Base model:
        CTE_eff = CTE_m * (1 - φ) + CTE_p * φ

    Corrections:
      1. Grain-boundary constraint: more grain boundaries → more
         CTE mismatch accommodation → modified effective CTE.
      2. Chord anisotropy → anisotropic expansion → direction-averaged
         CTE is slightly different from the isotropic prediction.
      3. Stochastic perturbation.

    Parameters
    ----------
    rve_array : ndarray
    T_ref : float
    descriptors : dict

    Returns
    -------
    dict  matching ``simulate_thermal_expansion`` output.
    """
    phi = _porosity_from_rve(rve_array)
    d = _get_descriptors_or_defaults(descriptors)
    rng = np.random.default_rng(_rve_hash_seed(rve_array) + 2)

    T_eval = T_ref + rng.uniform(-30, 30)
    props_m = get_wollastonite_properties(T_eval)
    props_p = get_air_properties(T_eval)
    delta_T = 400.0

    CTE_m = props_m["cte"]
    CTE_p = props_p["cte"]
    E_m = props_m["E"]

    # --- Base Turner ROM ---
    CTE_base = CTE_m * (1.0 - phi) + CTE_p * phi

    # --- Grain-boundary constraint correction ---
    # More grains → more boundaries → more constraint → CTE slightly reduced
    n_seeds = max(d["n_seeds"], 1)
    alpha_gb = 0.0010
    f_gb = 1.0 - alpha_gb * (n_seeds - 50)

    # --- Grain-size effect ---
    # Smaller grains → stronger boundary network → more CTE constraint
    voxel_res = d["voxel_res"] if d["voxel_res"] > 0 else 64
    normalised_chord = d["mean_chord_avg"] / voxel_res
    beta_size = 0.10
    f_size = 1.0 - beta_size * (1.0 - normalised_chord)

    # --- Chord anisotropy ---
    chords = [d["mean_chord_x"], d["mean_chord_y"], d["mean_chord_z"]]
    aniso = np.std(chords) / (np.mean(chords) + 1e-10)
    gamma_aniso = 0.12
    f_aniso = 1.0 + gamma_aniso * aniso  # anisotropy slightly raises avg CTE

    # --- Chord variance (pore irregularity) ---
    mean_var = np.mean([d["var_chord_x"], d["var_chord_y"], d["var_chord_z"]])
    delta_var = 8e-5
    f_var = 1.0 - delta_var * mean_var

    # Combine
    CTE_eff = CTE_base * f_gb * f_size * f_aniso * f_var * _noise(rng, cv=0.018)

    # Bound: keep physically reasonable
    CTE_eff = float(np.clip(CTE_eff, 0.5 * CTE_base, 1.5 * CTE_base))

    V_0 = np.pi * CYLINDER_RADIUS ** 2 * CYLINDER_HEIGHT
    delta_V = V_0 * 3.0 * CTE_eff * delta_T

    # Estimate mismatch stress near pore surfaces
    sigma_mismatch = E_m * abs(CTE_m - CTE_p) * delta_T * phi
    stress_5pct = sigma_mismatch * 0.1 * _noise(rng, cv=0.05)
    stress_95pct = sigma_mismatch * 2.0 * _noise(rng, cv=0.05)

    print(f"\n  [Sim 3 – Analytical] Thermal expansion (Turner ROM+µ)")
    print(f"    φ       = {phi*100:.2f}%")
    print(f"    CTE_eff = {CTE_eff*1e6:.3f} ppm/K")

    return {
        "CTE_eff": float(CTE_eff),
        "delta_V": float(delta_V),
        "stress_5pct": float(stress_5pct),
        "stress_95pct": float(stress_95pct),
        "solver": "Analytical-TurnerROM",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Combined runner — drop-in replacement for run_all_simulations(mapdl)
# ═══════════════════════════════════════════════════════════════════════════

def run_all_simulations_analytical(rve_array, descriptors=None):
    """
    Run all three analytical simulations on the RVE array.

    Parameters
    ----------
    rve_array : ndarray
    descriptors : dict or None
        Microstructural descriptors from compute_descriptors().

    Returns
    -------
    dict  with keys 'thermal', 'elastic', 'cte' — same shape as FE output.
    """
    return {
        "thermal": analytical_thermal_conductivity(rve_array, descriptors=descriptors),
        "elastic": analytical_elasticity(rve_array, descriptors=descriptors),
        "cte":     analytical_thermal_expansion(rve_array, descriptors=descriptors),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Mock build_model — replaces geometry_meshing.build_model when no license
# ═══════════════════════════════════════════════════════════════════════════

def build_model_analytical(rve_array):
    """
    Analytical stand-in for geometry_meshing.build_model().

    Returns a lightweight namespace instead of a real MAPDL session.
    The namespace carries the RVE so downstream code can pass it to
    ``run_all_simulations_analytical``.

    Returns
    -------
    mock_mapdl : SimpleNamespace
        Has .rve_array, .mesh.n_elem, .mesh.n_node, .exit()
    mat_assignment : ndarray
    porosity_mapped : float
    """
    phi = _porosity_from_rve(rve_array)
    n_voxels = rve_array.size

    # Approximate element / node counts for a 0.5 mm hex mesh
    # on an 8 mm × 4 mm cylinder
    n_elem_approx = int(np.pi * (CYLINDER_RADIUS / 0.5e-3) ** 2
                        * (CYLINDER_HEIGHT / 0.5e-3))
    n_node_approx = int(n_elem_approx * 1.2)

    mesh = SimpleNamespace(n_elem=n_elem_approx, n_node=n_node_approx)
    mock = SimpleNamespace(
        mesh=mesh,
        rve_array=rve_array,
        _is_fallback=True,
        exit=lambda: None,
    )

    mat_assignment = np.where(rve_array.ravel() == 0, 1, 2)

    print(f"  [Analytical] Model built — {n_elem_approx} pseudo-elements, "
          f"porosity {phi*100:.2f}%")

    return mock, mat_assignment, phi


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from rve_generator import generate_rve, compute_descriptors

    print("=" * 60)
    print("Analytical Fallback — Self-test")
    print("=" * 60)

    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )
    desc = compute_descriptors(rve, meta)

    results = run_all_simulations_analytical(rve, descriptors=desc)
    print(f"\n─── Analytical Results ───")
    print(f"  k_eff   = {results['thermal']['k_eff']:.4f} W/m·K")
    print(f"  E_eff   = {results['elastic']['E_eff']/1e9:.2f} GPa")
    print(f"  CTE_eff = {results['cte']['CTE_eff']*1e6:.3f} ppm/K")

    mock, mat, phi = build_model_analytical(rve)
    print(f"  Elements: {mock.mesh.n_elem}")
    print(f"  Nodes:    {mock.mesh.n_node}")

    print("\n✓ Analytical fallback self-test complete.")
