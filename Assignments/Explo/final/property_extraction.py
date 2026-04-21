"""
property_extraction.py
======================
Phase 5 — Property Extraction for Wollastonite (CaSiO₃)

Converts raw FE simulation outputs and RVE data into the four
target engineering properties:
  1. Effective thermal conductivity (from FE simulation)
  2. Bulk density (from voxel array — rule of mixtures)
  3. Porosity (from voxel array)
  4. Vickers hardness (Rice-Duckworth porosity correction)

Also packages everything into a single JSON-serializable record
for the structure–property database.
"""

import numpy as np
import json
import hashlib

from material_data import (
    WOLLASTONITE_DENSITY,
    WOLLASTONITE_HARDNESS_DENSE,
    AIR_DENSITY,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Thermal conductivity — direct from FE simulation
# ═══════════════════════════════════════════════════════════════════════════

def extract_thermal_conductivity(sim_result):
    """
    Extract effective thermal conductivity from Phase 4 simulation.

    Parameters
    ----------
    sim_result : dict
        Output of ``simulate_thermal_conductivity()``.
        Must contain key 'k_eff'.

    Returns
    -------
    k_eff : float
        Effective thermal conductivity (W/m·K).
    """
    return float(sim_result["k_eff"])


# ═══════════════════════════════════════════════════════════════════════════
#  Density and porosity — from RVE voxel array
# ═══════════════════════════════════════════════════════════════════════════

def extract_density_porosity(rve_array,
                              rho_matrix=WOLLASTONITE_DENSITY,
                              rho_air=AIR_DENSITY,
                              descriptors=None):
    """
    Compute effective density and porosity from the RVE voxel array.

    Porosity: fraction of voxels labelled as pore (value 1).
    Density:  rule of mixtures with microstructural corrections:
        ρ_eff = (1 − φ) · ρ_matrix + φ · ρ_air + corrections

    Corrections account for:
      - Grain-boundary density deficit (finer grains → more boundaries
        → slightly lower packing density)
      - Small measurement uncertainty noise

    Parameters
    ----------
    rve_array : ndarray
        3D array from Phase 2 (0 = matrix, 1 = pore).
    rho_matrix : float
        Matrix density (kg/m³). Default: Wollastonite.
    rho_air : float
        Pore fluid density (kg/m³). Default: air.
    descriptors : dict or None
        Microstructural descriptors from compute_descriptors().

    Returns
    -------
    density : float
        Effective density (kg/m³).
    porosity : float
        Porosity fraction (0 to 1).
    """
    total = rve_array.size
    n_pore = int(rve_array.sum())
    porosity = n_pore / total

    density = (1.0 - porosity) * rho_matrix + porosity * rho_air

    # --- Microstructural correction ---
    if descriptors is not None:
        h = hashlib.md5(rve_array.tobytes()).hexdigest()
        rng = np.random.default_rng(int(h[:8], 16) + 100)

        # Grain-boundary density deficit
        n_seeds = descriptors.get("n_seeds", 50)
        voxel_res = descriptors.get("voxel_res", 64)
        mean_chord = descriptors.get("mean_chord_avg", 32.0)
        normalised_chord = mean_chord / voxel_res if voxel_res > 0 else 0.5

        # More grain boundaries (smaller chord) → slight density reduction
        gb_correction = -rho_matrix * 0.003 * (1.0 - normalised_chord)
        density += gb_correction

        # Small measurement noise (~0.15% CV)
        density *= (1.0 + rng.normal(0.0, 0.0015))

    return float(density), float(porosity)


# ═══════════════════════════════════════════════════════════════════════════
#  Vickers hardness — Rice-Duckworth model
# ═══════════════════════════════════════════════════════════════════════════

def extract_hardness(porosity,
                     E_eff,
                     E_dense=None,
                     HV_dense=WOLLASTONITE_HARDNESS_DENSE,
                     b_porosity=7.0,
                     descriptors=None,
                     rve_array=None):
    """
    Estimate Vickers hardness using the Rice-Duckworth exponential model
    with a Hall-Petch-like grain-size correction.

    HV = HV_dense · exp(−b · φ) · (E_eff / E_dense) · f_hp · f_var

    The first term is the exponential porosity correction.
    The second term scales by the stiffness ratio.
    f_hp is a Hall-Petch grain-size correction (smaller grains → harder).
    f_var accounts for chord-length variance (pore regularity).

    Parameters
    ----------
    porosity : float
        Porosity fraction (0 to 1).
    E_eff : float
        Effective Young's modulus from FE simulation (Pa).
    E_dense : float or None
        Dense-material Young's modulus (Pa).
        Default: Wollastonite at 0 °C (96 GPa).
    HV_dense : float
        Hardness of fully dense material (HV).
    b_porosity : float
        Exponential decay coefficient (typical 4–10 for oxide ceramics).
    descriptors : dict or None
        Microstructural descriptors for grain-size correction.
    rve_array : ndarray or None
        RVE voxel array for noise seeding.

    Returns
    -------
    HV : float
        Estimated Vickers hardness (HV).
    """
    if E_dense is None:
        from material_data import WOLLASTONITE_YOUNGS_MODULUS
        E_dense = float(WOLLASTONITE_YOUNGS_MODULUS[0])   # 96 GPa at 0 °C

    # Rice-Duckworth with stiffness scaling
    HV = HV_dense * np.exp(-b_porosity * porosity) * (E_eff / E_dense)

    # --- Microstructural corrections ---
    if descriptors is not None:
        voxel_res = descriptors.get("voxel_res", 64)
        mean_chord = descriptors.get("mean_chord_avg", 32.0)
        normalised_chord = mean_chord / voxel_res if voxel_res > 0 else 0.5

        # Hall-Petch-like: smaller grains → harder
        # f_hp > 1 for fine grains, < 1 for coarse grains
        alpha_hp = 0.06
        f_hp = 1.0 + alpha_hp * (1.0 - normalised_chord)
        HV *= f_hp

        # Chord-length variance: more irregular pore shapes → stress
        # concentrators → slightly lower hardness
        mean_var = np.mean([
            descriptors.get("var_chord_x", 500.0),
            descriptors.get("var_chord_y", 500.0),
            descriptors.get("var_chord_z", 500.0),
        ])
        beta_var = 2e-5
        f_var = 1.0 - beta_var * mean_var
        HV *= f_var

        # n_seeds effect
        n_seeds = descriptors.get("n_seeds", 50)
        gamma_seeds = 0.0002
        f_seeds = 1.0 + gamma_seeds * (n_seeds - 80)
        HV *= f_seeds

    # --- Stochastic noise ---
    if rve_array is not None:
        h = hashlib.md5(rve_array.tobytes()).hexdigest()
        rng = np.random.default_rng(int(h[:8], 16) + 200)
        HV *= (1.0 + rng.normal(0.0, 0.012))

    return float(HV)


# ═══════════════════════════════════════════════════════════════════════════
#  Build a single database record
# ═══════════════════════════════════════════════════════════════════════════

def build_record(descriptors, sim_results, rve_array,
                 rho_matrix=WOLLASTONITE_DENSITY,
                 rho_air=AIR_DENSITY):
    """
    Package all descriptors and properties into a single
    JSON-serializable record for the structure–property database.

    Parameters
    ----------
    descriptors : dict
        Output of ``compute_descriptors()`` from Phase 2.
    sim_results : dict
        Output of ``run_all_simulations()`` from Phase 4.
        Expected keys: 'thermal', 'elastic', 'cte'.
    rve_array : ndarray
        RVE voxel array (0=matrix, 1=pore).
    rho_matrix, rho_air : float
        Phase densities.

    Returns
    -------
    record : dict
        Complete structure–property record.
    """
    # Extract properties
    k_eff = extract_thermal_conductivity(sim_results["thermal"])
    density, porosity = extract_density_porosity(
        rve_array, rho_matrix, rho_air, descriptors=descriptors
    )

    E_eff = sim_results["elastic"]["E_eff"]
    HV = extract_hardness(
        porosity, E_eff,
        descriptors=descriptors, rve_array=rve_array
    )

    CTE_eff = sim_results["cte"]["CTE_eff"]

    record = {
        # --- Microstructural descriptors ---
        "porosity": porosity,
        "mean_chord_x": descriptors.get("mean_chord_x", 0.0),
        "mean_chord_y": descriptors.get("mean_chord_y", 0.0),
        "mean_chord_z": descriptors.get("mean_chord_z", 0.0),
        "var_chord_x":  descriptors.get("var_chord_x", 0.0),
        "var_chord_y":  descriptors.get("var_chord_y", 0.0),
        "var_chord_z":  descriptors.get("var_chord_z", 0.0),
        "mean_chord_avg": descriptors.get("mean_chord_avg", 0.0),
        "n_seeds": descriptors.get("n_seeds", 0),
        "voxel_res": descriptors.get("voxel_res", 0),

        # --- Simulation outputs ---
        "k_eff_WpmK": k_eff,
        "E_eff_GPa": E_eff / 1e9,
        "CTE_eff_ppmK": CTE_eff * 1e6,
        "density_kgpm3": density,
        "porosity_pct": porosity * 100.0,
        "HV": HV,

        # --- Stress field statistics ---
        "stress_5pct_Pa": sim_results["cte"].get("stress_5pct", 0.0),
        "stress_95pct_Pa": sim_results["cte"].get("stress_95pct", 0.0),

        # --- Metadata ---
        "solver": sim_results["thermal"].get("solver", "unknown"),
    }

    return record


def save_records(records, filepath):
    """Save a list of records to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  → Database saved: {filepath} ({len(records)} records)")


def load_records(filepath):
    """Load records from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5 — Property Extraction")
    print("=" * 60)

    # Test with synthetic data
    from rve_generator import generate_rve, compute_descriptors

    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )
    desc = compute_descriptors(rve, meta)

    # Synthetic simulation results (as would come from Phase 4)
    sim_results = {
        "thermal": {"k_eff": 3.1, "Q_total": 0.005, "solver": "test"},
        "elastic": {"E_eff": 88e9, "sigma_avg": 88e6, "solver": "test"},
        "cte": {
            "CTE_eff": 7.0e-6, "delta_V": 1e-12,
            "stress_5pct": 1e5, "stress_95pct": 5e7,
            "solver": "test",
        },
    }

    record = build_record(desc, sim_results, rve)

    print("\nSample record:")
    for key, val in record.items():
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")

    print("\n✓ Property extraction complete.")
