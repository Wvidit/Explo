"""
material_data.py
================
Phase 1 — Material Data Assembly for Wollastonite (CaSiO₃)

Temperature-dependent material properties for:
  1. Dense Wollastonite (matrix phase)
  2. Air (pore phase)

Properties are tabulated at 0, 100, 200, 300, 400 °C and
interpolated via scipy for any intermediate temperature.
"""

import numpy as np
from scipy.interpolate import interp1d

# ─── Temperature grid (°C) ────────────────────────────────────────────────
TEMP_POINTS = np.array([0.0, 100.0, 200.0, 300.0, 400.0])

# ═══════════════════════════════════════════════════════════════════════════
#  WOLLASTONITE (CaSiO₃) — Dense, pore-free reference values
# ═══════════════════════════════════════════════════════════════════════════

# Young's modulus (Pa) — ~96 GPa, slight softening with temperature
WOLLASTONITE_YOUNGS_MODULUS = np.array([
    96.0e9, 95.5e9, 94.8e9, 94.0e9, 93.0e9
])

# Poisson's ratio — nearly constant
WOLLASTONITE_POISSONS_RATIO = np.array([
    0.27, 0.27, 0.27, 0.27, 0.27
])

# Bulk density (kg/m³) — constant (thermal expansion effect negligible)
WOLLASTONITE_DENSITY = 2900.0   # kg/m³

# Thermal conductivity (W/m·K) — decreasing with temperature
WOLLASTONITE_THERMAL_CONDUCTIVITY = np.array([
    3.50, 3.25, 3.00, 2.80, 2.60
])

# Coefficient of thermal expansion (1/K) — increasing with temperature
WOLLASTONITE_CTE = np.array([
    6.5e-6, 6.7e-6, 7.0e-6, 7.3e-6, 7.5e-6
])

# Specific heat capacity (J/kg·K) — increasing with temperature
WOLLASTONITE_SPECIFIC_HEAT = np.array([
    840.0, 865.0, 890.0, 915.0, 940.0
])

# Vickers hardness of fully dense Wollastonite (HV)
WOLLASTONITE_HARDNESS_DENSE = 570.0   # HV

# ═══════════════════════════════════════════════════════════════════════════
#  AIR — Pore phase properties
# ═══════════════════════════════════════════════════════════════════════════

# Near-zero stiffness to avoid FE singularity (Pa)
AIR_YOUNGS_MODULUS = np.array([
    1.0e3, 1.0e3, 1.0e3, 1.0e3, 1.0e3
])

# Poisson's ratio for air (numerical placeholder)
AIR_POISSONS_RATIO = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0
])

# Air density (kg/m³)
AIR_DENSITY = 1.2   # kg/m³

# Thermal conductivity of air (W/m·K)
AIR_THERMAL_CONDUCTIVITY = np.array([
    0.0243, 0.0314, 0.0386, 0.0457, 0.0529
])

# CTE for air — not physically meaningful, set to zero
AIR_CTE = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Specific heat of air at constant pressure (J/kg·K)
AIR_SPECIFIC_HEAT = np.array([
    1005.0, 1009.0, 1013.0, 1017.0, 1021.0
])


# ═══════════════════════════════════════════════════════════════════════════
#  Interpolation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_interp(values, kind="linear"):
    """Create an interpolation function over the standard temperature grid."""
    return interp1d(TEMP_POINTS, values, kind=kind,
                    bounds_error=False, fill_value="extrapolate")


# Pre-built interpolators for Wollastonite
wollastonite_E    = _make_interp(WOLLASTONITE_YOUNGS_MODULUS)
wollastonite_nu   = _make_interp(WOLLASTONITE_POISSONS_RATIO)
wollastonite_k    = _make_interp(WOLLASTONITE_THERMAL_CONDUCTIVITY)
wollastonite_cte  = _make_interp(WOLLASTONITE_CTE)
wollastonite_cp   = _make_interp(WOLLASTONITE_SPECIFIC_HEAT)

# Pre-built interpolators for Air
air_E    = _make_interp(AIR_YOUNGS_MODULUS)
air_nu   = _make_interp(AIR_POISSONS_RATIO)
air_k    = _make_interp(AIR_THERMAL_CONDUCTIVITY)
air_cte  = _make_interp(AIR_CTE)
air_cp   = _make_interp(AIR_SPECIFIC_HEAT)


def get_wollastonite_properties(T):
    """
    Return all Wollastonite properties at temperature T (°C).

    Parameters
    ----------
    T : float or array-like
        Temperature in °C (0–400 range).

    Returns
    -------
    dict with keys: E, nu, rho, k, cte, cp, HV_dense
    """
    return {
        "E":        float(wollastonite_E(T)),
        "nu":       float(wollastonite_nu(T)),
        "rho":      WOLLASTONITE_DENSITY,
        "k":        float(wollastonite_k(T)),
        "cte":      float(wollastonite_cte(T)),
        "cp":       float(wollastonite_cp(T)),
        "HV_dense": WOLLASTONITE_HARDNESS_DENSE,
    }


def get_air_properties(T):
    """
    Return all air (pore phase) properties at temperature T (°C).

    Parameters
    ----------
    T : float or array-like
        Temperature in °C.

    Returns
    -------
    dict with keys: E, nu, rho, k, cte, cp
    """
    return {
        "E":   float(air_E(T)),
        "nu":  float(air_nu(T)),
        "rho": AIR_DENSITY,
        "k":   float(air_k(T)),
        "cte": float(air_cte(T)),
        "cp":  float(air_cp(T)),
    }


def get_material_tables():
    """
    Return the full temperature-dependent property tables for both phases
    as a dict suitable for MAPDL material definition loops.

    Returns
    -------
    dict with keys 'wollastonite' and 'air', each containing numpy arrays
    keyed by property name, plus the shared 'temperatures' array.
    """
    return {
        "temperatures": TEMP_POINTS,
        "wollastonite": {
            "E":   WOLLASTONITE_YOUNGS_MODULUS,
            "nu":  WOLLASTONITE_POISSONS_RATIO,
            "rho": WOLLASTONITE_DENSITY,
            "k":   WOLLASTONITE_THERMAL_CONDUCTIVITY,
            "cte": WOLLASTONITE_CTE,
            "cp":  WOLLASTONITE_SPECIFIC_HEAT,
        },
        "air": {
            "E":   AIR_YOUNGS_MODULUS,
            "nu":  AIR_POISSONS_RATIO,
            "rho": AIR_DENSITY,
            "k":   AIR_THERMAL_CONDUCTIVITY,
            "cte": AIR_CTE,
            "cp":  AIR_SPECIFIC_HEAT,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 — Material Data Assembly")
    print("=" * 60)
    print(f"\nWollastonite (CaSiO₃) — Dense reference properties")
    print(f"  Density:         {WOLLASTONITE_DENSITY} kg/m³")
    print(f"  Hardness (dense): {WOLLASTONITE_HARDNESS_DENSE} HV")
    print()
    print(f"{'T (°C)':>8}  {'E (GPa)':>10}  {'k (W/mK)':>10}  "
          f"{'CTE (ppm/K)':>12}  {'cp (J/kgK)':>10}")
    print("-" * 60)
    for T in TEMP_POINTS:
        p = get_wollastonite_properties(T)
        print(f"{T:8.0f}  {p['E']/1e9:10.2f}  {p['k']:10.3f}  "
              f"{p['cte']*1e6:12.2f}  {p['cp']:10.1f}")

    print(f"\nAir (pore phase)")
    print(f"{'T (°C)':>8}  {'k (W/mK)':>10}")
    print("-" * 25)
    for T in TEMP_POINTS:
        a = get_air_properties(T)
        print(f"{T:8.0f}  {a['k']:10.4f}")

    print("\n✓ Material data assembly complete.")
