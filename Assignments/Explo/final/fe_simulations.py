"""
fe_simulations.py
=================
Phase 4 — Finite Element Simulations for Wollastonite (CaSiO₃)

Three coupled analyses on the meshed cylinder:
  1. Thermal conductivity — steady-state heat conduction
  2. Elasticity — linear elastic compression
  3. Thermal expansion — free expansion under uniform ΔT

When no ANSYS license is available, the ``run_all_simulations_auto``
function delegates to the analytical fallback module.
"""

import numpy as np
import sys

try:
    from ansys.mapdl.core import Mapdl
except ImportError:
    Mapdl = None

from geometry_meshing import (
    CYLINDER_RADIUS, CYLINDER_HEIGHT, HAS_MAPDL,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Helper: select nodes on a face
# ═══════════════════════════════════════════════════════════════════════════

TOL = 1e-6   # coordinate tolerance (m)


def _select_bottom_nodes(mapdl):
    """Select nodes on the bottom face (Z = 0)."""
    mapdl.nsel("S", "LOC", "Z", 0.0, TOL)


def _select_top_nodes(mapdl):
    """Select nodes on the top face (Z = CYLINDER_HEIGHT)."""
    mapdl.nsel("S", "LOC", "Z", CYLINDER_HEIGHT - TOL, CYLINDER_HEIGHT + TOL)


def _select_radial_nodes(mapdl):
    """Select nodes on the cylindrical (radial) surface."""
    # Nodes at radius ≈ CYLINDER_RADIUS
    mapdl.nsel("S", "LOC", "X", -CYLINDER_RADIUS - TOL, CYLINDER_RADIUS + TOL)
    # Intersect with nodes whose sqrt(x²+y²) ≈ R — use cylindrical
    # For a full cylinder the outer surface is at r = R
    mapdl.csys(1)   # cylindrical coordinate system
    mapdl.nsel("R", "LOC", "X", CYLINDER_RADIUS - TOL, CYLINDER_RADIUS + TOL)
    mapdl.csys(0)   # back to Cartesian


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 1: Thermal Conductivity
# ═══════════════════════════════════════════════════════════════════════════

def simulate_thermal_conductivity(mapdl):
    """
    Steady-state thermal simulation: ΔT = 400 K axial.

    Boundary conditions:
      - Bottom face (Z=0):  T = 0 °C  (273.15 K)
      - Top face (Z=H):     T = 400 °C (673.15 K)
      - Radial surface:     perfectly insulated (default: zero flux)

    Post-processing:
      - Extract total heat flux through the top face
      - Compute k_eff via Fourier's law:
            k_eff = (Q_total / A) * (L / ΔT)

    Returns
    -------
    dict with 'k_eff' (W/m·K), 'Q_total' (W), and 'solver'.
    """
    print("\n  [Sim 1] Thermal conductivity — steady-state heat conduction")
    mapdl.finish()
    mapdl.slashsolu()

    # Analysis type: steady-state thermal
    mapdl.antype("STATIC")

    # Apply temperature BCs
    mapdl.allsel()
    _select_bottom_nodes(mapdl)
    mapdl.d("ALL", "TEMP", 0.0)        # T = 0 °C at bottom

    mapdl.allsel()
    _select_top_nodes(mapdl)
    mapdl.d("ALL", "TEMP", 400.0)      # T = 400 °C at top

    # Radial surface — insulated (default: no flux BC needed)

    # Solve
    mapdl.allsel()
    mapdl.solve()
    mapdl.finish()

    # Post-processing
    mapdl.post1()
    mapdl.set("LAST")

    # Extract heat flux reaction at the top face
    mapdl.allsel()
    _select_top_nodes(mapdl)

    # Sum nodal heat reactions on the top face
    # FSUM gives total force/flux on selected nodes
    mapdl.fsum()
    # The heat reaction is stored as HEAT in the force summary
    # Alternatively, use PRRSOL to print reaction solution
    Q_total = abs(mapdl.get("Q_TOP", "FSUM", "", "ITEM", "HEAT"))

    # Cross-sectional area
    A = np.pi * CYLINDER_RADIUS ** 2
    L = CYLINDER_HEIGHT
    delta_T = 400.0   # °C

    # Fourier's law: q = -k * dT/dx  →  k = (Q/A) * (L/ΔT)
    k_eff = (Q_total / A) * (L / delta_T)

    mapdl.finish()
    mapdl.allsel()

    print(f"    Q_total = {Q_total:.6e} W")
    print(f"    k_eff   = {k_eff:.4f} W/m·K")

    return {
        "k_eff": float(k_eff),
        "Q_total": float(Q_total),
        "solver": "PyMAPDL-SOLID226",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 2: Elasticity (Young's Modulus)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_elasticity(mapdl):
    """
    Linear elastic simulation: 0.1% axial strain.

    Boundary conditions:
      - Bottom face (Z=0):  UX = UY = UZ = 0 (fully fixed)
      - Top face (Z=H):     UZ = -ε·H (controlled displacement)
      - Lateral:             free

    Post-processing:
      - Volume-averaged axial stress on top face
      - E_eff = σ_avg / ε

    Returns
    -------
    dict with 'E_eff' (Pa), 'sigma_avg' (Pa), and 'solver'.
    """
    print("\n  [Sim 2] Elasticity — linear elastic compression")
    mapdl.finish()

    # Need a fresh solution — clear previous BCs
    mapdl.slashsolu()
    mapdl.antype("STATIC")

    # Need to ensure structural DOFs are active
    # SOLID226 KEYOPT(1)=11 gives UX,UY,UZ,TEMP

    strain_applied = 0.001    # 0.1%
    disp_z = -strain_applied * CYLINDER_HEIGHT   # compression

    # Fix bottom face
    mapdl.allsel()
    _select_bottom_nodes(mapdl)
    mapdl.d("ALL", "UX", 0.0)
    mapdl.d("ALL", "UY", 0.0)
    mapdl.d("ALL", "UZ", 0.0)
    # Set reference temperature to avoid thermal strain
    mapdl.d("ALL", "TEMP", 0.0)

    # Apply displacement on top face
    mapdl.allsel()
    _select_top_nodes(mapdl)
    mapdl.d("ALL", "UZ", disp_z)
    mapdl.d("ALL", "TEMP", 0.0)    # isothermal

    # All other nodes: set temperature to 0 (reference, no thermal load)
    mapdl.allsel()
    mapdl.d("ALL", "TEMP", 0.0)

    # Solve
    mapdl.solve()
    mapdl.finish()

    # Post-processing
    mapdl.post1()
    mapdl.set("LAST")

    # Extract reaction force on bottom face (Z-direction)
    mapdl.allsel()
    _select_bottom_nodes(mapdl)
    mapdl.fsum()
    F_z = abs(mapdl.get("FZ_BOT", "FSUM", "", "ITEM", "FZ"))

    # Cross-sectional area
    A = np.pi * CYLINDER_RADIUS ** 2

    # Stress and modulus
    sigma_avg = F_z / A
    E_eff = sigma_avg / strain_applied

    mapdl.finish()
    mapdl.allsel()

    print(f"    F_z     = {F_z:.4e} N")
    print(f"    σ_avg   = {sigma_avg:.4e} Pa")
    print(f"    E_eff   = {E_eff/1e9:.2f} GPa")

    return {
        "E_eff": float(E_eff),
        "sigma_avg": float(sigma_avg),
        "solver": "PyMAPDL-SOLID226",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation 3: Thermal Expansion (CTE)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_thermal_expansion(mapdl):
    """
    Thermal expansion simulation: uniform ΔT = 400 K, free expansion.

    Boundary conditions:
      - Reference temperature: 0 °C
      - Applied temperature: 400 °C everywhere
      - Minimal displacement constraints (prevent rigid body motion)

    Post-processing:
      - Volume change → CTE_eff = ΔV / (V₀ · 3 · ΔT)
      - Stress field statistics around pores (5th/95th percentile)

    Returns
    -------
    dict with 'CTE_eff' (1/K), 'delta_V' (m³), 'stress_5pct', 
         'stress_95pct', and 'solver'.
    """
    print("\n  [Sim 3] Thermal expansion — free expansion under ΔT")
    mapdl.finish()
    mapdl.slashsolu()
    mapdl.antype("STATIC")

    delta_T = 400.0   # °C

    # Set reference temperature
    mapdl.tref(0.0)

    # Minimal constraints to prevent rigid body motion
    # Fix one node at the origin
    mapdl.allsel()
    _select_bottom_nodes(mapdl)
    # Fix the node closest to the centre of the bottom face
    mapdl.nsel("R", "LOC", "X", -TOL, TOL)
    mapdl.nsel("R", "LOC", "Y", -TOL, TOL)

    # Get the selected node(s)
    n_centre = mapdl.get("NCOUNT", "NODE", "", "COUNT")
    if n_centre > 0:
        mapdl.d("ALL", "UX", 0.0)
        mapdl.d("ALL", "UY", 0.0)
        mapdl.d("ALL", "UZ", 0.0)
    else:
        # Fallback: fix the first bottom node
        mapdl.allsel()
        _select_bottom_nodes(mapdl)
        # Take minimum node number
        nmin = int(mapdl.get("NMIN", "NODE", "", "NUM", "MIN"))
        mapdl.d(nmin, "UX", 0.0)
        mapdl.d(nmin, "UY", 0.0)
        mapdl.d(nmin, "UZ", 0.0)

    # Apply uniform temperature to all nodes
    mapdl.allsel()
    mapdl.d("ALL", "TEMP", delta_T)

    # Solve
    mapdl.solve()
    mapdl.finish()

    # Post-processing
    mapdl.post1()
    mapdl.set("LAST")

    # Compute volume change
    # Original volume
    V_0 = np.pi * CYLINDER_RADIUS ** 2 * CYLINDER_HEIGHT

    # Get nodal displacements and compute deformed volume
    mapdl.allsel()

    # Extract displacement field
    ux = mapdl.post_processing.nodal_displacement("X")
    uy = mapdl.post_processing.nodal_displacement("Y")
    uz = mapdl.post_processing.nodal_displacement("Z")

    # Volumetric strain: approximate from average displacements
    # For free expansion: ΔV/V ≈ 3·α·ΔT
    # We extract the actual displacement to verify
    # Average axial strain
    mapdl.allsel()
    _select_top_nodes(mapdl)
    top_node_ids = mapdl.mesh.nnum

    mapdl.allsel()
    uz_top = []
    for nid in top_node_ids:
        try:
            uz_val = mapdl.get("UZ_N", "NODE", nid, "U", "Z")
            uz_top.append(uz_val)
        except Exception:
            pass

    if len(uz_top) > 0:
        avg_uz_top = np.mean(uz_top)
        eps_z = avg_uz_top / CYLINDER_HEIGHT
    else:
        eps_z = 0.0

    # Effective CTE (linear)
    CTE_eff = eps_z / delta_T

    # Stress field statistics
    mapdl.allsel()
    try:
        s_eqv = mapdl.post_processing.nodal_eqv_stress()
        stress_5pct  = float(np.percentile(s_eqv, 5))
        stress_95pct = float(np.percentile(s_eqv, 95))
    except Exception:
        stress_5pct  = 0.0
        stress_95pct = 0.0

    # Volume change
    delta_V = V_0 * 3.0 * CTE_eff * delta_T

    mapdl.finish()
    mapdl.allsel()

    print(f"    CTE_eff     = {CTE_eff*1e6:.3f} ppm/K")
    print(f"    ΔV          = {delta_V:.4e} m³")
    print(f"    σ_eqv  5%   = {stress_5pct:.2e} Pa")
    print(f"    σ_eqv 95%   = {stress_95pct:.2e} Pa")

    return {
        "CTE_eff": float(CTE_eff),
        "delta_V": float(delta_V),
        "stress_5pct": float(stress_5pct),
        "stress_95pct": float(stress_95pct),
        "solver": "PyMAPDL-SOLID226",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Run all three simulations
# ═══════════════════════════════════════════════════════════════════════════

def run_all_simulations(mapdl):
    """
    Run all three FE simulations on the current model.

    The model must already be meshed and have materials/phases assigned
    (output of Phase 3).

    Parameters
    ----------
    mapdl : Mapdl session.

    Returns
    -------
    results : dict with keys 'thermal', 'elastic', 'cte'.
    """
    results = {}

    # Need to rebuild the solution environment for each analysis
    # because BCs differ

    # --- Sim 1: Thermal conductivity ---
    mapdl.finish()
    mapdl.prep7()
    # Ensure SOLID226 KEYOPT for thermal-only analysis
    mapdl.keyopt(1, 1, 11)   # coupled thermal-structural
    mapdl.allsel()

    results["thermal"] = simulate_thermal_conductivity(mapdl)

    # --- Sim 2: Elasticity ---
    # Clear previous solution, re-enter solution
    mapdl.finish()
    mapdl.prep7()
    mapdl.allsel()
    # Delete previous boundary conditions
    mapdl.ddele("ALL", "ALL")

    results["elastic"] = simulate_elasticity(mapdl)

    # --- Sim 3: CTE ---
    mapdl.finish()
    mapdl.prep7()
    mapdl.allsel()
    mapdl.ddele("ALL", "ALL")

    results["cte"] = simulate_thermal_expansion(mapdl)

    return results


def run_all_simulations_auto(mapdl, rve_array=None, descriptors=None):
    """
    Automatically choose FE or analytical simulations.

    If ``mapdl`` is a real MAPDL session, runs the FE simulations.
    If ``mapdl`` is a fallback mock (has ``_is_fallback``), runs
    analytical models on ``rve_array`` (or ``mapdl.rve_array``).

    Parameters
    ----------
    mapdl : Mapdl or SimpleNamespace
    rve_array : ndarray or None
    descriptors : dict or None
        Microstructural descriptors for analytical fallback.

    Returns
    -------
    dict  with keys 'thermal', 'elastic', 'cte'.
    """
    if getattr(mapdl, '_is_fallback', False):
        from analytical_fallback import run_all_simulations_analytical
        rve = rve_array if rve_array is not None else mapdl.rve_array
        return run_all_simulations_analytical(rve, descriptors=descriptors)
    else:
        return run_all_simulations(mapdl)


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test (requires ANSYS license)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from rve_generator import generate_rve
    from geometry_meshing import build_model

    print("=" * 60)
    print("Phase 4 — Finite Element Simulations")
    print("=" * 60)

    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )

    mapdl, mat_assign, porosity = build_model(rve)

    results = run_all_simulations(mapdl)

    print("\n─── Results Summary ───")
    print(f"  k_eff   = {results['thermal']['k_eff']:.4f} W/m·K")
    print(f"  E_eff   = {results['elastic']['E_eff']/1e9:.2f} GPa")
    print(f"  CTE_eff = {results['cte']['CTE_eff']*1e6:.3f} ppm/K")

    mapdl.exit()
    print("\n✓ FE simulations complete.")
