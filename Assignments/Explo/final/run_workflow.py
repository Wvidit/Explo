"""
run_workflow.py
===============
Master runner for the Wollastonite (CaSiO3) Simulation Workflow

Orchestrates all six phases:
  Phase 1: Material data assembly
  Phase 2: RVE microstructure generation
  Phase 3: Geometry construction & meshing (PyMAPDL or analytical)
  Phase 4: Finite element simulations  (PyMAPDL or analytical)
  Phase 5: Property extraction
  Phase 6: Database construction & ML surrogate

Automatically falls back to analytical models when ANSYS is unavailable.
"""

import os
import sys
import time
import json
import numpy as np

# --- Ensure local imports resolve ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def banner(text):
    """Print a formatted phase banner."""
    print(f"\n{'='*64}")
    print(f"  {text}")
    print(f"{'='*64}\n")


def run_single_rve_demo():
    """
    Run Phases 1-5 on a single RVE for demonstration / validation.
    Returns the record, MAPDL session (or mock), and the RVE array.
    """
    from material_data import get_wollastonite_properties, get_air_properties, TEMP_POINTS
    from rve_generator import generate_rve, compute_descriptors, visualize_rve
    from geometry_meshing import build_model, HAS_MAPDL
    from fe_simulations import run_all_simulations_auto
    from property_extraction import build_record

    use_fallback = not HAS_MAPDL
    if use_fallback:
        print("\n" + "#" * 64)
        print("  WARNING: ANSYS MAPDL NOT AVAILABLE - USING ANALYTICAL FALLBACK")
        print("#" * 64 + "\n")

    # -- Phase 1 --
    banner("Phase 1 - Material Data Assembly")
    print("Wollastonite (CaSiO3) properties loaded.")
    for T in TEMP_POINTS:
        p = get_wollastonite_properties(T)
        print(f"  T={T:5.0f} C  k={p['k']:.3f} W/mK  "
              f"E={p['E']/1e9:.1f} GPa  CTE={p['cte']*1e6:.2f} ppm/K")

    # -- Phase 2 --
    banner("Phase 2 - RVE Microstructure Generation")
    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )
    descriptors = compute_descriptors(rve, meta)
    print(f"  Porosity:       {meta['actual_porosity']*100:.2f}%")
    print(f"  Mean chord avg: {descriptors['mean_chord_avg']:.2f} voxels")

    # Save visualization
    vis_path = os.path.join(SCRIPT_DIR, "rve_visualization.png")
    visualize_rve(rve, grain_ids, save_path=vis_path)

    # -- Phase 3 --
    phase3_label = "Phase 3 - Geometry Construction & Meshing"
    if use_fallback:
        phase3_label += " [ANALYTICAL]"
    banner(phase3_label)
    mapdl, mat_assign, porosity_mapped = build_model(
        rve, use_fallback=use_fallback
    )
    print(f"  Elements: {mapdl.mesh.n_elem}")
    print(f"  Nodes:    {mapdl.mesh.n_node}")
    print(f"  Mapped porosity: {porosity_mapped*100:.2f}%")

    # -- Phase 4 --
    phase4_label = "Phase 4 - Finite Element Simulations"
    if use_fallback:
        phase4_label += " [ANALYTICAL]"
    banner(phase4_label)
    sim_results = run_all_simulations_auto(mapdl, rve_array=rve, descriptors=descriptors)

    # -- Phase 5 --
    banner("Phase 5 - Property Extraction")
    record = build_record(descriptors, sim_results, rve)

    print("\n  --- Extracted Properties ---")
    print(f"  Thermal conductivity: {record['k_eff_WpmK']:.4f} W/m K")
    print(f"  Young's modulus:      {record['E_eff_GPa']:.2f} GPa")
    print(f"  CTE:                  {record['CTE_eff_ppmK']:.3f} ppm/K")
    print(f"  Density:              {record['density_kgpm3']:.1f} kg/m3")
    print(f"  Porosity:             {record['porosity_pct']:.2f}%")
    print(f"  Vickers hardness:     {record['HV']:.1f} HV")
    print(f"  Solver:               {record['solver']}")

    return record, mapdl, rve


def run_database_and_ml(mapdl=None, n_samples=200):
    """
    Run Phase 6: build database + train ML surrogates + reverse design.
    """
    from ml_surrogate import (
        build_database, train_surrogates, save_models,
        plot_parity, plot_feature_importance,
        plot_property_vs_porosity, plot_reverse_design,
        reverse_design,
    )

    banner("Phase 6 - Database Construction & ML Surrogate")

    db_path = os.path.join(SCRIPT_DIR, "simulation_database.json")

    # Build database
    records = build_database(
        n_samples=n_samples,
        save_path=db_path,
        mapdl=mapdl,
    )

    if len(records) < 10:
        print("WARNING: Too few successful simulations for ML training.")
        return

    # Train surrogates
    models, metrics, X_test, Y_test = train_surrogates(records)

    # Generate all plots
    plot_parity(Y_test, metrics)
    plot_feature_importance(models, metrics)
    plot_property_vs_porosity(records)

    # Reverse design demo: find microstructure for k_eff = 3.0 W/m K
    target_k = 3.0
    print(f"\n  Reverse design: target k_eff = {target_k} W/m K")
    best, err, grid = reverse_design(models, "k_eff_WpmK", target_k, records=records)
    print(f"  -> Best: porosity={best['porosity']*100:.2f}%, "
          f"seeds={best['n_seeds']}, "
          f"predicted={best['predicted_value']:.3f} W/m K")
    plot_reverse_design(grid, "k_eff_WpmK", target_k, best)

    # Save models
    save_models(models)

    # Summary
    print(f"\n{'-'*64}")
    print(f"  ML Surrogate Performance Summary")
    print(f"{'-'*64}")
    for target, m in metrics.items():
        print(f"  {m['label']:35s}  R2={m['R2']:.4f}  MAE={m['MAE']:.4f}")

    return models, metrics


def main():
    """
    Complete workflow: single-RVE demo -> database build -> ML training.
    """
    from geometry_meshing import HAS_MAPDL

    banner("WOLLASTONITE (CaSiO3) SIMULATION WORKFLOW")
    print("  Material: Wollastonite (CaSiO3)")
    print("  Specimen: Cylinder 8 mm x 4 mm H")
    print("  Temperature range: 0 - 400 C")
    print("  Target properties: k_eff, density, porosity, HV")
    solver_str = "ANSYS MAPDL (PyMAPDL)" if HAS_MAPDL else "Analytical Fallback"
    print(f"  Solver: {solver_str}")

    t_start = time.time()

    # Step 1: Single-RVE demonstration (Phases 1-5)
    record, mapdl, rve = run_single_rve_demo()

    # Validate ranges
    print("\n  --- Validation Checks ---")
    checks = [
        ("k_eff", record["k_eff_WpmK"], 1.8, 3.6, "W/m K"),
        ("density", record["density_kgpm3"], 2750, 2910, "kg/m3"),
        ("porosity", record["porosity_pct"], 0.5, 3.5, "%"),
        ("HV", record["HV"], 350, 570, "HV"),
    ]
    for name, val, lo, hi, unit in checks:
        status = "OK" if lo <= val <= hi else "WARN"
        print(f"  [{status}] {name:12s} = {val:.2f} {unit}  "
              f"(expected {lo}-{hi})")

    # Step 2: Database build + ML (Phase 6)
    run_database_and_ml(mapdl=mapdl, n_samples=200)

    # Clean up
    try:
        mapdl.exit()
    except Exception:
        pass

    t_total = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"  WORKFLOW COMPLETE - Total time: {t_total/60:.1f} minutes")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
