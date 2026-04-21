"""
ml_surrogate.py
===============
Phase 6 — Database Construction & Machine Learning Surrogate

Orchestrates the full Phases 2–5 loop to build a structure–property
database, then trains gradient boosting surrogate models and enables
reverse design queries.

Automatically uses analytical fallback when no ANSYS license is available.
"""

import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

from rve_generator import generate_rve, compute_descriptors
from geometry_meshing import build_model, HAS_MAPDL
from fe_simulations import run_all_simulations_auto
from property_extraction import build_record, save_records, load_records

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
#  Database construction
# ═══════════════════════════════════════════════════════════════════════════

def build_database(n_samples=200,
                   porosity_range=(0.005, 0.035),
                   seed_range=(20, 150),
                   voxel_res=64,
                   element_size=0.5e-3,
                   save_path=None,
                   mapdl=None):
    """
    Build the structure–property database by looping Phases 2–5.

    Parameters
    ----------
    n_samples : int
        Number of RVEs to generate and simulate.
    porosity_range : tuple of float
        (min, max) porosity fraction.
    seed_range : tuple of int
        (min, max) number of Voronoi seeds.
    voxel_res : int
        Voxel resolution per axis.
    element_size : float
        FE element size (m).
    save_path : str or None
        Path to save the database JSON. Default: OUTPUT_DIR/simulation_database.json.
    mapdl : Mapdl or None
        Existing MAPDL session. If None, a new session is launched.

    Returns
    -------
    records : list of dict
        Complete structure–property database.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "simulation_database.json")

    use_fallback = not HAS_MAPDL

    # Launch MAPDL if available and not provided
    if mapdl is None and not use_fallback:
        from geometry_meshing import start_mapdl
        try:
            mapdl = start_mapdl()
        except RuntimeError:
            print("  ⚠ MAPDL launch failed — switching to analytical fallback.")
            use_fallback = True

    rng = np.random.default_rng(seed=42)
    records = []
    failed = 0

    solver_label = "Analytical" if use_fallback else "PyMAPDL"
    print(f"\n{'='*60}")
    print(f"Phase 6 — Building database: {n_samples} RVEs  [{solver_label}]")
    print(f"{'='*60}")

    for i in range(n_samples):
        t0 = time.time()

        # Random microstructure parameters
        target_porosity = rng.uniform(*porosity_range)
        n_seeds = rng.integers(seed_range[0], seed_range[1] + 1)

        print(f"\n[{i+1}/{n_samples}] seeds={n_seeds}, "
              f"porosity_target={target_porosity*100:.2f}%")

        try:
            # Phase 2: Generate RVE
            rve, grain_ids, meta = generate_rve(
                n_seeds=int(n_seeds),
                voxel_res=voxel_res,
                target_porosity=target_porosity,
                seed=int(rng.integers(0, 2**31)),
            )
            descriptors = compute_descriptors(rve, meta)

            if use_fallback:
                # Analytical path — no MAPDL needed
                from analytical_fallback import (
                    run_all_simulations_analytical,
                    build_model_analytical,
                )
                _, mat_assign, porosity_mapped = build_model_analytical(rve)
                sim_results = run_all_simulations_analytical(
                    rve, descriptors=descriptors
                )
            else:
                # FE path
                mapdl.clear()
                mapdl.prep7()
                from geometry_meshing import (
                    define_materials, create_cylinder_mesh, map_phases,
                )
                define_materials(mapdl)
                create_cylinder_mesh(mapdl, element_size)
                mat_assign, porosity_mapped = map_phases(mapdl, rve)
                sim_results = run_all_simulations_auto(
                    mapdl, rve_array=rve, descriptors=descriptors
                )

            # Phase 5: Extract properties and build record
            record = build_record(descriptors, sim_results, rve)
            record["sample_id"] = i
            records.append(record)

            dt = time.time() - t0
            print(f"    ✓ Done in {dt:.1f}s — k={record['k_eff_WpmK']:.3f}, "
                  f"E={record['E_eff_GPa']:.1f}, HV={record['HV']:.0f}")

        except Exception as e:
            failed += 1
            print(f"    ✗ FAILED: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Database complete: {len(records)} successful, {failed} failed")
    print(f"{'='*60}")

    # Save
    save_records(records, save_path)

    return records


# ═══════════════════════════════════════════════════════════════════════════
#  Machine learning surrogates
# ═══════════════════════════════════════════════════════════════════════════

# Features and targets
FEATURE_COLS = [
    "porosity", "mean_chord_x", "mean_chord_y", "mean_chord_z",
    "var_chord_x", "var_chord_y", "var_chord_z", "mean_chord_avg",
    "n_seeds",
]

TARGET_COLS = {
    "k_eff_WpmK":    "Thermal Conductivity (W/m·K)",
    "E_eff_GPa":     "Young's Modulus (GPa)",
    "CTE_eff_ppmK":  "CTE (ppm/K)",
    "density_kgpm3": "Density (kg/m³)",
    "HV":            "Vickers Hardness (HV)",
    "porosity_pct":  "Porosity (%)",
}


def _records_to_arrays(records):
    """Convert list of record dicts to feature matrix X and target dict Y."""
    X = np.array([[r[f] for f in FEATURE_COLS] for r in records])
    Y = {t: np.array([r[t] for r in records]) for t in TARGET_COLS}
    return X, Y


def train_surrogates(records,
                     test_size=0.25,
                     random_state=42,
                     n_estimators=200,
                     max_depth=5):
    """
    Train gradient boosting regressors for each target property.

    Parameters
    ----------
    records : list of dict
        Database from ``build_database()``.
    test_size : float
        Fraction for test split.
    random_state : int
        Random seed.
    n_estimators : int
        Number of boosting trees.
    max_depth : int
        Max tree depth.

    Returns
    -------
    models : dict
        Trained GBR model for each target.
    metrics : dict
        R² and MAE for each target.
    X_test, Y_test : ndarray, dict
        Test set for plotting.
    """
    X, Y = _records_to_arrays(records)
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, np.arange(len(records)), test_size=test_size,
        random_state=random_state,
    )

    models = {}
    metrics = {}
    Y_test = {}

    print(f"\nTraining ML surrogates ({len(X_train)} train / "
          f"{len(X_test)} test)")
    print("-" * 60)

    for target, label in TARGET_COLS.items():
        y = Y[target]
        y_train = y[idx_train]
        y_test  = y[idx_test]

        gbr = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            learning_rate=0.1,
            subsample=0.8,
        )
        gbr.fit(X_train, y_train)
        y_pred = gbr.predict(X_test)

        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        models[target]  = gbr
        metrics[target]  = {"R2": r2, "MAE": mae, "label": label}
        Y_test[target]  = (y_test, y_pred)

        print(f"  {label:35s}  R²={r2:.4f}  MAE={mae:.4f}")

    return models, metrics, X_test, Y_test


# ═══════════════════════════════════════════════════════════════════════════
#  Reverse design
# ═══════════════════════════════════════════════════════════════════════════

def reverse_design(models, target_property, target_value,
                   porosity_range=(0.005, 0.035),
                   seed_range=(20, 150),
                   n_grid=50):
    """
    Grid-search the microstructure parameter space to find
    combinations that produce a desired target property value.

    Parameters
    ----------
    models : dict
        Trained ML models from ``train_surrogates()``.
    target_property : str
        Key in TARGET_COLS, e.g. 'k_eff_WpmK'.
    target_value : float
        Desired property value.
    porosity_range : tuple
        (min, max) porosity to search.
    seed_range : tuple
        (min, max) n_seeds to search.
    n_grid : int
        Grid points per dimension.

    Returns
    -------
    best_params : dict
        Best microstructural parameters.
    best_error : float
        Absolute error from target.
    grid_results : ndarray
        Full grid of (porosity, n_seeds, predicted_value).
    """
    if target_property not in models:
        raise ValueError(f"No model for '{target_property}'. "
                         f"Available: {list(models.keys())}")

    model = models[target_property]

    porosities = np.linspace(*porosity_range, n_grid)
    seeds_arr  = np.linspace(*[float(s) for s in seed_range], n_grid)

    # Build a synthetic feature matrix
    # For chord lengths, use empirical approximation:
    #   mean_chord ≈ voxel_res / n_seeds^(1/3)
    voxel_res = 64   # assumed

    grid_results = []
    best_error = np.inf
    best_params = None

    for p in porosities:
        for ns in seeds_arr:
            mc = voxel_res / (ns ** (1.0/3.0))
            mc_var = mc * 0.1   # approximate

            features = np.array([[
                p,        # porosity
                mc, mc, mc,    # mean_chord_{x,y,z}
                mc_var, mc_var, mc_var,   # var_chord_{x,y,z}
                mc,       # mean_chord_avg
                ns,       # n_seeds
            ]])

            pred = model.predict(features)[0]
            err = abs(pred - target_value)

            grid_results.append((p, ns, pred))

            if err < best_error:
                best_error = err
                best_params = {
                    "porosity": p,
                    "n_seeds": int(round(ns)),
                    "predicted_value": pred,
                }

    grid_results = np.array(grid_results)

    return best_params, best_error, grid_results


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def plot_parity(Y_test, metrics, save_path=None):
    """Predicted vs actual scatter plots for all targets."""
    n = len(Y_test)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows))
    axes = np.atleast_2d(axes)

    for idx, (target, (y_true, y_pred)) in enumerate(Y_test.items()):
        ax = axes[idx // cols, idx % cols]
        ax.scatter(y_true, y_pred, s=20, alpha=0.6, edgecolors="k", lw=0.3)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                "r--", lw=1.5, label="ideal")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        r2 = metrics[target]["R2"]
        ax.set_title(f"{metrics[target]['label']}\nR²={r2:.3f}", fontsize=10)
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "ml_parity_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved parity plots: {save_path}")
    plt.close()


def plot_feature_importance(models, metrics, save_path=None):
    """Feature importance bar charts for all models."""
    n = len(models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.atleast_2d(axes)

    for idx, (target, model) in enumerate(models.items()):
        ax = axes[idx // cols, idx % cols]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        ax.barh(
            [FEATURE_COLS[i] for i in sorted_idx],
            importances[sorted_idx],
            color="steelblue",
        )
        ax.set_title(metrics[target]["label"], fontsize=10)
        ax.set_xlabel("Importance")

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved feature importance: {save_path}")
    plt.close()


def plot_property_vs_porosity(records, save_path=None):
    """Plot all four target properties vs porosity."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    porosities = [r["porosity_pct"] for r in records]

    targets = [
        ("k_eff_WpmK",    "Thermal Conductivity (W/m·K)"),
        ("density_kgpm3", "Density (kg/m³)"),
        ("HV",            "Vickers Hardness (HV)"),
        ("E_eff_GPa",     "Young's Modulus (GPa)"),
    ]

    for ax, (key, label) in zip(axes.flat, targets):
        vals = [r[key] for r in records]
        ax.scatter(porosities, vals, s=15, alpha=0.6, edgecolors="k", lw=0.3)
        ax.set_xlabel("Porosity (%)")
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "property_vs_porosity.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved property vs porosity: {save_path}")
    plt.close()


def plot_reverse_design(grid_results, target_property, target_value,
                        best_params, save_path=None):
    """Heatmap of predicted property over porosity × n_seeds grid."""
    porosities = grid_results[:, 0]
    seeds      = grid_results[:, 1]
    values     = grid_results[:, 2]

    n_grid = int(np.sqrt(len(grid_results)))
    P = porosities.reshape(n_grid, n_grid)
    S = seeds.reshape(n_grid, n_grid)
    V = values.reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(P * 100, S, V, levels=30, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=TARGET_COLS.get(target_property, target_property))

    # Mark target contour
    ax.contour(P * 100, S, V, levels=[target_value], colors="red", linewidths=2)

    # Mark best point
    ax.plot(best_params["porosity"] * 100, best_params["n_seeds"],
            "r*", markersize=15, label=f"Best: {best_params['predicted_value']:.3f}")

    ax.set_xlabel("Porosity (%)")
    ax.set_ylabel("Number of Seeds (grain count)")
    ax.set_title(f"Reverse Design: Target {TARGET_COLS.get(target_property, target_property)} = {target_value}")
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "reverse_design_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  → Saved reverse design map: {save_path}")
    plt.close()


def save_models(models, directory=None):
    """Persist trained models with joblib."""
    if directory is None:
        directory = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(directory, exist_ok=True)
    for target, model in models.items():
        path = os.path.join(directory, f"{target}_gbr.joblib")
        joblib.dump(model, path)
    print(f"  → Models saved to: {directory}")


def load_models(directory=None):
    """Load trained models from joblib files."""
    if directory is None:
        directory = os.path.join(OUTPUT_DIR, "models")
    models = {}
    for fname in os.listdir(directory):
        if fname.endswith("_gbr.joblib"):
            target = fname.replace("_gbr.joblib", "")
            models[target] = joblib.load(os.path.join(directory, fname))
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test (requires ANSYS license for database building)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6 — ML Surrogate")
    print("=" * 60)

    db_path = os.path.join(OUTPUT_DIR, "simulation_database.json")

    # Build database (requires ANSYS)
    records = build_database(n_samples=200, save_path=db_path)

    # Train surrogates
    models, metrics, X_test, Y_test = train_surrogates(records)

    # Plots
    plot_parity(Y_test, metrics)
    plot_feature_importance(models, metrics)
    plot_property_vs_porosity(records)

    # Reverse design demo
    target_k = 3.0   # W/m·K
    best, err, grid = reverse_design(models, "k_eff_WpmK", target_k)
    print(f"\nReverse design for k_eff = {target_k} W/m·K:")
    print(f"  Best: porosity={best['porosity']*100:.2f}%, "
          f"seeds={best['n_seeds']}, "
          f"predicted={best['predicted_value']:.3f}")
    plot_reverse_design(grid, "k_eff_WpmK", target_k, best)

    # Save models
    save_models(models)

    print("\n✓ ML surrogate complete.")
