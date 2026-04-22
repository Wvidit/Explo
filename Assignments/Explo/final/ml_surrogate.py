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
                   n_grid=50,
                   records=None):
    """
    Grid-search the microstructure parameter space to find
    combinations that produce a desired target property value.

    Uses the training database to fit empirical mappings from
    (porosity, n_seeds) → chord features, ensuring the synthetic
    query points stay within the model's training distribution.

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
    records : list of dict or None
        Training records for fitting empirical feature maps.
        If None, falls back to simple approximation.

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

    # ── Fit empirical feature maps from the database ────────────────────
    # Instead of crude approximations, learn how chord features relate to
    # (porosity, n_seeds) from the actual training data.
    if records is not None and len(records) > 10:
        from sklearn.linear_model import LinearRegression

        db_porosity = np.array([r["porosity"] for r in records])
        db_nseeds   = np.array([r["n_seeds"]  for r in records])

        # Design matrix: [porosity, n_seeds, porosity², n_seeds², p*ns]
        X_design = np.column_stack([
            db_porosity,
            db_nseeds,
            db_porosity ** 2,
            db_nseeds ** 2,
            db_porosity * db_nseeds,
        ])

        # Fit a simple polynomial for each chord feature
        chord_models = {}
        chord_feature_keys = [
            "mean_chord_x", "mean_chord_y", "mean_chord_z",
            "var_chord_x", "var_chord_y", "var_chord_z",
            "mean_chord_avg",
        ]
        for key in chord_feature_keys:
            y = np.array([r[key] for r in records])
            lr = LinearRegression().fit(X_design, y)
            chord_models[key] = lr

        def _predict_features(p, ns):
            x = np.array([[p, ns, p**2, ns**2, p * ns]])
            feats = {k: float(chord_models[k].predict(x)[0])
                     for k in chord_feature_keys}
            return feats
    else:
        # Fallback: simple approximation
        def _predict_features(p, ns):
            mc = 64 / max(ns, 1) ** (1.0/3.0)
            mc_var = mc * 0.15 + 450
            return {
                "mean_chord_x": mc, "mean_chord_y": mc, "mean_chord_z": mc,
                "var_chord_x": mc_var, "var_chord_y": mc_var,
                "var_chord_z": mc_var, "mean_chord_avg": mc,
            }

    # ── Grid search ─────────────────────────────────────────────────────
    grid_results = []
    best_error = np.inf
    best_params = None

    for p in porosities:
        for ns in seeds_arr:
            feats = _predict_features(p, ns)

            features = np.array([[
                p,
                feats["mean_chord_x"], feats["mean_chord_y"],
                feats["mean_chord_z"],
                feats["var_chord_x"], feats["var_chord_y"],
                feats["var_chord_z"],
                feats["mean_chord_avg"],
                ns,
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

def _apply_dark_theme():
    """Apply the shared dark theme used across all plots."""
    BG       = "#0f1117"
    PANEL_BG = "#181c25"
    TEXT     = "#e0e4ec"
    GRID_CLR = "#2a2f3a"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL_BG,
        "axes.edgecolor":   GRID_CLR,
        "axes.labelcolor":  TEXT,
        "text.color":       TEXT,
        "xtick.color":      TEXT,
        "ytick.color":      TEXT,
        "grid.color":       GRID_CLR,
        "grid.alpha":       0.3,
        "font.family":      "sans-serif",
        "font.size":        10,
    })
    return BG, PANEL_BG, TEXT, GRID_CLR


def plot_parity(Y_test, metrics, save_path=None):
    """Dark-themed predicted vs actual scatter plots for all targets."""
    BG, PANEL_BG, TEXT, GRID_CLR = _apply_dark_theme()
    ACCENT  = "#6ee7b7"
    ACCENT2 = "#f472b6"

    # Bright scatter palette for each subplot
    palette = ["#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c"]

    n = len(Y_test)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 5*rows))
    fig.suptitle("ML Surrogate — Parity Plots",
                 fontsize=16, fontweight="bold", color=ACCENT, y=0.98)
    axes = np.atleast_2d(axes)

    for idx, (target, (y_true, y_pred)) in enumerate(Y_test.items()):
        ax = axes[idx // cols, idx % cols]
        color = palette[idx % len(palette)]

        ax.scatter(y_true, y_pred, s=28, alpha=0.7,
                   color=color, edgecolors="none", rasterized=True)

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = (hi - lo) * 0.08
        ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin],
                color=ACCENT, lw=1.5, ls="--", alpha=0.8, label="Ideal")

        ax.set_xlabel("Actual", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        r2 = metrics[target]["R2"]
        mae = metrics[target]["MAE"]
        ax.set_title(f"{metrics[target]['label']}\nR²={r2:.3f}  MAE={mae:.4f}",
                     fontsize=10, fontweight="bold", color=TEXT, pad=8)
        ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_CLR,
                  labelcolor=TEXT)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=8)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "ml_parity_plots.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  → Saved parity plots: {save_path}")
    plt.close()
    plt.rcdefaults()


def plot_feature_importance(models, metrics, save_path=None):
    """Dark-themed feature importance bar charts for all models."""
    BG, PANEL_BG, TEXT, GRID_CLR = _apply_dark_theme()
    ACCENT = "#6ee7b7"

    # Gradient colors for bars
    bar_colors = ["#60a5fa", "#34d399", "#f472b6", "#fbbf24", "#a78bfa", "#fb923c"]

    n = len(models)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 4.5*rows))
    fig.suptitle("Feature Importance — Gradient Boosting",
                 fontsize=16, fontweight="bold", color=ACCENT, y=0.98)
    axes = np.atleast_2d(axes)

    for idx, (target, model) in enumerate(models.items()):
        ax = axes[idx // cols, idx % cols]
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)

        labels = [FEATURE_COLS[i] for i in sorted_idx]
        vals = importances[sorted_idx]
        color = bar_colors[idx % len(bar_colors)]

        bars = ax.barh(labels, vals, color=color, alpha=0.85,
                       edgecolor="none", height=0.7)

        # Add value labels
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{v:.2f}", va="center", fontsize=7, color="#94a3b8")

        ax.set_title(metrics[target]["label"], fontsize=10,
                     fontweight="bold", color=TEXT, pad=8)
        ax.set_xlabel("Importance", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="x", alpha=0.15)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  → Saved feature importance: {save_path}")
    plt.close()
    plt.rcdefaults()


def plot_property_vs_porosity(records, save_path=None):
    """Dark-themed property vs porosity scatter plots."""
    BG, PANEL_BG, TEXT, GRID_CLR = _apply_dark_theme()
    ACCENT = "#6ee7b7"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Structure–Property Relationships",
                 fontsize=16, fontweight="bold", color=ACCENT, y=0.98)

    porosities = np.array([r["porosity_pct"] for r in records])
    n_seeds = np.array([r["n_seeds"] for r in records])

    targets = [
        ("k_eff_WpmK",    "Thermal Conductivity (W/m·K)", "#60a5fa"),
        ("density_kgpm3", "Density (kg/m³)",               "#34d399"),
        ("HV",            "Vickers Hardness (HV)",         "#f472b6"),
        ("E_eff_GPa",     "Young's Modulus (GPa)",         "#fbbf24"),
    ]

    for ax, (key, label, color) in zip(axes.flat, targets):
        vals = np.array([r[key] for r in records])

        # Colour by n_seeds for multi-dimensional insight
        sc = ax.scatter(porosities, vals, c=n_seeds, cmap="magma",
                        s=25, alpha=0.75, edgecolors="none", rasterized=True)

        ax.set_xlabel("Porosity (%)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold", color=TEXT, pad=8)
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=8)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("n_seeds", fontsize=8, color="#94a3b8")
        cbar.ax.tick_params(labelsize=7, colors="#94a3b8")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "property_vs_porosity.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  → Saved property vs porosity: {save_path}")
    plt.close()
    plt.rcdefaults()


def plot_reverse_design(grid_results, target_property, target_value,
                        best_params, save_path=None):
    """Publication-quality reverse design heatmap with dark theme."""
    porosities = grid_results[:, 0]
    seeds      = grid_results[:, 1]
    values     = grid_results[:, 2]

    n_grid = int(np.sqrt(len(grid_results)))
    P = porosities.reshape(n_grid, n_grid)
    S = seeds.reshape(n_grid, n_grid)
    V = values.reshape(n_grid, n_grid)

    # ── Dark theme ──────────────────────────────────────────────────────
    BG       = "#0f1117"
    PANEL_BG = "#181c25"
    TEXT     = "#e0e4ec"
    ACCENT   = "#6ee7b7"
    GRID_CLR = "#2a2f3a"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL_BG,
        "axes.edgecolor":   GRID_CLR,
        "axes.labelcolor":  TEXT,
        "text.color":       TEXT,
        "xtick.color":      TEXT,
        "ytick.color":      TEXT,
        "font.family":      "sans-serif",
        "font.size":        11,
    })

    fig, ax = plt.subplots(figsize=(10, 7))

    prop_label = TARGET_COLS.get(target_property, target_property)

    # Smooth filled contour
    cf = ax.contourf(
        P * 100, S, V,
        levels=40, cmap="magma",
    )
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(prop_label, fontsize=12, color=TEXT)
    cbar.ax.tick_params(colors=TEXT, labelsize=9)

    # Target contour line
    target_cs = ax.contour(
        P * 100, S, V,
        levels=[target_value], colors=[ACCENT], linewidths=2.5,
        linestyles="--",
    )
    ax.clabel(target_cs, fmt=f"Target = {target_value:.2f}",
              fontsize=9, colors=[ACCENT])

    # Best point
    ax.plot(
        best_params["porosity"] * 100, best_params["n_seeds"],
        marker="*", markersize=20, color="#f472b6",
        markeredgecolor="white", markeredgewidth=1.2,
        zorder=10,
    )
    # Annotation box for best point
    pred_val = best_params["predicted_value"]
    ax.annotate(
        f"Best: {pred_val:.3f}\n"
        f"φ = {best_params['porosity']*100:.2f}%\n"
        f"seeds = {best_params['n_seeds']}",
        xy=(best_params["porosity"] * 100, best_params["n_seeds"]),
        xytext=(20, 25), textcoords="offset points",
        fontsize=10, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e2330",
                  edgecolor=ACCENT, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
        zorder=11,
    )

    ax.set_xlabel("Porosity (%)", fontsize=13, labelpad=8)
    ax.set_ylabel("Number of Seeds (grain count)", fontsize=13, labelpad=8)
    ax.set_title(
        f"Reverse Design Map — Target {prop_label} = {target_value}",
        fontsize=15, fontweight="bold", color=ACCENT, pad=14,
    )
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "reverse_design_map.png")
    plt.savefig(save_path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"  → Saved reverse design map: {save_path}")
    plt.close()
    plt.rcdefaults()


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
