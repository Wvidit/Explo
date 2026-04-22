# Wollastonite (CaSiO₃) — Top-Down Ceramic Design Workflow

## Overview

A complete computational materials design pipeline for Wollastonite ceramic,
following the Pirkelmann et al. methodology. The workflow generates porous
microstructures via Voronoi tessellation, simulates them with coupled
thermal-structural FEA, extracts four target properties, and trains ML
surrogates for reverse design.

**Material**: Wollastonite (CaSiO₃) — single-phase calcium silicate ceramic  
**Specimen**: Cylinder 8 mm Ø × 4 mm H  
**Temperature range**: 0 – 400 °C  
**Target properties**: Thermal conductivity, Bulk density, Porosity, Vickers hardness

> ℹ️ **Dual-Mode Operation** — If an ANSYS/PyMAPDL license is available, the
> workflow runs full FE simulations. If not, it automatically falls back to
> microstructure-aware analytical homogenisation models with grain-boundary,
> tortuosity, and pore-morphology corrections, producing realistic scatter
> in structure–property relationships.

---

## Architecture

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `material_data.py` | Temperature-dependent Wollastonite & air properties |
| 2 | `rve_generator.py` | 3D Voronoi RVE with triple-junction porosity |
| 3 | `geometry_meshing.py` | Cylindrical FE mesh (SOLID226) + phase mapping / analytical mock |
| 4 | `fe_simulations.py` | Thermal, elastic, and CTE simulations (FE or analytical) |
| 5 | `property_extraction.py` | k_eff, density, porosity, hardness extraction |
| 6 | `ml_surrogate.py` | Database loop, gradient boosting surrogates, reverse design |
| — | `analytical_fallback.py` | **Microstructure-aware analytical fallback (no license needed)** |
| — | `run_workflow.py` | Master runner — auto-detects solver availability |

---

## Installation

### 1. Install System Dependencies & Python Packages

Use the provided setup script (works with your existing `pyenv` global Python):

```bash
chmod +x setup_ansys_mapdl.sh
./setup_ansys_mapdl.sh              # installs OS libs + PyAnsys packages
# or skip OS libs if you already have them:
./setup_ansys_mapdl.sh --python-only
```

Or install manually:

```bash
pip install -r requirements.txt
```

**Python dependencies**: `numpy`, `scipy`, `matplotlib`, `scikit-learn`, `joblib`, `ansys-mapdl-core`

### 2. ANSYS Solver (Optional)

The full MAPDL solver requires a licensed installer from
[customer.ansys.com](https://customer.ansys.com) or a university license.
**Students without a license can skip this step** — the analytical fallback
will activate automatically.

---

## Running

```bash
# Full workflow (auto-detects ANSYS availability)
python run_workflow.py

# Analytical fallback only (no license, runs instantly)
python analytical_fallback.py

# Individual phases
python material_data.py        # Phase 1 self-test (no license)
python rve_generator.py        # Phase 2 demo (no license)
python geometry_meshing.py     # Phase 3 (ANSYS or fallback)
python fe_simulations.py       # Phase 4 (ANSYS or fallback)
python property_extraction.py  # Phase 5 self-test (no license)
python ml_surrogate.py         # Phase 6 (ANSYS or fallback)
```

---

## Solver Modes

| Mode | Requirement | Phases 3–4 Method | Accuracy |
|------|-------------|-------------------|----------|
| **PyMAPDL (FE)** | ANSYS license | SOLID226 FEM solver | High (numerical) |
| **Analytical Fallback** | None | Homogenisation models | Moderate (validated) |

### Fallback Models Used

Each fallback model combines a classical homogenisation formula with **microstructural corrections** derived from the RVE descriptors (chord lengths, variance, grain count) plus controlled stochastic noise (~1.5–2% CV) for realistic scatter:

| Simulation | Base Model | Microstructural Corrections |
|------------|-----------|----------------------------|
| Thermal conductivity | **Maxwell-Eucken** | Grain-boundary scattering, tortuosity (chord anisotropy), pore irregularity (chord variance), grain count effect, T_ref ±30°C variation |
| Young's modulus | **Mori-Tanaka** | Pore aspect-ratio (chord CV), grain-size bidirectional correction, chord anisotropy, grain count effect |
| CTE | **Turner / Voigt ROM** | Grain-boundary constraint, grain-size effect, chord anisotropy, pore irregularity |
| Density | **Rule of Mixtures** | Grain-boundary density deficit, measurement noise |
| Hardness | **Rice-Duckworth** | Hall-Petch grain-size correction, pore irregularity, grain count effect |

---

## Output Files

| File | Description |
|------|-------------|
| `rve_visualization.png` | 2×2 dark-themed RVE panel: 3D grain boundaries, pore network, 2D cross-section, and statistics |
| `property_vs_porosity.png` | All 4 properties vs porosity scatter plots |
| `ml_parity_plots.png` | Predicted vs actual for each ML model |
| `feature_importance.png` | Gradient boosting feature importance |
| `reverse_design_map.png` | Target property → microstructure parameter map |
| `simulation_database.json` | Full structure–property database (200+ records) |
| `models/` | Serialised gradient boosting models (joblib) |

---

## Key Physical Parameters

| Property | Dense Wollastonite | Unit |
|----------|--------------------|------|
| Young's modulus | 96 GPa (0 °C) → 93 GPa (400 °C) | GPa |
| Poisson's ratio | 0.27 | — |
| Density | 2900 | kg/m³ |
| Thermal conductivity | 3.50 (0 °C) → 2.60 (400 °C) | W/m·K |
| CTE | 6.5 → 7.5 | ppm/K |
| Specific heat | 840 → 940 | J/kg·K |
| Vickers hardness | 570 | HV |

---

## FE Simulation Details (When ANSYS is Available)

- **Element type**: SOLID226 (coupled thermal-structural, KEYOPT(1)=11)
- **Mesh size**: ~0.5 mm element edge length
- **Thermal sim**: Steady-state, ΔT = 400 K axial, insulated radial surfaces
- **Elastic sim**: 0.1% axial strain, fully fixed bottom face
- **CTE sim**: Uniform ΔT = 400 K, free expansion, minimal rigid-body constraints

---

## ML Surrogate

- **Algorithm**: Gradient Boosting Regressor (scikit-learn)
- **Features**: Porosity, mean chord lengths (x/y/z), chord length variance (x/y/z), mean chord avg, grain count
- **Targets**: k_eff, E_eff, CTE, density, porosity, HV
- **Split**: 75% train / 25% test
- **Typical R²**: 0.79 (k), 0.31 (E), 0.87 (CTE), 0.95 (density), 0.85 (HV)
- **Reverse design**: Database-fitted grid search over porosity × grain count to match target property
