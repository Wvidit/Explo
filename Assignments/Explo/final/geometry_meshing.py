"""
geometry_meshing.py
===================
Phase 3 — Geometry Construction & Meshing for Wollastonite (CaSiO₃)

Builds the cylindrical specimen (8 mm Ø × 4 mm H) in PyMAPDL,
meshes with SOLID226 coupled thermal-structural elements, and maps
the RVE microstructure onto element material IDs.

When no ANSYS license is available, falls back to analytical models.
"""

import numpy as np
import sys

import shutil
import os
import glob

# ── Detect MAPDL solver binary FIRST (filesystem only, no ansys imports) ──
# We check for the binary BEFORE importing ansys.mapdl.core because that
# import alone can hang for 30+ seconds even when the solver isn't installed.

def _find_mapdl_binary():
    """Non-interactive search for the MAPDL solver executable."""
    # Strategy 1: ANSYS_DIR environment variable
    ansys_dir = os.environ.get("ANSYS_DIR", "")
    if ansys_dir:
        bin_dir = os.path.join(ansys_dir, "ansys", "bin")
        if os.path.isdir(bin_dir) and glob.glob(os.path.join(bin_dir, "ansys*")):
            return True

    # Strategy 2: Check PATH for ansysXXX executables
    for ver in range(260, 200, -1):
        if shutil.which(f"ansys{ver}"):
            return True

    # Strategy 3: Common Linux install locations
    for pattern in [
        os.path.expanduser("~/ansys_inc/v*/ansys/bin/ansys*"),
        "/ansys_inc/v*/ansys/bin/ansys*",
        "/usr/ansys_inc/v*/ansys/bin/ansys*",
        "/opt/ansys_inc/v*/ansys/bin/ansys*",
    ]:
        if glob.glob(pattern):
            return True

    return False

_MAPDL_EXEC_FOUND = _find_mapdl_binary()
launch_mapdl = None
HAS_MAPDL = False

if _MAPDL_EXEC_FOUND:
    # Only now do we pay the cost of importing the heavy ansys package
    try:
        from ansys.mapdl.core import launch_mapdl
        HAS_MAPDL = True
    except ImportError:
        print("WARNING: MAPDL binary found but ansys-mapdl-core not installed.")
        print("         Install with: pip install ansys-mapdl-core")
else:
    print("INFO: MAPDL solver binary not found — analytical fallback will be used.")

from material_data import get_material_tables, TEMP_POINTS


# ═══════════════════════════════════════════════════════════════════════════
#  Constants — Specimen geometry
# ═══════════════════════════════════════════════════════════════════════════
CYLINDER_DIAMETER = 8.0e-3    # 8 mm  → metres
CYLINDER_RADIUS   = CYLINDER_DIAMETER / 2.0
CYLINDER_HEIGHT   = 4.0e-3    # 4 mm  → metres
ELEMENT_SIZE      = 0.5e-3    # 0.5 mm target element edge length


# ═══════════════════════════════════════════════════════════════════════════
#  MAPDL session management
# ═══════════════════════════════════════════════════════════════════════════

def start_mapdl(**kwargs):
    """
    Launch a PyMAPDL session.

    All keyword arguments are forwarded to ``launch_mapdl``.
    Raises RuntimeError if the license check fails.

    Returns
    -------
    mapdl : ansys.mapdl.core.Mapdl
        Active MAPDL session.
    """
    if not HAS_MAPDL:
        raise RuntimeError(
            "ansys-mapdl-core is not installed. Cannot launch MAPDL. "
            "Use build_model(rve, use_fallback=True) for analytical mode."
        )
    try:
        mapdl = launch_mapdl(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to launch ANSYS MAPDL — ensure a valid license is "
            f"available.\n  Original error: {e}"
        ) from e

    # Quick license verification
    mapdl.prep7()
    return mapdl


# ═══════════════════════════════════════════════════════════════════════════
#  Material definition
# ═══════════════════════════════════════════════════════════════════════════

def define_materials(mapdl):
    """
    Define temperature-dependent material properties in MAPDL.

    Material 1: Wollastonite (matrix)
    Material 2: Air (pore)

    Parameters
    ----------
    mapdl : Mapdl session (must be in /PREP7).
    """
    tables = get_material_tables()
    temps = tables["temperatures"]

    # --- Material 1: Wollastonite ---
    mapdl.mp("DENS", 1, tables["wollastonite"]["rho"])  # constant density

    for i, T in enumerate(temps):
        mapdl.mptemp(i + 1, T)

    # Young's modulus
    for i, T in enumerate(temps):
        mapdl.mpdata("EX", 1, i + 1, float(tables["wollastonite"]["E"][i]))

    # Poisson's ratio
    for i, T in enumerate(temps):
        mapdl.mpdata("PRXY", 1, i + 1, float(tables["wollastonite"]["nu"][i]))

    # Thermal conductivity
    for i, T in enumerate(temps):
        mapdl.mpdata("KXX", 1, i + 1, float(tables["wollastonite"]["k"][i]))

    # CTE
    for i, T in enumerate(temps):
        mapdl.mpdata("ALPX", 1, i + 1, float(tables["wollastonite"]["cte"][i]))

    # Specific heat
    for i, T in enumerate(temps):
        mapdl.mpdata("C", 1, i + 1, float(tables["wollastonite"]["cp"][i]))

    # --- Material 2: Air (pore phase) ---
    mapdl.mptemp("")  # clear temp table

    mapdl.mp("DENS", 2, tables["air"]["rho"])

    for i, T in enumerate(temps):
        mapdl.mptemp(i + 1, T)

    for i, T in enumerate(temps):
        mapdl.mpdata("EX", 2, i + 1, float(tables["air"]["E"][i]))

    for i, T in enumerate(temps):
        mapdl.mpdata("PRXY", 2, i + 1, float(tables["air"]["nu"][i]))

    for i, T in enumerate(temps):
        mapdl.mpdata("KXX", 2, i + 1, float(tables["air"]["k"][i]))

    for i, T in enumerate(temps):
        mapdl.mpdata("ALPX", 2, i + 1, float(tables["air"]["cte"][i]))

    for i, T in enumerate(temps):
        mapdl.mpdata("C", 2, i + 1, float(tables["air"]["cp"][i]))

    print("  → Materials defined: 1 = Wollastonite, 2 = Air")


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry & Mesh
# ═══════════════════════════════════════════════════════════════════════════

def create_cylinder_mesh(mapdl, element_size=ELEMENT_SIZE):
    """
    Create the cylindrical specimen and mesh it with SOLID226.

    Parameters
    ----------
    mapdl : Mapdl session.
    element_size : float
        Target element edge length (m). Default 0.5 mm.

    Returns
    -------
    n_elements : int
        Number of elements in the mesh.
    """
    mapdl.prep7()

    # Element type: SOLID226 with KEYOPT(1)=11 for coupled structural-thermal
    mapdl.et(1, "SOLID226", 11)

    # Create cylinder aligned along Z axis, centred at origin
    # CYLIND: RAD1, RAD2, Z1, Z2
    mapdl.cylind(0, CYLINDER_RADIUS, 0, CYLINDER_HEIGHT)

    # Global element size
    mapdl.esize(element_size)

    # Volume mesh
    mapdl.vmesh("ALL")

    # Query element count
    n_elements = mapdl.mesh.n_elem
    print(f"  → Cylinder meshed: {n_elements} elements "
          f"(size = {element_size*1e3:.2f} mm)")

    return n_elements


# ═══════════════════════════════════════════════════════════════════════════
#  Phase mapping — RVE → FE mesh
# ═══════════════════════════════════════════════════════════════════════════

def map_phases(mapdl, rve_array):
    """
    Map the RVE voxel microstructure onto the FE mesh.

    For each element, compute the centroid coordinates, translate them
    to voxel indices in the RVE array, and assign:
        material 1 (Wollastonite) if matrix voxel
        material 2 (Air)          if pore voxel

    The RVE is tiled periodically to cover the full cylinder.

    Parameters
    ----------
    mapdl : Mapdl session (mesh must already exist).
    rve_array : ndarray, shape (Nv, Nv, Nv)
        0 = matrix, 1 = pore.

    Returns
    -------
    mat_assignment : ndarray of int
        Material ID (1 or 2) for each element.
    porosity_mapped : float
        Fraction of elements assigned as pore.
    """
    Nv = rve_array.shape[0]

    # Get element centroids
    elem_ids = mapdl.mesh.enum
    centroids = mapdl.mesh.elem_centroids  # shape (n_elem, 3)

    # Coordinate ranges of the mesh
    x_min, x_max = centroids[:, 0].min(), centroids[:, 0].max()
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    z_min, z_max = centroids[:, 2].min(), centroids[:, 2].max()

    # Normalise centroid coordinates to [0, Nv) for RVE lookup
    def _norm(vals, vmin, vmax):
        span = vmax - vmin if vmax > vmin else 1.0
        return ((vals - vmin) / span * Nv).astype(int) % Nv

    ix = _norm(centroids[:, 0], x_min, x_max)
    iy = _norm(centroids[:, 1], y_min, y_max)
    iz = _norm(centroids[:, 2], z_min, z_max)

    # Look up voxel phase
    phases = rve_array[ix, iy, iz]   # 0 = matrix, 1 = pore
    mat_assignment = np.where(phases == 0, 1, 2)   # MAPDL mat IDs

    # Apply material assignments element by element
    n_pore = 0
    for i, eid in enumerate(elem_ids):
        mat_id = int(mat_assignment[i])
        mapdl.emodif(eid, "MAT", mat_id)
        if mat_id == 2:
            n_pore += 1

    porosity_mapped = n_pore / len(elem_ids) if len(elem_ids) > 0 else 0.0

    print(f"  → Phase mapping complete: "
          f"{n_pore}/{len(elem_ids)} elements → pore "
          f"({porosity_mapped*100:.2f}% porosity)")

    return mat_assignment, porosity_mapped


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

def build_model(rve_array, element_size=ELEMENT_SIZE, mapdl=None,
                use_fallback=False, **mapdl_kwargs):
    """
    Complete Phase 3: launch MAPDL → define materials → mesh → map phases.

    When ``use_fallback=True`` or MAPDL is unavailable, delegates to
    ``analytical_fallback.build_model_analytical()``.

    Parameters
    ----------
    rve_array : ndarray, shape (Nv, Nv, Nv)
        RVE from Phase 2 (0=matrix, 1=pore).
    element_size : float
        Target element size in metres.
    mapdl : Mapdl or None
        Existing session; if None, a new one will be launched.
    use_fallback : bool
        If True, skip MAPDL and use analytical models.
    **mapdl_kwargs
        Forwarded to launch_mapdl if creating a new session.

    Returns
    -------
    mapdl : Mapdl session or SimpleNamespace (fallback).
    mat_assignment : ndarray
    porosity_mapped : float
    """
    if use_fallback or not HAS_MAPDL:
        from analytical_fallback import build_model_analytical
        return build_model_analytical(rve_array)

    if mapdl is None:
        mapdl = start_mapdl(**mapdl_kwargs)

    mapdl.clear()
    mapdl.prep7()

    define_materials(mapdl)
    create_cylinder_mesh(mapdl, element_size)
    mat_assignment, porosity_mapped = map_phases(mapdl, rve_array)

    return mapdl, mat_assignment, porosity_mapped


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test (requires ANSYS license)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from rve_generator import generate_rve

    print("=" * 60)
    print("Phase 3 — Geometry Construction & Meshing")
    print("=" * 60)

    # Generate a sample RVE
    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )

    # Build the full model
    mapdl, mat_assign, porosity = build_model(rve)

    print(f"\n  Elements: {mapdl.mesh.n_elem}")
    print(f"  Nodes:    {mapdl.mesh.n_node}")
    print(f"  Mapped porosity: {porosity*100:.2f}%")

    mapdl.exit()
    print("\n✓ Geometry & meshing complete.")
