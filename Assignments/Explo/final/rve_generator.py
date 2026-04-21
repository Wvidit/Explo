"""
rve_generator.py
================
Phase 2 — RVE Microstructure Generation for Wollastonite (CaSiO₃)

Generates 3D representative volume elements (RVEs) by:
  1. Placing random seed points in a cubic domain
  2. Voronoi tessellation → polyhedral grains
  3. Introducing pores at triple junctions (≥3 grain boundaries)
  4. Computing structural descriptors (porosity, chord lengths)
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════
#  Voronoi tessellation
# ═══════════════════════════════════════════════════════════════════════════

def _voronoi_tessellation(n_seeds, voxel_res):
    """
    Create a 3D Voronoi tessellation on a voxel grid.

    Parameters
    ----------
    n_seeds : int
        Number of grain nuclei (seeds).
    voxel_res : int
        Cubic voxel resolution (e.g. 64 → 64³ grid).

    Returns
    -------
    grain_ids : ndarray, shape (voxel_res, voxel_res, voxel_res)
        Integer grain ID for each voxel.
    seeds : ndarray, shape (n_seeds, 3)
        Seed coordinates in voxel space.
    """
    rng = np.random.default_rng()
    seeds = rng.uniform(0, voxel_res, size=(n_seeds, 3))

    # Build grid of voxel centres
    coords = np.arange(voxel_res) + 0.5
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Nearest-seed assignment via KD-tree
    tree = cKDTree(seeds)
    _, grain_ids_flat = tree.query(grid_points)
    grain_ids = grain_ids_flat.reshape((voxel_res, voxel_res, voxel_res))

    return grain_ids, seeds


def _find_triple_junctions(grain_ids):
    """
    Identify voxels at triple junctions (≥3 distinct grain neighbours).

    A voxel is at a triple junction if, among its 26 neighbours,
    at least 3 different grain IDs are present (including itself).

    Parameters
    ----------
    grain_ids : ndarray, shape (N, N, N)

    Returns
    -------
    junction_mask : ndarray of bool, shape (N, N, N)
        True where the voxel sits at a triple junction.
    """
    N = grain_ids.shape[0]
    junction_mask = np.zeros_like(grain_ids, dtype=bool)

    # Pad with -1 so boundary voxels have valid neighbours
    padded = np.pad(grain_ids, 1, mode="constant", constant_values=-1)

    for i in range(N):
        for j in range(N):
            for k in range(N):
                neighbourhood = padded[i:i+3, j:j+3, k:k+3].ravel()
                unique_ids = np.unique(neighbourhood)
                # Remove the padding sentinel
                unique_ids = unique_ids[unique_ids >= 0]
                if len(unique_ids) >= 3:
                    junction_mask[i, j, k] = True

    return junction_mask


def _find_triple_junctions_fast(grain_ids):
    """
    Vectorised triple-junction detection using shifted arrays.
    Much faster than the loop-based version for large grids.
    """
    N = grain_ids.shape[0]
    padded = np.pad(grain_ids, 1, mode="wrap")   # periodic BCs

    unique_count = np.zeros((N, N, N), dtype=np.int32)

    # Collect all 27 neighbours (including self) into a list
    neighbours = []
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                neighbours.append(padded[di:di+N, dj:dj+N, dk:dk+N])

    # Stack and count unique along last axis
    stack = np.stack(neighbours, axis=-1)  # shape (N, N, N, 27)

    # For each voxel, count unique grain IDs among its 27 neighbours
    for i in range(N):
        for j in range(N):
            for k in range(N):
                unique_count[i, j, k] = len(np.unique(stack[i, j, k]))

    return unique_count >= 3


def _find_triple_junctions_vectorized(grain_ids):
    """
    Fully vectorised triple-junction detection.
    Uses the fact that a voxel at a grain boundary has at least one
    neighbour with a different grain ID. Triple junctions have ≥3
    distinct IDs in their local 3×3×3 neighbourhood.
    """
    N = grain_ids.shape[0]
    padded = np.pad(grain_ids, 1, mode="wrap")

    # Collect all 27 shifts
    shifts = []
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                shifts.append(padded[di:di+N, dj:dj+N, dk:dk+N])

    # Stack → (N, N, N, 27)
    stack = np.stack(shifts, axis=-1)

    # Sort along the neighbour axis and count transitions
    sorted_stack = np.sort(stack, axis=-1)
    # Count unique: wherever sorted value changes = new unique ID
    diffs = np.diff(sorted_stack, axis=-1) > 0
    n_unique = diffs.sum(axis=-1) + 1  # +1 for the first element

    return n_unique >= 3


# ═══════════════════════════════════════════════════════════════════════════
#  RVE generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_rve(n_seeds=50, voxel_res=64, target_porosity=0.02, seed=None):
    """
    Generate a 3D RVE with Voronoi grains and triple-junction porosity.

    Parameters
    ----------
    n_seeds : int
        Number of grain seeds (controls grain size).
    voxel_res : int
        Voxel grid resolution per axis.
    target_porosity : float
        Desired porosity fraction (0 to 1), e.g. 0.02 for 2%.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rve : ndarray of int, shape (voxel_res, voxel_res, voxel_res)
        0 = matrix (Wollastonite), 1 = pore (air).
    grain_ids : ndarray of int, same shape
        Grain label for each voxel (before pore insertion).
    metadata : dict
        Contains n_seeds, voxel_res, target_porosity, actual_porosity.
    """
    if seed is not None:
        np.random.seed(seed)

    # Step 1: Voronoi tessellation
    grain_ids, seeds = _voronoi_tessellation(n_seeds, voxel_res)

    # Step 2: Find triple-junction voxels
    junction_mask = _find_triple_junctions_vectorized(grain_ids)
    junction_indices = np.argwhere(junction_mask)

    # Step 3: Convert a subset of junction voxels to pores
    total_voxels = voxel_res ** 3
    n_pore_target = int(target_porosity * total_voxels)

    rve = np.zeros_like(grain_ids, dtype=np.int32)  # 0 = matrix

    if len(junction_indices) > 0:
        if n_pore_target <= len(junction_indices):
            # Randomly select from junction voxels
            chosen = np.random.choice(
                len(junction_indices), size=n_pore_target, replace=False
            )
            for idx in chosen:
                i, j, k = junction_indices[idx]
                rve[i, j, k] = 1   # 1 = pore
        else:
            # All junctions become pores, then add random ones for remainder
            for i, j, k in junction_indices:
                rve[i, j, k] = 1
            remaining = n_pore_target - len(junction_indices)
            matrix_indices = np.argwhere(rve == 0)
            if remaining > 0 and len(matrix_indices) > 0:
                chosen = np.random.choice(
                    len(matrix_indices),
                    size=min(remaining, len(matrix_indices)),
                    replace=False,
                )
                for idx in chosen:
                    i, j, k = matrix_indices[idx]
                    rve[i, j, k] = 1

    actual_porosity = float(rve.sum()) / total_voxels

    metadata = {
        "n_seeds": n_seeds,
        "voxel_res": voxel_res,
        "target_porosity": target_porosity,
        "actual_porosity": actual_porosity,
        "n_junction_voxels": int(junction_mask.sum()),
        "n_pore_voxels": int(rve.sum()),
    }

    return rve, grain_ids, metadata


# ═══════════════════════════════════════════════════════════════════════════
#  Structural descriptors
# ═══════════════════════════════════════════════════════════════════════════

def compute_chord_lengths(rve):
    """
    Compute mean chord length and variance along each axis.

    A chord is a maximal contiguous run of matrix voxels (value 0)
    along a single row/column. This characterises the effective
    grain size in each direction.

    Parameters
    ----------
    rve : ndarray, shape (N, N, N)
        0 = matrix, 1 = pore.

    Returns
    -------
    dict with 'mean_chord_x', 'mean_chord_y', 'mean_chord_z',
              'var_chord_x',  'var_chord_y',  'var_chord_z'.
    """
    result = {}
    N = rve.shape[0]

    for axis, label in enumerate(["x", "y", "z"]):
        chords = []
        for i in range(N):
            for j in range(N):
                # Extract 1D line along the current axis
                if axis == 0:
                    line = rve[:, i, j]
                elif axis == 1:
                    line = rve[i, :, j]
                else:
                    line = rve[i, j, :]

                # Find runs of matrix (0)
                run_len = 0
                for v in line:
                    if v == 0:
                        run_len += 1
                    else:
                        if run_len > 0:
                            chords.append(run_len)
                        run_len = 0
                if run_len > 0:
                    chords.append(run_len)

        if len(chords) > 0:
            result[f"mean_chord_{label}"] = float(np.mean(chords))
            result[f"var_chord_{label}"]  = float(np.var(chords))
        else:
            result[f"mean_chord_{label}"] = 0.0
            result[f"var_chord_{label}"]  = 0.0

    return result


def compute_descriptors(rve, metadata=None):
    """
    Compute the full set of microstructural descriptors for an RVE.

    Parameters
    ----------
    rve : ndarray, shape (N, N, N)
    metadata : dict, optional
        Output from generate_rve; used for n_seeds, etc.

    Returns
    -------
    dict of descriptors.
    """
    total = rve.size
    porosity = float(rve.sum()) / total

    chord_data = compute_chord_lengths(rve)

    descriptors = {
        "porosity": porosity,
        "mean_chord_x": chord_data["mean_chord_x"],
        "mean_chord_y": chord_data["mean_chord_y"],
        "mean_chord_z": chord_data["mean_chord_z"],
        "var_chord_x":  chord_data["var_chord_x"],
        "var_chord_y":  chord_data["var_chord_y"],
        "var_chord_z":  chord_data["var_chord_z"],
        "mean_chord_avg": np.mean([
            chord_data["mean_chord_x"],
            chord_data["mean_chord_y"],
            chord_data["mean_chord_z"],
        ]),
    }

    if metadata is not None:
        descriptors["n_seeds"] = metadata.get("n_seeds", 0)
        descriptors["voxel_res"] = metadata.get("voxel_res", 0)

    return descriptors


# ═══════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════

def visualize_rve(rve, grain_ids, save_path=None, show_grains=True):
    """
    3D voxel visualisation of the RVE.

    Parameters
    ----------
    rve : ndarray, shape (N, N, N)
        0 = matrix, 1 = pore.
    grain_ids : ndarray, shape (N, N, N)
        Grain labels (for colouring).
    save_path : str or None
        If given, save figure to this path.
    show_grains : bool
        If True, colour matrix by grain ID. If False, uniform colour.
    """
    fig = plt.figure(figsize=(14, 6))

    # --- Subplot 1: grain structure (surface voxels only) ---
    ax1 = fig.add_subplot(121, projection="3d")
    N = rve.shape[0]

    # Show only surface and boundary voxels for performance
    # For large grids, subsample
    step = max(1, N // 32)
    sub_rve = rve[::step, ::step, ::step]
    sub_grains = grain_ids[::step, ::step, ::step]
    sN = sub_rve.shape[0]

    # Matrix voxels at boundaries (where grain changes)
    mask_matrix = sub_rve == 0
    # Detect boundaries
    boundary = np.zeros_like(mask_matrix)
    boundary[:-1, :, :] |= (sub_grains[:-1, :, :] != sub_grains[1:, :, :])
    boundary[:, :-1, :] |= (sub_grains[:, :-1, :] != sub_grains[:, 1:, :])
    boundary[:, :, :-1] |= (sub_grains[:, :, :-1] != sub_grains[:, :, 1:])

    # Plot grain boundaries
    bi, bj, bk = np.where(boundary & mask_matrix)
    colors = sub_grains[boundary & mask_matrix] if show_grains else None
    ax1.scatter(bi, bj, bk, c=colors, cmap="tab20", s=1, alpha=0.4)
    ax1.set_title("Grain Boundaries", fontsize=11)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    # --- Subplot 2: pore distribution ---
    ax2 = fig.add_subplot(122, projection="3d")
    pi, pj, pk = np.where(rve == 1)
    if len(pi) > 5000:
        idx = np.random.choice(len(pi), 5000, replace=False)
        pi, pj, pk = pi[idx], pj[idx], pk[idx]
    ax2.scatter(pi, pj, pk, c="red", s=3, alpha=0.7, label="Pores")
    ax2.set_title(f"Pore Distribution ({rve.sum()}/{rve.size} voxels)", fontsize=11)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved RVE visualization: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 — RVE Microstructure Generation")
    print("=" * 60)

    # Generate a sample RVE
    rve, grain_ids, meta = generate_rve(
        n_seeds=50, voxel_res=64, target_porosity=0.02, seed=42
    )
    print(f"\nRVE generated: {meta['voxel_res']}³ voxels, "
          f"{meta['n_seeds']} grains")
    print(f"  Target porosity:  {meta['target_porosity']*100:.1f}%")
    print(f"  Actual porosity:  {meta['actual_porosity']*100:.2f}%")
    print(f"  Junction voxels:  {meta['n_junction_voxels']}")
    print(f"  Pore voxels:      {meta['n_pore_voxels']}")

    # Compute descriptors
    desc = compute_descriptors(rve, meta)
    print(f"\nStructural descriptors:")
    for key, val in desc.items():
        print(f"  {key:20s}: {val:.4f}" if isinstance(val, float)
              else f"  {key:20s}: {val}")

    # Visualize
    save_path = os.path.join(OUTPUT_DIR, "rve_visualization.png")
    visualize_rve(rve, grain_ids, save_path=save_path)

    print("\n✓ RVE generation complete.")
