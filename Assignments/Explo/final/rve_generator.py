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
    Publication-quality 2×2 RVE visualisation.

    Layout:
      Top-left:     3D grain boundaries (coloured by grain ID)
      Top-right:    3D pore network (depth-coloured)
      Bottom-left:  2D mid-slice cross-section (grains + pores)
      Bottom-right: Microstructure statistics panel

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
    from matplotlib.colors import Normalize
    from matplotlib import cm

    N = rve.shape[0]
    porosity = float(rve.sum()) / rve.size
    n_grains = len(np.unique(grain_ids))

    # ── Dark theme ──────────────────────────────────────────────────────
    BG       = "#0f1117"
    PANEL_BG = "#181c25"
    TEXT     = "#e0e4ec"
    ACCENT   = "#6ee7b7"
    ACCENT2  = "#f472b6"
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

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Wollastonite RVE Microstructure",
        fontsize=18, fontweight="bold", color=ACCENT, y=0.97,
    )

    # ── (1) Top-left — 3D Grain Boundaries ──────────────────────────────
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_facecolor(PANEL_BG)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor(GRID_CLR)
    ax1.yaxis.pane.set_edgecolor(GRID_CLR)
    ax1.zaxis.pane.set_edgecolor(GRID_CLR)
    ax1.grid(True, alpha=0.15)

    # Detect grain boundaries at full resolution
    boundary = np.zeros((N, N, N), dtype=bool)
    boundary[:-1, :, :] |= (grain_ids[:-1, :, :] != grain_ids[1:, :, :])
    boundary[:, :-1, :] |= (grain_ids[:, :-1, :] != grain_ids[:, 1:, :])
    boundary[:, :, :-1] |= (grain_ids[:, :, :-1] != grain_ids[:, :, 1:])
    mask_matrix = rve == 0
    boundary_pts = boundary & mask_matrix

    # Subsample for rendering (max ~12k points)
    bi, bj, bk = np.where(boundary_pts)
    max_pts = 12000
    if len(bi) > max_pts:
        idx = np.random.choice(len(bi), max_pts, replace=False)
        bi, bj, bk = bi[idx], bj[idx], bk[idx]

    if show_grains and len(bi) > 0:
        c_vals = grain_ids[bi, bj, bk].astype(float)
        ax1.scatter(
            bi, bj, bk,
            c=c_vals, cmap="Spectral", s=1.0, alpha=0.55,
            edgecolors="none", rasterized=True,
        )
    elif len(bi) > 0:
        ax1.scatter(
            bi, bj, bk,
            color=ACCENT, s=0.8, alpha=0.4,
            edgecolors="none", rasterized=True,
        )

    ax1.set_title("Grain Boundaries", fontsize=13, fontweight="bold",
                  color=TEXT, pad=12)
    ax1.set_xlabel("X", labelpad=8)
    ax1.set_ylabel("Y", labelpad=8)
    ax1.set_zlabel("Z", labelpad=8)
    ax1.set_xlim(0, N); ax1.set_ylim(0, N); ax1.set_zlim(0, N)
    ax1.view_init(elev=25, azim=135)
    ax1.tick_params(labelsize=8)

    # ── (2) Top-right — 3D Pore Network ─────────────────────────────────
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.set_facecolor(PANEL_BG)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.pane.set_edgecolor(GRID_CLR)
    ax2.yaxis.pane.set_edgecolor(GRID_CLR)
    ax2.zaxis.pane.set_edgecolor(GRID_CLR)
    ax2.grid(True, alpha=0.15)

    pi, pj, pk = np.where(rve == 1)
    max_pore_pts = 8000
    if len(pi) > max_pore_pts:
        idx = np.random.choice(len(pi), max_pore_pts, replace=False)
        pi, pj, pk = pi[idx], pj[idx], pk[idx]

    if len(pi) > 0:
        # Depth-colour by Z coordinate for 3D perception
        depth = pk.astype(float) / N
        ax2.scatter(
            pi, pj, pk,
            c=depth, cmap="magma", s=4, alpha=0.75,
            edgecolors="none", rasterized=True,
        )

    n_pore = int(rve.sum())
    ax2.set_title(
        f"Pore Network  ({n_pore:,} / {rve.size:,} voxels)",
        fontsize=13, fontweight="bold", color=TEXT, pad=12,
    )
    ax2.set_xlabel("X", labelpad=8)
    ax2.set_ylabel("Y", labelpad=8)
    ax2.set_zlabel("Z", labelpad=8)
    ax2.set_xlim(0, N); ax2.set_ylim(0, N); ax2.set_zlim(0, N)
    ax2.view_init(elev=25, azim=135)
    ax2.tick_params(labelsize=8)

    # ── (3) Bottom-left — 2D Mid-slice Cross-section ────────────────────
    ax3 = fig.add_subplot(223)
    mid = N // 2
    slice_grains = grain_ids[:, :, mid].astype(float)
    slice_pores  = rve[:, :, mid]

    # Grain map with pore overlay
    ax3.imshow(
        slice_grains.T, origin="lower", cmap="Spectral",
        interpolation="nearest", alpha=0.85,
    )
    # Overlay pores as bright pink
    pore_overlay = np.ma.masked_where(slice_pores == 0, slice_pores)
    ax3.imshow(
        pore_overlay.T, origin="lower", cmap="spring",
        interpolation="nearest", alpha=1.0, vmin=0, vmax=1,
    )

    ax3.set_title(
        f"Cross-section at Z = {mid}",
        fontsize=13, fontweight="bold", color=TEXT, pad=10,
    )
    ax3.set_xlabel("X"); ax3.set_ylabel("Y")
    ax3.tick_params(labelsize=8)

    # Add scale bar
    bar_len = N // 4
    bar_y = N * 0.05
    ax3.plot([N*0.05, N*0.05 + bar_len], [bar_y, bar_y],
             color=ACCENT, linewidth=3, solid_capstyle="butt")
    ax3.text(N*0.05 + bar_len/2, bar_y + N*0.04,
             f"{bar_len} voxels", color=ACCENT,
             ha="center", fontsize=9, fontweight="bold")

    # ── (4) Bottom-right — Statistics Panel ─────────────────────────────
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor(PANEL_BG)
    ax4.axis("off")

    # Compute quick stats
    n_junction = int(boundary_pts.sum())

    stats_lines = [
        ("MICROSTRUCTURE SUMMARY", "", True),
        ("", "", False),
        ("Grid resolution", f"{N}³  =  {N**3:,} voxels", False),
        ("Grain count", f"{n_grains}", False),
        ("Pore voxels", f"{n_pore:,}", False),
        ("Porosity", f"{porosity*100:.2f} %", False),
        ("Boundary voxels", f"{n_junction:,}", False),
        ("", "", False),
        ("POROSITY PROFILE", "", True),
    ]

    y_pos = 0.92
    for label, value, is_header in stats_lines:
        if is_header:
            ax4.text(0.5, y_pos, label, transform=ax4.transAxes,
                     fontsize=13, fontweight="bold", color=ACCENT,
                     ha="center", va="top")
        elif label == "":
            pass  # spacer
        else:
            ax4.text(0.10, y_pos, label, transform=ax4.transAxes,
                     fontsize=11, color="#94a3b8", ha="left", va="top")
            ax4.text(0.90, y_pos, value, transform=ax4.transAxes,
                     fontsize=11, color=TEXT, fontweight="bold",
                     ha="right", va="top")
        y_pos -= 0.065

    # Mini porosity-vs-Z profile
    z_porosity = np.array([
        rve[:, :, z].sum() / (N * N) * 100 for z in range(N)
    ])

    # Inset axes for the profile plot
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax4, width="80%", height="38%",
                          loc="lower center",
                          bbox_to_anchor=(0, 0.02, 1, 0.45),
                          bbox_transform=ax4.transAxes)
    ax_inset.set_facecolor("#1e2330")
    ax_inset.fill_between(range(N), z_porosity, alpha=0.35, color=ACCENT2)
    ax_inset.plot(range(N), z_porosity, color=ACCENT2, linewidth=1.5)
    ax_inset.axhline(porosity * 100, color=ACCENT, linewidth=1,
                     linestyle="--", alpha=0.7, label=f"Mean {porosity*100:.2f}%")
    ax_inset.set_xlabel("Z slice", fontsize=9, color="#94a3b8")
    ax_inset.set_ylabel("Porosity %", fontsize=9, color="#94a3b8")
    ax_inset.tick_params(labelsize=7, colors="#94a3b8")
    ax_inset.spines["top"].set_visible(False)
    ax_inset.spines["right"].set_visible(False)
    ax_inset.spines["bottom"].set_color(GRID_CLR)
    ax_inset.spines["left"].set_color(GRID_CLR)
    ax_inset.legend(fontsize=8, loc="upper right",
                    facecolor="#1e2330", edgecolor=GRID_CLR,
                    labelcolor=TEXT)

    # ── Finalise ────────────────────────────────────────────────────────
    plt.subplots_adjust(
        left=0.04, right=0.96, bottom=0.05, top=0.92,
        wspace=0.15, hspace=0.28,
    )
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"  → Saved RVE visualization: {save_path}")
    plt.close()

    # Reset rcParams to defaults
    plt.rcdefaults()


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
