#!/usr/bin/env python3
"""
Anatomical Passport Visualizer
===============================

Creates multi-panel visualizations from passport JSON files generated
by 06_inferenceAndPassport.py.

Panels:
  1. 3D organ map (centroids + principal axes + boundary cloud)
  2. Organ volume bar chart (sorted, with labels)
  3. Three 2D anatomical projections (axial, coronal, sagittal)

Usage:
    python 08_visualize_passport.py <passport.json> [--modality ct|mri]
    python 08_visualize_passport.py benchmarks/inference_and_passport_results_mri/passports/amos_0508.nii_passport.json --modality mri
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# ===== Label name maps (from TotalSegmentator) =====
CT_LABELS = {
    1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_R",
    9: "adrenal_L", 10: "lung_UL_L", 11: "lung_LL_L", 12: "lung_UL_R",
    13: "lung_ML_R", 14: "lung_LL_R", 15: "esophagus", 16: "trachea",
    17: "thyroid", 18: "small_bowel", 19: "duodenum", 20: "colon",
    21: "bladder", 22: "prostate", 23: "kidney_cyst_L", 24: "kidney_cyst_R",
    25: "sacrum", 26: "vert_S1", 27: "vert_L5", 28: "vert_L4",
    29: "vert_L3", 30: "vert_L2", 31: "vert_L1", 32: "vert_T12",
    33: "vert_T11", 34: "vert_T10", 35: "vert_T9", 36: "vert_T8",
    37: "vert_T7", 38: "vert_T6", 39: "vert_T5", 40: "vert_T4",
    41: "vert_T3", 42: "vert_T2", 43: "vert_T1", 44: "vert_C7",
    45: "vert_C6", 46: "vert_C5", 47: "vert_C4", 48: "vert_C3",
    49: "vert_C2", 50: "vert_C1", 51: "heart", 52: "aorta",
    53: "pulm_vein", 54: "brachioceph_trunk", 55: "subclavian_R",
    56: "subclavian_L", 57: "carotid_R", 58: "carotid_L",
    59: "brachioceph_vein_L", 60: "brachioceph_vein_R",
    61: "atrial_app_L", 62: "SVC", 63: "IVC", 64: "portal_vein",
    65: "iliac_art_L", 66: "iliac_art_R", 67: "iliac_vein_L",
    68: "iliac_vein_R", 69: "humerus_L", 70: "humerus_R",
    71: "scapula_L", 72: "scapula_R", 73: "clavicula_L", 74: "clavicula_R",
    75: "femur_L", 76: "femur_R", 77: "hip_L", 78: "hip_R",
    79: "spinal_cord", 80: "glut_max_L", 81: "glut_max_R",
    82: "glut_med_L", 83: "glut_med_R", 84: "glut_min_L",
    85: "glut_min_R", 86: "autochthon_L", 87: "autochthon_R",
    88: "iliopsoas_L", 89: "iliopsoas_R", 90: "brain", 91: "skull",
    92: "rib_L1", 93: "rib_L2", 94: "rib_L3", 95: "rib_L4",
    96: "rib_L5", 97: "rib_L6", 98: "rib_L7", 99: "rib_L8",
    100: "rib_L9", 101: "rib_L10", 102: "rib_L11", 103: "rib_L12",
    104: "rib_R1", 105: "rib_R2", 106: "rib_R3", 107: "rib_R4",
    108: "rib_R5", 109: "rib_R6", 110: "rib_R7", 111: "rib_R8",
    112: "rib_R9", 113: "rib_R10", 114: "rib_R11", 115: "rib_R12",
    116: "sternum", 117: "costal_cartilages",
}

MRI_LABELS = {
    1: "spleen", 2: "kidney_R", 3: "kidney_L", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_R",
    9: "adrenal_L", 10: "lung_L", 11: "lung_R", 12: "esophagus",
    13: "small_bowel", 14: "duodenum", 15: "colon", 16: "bladder",
    17: "prostate", 18: "sacrum", 19: "vertebrae", 20: "IVD",
    21: "spinal_cord", 22: "heart", 23: "aorta", 24: "IVC",
    25: "portal_vein", 26: "iliac_art_L", 27: "iliac_art_R",
    28: "iliac_vein_L", 29: "iliac_vein_R", 30: "humerus_L",
    31: "humerus_R", 32: "scapula_L", 33: "scapula_R",
    34: "clavicula_L", 35: "clavicula_R", 36: "femur_L", 37: "femur_R",
    38: "hip_L", 39: "hip_R", 40: "glut_max_L", 41: "glut_max_R",
    42: "glut_med_L", 43: "glut_med_R", 44: "glut_min_L",
    45: "glut_min_R", 46: "autochthon_L", 47: "autochthon_R",
    48: "iliopsoas_L", 49: "iliopsoas_R", 50: "brain",
}

# Organ category colors for grouping
ORGAN_CATEGORIES = {
    "bone": ["sacrum", "vertebrae", "vert_", "humerus", "scapula",
             "clavicula", "femur", "hip", "rib_", "skull", "sternum",
             "costal"],
    "muscle": ["glut_max", "glut_med", "glut_min", "autochthon",
               "iliopsoas"],
    "lung": ["lung"],
    "vessel": ["aorta", "IVC", "portal_vein", "iliac_art", "iliac_vein"],
    "organ": ["spleen", "kidney", "liver", "stomach", "pancreas",
              "gallbladder", "heart", "bladder", "prostate", "brain",
              "adrenal"],
    "gi": ["esophagus", "small_bowel", "duodenum", "colon"],
    "neuro": ["spinal_cord", "IVD"],
}

CATEGORY_COLORS = {
    "bone": "#e6c44c",
    "muscle": "#c0392b",
    "lung": "#5dade2",
    "vessel": "#e74c3c",
    "organ": "#27ae60",
    "gi": "#f39c12",
    "neuro": "#8e44ad",
    "other": "#95a5a6",
}


def get_organ_name(label_int, label_map):
    return label_map.get(label_int, f"label_{label_int}")


def get_category(organ_name):
    name_lower = organ_name.lower()
    for cat, keywords in ORGAN_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return cat
    return "other"


def load_passport(path):
    with open(path) as f:
        data = json.load(f)
    # Keys may be strings from JSON — ensure int
    return {int(k): v for k, v in data.items()}


def plot_3d_organ_map(ax, passport, label_map):
    """Panel 1: 3D centroids with principal axes and boundary cloud."""
    ax.set_title("3D Organ Map", fontsize=12, fontweight="bold")

    for label, info in passport.items():
        name = get_organ_name(label, label_map)
        cat = get_category(name)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])

        cx, cy, cz = info["centroid"]

        # Plot boundary points (faded)
        bp = np.array(info.get("boundary_points", []))
        if len(bp) > 0:
            # Subsample for speed
            if len(bp) > 100:
                idx = np.random.choice(len(bp), 100, replace=False)
                bp = bp[idx]
            ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2],
                       c=color, alpha=0.03, s=1, rasterized=True)

        # Plot centroid
        ax.scatter(cx, cy, cz, c=color, s=40, edgecolors="k",
                   linewidths=0.3, zorder=5)

        # Plot principal axes (eigenvectors scaled by sqrt of eigenvalues)
        evals = np.array(info["eigenvalues"])
        evecs = np.array(info["eigenvectors"])
        for i in range(3):
            scale = np.sqrt(max(evals[i], 0)) * 0.3
            direction = evecs[:, i] * scale
            ax.plot([cx, cx + direction[0]],
                    [cy, cy + direction[1]],
                    [cz, cz + direction[2]],
                    color=color, alpha=0.5, linewidth=0.8)

    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.tick_params(labelsize=6)


def plot_volume_chart(ax, passport, label_map, top_n=25):
    """Panel 2: Horizontal bar chart of organ volumes."""
    ax.set_title(f"Organ Volumes (Top {top_n})", fontsize=12,
                 fontweight="bold")

    volumes = []
    for label, info in passport.items():
        name = get_organ_name(label, label_map)
        vol_cc = info["volume_mm3"] / 1000.0  # mm³ → cm³ (cc)
        cat = get_category(name)
        volumes.append((name, vol_cc, cat))

    # Sort by volume descending, take top_n
    volumes.sort(key=lambda x: x[1], reverse=True)
    volumes = volumes[:top_n]
    volumes.reverse()  # Flip so largest is at top in horizontal bar

    names = [v[0] for v in volumes]
    vals = [v[1] for v in volumes]
    colors = [CATEGORY_COLORS.get(v[2], CATEGORY_COLORS["other"])
              for v in volumes]

    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white",
                   linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Volume (cm³)", fontsize=9)

    # Add value labels on bars
    for bar, val in zip(bars, vals):
        if val > 0.5:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}", va="center", fontsize=6, color="#555")


def plot_2d_projection(ax, passport, label_map, plane="axial"):
    """Panel 3/4/5: 2D anatomical projections with organ labels."""
    plane_config = {
        "axial": {"x_idx": 0, "y_idx": 1, "xlabel": "X (R→L)",
                  "ylabel": "Y (A→P)", "title": "Axial (top-down)"},
        "coronal": {"x_idx": 0, "y_idx": 2, "xlabel": "X (R→L)",
                    "ylabel": "Z (I→S)", "title": "Coronal (front)"},
        "sagittal": {"x_idx": 1, "y_idx": 2, "xlabel": "Y (A→P)",
                     "ylabel": "Z (I→S)", "title": "Sagittal (side)"},
    }
    cfg = plane_config[plane]
    ax.set_title(cfg["title"], fontsize=11, fontweight="bold")

    for label, info in passport.items():
        name = get_organ_name(label, label_map)
        cat = get_category(name)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])

        c = info["centroid"]
        x, y = c[cfg["x_idx"]], c[cfg["y_idx"]]

        # Draw ellipse from eigenvalues/eigenvectors projected onto plane
        evals = np.array(info["eigenvalues"])
        evecs = np.array(info["eigenvectors"])

        # Project eigenvectors onto the 2D plane
        scale = np.sqrt(np.maximum(evals, 0)) * 0.4
        for i in range(3):
            dx = evecs[cfg["x_idx"], i] * scale[i]
            dy = evecs[cfg["y_idx"], i] * scale[i]
            ax.plot([x - dx, x + dx], [y - dy, y + dy],
                    color=color, alpha=0.3, linewidth=1.5)

        # Centroid marker
        vol_cc = info["volume_mm3"] / 1000.0
        marker_size = max(10, min(80, vol_cc * 0.15))
        ax.scatter(x, y, c=color, s=marker_size, edgecolors="k",
                   linewidths=0.3, zorder=5, alpha=0.8)

        # Label (offset to avoid overlap)
        ax.annotate(name, (x, y), fontsize=4.5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(cfg["xlabel"], fontsize=8)
    ax.set_ylabel(cfg["ylabel"], fontsize=8)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.15)


def plot_category_legend(ax):
    """Draw a legend showing organ category colors."""
    ax.set_title("Categories", fontsize=10, fontweight="bold")
    ax.axis("off")

    categories = list(CATEGORY_COLORS.items())
    for i, (cat, color) in enumerate(categories):
        y = 1.0 - (i + 1) * 0.12
        ax.add_patch(plt.Rectangle((0.05, y - 0.03), 0.08, 0.06,
                                   facecolor=color, edgecolor="k",
                                   linewidth=0.5,
                                   transform=ax.transAxes))
        ax.text(0.18, y, cat.capitalize(), fontsize=9,
                transform=ax.transAxes, va="center")


def render_organ_surface(ax, points, color, alpha=0.25):
    """Render organ surface using ConvexHull triangulation."""
    if len(points) < 4:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=2, alpha=0.5)
        return
    try:
        hull = ConvexHull(points)
        triangles = []
        for simplex in hull.simplices:
            triangle = [points[simplex[0]], points[simplex[1]],
                        points[simplex[2]]]
            triangles.append(triangle)
        mesh = Poly3DCollection(triangles, alpha=alpha, linewidths=0.1,
                                edgecolors=color)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
    except Exception:
        # Fall back to point cloud if hull fails
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=2, alpha=0.3)


def visualize_organs(passport_path, modality="ct", output_path=None):
    """Generate 3D organ shape visualization using boundary points."""
    passport = load_passport(passport_path)
    label_map = MRI_LABELS if modality == "mri" else CT_LABELS

    subject_name = Path(passport_path).stem.replace("_passport", "")
    n_organs = len(passport)

    # Sort organs by volume (largest first for back-to-front rendering)
    sorted_organs = sorted(passport.items(),
                           key=lambda x: x[1]["volume_mm3"], reverse=True)

    # 4 views: front, top, side, 3/4 angle
    views = [
        {"elev": 0, "azim": 0, "title": "Front (Coronal)"},
        {"elev": 90, "azim": 0, "title": "Top (Axial)"},
        {"elev": 0, "azim": 90, "title": "Side (Sagittal)"},
        {"elev": 25, "azim": -60, "title": "3D View"},
    ]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"Organ Shapes: {subject_name}\n"
        f"Modality: {modality.upper()} | {n_organs} organs rendered "
        f"from boundary points",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = fig.add_gridspec(2, 3, hspace=0.2, wspace=0.15)

    # 4 view panels
    axes_pos = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0:2]]
    axes = []
    for pos in axes_pos:
        ax = fig.add_subplot(pos, projection="3d")
        axes.append(ax)

    # Legend panel (2D)
    ax_legend = fig.add_subplot(gs[1, 2])

    # Collect all boundary points for axis limits
    all_pts = []
    organ_render_data = []
    for label, info in sorted_organs:
        bp = np.array(info.get("boundary_points", []))
        if len(bp) < 4:
            continue
        name = get_organ_name(label, label_map)
        cat = get_category(name)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
        all_pts.append(bp)
        organ_render_data.append((name, bp, color, info["volume_mm3"]))

    if not all_pts:
        print("No boundary points found in passport.")
        return

    all_pts = np.vstack(all_pts)
    margin = 20
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    np.random.seed(42)

    for ax, view in zip(axes, views):
        ax.set_title(view["title"], fontsize=11, fontweight="bold")

        for name, bp, color, vol in organ_render_data:
            render_organ_surface(ax, bp, color, alpha=0.20)

            # Add label at centroid
            centroid = bp.mean(axis=0)
            ax.text(centroid[0], centroid[1], centroid[2], name,
                    fontsize=3.5, alpha=0.6, ha="center")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=view["elev"], azim=view["azim"])
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=5)

    # Legend with top organs by volume
    ax_legend.set_title("Top Organs by Volume", fontsize=10,
                        fontweight="bold")
    ax_legend.axis("off")

    top_organs = organ_render_data[:20]
    for i, (name, _, color, vol) in enumerate(top_organs):
        y = 1.0 - (i + 1) * 0.048
        ax_legend.add_patch(plt.Rectangle(
            (0.02, y - 0.015), 0.05, 0.03, facecolor=color,
            edgecolor="k", linewidth=0.3, transform=ax_legend.transAxes))
        vol_cc = vol / 1000.0
        ax_legend.text(0.09, y, f"{name} ({vol_cc:.0f} cm³)",
                       fontsize=7, transform=ax_legend.transAxes,
                       va="center")

    fig.subplots_adjust(top=0.92)

    if output_path is None:
        output_path = str(Path(passport_path).with_suffix("")) + "_organs.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()

    print(f"Organ visualization saved to: {output_path}")
    return output_path


def visualize_passport(passport_path, modality="ct", output_path=None):
    """Generate multi-panel passport visualization."""
    passport = load_passport(passport_path)
    label_map = MRI_LABELS if modality == "mri" else CT_LABELS

    subject_name = Path(passport_path).stem.replace("_passport", "")
    n_organs = len(passport)
    total_vol_cc = sum(v["volume_mm3"] for v in passport.values()) / 1000.0

    # Create figure with 6 panels
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Anatomical Passport: {subject_name}\n"
        f"Modality: {modality.upper()} | Organs: {n_organs} | "
        f"Total Volume: {total_vol_cc:.0f} cm³",
        fontsize=14, fontweight="bold", y=0.98
    )

    # Panel layout using GridSpec for mixed column counts
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.35)

    # Row 1: 3D map (3 cols wide) + volume chart (1 col)
    ax_3d = fig.add_subplot(gs[0, 0:3], projection="3d")
    ax_vol = fig.add_subplot(gs[0, 3])

    # Row 2: Axial + Coronal + Sagittal + Legend
    ax_axial = fig.add_subplot(gs[1, 0])
    ax_coronal = fig.add_subplot(gs[1, 1])
    ax_sagittal = fig.add_subplot(gs[1, 2])
    ax_legend = fig.add_subplot(gs[1, 3])

    # Render panels
    np.random.seed(42)  # Reproducible boundary subsampling
    plot_3d_organ_map(ax_3d, passport, label_map)
    plot_volume_chart(ax_vol, passport, label_map)
    plot_2d_projection(ax_axial, passport, label_map, plane="axial")
    plot_2d_projection(ax_coronal, passport, label_map, plane="coronal")
    plot_2d_projection(ax_sagittal, passport, label_map, plane="sagittal")
    plot_category_legend(ax_legend)

    fig.subplots_adjust(top=0.92)

    # Save or show
    if output_path is None:
        output_path = Path(passport_path).with_suffix(".png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()

    print(f"Visualization saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize anatomical passport from JSON file"
    )
    parser.add_argument(
        "passport",
        type=str,
        help="Path to passport JSON file"
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["ct", "mri"],
        default="ct",
        help="Imaging modality for label names (default: ct)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: same name as input with .png)"
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["summary", "organs", "both"],
        default="both",
        help="Visualization type: summary (dashboard), organs (3D shapes), "
             "or both (default: both)"
    )

    args = parser.parse_args()

    if args.view in ("summary", "both"):
        visualize_passport(args.passport, modality=args.modality,
                           output_path=args.output)
    if args.view in ("organs", "both"):
        visualize_organs(args.passport, modality=args.modality,
                         output_path=None if args.view == "both"
                         else args.output)


if __name__ == "__main__":
    main()
