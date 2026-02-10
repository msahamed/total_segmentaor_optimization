#!/usr/bin/env python3
"""
Spine Visualization from Anatomical Passport
=============================================

Extracts and visualizes spine structures from passport JSON files.
Useful for spine autolabeling projects.

Features:
  - Individual vertebrae rendering (CT: C1-C7, T1-T12, L1-L5, S1)
  - Spinal curve fitting and curvature analysis
  - Inter-vertebral distances and angles
  - Spinal cord visualization
  - IVD visualization (MRI)

Usage:
    python 09_visualize_spine.py <passport.json> [--modality ct|mri]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

# ===== Spine label definitions =====
CT_SPINE_LABELS = {
    25: ("sacrum", "sacral"),
    26: ("S1", "sacral"),
    27: ("L5", "lumbar"), 28: ("L4", "lumbar"), 29: ("L3", "lumbar"),
    30: ("L2", "lumbar"), 31: ("L1", "lumbar"),
    32: ("T12", "thoracic"), 33: ("T11", "thoracic"), 34: ("T10", "thoracic"),
    35: ("T9", "thoracic"), 36: ("T8", "thoracic"), 37: ("T7", "thoracic"),
    38: ("T6", "thoracic"), 39: ("T5", "thoracic"), 40: ("T4", "thoracic"),
    41: ("T3", "thoracic"), 42: ("T2", "thoracic"), 43: ("T1", "thoracic"),
    44: ("C7", "cervical"), 45: ("C6", "cervical"), 46: ("C5", "cervical"),
    47: ("C4", "cervical"), 48: ("C3", "cervical"), 49: ("C2", "cervical"),
    50: ("C1", "cervical"),
    79: ("spinal_cord", "cord"),
}

MRI_SPINE_LABELS = {
    18: ("sacrum", "sacral"),
    19: ("vertebrae", "merged"),
    20: ("IVD", "disc"),
    21: ("spinal_cord", "cord"),
}

# Anatomical order: inferior → superior
VERTEBRA_ORDER = [
    "sacrum", "S1", "L5", "L4", "L3", "L2", "L1",
    "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4",
    "T3", "T2", "T1", "C7", "C6", "C5", "C4", "C3", "C2", "C1",
]

REGION_COLORS = {
    "sacral": "#8B4513",
    "lumbar": "#DAA520",
    "thoracic": "#2E86C1",
    "cervical": "#27AE60",
    "cord": "#E74C3C",
    "disc": "#9B59B6",
    "merged": "#7F8C8D",
}


def load_passport(path):
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def extract_spine_data(passport, modality="ct"):
    """Extract spine structures from passport, ordered inferior to superior."""
    label_map = CT_SPINE_LABELS if modality == "ct" else MRI_SPINE_LABELS

    spine = []
    for label, info in passport.items():
        if label not in label_map:
            continue
        name, region = label_map[label]
        spine.append({
            "label": label,
            "name": name,
            "region": region,
            "centroid": np.array(info["centroid"]),
            "eigenvalues": np.array(info["eigenvalues"]),
            "eigenvectors": np.array(info["eigenvectors"]),
            "boundary_points": np.array(info.get("boundary_points", [])),
            "volume_mm3": info["volume_mm3"],
        })

    # Sort by anatomical order (inferior to superior)
    order_map = {name: i for i, name in enumerate(VERTEBRA_ORDER)}
    spine.sort(key=lambda x: order_map.get(x["name"], 999))

    return spine


def compute_spine_metrics(vertebrae):
    """Compute inter-vertebral distances, angles, and spine curve."""
    # Filter to only vertebrae (not cord/disc)
    verts = [v for v in vertebrae
             if v["region"] not in ("cord", "disc", "merged")]

    if len(verts) < 2:
        return None

    centroids = np.array([v["centroid"] for v in verts])
    names = [v["name"] for v in verts]

    # Inter-vertebral distances
    distances = []
    for i in range(len(centroids) - 1):
        d = np.linalg.norm(centroids[i + 1] - centroids[i])
        distances.append({
            "from": names[i],
            "to": names[i + 1],
            "distance_mm": float(d),
        })

    # Angles between consecutive vertebra direction vectors
    angles = []
    for i in range(1, len(centroids) - 1):
        v1 = centroids[i] - centroids[i - 1]
        v2 = centroids[i + 1] - centroids[i]
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        angles.append({
            "vertex": names[i],
            "angle_deg": float(angle_deg),
        })

    # Fit spline through centroids for spine curve
    # Simple: parameterize by cumulative arc length
    t = np.zeros(len(centroids))
    for i in range(1, len(centroids)):
        t[i] = t[i - 1] + np.linalg.norm(centroids[i] - centroids[i - 1])

    # Total spine length
    total_length = t[-1]

    # Approximate Cobb-like angle: angle between first and last segment
    if len(centroids) >= 3:
        first_dir = centroids[1] - centroids[0]
        last_dir = centroids[-1] - centroids[-2]
        first_norm = first_dir / (np.linalg.norm(first_dir) + 1e-8)
        last_norm = last_dir / (np.linalg.norm(last_dir) + 1e-8)
        cos_cobb = np.clip(np.dot(first_norm, last_norm), -1, 1)
        cobb_angle = np.degrees(np.arccos(cos_cobb))
    else:
        cobb_angle = 0.0

    return {
        "distances": distances,
        "angles": angles,
        "total_length_mm": float(total_length),
        "cobb_angle_approx_deg": float(cobb_angle),
        "centroids": centroids,
        "names": names,
    }


def render_vertebra(ax, bp, color, alpha=0.3):
    """Render vertebra surface from boundary points."""
    if len(bp) < 4:
        ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2], c=color, s=3, alpha=0.5)
        return
    try:
        hull = ConvexHull(bp)
        triangles = [
            [bp[s[0]], bp[s[1]], bp[s[2]]] for s in hull.simplices
        ]
        mesh = Poly3DCollection(triangles, alpha=alpha, linewidths=0.1,
                                edgecolors=color)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
    except Exception:
        ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2], c=color, s=3, alpha=0.3)


def visualize_spine(passport_path, modality="ct", output_path=None):
    """Generate spine-focused visualization."""
    passport = load_passport(passport_path)
    spine = extract_spine_data(passport, modality)

    if not spine:
        print("No spine structures found in passport.")
        return

    subject_name = Path(passport_path).stem.replace("_passport", "")

    # Compute metrics
    metrics = compute_spine_metrics(spine)

    # Figure: 5 panels
    fig = plt.figure(figsize=(22, 14))

    n_vert = len([s for s in spine if s["region"] not in ("cord", "disc")])
    fig.suptitle(
        f"Spine Analysis: {subject_name}\n"
        f"Modality: {modality.upper()} | {n_vert} vertebrae detected",
        fontsize=14, fontweight="bold", y=0.99
    )

    gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.3)

    # Panel 1: 3D spine with labeled vertebrae (wide)
    ax_3d = fig.add_subplot(gs[0, 0:2], projection="3d")
    # Panel 2: Coronal view (AP)
    ax_coronal = fig.add_subplot(gs[0, 2])
    # Panel 3: Sagittal view (lateral)
    ax_sagittal = fig.add_subplot(gs[0, 3])
    # Panel 4: Inter-vertebral distances
    ax_dist = fig.add_subplot(gs[1, 0:2])
    # Panel 5: Metrics summary
    ax_metrics = fig.add_subplot(gs[1, 2:4])

    # ===== Panel 1: 3D Spine =====
    ax_3d.set_title("3D Spine View", fontsize=12, fontweight="bold")

    for vert in spine:
        color = REGION_COLORS.get(vert["region"], "#999")
        bp = vert["boundary_points"]

        if len(bp) > 0:
            render_vertebra(ax_3d, bp, color, alpha=0.35)

        c = vert["centroid"]
        ax_3d.scatter(c[0], c[1], c[2], c=color, s=50, edgecolors="k",
                      linewidths=0.5, zorder=5)
        ax_3d.text(c[0] + 5, c[1], c[2], vert["name"], fontsize=7,
                   fontweight="bold", color=color)

        # Draw principal axis (largest eigenvector)
        evals = vert["eigenvalues"]
        evecs = vert["eigenvectors"]
        main_axis = evecs[:, 2] * np.sqrt(max(evals[2], 0)) * 0.3
        ax_3d.plot([c[0] - main_axis[0], c[0] + main_axis[0]],
                   [c[1] - main_axis[1], c[1] + main_axis[1]],
                   [c[2] - main_axis[2], c[2] + main_axis[2]],
                   color=color, linewidth=1.5, alpha=0.6)

    # Draw spine curve through centroids
    if metrics and len(metrics["centroids"]) > 1:
        curve = metrics["centroids"]
        ax_3d.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                   'k--', linewidth=1.5, alpha=0.5, label="spine curve")

    ax_3d.view_init(elev=15, azim=-70)
    ax_3d.set_xlabel("X (mm)", fontsize=8)
    ax_3d.set_ylabel("Y (mm)", fontsize=8)
    ax_3d.set_zlabel("Z (mm)", fontsize=8)
    ax_3d.tick_params(labelsize=6)

    # ===== Panel 2: Coronal (X vs Z) =====
    ax_coronal.set_title("Coronal View (A→P)", fontsize=12,
                         fontweight="bold")
    for vert in spine:
        color = REGION_COLORS.get(vert["region"], "#999")
        c = vert["centroid"]
        vol_cc = vert["volume_mm3"] / 1000
        size = max(20, min(120, vol_cc * 3))

        ax_coronal.scatter(c[0], c[2], c=color, s=size, edgecolors="k",
                           linewidths=0.5, zorder=5, alpha=0.8)
        ax_coronal.annotate(vert["name"], (c[0], c[2]), fontsize=7,
                            fontweight="bold", color=color,
                            xytext=(8, 0), textcoords="offset points")

        # Draw eigenvector projection
        evecs = vert["eigenvectors"]
        evals = vert["eigenvalues"]
        for i in range(3):
            scale = np.sqrt(max(evals[i], 0)) * 0.3
            dx = evecs[0, i] * scale
            dz = evecs[2, i] * scale
            ax_coronal.plot([c[0] - dx, c[0] + dx],
                            [c[2] - dz, c[2] + dz],
                            color=color, alpha=0.3, linewidth=1)

    # Connect vertebrae with spine curve
    if metrics:
        curve = metrics["centroids"]
        ax_coronal.plot(curve[:, 0], curve[:, 2], 'k--', linewidth=1,
                        alpha=0.4)

    ax_coronal.set_xlabel("X: Right → Left (mm)", fontsize=9)
    ax_coronal.set_ylabel("Z: Inferior → Superior (mm)", fontsize=9)
    ax_coronal.grid(True, alpha=0.2)
    ax_coronal.set_aspect("equal")

    # ===== Panel 3: Sagittal (Y vs Z) =====
    ax_sagittal.set_title("Sagittal View (Lateral)", fontsize=12,
                          fontweight="bold")
    for vert in spine:
        color = REGION_COLORS.get(vert["region"], "#999")
        c = vert["centroid"]
        vol_cc = vert["volume_mm3"] / 1000
        size = max(20, min(120, vol_cc * 3))

        ax_sagittal.scatter(c[1], c[2], c=color, s=size, edgecolors="k",
                            linewidths=0.5, zorder=5, alpha=0.8)
        ax_sagittal.annotate(vert["name"], (c[1], c[2]), fontsize=7,
                             fontweight="bold", color=color,
                             xytext=(8, 0), textcoords="offset points")

        evecs = vert["eigenvectors"]
        evals = vert["eigenvalues"]
        for i in range(3):
            scale = np.sqrt(max(evals[i], 0)) * 0.3
            dy = evecs[1, i] * scale
            dz = evecs[2, i] * scale
            ax_sagittal.plot([c[1] - dy, c[1] + dy],
                             [c[2] - dz, c[2] + dz],
                             color=color, alpha=0.3, linewidth=1)

    if metrics:
        curve = metrics["centroids"]
        ax_sagittal.plot(curve[:, 1], curve[:, 2], 'k--', linewidth=1,
                         alpha=0.4)

    ax_sagittal.set_xlabel("Y: Anterior → Posterior (mm)", fontsize=9)
    ax_sagittal.set_ylabel("Z: Inferior → Superior (mm)", fontsize=9)
    ax_sagittal.grid(True, alpha=0.2)
    ax_sagittal.set_aspect("equal")

    # ===== Panel 4: Inter-vertebral distances =====
    ax_dist.set_title("Inter-Vertebral Distances & Angles", fontsize=12,
                      fontweight="bold")

    if metrics and metrics["distances"]:
        dist_data = metrics["distances"]
        pair_names = [f"{d['from']}→{d['to']}" for d in dist_data]
        dists = [d["distance_mm"] for d in dist_data]

        # Color by region of the lower vertebra
        bar_colors = []
        for d in dist_data:
            for v in spine:
                if v["name"] == d["from"]:
                    bar_colors.append(
                        REGION_COLORS.get(v["region"], "#999"))
                    break
            else:
                bar_colors.append("#999")

        x_pos = np.arange(len(pair_names))
        bars = ax_dist.bar(x_pos, dists, color=bar_colors, edgecolor="white",
                           linewidth=0.3, alpha=0.8)

        # Add distance labels
        for bar, d in zip(bars, dists):
            ax_dist.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{d:.1f}", ha="center", va="bottom", fontsize=6)

        ax_dist.set_xticks(x_pos)
        ax_dist.set_xticklabels(pair_names, rotation=45, ha="right",
                                fontsize=7)
        ax_dist.set_ylabel("Distance (mm)", fontsize=9)
        ax_dist.grid(axis="y", alpha=0.2)

        # Overlay angles on secondary axis
        if metrics["angles"]:
            ax_angle = ax_dist.twinx()
            angle_data = metrics["angles"]
            angle_x = []
            angle_vals = []
            for a in angle_data:
                # Find position matching vertex name
                for i, d in enumerate(dist_data):
                    if d["to"] == a["vertex"]:
                        angle_x.append(i + 0.5)
                        angle_vals.append(a["angle_deg"])
                        break
            if angle_x:
                ax_angle.plot(angle_x, angle_vals, "r^-", markersize=6,
                              linewidth=1, alpha=0.7, label="angle")
                ax_angle.set_ylabel("Angle (°)", fontsize=9, color="red")
                ax_angle.tick_params(axis="y", labelcolor="red", labelsize=7)
    else:
        ax_dist.text(0.5, 0.5, "Insufficient vertebrae\nfor distance analysis",
                     ha="center", va="center", fontsize=12, color="#999",
                     transform=ax_dist.transAxes)
        ax_dist.set_xticks([])

    # ===== Panel 5: Metrics Summary =====
    ax_metrics.axis("off")
    ax_metrics.set_title("Spine Metrics", fontsize=12, fontweight="bold")

    lines = []
    lines.append(f"Subject: {subject_name}")
    lines.append(f"Modality: {modality.upper()}")
    lines.append("")

    # Vertebrae summary
    regions = {}
    for v in spine:
        if v["region"] in ("cord", "disc", "merged"):
            continue
        regions.setdefault(v["region"], []).append(v["name"])

    lines.append("VERTEBRAE DETECTED:")
    for region in ["cervical", "thoracic", "lumbar", "sacral"]:
        if region in regions:
            names = ", ".join(regions[region])
            lines.append(f"  {region.capitalize()}: {names}")

    lines.append("")

    # Volume per vertebra
    lines.append("VERTEBRA VOLUMES:")
    for v in spine:
        if v["region"] in ("cord", "disc", "merged"):
            continue
        vol = v["volume_mm3"] / 1000
        lines.append(f"  {v['name']:<8} {vol:6.1f} cm³")

    lines.append("")

    # Cord/disc info
    for v in spine:
        if v["region"] == "cord":
            vol = v["volume_mm3"] / 1000
            lines.append(f"Spinal Cord: {vol:.1f} cm³")
        if v["region"] == "disc":
            vol = v["volume_mm3"] / 1000
            lines.append(f"IVD (total): {vol:.1f} cm³")

    lines.append("")

    # Metrics
    if metrics:
        lines.append("SPINE CURVE METRICS:")
        lines.append(f"  Total length: {metrics['total_length_mm']:.1f} mm")
        lines.append(f"  Cobb angle (approx): "
                     f"{metrics['cobb_angle_approx_deg']:.1f}°")
        if metrics["distances"]:
            avg_dist = np.mean([d["distance_mm"]
                                for d in metrics["distances"]])
            lines.append(f"  Avg inter-vertebral dist: {avg_dist:.1f} mm")
        if metrics["angles"]:
            max_angle = max(a["angle_deg"] for a in metrics["angles"])
            max_vertex = [a["vertex"] for a in metrics["angles"]
                          if a["angle_deg"] == max_angle][0]
            lines.append(f"  Max angulation: {max_angle:.1f}° at {max_vertex}")

    # Render text
    text = "\n".join(lines)
    ax_metrics.text(0.05, 0.95, text, transform=ax_metrics.transAxes,
                    fontsize=8, fontfamily="monospace", verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                              edgecolor="#ddd"))

    # Region legend
    legend_y = 0.05
    for region, color in REGION_COLORS.items():
        if region in ("merged",):
            continue
        ax_metrics.add_patch(plt.Rectangle(
            (0.05, legend_y), 0.04, 0.025, facecolor=color, edgecolor="k",
            linewidth=0.3, transform=ax_metrics.transAxes))
        ax_metrics.text(0.11, legend_y + 0.012, region.capitalize(),
                        fontsize=7, transform=ax_metrics.transAxes,
                        va="center")
        legend_y += 0.035

    fig.subplots_adjust(top=0.93)

    # Save
    if output_path is None:
        output_path = str(Path(passport_path).with_suffix("")) + "_spine.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Spine visualization saved to: {output_path}")

    # Also print metrics to console
    if metrics:
        print(f"\nSpine Metrics:")
        print(f"  Vertebrae: {len(metrics['names'])} "
              f"({metrics['names'][0]} → {metrics['names'][-1]})")
        print(f"  Total length: {metrics['total_length_mm']:.1f} mm")
        print(f"  Cobb angle (approx): "
              f"{metrics['cobb_angle_approx_deg']:.1f}°")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize spine structures from anatomical passport"
    )
    parser.add_argument(
        "passport", type=str, help="Path to passport JSON file"
    )
    parser.add_argument(
        "--modality", type=str, choices=["ct", "mri"], default="ct",
        help="Imaging modality (default: ct)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output image path (default: <input>_spine.png)"
    )

    args = parser.parse_args()
    visualize_spine(args.passport, modality=args.modality,
                    output_path=args.output)


if __name__ == "__main__":
    main()
