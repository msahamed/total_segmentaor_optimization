#!/usr/bin/env python3
"""
TotalSegmentator: Optimized Inference + Passport Extraction Benchmark
======================================================================

This script combines the optimized Python-ONNX inference pipeline with
anatomical passport extraction to measure the TOTAL latency for registration
preprocessing.

Purpose:
- Measure how much latency passport extraction adds to the inference pipeline
- Determine if Rust optimization is worth pursuing for passport extraction
- Provide baseline for future optimization work

Key Optimization:
- Passport is extracted from 3mm resampled volume (10-50× faster than original resolution)
- 3mm resolution provides sufficient precision for registration landmarks
- Matches the approach used in production registration pipeline

Outputs:
- Segmentation masks (original resolution for dice score validation)
- Anatomical passports (JSON files with centroids, eigenvectors, boundary points)
- Detailed timing breakdown (preprocessing, inference, passport, postprocessing)
- Benchmark results JSON

Author: Optimized for production deployment
Date: 2026-01-12 (Updated to use 3mm passport extraction)
"""

import os
import sys
import time
import json
import logging
import multiprocessing
from pathlib import Path
import nibabel as nib
import numpy as np
import onnxruntime as ort
from scipy import ndimage
from totalsegmentator.resampling import change_spacing

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_datasets(base_dir, max_samples=20):
    """
    Find CT scan datasets from multiple directories.
    Priority: Root ct_data -> learn2reg/scans
    """
    datasets = []

    # 1. Root ct_data
    for f in base_dir.glob("*.nii.gz"):
        datasets.append(("Root", f))
        if len(datasets) >= max_samples:
            break

    # 2. learn2reg/scans
    if len(datasets) < max_samples:
        l2r_dir = base_dir / "learn2reg" / "scans"
        if l2r_dir.exists():
            for f in sorted(l2r_dir.glob("*.nii.gz")):
                datasets.append(("Learn2Reg", f))
                if len(datasets) >= max_samples:
                    break

    logger.info(f"Found {len(datasets)} CT scans for benchmarking")
    return datasets


def postprocess(prediction, img_resampled, img_canonical, original_img):
    """
    Postprocess segmentation: argmax, transpose, resample back to original resolution.

    This maintains compatibility with the original TotalSegmentator output format.
    """
    # 1. Argmax to get class labels
    seg_data = np.argmax(prediction[0], axis=0).astype(np.uint8)  # (Z, Y, X)

    # 2. Transpose back to (X, Y, Z) for nibabel convention
    seg_data = seg_data.transpose(2, 1, 0)

    # 3. Create nibabel image for resampled segmentation
    seg_img_resampled = nib.Nifti1Image(seg_data, img_resampled.affine)

    # 4. Resample back to canonical spacing (nearest neighbor for labels)
    seg_img_canonical = change_spacing(
        seg_img_resampled,
        new_spacing=img_canonical.header.get_zooms(),
        target_shape=img_canonical.shape,
        order=0  # Nearest neighbor
    )

    # 5. Reorient back to original orientation
    orig_axcodes = nib.aff2axcodes(original_img.affine)
    canon_axcodes = nib.aff2axcodes(img_canonical.affine)

    if orig_axcodes != canon_axcodes:
        orig_ornt = nib.orientations.axcodes2ornt(orig_axcodes)
        canon_ornt = nib.orientations.axcodes2ornt(canon_axcodes)
        inv_transform = nib.orientations.ornt_transform(canon_ornt, orig_ornt)

        seg_data_canonical = seg_img_canonical.get_fdata().astype(np.uint8)
        seg_data_original = nib.orientations.apply_orientation(
            seg_data_canonical, inv_transform
        )
        seg_img_final = nib.Nifti1Image(
            seg_data_original, original_img.affine, header=original_img.header
        )
    else:
        seg_img_final = nib.Nifti1Image(
            seg_img_canonical.get_fdata().astype(np.uint8),
            original_img.affine,
            header=original_img.header
        )

    return seg_img_final


def extract_passport(mask_data, affine, decimation_target=500):
    """
    Extract anatomical passport from segmentation mask.

    For each organ label, extracts:
    - Centroid (mean world coordinates)
    - Covariance matrix (spatial distribution)
    - Eigenvalues & Eigenvectors (principal axes for orientation)
    - Boundary points (organ surface, decimated to ~500 points)
    - Volume (mm³)

    Args:
        mask_data: 3D numpy array with organ labels
        affine: 4x4 affine matrix mapping voxels to world coordinates
        decimation_target: Target number of boundary points per organ

    Returns:
        dict: Passport data structure {label: {centroid, covariance, ...}}
    """
    passport = {}
    labels = np.unique(mask_data)
    labels = labels[labels > 0]  # Skip background

    logger.debug(f"Extracting passport for {len(labels)} organs...")

    for label in labels:
        label_mask = (mask_data == label)
        indices = np.argwhere(label_mask)

        if len(indices) == 0:
            continue

        # Map voxel coordinates to world coordinates
        ones = np.ones((indices.shape[0], 1))
        coords_vox = np.hstack([indices, ones])
        coords_world = (affine @ coords_vox.T).T[:, :3]

        # Centroid
        centroid = np.mean(coords_world, axis=0)

        # Covariance matrix and principal axes
        centered = coords_world - centroid
        cov = (centered.T @ centered) / len(centered)
        evals, evecs = np.linalg.eigh(cov)

        # Boundary extraction using binary erosion
        eroded = ndimage.binary_erosion(label_mask)
        boundary_vox = np.argwhere(label_mask ^ eroded)

        # Decimate boundary points to reduce storage/computation
        if len(boundary_vox) > decimation_target:
            idx = np.random.choice(len(boundary_vox), decimation_target, replace=False)
            boundary_vox = boundary_vox[idx]

        # Map boundary to world coordinates
        if len(boundary_vox) > 0:
            b_ones = np.ones((boundary_vox.shape[0], 1))
            b_coords_world = (affine @ np.hstack([boundary_vox, b_ones]).T).T[:, :3]
            boundary_points = b_coords_world.tolist()
        else:
            boundary_points = []

        # Volume in mm³
        volume_mm3 = float(np.sum(label_mask) * np.abs(np.linalg.det(affine[:3, :3])))

        passport[int(label)] = {
            "centroid": centroid.tolist(),
            "covariance": cov.tolist(),
            "eigenvalues": evals.tolist(),
            "eigenvectors": evecs.tolist(),
            "boundary_points": boundary_points,
            "volume_mm3": volume_mm3
        }

    return passport


def run_inference_and_passport_benchmark(max_samples=20):
    """
    Run complete benchmark: Inference + Passport Extraction.

    This measures the TOTAL latency for registration preprocessing:
    1. Preprocessing (resampling, normalization, padding)
    2. ONNX Inference (model forward pass)
    3. Postprocessing (argmax, resample back to original resolution)
    4. Passport Extraction (centroids, eigenvectors, boundary points)
    """
    model_path = "models/totalsegmentator_total_fast_fp32.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    # ===== ONNX Session Setup (Optimized) =====
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Thread optimization for CPU
    # Using 4 threads is 15-20% faster than using all cores (0)
    # Reduces thread contention and cache thrashing
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Prioritize CUDA if available
    available_providers = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
        logger.info("GPU (CUDA) detected - using GPU acceleration")
    providers.append('CPUExecutionProvider')

    logger.info(f"ONNX Runtime providers: {providers}")
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name

    # ===== Find datasets =====
    base_dir = Path("ct_data")
    datasets = find_datasets(base_dir, max_samples=max_samples)

    if len(datasets) == 0:
        logger.error("No datasets found. Please run 01_download_data.py first.")
        return

    # ===== Setup output directories =====
    results = []
    output_dir = Path("benchmarks/inference_and_passport_results")
    mask_dir = output_dir / "masks"
    passport_dir = output_dir / "passports"
    mask_dir.mkdir(parents=True, exist_ok=True)
    passport_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INFERENCE + PASSPORT EXTRACTION BENCHMARK")
    print("=" * 80)
    print(f"Samples: {len(datasets)}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # ===== Process each dataset =====
    for i, (category, img_path) in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] Processing {category}: {img_path.name}")

        try:
            start_total = time.time()
            original_img = nib.load(str(img_path))

            # ----- 1. PREPROCESSING -----
            pre_start = time.time()
            img_canonical = nib.as_closest_canonical(original_img)

            # Use order=1 (Linear) for 30-50% speedup with minimal accuracy loss
            img_resampled = change_spacing(img_canonical, new_spacing=3.0, order=1)

            data = img_resampled.get_fdata().astype(np.float32)
            data = np.clip(data, -1004.0, 1588.0)
            data = (data - (-50.3869)) / 503.3923

            input_data = data.transpose(2, 1, 0)[np.newaxis, np.newaxis, :, :, :]
            original_shape_resampled = input_data.shape[2:]

            # Pad to multiple of 32 (model requirement)
            m = 32
            pad_z = (m - original_shape_resampled[0] % m) % m
            pad_y = (m - original_shape_resampled[1] % m) % m
            pad_x = (m - original_shape_resampled[2] % m) % m

            if pad_z > 0 or pad_y > 0 or pad_x > 0:
                input_data = np.pad(
                    input_data,
                    ((0, 0), (0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
                    mode='constant',
                    constant_values=input_data.min()
                )

            pre_time = time.time() - pre_start

            # ----- 2. INFERENCE -----
            inf_start = time.time()
            outputs = session.run(None, {input_name: input_data})
            inf_time = time.time() - inf_start

            # ----- 3. PASSPORT EXTRACTION (on 3mm resampled volume) -----
            # IMPORTANT: Extract passport from 3mm resampled segmentation
            # This is 10-50× faster than extracting from original resolution
            # and provides sufficient precision for registration landmarks
            passport_start = time.time()
            prediction = outputs[0]
            prediction = prediction[:, :, :original_shape_resampled[0],
                                    :original_shape_resampled[1],
                                    :original_shape_resampled[2]]

            # Get segmentation at 3mm resolution (argmax)
            seg_3mm = np.argmax(prediction[0], axis=0).astype(np.uint8)  # (Z, Y, X)
            seg_3mm = seg_3mm.transpose(2, 1, 0)  # (X, Y, Z) for nibabel

            # Extract passport from 3mm volume
            passport = extract_passport(seg_3mm, img_resampled.affine)
            passport_time = time.time() - passport_start

            # ----- 4. POSTPROCESSING (resample back to original resolution) -----
            # For dice score validation and clinical use
            post_start = time.time()
            seg_img_final = postprocess(
                prediction, img_resampled, img_canonical, original_img
            )
            post_time = time.time() - post_start

            total_time = time.time() - start_total

            # ----- 5. SAVE OUTPUTS -----
            # Save segmentation mask
            mask_path = mask_dir / f"{img_path.stem}_mask.nii.gz"
            nib.save(seg_img_final, str(mask_path))

            # Save passport
            passport_path = passport_dir / f"{img_path.stem}_passport.json"
            with open(passport_path, 'w') as f:
                json.dump(passport, f, indent=2)

            # ----- 6. RECORD RESULTS -----
            meta = {
                "subject": img_path.name,
                "category": category,
                "latency_total_sec": round(total_time, 3),
                "breakdown": {
                    "preprocessing": round(pre_time, 3),
                    "inference": round(inf_time, 3),
                    "postprocessing": round(post_time, 3),
                    "passport_extraction": round(passport_time, 3)
                },
                "num_organs": len(passport),
                "mask_path": str(mask_path),
                "passport_path": str(passport_path),
                "passport_size_kb": round(passport_path.stat().st_size / 1024, 2)
            }
            results.append(meta)

            # ----- 7. PRINT PROGRESS -----
            print(f"  Total: {total_time:.3f}s | Pre: {pre_time:.3f}s | "
                  f"Inf: {inf_time:.3f}s | Post: {post_time:.3f}s | "
                  f"Passport: {passport_time:.3f}s")
            print(f"  Organs detected: {len(passport)} | "
                  f"Passport size: {meta['passport_size_kb']:.2f} KB")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}", exc_info=True)
            results.append({
                "subject": img_path.name,
                "error": str(e)
            })

    # ===== SAVE BENCHMARK RESULTS =====
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # ===== COMPUTE SUMMARY STATISTICS =====
    successful_results = [r for r in results if "error" not in r]

    if successful_results:
        avg_total = np.mean([r["latency_total_sec"] for r in successful_results])
        avg_pre = np.mean([r["breakdown"]["preprocessing"] for r in successful_results])
        avg_inf = np.mean([r["breakdown"]["inference"] for r in successful_results])
        avg_post = np.mean([r["breakdown"]["postprocessing"] for r in successful_results])
        avg_passport = np.mean([r["breakdown"]["passport_extraction"] for r in successful_results])
        avg_organs = np.mean([r["num_organs"] for r in successful_results])

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Successful runs: {len(successful_results)}/{len(results)}")
        print(f"Average total latency: {avg_total:.3f}s")
        print(f"  - Preprocessing:        {avg_pre:.3f}s ({avg_pre/avg_total*100:.1f}%)")
        print(f"  - Inference:            {avg_inf:.3f}s ({avg_inf/avg_total*100:.1f}%)")
        print(f"  - Postprocessing:       {avg_post:.3f}s ({avg_post/avg_total*100:.1f}%)")
        print(f"  - Passport Extraction:  {avg_passport:.3f}s ({avg_passport/avg_total*100:.1f}%)")
        print(f"Average organs detected: {avg_organs:.1f}")
        print(f"\nPassport extraction overhead: {avg_passport:.3f}s "
              f"({avg_passport/avg_total*100:.1f}% of total pipeline)")
        print("=" * 80)
        print(f"\nResults saved to: {results_path}")

        # Additional insight
        if avg_passport > 0.5:
            print(f"\n⚠️  INSIGHT: Passport extraction takes {avg_passport:.3f}s")
            print("   Consider profiling to identify bottlenecks:")
            print("   - Binary erosion (likely culprit)")
            print("   - Coordinate transformations")
            print("   - Random sampling overhead")
        else:
            print(f"\n✅ INSIGHT: Passport extraction is fast ({avg_passport:.3f}s)")
            print("   Extracting from 3mm resampled volume is optimal.")
            print("   No further optimization needed for this component.")
    else:
        logger.error("All benchmarks failed. Check logs for details.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark TotalSegmentator inference + passport extraction"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of CT scans to process (default: 20)"
    )

    args = parser.parse_args()

    run_inference_and_passport_benchmark(max_samples=args.max_samples)
