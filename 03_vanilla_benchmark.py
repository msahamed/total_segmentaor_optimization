#!/usr/bin/env python3
import os
import time
import json
import logging
from pathlib import Path
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_datasets(base_dir, max_samples=20):
    datasets = []
    
    # 1. Root ct_data
    for f in base_dir.glob("*.nii.gz"):
        datasets.append(("Root", f))
        if len(datasets) >= max_samples:
            break
            
    # 2. learn2reg/scans (if more needed)
    if len(datasets) < max_samples:
        l2r_dir = base_dir / "learn2reg" / "scans"
        if l2r_dir.exists():
            for f in sorted(l2r_dir.glob("*.nii.gz")):
                datasets.append(("Learn2Reg", f))
                if len(datasets) >= max_samples:
                    break
            
    return datasets

def run_benchmarks(max_samples=20):
    # base_dir should be the root ct_data folder
    base_dir = Path("ct_data")
    datasets = find_datasets(base_dir, max_samples)
    
    if len(datasets) < max_samples:
        logger.warning(f"Only found {len(datasets)} datasets. Proceeding with available ones.")



    results = []
    output_dir = Path("benchmarks/vanilla_benchmark_results")
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ESTABLISHING VANILLA GROUND TRUTH (SAVING MASKS) | Max Samples: {max_samples}")
    print("=" * 70)

    for i, (category, img_path) in enumerate(datasets):
        if not img_path.exists():
            logger.warning(f"File not found: {img_path}")
            continue

        print(f"\n[{i+1}/{len(datasets)}] Processing {category}: {img_path.name}")
        
        # Load image once to get metadata
        img = nib.load(str(img_path))
        shape = img.shape
        zooms = [round(float(z), 3) for z in img.header.get_zooms()]
        
        start_time = time.time()
        
        try:
            # Run vanilla totalsegmentator (fast=True for 3mm)
            # Save the mask to the masks directory
            mask_path = mask_dir / f"{img_path.stem}_mask.nii.gz"
            seg_img = totalsegmentator(img_path, None, fast=True, ml=True, quiet=True)
            nib.save(seg_img, str(mask_path))
            
            end_time = time.time()
            latency = end_time - start_time
            
            seg_data = seg_img.get_fdata()
            unique_labels = np.unique(seg_data)
            # Remove background (0)
            organs = [int(l) for l in unique_labels if l > 0]
            num_organs = len(organs)
            
            # Record metrics
            meta = {
                "subject": img_path.name,
                "category": category,
                "path": str(img_path),
                "mask_path": str(mask_path),
                "shape": list(shape),
                "spacing": list(zooms),
                "latency_sec": round(latency, 2),
                "num_organs": num_organs,
                "organs_present": organs
            }
            results.append(meta)
            
            print(f"  Shape: {shape}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Organs Detected: {num_organs}")
            print(f"  Saved to: {mask_path.name}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    # Save final results in benchmarks folder
    with open("benchmarks/vanilla_ground_truth.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE. Results: 'benchmarks/vanilla_ground_truth.json'")
    print(f"Masks stored in: 'benchmarks/vanilla_benchmark_results/masks/'")
    print("=" * 70)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run vanilla TotalSegmentator benchmark")
    parser.add_argument("--max-samples", type=int, default=20, help="Maximum number of samples to process")
    args = parser.parse_args()
    
    run_benchmarks(max_samples=args.max_samples)
