#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import nibabel as nib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_dice(seg1, seg2, labels):
    """
    Calculate Dice score for each label in the segmentation masks.
    """
    dice_scores = {}
    for label in labels:
        if label == 0: continue
        mask1 = (seg1 == label)
        mask2 = (seg2 == label)
        intersection = np.logical_and(mask1, mask2).sum()
        sum_masks = mask1.sum() + mask2.sum()
        if sum_masks == 0:
            dice_scores[label] = 1.0
        else:
            dice_scores[label] = 2.0 * intersection / sum_masks
    return dice_scores

def compare():
    # Define paths
    vanilla_path = Path("benchmarks/vanilla_ground_truth.json")
    optimized_path = Path("benchmarks/inference_and_passport_results/benchmark_results.json")
    
    if not vanilla_path.exists():
        print(f"Error: Vanilla results not found at {vanilla_path}")
        return
    if not optimized_path.exists():
        print(f"Error: Optimized results not found at {optimized_path}")
        return

    # Load results
    with open(vanilla_path, "r") as f:
        vanilla_data = json.load(f)
    with open(optimized_path, "r") as f:
        optimized_data = json.load(f)

    # Filter out outliers if necessary (e.g. initial compilation runs)
    # vanilla_data = [v for v in vanilla_data if v["latency_sec"] < 1000]
    
    # Create dictionary for optimized results for O(1) lookup
    # Only consider successful runs
    opt_dict = {o["subject"]: o for o in optimized_data if "error" not in o and "latency_total_sec" in o}
    
    comparisons = []
    
    # ... (previous code calculating comparisons) ...

    # Prepare report content
    report_lines = []
    report_lines.append("# Detailed Vanilla vs Optimized Comparison Report")
    report_lines.append(f"\n**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Vanilla Source**: `{vanilla_path}`")
    report_lines.append(f"**Optimized Source**: `{optimized_path}`\n")
    
    report_lines.append("| Subject | Shape | Vanilla Latency | Optimized Latency | Speedup | V-Organs | O-Organs | Org-Diff | Dice (Mean) | Org-Details |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    speedups = [] # Fix: Re-initialize speedups list
    all_dice = [] # Fix: Re-initialize all_dice list
    
    # Iterate through vanilla results and find matches
    for v in vanilla_data:
        subj = v["subject"]
        if subj in opt_dict:
            o = opt_dict[subj]
            
            # Latency and Speedup
            v_lat = v.get("latency_sec", 0)
            o_lat = o.get("latency_total_sec", 0)
            
            if o_lat == 0:
                continue
                
            speedup = v_lat / o_lat
            speedups.append(speedup)
            
            # Organs
            v_org = v.get("num_organs", 0)
            o_org = o.get("num_organs", 0)
            org_diff = o_org - v_org
            
            # Detailed Organ Match
            v_organs_set = set(v.get("organs_present", []))
            o_organs_set = set(o.get("organs_present", []))
            
            missing_in_optimized = v_organs_set - o_organs_set
            extra_in_optimized = o_organs_set - v_organs_set
            
            org_details = ""
            if missing_in_optimized:
                org_details += f"Missing: {list(missing_in_optimized)} "
            if extra_in_optimized:
                org_details += f"Extra: {list(extra_in_optimized)}"
            
            # Throughput (MVox/s)
            shape = v.get("shape", [0,0,0])
            voxels = np.prod(shape)
            v_tp = voxels / v_lat / 1e6 if v_lat > 0 else 0
            o_tp = voxels / o_lat / 1e6 if o_lat > 0 else 0
            
            # Dice Calculation
            dice_mean = 0.0
            dice_str = "N/A"
            
            try:
                v_mask_path = Path(v.get("mask_path", ""))
                o_mask_path = Path(o.get("mask_path", ""))
                
                if v_mask_path.exists() and o_mask_path.exists():
                    v_img = nib.load(str(v_mask_path))
                    o_img = nib.load(str(o_mask_path))
                    v_mask = v_img.get_fdata().astype(np.uint8)
                    o_mask = o_img.get_fdata().astype(np.uint8)
                    
                    if v_mask.shape != o_mask.shape:
                        dice_str = f"ShapeMismatch"
                    else:
                        labels = set(np.unique(v_mask)).union(set(np.unique(o_mask)))
                        dice_vals = calculate_dice(v_mask, o_mask, labels)
                        if dice_vals:
                            dice_mean = np.mean(list(dice_vals.values()))
                            all_dice.append(dice_mean)
                            dice_str = f"{dice_mean:.4f}"
                        else:
                            dice_str = "0.0000" # No foreground
                else:
                    dice_str = "MissingFile"
            except Exception as e:
                dice_str = "Error"
                logger.error(f"Error calculating dice for {subj}: {e}")
                
                
            line = f"| {subj} | {shape} | {v_lat:.2f}s | {o_lat:.2f}s | {speedup:.2f}x | {v_org} | {o_org} | {org_diff:+} | {dice_str} | {org_details} |"
            report_lines.append(line)
            
            comparisons.append({
                "subject": subj,
                "speedup": speedup,
                "dice": dice_mean if dice_mean > 0 else None,
                "org_diff": org_diff
            })

    # Summary Statistics
    if comparisons:
        report_lines.append(f"\n## Summary Statistics (N={len(comparisons)})")
        report_lines.append(f"- **Average Speedup**: **{np.mean(speedups):.2f}x**")
        
        valid_dice = [c["dice"] for c in comparisons if c["dice"] is not None]
        if valid_dice:
            report_lines.append(f"- **Mean Dice Score**: **{np.mean(valid_dice):.4f}**")
            report_lines.append(f"- **Median Dice Score**: **{np.median(valid_dice):.4f}**")
        
        report_lines.append(f"- **Avg Organ Count Diff**: **{np.mean([c['org_diff'] for c in comparisons]):.2f}**")
        
        # Highlight Outliers
        report_lines.append("\n## Notable Outliers")
        if valid_dice:
            # Sort by dice ascending
            sorted_dice = sorted([c for c in comparisons if c["dice"] is not None], key=lambda x: x["dice"])
            if sorted_dice:
                report_lines.append(f"- **Lowest Agreement**: {sorted_dice[0]['subject']} (Dice: {sorted_dice[0]['dice']:.4f})")
                report_lines.append(f"- **Highest Agreement**: {sorted_dice[-1]['subject']} (Dice: {sorted_dice[-1]['dice']:.4f})")
        
        sorted_speed = sorted(comparisons, key=lambda x: x["speedup"], reverse=True)
        if sorted_speed:
            report_lines.append(f"- **Fastest Speedup**: {sorted_speed[0]['subject']} ({sorted_speed[0]['speedup']:.2f}x speedup)")
    else:
        report_lines.append("\nNo matching subjects found between Vanilla and Optimized results.")

    # Print and Save
    report_content = "\n".join(report_lines)
    print(report_content)
    
    output_file = Path("benchmarks/VANILLA_VS_OPTIMIZED_REPORT.md")
    with open(output_file, "w") as f:
        f.write(report_content)
    print(f"\n✅ Report saved to: {output_file}")

if __name__ == "__main__":
    import time
    compare()
