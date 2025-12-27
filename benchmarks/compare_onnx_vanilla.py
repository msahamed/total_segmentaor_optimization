import json
import numpy as np
from pathlib import Path
import nibabel as nib

def calculate_dice(seg1, seg2, labels):
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
    vanilla_path = Path("benchmarks/vanilla_ground_truth.json")
    onnx_path = Path("benchmarks/onnx_ground_truth.json")
    
    if not vanilla_path.exists() or not onnx_path.exists():
        print("Required JSON files not found.")
        return

    with open(vanilla_path, "r") as f:
        vanilla_data = json.load(f)
    with open(onnx_path, "r") as f:
        onnx_data = json.load(f)

    # Filter out sleep outliers from vanilla if any (already handled in analysis, but let's be safe)
    vanilla_data = [v for v in vanilla_data if v["latency_sec"] < 1000]
    
    # Match by subject
    onnx_dict = {o["subject"]: o for o in onnx_data}
    
    comparisons = []
    
    print("# Detailed ONNX vs Vanilla Comparison Report")
    print("\n| Subject | Shape | Vanilla Latency | ONNX Latency | Speedup | V-Organs | O-Organs | Org-Diff | Dice (Mean) | V-Throughput | O-Throughput |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    all_dice = []
    speedups = []
    
    for v in vanilla_data:
        subj = v["subject"]
        if subj in onnx_dict:
            o = onnx_dict[subj]
            
            # Latency and Speedup
            v_lat = v["latency_sec"]
            o_lat = o["latency_sec"]
            speedup = v_lat / o_lat
            speedups.append(speedup)
            
            # Organs
            v_org = v["num_organs"]
            o_org = o["num_organs"]
            org_diff = o_org - v_org
            
            # Throughput (MVox/s)
            voxels = np.prod(v["shape"])
            v_tp = voxels / v_lat / 1e6
            o_tp = voxels / o_lat / 1e6
            
            # Dice Calculation (For all 20 cases now)
            dice_mean = 0.0
            try:
                v_img = nib.load(v["mask_path"])
                o_img = nib.load(o["mask_path"])
                v_mask = v_img.get_fdata().astype(np.uint8)
                o_mask = o_img.get_fdata().astype(np.uint8)
                
                labels = set(np.unique(v_mask)).union(set(np.unique(o_mask)))
                dice_vals = calculate_dice(v_mask, o_mask, labels)
                dice_mean = np.mean(list(dice_vals.values()))
                all_dice.append(dice_mean)
                dice_str = f"{dice_mean:.4f}"
            except Exception as e:
                dice_str = "Error"
                
            print(f"| {subj} | {v['shape']} | {v_lat:.2f}s | {o_lat:.2f}s | {speedup:.2f}x | {v_org} | {o_org} | {org_diff:+} | {dice_str} | {v_tp:.2f} | {o_tp:.2f} |")
            
            comparisons.append({
                "subject": subj,
                "speedup": speedup,
                "dice": dice_mean,
                "org_diff": org_diff
            })

    print(f"\n## Summary Statistics (N={len(comparisons)})")
    print(f"- **Average Speedup**: **{np.mean(speedups):.2f}x**")
    if all_dice:
        print(f"- **Mean Dice Score**: **{np.mean(all_dice):.4f}**")
        print(f"- **Median Dice Score**: **{np.median(all_dice):.4f}**")
    print(f"- **Avg Organ Count Diff**: **{np.mean([c['org_diff'] for c in comparisons]):.2f}**")
    
    # Highlight Outliers
    print("\n## Notable Outliers")
    sorted_dice = sorted(comparisons, key=lambda x: x["dice"] if x["dice"] is not None else -1)
    print(f"- **Lowest Agreement**: {sorted_dice[0]['subject']} (Dice: {sorted_dice[0]['dice']:.4f})")
    print(f"- **Highest Agreement**: {sorted_dice[-1]['subject']} (Dice: {sorted_dice[-1]['dice']:.4f})")
    
    sorted_speed = sorted(comparisons, key=lambda x: x["speedup"], reverse=True)
    print(f"- **Fastest ONNX Run**: {sorted_speed[0]['subject']} ({sorted_speed[0]['speedup']:.2f}x speedup)")

if __name__ == "__main__":
    compare()

