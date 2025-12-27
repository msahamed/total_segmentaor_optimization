#!/usr/bin/env python3
import os
import time
import json
import logging
from pathlib import Path
import nibabel as nib
import numpy as np
import onnxruntime as ort
from totalsegmentator.resampling import change_spacing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_datasets(base_dir):
    datasets = []
    # 1. Root ct_data
    for f in base_dir.glob("*.nii.gz"):
        datasets.append(("Root", f))
        if len(datasets) >= 20: break
    # 2. learn2reg/scans
    if len(datasets) < 20:
        l2r_dir = base_dir / "learn2reg" / "scans"
        if l2r_dir.exists():
            for f in sorted(l2r_dir.glob("*.nii.gz")):
                datasets.append(("Learn2Reg", f))
                if len(datasets) >= 20: break
    return datasets



def postprocess(prediction, img_resampled, img_canonical, original_img):
    # prediction shape: [1, 118, Z, Y, X]
    # 1. Argmax
    seg_data = np.argmax(prediction[0], axis=0).astype(np.uint8) # (Z, Y, X)
    
    # 2. Transpose back to (X, Y, Z)
    seg_data = seg_data.transpose(2, 1, 0)
    
    # 3. Create nibabel image for resampled segmentation
    seg_img_resampled = nib.Nifti1Image(seg_data, img_resampled.affine)
    
    # 4. Resample back to canonical spacing (nearest neighbor)
    seg_img_canonical = change_spacing(seg_img_resampled, 
                                       new_spacing=img_canonical.header.get_zooms(), 
                                       target_shape=img_canonical.shape, 
                                       order=0)

    # 5. Reorient back to original orientation
    orig_axcodes = nib.aff2axcodes(original_img.affine)
    canon_axcodes = nib.aff2axcodes(img_canonical.affine)
    
    if orig_axcodes != canon_axcodes:
        orig_ornt = nib.orientations.axcodes2ornt(orig_axcodes)
        canon_ornt = nib.orientations.axcodes2ornt(canon_axcodes)
        inv_transform = nib.orientations.ornt_transform(canon_ornt, orig_ornt)
        
        seg_data_canonical = seg_img_canonical.get_fdata().astype(np.uint8)
        seg_data_original = nib.orientations.apply_orientation(seg_data_canonical, inv_transform)
        seg_img_final = nib.Nifti1Image(seg_data_original, original_img.affine, header=original_img.header)
    else:
        seg_img_final = nib.Nifti1Image(seg_img_canonical.get_fdata().astype(np.uint8), original_img.affine, header=original_img.header)
    
    return seg_img_final

def run_onnx_benchmarks():
    model_path = "models/totalsegmentator_total_fast_fp32.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    # Load ONNX session (CPU)
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    base_dir = Path("ct_data")
    datasets = find_datasets(base_dir)

    results = []
    output_dir = Path("benchmarks/onnx_benchmark_results")
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ESTABLISHING ONNX BASELINE (PYTHON + ORT CPU)")
    print("=" * 70)

    for i, (category, img_path) in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] Processing {category}: {img_path.name}")
        
        try:
            # 1. Preprocess
            start_time = time.time()
            original_img = nib.load(str(img_path))
            
            # Canonical orient
            img_canonical = nib.as_closest_canonical(original_img)
            # Resample to 3mm
            img_resampled = change_spacing(img_canonical, new_spacing=3.0, order=1)
            
            # Norm
            data = img_resampled.get_fdata().astype(np.float32)
            data = np.clip(data, -1004.0, 1588.0)
            data = (data - (-50.3869)) / 503.3923
            
            # (X, Y, Z) -> (Z, Y, X) and add dims
            input_data = data.transpose(2, 1, 0)[np.newaxis, np.newaxis, :, :, :]
            original_shape_resampled = input_data.shape[2:]
            
            # Pad to multiple of 32 to avoid ONNX Concat errors
            m = 32

            pad_z = (m - original_shape_resampled[0] % m) % m
            pad_y = (m - original_shape_resampled[1] % m) % m
            pad_x = (m - original_shape_resampled[2] % m) % m
            
            if pad_z > 0 or pad_y > 0 or pad_x > 0:
                input_data = np.pad(input_data, ((0,0), (0,0), (0, pad_z), (0, pad_y), (0, pad_x)), 
                                    mode='constant', constant_values=input_data.min())
            
            pre_time = time.time() - start_time
            
            # 2. Inference
            inf_start = time.time()
            outputs = session.run(None, {input_name: input_data})
            inf_time = time.time() - inf_start
            
            # 3. Postprocess
            post_start = time.time()
            prediction = outputs[0]
            # Crop back to original resampled shape
            prediction = prediction[:, :, :original_shape_resampled[0], :original_shape_resampled[1], :original_shape_resampled[2]]
            
            seg_img_final = postprocess(prediction, img_resampled, img_canonical, original_img)
            post_time = time.time() - post_start


            
            total_time = time.time() - start_time
            
            # Metrics
            seg_data = seg_img_final.get_fdata().astype(np.uint8)
            unique_labels = np.unique(seg_data)
            organs = [int(l) for l in unique_labels if l > 0]
            num_organs = len(organs)
            
            # Save mask
            mask_path = mask_dir / f"{img_path.stem}_onnx_mask.nii.gz"
            nib.save(seg_img_final, str(mask_path))
            
            meta = {
                "subject": img_path.name,
                "category": category,
                "latency_sec": round(total_time, 2),
                "breakdown": {
                    "preprocess": round(pre_time, 2),
                    "inference": round(inf_time, 2),
                    "postprocess": round(post_time, 2)
                },
                "num_organs": num_organs,
                "organs_present": organs,
                "mask_path": str(mask_path)
            }
            results.append(meta)
            
            print(f"  Latency: {total_time:.2f}s (Inf: {inf_time:.2f}s)")
            print(f"  Organs Detected: {num_organs}")

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    # Save results
    with open("benchmarks/onnx_ground_truth.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETE. Saved to 'benchmarks/onnx_ground_truth.json'")
    print("=" * 70)

if __name__ == "__main__":
    run_onnx_benchmarks()
