#!/usr/bin/env python3
import os
import time
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import onnxruntime as ort
from totalsegmentator.resampling import change_spacing

def run_linux_benchmark():
    model_path = "models/totalsegmentator_total_fast_fp32.onnx"
    img_path = "ct_data/learn2reg/scans/case_001_exp.nii.gz" # Use a standard case
    
    # Session options optimized for Linux/AMD
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # On Linux, setting this to 0 (auto) or a fixed number of physical cores 
    # works consistently.
    so.intra_op_num_threads = 0 
    
    session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Pre-load image to exclude Disk I/O from the "Logic" benchmark
    img = nib.load(img_path)
    
    print(f"--- Linux Production Benchmark (AMD) ---")
    
    # Run 5 iterations to show consistency
    for i in range(5):
        start = time.time()
        
        # 1. Preprocess (Order=1 for speed/accuracy balance)
        img_canonical = nib.as_closest_canonical(img)
        img_resampled = change_spacing(img_canonical, new_spacing=3.0, order=1)
        data = img_resampled.get_fdata().astype(np.float32)
        data = np.clip(data, -1004.0, 1588.0)
        data = (data - (-50.3869)) / 503.3923
        input_data = data.transpose(2, 1, 0)[np.newaxis, np.newaxis, :, :, :]
        
        m = 32
        p = [(0,0), (0,0), 
             (0, (m - input_data.shape[2] % m) % m), 
             (0, (m - input_data.shape[3] % m) % m), 
             (0, (m - input_data.shape[4] % m) % m)]
        input_data = np.pad(input_data, p, mode='constant')
        
        # 2. Inference
        inf_start = time.time()
        session.run(None, {input_name: input_data})
        inf_time = time.time() - inf_start
        
        total = time.time() - start
        print(f"Iteration {i+1}: Total={total:.2f}s (Inf={inf_time:.2f}s)")

if __name__ == "__main__":
    run_linux_benchmark()
