# TotalSegmentator Inference + Passport Extraction - Testing Guide

**Script:** `06_inferenceAndPassport.py`
**Purpose:** Benchmark TotalSegmentator inference with anatomical passport extraction
**Modalities:** CT (117 classes) and MRI (50 classes)
**Target Users:** Evaluators, QA testers, researchers validating performance

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Running the Benchmark](#running-the-benchmark)
6. [Understanding the Results](#understanding-the-results)
7. [Validation Guidelines](#validation-guidelines)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

For evaluators who want to run immediately:

**CT:**
```bash
# 1. Clone the repository (models are already included!)
git clone <repository-url>
cd total_segmentator

# 2. Install dependencies
pip install numpy scipy nibabel onnxruntime totalsegmentator

# 3. Place your CT scans in ct_data/ directory
cp your_ct_scan.nii.gz ct_data/

# 4. Run the benchmark (default: 20 scans)
python 06_inferenceAndPassport.py --modality ct

# 5. Check results
cat benchmarks/inference_and_passport_results/benchmark_results.json
```

**MRI:**
```bash
# 1. Place your MRI scans in mri_data/ directory
cp your_mri_scan.nii.gz mri_data/

# 2. Run the benchmark
python 06_inferenceAndPassport.py --modality mri

# 3. Check results
cat benchmarks/inference_and_passport_results_mri/benchmark_results.json
```

**Note:** Both ONNX models are included in the repository — CT (66MB) and MRI (63MB).

---

## 💻 System Requirements

### Hardware Requirements

- **CPU:** Multi-core processor (4+ cores recommended)
- **RAM:** 8GB minimum, 16GB+ recommended for large CT volumes
- **Storage:** 10GB free space for test data and outputs

### Software Requirements

- **Python:** 3.8 or higher
- **Operating System:** Linux, macOS, or Windows (tested on macOS)

### Expected Performance

| Metric | CT (N=49) | MRI (N=5) |
|--------|-----------|-----------|
| **Average Latency** | ~12s per scan | ~11.2s per scan |
| **Range** | 2.6s - 24s | 8s - 15s |
| **Success Rate** | 100% | 100% |
| **Organs Detected** | 50-90 (117 classes) | 26-40 (50 classes) |

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
# Install required packages
pip install numpy scipy nibabel onnxruntime

# Install TotalSegmentator (for resampling utilities)
pip install totalsegmentator
```

**Key Dependencies:**
- `numpy`: Array operations
- `scipy`: Signal processing (resampling, binary erosion)
- `nibabel`: Medical image I/O (NIfTI format)
- `onnxruntime`: ONNX model inference
- `totalsegmentator`: Official library (for resampling functions)

### Step 2: ONNX Models (Included)

✅ **Both ONNX models are already included in this repository!**

| Model | Path | Classes | Size |
|-------|------|---------|------|
| **CT** | `models/totalsegmentator_total_fast_fp32.onnx` | 117 | 66MB |
| **MRI** | `models/totalsegmentator_total_mr_fast_fp32.onnx` | 50 | 63MB |

**Note:** If you need to re-export the models:
```bash
python 02_export_model.py --task total --fast      # CT model
python 02_export_model.py --task total_mr --fast    # MRI model
```

---

## 📁 Data Preparation

### Supported Data Format

The script accepts **NIfTI format** (`.nii.gz` or `.nii`) for both CT and MRI scans.

### Download Benchmark Dataset (Optional)

To replicate the exact N=49 comprehensive validation, download the complete benchmark dataset:

📥 **Google Drive:** [Download Benchmark Dataset](https://drive.google.com/file/d/1g_a08EMAG1NTAWM5m2JV4mPgCcSqDuJs/view?usp=sharing)

**Dataset Contents:**
- **MSD Spleen (Task09):** 5 subjects
- **Learn2Reg Challenge:** 42 subjects (21 paired exp/insp)
- **Custom samples:** 2 subjects

**Installation:**
```bash
# Download and extract the dataset
# Place contents in ct_data/ directory
unzip benchmark_dataset.zip -d ct_data/
```

**Note:** This is optional. You can use your own CT scans for testing.

### Directory Structure

```
total_segmentator/
├── ct_data/                          # CT scans
│   ├── your_scan_001.nii.gz
│   └── learn2reg/scans/             # Learn2Reg dataset (optional)
│       ├── case_001_exp.nii.gz
│       └── ...
│
└── mri_data/                         # MRI scans
    ├── your_mri_001.nii.gz
    └── ...
```

### Adding Your Custom Dataset

```bash
# CT scans
cp /path/to/your/ct_scans/*.nii.gz ct_data/

# MRI scans
cp /path/to/your/mri_scans/*.nii.gz mri_data/
```

### Data Requirements

✅ **Accepted:**
- NIfTI format (`.nii.gz` or `.nii`)
- CT scans (any body region) — use `--modality ct`
- MRI scans (abdominal) — use `--modality mri`
- Any resolution/spacing (will be resampled to 3mm)
- Any orientation (will be canonicalized)

❌ **Not Supported:**
- DICOM format (convert to NIfTI first)
- Corrupted or incomplete scans

### Data Conversion (if needed)

If you have DICOM files:

```bash
# Using dcm2niix (install via: brew install dcm2niix)
dcm2niix -o ct_data/ -f scan_%p /path/to/dicom/folder

# Using Python (simple-dicom-to-nifti)
pip install dicom2nifti
python -c "import dicom2nifti; dicom2nifti.convert_directory('/path/to/dicom', 'ct_data/')"
```

---

## 🏃 Running the Benchmark

### Basic Usage

```bash
# CT (default)
python 06_inferenceAndPassport.py --modality ct

# MRI
python 06_inferenceAndPassport.py --modality mri
```

### Advanced Options

```bash
# Process only 5 scans (quick test)
python 06_inferenceAndPassport.py --modality ct --max-samples 5

# Process all available scans
python 06_inferenceAndPassport.py --modality ct --max-samples 1000
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--modality` | str | ct | Imaging modality: `ct` or `mri` |
| `--max-samples` | int | 20 | Maximum number of scans to process |

### What Happens During Execution

The script will:

1. **Find scans** in `ct_data/` (CT) or `mri_data/` (MRI)
2. **For each scan:**
   - Preprocess (resample to 3mm, modality-specific normalization, pad)
   - Run ONNX inference (CT: 117 classes, MRI: 50 classes)
   - Postprocess (resample back to original resolution)
   - Extract anatomical passport (centroids, eigenvectors, boundaries)
   - Save outputs (mask + passport JSON)
3. **Generate summary** with performance statistics

### Expected Console Output

```
2026-01-13 08:06:41,245 - INFO - ONNX Runtime providers: ['CPUExecutionProvider']
2026-01-13 08:06:41,284 - INFO - Found 49 CT scans for benchmarking
================================================================================
INFERENCE + PASSPORT EXTRACTION BENCHMARK
================================================================================
Samples: 49
Model: models/totalsegmentator_total_fast_fp32.onnx
Output: benchmarks/inference_and_passport_results
================================================================================

[1/49] Processing Root: your_scan_001.nii.gz
  Total: 9.575s | Pre: 0.264s | Inf: 6.853s | Post: 0.802s | Passport: 1.654s
  Organs detected: 60 | Passport size: 1699.96 KB

[2/49] Processing Root: your_scan_002.nii.gz
  Total: 8.635s | Pre: 0.172s | Inf: 5.998s | Post: 0.276s | Passport: 1.187s
  Organs detected: 55 | Passport size: 1376.13 KB

...

================================================================================
BENCHMARK SUMMARY
================================================================================
Successful runs: 49/49
Average total latency: 9.968s
  - Preprocessing:        0.227s (2.3%)
  - Inference:            6.768s (67.9%)
  - Postprocessing:       0.864s (8.7%)
  - Passport Extraction:  2.105s (21.1%)
Average organs detected: 68.7

Passport extraction overhead: 2.105s (21.1% of total pipeline)
================================================================================

Results saved to: benchmarks/inference_and_passport_results/benchmark_results.json
```

---

## 📊 Understanding the Results

### Output Files

After running the benchmark, you'll find:

```
# CT outputs
benchmarks/inference_and_passport_results/
├── benchmark_results.json
├── masks/
│   └── your_scan_001_mask.nii.gz
└── passports/
    └── your_scan_001_passport.json

# MRI outputs
benchmarks/inference_and_passport_results_mri/
├── benchmark_results.json
├── masks/
│   └── your_mri_001_mask.nii.gz
└── passports/
    └── your_mri_001_passport.json
```

### Benchmark Results JSON

**Location:** `benchmarks/inference_and_passport_results/benchmark_results.json` (CT) or `benchmarks/inference_and_passport_results_mri/benchmark_results.json` (MRI)

**Structure:**

```json
[
  {
    "subject": "your_scan_001.nii.gz",
    "latency_total_sec": 9.575,
    "breakdown": {
      "preprocessing": 0.264,
      "inference": 6.853,
      "postprocessing": 0.802,
      "passport_extraction": 1.654
    },
    "num_organs": 60,
    "passport_size_kb": 1699.96,
    "mask_path": "benchmarks/.../masks/your_scan_001_mask.nii.gz",
    "passport_path": "benchmarks/.../passports/your_scan_001_passport.json"
  },
  ...
]
```

### Performance Metrics Explained

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| **latency_total_sec** | Total processing time (sec) | 2.6s - 24s |
| **preprocessing** | Resampling, normalization, padding | 0.1s - 0.7s |
| **inference** | ONNX model forward pass | 2.0s - 16s |
| **postprocessing** | Resample back to original resolution | 0.3s - 2.1s |
| **passport_extraction** | Extract landmarks from 3mm volume | 0.2s - 7.1s |
| **num_organs** | Number of detected anatomical structures | 13 - 88 |
| **passport_size_kb** | Size of passport JSON file | 376 KB - 3200 KB |

### Anatomical Passport Format

**Location:** `benchmarks/inference_and_passport_results/passports/*.json`

**Structure:**

```json
{
  "1": {  // Organ label (e.g., 1 = spleen)
    "centroid": [x, y, z],              // 3D center in world coordinates (mm)
    "covariance": [[3x3 matrix]],       // Spatial distribution
    "eigenvalues": [λ1, λ2, λ3],        // Principal component magnitudes
    "eigenvectors": [[3x3 matrix]],     // Principal axes (orientation)
    "boundary_points": [[x,y,z], ...],  // ~500 surface points per organ
    "volume_mm3": 12345.67              // Organ volume
  },
  "2": { ... },  // Liver
  ...
}
```

**Use Cases:**
- Registration preprocessing (rigid + deformable alignment)
- Anatomical feature extraction
- Organ volume analysis
- Shape-based matching

---

## ✅ Validation Guidelines

### Performance Targets

**CT (Based on N=49 Validation):**

| Target | Threshold | Expected |
|--------|-----------|----------|
| **Mean Latency** | < 15s | ~12s |
| **Success Rate** | > 95% | 100% |
| **Organs Detected** | 50-90 | ~69 avg |
| **Dice vs Vanilla** | > 0.85 | 0.91 |

**MRI (Based on N=5 Validation):**

| Target | Threshold | Expected |
|--------|-----------|----------|
| **Mean Latency** | < 15s | ~11.2s |
| **Success Rate** | > 95% | 100% |
| **Organs Detected** | 20-40 | 26-40 |

### Expected Performance by Scan Type

| Scan Type | Typical Latency | Organ Count |
|-----------|----------------|-------------|
| **CT - Small volumes** (limited FOV) | 2-5s | 10-20 |
| **CT - Thoracic (expiration)** | 7-10s | 50-60 |
| **CT - Thoracic (inspiration)** | 8-11s | 80-90 |
| **CT - Abdominal** | 9-15s | 60-75 |
| **CT - Large whole-body** | 15-24s | 70-85 |
| **MRI - Abdominal** | 8-15s | 26-40 |

### Quality Checks

#### 1. Latency Validation

```bash
# Extract average latency from results
cat benchmarks/inference_and_passport_results/benchmark_results.json | \
  python -c "import json, sys; data=json.load(sys.stdin); print(f'Average: {sum(r[\"latency_total_sec\"] for r in data)/len(data):.3f}s')"
```

**Expected:** ~10 seconds average

#### 2. Success Rate Validation

```bash
# Check for errors
cat benchmarks/inference_and_passport_results/benchmark_results.json | \
  python -c "import json, sys; data=json.load(sys.stdin); errors=[r for r in data if 'error' in r]; print(f'Success: {len(data)-len(errors)}/{len(data)}')"
```

**Expected:** 100% success (no errors)

#### 3. Organ Detection Validation

```bash
# Extract organ count statistics
cat benchmarks/inference_and_passport_results/benchmark_results.json | \
  python -c "import json, sys; import numpy as np; data=json.load(sys.stdin); counts=[r['num_organs'] for r in data]; print(f'Organs: {np.mean(counts):.1f} ± {np.std(counts):.1f}')"
```

**Expected:** 50-90 organs depending on scan type

#### 4. Visual Inspection (Optional)

```bash
# Open segmentation mask in your NIfTI viewer
# ITK-SNAP, 3D Slicer, or FSLeyes
itksnap -g ct_data/your_scan.nii.gz \
        -s benchmarks/inference_and_passport_results/masks/your_scan_mask.nii.gz
```

**Check:**
- Organs are correctly segmented
- No major misalignments
- Labels match anatomical structures

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: "No CT scans found"

**Error:**
```
INFO - Found 0 CT scans for benchmarking
```

**Solution:**
- Verify files exist: `ls ct_data/*.nii.gz`
- Check file permissions: `chmod 644 ct_data/*.nii.gz`
- Ensure files are NIfTI format (`.nii.gz` or `.nii`)

---

#### Issue 2: "Model file not found"

**Error:**
```
FileNotFoundError: models/totalsegmentator_total_fast_fp32.onnx
```

**Solution:**
```bash
# Export the model first
python 02_export_model.py

# Verify model exists
ls -lh models/totalsegmentator_total_fast_fp32.onnx
```

---

#### Issue 3: "Out of memory"

**Error:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Close other applications to free RAM
- Process fewer scans at once: `--max-samples 5`
- Use a machine with more RAM (16GB+ recommended)

---

#### Issue 4: Slow performance (>15s average)

**Possible Causes:**
1. **High system load:** Close background applications
2. **Disk I/O bottleneck:** Use SSD instead of HDD
3. **CPU throttling:** Check thermal management

**Debug:**
```bash
# Check system load
top

# Monitor during execution
python 06_inferenceAndPassport.py --max-samples 5
```

**Expected component breakdown:**
- Preprocessing: 2-3%
- **Inference: 67-68%** (dominant)
- Postprocessing: 8-9%
- Passport: 21-22%

---

#### Issue 5: "Segmentation looks incorrect"

**Checklist:**
1. **Verify modality:** Make sure you're using the correct `--modality` flag (ct or mri)
2. **Check orientation:** TotalSegmentator expects RAS orientation (handled automatically)
3. **Inspect values:** CT should have Hounsfield units (-1024 to +3071). MRI values vary by sequence.

**Validation:**
```python
import nibabel as nib
img = nib.load("ct_data/your_scan.nii.gz")
data = img.get_fdata()
print(f"Min: {data.min()}, Max: {data.max()}")
# CT: should be around [-1024, +3071]
# MRI: varies by acquisition (no fixed range)
```

---

#### Issue 6: Different results than documentation

**Expected Variations:**

| Factor | Impact |
|--------|--------|
| **Scan size** | Larger volumes = longer inference |
| **Anatomy** | More organs = longer passport extraction |
| **CPU model** | Affects absolute timing (not percentages) |
| **System load** | Background processes slow down inference |

**Acceptable Ranges:**
- Mean latency: 8-12s (target: ~10s)
- Inference: 60-75% of total time
- Passport: 15-25% of total time

---

### Getting Help

If you encounter issues not covered here:

1. **Check existing benchmarks:**
   - `COMPREHENSIVE_BENCHMARK_N49_RESULTS.md` (N=49 validation)
   - `INFERENCE_PERFORMANCE_INVESTIGATION.md` (thread tuning)

2. **Verify system specs:**
   - Python version: `python --version` (need 3.8+)
   - Package versions: `pip list | grep -E "numpy|scipy|nibabel|onnxruntime"`

3. **Report issues:**
   - Include error message
   - Attach benchmark results JSON
   - Specify CT scan characteristics (size, region)

---

## 📚 Additional Resources

### Related Documentation

- **`README.md`** - Project overview and key results
- **`COMPREHENSIVE_BENCHMARK_N49_RESULTS.md`** - Full CT validation report (N=49 scans)
- **`TOTALSEGMENTATOR_TEST_DATASETS_RESEARCH.md`** - Official test datasets

### Other Benchmark Scripts

| Script | Purpose | Modality |
|--------|---------|----------|
| `03_vanilla_benchmark.py` | Baseline PyTorch performance | CT, MRI (`--modality`) |
| `06_inferenceAndPassport.py` | **Optimized pipeline** | CT, MRI (`--modality`) |
| `07_compare_vanilla_vs_optimized.py` | Dice score + speedup comparison | CT, MRI (`--modality`) |

### Performance Comparison

| Implementation | CT Latency | MRI Latency | CT Speedup |
|----------------|-----------|-------------|------------|
| Vanilla PyTorch | ~43.8s | TBD | 1.0x |
| **Optimized ONNX** | **~12s** | **~11.2s** | **4.3x** |

---

## 🎯 Success Criteria

Your benchmark is successful if:

- ✅ **100% success rate** (no failed scans)
- ✅ **Mean latency < 15s**
- ✅ **Inference dominates** (65-70% of total time)
- ✅ **Passport extraction < 25%** of total time
- ✅ **CT organs detected: 50-90** (depending on anatomy)
- ✅ **MRI organs detected: 20-40** (abdominal structures)

---

## 🆚 Comparing Performance: Vanilla vs Optimized

If you want to benchmark the speedup and accuracy against the original TotalSegmentator implementation, follow this 3-step workflow. All scripts support `--modality ct|mri`.

### CT Comparison

```bash
# Step 1: Run vanilla baseline (saves masks + ground truth JSON)
python 03_vanilla_benchmark.py --modality ct --max-samples 20

# Step 2: Run optimized pipeline
python 06_inferenceAndPassport.py --modality ct --max-samples 20

# Step 3: Compare (generates Dice scores, speedup report)
python 07_compare_vanilla_vs_optimized.py --modality ct
# Report: benchmarks/VANILLA_VS_OPTIMIZED_REPORT.md
```

### MRI Comparison

```bash
# Step 1: Run vanilla baseline
python 03_vanilla_benchmark.py --modality mri --max-samples 5

# Step 2: Run optimized pipeline
python 06_inferenceAndPassport.py --modality mri --max-samples 5

# Step 3: Compare
python 07_compare_vanilla_vs_optimized.py --modality mri
# Report: benchmarks/VANILLA_VS_OPTIMIZED_REPORT_MRI.md
```

### Expected Output

| Metric | CT Expected | MRI Expected |
| :--- | :--- | :--- |
| **Speedup** | **4.0x - 5.0x** | TBD |
| **Mean Dice** | **> 0.90** | TBD |

---

## 📝 Example Validation Report

After running your benchmark, create a validation report:

```markdown
## Validation Report - Custom Dataset

**Date:** 2026-01-13
**Dataset:** My Hospital CT Scans
**Samples:** 25 abdominal CT scans

### Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Success Rate | 25/25 (100%) | >95% | ✅ PASS |
| Mean Latency | 11.2s | <12s | ✅ PASS |
| Organs Detected | 72.4 ± 8.1 | 50-90 | ✅ PASS |
| Inference Time | 7.6s (67.8%) | 60-75% | ✅ PASS |
| Passport Time | 2.4s (21.4%) | 15-25% | ✅ PASS |

### Conclusion

The TotalSegmentator optimized pipeline successfully processes our
clinical dataset with 100% reliability and meets all performance targets.
Average processing time of 11.2s is within acceptable range for
production deployment.
```

---

## 🔬 Advanced: Dice Score Validation (Optional)

If you have ground truth segmentations, calculate Dice scores:

```python
import nibabel as nib
import numpy as np

def dice_score(pred_path, gt_path):
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()

    intersection = np.sum(pred * gt)
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt))
    return dice

# Example
dice = dice_score(
    "benchmarks/inference_and_passport_results/masks/scan_mask.nii.gz",
    "ground_truth/scan_gt.nii.gz"
)
print(f"Dice Score: {dice:.4f}")  # Target: >0.85, Expected: ~0.91
```

**Expected Performance:**
- Mean Dice: ~0.91 (validated on N=20 subset)
- Clinical threshold: >0.85

---

**Document Version:** 2.0
**Last Updated:** 2026-02-10
**Validated On:** CT: N=49 (100% success), MRI: N=5 (100% success)
