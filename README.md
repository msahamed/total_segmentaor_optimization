# TotalSegmentator Optimization Study

This project evaluates and optimizes different implementations of the **TotalSegmentator** (Fast Model) pipeline to achieve the best balance of latency and segmentation accuracy (Dice score). Supports both **CT** and **MRI** modalities.

## 🚀 Key Results

### CT Performance (N=49 Comprehensive Validation)

| Implementation | Typical Latency | Mean Dice | Speedup | Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla PyTorch** | ~43.8s | ~0.94 | 1.0x | Python/PyTorch |
| **Python-ONNX (Optimized)** | **~12.0s** | **0.91** | **4.3x** | Python/ORT |

*Validated on N=49 diverse CT scans (100% success rate). Performance range: 2.6s-24s depending on volume size.*

### MRI Performance (N=5 Validation)

| Implementation | Typical Latency | Organs Detected | Success Rate | Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Python-ONNX (Optimized)** | **~11.2s** | **26-40** | **100%** | Python/ORT |

*Validated on 5 AMOS abdominal MRI scans. 50 organ classes available (Task 852).*

## 📋 Project Objective
The goal was to move from the original PyTorch implementation to a high-performance deployment environment. The **Optimized Python-ONNX** pipeline emerged as the winner due to:
1.  **Superior Accuracy**: Cubic resampling (`order=3`) matching vanilla TotalSegmentator's training pipeline.
2.  **Maximum Throughput**: Aggressive ONNX session tuning (4-thread config, sequential execution).
3.  **Multi-Modality**: Single pipeline supporting both CT (117 classes) and MRI (50 classes).

## 🛠 Usage Pipeline

The project is organized into sequential scripts:

1.  **`01_download_data.py`**: Downloads the 20-subject benchmark dataset (MSD Spleen & Learn2Reg).
2.  **`02_export_model.py`**: Exports TotalSegmentator models to ONNX format. Supports `--task total` (CT) and `--task total_mr` (MRI).
3.  **`03_vanilla_benchmark.py`**: Establishes the baseline performance using raw PyTorch.
4.  **`04_onnx_benchmark.py`**: Baselines the standard Python ONNX Runtime implementation.
5.  **`05_optimized_python_benchmark.py`**: Demonstrates 30-50% speedups via session tuning and resampling efficiency.
6.  **`06_inferenceAndPassport.py`**: Combines optimized inference with anatomical passport extraction. Supports `--modality ct|mri`. **[See Testing Guide](TESTING_GUIDE.md)**
7.  **`07_compare_vanilla_vs_optimized.py`**: Compares vanilla vs optimized results (Dice score, speedup).
8.  **`08_visualize_passport.py`**: Multi-panel passport visualization (organ map, volumes, projections, organ shapes).
9.  **`09_visualize_spine.py`**: Spine-focused visualization with inter-vertebral distances, angles, and Cobb angle.

## 📁 Repository Structure
- `models/`: ONNX model storage.
  - `totalsegmentator_total_fast_fp32.onnx` — CT model (66MB, 117 classes, Task 297)
  - `totalsegmentator_total_mr_fast_fp32.onnx` — MRI model (63MB, 50 classes, Task 852)
- `ct_data/`: CT scan data (.nii.gz files).
- `mri_data/`: MRI scan data (.nii.gz files).
- `benchmarks/`: Latency and Dice score reports.
  - `inference_and_passport_results/` — CT benchmark outputs and passports.
  - `inference_and_passport_results_mri/` — MRI benchmark outputs and passports.
- `registration/`: Metadata-driven registration system for medical imaging.
  - `scripts/`: Registration engine and passport extraction tools.
  - `metadata/`: Pre-computed anatomical passports (centroids, eigenvectors, boundaries).
  - `output/`: Registration results and visualizations.
- `rust/`: Experimental Rust implementation (for study purposes).

## 📥 Dataset Download

### CT Dataset (N=49)
**Google Drive:** [Download Benchmark Dataset](https://drive.google.com/file/d/1g_a08EMAG1NTAWM5m2JV4mPgCcSqDuJs/view?usp=sharing)

**Contents:**
- MSD Spleen (Task09): 5 subjects
- Learn2Reg Challenge: 42 subjects (21 paired exp/insp)
- Custom samples: 2 subjects

**Usage:** Extract to `ct_data/` directory.

### MRI Dataset (N=5)
AMOS abdominal MRI scans (T1-weighted). Place `.nii.gz` files in `mri_data/` directory.

## 🔬 Registration Pipeline: Anatomical Passport System

The project includes a **metadata-driven registration system** that achieves **1000x speedup** over traditional intensity-based methods while maintaining clinical accuracy (~1.7mm error).

### What is an Anatomical Passport?

An anatomical passport is a compact JSON representation of organ geometry extracted from segmentation masks:

```json
{
  "1": {  // Organ label (e.g., 1 = spleen)
    "centroid": [x, y, z],              // 3D organ center in world coordinates
    "covariance": [[3x3 matrix]],       // Spatial distribution
    "eigenvalues": [λ1, λ2, λ3],        // Principal component magnitudes
    "eigenvectors": [[3x3 matrix]],     // Principal axes (orientation)
    "boundary_points": [[x,y,z], ...],  // ~500 surface points per organ
    "volume_mm3": 12345.67              // Organ volume
  }
}
```

### Two-Tier Registration Architecture

**Tier 1: Shape-Aware Rigid Alignment (~10ms)**
- Uses centroids AND principal axes (eigenvectors) to capture organ orientation
- Weighted Procrustes analysis with anatomical stability scores:
  - Vertebrae/Pelvis: 1.0 (very stable)
  - Ribs: 0.9
  - Major organs: 0.6
  - Lungs: 0.4 (respiratory motion)
- Produces rotation matrix R + translation vector t

**Tier 2: Deformable Registration (~1s)**
- Thin Plate Spline (TPS) interpolation on ~5000 boundary points
- KDTree nearest neighbor correspondence matching
- Adaptive outlier filtering (3×median or 95th percentile)
- Smoothing parameter: 2.0 (optimized for tight fitting)
- Sub-sampled displacement field (4× downsampling for speed)

### Performance Comparison

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Traditional (Intensity-based)** | 1-60 minutes | 0.5-2mm | High-precision research |
| **Metadata-driven (This project)** | **1 second** | 1.7mm | Real-time clinical workflows |

### Passport Extraction Benchmark (Script 06)

`06_inferenceAndPassport.py` measures the **complete registration preprocessing pipeline**:

1. **Preprocessing** (resampling, normalization, padding)
2. **Inference** (ONNX model forward pass)
3. **Postprocessing** (argmax, resample back to original resolution)
4. **Passport Extraction** (centroids, eigenvectors, boundary points)

**Purpose:**
- Quantify passport extraction overhead
- Determine if Rust optimization is worthwhile for this step
- Establish baseline for future optimization work

**Key Metrics Tracked:**
- Total latency breakdown (ms per component)
- Number of organs detected
- Passport file size (KB)
- Passport extraction as % of total pipeline time

**Decision Criteria for Rust Optimization:**
- If passport extraction > 500ms → **Consider Rust**
- If passport extraction < 500ms → **Python is sufficient**

**✅ BENCHMARK RESULTS (2026-01-12):**
- **Passport extraction: 17.6s average (58.5% of total pipeline)** ← **MAJOR BOTTLENECK**
- **Decision: PROCEED with Rust optimization**
- **Expected speedup: 6-8× (17.6s → 2-3s)**
- See `PASSPORT_EXTRACTION_ANALYSIS.md` for detailed analysis

This data-driven approach ensures optimization efforts are focused on actual bottlenecks.

### Registration Scripts

Located in `registration/scripts/`:
- **`InferenceAndPassport.py`**: Legacy passport extraction (reference implementation)
- **`AnatomicalRegister.py`**: Core two-tier registration engine (619 lines)
- **`sanity_check_registration.py`**: Comprehensive validation with multi-view visualizations
- **`Learn2RegBenchmark.py`**: Batch processing for multiple cases
- **`ClinicalApp.py`**: Streamlit web UI for clinical deployment
- **`brain_self_registration_check.py`** / **`lung_self_registration_check.py`**: Self-registration tests (should be ~0mm error)

See `registration/sabber_registration_notes.md` for comprehensive documentation (847 lines).

## ✅ Comprehensive Validation

### CT Validation (N=49)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Success Rate** | **100%** (49/49) | >95% | ✅ |
| **Mean Latency** | **~12s** | <15s | ✅ |
| **Mean Dice vs Vanilla** | **0.91** | >0.85 | ✅ |
| **Speedup** | **4.3x** | >3x | ✅ |

### MRI Validation (N=5)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Success Rate** | **100%** (5/5) | >95% | ✅ |
| **Mean Latency** | **~11.2s** | <15s | ✅ |
| **Organs Detected** | **26-40** per scan | >20 | ✅ |

### Key Optimizations Applied
1. ✅ **Cubic resampling (`order=3`)** matching vanilla training pipeline
2. ✅ **Passport extraction from 3mm volume** (8.4x faster than original resolution)
3. ✅ **4-thread ONNX configuration** (15-20% faster than using all cores)
4. ✅ **Modality-aware normalization** (CT: dataset-level z-score with HU clipping; MRI: per-image z-score)

## 🔬 CT vs MRI Model Details

| Property | CT Model | MRI Model |
|----------|----------|-----------|
| **Task ID** | 297 | 852 |
| **Output Classes** | 117 | 50 |
| **Normalization** | CT: clip [-1004, 1588] HU, z-score (dataset-level) | Z-score (per-image, no clipping) |
| **Trainer** | nnUNetTrainer_4000epochs_NoMirroring | nnUNetTrainer_2000epochs_NoMirroring |
| **Model Size** | 66 MB | 63 MB |
| **Structures** | Full body: organs, bones (individual vertebrae C1-S1, 24 ribs), vessels, muscles | Abdominal: organs, bones (merged vertebrae), vessels, muscles |

## 🧪 Testing with Custom Datasets

### CT Inference
```bash
# 1. Place CT scans (.nii.gz) in ct_data/ directory
cp your_ct_scans/*.nii.gz ct_data/

# 2. Run benchmark
python 06_inferenceAndPassport.py --modality ct

# 3. View results
cat benchmarks/inference_and_passport_results/benchmark_results.json
```

### MRI Inference
```bash
# 1. Place MRI scans (.nii.gz) in mri_data/ directory
cp your_mri_scans/*.nii.gz mri_data/

# 2. Run benchmark
python 06_inferenceAndPassport.py --modality mri

# 3. View results
cat benchmarks/inference_and_passport_results_mri/benchmark_results.json
```

### Visualization
```bash
# Static passport dashboard
python 08_visualize_passport.py <passport.json> --modality ct

# Spine analysis
python 09_visualize_spine.py <passport.json> --modality ct
```

**For comprehensive evaluator instructions, see:** **[TESTING_GUIDE.md](TESTING_GUIDE.md)**

## 💡 Recommendation
For production deployment, use the **Optimized Python-ONNX** pipeline with `--modality ct|mri`. It provides a **4.3x speedup** over Vanilla PyTorch while maintaining clinical accuracy (>0.90 Dice for CT).

**Validated Performance:**
- **CT**: ~12s average per scan, 0.91 Dice, 100% reliability across 49 scans
- **MRI**: ~11.2s average per scan, 26-40 organs detected, 100% reliability across 5 scans

For **registration workflows**, the metadata-driven approach offers dramatic speedups (1000x) with acceptable accuracy for most clinical applications.
