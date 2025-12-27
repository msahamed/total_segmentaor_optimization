# Detailed 4-Way Performance Comparison

| Subject | Vanilla (s) | Python (Baseline) (s) | Python (Optimized) (s) | Rust (s) | Best ONNX |
| :--- | :---: | :---: | :---: | :---: | :---: |
| case_001_exp.nii.gz | 39.12 | 6.84 | 17.27 | 7.05 | **Py-Base** |
| case_001_insp.nii.gz | 60.19 | 8.71 | 8.11 | 6.96 | **Rust** |
| case_002_exp.nii.gz | 58.40 | 7.90 | 6.94 | 6.75 | **Rust** |
| case_002_insp.nii.gz | 36.83 | 10.67 | 7.01 | 6.90 | **Rust** |
| case_003_exp.nii.gz | 24.33 | 6.86 | 6.59 | 7.55 | **Py-Opt** |
| case_003_insp.nii.gz | 35.90 | 6.50 | 11.60 | 7.86 | **Py-Base** |
| case_004_exp.nii.gz | 33.15 | 6.74 | 11.92 | 9.77 | **Py-Base** |
| case_004_insp.nii.gz | 23.32 | 8.48 | 20.11 | 9.15 | **Py-Base** |
| case_005_exp.nii.gz | 23.21 | 7.70 | 6.66 | 16.10 | **Py-Opt** |
| case_005_insp.nii.gz | 33.33 | 8.99 | 15.36 | 11.10 | **Py-Base** |
| case_006_exp.nii.gz | 23.08 | 7.42 | 9.02 | 10.91 | **Py-Base** |
| case_006_insp.nii.gz | 37.28 | 7.42 | 7.93 | 8.64 | **Py-Base** |
| case_007_exp.nii.gz | 58.27 | 7.22 | 6.70 | 10.08 | **Py-Opt** |
| ct_abdomen.nii.gz | 44.07 | 8.26 | 9.34 | 7.11 | **Rust** |
| ct_philips.nii.gz | 33.38 | 3.30 | 2.15 | 2.92 | **Py-Opt** |
| msd_spleen_spleen_19.nii.gz | 51.65 | 12.77 | 19.40 | 10.17 | **Rust** |
| msd_spleen_spleen_3.nii.gz | 53.70 | 8.24 | 6.88 | 9.57 | **Py-Opt** |
| msd_spleen_spleen_31.nii.gz | 42.62 | 8.67 | 12.58 | 8.80 | **Py-Base** |
| msd_spleen_spleen_40.nii.gz | 61.13 | 15.65 | 15.44 | 14.60 | **Rust** |
| msd_spleen_spleen_52.nii.gz | 104.56 | 19.28 | 39.32 | 25.62 | **Py-Base** |
| **AVERAGE** | **43.88** | **8.88** | **12.02** | **9.88** | |

## Summary
- **Vanilla Avg**: 43.88s
- **Python Baseline Avg**: 8.88s
- **Python Optimized Avg**: 12.02s
- **Rust Avg**: 9.88s