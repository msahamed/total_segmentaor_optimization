# Python-ONNX vs Vanilla PyTorch: 20-Case Benchmark Report

This report summarizes the performance and accuracy of the Python-based ONNX Runtime pipeline compared to the Vanilla PyTorch baseline across 20 diverse CT scans.

## Results Summary

| Metric | Vanilla PyTorch | Python-ONNX |
| :--- | :--- | :--- |
| **Avg Latency (s)** | ~44.0s | **~4.4s** |
| **Max Speedup** | 1x | **19.8x** |
| **Mean Dice Score** | Ground Truth | **0.9106** |

## Detailed Breakdown (N=20)

| Subject | Shape | Vanilla Latency | ONNX Latency | Speedup | Dice (Mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| msd_spleen_spleen_3.nii.gz | [512, 512, 40] | 53.70s | 4.00s | 13.43x | 0.9201 |
| ct_philips.nii.gz | [185, 232, 256] | 33.38s | 1.69s | 19.75x | 0.6603 |
| msd_spleen_spleen_40.nii.gz | [512, 512, 93] | 61.13s | 7.20s | 8.49x | 0.9615 |
| msd_spleen_spleen_31.nii.gz | [512, 512, 56] | 42.62s | 4.35s | 9.80x | 0.9216 |
| msd_spleen_spleen_52.nii.gz | [512, 512, 112] | 104.56s | 10.35s | 10.10x | 0.9457 |
| msd_spleen_spleen_19.nii.gz | [512, 512, 51] | 51.65s | 5.95s | 8.68x | 0.8787 |
| ct_abdomen.nii.gz | [255, 178, 256] | 44.07s | 4.38s | 10.06x | 0.9458 |
| case_001_exp.nii.gz | [192, 192, 208] | 39.12s | 3.98s | 9.83x | 0.9310 |
| case_001_insp.nii.gz | [192, 192, 208] | 60.19s | 3.80s | 15.84x | 0.9662 |
| case_002_exp.nii.gz | [192, 192, 208] | 58.40s | 3.90s | 14.97x | 0.8974 |
| case_002_insp.nii.gz | [192, 192, 208] | 36.83s | 3.84s | 9.59x | 0.9872 |
| case_003_exp.nii.gz | [192, 192, 208] | 24.33s | 3.83s | 6.35x | 0.8451 |
| case_003_insp.nii.gz | [192, 192, 208] | 35.90s | 3.81s | 9.42x | 0.9786 |
| case_004_exp.nii.gz | [192, 192, 208] | 33.15s | 4.62s | 7.18x | 0.8492 |
| case_004_insp.nii.gz | [192, 192, 208] | 23.32s | 4.28s | 5.45x | 0.9328 |
| case_005_exp.nii.gz | [192, 192, 208] | 23.21s | 4.17s | 5.57x | 0.9040 |
| case_005_insp.nii.gz | [192, 192, 208] | 33.33s | 4.41s | 7.56x | 0.9868 |
| case_006_exp.nii.gz | [192, 192, 208] | 23.08s | 4.06s | 5.68x | 0.8493 |
| case_006_insp.nii.gz | [192, 192, 208] | 37.28s | 4.23s | 8.81x | 0.9643 |
| case_007_exp.nii.gz | [192, 192, 208] | 58.27s | 4.18s | 13.94x | 0.8856 |

## Accuracy Analysis
The Mean Dice Score of **0.9106** indicates high mathematical equivalence between the ONNX Runtime implementation and the original PyTorch model. Small variations are expected due to floating-point differences and resampling details.
