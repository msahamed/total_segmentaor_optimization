# Detailed ONNX vs Vanilla Comparison Report

| Subject | Shape | Vanilla Latency | ONNX Latency | Speedup | V-Organs | O-Organs | Org-Diff | Dice (Mean) | V-Throughput | O-Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| msd_spleen_spleen_3.nii.gz | [512, 512, 40] | 53.70s | 4.00s | 13.43x | 58 | 60 | +2 | 0.9201 | 0.20 | 2.62 |
| ct_philips.nii.gz | [185, 232, 256] | 33.38s | 1.69s | 19.75x | 13 | 13 | +0 | 0.6603 | 0.33 | 6.50 |
| msd_spleen_spleen_40.nii.gz | [512, 512, 93] | 61.13s | 7.20s | 8.49x | 85 | 85 | +0 | 0.9615 | 0.40 | 3.39 |
| msd_spleen_spleen_31.nii.gz | [512, 512, 56] | 42.62s | 4.35s | 9.80x | 67 | 64 | -3 | 0.9216 | 0.34 | 3.37 |
| msd_spleen_spleen_52.nii.gz | [512, 512, 112] | 104.56s | 10.35s | 10.10x | 72 | 72 | +0 | 0.9457 | 0.28 | 2.84 |
| msd_spleen_spleen_19.nii.gz | [512, 512, 51] | 51.65s | 5.95s | 8.68x | 56 | 53 | -3 | 0.8787 | 0.26 | 2.25 |
| ct_abdomen.nii.gz | [255, 178, 256] | 44.07s | 4.38s | 10.06x | 87 | 87 | +0 | 0.9458 | 0.26 | 2.65 |
| case_001_exp.nii.gz | [192, 192, 208] | 39.12s | 3.98s | 9.83x | 54 | 53 | -1 | 0.9310 | 0.20 | 1.93 |
| case_001_insp.nii.gz | [192, 192, 208] | 60.19s | 3.80s | 15.84x | 85 | 84 | -1 | 0.9662 | 0.13 | 2.02 |
| case_002_exp.nii.gz | [192, 192, 208] | 58.40s | 3.90s | 14.97x | 59 | 58 | -1 | 0.8974 | 0.13 | 1.97 |
| case_002_insp.nii.gz | [192, 192, 208] | 36.83s | 3.84s | 9.59x | 84 | 84 | +0 | 0.9872 | 0.21 | 2.00 |
| case_003_exp.nii.gz | [192, 192, 208] | 24.33s | 3.83s | 6.35x | 55 | 54 | -1 | 0.8451 | 0.32 | 2.00 |
| case_003_insp.nii.gz | [192, 192, 208] | 35.90s | 3.81s | 9.42x | 83 | 83 | +0 | 0.9786 | 0.21 | 2.01 |
| case_004_exp.nii.gz | [192, 192, 208] | 33.15s | 4.62s | 7.18x | 59 | 57 | -2 | 0.8492 | 0.23 | 1.66 |
| case_004_insp.nii.gz | [192, 192, 208] | 23.32s | 4.28s | 5.45x | 86 | 87 | +1 | 0.9328 | 0.33 | 1.79 |
| case_005_exp.nii.gz | [192, 192, 208] | 23.21s | 4.17s | 5.57x | 55 | 55 | +0 | 0.9040 | 0.33 | 1.84 |
| case_005_insp.nii.gz | [192, 192, 208] | 33.33s | 4.41s | 7.56x | 85 | 85 | +0 | 0.9868 | 0.23 | 1.74 |
| case_006_exp.nii.gz | [192, 192, 208] | 23.08s | 4.06s | 5.68x | 55 | 57 | +2 | 0.8493 | 0.33 | 1.89 |
| case_006_insp.nii.gz | [192, 192, 208] | 37.28s | 4.23s | 8.81x | 82 | 82 | +0 | 0.9643 | 0.21 | 1.81 |
| case_007_exp.nii.gz | [192, 192, 208] | 58.27s | 4.18s | 13.94x | 57 | 57 | +0 | 0.8856 | 0.13 | 1.83 |

## Summary Statistics (N=20)
- **Average Speedup**: **10.03x**
- **Mean Dice Score**: **0.9106**
- **Median Dice Score**: **0.9263**
- **Avg Organ Count Diff**: **-0.35**

## Notable Outliers
- **Lowest Agreement**: ct_philips.nii.gz (Dice: 0.6603)
- **Highest Agreement**: case_002_insp.nii.gz (Dice: 0.9872)
- **Fastest ONNX Run**: ct_philips.nii.gz (19.75x speedup)
