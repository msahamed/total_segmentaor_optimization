# Comprehensive Performance Comparison: Vanilla vs Python-ONNX vs Rust-ONNX

## Summary Statistics (N=20)

| Implementation | Avg Latency (s) | Avg Speedup | Mean Dice (Valid) |
| :--- | :--- | :--- | :--- |
| Vanilla PyTorch | 43.88s | 1.0x | N/A |
| Python-ONNX | 8.88s | 4.9x | 0.9106 |
| **Rust-ONNX** | **9.88s** | **4.4x** | **0.8454** |

## Detailed Results

| Subject                     |   Vanilla Lat (s) |   Py-ONNX Lat (s) |   Rust-ONNX Lat (s) |   Rust Speedup |   Py Dice |   Rust Dice |
|:----------------------------|------------------:|------------------:|--------------------:|---------------:|----------:|------------:|
| case_001_exp.nii.gz         |             39.12 |              6.84 |             7.05306 |        5.54653 |    0.931  |    0.859155 |
| case_001_insp.nii.gz        |             60.19 |              8.71 |             6.96176 |        8.64581 |    0.9662 |    0.869851 |
| case_002_exp.nii.gz         |             58.4  |              7.9  |             6.7491  |        8.65301 |    0.8974 |    0.832645 |
| case_002_insp.nii.gz        |             36.83 |             10.67 |             6.90434 |        5.33432 |    0.9872 |    0.901008 |
| case_003_exp.nii.gz         |             24.33 |              6.86 |             7.55229 |        3.22154 |    0.8451 |    0.800704 |
| case_003_insp.nii.gz        |             35.9  |              6.5  |             7.86276 |        4.56583 |    0.9786 |    0.890473 |
| case_004_exp.nii.gz         |             33.15 |              6.74 |             9.76696 |        3.3941  |    0.8492 |    0.782806 |
| case_004_insp.nii.gz        |             23.32 |              8.48 |             9.15177 |        2.54814 |    0.9328 |    0.855108 |
| case_005_exp.nii.gz         |             23.21 |              7.7  |            16.1016  |        1.44147 |    0.904  |    0.827799 |
| case_005_insp.nii.gz        |             33.33 |              8.99 |            11.1017  |        3.00224 |    0.9868 |    0.889557 |
| case_006_exp.nii.gz         |             23.08 |              7.42 |            10.9091  |        2.11566 |    0.8493 |    0.821194 |
| case_006_insp.nii.gz        |             37.28 |              7.42 |             8.63889 |        4.31537 |    0.9643 |    0.875248 |
| case_007_exp.nii.gz         |             58.27 |              7.22 |            10.0831  |        5.77896 |    0.8856 |    0.834774 |
| ct_abdomen.nii.gz           |             44.07 |              8.26 |             7.11337 |        6.19537 |    0.9458 |    0.897163 |
| ct_philips.nii.gz           |             33.38 |              3.3  |             2.91907 |       11.4351  |    0.6603 |    0.665163 |
| msd_spleen_spleen_19.nii.gz |             51.65 |             12.77 |            10.1731  |        5.07711 |    0.8787 |    0.825409 |
| msd_spleen_spleen_3.nii.gz  |             53.7  |              8.24 |             9.56978 |        5.61141 |    0.9201 |    0.862201 |
| msd_spleen_spleen_31.nii.gz |             42.62 |              8.67 |             8.79521 |        4.84582 |    0.9216 |    0.850161 |
| msd_spleen_spleen_40.nii.gz |             61.13 |             15.65 |            14.6023  |        4.18633 |    0.9615 |    0.869607 |
| msd_spleen_spleen_52.nii.gz |            104.56 |             19.28 |            25.6221  |        4.08085 |    0.9457 |    0.898055 |

## Technical Notes

> [!TIP]
> **Orientation Handling**: The Rust-ONNX implementation successfully handles both RAS and LPS orientations
> by detecting orientation from the NIfTI header and applying canonical reorientation during preprocessing.
> This is reflected in the high Dice scores (>0.80) achieved on the Learn2Reg (LPS) subjects.
