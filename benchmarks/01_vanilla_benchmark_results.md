# Vanilla Baseline Analysis Report (Fast Model)

## Overall Statistics (N=20)
- **Avg Latency**: 43.88s (±18.71s)
- **Min/Max Latency**: 23.08s / 104.56s
- **Avg Organs**: 66.85
- **Avg Throughput**: 0.25 MVox/s

## Categorical Breakdown
| Category | Count | Avg Latency | Avg Organs | Avg Throughput |
| :--- | :--- | :--- | :--- | :--- |
| Learn2Reg | 13 | 37.42s | 69.15 | 0.23 MVox/s |
| Root | 7 | 55.87s | 62.57 | 0.30 MVox/s |

## Top 5 Slowest Subjects
| Subject | Category | Shape | Latency |
| :--- | :--- | :--- | :--- |
| msd_spleen_spleen_52.nii.gz | Root | [512, 512, 112] | 104.56s |
| msd_spleen_spleen_40.nii.gz | Root | [512, 512, 93] | 61.13s |
| case_001_insp.nii.gz | Learn2Reg | [192, 192, 208] | 60.19s |
| case_002_exp.nii.gz | Learn2Reg | [192, 192, 208] | 58.40s |
| case_007_exp.nii.gz | Learn2Reg | [192, 192, 208] | 58.27s |
