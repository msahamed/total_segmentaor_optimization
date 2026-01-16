# Cost Analysis: TotalSegmentator Production Deployment (AWS)

This document outlines the cost components and estimated expenses for deploying the TotalSegmentator pipeline on AWS.

## 1. System Assumptions
To provide a concrete estimate, we base calculations on the following observed metrics from our local benchmarks:
- **Average Throughput**: 1 scan every ~5 seconds (Optimized CPU pipeline).
- **Input Data Size**: ~50 MB per CT scan (compressed NIfTI).
- **Output Data Size**: ~5 MB (Segmentation masks + JSON Passport).
- **Volume Scenario**: 10,000 scans per month.

---

## 2. Key Cost Components

### A. Compute (The Processing Engine)
This is typically the largest cost driver for AI workloads.
*   **Latency**: High latency = longer instance running time = higher cost.
*   **Scaling**: Using Auto Scaling Groups (ASG) or AWS Batch to scale down to 0 when idle is crucial.

**Options:**
1.  **CPU-based (e.g., c6i.xlarge)**: 
    -   *Pros*: Cheaper per hour, good for our ~5s optimized inference.
    -   *Cons*: Slower than GPU.
2.  **GPU-based (e.g., g4dn.xlarge)**:
    -   *Pros*: Fastest inference (~1-2s).
    -   *Cons*: Higher hourly cost, initialization overhead can be high for spiky workloads.

### B. Storage (S3)
Storing the raw DICOM/NIfTI files and the generated masks.
*   **Input Storage**: *Excluded (Pre-existing S3 bucket)*.
*   **Output Storage**: 10k scans * 5 MB = 50 GB/month.
*   **Lifecycle Policy**: Move old data to S3 Glacier Instant Retrieval after 30 days to save 60-80%.

### C. Database (Metadata)
Storing the "Anatomical Passport" (organ volumes, presence/absence) for quick querying.
*   **Service**: DynamoDB (NoSQL) is ideal for this JSON-like data.
*   **Volume**: 2KB per record is negligible in terms of storage cost; cost is driven by Write Capacity Units (WCUs).

### D. Data Transfer (Networking)
*   **Ingress (Inbound)**: Free on AWS.
*   **Egress (Outbound)**: Sending the masks/reports back to the user/PACS. AWS charges ~$0.09 per GB.

---

## 3. Estimated Monthly Bill (Scenario: 10,000 Scans)

### Architecture: Serverless / On-Demand (AWS Batch + Fargate)
*Recommended for variable workload (not 24/7).*

| Component | Calculation | Estimated Cost |
| :--- | :--- | :--- |
| **Compute (Fargate vs EC2)** | **CPU (4 vCPU, 8GB RAM)**: ~$0.05/hour.<br>Time per scan: 10s (inc. startup/overhead).<br>Total Compute Hours: (10,000 * 10s) / 3600 = ~28 hours.<br>Cost: 28 * $0.20 (Fargate spot pricing approx) | **~$6.00** |
| **GPU Option (g4dn.xlarge)** | ~$0.526/hour.<br>Time per scan: 5s.<br>Hours: ~14 hours.<br>Cost: 14 * $0.526 | **~$7.50** |
| **Storage (S3 Standard)** | 50 GB (masks only) * $0.023/GB | **$1.15** |
| **Database (DynamoDB)** | On-Demand mode. <br>Writes: 10k units ($1.25/million) <br>Reads: Variable. | **<$1.00** |
| **Data Transfer Out** | 50 GB (masks) * $0.09/GB | **$4.50** |
| **CloudWatch Logs** | Logging inference steps (~10MB/day) | **<$1.00** |
| **TOTAL ESTIMATE** | | **~$12 - $15 / month** |

*> **Note**: The estimates above use **Spot Instances / Fargate Spot** which offers up to 70% savings. On-demand prices would be ~3x higher.*

---

## 4. Optimization Opportunities (Cost Reduction)

1.  **Spot Instances**: Use AWS Batch with Spot Instances (g4dn.xlarge or c6i types). This is the single biggest saver.
2.  **S3 Lifecycle Policies**: Automatically move raw CT scans to Glacier after 7 days if they are only needed for processing.
3.  **Model Optimization**: We already achieved ~4-7x speedup. This directly reduces compute cost by ~80% compared to the vanilla implementation.
4.  **Shared Volumes**: If processing many files at once, mounting Amazon EFS can be faster than S3 copy per file, though EFS storage is more expensive ($0.30/GB). Stick to S3 for batch processing.

## 5. Summary Recommendation
For 10,000 scans/month:
*   **Architecture**: AWS Batch (ECS Fargate or EC2 Spot).
*   **Instance Type**: `c6i.2xlarge` (Compute Optimized) or `g4dn.xlarge` (GPU) depending on SLAs. Since our Optimized CPU version is fast (~5s), **CPU Spot instances** are likely the most cost-efficient robust choice.
*   **Total Expected Cost**: **<$15/month** (extremely efficient).
