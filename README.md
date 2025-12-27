# TotalSegmentator Optimization Study

This project evaluates and optimizes different implementations of the **TotalSegmentator** (Fast Model) pipeline to achieve the best balance of latency and segmentation accuracy (Dice score).

## 🚀 Key Results (N=20 Benchmark)

| Implementation | Typical Latency | Mean Dice | Speedup | Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla PyTorch** | ~43.8s | ~0.94 | 1.0x | Python/PyTorch |
| **Python-ONNX (Baseline)** | ~8.9s | 0.9106 | 4.9x | Python/ORT |
| **Rust-ONNX** | ~9.8s | 0.8454 | 4.4x | Rust/ORT |
| **Python-ONNX (Optimized)** | **~5.8s*** | **0.9099** | **7.5x** | Python/ORT |

*\*Performance floor of **2.15s** achieved on smaller volumes. Typical performance on a cold CPU is ~5.8s.*

## 📋 Project Objective
The goal was to move from the original PyTorch implementation to a high-performance deployment environment. While Rust was explored for its low-level efficiency, the **Optimized Python-ONNX** pipeline emerged as the winner due to:
1.  **Superior Accuracy**: Utilizing `scipy`'s high-fidelity resampling kernels.
2.  **Maximum Throughput**: Achieving lower latencies than Rust through aggressive ONNX session tuning and efficient `order=1` resampling.

## 🛠 Usage Pipeline

The project is organized into sequential scripts:

1.  **`01_download_data.py`**: Downloads the 20-subject benchmark dataset (MSD Spleen & Learn2Reg).
2.  **`02_export_model.py`**: Exports the TotalSegmentator model to ONNX format.
3.  **`03_vanilla_benchmark.py`**: Establises the baseline performance using raw PyTorch.
4.  **`04_onnx_benchmark.py`**: Baselines the standard Python ONNX Runtime implementation.
5.  **`05_optimized_python_benchmark.py`**: Demonstrates 30-50% speedups via session tuning and resampling efficiency.
6.  **`06_linux_production_benchmark.py`**: Final production-ready script for Linux/AMD environments (includes `taskset` tuning).

## 📁 Repository Structure
- `rust/`: Experimental Rust implementation (for study purposes).
- `benchmarks/`: Comprehensive latency and Dice score reports.
- `models/`: ONNX model storage.
- `ct_data/`: (Ignored) Raw medical image data.

## 💡 Recommendation
For production deployment, use the **Optimized Python-ONNX** pipeline. It provides a **7.5x speedup** over Vanilla PyTorch while maintaining the necessary accuracy (>0.90 Dice) for clinical reliability.
