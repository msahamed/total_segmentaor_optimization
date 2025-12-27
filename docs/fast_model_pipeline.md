# TotalSegmentator 'fast' Model Pipeline Documentation

This document describes the input, output, and processing pipeline for the TotalSegmentator `fast` model (Task 297).

## Model Overview
- **Task ID**: 297
- **Model Architecture**: nnU-Net 3d_fullres (PlainConvUNet)
- **Target Resolution**: 3.0mm isotropic (3.0, 3.0, 3.0)
- **Patch Size**: [112, 112, 128] (Used during training)
- **Input Channels**: 1 (CT image)

## 1. Preprocessing Pipeline

| Step | Method | Description |
| :--- | :--- | :--- |
| **Orientation** | `as_closest_canonical` | Ensures image is in canonical orientation (usually RAS). |
| **Resampling** | `change_spacing` | Resamples original CT to 3.0mm isotropic spacing using 3rd order spline interpolation. |
| **Clipping** | `CTNormalization` | Intensity values are clipped to `[-1004.0, 1588.0]` (0.5th and 99.5th percentiles). |
| **Normalization** | Z-Score | `(x - (-50.38)) / 503.39` |

## 2. Model Input/Output
- **Input**: Tensor of shape `[1, 1, D, H, W]` (normalized HU values).
- **Output**: Softmax probabilities or class indices of shape `[1, C, D, H, W]`.
- **Classes**: 118 labels (TotalSegmentator v2 standard set).

## 3. Postprocessing Pipeline

| Step | Method | Description |
| :--- | :--- | :--- |
| **Argmax** | `argmax(axis=1)` | Converts multi-channel probabilities to a single-channel label map. |
| **Resampling** | `change_spacing` | Resamples the 3mm label map back to the original CT spacing using nearest-neighbor interpolation. |
| **Reorientation** | Inverse Affine | Maps the canonical labels back to the original image orientation. |
| **Label Mapping** | `class_map` | Maps predicted indices (1-117) to specific anatomy names (e.g., 1=spleen). |

## 4. Optimized Implementation Notes (Rust/ONNX)

> [!TIP]
> To match the vanilla performance, the Rust implementation should focus on:
> 1.  Fast 3D Spline Interpolation for resampling.
> 2.  Efficient Z-score normalization using SIMD.
> 3.  Sliding window inference with overlap (TotalSegmentator uses 0.5 step size).
> 4.  Handling dynamic input shapes (arbitrary volume size).
