#!/usr/bin/env python3
"""
Step 2: Export TotalSegmentator Model to ONNX
==============================================

Exports the TotalSegmentator PyTorch model to ONNX format for production deployment.
ONNX provides cross-platform compatibility and optimization opportunities.

Usage:
    python 2_export_model.py [--task total] [--fast]
"""

import argparse
import time
from pathlib import Path

import onnx
import torch
from totalsegmentator.python_api import setup_nnunet

# Configuration
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)


def get_model_info(task: str, fast: bool):
    """Get nnUNet task information based on task name and fast flag"""
    if task == "total":
        if fast:
            return [297], "nnUNetTrainer_4000epochs_NoMirroring", "nnUNetPlans", "3d_fullres"
        else:
            # Standard 'total' task uses 5 models
            return [291, 292, 293, 294, 295], "nnUNetTrainerNoMirroring", "nnUNetPlans", "3d_fullres"
    elif task == "lung_vessels":
        return [258], "nnUNetTrainer", "nnUNetPlans", "3d_fullres"
    elif task == "body":
        return [300], "nnUNetTrainer", "nnUNetPlans", "3d_fullres"
    else:
        # Default to trial-and-error or specific ID if task is numeric
        try:
            tid = int(task)
            return [tid], "nnUNetTrainer", "nnUNetPlans", "3d_fullres"
        except ValueError:
            print(f"⚠ Warning: Unknown task '{task}'. Defaulting to Task 297 (fast).")
            return [297], "nnUNetTrainer_4000epochs_NoMirroring", "nnUNetPlans", "3d_fullres"


def export_single_model(
    task_id: int, 
    trainer: str, 
    plans: str, 
    model_config: str, 
    output_path: Path, 
    patch_size: tuple = (128, 128, 128),
    device: str = "cpu"
):
    """Export a single nnUNet model to ONNX"""
    print(f"\n--- Exporting Task {task_id} to {output_path.name} ---")
    
    from totalsegmentator.nnunet import get_output_folder
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from totalsegmentator.libs import download_pretrained_weights
    import os

    # 1. Download weights
    try:
        download_pretrained_weights(task_id)
    except Exception as e:
        print(f"Warning: Download weights failed or skipped: {e}")

    # 2. Resolve dataset name
    results_dir = os.environ.get("nnUNet_results")
    if not results_dir:
        print("ERROR: nnUNet_results environment variable not set.")
        return False

    dataset_name = None
    results_path = Path(results_dir)
    for item in results_path.iterdir():
        if item.is_dir() and item.name.startswith(f"Dataset{task_id:03d}"):
            dataset_name = item.name
            break
    
    if not dataset_name:
        # Try without leading zeros just in case
        for item in results_path.iterdir():
            if item.is_dir() and item.name.startswith(f"Dataset{task_id}"):
                dataset_name = item.name
                break
    
    if not dataset_name:
        print(f"⚠ Could not find dataset folder for ID {task_id} in {results_dir}")
        dataset_name = f"Dataset{task_id:03d}" # Predictor might find it

    # 3. Get model folder
    try:
        model_folder = get_output_folder(dataset_name, trainer, plans, model_config)
        print(f"Model folder: {model_folder}")
    except Exception as e:
        print(f"ERROR: Could not resolve model folder: {e}")
        return False

    # 4. Load weights into Predictor
    torch_device = torch.device(device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    
    try:
        predictor.initialize_from_trained_model_folder(
            str(model_folder),
            use_folds=[0],
            checkpoint_name="checkpoint_final.pth",
        )
    except Exception as e:
        print(f"ERROR: Prediction initialization failed: {e}")
        return False
        
    model = predictor.network
    model.eval()

    # 5. Create dummy input
    batch_size = 1
    channels = 1
    depth, height, width = patch_size
    dummy_input = torch.randn(batch_size, channels, depth, height, width).to(torch_device)

    # 6. Export to ONNX
    print(f"Starting export (patch size: {patch_size})...")
    start_time = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
                    "output": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
                },
                verbose=False,
            )
        print(f"✓ Export completed in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"⚠ Export failed: {e}")
        return False

    # 7. Verification
    try:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ ONNX model valid. Size: {size_mb:.2f} MB")
    except Exception as e:
        print(f"⚠ Verification failed: {e}")
        return False

    return True


def main():
    """Main execution with command-line arguments"""
    parser = argparse.ArgumentParser(description="Export TotalSegmentator to ONNX")
    parser.add_argument("--task", type=str, default="total", help="Task name (default: total)")
    parser.add_argument("--fast", action="store_true", help="Use fast model")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size (default: 128)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for export (cpu/cuda)")

    args = parser.parse_args()
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    
    # Initialize nnUNet environment
    from totalsegmentator.config import setup_nnunet
    setup_nnunet()

    task_ids, trainer, plans, model_config = get_model_info(args.task, args.fast)
    
    print("=" * 70)
    print(f"EXPORT PLAN: {args.task} ({'fast' if args.fast else 'standard'})")
    print(f"Tasks to export: {task_ids}")
    print("=" * 70)

    success_count = 0
    for tid in task_ids:
        # Determine filename
        if len(task_ids) == 1:
            name = f"totalsegmentator_{args.task}{'_fast' if args.fast else ''}_fp32.onnx"
        else:
            # Map task ID to a friendly name if it's the 5-part 'total'
            part_map = {291: "part1", 292: "part2", 293: "part3", 294: "part4", 295: "part5"}
            part_name = part_map.get(tid, f"task{tid}")
            name = f"totalsegmentator_{part_name}_fp32.onnx"
            
        output_path = MODEL_DIR / name
        if export_single_model(tid, trainer, plans, model_config, output_path, patch_size, args.device):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: Exported {success_count}/{len(task_ids)} models successfully.")
    print("=" * 70)
    
    return 0 if success_count == len(task_ids) else 1



if __name__ == "__main__":
    exit(main())
