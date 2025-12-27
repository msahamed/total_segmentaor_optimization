#!/usr/bin/env python3
"""
Step 1: Download Real CT Data from Hugging Face
================================================

Downloads real CT scans from Hugging Face datasets for testing TotalSegmentator.
Using the CT-RATE dataset which contains chest CT scans in NIfTI format.

Usage:
    python 1_download_ct_data.py
"""

import os
from pathlib import Path

import nibabel as nib
from huggingface_hub import hf_hub_download

# Configuration
DATA_DIR = Path("./ct_data")
DATA_DIR.mkdir(exist_ok=True)

# Hugging Face dataset info
DATASET_REPO = "ibrahimhamamci/CT-RATE"
SAMPLE_FILES = [
    # Download a few sample CT scans
    "ct_scans/train/volume_001.nii.gz",  # These are example paths
]


def download_msd_spleen():
    """Download Task09_Spleen from Medical Segmentation Decathlon"""
    print("=" * 70)
    print("Downloading MSD Task09_Spleen (Abdominal CT)")
    print("=" * 70)
    
    URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    TAR_PATH = DATA_DIR / "Task09_Spleen.tar"
    EXTRACT_DIR = DATA_DIR / "Task09_Spleen"
    
    import urllib.request
    import tarfile
    import shutil
    
    # 1. Download Tar
    if not TAR_PATH.exists() and not EXTRACT_DIR.exists():
        print(f"Downloading {URL}...")
        print("This is ~1.5GB. Please wait...")
        try:
            # simple progress hook
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = downloaded * 100 / total_size
                    if block_num % 1000 == 0:
                        print(f"\rProgress: {percent:.1f}% ({downloaded/1024/1024:.1f} MB)", end="")
            
            urllib.request.urlretrieve(URL, TAR_PATH, reporthook=show_progress)
            print("\n✓ Download complete.")
        except Exception as e:
            print(f"\n⚠ Download failed: {e}")
            return []
            
    # 2. Extract
    if TAR_PATH.exists():
        print("Extracting tar archive...")
        try:
            with tarfile.open(TAR_PATH, "r") as tar:
                # Only extract imagesTr folder (Training images)
                # We only need 10 files
                members = []
                count = 0
                for member in tar.getmembers():
                    if "imagesTr" in member.name and member.name.endswith(".nii.gz") and not member.name.startswith("._"):
                        members.append(member)
                        count += 1
                        if count >= 10:
                            break
                            
                tar.extractall(path=DATA_DIR, members=members)
                print(f"✓ Extracted {len(members)} CT scans.")
        except Exception as e:
            print(f"⚠ Extraction failed: {e}")
            return []
            
        # Cleanup tar
        print("Removing tar file...")
        TAR_PATH.unlink()
        
    # 3. Move files to main folder
    downloaded = []
    source_dir = DATA_DIR / "Task09_Spleen" / "imagesTr"
    if source_dir.exists():
        for f in source_dir.glob("*.nii.gz"):
            dest = DATA_DIR / f"msd_spleen_{f.name}"
            shutil.move(str(f), str(dest))
            downloaded.append(str(dest))
            
        # Cleanup extraction dir
        shutil.rmtree(DATA_DIR / "Task09_Spleen")
        
    return downloaded

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("TotalSegmentator Data Expansion")
    print("=" * 70)

    # 1. Download MSD Data
    files = download_msd_spleen()
    
    # 2. List all data
    print("\n" + "=" * 70)
    print("Current Dataset Status")
    print("=" * 70)
    all_files = list(DATA_DIR.glob("*.nii.gz"))
    print(f"Total CT Scans: {len(all_files)}")
    
    if len(all_files) >= 10:
        print("✓ Sufficient data available for benchmarking.")
    else:
        print("⚠ Limited data available.")

    return [str(f) for f in all_files]

if __name__ == "__main__":
    main()
