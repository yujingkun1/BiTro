#!/usr/bin/env python3
"""
Split feature files with an 80/20 ratio and copy them to train and test folders.
"""
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def split_features():
    # Configure paths (override via env if needed)
    base_dir = Path(os.environ.get("SPLIT_BASE_DIR", "/data/yujk/hovernet2feature/BiTro/demo_data/Feature/Bulk"))
    source_dir = Path(os.environ.get("SPLIT_SOURCE_DIR", base_dir / "all"))
    train_dir = Path(os.environ.get("SPLIT_TRAIN_DIR", base_dir / "train"))
    test_dir = Path(os.environ.get("SPLIT_TEST_DIR", base_dir / "test"))
    
    # Check whether the source directory exists
    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        return
    
    # Create destination directories
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    print(f"[INFO] Destination directories ready: {train_dir}, {test_dir}")
    
    # Collect all parquet files (excluding .bak files)
    print(f"[INFO] Scanning feature files: {source_dir}")
    all_files = []
    for file in source_dir.glob("*.parquet"):
        if not file.name.endswith(".bak"):
            all_files.append(file)
    
    print(f"[INFO] Found {len(all_files)} feature files")
    
    if len(all_files) == 0:
        print("[ERROR] No feature files found")
        return
    
    # Shuffle the file list
    random.seed(42)  # Set random seed for reproducibility
    random.shuffle(all_files)
    
    # Split by 80/20 ratio
    total_files = len(all_files)
    train_size = int(total_files * 0.8)
    test_size = total_files - train_size
    
    train_files = all_files[:train_size]
    test_files = all_files[train_size:]
    
    print(f"\n[INFO] Split result:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Test: {len(test_files)} files ({len(test_files)/total_files*100:.1f}%)")
    
    # Copy training files
    print(f"\n[INFO] Copying training files to {train_dir}...")
    failed_files = []
    for file in tqdm(train_files, desc="Copy train"):
        dest = train_dir / file.name
        if not dest.exists():  # Avoid duplicate copies
            try:
                # Check file readability
                if not os.access(file, os.R_OK):
                    failed_files.append((file.name, "Permission denied: cannot read"))
                    continue
                shutil.copy2(file, dest)
            except PermissionError as e:
                failed_files.append((file.name, f"Permission error: {str(e)}"))
                print(f"\n[WARN] Skipped file (permission denied): {file.name}")
            except Exception as e:
                failed_files.append((file.name, f"Error: {str(e)}"))
                print(f"\n[WARN] Skipped file: {file.name} - {str(e)}")
        else:
            print(f"[WARN] File already exists, skipping: {dest.name}")
    
    # Copy test files
    print(f"\n[INFO] Copying test files to {test_dir}...")
    for file in tqdm(test_files, desc="Copy test"):
        dest = test_dir / file.name
        if not dest.exists():  # Avoid duplicate copies
            try:
                # Check file readability
                if not os.access(file, os.R_OK):
                    failed_files.append((file.name, "Permission denied: cannot read"))
                    continue
                shutil.copy2(file, dest)
            except PermissionError as e:
                failed_files.append((file.name, f"Permission error: {str(e)}"))
                print(f"\n[WARN] Skipped file (permission denied): {file.name}")
            except Exception as e:
                failed_files.append((file.name, f"Error: {str(e)}"))
                print(f"\n[WARN] Skipped file: {file.name} - {str(e)}")
        else:
            print(f"[WARN] File already exists, skipping: {dest.name}")
    
    # Report failed files
    if failed_files:
        print(f"\n[WARN] {len(failed_files)} files could not be copied due to permissions")
        print("[INFO] Suggested commands to fix permissions:")
        print(f"  sudo chmod -R 644 {source_dir}/*.parquet")
        print("  Or run this script with sudo: sudo python3 split_features.py")
    
    # Verify results
    train_count = len(list(train_dir.glob("*.parquet")))
    test_count = len(list(test_dir.glob("*.parquet")))
    
    print(f"\n[INFO] Split completed!")
    print(f"  Train dir: {train_dir} ({train_count} files)")
    print(f"  Test dir: {test_dir} ({test_count} files)")
    print(f"  Total: {train_count + test_count} files")
    
    # Show first few file names as examples
    print(f"\n[INFO] First 5 train files:")
    for file in train_files[:5]:
        print(f"  {file.name}")
    
    print(f"\n[INFO] First 5 test files:")
    for file in test_files[:5]:
        print(f"  {file.name}")

if __name__ == "__main__":
    split_features()
