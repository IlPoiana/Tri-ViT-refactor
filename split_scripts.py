import os
import shutil
import random
import argparse
from pathlib import Path

source_dir = "../IXI"
output_dir = "./data/"

def split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    move_files=False
):
    """
    Splits files from source_dir into train/val/test directories.

    Args:
        source_dir (str): Path to directory containing data files.
        output_dir (str): Path where split folders will be created.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        seed (int): Random seed for reproducibility.
        move_files (bool): If True, files are moved instead of copied.
    """

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if total != 1.0:
        raise ValueError("Train/Val/Test ratios must sum to 1.0")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    # Collect all files (ignore subdirectories)
    files = [f for f in source_path.iterdir() if f.is_file()]
    if len(files) == 0:
        raise ValueError("No files found in source directory.")

    # Shuffle
    random.seed(seed)
    random.shuffle(files)

    # Compute split indices
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:(n_train + n_val)]
    test_files = files[(n_train + n_val):]

    # Create output directories
    train_dir = output_path / "IXI_train"
    val_dir = output_path / "IXI_validate"
    test_dir = output_path / "IXI_test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy or move files
    operation = shutil.move if move_files else shutil.copy2

    for f in train_files:
        operation(f, train_dir / f.name)

    for f in val_files:
        operation(f, val_dir / f.name)

    for f in test_files:
        operation(f, test_dir / f.name)

    print(f"Total files: {n_total}")
    print(f"Train: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    print("Split complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into 70/15/15.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    split_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        move_files=args.move
    )
