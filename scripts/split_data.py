#!/usr/bin/env python3
"""
Split training data into train/validation/test sets while maintaining
the ratio of co_vi_pham/ko_vi_pham across all splits.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = Path("data")
co_vi_pham_file = data_dir / "training_data_co_vi_pham_audited.jsonl"
ko_vi_pham_file = data_dir / "training_data_ko_vi_pham_audited.jsonl"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"


def load_jsonl(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_data(data, train_ratio, val_ratio, test_ratio):
    """Split data into train/val/test sets."""
    # Shuffle data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    n = len(shuffled_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled_data[:train_end]
    val = shuffled_data[train_end:val_end]
    test = shuffled_data[val_end:]

    return train, val, test


def main():
    print("Loading data...")
    co_vi_pham_data = load_jsonl(co_vi_pham_file)
    ko_vi_pham_data = load_jsonl(ko_vi_pham_file)

    print(f"\nOriginal data:")
    print(f"  Co vi pham: {len(co_vi_pham_data)} samples")
    print(f"  Ko vi pham: {len(ko_vi_pham_data)} samples")
    print(f"  Total: {len(co_vi_pham_data) + len(ko_vi_pham_data)} samples")
    print(f"  Ratio (co/ko): {len(co_vi_pham_data) / len(ko_vi_pham_data):.4f}")

    # Split each class separately to maintain ratio
    print(f"\nSplitting data (train: {TRAIN_RATIO}, val: {VAL_RATIO}, test: {TEST_RATIO})...")
    co_train, co_val, co_test = split_data(co_vi_pham_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    ko_train, ko_val, ko_test = split_data(ko_vi_pham_data, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)

    # Combine and shuffle each split
    train_data = co_train + ko_train
    val_data = co_val + ko_val
    test_data = co_test + ko_test

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Print statistics
    print(f"\nSplit statistics:")
    print(f"\nTrain set: {len(train_data)} samples")
    print(f"  Co vi pham: {len(co_train)} samples ({len(co_train)/len(train_data)*100:.2f}%)")
    print(f"  Ko vi pham: {len(ko_train)} samples ({len(ko_train)/len(train_data)*100:.2f}%)")
    print(f"  Ratio (co/ko): {len(co_train) / len(ko_train):.4f}")

    print(f"\nValidation set: {len(val_data)} samples")
    print(f"  Co vi pham: {len(co_val)} samples ({len(co_val)/len(val_data)*100:.2f}%)")
    print(f"  Ko vi pham: {len(ko_val)} samples ({len(ko_val)/len(val_data)*100:.2f}%)")
    print(f"  Ratio (co/ko): {len(co_val) / len(ko_val):.4f}")

    print(f"\nTest set: {len(test_data)} samples")
    print(f"  Co vi pham: {len(co_test)} samples ({len(co_test)/len(test_data)*100:.2f}%)")
    print(f"  Ko vi pham: {len(ko_test)} samples ({len(ko_test)/len(test_data)*100:.2f}%)")
    print(f"  Ratio (co/ko): {len(co_test) / len(ko_test):.4f}")

    # Save splits
    print(f"\nSaving splits...")
    save_jsonl(train_data, data_dir / "train.jsonl")
    save_jsonl(val_data, data_dir / "val.jsonl")
    save_jsonl(test_data, data_dir / "test.jsonl")

    print(f"\nDone! Files saved:")
    print(f"  - {data_dir / 'train.jsonl'}")
    print(f"  - {data_dir / 'val.jsonl'}")
    print(f"  - {data_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
