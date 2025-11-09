#!/usr/bin/env python3
"""
Analyze dataset to determine optimal training parameters using actual tokenizer
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_data, prepare_dataset
from transformers import AutoTokenizer


def analyze_dataset_with_tokenizer(
    dataset_path: str,
    model_name: str,
    dataset_name: str = "dataset",
    system_prompt: str = "You are a helpful assistant.",
    chat_template: str = "auto"
):
    """Analyze dataset using actual tokenizer"""
    print(f"\n{'='*70}")
    print(f"Analyzing {dataset_name} with {model_name} tokenizer")
    print(f"{'='*70}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load and prepare dataset
    dataset = load_data(dataset_path, dataset_name=dataset_name)
    formatted_dataset = prepare_dataset(
        dataset,
        model_name=model_name,
        text_field="messages",
        system_prompt=system_prompt,
        chat_template=chat_template
    )

    # Tokenize and analyze
    print(f"\n{'─'*70}")
    print("Tokenizing dataset (this may take a moment)...")
    print(f"{'─'*70}")

    token_lengths = []
    for i, example in enumerate(formatted_dataset):
        text = example["text"]
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))

        if i < 3:  # Show first 3 examples
            print(f"\nExample {i+1}:")
            print(f"  Text length: {len(text)} chars")
            print(f"  Token count: {len(tokens)} tokens")
            print(f"  First 200 chars: {text[:200]}...")

    # Calculate statistics
    token_lengths = np.array(token_lengths)

    print(f"\n{'='*70}")
    print(f"TOKEN LENGTH STATISTICS - {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Total samples: {len(token_lengths)}")
    print(f"\nBasic Statistics:")
    print(f"  Mean:   {np.mean(token_lengths):.1f} tokens")
    print(f"  Median: {np.median(token_lengths):.0f} tokens")
    print(f"  Std:    {np.std(token_lengths):.1f} tokens")
    print(f"  Min:    {np.min(token_lengths)} tokens")
    print(f"  Max:    {np.max(token_lengths)} tokens")

    print(f"\nPercentiles:")
    percentiles = [50, 75, 90, 95, 97, 99, 99.5, 99.9]
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        count = np.sum(token_lengths <= value)
        percentage = (count / len(token_lengths)) * 100
        print(f"  {p:5.1f}th: {value:6.0f} tokens (covers {count:5d}/{len(token_lengths)} = {percentage:.1f}%)")

    # Distribution by common seq lengths
    print(f"\nCoverage by common max_seq_length values:")
    for max_len in [512, 1024, 2048, 4096, 8192]:
        count = np.sum(token_lengths <= max_len)
        percentage = (count / len(token_lengths)) * 100
        truncated = len(token_lengths) - count
        print(f"  {max_len:5d}: {count:5d}/{len(token_lengths)} samples ({percentage:5.1f}%) - {truncated:4d} would be truncated")

    return {
        'dataset_name': dataset_name,
        'total_samples': len(token_lengths),
        'mean': float(np.mean(token_lengths)),
        'median': float(np.median(token_lengths)),
        'std': float(np.std(token_lengths)),
        'min': int(np.min(token_lengths)),
        'max': int(np.max(token_lengths)),
        'p50': float(np.percentile(token_lengths, 50)),
        'p75': float(np.percentile(token_lengths, 75)),
        'p90': float(np.percentile(token_lengths, 90)),
        'p95': float(np.percentile(token_lengths, 95)),
        'p99': float(np.percentile(token_lengths, 99)),
        'p99_5': float(np.percentile(token_lengths, 99.5)),
        'token_lengths': token_lengths
    }


def recommend_params(train_stats, val_stats, test_stats):
    """Recommend training parameters based on all datasets"""
    print(f"\n{'='*70}")
    print("TRAINING PARAMETER RECOMMENDATIONS")
    print(f"{'='*70}")

    # Combine all stats
    all_stats = [train_stats, val_stats, test_stats]
    p95_max = max(s['p95'] for s in all_stats)
    p99_max = max(s['p99'] for s in all_stats)
    p99_5_max = max(s['p99_5'] for s in all_stats)

    print(f"\nCombined statistics across all splits:")
    print(f"  95th percentile:   {p95_max:.0f} tokens")
    print(f"  99th percentile:   {p99_max:.0f} tokens")
    print(f"  99.5th percentile: {p99_5_max:.0f} tokens")

    # Recommend max_seq_length
    print(f"\n{'─'*70}")
    print("1. MAX_SEQ_LENGTH RECOMMENDATION:")
    print(f"{'─'*70}")

    if p99_max <= 512:
        recommended = 512
        reasoning = "99% of samples fit in 512 tokens"
    elif p99_max <= 768:
        recommended = 1024
        reasoning = "99% fit in ≤768, use 1024 for safety"
    elif p99_max <= 1024:
        recommended = 1024
        reasoning = "99% of samples fit in 1024 tokens"
    elif p99_max <= 1536:
        recommended = 2048
        reasoning = "99% fit in ≤1536, use 2048 for safety"
    elif p99_max <= 2048:
        recommended = 2048
        reasoning = "99% of samples fit in 2048 tokens"
    elif p99_max <= 3072:
        recommended = 4096
        reasoning = "99% fit in ≤3072, use 4096 for safety"
    else:
        recommended = 4096
        reasoning = "Samples are long, 4096 balances coverage and memory"

    print(f"\n  RECOMMENDED: {recommended} tokens")
    print(f"  Reasoning: {reasoning}")

    # Calculate actual coverage
    for stats in all_stats:
        covered = np.sum(stats['token_lengths'] <= recommended)
        total = len(stats['token_lengths'])
        pct = (covered / total) * 100
        truncated = total - covered
        print(f"  - {stats['dataset_name']:12s}: {covered:5d}/{total:5d} ({pct:5.1f}%) - {truncated:3d} truncated")

    # Batch size recommendations
    total_samples = train_stats['total_samples']
    print(f"\n{'─'*70}")
    print("2. BATCH SIZE RECOMMENDATION:")
    print(f"{'─'*70}")
    print(f"  Training samples: {total_samples}")
    print(f"  Target effective batch size: 16-32 (good for most tasks)")

    if recommended <= 512:
        per_device = 8
        grad_accum = 4
    elif recommended <= 1024:
        per_device = 4
        grad_accum = 4
    elif recommended <= 2048:
        per_device = 2
        grad_accum = 8
    else:
        per_device = 1
        grad_accum = 16

    effective = per_device * grad_accum
    steps_per_epoch = total_samples // effective

    print(f"\n  For max_seq_length={recommended}:")
    print(f"    per_device_train_batch_size: {per_device}")
    print(f"    gradient_accumulation_steps:  {grad_accum}")
    print(f"    Effective batch size:         {effective}")
    print(f"    Steps per epoch (approx):     {steps_per_epoch}")

    # Learning rate
    print(f"\n{'─'*70}")
    print("3. LEARNING RATE:")
    print(f"{'─'*70}")
    print(f"  LoRA training:        2e-4 (current default is optimal)")
    print(f"  Full fine-tuning:     2e-5 (current default is optimal)")

    # Warmup and scheduler
    total_steps = steps_per_epoch * 3  # Assuming 3 epochs
    warmup_steps = int(total_steps * 0.1)

    print(f"\n{'─'*70}")
    print("4. WARMUP & SCHEDULER:")
    print(f"{'─'*70}")
    print(f"  Total training steps (3 epochs): ~{total_steps}")
    print(f"  Recommended warmup_steps: {warmup_steps} (10% of total)")
    print(f"  OR use warmup_ratio: 0.1")
    print(f"  Scheduler: 'cosine' or 'linear' (both work well)")

    # Save frequency
    save_steps = max(100, steps_per_epoch // 4)
    print(f"\n{'─'*70}")
    print("5. CHECKPOINTING:")
    print(f"{'─'*70}")
    print(f"  Recommended save_steps: {save_steps} (~4 saves per epoch)")
    print(f"  Recommended eval_steps: {save_steps} (evaluate when saving)")

    # Final summary
    print(f"\n{'='*70}")
    print("SUGGESTED COMMAND LINE PARAMETERS:")
    print(f"{'='*70}")
    print(f"\nFor LoRA training (recommended for large models):")
    print(f"  --max_seq_length {recommended} \\")
    print(f"  --per_device_train_batch_size {per_device} \\")
    print(f"  --gradient_accumulation_steps {grad_accum} \\")
    print(f"  --learning_rate 2e-4 \\")
    print(f"  --warmup_ratio 0.1 \\")
    print(f"  --lr_scheduler_type cosine \\")
    print(f"  --num_train_epochs 3 \\")
    print(f"  --save_steps {save_steps} \\")
    print(f"  --logging_steps 10")

    print(f"\nFor Full fine-tuning (small models only):")
    print(f"  --max_seq_length {recommended} \\")
    print(f"  --per_device_train_batch_size {max(1, per_device // 2)} \\")
    print(f"  --gradient_accumulation_steps {grad_accum * 2} \\")
    print(f"  --learning_rate 2e-5 \\")
    print(f"  --warmup_ratio 0.1 \\")
    print(f"  --lr_scheduler_type cosine \\")
    print(f"  --num_train_epochs 3 \\")
    print(f"  --save_steps {save_steps}")

    return recommended, per_device, grad_accum


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset and recommend training parameters"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Model name for tokenizer (default: Qwen/Qwen2.5-0.5B)"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/train.jsonl",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="./data/val.jsonl",
        help="Path to validation dataset"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="./data/test.jsonl",
        help="Path to test dataset"
    )

    args = parser.parse_args()

    # Analyze all datasets
    train_stats = analyze_dataset_with_tokenizer(
        args.train_path,
        args.model_name,
        dataset_name="train"
    )

    val_stats = analyze_dataset_with_tokenizer(
        args.val_path,
        args.model_name,
        dataset_name="validation"
    )

    test_stats = analyze_dataset_with_tokenizer(
        args.test_path,
        args.model_name,
        dataset_name="test"
    )

    # Get recommendations
    recommend_params(train_stats, val_stats, test_stats)


if __name__ == "__main__":
    main()
