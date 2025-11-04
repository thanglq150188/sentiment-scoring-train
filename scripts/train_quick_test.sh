#!/bin/bash
# Quick Test Training Script
# Fast training for testing the pipeline

python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --lora_r 8 \
  --max_steps 20 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --output_dir "./outputs/quick_test" \
  --save_strategy "no" \
  --test_after_training \
  --logging_steps 1
