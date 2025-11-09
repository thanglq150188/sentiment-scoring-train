#!/bin/bash
# Full Fine-tuning Script
# Trains all parameters - use for small models only
# Note: Qwen3 has issues with gradient_accumulation, using batch_size only

python train.py \
  --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset_path "./data/train.jsonl" \
  --val_dataset_path "./data/val.jsonl" \
  --test_dataset_path "./data/test.jsonl" \
  --max_seq_length 4096 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing \
  --output_dir "./outputs/Qwen/Qwen2.5-1.5B-Instruct" \
  --save_steps 500 \
  --save_model \
  --save_model_path "./models/Qwen/Qwen2.5-1.5B-Instruct" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 5
