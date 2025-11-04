#!/bin/bash
# Full Fine-tuning Script
# Trains all parameters - use for small models only

python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing \
  --output_dir "./outputs/full" \
  --save_model \
  --save_model_path "./models/qwen_full" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 5
