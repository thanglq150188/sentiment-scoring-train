#!/bin/bash
# LoRA Fine-tuning Script - Optimized for A6000 48GB
# Maximum performance configuration

python train.py \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --dataset_path "./data/train.jsonl" \
  --val_dataset_path "./data/val.jsonl" \
  --test_dataset_path "./data/test.jsonl" \
  --max_seq_length 4096 \
  --use_lora \
  --dtype bfloat16 \
  --lora_r 64 \
  --lora_alpha 128 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --output_dir "./outputs/Qwen/Qwen2.5-0.5B-Instruct" \
  --save_steps 50 \
  --save_model \
  --save_model_path "./models/Qwen/Qwen2.5-0.5B-Instruct-lora" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 5
