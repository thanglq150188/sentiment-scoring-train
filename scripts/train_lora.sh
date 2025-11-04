#!/bin/bash
# LoRA Fine-tuning Script
# Memory efficient training for large models

python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --warmup_steps 10 \
  --lr_scheduler_type "linear" \
  --output_dir "./outputs/lora" \
  --save_model \
  --save_model_path "./models/qwen_lora" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 5
