#!/bin/bash
# LoRA + 4-bit Quantization Script
# Most memory efficient - for very large models (7B+)

python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --load_in_4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --warmup_steps 20 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing \
  --output_dir "./outputs/lora_4bit" \
  --save_model \
  --save_model_path "./models/llama2_lora_4bit" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 10
