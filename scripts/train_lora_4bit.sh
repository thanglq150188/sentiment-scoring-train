#!/bin/bash
# LoRA + 4-bit Quantization Script
# Most memory efficient - for very large models (7B+)

python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --dataset_path "./data/train.jsonl" \
  --val_dataset_path "./data/val.jsonl" \
  --test_dataset_path "./data/test.jsonl" \
  --max_seq_length 4096 \
  --use_lora \
  --load_in_4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing \
  --output_dir "./outputs/lora_4bit" \
  --save_steps 100 \
  --save_model \
  --save_model_path "./models/llama2_lora_4bit" \
  --save_merged_16bit \
  --test_after_training \
  --logging_steps 10
