#!/bin/bash
# Training with HuggingFace Hub Upload
# Set your HF_TOKEN environment variable first

if [ -z "$HF_TOKEN" ]; then
    echo "Error: Please set HF_TOKEN environment variable"
    echo "export HF_TOKEN=your_token_here"
    exit 1
fi

if [ -z "$HF_MODEL_ID" ]; then
    echo "Error: Please set HF_MODEL_ID environment variable"
    echo "export HF_MODEL_ID=username/model-name"
    exit 1
fi

python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./data/train.jsonl" \
  --val_dataset_path "./data/val.jsonl" \
  --test_dataset_path "./data/test.jsonl" \
  --max_seq_length 4096 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --output_dir "./outputs/hub_upload" \
  --save_steps 100 \
  --save_model \
  --save_model_path "./models/for_hub" \
  --save_merged_16bit \
  --push_to_hub \
  --hub_model_id "$HF_MODEL_ID" \
  --hub_token "$HF_TOKEN" \
  --test_after_training \
  --logging_steps 10
