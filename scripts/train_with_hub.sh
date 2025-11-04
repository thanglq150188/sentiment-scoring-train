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
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --output_dir "./outputs/hub_upload" \
  --save_model \
  --save_model_path "./models/for_hub" \
  --save_merged_16bit \
  --push_to_hub \
  --hub_model_id "$HF_MODEL_ID" \
  --hub_token "$HF_TOKEN" \
  --test_after_training
