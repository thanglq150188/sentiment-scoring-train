# Training Scripts

This directory contains ready-to-use bash scripts for different training scenarios.

## Available Scripts

### 1. train_lora.sh
**LoRA Fine-tuning** - Recommended for most use cases
- Uses parameter-efficient LoRA
- Balanced memory usage and performance
- Good for models up to 7B parameters

```bash
./scripts/train_lora.sh
```

### 2. train_full.sh
**Full Fine-tuning** - For small models
- Trains all model parameters
- Higher memory requirements
- Best for models under 1B parameters

```bash
./scripts/train_full.sh
```

### 3. train_lora_4bit.sh
**LoRA + 4-bit Quantization** - For large models
- Most memory efficient
- Use for 7B+ parameter models
- Requires less GPU memory

```bash
./scripts/train_lora_4bit.sh
```

### 4. train_quick_test.sh
**Quick Test** - For testing the pipeline
- Runs for only 20 steps
- Fast validation of setup
- No model saving

```bash
./scripts/train_quick_test.sh
```

### 5. train_with_hub.sh
**Training with HuggingFace Hub Upload**
- Trains and uploads to HF Hub
- Requires HF token and model ID

```bash
export HF_TOKEN="your_hf_token"
export HF_MODEL_ID="username/model-name"
./scripts/train_with_hub.sh
```

## Customization

To customize any script:

1. Edit the script file
2. Modify parameters as needed:
   - `--model_name`: Change the base model
   - `--dataset_path`: Point to your dataset
   - `--num_train_epochs`: Adjust training duration
   - `--learning_rate`: Tune learning rate
   - `--per_device_train_batch_size`: Adjust for your GPU

## Example: Custom Training

Create your own script:

```bash
#!/bin/bash
python train.py \
  --model_name "your/model" \
  --dataset_path "your/data.jsonl" \
  --use_lora \
  --num_train_epochs 5 \
  --learning_rate 3e-4 \
  --output_dir "./outputs/custom" \
  --save_model \
  --save_model_path "./models/custom"
```

## GPU Memory Guide

| Script | Model Size | Minimum GPU Memory |
|--------|------------|-------------------|
| train_quick_test.sh | 0.6B | 4GB |
| train_lora.sh | 0.6B - 3B | 8GB |
| train_full.sh | 0.6B - 1B | 16GB |
| train_lora_4bit.sh | 7B - 13B | 16GB |

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16
```

### Slow Training
Increase batch size if you have memory:
```bash
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2
```

### Poor Results
- Increase training epochs
- Try different learning rates
- Increase LoRA rank (--lora_r 32)
