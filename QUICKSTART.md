# Quick Start Guide

Get started with training in 5 minutes!

## 1. Install Dependencies

```bash
pip install torch transformers datasets trl unsloth
```

## 2. Test the Pipeline (30 seconds)

```bash
./scripts/train_quick_test.sh
```

This runs a quick test with the sample dataset to verify everything works.

## 3. Train Your Model

### Basic LoRA Training (Recommended)

```bash
./scripts/train_lora.sh
```

Edit the script to point to your dataset:
- Change `--dataset_path` to your data file
- Adjust `--model_name` if using a different model

### Your Dataset Format

Create a JSONL file with this format:

```json
{"messages": [{"role": "user", "content": "Your question"}, {"role": "assistant", "content": "The answer"}]}
{"messages": [{"role": "user", "content": "Another question"}, {"role": "assistant", "content": "Another answer"}]}
```

## 4. Common Use Cases

### Small Dataset (< 1000 samples)
```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "your_data.jsonl" \
  --use_lora \
  --num_train_epochs 5 \
  --save_model \
  --save_model_path "./my_model"
```

### Large Dataset (> 10k samples)
```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "your_data.jsonl" \
  --use_lora \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --save_model
```

### Large Model (7B+)
```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --dataset_path "your_data.jsonl" \
  --use_lora \
  --load_in_4bit \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --save_model
```

## 5. Monitor Training

Watch the loss decrease:
```
Step: 1/60 | Loss: 2.45
Step: 2/60 | Loss: 2.31
Step: 3/60 | Loss: 2.18
...
```

Lower loss = better training

## 6. Test Your Model

Add `--test_after_training` to any command:

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "your_data.jsonl" \
  --use_lora \
  --test_after_training
```

## Quick Parameter Guide

### Must Change
- `--model_name`: Your base model
- `--dataset_path`: Your training data

### Commonly Adjusted
- `--num_train_epochs`: More epochs = more training (default: 3)
- `--learning_rate`: How fast to learn (default: 2e-4)
- `--per_device_train_batch_size`: Batch size (reduce if OOM)

### For Memory Issues
- `--load_in_4bit`: Use 4-bit quantization
- `--per_device_train_batch_size 1`: Minimum batch size
- `--gradient_accumulation_steps 16`: Compensate for small batch

## Troubleshooting

### "CUDA out of memory"
```bash
# Add these flags
--load_in_4bit \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing
```

### "Training is too slow"
```bash
# Increase batch size
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2
```

### "Model quality is poor"
```bash
# Train longer
--num_train_epochs 5

# Or adjust learning rate
--learning_rate 3e-4
```

## Next Steps

1. Read the [full README](README.md) for all options
2. Check [scripts/README.md](scripts/README.md) for more examples
3. Customize scripts for your needs
4. Monitor training and adjust parameters

## Getting Help

Run with `--help` to see all options:
```bash
python train.py --help
```

Happy training!
