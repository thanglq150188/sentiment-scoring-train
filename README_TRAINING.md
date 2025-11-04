# LLM Fine-tuning Training Script

A comprehensive training script for fine-tuning LLMs using Unsloth with support for both LoRA and full fine-tuning methods. Supports OpenAI message format for datasets.

## Features

- Support for both LoRA (parameter-efficient) and full fine-tuning
- OpenAI message format compatibility
- Multiple chat templates (Qwen, Llama, auto-detect)
- Comprehensive CLI with standard training parameters
- Dataset loading from local files (JSON/JSONL) or HuggingFace
- Model saving with merged 16-bit option
- HuggingFace Hub integration
- Post-training inference testing

## Installation

```bash
pip install torch transformers datasets trl unsloth
```

## Dataset Format

The script expects datasets in OpenAI message format where each sample contains a `messages` field with a list of message dictionaries:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Supported file formats:
- **JSONL**: One JSON object per line
- **JSON**: Array of objects or dict with split keys

See [sample_dataset.jsonl](sample_dataset.jsonl) for an example.

## Usage Examples

### 1. Basic LoRA Fine-tuning

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-4 \
  --output_dir "./outputs" \
  --save_model \
  --save_model_path "./my_finetuned_model"
```

### 2. Full Fine-tuning (All Parameters)

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --output_dir "./outputs" \
  --save_model \
  --save_model_path "./my_finetuned_model"
```

### 3. LoRA with 4-bit Quantization (Memory Efficient)

```bash
python train.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --load_in_4bit \
  --lora_r 16 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --output_dir "./outputs" \
  --save_model \
  --save_model_path "./llama2_lora"
```

### 4. Training with Testing and Saving to Hub

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 2e-4 \
  --output_dir "./outputs" \
  --save_model \
  --save_model_path "./my_model" \
  --save_merged_16bit \
  --test_after_training \
  --push_to_hub \
  --hub_model_id "username/model-name" \
  --hub_token "hf_..."
```

### 5. Custom LoRA Configuration

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "./sample_dataset.jsonl" \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --output_dir "./outputs" \
  --save_model \
  --save_model_path "./custom_lora_model"
```

### 6. HuggingFace Dataset

```bash
python train.py \
  --model_name "Qwen/Qwen3-0.6B" \
  --dataset_path "tatsu-lab/alpaca" \
  --dataset_split "train" \
  --dataset_text_field "conversations" \
  --use_lora \
  --num_train_epochs 3 \
  --output_dir "./outputs"
```

## Key Parameters

### Model Configuration
- `--model_name`: HuggingFace model identifier (required)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--load_in_4bit`: Enable 4-bit quantization
- `--dtype`: Data type (auto, float16, bfloat16, float32)

### Training Method
- `--use_lora`: Enable LoRA fine-tuning
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 16)
- `--lora_dropout`: LoRA dropout (default: 0.0)
- `--lora_target_modules`: Target modules for LoRA

### Dataset
- `--dataset_path`: Path to dataset file or HF dataset name (required)
- `--dataset_split`: Dataset split (default: train)
- `--dataset_text_field`: Field containing messages (default: messages)
- `--system_prompt`: Default system prompt
- `--chat_template`: Chat template (qwen, llama, auto)

### Training
- `--num_train_epochs`: Number of epochs (default: 3)
- `--max_steps`: Max training steps (overrides epochs if set)
- `--per_device_train_batch_size`: Batch size per device (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--warmup_steps`: Warmup steps (default: 10)
- `--warmup_ratio`: Warmup ratio (overrides warmup_steps)
- `--lr_scheduler_type`: LR scheduler type
- `--weight_decay`: Weight decay (default: 0.01)
- `--optim`: Optimizer (default: adamw_8bit)
- `--gradient_checkpointing`: Enable gradient checkpointing

### Output
- `--output_dir`: Output directory for checkpoints
- `--save_model`: Save final model
- `--save_model_path`: Path to save model
- `--save_merged_16bit`: Save merged 16-bit version
- `--push_to_hub`: Push to HuggingFace Hub
- `--hub_model_id`: HF Hub model ID
- `--hub_token`: HF Hub token

### Testing
- `--test_after_training`: Test model after training
- `--test_prompt`: Custom test prompt
- `--max_new_tokens`: Max tokens for generation (default: 128)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Top-p sampling (default: 0.9)

## LoRA vs Full Fine-tuning

### LoRA (Recommended for large models)
- Faster training
- Lower memory usage
- Only trains adapter weights (~few MB)
- Good performance with proper configuration
- Use with `--use_lora` flag

### Full Fine-tuning
- Trains all model parameters
- Higher memory requirements
- Better performance on some tasks
- Recommended only for smaller models (< 1B parameters)
- Don't use `--use_lora` flag

## Tips

1. **For large models (7B+)**: Use LoRA with 4-bit quantization
   ```bash
   --use_lora --load_in_4bit --lora_r 16
   ```

2. **For small models (< 1B)**: Full fine-tuning works well
   ```bash
   --per_device_train_batch_size 1 --gradient_accumulation_steps 8
   ```

3. **Memory issues**: Reduce batch size and increase gradient accumulation
   ```bash
   --per_device_train_batch_size 1 --gradient_accumulation_steps 16
   ```

4. **Faster training**: Enable packing for short sequences
   ```bash
   --packing
   ```

5. **Better convergence**: Use cosine scheduler with warmup
   ```bash
   --lr_scheduler_type cosine --warmup_ratio 0.1
   ```

## Output Structure

```
outputs/
├── checkpoint-500/
├── checkpoint-1000/
└── ...

final_model/
├── config.json
├── tokenizer.json
├── adapter_model.bin  (if LoRA)
└── pytorch_model.bin  (if full fine-tuning)

final_model_16bit/     (if --save_merged_16bit)
└── ...
```

## Troubleshooting

### Out of Memory
- Reduce `--per_device_train_batch_size` to 1
- Increase `--gradient_accumulation_steps`
- Use `--load_in_4bit`
- Enable `--gradient_checkpointing`

### Slow Training
- Increase `--per_device_train_batch_size`
- Enable `--packing` for short sequences
- Use `--load_in_4bit` with LoRA

### Poor Performance
- Increase training epochs
- Adjust learning rate (try 1e-4 to 5e-4 for LoRA)
- Increase LoRA rank (`--lora_r 32` or `--lora_r 64`)
- Use more training data

## License

MIT License
