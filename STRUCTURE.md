# Project Structure

## Overview

This is a well-organized, modular Python package for LLM fine-tuning.

```
llm-finetuning/
├── src/                      # Main package
│   ├── __init__.py          # Package initialization
│   ├── __main__.py          # Module entry point
│   ├── cli.py               # Command-line interface
│   ├── dataset.py           # Dataset loading & formatting
│   ├── model.py             # Model operations
│   └── trainer.py           # Training & inference
│
├── scripts/                  # Ready-to-use bash scripts
│   ├── train_lora.sh        # LoRA training
│   ├── train_full.sh        # Full fine-tuning
│   ├── train_lora_4bit.sh   # 4-bit quantized training
│   ├── train_quick_test.sh  # Quick pipeline test
│   ├── train_with_hub.sh    # Train + HF Hub upload
│   └── README.md            # Script documentation
│
├── train.py                  # Simple entry point (calls src.cli)
├── setup.py                  # Package installation config
├── sample_dataset.jsonl      # Example dataset
│
├── README.md                 # Main documentation
├── QUICKSTART.md            # Quick start guide
├── README_TRAINING.md       # Detailed training guide
└── STRUCTURE.md             # This file
```

## Module Organization

### src/cli.py
- Main command-line interface
- Argument parsing
- Orchestrates the training pipeline
- Entry point for all training operations

### src/dataset.py
- `load_data()` - Load from files or HuggingFace
- `prepare_dataset()` - Format for training
- `get_chat_template()` - Chat template management
- `format_messages_to_text()` - Message conversion

### src/model.py
- `load_model_and_tokenizer()` - Model initialization
- `save_model()` - Model persistence
- Handles both LoRA and full fine-tuning configurations

### src/trainer.py
- `create_training_args()` - Training configuration
- `train_model()` - Training loop
- `test_model()` - Inference testing

## Usage Methods

### 1. Direct Script Execution (Recommended)
```bash
python train.py --model_name "Qwen/Qwen3-0.6B" --dataset_path "data.jsonl" --use_lora
```

### 2. Bash Scripts
```bash
./scripts/train_lora.sh
```

### 3. Python Module
```bash
python -m src --model_name "Qwen/Qwen3-0.6B" --dataset_path "data.jsonl"
```

### 4. Installed Command (after setup.py install)
```bash
pip install -e .
llm-train --model_name "Qwen/Qwen3-0.6B" --dataset_path "data.jsonl"
```

### 5. Programmatic Usage
```python
from src.dataset import load_data, prepare_dataset
from src.model import load_model_and_tokenizer
from src.trainer import train_model, create_training_args

# Use modules directly in your code
dataset = load_data("data.jsonl")
model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-0.6B", use_lora=True)
training_args = create_training_args(num_train_epochs=3)
train_model(model, tokenizer, dataset, training_args)
```

## Why This Structure?

### Benefits:

1. **Modularity**: Each file has a single responsibility
   - Easy to find and modify specific functionality
   - Can import and use individual modules

2. **Maintainability**: Clear separation of concerns
   - CLI logic separate from training logic
   - Dataset handling separate from model operations

3. **Testability**: Each module can be tested independently
   ```python
   from src.dataset import format_messages_to_text
   # Test just the formatting function
   ```

4. **Reusability**: Use modules in other projects
   ```python
   # Use just the dataset module
   from src.dataset import load_data
   ```

5. **Professional**: Follows Python best practices
   - Standard package structure
   - Can be installed with pip
   - Can be published to PyPI

6. **Flexibility**: Multiple ways to use
   - Command-line for quick use
   - Scripts for common patterns
   - Module imports for custom workflows

## Installation

### Development Mode (Recommended)
```bash
pip install -e .
```

This installs the package in editable mode, so changes are immediately available.

### Regular Installation
```bash
pip install .
```

### Manual (No Installation)
```bash
# Just run train.py - it works without installation
python train.py --help
```

## Adding New Features

### Adding a New Training Method
1. Add function to `src/trainer.py`
2. Add CLI argument to `src/cli.py`
3. Call function in main pipeline

### Adding a New Dataset Format
1. Add loader function to `src/dataset.py`
2. Update `load_data()` to handle new format
3. Document in README

### Adding a New Script
1. Create script in `scripts/`
2. Make executable: `chmod +x scripts/your_script.sh`
3. Document in `scripts/README.md`

## Development Workflow

```bash
# Edit source files
vim src/trainer.py

# No need to reinstall if using -e
python train.py --help  # Changes are live

# Run tests
python -m pytest tests/  # (if you add tests)

# Run training
./scripts/train_quick_test.sh
```

## Best Practices

1. **Keep train.py minimal** - It's just an entry point
2. **Put logic in src/** - All functionality goes in modules
3. **Use scripts/** - For common use cases
4. **Document changes** - Update READMEs when adding features
5. **Test before commit** - Run quick test to verify changes

## Comparison to Old Structure

### Before (Monolithic)
```
train.py (600+ lines, everything mixed together)
```

### After (Modular)
```
train.py (5 lines, entry point only)
src/cli.py (400 lines, CLI logic)
src/dataset.py (130 lines, dataset handling)
src/model.py (90 lines, model operations)
src/trainer.py (140 lines, training logic)
```

### Advantages of New Structure
- Easier to find code
- Easier to test
- Easier to reuse
- More professional
- Better for collaboration
- Can be packaged and distributed
