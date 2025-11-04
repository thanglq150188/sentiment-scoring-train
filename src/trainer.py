"""Training and inference utilities"""

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


def create_training_args(
    output_dir: str = "./outputs",
    num_train_epochs: int = 3,
    max_steps: int = -1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "linear",
    weight_decay: float = 0.01,
    optim: str = "adamw_8bit",
    logging_steps: int = 1,
    save_steps: int = 500,
    save_strategy: str = "steps",
    save_total_limit: int = 2,
    seed: int = 3407,
    dtype: str = "auto"
) -> TrainingArguments:
    """Create training arguments"""
    print(f"\n{'='*50}")
    print("Training Configuration")
    print(f"{'='*50}")

    # Determine precision
    fp16 = False
    bf16 = False
    if dtype == "auto":
        fp16 = not torch.cuda.is_bf16_supported()
        bf16 = torch.cuda.is_bf16_supported()
    elif dtype == "float16":
        fp16 = True
    elif dtype == "bfloat16":
        bf16 = True

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps if warmup_ratio == 0.0 else 0,
        warmup_ratio=warmup_ratio if warmup_ratio > 0.0 else 0.0,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=save_total_limit,
        report_to="none",
    )

    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    print(f"Batch size: {per_device_train_batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Precision: {'FP16' if fp16 else 'BF16' if bf16 else 'FP32'}")
    print(f"Optimizer: {optim}")

    return training_args


def train_model(
    model,
    tokenizer,
    dataset: Dataset,
    training_args: TrainingArguments,
    max_seq_length: int = 2048,
    packing: bool = False
):
    """Train the model"""
    print(f"\n{'='*50}")
    print("Starting Training")
    print(f"{'='*50}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=packing,
        args=training_args,
    )

    trainer_stats = trainer.train()

    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")

    return trainer_stats


def test_model(
    model,
    tokenizer,
    test_prompt: str = None,
    dataset: Dataset = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9
):
    """Test the trained model"""
    print(f"\n{'='*50}")
    print("Testing Model")
    print(f"{'='*50}")

    FastLanguageModel.for_inference(model)

    # Get test prompt
    if test_prompt:
        test_text = test_prompt
    elif dataset and len(dataset) > 0:
        test_text = dataset[0]["text"]
        # Remove the assistant's response for testing
        if "<|im_start|>assistant" in test_text:
            test_text = test_text.split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
        elif "<|start_header_id|>assistant<|end_header_id|>" in test_text:
            test_text = test_text.split("<|start_header_id|>assistant<|end_header_id|>")[0] + "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        print("No test prompt provided and no dataset available")
        return

    print("Input prompt:")
    print("-" * 50)
    print(test_text[:500] + ("..." if len(test_text) > 500 else ""))
    print("-" * 50)

    # Generate
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )

    response = tokenizer.batch_decode(outputs)[0]

    print("\nGenerated Response:")
    print("=" * 50)
    print(response)
    print("=" * 50)
