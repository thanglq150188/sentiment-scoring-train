"""Model loading and configuration utilities"""

import torch
from unsloth import FastLanguageModel


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: str = "auto",
    load_in_4bit: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: list = None,
    gradient_checkpointing: bool = False,
    seed: int = 3407
):
    """Load model and tokenizer with optional LoRA"""
    print(f"\n{'='*50}")
    print("Loading Model and Tokenizer")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Method: {'LoRA' if use_lora else 'Full Fine-tuning'}")

    # Handle dtype
    dtype_map = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype_value = dtype_map[dtype]

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype_value,
        load_in_4bit=load_in_4bit,
    )

    # Configure for training
    if use_lora:
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]

        print(f"\nConfiguring LoRA:")
        print(f"  - Rank: {lora_r}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - Target modules: {lora_target_modules}")

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
            random_state=seed,
            use_rslora=False,
            loftq_config=None,
        )
    else:
        print("\nConfiguring for full fine-tuning...")
        model = FastLanguageModel.for_training(model)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel loaded successfully")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Trainable %: {100 * trainable_params / total_params:.2f}%")

    return model, tokenizer


def save_model(
    model,
    tokenizer,
    save_path: str,
    save_merged_16bit: bool = False,
    push_to_hub: bool = False,
    hub_model_id: str = None,
    hub_token: str = None
):
    """Save the trained model"""
    print(f"\n{'='*50}")
    print("Saving Model")
    print(f"{'='*50}")

    # Save base model
    print(f"Saving to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    # Save merged 16-bit version
    if save_merged_16bit:
        merged_path = f"{save_path}_16bit"
        print(f"Saving merged 16-bit model to: {merged_path}")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
        print(f"Merged 16-bit model saved to {merged_path}")

    # Push to hub
    if push_to_hub:
        if not hub_model_id:
            print("--hub_model_id not provided, skipping push to hub")
        else:
            print(f"Pushing to HuggingFace Hub: {hub_model_id}")
            model.push_to_hub_merged(
                hub_model_id,
                tokenizer,
                save_method="merged_16bit",
                token=hub_token
            )
            print(f"Model pushed to {hub_model_id}")
