#!/usr/bin/env python3
"""
Training Script for LLM Fine-tuning with Unsloth
Supports both LoRA and Full Fine-tuning with OpenAI message format
"""

import argparse
import sys

from .dataset import load_data, load_optional_data, prepare_dataset
from .model import load_model_and_tokenizer, save_model
from .trainer import create_training_args, train_model, test_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs using Unsloth with OpenAI message format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Qwen/Qwen3-0.6B)"
    )
    model_group.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    model_group.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization"
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Data type for model weights"
    )

    # Training method
    method_group = parser.add_argument_group("Training Method")
    method_group.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    method_group.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    method_group.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    method_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )
    method_group.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA"
    )

    # Dataset arguments
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to training dataset file (JSON/JSONL) or HuggingFace dataset name"
    )
    data_group.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="Path to validation dataset file (optional)"
    )
    data_group.add_argument(
        "--test_dataset_path",
        type=str,
        default=None,
        help="Path to test dataset file (optional)"
    )
    data_group.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to use (for HuggingFace datasets)"
    )
    data_group.add_argument(
        "--dataset_text_field",
        type=str,
        default="messages",
        help="Field name containing the messages"
    )
    data_group.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt to use"
    )
    data_group.add_argument(
        "--chat_template",
        type=str,
        choices=["qwen", "llama", "auto"],
        default="auto",
        help="Chat template format to use"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints"
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum training steps (overrides num_train_epochs if set)"
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per device"
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    train_group.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps"
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup ratio (overrides warmup_steps if > 0)"
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler"
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    train_group.add_argument(
        "--optim",
        type=str,
        default="adamw_8bit",
        help="Optimizer"
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every N steps"
    )
    train_group.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    train_group.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Save strategy"
    )
    train_group.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed"
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    train_group.add_argument(
        "--packing",
        action="store_true",
        help="Enable sequence packing for efficiency"
    )

    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--save_model",
        action="store_true",
        help="Save the final model"
    )
    output_group.add_argument(
        "--save_model_path",
        type=str,
        default="./final_model",
        help="Path to save the final model"
    )
    output_group.add_argument(
        "--save_merged_16bit",
        action="store_true",
        help="Save merged model in 16-bit format"
    )
    output_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub"
    )
    output_group.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID"
    )
    output_group.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="HuggingFace Hub token"
    )

    # Testing
    test_group = parser.add_argument_group("Testing Configuration")
    test_group.add_argument(
        "--test_after_training",
        action="store_true",
        help="Test model after training"
    )
    test_group.add_argument(
        "--test_prompt",
        type=str,
        default=None,
        help="Test prompt"
    )
    test_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens for generation"
    )
    test_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    test_group.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )

    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()

    print("\n" + "=" * 50)
    print("LLM Fine-tuning with Unsloth")
    print("=" * 50)

    try:
        # 1. Load datasets
        train_dataset = load_data(args.dataset_path, args.dataset_split, dataset_name="train")

        # Load optional validation dataset
        val_dataset = None
        if args.val_dataset_path:
            val_dataset = load_optional_data(args.val_dataset_path, dataset_name="validation")

        # Load optional test dataset
        test_dataset = None
        if args.test_dataset_path:
            test_dataset = load_optional_data(args.test_dataset_path, dataset_name="test")

        # 2. Prepare datasets
        formatted_train_dataset = prepare_dataset(
            train_dataset,
            model_name=args.model_name,
            text_field=args.dataset_text_field,
            system_prompt=args.system_prompt,
            chat_template=args.chat_template
        )

        formatted_val_dataset = None
        if val_dataset is not None:
            formatted_val_dataset = prepare_dataset(
                val_dataset,
                model_name=args.model_name,
                text_field=args.dataset_text_field,
                system_prompt=args.system_prompt,
                chat_template=args.chat_template
            )

        formatted_test_dataset = None
        if test_dataset is not None:
            formatted_test_dataset = prepare_dataset(
                test_dataset,
                model_name=args.model_name,
                text_field=args.dataset_text_field,
                system_prompt=args.system_prompt,
                chat_template=args.chat_template
            )

        # 3. Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed
        )

        # 4. Create training arguments
        training_args = create_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            optim=args.optim,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy=args.save_strategy,
            save_total_limit=args.save_total_limit,
            seed=args.seed,
            dtype=args.dtype
        )

        # 5. Train
        train_model(
            model,
            tokenizer,
            formatted_train_dataset,
            training_args,
            max_seq_length=args.max_seq_length,
            packing=args.packing,
            eval_dataset=formatted_val_dataset
        )

        # 6. Test
        if args.test_after_training:
            # Use test dataset if available, otherwise use train dataset
            test_data = formatted_test_dataset if formatted_test_dataset is not None else formatted_train_dataset
            test_model(
                model,
                tokenizer,
                test_prompt=args.test_prompt,
                dataset=test_data,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )

        # 7. Save
        if args.save_model:
            save_model(
                model,
                tokenizer,
                save_path=args.save_model_path,
                save_merged_16bit=args.save_merged_16bit,
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
                hub_token=args.hub_token
            )

        print("\n" + "=" * 50)
        print("Training Pipeline Complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
