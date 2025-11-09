"""Dataset loading and formatting utilities"""

import json
import os
from typing import Dict, List

from datasets import Dataset, load_dataset


def load_data(dataset_path: str, dataset_split: str = "train", dataset_name: str = "train") -> Dataset:
    """Load dataset from file or HuggingFace

    Args:
        dataset_path: Path to dataset file or HuggingFace dataset name
        dataset_split: Split to use if loading from HuggingFace
        dataset_name: Name to display (e.g., "train", "validation", "test")

    Returns:
        Dataset object or None if file doesn't exist
    """
    print(f"\n{'='*50}")
    print(f"Loading {dataset_name.capitalize()} Dataset")
    print(f"{'='*50}")

    if os.path.exists(dataset_path):
        print(f"Loading from local file: {dataset_path}")

        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif dataset_path.endswith(".json"):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and dataset_split in data:
                    data = data[dataset_split]
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")

        dataset = Dataset.from_list(data)
    else:
        print(f"Loading from HuggingFace: {dataset_path}")
        dataset = load_dataset(dataset_path, split=dataset_split)

    print(f"{dataset_name.capitalize()} dataset loaded: {len(dataset)} examples")
    print(f"\nSample data (first example):")
    print(json.dumps(dataset[0], indent=2))

    return dataset


def load_optional_data(dataset_path: str, dataset_name: str = "validation") -> Dataset | None:
    """Load optional dataset (validation or test)

    Args:
        dataset_path: Path to dataset file
        dataset_name: Name to display (e.g., "validation", "test")

    Returns:
        Dataset object or None if file doesn't exist
    """
    if not dataset_path or not os.path.exists(dataset_path):
        print(f"\n{dataset_name.capitalize()} dataset not found at: {dataset_path or 'Not provided'}")
        print(f"Skipping {dataset_name} dataset")
        return None

    return load_data(dataset_path, dataset_name=dataset_name)


def get_chat_template(model_name: str, template_type: str = "auto") -> Dict[str, str]:
    """Get chat template based on model type"""
    if template_type == "auto":
        model_lower = model_name.lower()
        if "qwen" in model_lower:
            template_type = "qwen"
        elif "llama" in model_lower:
            template_type = "llama"
        else:
            template_type = "qwen"

    templates = {
        "qwen": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
            "assistant_start": "<|im_start|>assistant\n"
        },
        "llama": {
            "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
            "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
    }

    return templates.get(template_type, templates["qwen"])


def format_messages_to_text(messages: List[Dict], template: Dict, system_prompt: str = None) -> str:
    """Convert OpenAI message format to text using chat template"""
    text_parts = []
    has_system = False

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            has_system = True
            text_parts.append(template["system"].format(content=content))
        elif role == "user":
            text_parts.append(template["user"].format(content=content))
        elif role == "assistant":
            text_parts.append(template["assistant"].format(content=content))

    if not has_system and system_prompt:
        text_parts.insert(0, template["system"].format(content=system_prompt))

    return "".join(text_parts)


def prepare_dataset(
    dataset: Dataset,
    model_name: str,
    text_field: str = "messages",
    system_prompt: str = "You are a helpful assistant.",
    chat_template: str = "auto"
) -> Dataset:
    """Format dataset for training"""
    print(f"\n{'='*50}")
    print("Preparing Dataset")
    print(f"{'='*50}")

    template = get_chat_template(model_name, chat_template)

    def formatting_func(examples):
        """Convert messages to formatted text"""
        texts = []
        messages_list = examples[text_field]

        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]

        for messages in messages_list:
            text = format_messages_to_text(messages, template, system_prompt)
            texts.append(text)

        return {"text": texts}

    formatted_dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    print(f"Dataset formatted: {len(formatted_dataset)} examples")
    print(f"\nSample formatted text:")
    print("-" * 50)
    print(formatted_dataset[0]["text"][:500] + "...")
    print("-" * 50)

    return formatted_dataset
