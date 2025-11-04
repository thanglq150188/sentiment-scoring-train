# ============================================
# STEP 2: Import Libraries
# ============================================
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================
# STEP 3: Load Model and Tokenizer
# ============================================
max_seq_length = 2048  # Can adjust based on your needs
dtype = None  # Auto-detect. Use Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",  # Qwen 3 equivalent
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================
# STEP 4: Configure LoRA Adapters
# ============================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # Long context support
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ============================================
# STEP 5: Prepare QA Dataset
# ============================================
# Example: Using a sample QA dataset
# You can replace this with your own dataset

# Sample QA data format
sample_data = {
    "question": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
    ],
    "answer": [
        "The capital of France is Paris.",
        "William Shakespeare wrote Romeo and Juliet.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
    ]
}

from datasets import Dataset
dataset = Dataset.from_dict(sample_data)

# Or load from HuggingFace:
# dataset = load_dataset("your-dataset-name", split="train")

# ============================================
# STEP 6: Format Dataset for Training
# ============================================
def formatting_prompts_func(examples):
    """Convert QA pairs to chat format"""
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    
    for question, answer in zip(questions, answers):
        # Qwen chat template format
        text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""
        texts.append(text)
    
    return {"text": texts}

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================
# STEP 7: Configure Training Arguments
# ============================================
training_args = TrainingArguments(
    per_device_train_batch_size=10,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,  # Adjust based on your dataset size
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

# ============================================
# STEP 8: Initialize Trainer
# ============================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences
    args=training_args,
)

# ============================================
# STEP 9: Train the Model
# ============================================
print("Starting training...")
trainer_stats = trainer.train()
print("Training complete!")

# ============================================
# STEP 10: Test the Finetuned Model
# ============================================
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Test with a sample question
inputs = tokenizer(
    """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
""", 
    return_tensors="pt"
).to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    use_cache=True,
    temperature=0.7,
    top_p=0.9
)

print("\n" + "="*50)
print("MODEL OUTPUT:")
print("="*50)
print(tokenizer.batch_decode(outputs)[0])

# ============================================
# STEP 11: Save the Model
# ============================================
# Save to local directory
model.save_pretrained("qwen3_qa_finetuned")
tokenizer.save_pretrained("qwen3_qa_finetuned")

# Save to 16bit for loading later
model.save_pretrained_merged("qwen3_qa_16bit", tokenizer, save_method="merged_16bit")

# Push to HuggingFace Hub (optional)
# model.push_to_hub_merged("your-username/qwen3-qa-finetuned", tokenizer, save_method="merged_16bit", token="YOUR_HF_TOKEN")

print("\n✅ Model saved successfully!")