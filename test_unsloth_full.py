# ============================================
# STEP 2: Import Libraries
# ============================================
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================
# STEP 3: Load Model and Tokenizer (Full Precision)
# ============================================
max_seq_length = 2048
dtype = None  # Auto-detect. Use Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Set to False for full finetuning

print("Loading Qwen 3 0.6B model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================
# STEP 4: Prepare for Full Finetuning
# ============================================
# For full finetuning, we don't use LoRA/PEFT
# Just prepare the model for training
model = FastLanguageModel.for_training(model)
model.gradient_checkpointing_enable()

print(f"Model loaded: {model.config.model_type}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================
# STEP 5: Prepare QA Dataset
# ============================================
# Sample QA data format
sample_data = {
    "question": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "What is the smallest unit of life?",
        "Who invented the telephone?",
    ],
    "answer": [
        "The capital of France is Paris.",
        "William Shakespeare wrote Romeo and Juliet.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "Jupiter is the largest planet in our solar system.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "The speed of light is approximately 299,792,458 meters per second.",
        "The cell is the smallest unit of life.",
        "Alexander Graham Bell invented the telephone.",
    ]
}

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

print(f"\nDataset size: {len(dataset)} examples")
print("\nSample formatted text:")
print(dataset[0]["text"])

# ============================================
# STEP 7: Configure Training Arguments for Full Finetuning
# ============================================
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Smaller batch for full finetuning
    gradient_accumulation_steps=8,  # Effective batch size = 1 * 8 = 8
    warmup_steps=10,
    num_train_epochs=3,  # Use epochs instead of max_steps
    learning_rate=2e-5,  # Lower learning rate for full finetuning
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_torch",  # Standard AdamW for full finetuning
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",  # Disable wandb/tensorboard
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
    packing=False,
    args=training_args,
)

# ============================================
# STEP 9: Train the Model (Full Finetuning)
# ============================================
print("\n" + "="*50)
print("Starting FULL finetuning...")
print("="*50)
print("⚠️  This will update ALL model weights")
print("⚠️  Requires more GPU memory than LoRA")
print("="*50 + "\n")

trainer_stats = trainer.train()

print("\n" + "="*50)
print("✅ Training complete!")
print("="*50)

# ============================================
# STEP 10: Test the Finetuned Model
# ============================================
print("\nPreparing model for inference...")
FastLanguageModel.for_inference(model)

# Test with a sample question
test_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
"""

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

print("\n" + "="*50)
print("TESTING FINETUNED MODEL")
print("="*50)
print(f"Input: What is the capital of France?\n")

outputs = model.generate(
    **inputs, 
    max_new_tokens=128, 
    use_cache=True,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.batch_decode(outputs)[0]
print("Full Response:")
print(response)

# Extract just the answer
answer_start = response.find("<|im_start|>assistant") + len("<|im_start|>assistant")
answer = response[answer_start:].strip()
print(f"\nExtracted Answer: {answer}")

# ============================================
# STEP 11: Save the Fully Finetuned Model
# ============================================
print("\n" + "="*50)
print("Saving model...")
print("="*50)

# Save the full model
output_dir = "qwen3_qa_full_finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to: {output_dir}")

# Save in 16-bit format for deployment
output_dir_16bit = "qwen3_qa_full_finetuned_16bit"
model.save_pretrained_merged(output_dir_16bit, tokenizer, save_method="merged_16bit")
print(f"✅ 16-bit model saved to: {output_dir_16bit}")

# Optional: Push to HuggingFace Hub
# model.push_to_hub_merged(
#     "your-username/qwen3-0.6b-qa-finetuned", 
#     tokenizer, 
#     save_method="merged_16bit",
#     token="YOUR_HF_TOKEN"
# )

print("\n" + "="*50)
print("🎉 Full finetuning complete!")
print("="*50)
print(f"Total parameters trained: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")