"""
Fine-tunes a text generation model on a CPU-only environment using LoRA (PEFT).
This script is designed for GitHub Actions CPU runners.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 1. Configuration for CPU-Only Fine-tuning
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
DATASET_NAME = os.getenv("DATASET_NAME", "Abirate/english_quotes")
OUTPUT_DIR = "output_model"

# LoRA Configuration: Low-rank adaptation to minimize memory usage
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,               # Low-rank matrix
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"]  # Standard target for GPT-like models
)

def main():
    print(f"Starting CPU-only fine-tuning of {MODEL_NAME}...")

    # 2. Load Tokenizer and Dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET_NAME, split="train[:10%]")  # Sub-sample to avoid CPU OOM

    def tokenize_function(examples):
        # We will map the 'quote' field to text
        return tokenizer(examples["quote"], padding="max_length", truncation=True, max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["author", "tags"])

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Standard for CPU (avoid half-precision overhead)
    )
    model.enable_input_require_grads()

    # Apply LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # 4. Training Arguments (Optimized for CPU)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=1,  # Keep epochs low for CPU speed
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        use_cpu=True,          # Force CPU usage
        report_to="none",      # Disable wandb/tensorboard for simplicity in CI
        push_to_hub=False,
    )

    # 5. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # 6. Save the adapter (Only LoRA weights, which are extremely small)
    model.save_pretrained(os.path.join(OUTPUT_DIR, "adapter"))

    print("Training complete! Adapter saved to 'output_model/adapter'.")


if __name__ == "__main__":
    main()
