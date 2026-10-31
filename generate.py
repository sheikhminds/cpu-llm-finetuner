"""
Demo: load the base model with the fine-tuned LoRA adapter and generate text.
Run after the GitHub Actions training job has produced the adapter artifact.

Usage:
  python generate.py --adapter output_model/adapter --prompt "The best way to learn"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Generate text with the fine-tuned adapter")
    parser.add_argument("--adapter", default="output_model/adapter", help="Path to the LoRA adapter directory")
    parser.add_argument("--model", default="distilgpt2", help="Base model name")
    parser.add_argument("--prompt", default="The best way to learn", help="Text prompt to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    print(f"Loading base model '{args.model}' on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    print(f"Attaching LoRA adapter from '{args.adapter}'...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n--- Generated text ---")
    print(text)

if __name__ == "__main__":
    main()
