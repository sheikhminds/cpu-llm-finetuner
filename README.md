# CPU-LLM-Finetuner

A proof-of-concept to fine-tune text generation models using **CPU only** and **GitHub Actions** (No GPU required).

## Overview
Fine-tuning LLMs is usually expensive. This repository demonstrates how to fine-tune a small language model (such as `distilgpt2`) on GitHub Actions CPU runners using Parameter-Efficient Fine-Tuning (LoRA).

## How It Works
1. **Model Selection:** `distilgpt2` (82M parameters) — ideal for CPU tasks.
2. **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) — minimizes memory and compute cost.
3. **Orchestration:** GitHub Actions CI/CD pipeline runs the training script on standard CPU runners.
4. **Output:** Produces a lightweight LoRA adapter that can be merged with the base model for inference.

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally (CPU):**
   ```bash
   python train.py
   ```

3. **Run via GitHub Actions:**
   Push to `main` or trigger the workflow manually via the Actions tab.

## Contributing
PRs are welcome. Please ensure CI passes.

## License
MIT License
