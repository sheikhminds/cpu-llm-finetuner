# PLAN.md — CPU-Only LLM Fine-Tuner on GitHub Actions

## NEXT_ACTION:
Write and validate a Dockerfile and a Hugging Face `transformers` training script optimized for CPU-only execution (no GPU), then create the `.github/workflows/train.yml` to trigger the pipeline.

---

## 1. Problem Statement
Fine-tuning text generation models usually requires GPU/TPU resources. This project aims to democratize the fine-tuning of smaller parameter models (e.g., 7B or smaller, or distilled models) using **CPU-only resources** available in **GitHub Actions** runners.

## 2. Target User
Developers, small tech teams, and AI researchers who want to experiment with fine-tuning LLMs but have limited access to paid GPU infrastructure (e.g., zero-budget hobbyists, local dev environments).

## 3. MVP Definition
- A clear `PLAN.md` outlining the strategy.
- A functional Hugging Face `transformers` + `accelerate` training loop adapted for CPU execution.
- A `.github/workflows/train.yml` GitHub Action pipeline that runs the training process in a CPU-based container.
- A demonstration of the fine-tuning process using a small/medium open-source model.

## 4. Technical Strategy (No GPU Constraint)
1. **Model Selection:** Use a small, CPU-friendly model like `TinyLlama-1.1B-Chat` or `Phi-3-mini-4k-instruct` (or similar quantized variants).
2. **Quantization/Offloading:** Utilize frameworks supporting CPU quantization (e.g., ONNX Runtime, OpenVINO, or Hugging Face `transformers` with `torch_dtype=torch.float32`).
3. **Optimization:** Use `PEFT`/`LoRA` for parameter-efficient fine-tuning (significantly less memory and computation required than full-model training).
4. **GitHub Actions Runners:** Target CPU runners (e.g., `ubuntu-latest` with standard specs) and optimize for RAM usage to prevent OOM (Out Of Memory) errors.

## 5. Deliverables & Monetization
- **Baseline:** Open-source repository demonstrating the feasibility.
- **Pro (Future):** A GitHub Action setup script or a paid tier of a customized "one-click" CPU fine-tuner for specific enterprise-like niche tasks (e.g., compliance data pre-processing).
- **Sponsorship:** Add GitHub Sponsors link upon stable proof of concept.

## 6. Success Criteria
- A repository setup that successfully triggers a GitHub Action that runs a model training step without GPU hardware.
- A working, documented `README.md` with installation and usage instructions.
- 1 passing test suite verifying the training loop runs without OOM on a standard runner.

## 7. Milestones
- **Phase 1:** Repo setup, `.gitignore` configuration, baseline environment.
- **Phase 2:** Training script and CI/CD pipeline.
- **Phase 3:** Public release (v1.0.0), demo video/GIF, community outreach (Reddit/HN).
