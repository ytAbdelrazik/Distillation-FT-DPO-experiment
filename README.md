# AlignBench

A research-grade ablation study framework that systematically compares **6 LLM training conditions** on a fixed student model — isolating the contributions of LoRA fine-tuning, knowledge distillation, and RLHF (DPO), both individually and in combination.

---

## Research Questions

1. How much does LoRA SFT alone improve instruction-following over a raw base model?
2. Does knowledge distillation from a larger teacher (9B) outperform direct SFT on the same data?
3. Does combining distillation + LoRA (C4) yield compounding gains over either alone?
4. Does DPO alignment further improve models already trained with distillation or SFT?
5. What are the compute-quality trade-offs across all 6 conditions?

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION (one-time, before training)                   │
│                                                                  │
│  mlx-community/Qwen3.5-9B-OptiQ-4bit                           │
│         │                                                        │
│         │ mlx-lm (Apple Silicon only)                           │
│         │ data/generate_teacher_outputs.py                       │
│         ▼                                                        │
│  data/teacher_outputs.jsonl  ◄── soft labels, token_ids        │
│         │                                                        │
│  data/alpaca.jsonl ──────────────────────────────────────┐      │
│  data/hh_rlhf.jsonl ─────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ All training: HuggingFace + peft
                           │ (PyTorch on MPS / CUDA / CPU)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING (train/ scripts, no teacher model in memory)          │
│                                                                  │
│  Qwen/Qwen3-0.6B (student base)                                 │
│  ├── C1: baseline        → checkpoints/baseline (eval only)     │
│  ├── C2: lora_sft        → checkpoints/lora_sft                 │
│  ├── C3: distillation    → checkpoints/distillation             │
│  ├── C4: distill_lora    → checkpoints/distill_lora             │
│  ├── C5: distill_dpo     → checkpoints/distill_dpo              │
│  └── C6: lora_sft_dpo    → checkpoints/lora_sft_dpo            │
│                                                                  │
│  reward_model/train_rm.py → checkpoints/reward_model           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           │ Merge LoRA → base model
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION (eval/run_eval.py)                                  │
│                                                                  │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Perplexity     │  │  HellaSwag       │  │  ROUGE-L      │  │
│  │  (1K Alpaca)    │  │  TruthfulQA      │  │  BERTScore F1 │  │
│  │                 │  │  MMLU            │  │  (CNN/DM)     │  │
│  │                 │  │  (lm-eval)       │  │               │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Win rate vs baseline — teacher-as-judge via mlx-lm      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  compare/dashboard.py                                           │
│  → results/comparison_table.md  + comparison_table.csv         │
│  → results/plots/bar_*.png                                      │
│  → results/plots/radar_chart.png                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 6 Conditions

| ID | Name | Description |
|----|------|-------------|
| C1 | `baseline` | Raw student (`Qwen/Qwen3-0.6B`), no training — eval only |
| C2 | `lora_sft` | LoRA fine-tuning on Alpaca instruction dataset |
| C3 | `distillation` | KD from pre-generated teacher soft labels (KL divergence) |
| C4 | `distill_lora` | Distillation loss + LoRA adapters combined |
| C5 | `distill_dpo` | Distillation (C3) → DPO alignment on HH-RLHF |
| C6 | `lora_sft_dpo` | LoRA SFT (C2) → DPO alignment on HH-RLHF |

---

## Setup

### Apple Silicon (M1/M2/M3)

```bash
# Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install training dependencies
pip install -r requirements-train.txt

# Install eval dependencies
pip install -r requirements-eval.txt

# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
```

### CUDA GPU (A100/T4)

```bash
python3 -m venv .venv && source .venv/bin/activate

# Same requirements — device is auto-detected at runtime
pip install -r requirements-train.txt
pip install -r requirements-eval.txt

# mlx-lm is Apple Silicon only; on CUDA, skip teacher generation
# and use pre-generated teacher_outputs.jsonl from disk
```

> **Note**: `mlx-lm` (for teacher inference) only runs on Apple Silicon.
> On CUDA, generate teacher outputs on a Mac first and copy `data/teacher_outputs.jsonl` to the GPU machine.

---

## Running Individual Conditions

```bash
# Step 1: Prepare data
python data/prepare_alpaca.py
python data/prepare_hh_rlhf.py

# Step 2: Generate teacher outputs (Apple Silicon only, one-time)
python data/generate_teacher_outputs.py

# C2: LoRA SFT
python train/lora_sft.py --config configs/lora_sft.yaml --seed 42

# C3: Knowledge Distillation
python train/distillation.py --config configs/distillation.yaml --seed 42

# C4: Distillation + LoRA
python train/distill_lora.py --config configs/distill_lora.yaml --seed 42

# C5: Distillation → DPO (requires C3 checkpoint first)
python train/dpo.py --condition distill_dpo --config configs/dpo.yaml --seed 42

# C6: LoRA SFT → DPO (requires C2 checkpoint first)
python train/dpo.py --condition lora_sft_dpo --config configs/dpo.yaml --seed 42

# Reward model
python reward_model/train_rm.py --seed 42

# LoRA rank ablation (r ∈ {4, 8, 16, 32, 64})
python train/lora_sft.py --rank-ablation --seed 42

# Eval a single condition
python eval/run_eval.py --condition lora_sft

# Eval all conditions
python eval/run_eval.py

# Dashboard
python compare/dashboard.py
```

---

## Running the Full Pipeline

```bash
# Full run (takes several hours on M1 Pro)
bash run_all.sh

# Fast iteration (10K samples, quick sanity check)
bash run_all.sh --fast

# Dry run (print plan without executing)
bash run_all.sh --dry-run

# With custom seed, no W&B
bash run_all.sh --seed 123 --no-wandb

# Run only specific conditions
bash run_all.sh --condition lora_sft --condition lora_sft_dpo

# Skip teacher generation (use existing teacher_outputs.jsonl)
bash run_all.sh --skip-teacher
```

---

## Results Table

*(Filled after training — run `bash run_all.sh` then `python compare/dashboard.py`)*

| Condition | Perplexity ↓ | HellaSwag ↑ | TruthfulQA ↑ | MMLU ↑ | ROUGE-L ↑ | BERTScore ↑ | Win Rate ↑ | Train Time ↓ |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| C1: Baseline | — | — | — | — | — | — | 50% | 0s |
| C2: LoRA SFT | — | — | — | — | — | — | — | — |
| C3: Distillation | — | — | — | — | — | — | — | — |
| C4: Distill+LoRA | — | — | — | — | — | — | — | — |
| C5: Distill+DPO | — | — | — | — | — | — | — | — |
| C6: LoRA+DPO | — | — | — | — | — | — | — | — |

---

## Metric Explanations

| Metric | Description |
|--------|-------------|
| **Perplexity** | Exponentiated average NLL on 1K held-out Alpaca prompts. Lower = better. Measures fluency and language modeling quality. |
| **HellaSwag Acc** | Commonsense inference accuracy. Tests whether the model picks the most plausible sentence continuation. |
| **TruthfulQA Acc** | Measures truthfulness — whether the model avoids common misconceptions and false answers. |
| **MMLU Acc** | Massive Multitask Language Understanding: 57 subjects of academic knowledge. Broad capability measure. |
| **ROUGE-L** | Longest common subsequence F-measure between generated and reference summaries on CNN/DailyMail. |
| **BERTScore F1** | Semantic similarity between generated and reference summaries using BERT embeddings. |
| **Win Rate** | Fraction of pairwise comparisons (condition vs C1 baseline) where the teacher model judges the condition's response as better. |
| **Peak Memory** | Peak allocated GPU/MPS memory during training in MB. |
| **Train Time** | Wall-clock seconds for the full training run. |

---

## Key Design Decisions

- **Teacher inference via `mlx-lm`**: The 9B teacher fits in 16GB at 4-bit using Apple's MLX framework, which is not compatible with PyTorch. Teacher outputs are pre-generated and saved to disk before any training begins.
- **Student training via `peft` + PyTorch MPS**: LoRA and KD training use pure HuggingFace/PyTorch — no quantization needed for the 0.6B student.
- **The two frameworks never coexist in memory**: `mlx-lm` and `transformers` model instances are never loaded simultaneously.
- **`batch_size=4, grad_accum=8`**: Effective batch of 32 within 16GB unified memory.
- **Reproducibility**: Every condition accepts `--seed` and logs hyperparameters to W&B.

---

## References

- **LoRA**: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Knowledge Distillation**: Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **InstructGPT / RLHF**: Ouyang et al. (2022). *Training language models to follow instructions with human feedback*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **DPO**: Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
