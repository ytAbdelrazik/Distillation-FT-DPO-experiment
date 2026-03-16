# AlignBench — Claude Code Prompt

---

## Project Overview

Build **AlignBench**: a research-grade ablation study framework that systematically compares 6 different LLM training conditions on a fixed student model. The goal is to isolate the contribution of LoRA fine-tuning, knowledge distillation, and RLHF (DPO) — both individually and combined — and evaluate every condition on a shared multi-metric benchmark.

---

## Hardware Target

- **Training device**: Apple M1 Pro, 16GB unified memory (MPS backend)
- **Eval device**: same machine (MPS / CPU)
- **No CUDA required**: all training runs locally via PyTorch MPS
- **Cloud option**: code must also run on a CUDA GPU (A100/T4) unchanged — device is auto-detected at runtime

---

## Models

- **Teacher**: `mlx-community/Qwen3.5-9B-OptiQ-4bit` — loaded via `mlx-lm` for inference only, outputs pre-generated and saved to disk before any training begins. Never loaded during training.
- **Student**: `Qwen/Qwen3-0.6B` — loaded via HuggingFace `transformers`, trained with `peft` LoRA on MPS
- **Reward Model base**: same student architecture (`Qwen/Qwen3-0.6B`), separate linear head

> **Why this split**: The teacher is used exclusively to pre-generate soft labels and win-rate judgments via `mlx-lm` (Apple-native, fits in 16GB at 4-bit). All training uses HuggingFace + `peft` on MPS — `peft` and LoRA are pure PyTorch ops and work natively on MPS. The two frameworks never coexist in memory simultaneously.

---

## Dataset

- **Instruction data (SFT)**: `yahma/alpaca-cleaned` (52K samples)
- **Distillation data**: generate teacher completions on the same Alpaca prompts using `mlx-lm`, store as `data/teacher_outputs.jsonl`. This step runs once upfront; training scripts load from disk.
- **Preference data (RLHF)**: `Anthropic/hh-rlhf` (chosen / rejected pairs)
- All datasets must be preprocessed into a unified format and cached under `data/`

---

## The 6 Experimental Conditions

Implement each as a separate, reproducible training script under `train/`. Every condition saves its adapter/checkpoint to `checkpoints/<condition_name>/`.

| ID | Condition | Description |
|----|-----------|-------------|
| C1 | `baseline` | Raw student, no training. Eval only. |
| C2 | `lora_sft` | LoRA fine-tuning on Alpaca instruction dataset |
| C3 | `distillation` | Knowledge distillation from pre-generated teacher soft labels (KL divergence loss) |
| C4 | `distill_lora` | Distillation loss + LoRA adapters combined |
| C5 | `distill_dpo` | Distillation (C3) → then DPO alignment on HH-RLHF |
| C6 | `lora_sft_dpo` | LoRA SFT (C2) → then DPO alignment on HH-RLHF (control) |

---

## Project Structure to Build

```
alignbench/
├── data/
│   ├── prepare_alpaca.py              # Download + format Alpaca
│   ├── generate_teacher_outputs.py    # Run teacher via mlx-lm, save soft labels to disk
│   └── prepare_hh_rlhf.py            # Download + format preference data
│
├── train/
│   ├── base_trainer.py                # Shared trainer config, device detection, logging, saving
│   ├── lora_sft.py                    # C2: LoRA supervised fine-tuning (peft on MPS)
│   ├── distillation.py                # C3: KD loading soft labels from disk (KL loss, pure PyTorch)
│   ├── distill_lora.py                # C4: Combined KD + LoRA
│   └── dpo.py                         # C5 & C6: DPO via trl DPOTrainer on MPS
│
├── reward_model/
│   └── train_rm.py                    # Bradley-Terry reward model, pure PyTorch, runs on MPS
│
├── eval/
│   ├── run_eval.py                    # Master eval runner for all conditions
│   ├── perplexity.py                  # Perplexity on held-out set
│   ├── benchmarks.py                  # HellaSwag, TruthfulQA, MMLU via lm-evaluation-harness
│   ├── win_rate.py                    # Pairwise win rate judged by teacher via mlx-lm
│   └── compute_metrics.py             # ROUGE, BERTScore on summarization
│
├── compare/
│   └── dashboard.py                   # Load all results, produce comparison plots + table
│
├── configs/
│   ├── lora_sft.yaml
│   ├── distillation.yaml
│   ├── distill_lora.yaml
│   └── dpo.yaml
│
├── requirements-train.txt             # MPS training deps (no bitsandbytes)
├── requirements-eval.txt              # HuggingFace eval deps
├── run_all.sh                         # Single script to run full pipeline end to end
└── README.md
```

---

## Implementation Requirements

### Device Detection (all training scripts)

Every training script must auto-detect the available device at startup:

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

No hardcoded `"cuda"` anywhere. All `.to(device)` calls use this function.

### LoRA (C2, C4, C6)
- Use HuggingFace `peft` library — works natively on MPS, no changes needed
- Target modules: `q_proj`, `v_proj`
- Default rank: `r=16`, `alpha=32`, `dropout=0.05`
- Make rank configurable via YAML config
- Include a rank ablation helper: loop r ∈ {4, 8, 16, 32, 64} and log results

### Knowledge Distillation (C3, C4, C5)
- Teacher outputs are **pre-generated and loaded from `data/teacher_outputs.jsonl`** — the teacher model is never loaded during training
- Each record in `teacher_outputs.jsonl` contains: `{"prompt": "...", "soft_labels": [...], "token_ids": [...]}`
- Temperature-scaled soft labels (T=4 default, configurable)
- Loss = `alpha * KL(student || teacher) + (1 - alpha) * CrossEntropy(student, hard_labels)`
- Alpha configurable (default 0.7)
- Pure PyTorch loss — no `bitsandbytes`, no quantization needed during training

### Teacher Output Generation (`data/generate_teacher_outputs.py`)
- Load `mlx-community/Qwen3.5-9B-OptiQ-4bit` via `mlx-lm`
- Run on all Alpaca prompts, extract logits/soft labels, save to `data/teacher_outputs.jsonl`
- Also used by `eval/win_rate.py` to judge pairwise responses
- This script is the **only place `mlx-lm` is used** — all other scripts are pure HuggingFace/PyTorch

### DPO (C5, C6)
- Use HuggingFace `trl` library's `DPOTrainer` — MPS-compatible in `trl>=0.8.0`
- Load from the corresponding checkpoint (C3 for C5, C2 for C6)
- Apply LoRA adapters during DPO as well (do not full fine-tune)
- Log reward margin (chosen reward - rejected reward) per step to W&B

### Reward Model
- Linear head on top of student's last hidden state
- Train with Bradley-Terry ranking loss on HH-RLHF chosen/rejected pairs — pure PyTorch, runs on MPS
- Save to `checkpoints/reward_model/`
- Report Kendall's tau on a held-out eval set

---

## Evaluation Suite

Run `eval/run_eval.py --condition <name>` to evaluate any single condition. Run without `--condition` to evaluate all 6.

Eval loads checkpoints in HuggingFace format. After training, LoRA adapters are merged into the base model before eval:

```bash
# Merge adapter into base model for eval
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
model = PeftModel.from_pretrained(base, 'checkpoints/lora_sft')
model.merge_and_unload().save_pretrained('checkpoints/lora_sft_merged')
"
```

This merge step is automated inside `eval/run_eval.py` before running any metrics.

### Metrics to compute and log for every condition:

| Metric | Tool |
|--------|------|
| Perplexity | Custom, on 1K held-out Alpaca prompts |
| MT-Bench score | `lm-evaluation-harness` or teacher-as-judge via `mlx-lm` |
| HellaSwag accuracy | `lm-evaluation-harness` |
| TruthfulQA accuracy | `lm-evaluation-harness` |
| MMLU accuracy | `lm-evaluation-harness` |
| ROUGE-L | On CNN/DailyMail summarization subset |
| BERTScore F1 | On CNN/DailyMail summarization subset |
| Win rate vs baseline | Pairwise, judged by teacher via `mlx-lm` |
| Peak memory (MB) | `torch.mps.current_allocated_memory()` on MPS, `torch.cuda.max_memory_allocated()` on CUDA |
| Training time (sec) | Wall clock per condition |

Store all results in `results/<condition_name>.json`.

---

## Comparison Dashboard

`compare/dashboard.py` should:
- Load all `results/*.json` files
- Produce a **single comparison table** (markdown + CSV) with all conditions × all metrics
- Produce **bar charts** comparing each metric across conditions
- Produce a **radar chart** showing overall profile of each condition
- Highlight the best condition per metric in green
- Save all plots to `results/plots/`
- Print a summary: which condition wins overall, which is most compute-efficient

---

## Configuration

Each training script reads from a YAML config. Example `configs/lora_sft.yaml`:

```yaml
model:
  student: Qwen/Qwen3-0.6B
  teacher_outputs: data/teacher_outputs.jsonl   # pre-generated, no live teacher loading
  # Teacher model loaded separately via mlx-lm only in data/generate_teacher_outputs.py

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, v_proj]

training:
  epochs: 3
  batch_size: 4                        # reduced from 8 for 16GB unified memory
  gradient_accumulation_steps: 8       # effective batch = 32, same as original
  learning_rate: 2e-4
  warmup_ratio: 0.03
  lr_scheduler: cosine
  max_seq_length: 512
  seed: 42

distillation:
  temperature: 4.0
  alpha: 0.7

logging:
  wandb_project: alignbench
  save_steps: 500
  eval_steps: 500
```

---

## `run_all.sh`

Write a single shell script that:
1. Prepares all datasets (HuggingFace `datasets`, works on Mac)
2. Generates teacher outputs via `mlx-lm` (one-time, saved to disk)
3. Trains all 6 conditions in order using HuggingFace + `peft` on MPS (C1 skipped, eval only)
4. Trains the reward model
5. Merges LoRA adapters into base model for each condition
6. Runs full eval on all conditions
7. Generates the comparison dashboard

Add `--dry-run` flag that prints the plan without executing.
Add `--fast` flag that reduces dataset to 10K samples for quick iteration.

---

## Requirements

Split into two files:

### `requirements-train.txt`
```
# Training — runs on Apple MPS or CUDA, no bitsandbytes needed
torch>=2.1.0
transformers>=4.51.0          # minimum for Qwen3 support
peft>=0.10.0
trl>=0.8.0
datasets>=2.18.0
accelerate>=0.28.0
pyyaml>=6.0
wandb>=0.16.0
tqdm>=4.66.0
mlx-lm>=0.18.0                # teacher inference only (Apple Silicon)
```

### `requirements-eval.txt`
```
# Eval — HuggingFace ecosystem, runs on MPS/CPU
torch>=2.1.0
transformers>=4.51.0
datasets>=2.18.0
evaluate>=0.4.0
bert-score>=0.3.13
rouge-score>=0.1.2
lm-eval>=0.4.0
wandb>=0.16.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm>=4.66.0
```

> **Note**: `bitsandbytes` is removed entirely — it is CUDA-only and not needed since teacher is loaded via `mlx-lm` and student training is pure PyTorch on MPS.

---

## README Requirements

The README must include:
- Project overview and research questions
- Architecture diagram (ASCII) — including the MLX teacher → disk → PyTorch training flow
- Setup instructions for Apple Silicon (M1/M2/M3) and CUDA GPUs
- How to run each condition individually
- How to run the full pipeline
- Results table (placeholder, to be filled after training)
- Explanation of each metric
- References to key papers: LoRA, DistilBERT, InstructGPT, DPO

---

## Key Constraints

- **Primary target**: Apple M1 Pro, 16GB unified memory — all training must complete locally
- **Secondary target**: single CUDA GPU (A100/T4) — same codebase, device auto-detected
- **batch_size=4** with **gradient_accumulation_steps=8** to match effective batch of 32 within 16GB
- Every condition must be **fully reproducible** via a single command with `--seed` argument
- Add `--fast` flag that reduces dataset size to 10K samples (useful for Mac development iteration)
- Code must be **modular** — each training script importable as a Python module, not just a CLI script
- Use `wandb` for all experiment tracking; log hyperparameters, losses, and eval metrics per run
- Teacher model (`mlx-community/Qwen3.5-9B-OptiQ-4bit`) is **never loaded during PyTorch training** — only pre-generation and eval judging via `mlx-lm`
- `transformers>=4.51.0` is required — earlier versions do not support Qwen3