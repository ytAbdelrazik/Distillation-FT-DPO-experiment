# AlignBench — Step-by-Step Instructions

Run these commands in order, one at a time.

---

## 1. Setup

```bash
cd /Users/youssefabdelrazik/alignbench
```

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements-train.txt
```

```bash
pip install -r requirements-eval.txt
```

---

## 2. Prepare Data

```bash
python data/prepare_alpaca.py --fast
```

```bash
python data/prepare_hh_rlhf.py --fast
```

---

## 3. Generate Teacher Outputs
*(Apple Silicon only — takes ~15 min in fast mode)*

```bash
python data/generate_teacher_outputs.py --fast
```

---

## 4. Train Each Condition

### C2 — LoRA SFT

```bash
python train/lora_sft.py --config configs/lora_sft.yaml --seed 42 --fast --no-wandb
```

### C3 — Knowledge Distillation

```bash
python train/distillation.py --config configs/distillation.yaml --seed 42 --fast --no-wandb
```

### C4 — Distillation + LoRA

```bash
python train/distill_lora.py --config configs/distill_lora.yaml --seed 42 --fast --no-wandb
```

### C5 — Distillation → DPO
*(requires C3 to finish first)*

```bash
python train/dpo.py --condition distill_dpo --config configs/dpo.yaml --seed 42 --fast --no-wandb
```

### C6 — LoRA SFT → DPO
*(requires C2 to finish first)*

```bash
python train/dpo.py --condition lora_sft_dpo --config configs/dpo.yaml --seed 42 --fast --no-wandb
```

---

## 5. Train Reward Model

```bash
python reward_model/train_rm.py --seed 42 --fast --no-wandb
```

---

## 6. Evaluate All Conditions

```bash
python eval/run_eval.py --fast --skip-win-rate
```

> `--skip-win-rate` skips the teacher judge step (saves time).
> Remove it if you want the full win-rate metric (requires mlx-lm + teacher model loaded).

---

## 7. Generate Dashboard

```bash
python compare/dashboard.py
```

---

## Output Locations

| What | Where |
|------|-------|
| Trained checkpoints | `checkpoints/<condition>/` |
| Merged models (for eval) | `checkpoints/<condition>_merged/` |
| Eval results (JSON) | `results/<condition>.json` |
| Comparison table | `results/comparison_table.md` |
| Comparison CSV | `results/comparison_table.csv` |
| Bar charts | `results/plots/bar_*.png` |
| Radar chart | `results/plots/radar_chart.png` |

---

## Notes

- All commands use `--fast` (10K samples) for quick iteration.
  Remove `--fast` from every command for a full training run.
- `--no-wandb` disables Weights & Biases logging.
  Remove it if you have a W&B account set up.
- C1 (baseline) needs no training — the evaluator loads `Qwen/Qwen3-0.6B` directly.
- If a training step fails, fix the issue and rerun just that step — other checkpoints are not affected.
