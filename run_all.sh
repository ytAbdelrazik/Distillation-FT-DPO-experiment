#!/usr/bin/env bash
# AlignBench — Full pipeline runner
#
# Runs the complete AlignBench pipeline end-to-end:
#   1. Prepare datasets
#   2. Generate teacher outputs (mlx-lm, one-time)
#   3. Train all 6 conditions (C1 is baseline, eval only)
#   4. Train reward model
#   5. Merge LoRA adapters for eval
#   6. Run full evaluation suite
#   7. Generate comparison dashboard
#
# Flags:
#   --dry-run   Print the plan without executing
#   --fast      Use 10K samples for quick iteration
#   --seed N    Random seed (default: 42)
#   --no-wandb  Disable W&B logging
#   --skip-teacher  Skip teacher output generation (use existing)
#   --condition C   Run only a specific condition (can be repeated)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DRY_RUN=false
FAST=false
SEED=42
NO_WANDB=false
SKIP_TEACHER=false
SELECTED_CONDITIONS=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)      DRY_RUN=true; shift ;;
        --fast)         FAST=true; shift ;;
        --seed)         SEED="$2"; shift 2 ;;
        --no-wandb)     NO_WANDB=true; shift ;;
        --skip-teacher) SKIP_TEACHER=true; shift ;;
        --condition)    SELECTED_CONDITIONS+=("$2"); shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--fast] [--seed N] [--no-wandb] [--skip-teacher] [--condition NAME]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FAST_FLAG=""
NO_WANDB_FLAG=""
if [ "$FAST" = true ]; then FAST_FLAG="--fast"; fi
if [ "$NO_WANDB" = true ]; then NO_WANDB_FLAG="--no-wandb"; fi

run() {
    # Run a command, or just echo it in dry-run mode
    echo ""
    echo ">>> $*"
    if [ "$DRY_RUN" = false ]; then
        eval "$@"
    fi
}

step() {
    echo ""
    echo "========================================================"
    echo "  STEP: $*"
    echo "========================================================"
}

should_run() {
    # Returns true if no --condition filter is set, or if condition matches
    local cond="$1"
    if [ ${#SELECTED_CONDITIONS[@]} -eq 0 ]; then
        return 0
    fi
    for c in "${SELECTED_CONDITIONS[@]}"; do
        if [ "$c" = "$cond" ]; then return 0; fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "AlignBench Pipeline — DRY RUN"
    echo "  Fast mode:  $FAST"
    echo "  Seed:       $SEED"
    echo "  No W&B:     $NO_WANDB"
    echo "  Skip teacher: $SKIP_TEACHER"
    if [ ${#SELECTED_CONDITIONS[@]} -gt 0 ]; then
        echo "  Conditions: ${SELECTED_CONDITIONS[*]}"
    else
        echo "  Conditions: all"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Prepare datasets
# ---------------------------------------------------------------------------
step "1/7 — Prepare datasets"

run "python data/prepare_alpaca.py $FAST_FLAG"
run "python data/prepare_hh_rlhf.py $FAST_FLAG"

# ---------------------------------------------------------------------------
# Step 2: Generate teacher outputs (mlx-lm, one-time)
# ---------------------------------------------------------------------------
step "2/7 — Generate teacher outputs via mlx-lm"

if [ "$SKIP_TEACHER" = true ]; then
    echo "Skipping teacher output generation (--skip-teacher)"
else
    run "python data/generate_teacher_outputs.py $FAST_FLAG"
fi

# ---------------------------------------------------------------------------
# Step 3: Train all conditions
# ---------------------------------------------------------------------------
step "3/7 — Train all conditions"

# C1: Baseline — no training, eval only
echo "C1 (baseline): Raw student — no training needed."

# C2: LoRA SFT
if should_run "lora_sft"; then
    step "C2: LoRA SFT"
    run "python train/lora_sft.py --config configs/lora_sft.yaml --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"
fi

# C3: Distillation
if should_run "distillation"; then
    step "C3: Knowledge Distillation"
    run "python train/distillation.py --config configs/distillation.yaml --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"
fi

# C4: Distill + LoRA
if should_run "distill_lora"; then
    step "C4: Distillation + LoRA"
    run "python train/distill_lora.py --config configs/distill_lora.yaml --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"
fi

# C5: Distill → DPO (requires C3 checkpoint)
if should_run "distill_dpo"; then
    step "C5: Distillation → DPO"
    run "python train/dpo.py --condition distill_dpo --config configs/dpo.yaml --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"
fi

# C6: LoRA SFT → DPO (requires C2 checkpoint)
if should_run "lora_sft_dpo"; then
    step "C6: LoRA SFT → DPO"
    run "python train/dpo.py --condition lora_sft_dpo --config configs/dpo.yaml --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"
fi

# ---------------------------------------------------------------------------
# Step 4: Train reward model
# ---------------------------------------------------------------------------
step "4/7 — Train reward model"

run "python reward_model/train_rm.py --seed $SEED $FAST_FLAG $NO_WANDB_FLAG"

# ---------------------------------------------------------------------------
# Step 5: Merge LoRA adapters for eval
# ---------------------------------------------------------------------------
step "5/7 — Merge LoRA adapters"

# LoRA conditions that need merging
LORA_CONDITIONS=("lora_sft" "distill_lora" "distill_dpo" "lora_sft_dpo")

for cond in "${LORA_CONDITIONS[@]}"; do
    if should_run "$cond"; then
        merged_path="checkpoints/${cond}_merged"
        if [ -d "$merged_path" ]; then
            echo "$cond: merged checkpoint already exists at $merged_path"
        else
            run "python -c \"
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('Merging: $cond')
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(base, 'checkpoints/$cond')
merged = model.merge_and_unload()
merged.save_pretrained('checkpoints/${cond}_merged')
tok = AutoTokenizer.from_pretrained('checkpoints/$cond', trust_remote_code=True)
tok.save_pretrained('checkpoints/${cond}_merged')
print('Saved: checkpoints/${cond}_merged')
\""
        fi
    fi
done

# ---------------------------------------------------------------------------
# Step 6: Run full evaluation
# ---------------------------------------------------------------------------
step "6/7 — Run evaluation suite"

FAST_EVAL_FLAG=""
if [ "$FAST" = true ]; then FAST_EVAL_FLAG="--fast"; fi

if [ ${#SELECTED_CONDITIONS[@]} -gt 0 ]; then
    for cond in "${SELECTED_CONDITIONS[@]}"; do
        run "python eval/run_eval.py --condition $cond $FAST_EVAL_FLAG"
    done
else
    run "python eval/run_eval.py $FAST_EVAL_FLAG"
fi

# ---------------------------------------------------------------------------
# Step 7: Generate comparison dashboard
# ---------------------------------------------------------------------------
step "7/7 — Generate comparison dashboard"

run "python compare/dashboard.py --results-dir results --output-dir results/plots"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  AlignBench pipeline complete!"
echo "  Results: results/"
echo "  Plots:   results/plots/"
echo "========================================================"
