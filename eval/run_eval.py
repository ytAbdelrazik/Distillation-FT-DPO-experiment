"""
Master evaluation runner for AlignBench.

Evaluates one or all 6 conditions on the full metric suite:
  - Perplexity
  - HellaSwag, TruthfulQA, MMLU (via lm-evaluation-harness)
  - ROUGE-L, BERTScore F1 (CNN/DailyMail)
  - Win rate vs baseline (teacher-as-judge via mlx-lm)
  - Peak memory (MB)
  - Training time (seconds, loaded from training metadata if available)

Before running metrics, LoRA adapters are merged into the base model.

Results saved to results/<condition_name>.json

Usage:
    python eval/run_eval.py                          # eval all 6 conditions
    python eval/run_eval.py --condition lora_sft     # eval single condition
    python eval/run_eval.py --condition lora_sft --fast  # quick iteration
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.base_trainer import get_device, get_peak_memory_mb

# All 6 conditions
ALL_CONDITIONS = ["baseline", "lora_sft", "distillation", "distill_lora", "distill_dpo", "lora_sft_dpo"]

# Conditions that use LoRA (need merging before eval)
LORA_CONDITIONS = {"lora_sft", "distill_lora", "distill_dpo", "lora_sft_dpo"}

STUDENT_MODEL = "Qwen/Qwen3-0.6B"
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# LoRA merge helper
# ---------------------------------------------------------------------------

def merge_lora_for_eval(condition: str) -> str:
    """
    Merge LoRA adapters into base model and save to checkpoints/<condition>_merged.

    Returns the merged checkpoint path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = f"checkpoints/{condition}"
    merged_path = f"checkpoints/{condition}_merged"

    if os.path.exists(merged_path):
        print(f"[run_eval] Merged checkpoint already exists: {merged_path}")
        return merged_path

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {adapter_path}\n"
            f"Run train/{condition}.py first."
        )

    print(f"[run_eval] Merging LoRA adapter: {adapter_path} -> {merged_path}")
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_path)

    print(f"[run_eval] Merged checkpoint saved: {merged_path}")
    return merged_path


def get_eval_checkpoint(condition: str) -> str:
    """
    Get the eval-ready checkpoint path for a condition.
    Merges LoRA if needed.
    """
    if condition == "baseline":
        # Baseline: raw student model (no training)
        return STUDENT_MODEL

    if condition in LORA_CONDITIONS:
        return merge_lora_for_eval(condition)

    # Full fine-tune (distillation C3) — use checkpoint directly
    ckpt = f"checkpoints/{condition}"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# Per-metric evaluation wrappers
# ---------------------------------------------------------------------------

def eval_perplexity(checkpoint: str, fast: bool = False) -> float:
    from eval.perplexity import compute_perplexity
    return compute_perplexity(
        model_path=checkpoint,
        data_path="data/alpaca.jsonl",
        batch_size=4,
        n_samples=100 if fast else 1000,
    )


def eval_benchmarks(checkpoint: str, fast: bool = False) -> dict:
    from eval.benchmarks import run_benchmarks
    # Skip MMLU in fast mode (slow)
    tasks = ["hellaswag", "truthfulqa_mc2"] if fast else ["hellaswag", "truthfulqa_mc2", "mmlu"]
    return run_benchmarks(model_path=checkpoint, tasks=tasks, batch_size=4)


def eval_rouge_bertscore(checkpoint: str, fast: bool = False) -> dict:
    from eval.compute_metrics import compute_rouge_bertscore
    return compute_rouge_bertscore(model_path=checkpoint, batch_size=4, fast=fast)


def eval_win_rate(condition: str, checkpoint: str, fast: bool = False) -> dict:
    from eval.win_rate import compute_win_rate
    if condition == "baseline":
        return {"win_rate": 0.5, "tie_rate": 0.0, "loss_rate": 0.5, "n_comparisons": 0}
    baseline_ckpt = get_eval_checkpoint("baseline")
    return compute_win_rate(
        condition_checkpoint=checkpoint,
        baseline_checkpoint=baseline_ckpt,
        data_path="data/alpaca.jsonl",
        fast=fast,
    )


def load_training_metadata(condition: str) -> dict:
    """Load training time and peak memory from training results if available."""
    result_path = os.path.join(RESULTS_DIR, f"{condition}.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)
        return {
            "training_time_sec": existing.get("training_time_sec", None),
            "peak_memory_mb": existing.get("peak_memory_mb", None),
        }
    return {"training_time_sec": None, "peak_memory_mb": None}


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    fast: bool = False,
    skip_win_rate: bool = False,
    skip_benchmarks: bool = False,
) -> dict:
    """
    Run the full evaluation suite on a single condition.

    Args:
        condition: Condition name (e.g., 'lora_sft').
        fast: Use smaller subsets for all metrics.
        skip_win_rate: Skip win rate eval (saves time if mlx-lm not available).
        skip_benchmarks: Skip lm-eval benchmarks.

    Returns:
        Dict with all metric results.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating condition: {condition}")
    print(f"{'='*60}")

    device = get_device()

    # Get eval-ready checkpoint
    checkpoint = get_eval_checkpoint(condition)
    print(f"[run_eval] Using checkpoint: {checkpoint}")

    results = {
        "condition": condition,
        "checkpoint": checkpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load training metadata (time, memory)
    meta = load_training_metadata(condition)
    results.update(meta)

    # 1. Perplexity
    print("\n--- Perplexity ---")
    try:
        ppl = eval_perplexity(checkpoint, fast=fast)
        results["perplexity"] = ppl
    except Exception as e:
        print(f"  ERROR: {e}")
        results["perplexity"] = None

    # 2. Benchmarks (HellaSwag, TruthfulQA, MMLU)
    if not skip_benchmarks:
        print("\n--- Benchmarks (HellaSwag, TruthfulQA, MMLU) ---")
        try:
            bench = eval_benchmarks(checkpoint, fast=fast)
            results.update(bench)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["hellaswag_acc"] = None
            results["truthfulqa_acc"] = None
            results["mmlu_acc"] = None
    else:
        results["hellaswag_acc"] = None
        results["truthfulqa_acc"] = None
        results["mmlu_acc"] = None

    # 3. ROUGE-L and BERTScore
    print("\n--- ROUGE-L and BERTScore ---")
    try:
        nlg = eval_rouge_bertscore(checkpoint, fast=fast)
        results["rouge_l"] = nlg.get("rouge_l")
        results["bertscore_f1"] = nlg.get("bertscore_f1")
    except Exception as e:
        print(f"  ERROR: {e}")
        results["rouge_l"] = None
        results["bertscore_f1"] = None

    # 4. Win rate vs baseline
    if not skip_win_rate:
        print("\n--- Win Rate vs Baseline ---")
        try:
            wr = eval_win_rate(condition, checkpoint, fast=fast)
            results["win_rate"] = wr.get("win_rate")
            results["tie_rate"] = wr.get("tie_rate")
            results["loss_rate"] = wr.get("loss_rate")
            results["n_comparisons"] = wr.get("n_comparisons")
        except Exception as e:
            print(f"  ERROR: {e}")
            results["win_rate"] = None
    else:
        results["win_rate"] = None

    # 5. Peak memory (current device)
    if results.get("peak_memory_mb") is None:
        results["peak_memory_mb"] = get_peak_memory_mb(device)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{condition}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[run_eval] Results saved to {result_path}")
    print_summary(results)

    return results


def print_summary(results: dict) -> None:
    """Print a formatted summary of eval results."""
    print(f"\n--- Results Summary: {results['condition']} ---")
    metrics = [
        ("Perplexity", "perplexity", "{:.2f}"),
        ("HellaSwag Acc", "hellaswag_acc", "{:.3f}"),
        ("TruthfulQA Acc", "truthfulqa_acc", "{:.3f}"),
        ("MMLU Acc", "mmlu_acc", "{:.3f}"),
        ("ROUGE-L", "rouge_l", "{:.4f}"),
        ("BERTScore F1", "bertscore_f1", "{:.4f}"),
        ("Win Rate vs Baseline", "win_rate", "{:.1%}"),
        ("Training Time (s)", "training_time_sec", "{:.1f}"),
        ("Peak Memory (MB)", "peak_memory_mb", "{:.1f}"),
    ]
    for label, key, fmt in metrics:
        val = results.get(key)
        if val is not None:
            try:
                print(f"  {label}: {fmt.format(val)}")
            except (ValueError, TypeError):
                print(f"  {label}: {val}")
        else:
            print(f"  {label}: N/A")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AlignBench master evaluation runner")
    parser.add_argument(
        "--condition",
        default=None,
        choices=ALL_CONDITIONS + [None],
        help="Condition to evaluate (default: all)",
    )
    parser.add_argument("--fast", action="store_true", help="Use smaller subsets for all metrics")
    parser.add_argument("--skip-win-rate", action="store_true", help="Skip win rate eval")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip lm-eval benchmarks")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else ALL_CONDITIONS

    all_results = {}
    for cond in conditions:
        try:
            results = evaluate_condition(
                condition=cond,
                fast=args.fast,
                skip_win_rate=args.skip_win_rate,
                skip_benchmarks=args.skip_benchmarks,
            )
            all_results[cond] = results
        except Exception as e:
            print(f"\nERROR evaluating {cond}: {e}")
            all_results[cond] = {"condition": cond, "error": str(e)}

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[run_eval] All results saved to {combined_path}")


if __name__ == "__main__":
    main()
