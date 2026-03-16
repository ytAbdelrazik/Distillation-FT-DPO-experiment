"""
Pairwise win rate evaluation — judged by teacher model via mlx-lm.

Compares a given condition's responses against the baseline (C1, raw student).
Teacher (mlx-community/Qwen3.5-9B-OptiQ-4bit) acts as judge.

This is the SECOND place mlx-lm is used (alongside data/generate_teacher_outputs.py).

Usage:
    python eval/win_rate.py --condition lora_sft --baseline-checkpoint checkpoints/baseline_merged
"""

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEACHER_MODEL = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
N_EVAL_PROMPTS = 200  # Number of prompts for win rate eval


def generate_responses(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[str]:
    """Generate responses from a HuggingFace model checkpoint."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from train.base_trainer import get_device

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, out in enumerate(output_ids):
            # Only decode the newly generated tokens
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = out[prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response.strip())

        if i % (batch_size * 10) == 0:
            print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} responses...")

    return responses


def compute_win_rate(
    condition_checkpoint: str,
    baseline_checkpoint: str,
    data_path: str = "data/alpaca.jsonl",
    n_prompts: int = N_EVAL_PROMPTS,
    fast: bool = False,
) -> dict:
    """
    Compute win rate of condition vs baseline using teacher-as-judge.

    Args:
        condition_checkpoint: Path to the condition's merged model.
        baseline_checkpoint: Path to the baseline (C1) model.
        data_path: Alpaca JSONL for prompts.
        n_prompts: Number of prompts to evaluate.
        fast: If True, use 50 prompts only.

    Returns:
        Dict with win_rate, tie_rate, loss_rate, n_comparisons.
    """
    # Import judge from generate_teacher_outputs
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_teacher_outputs import judge_pairwise

    if fast:
        n_prompts = 50

    # Load prompts
    with open(data_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Use prompts from the middle of dataset (not held-out tail, not train head)
    start_idx = int(0.7 * len(records))
    eval_records = records[start_idx:start_idx + n_prompts]
    prompts = [r["prompt"].split("### Response:")[0] + "### Response:" for r in eval_records]

    print(f"[win_rate] Generating condition responses from {condition_checkpoint}")
    condition_responses = generate_responses(condition_checkpoint, prompts)

    print(f"[win_rate] Generating baseline responses from {baseline_checkpoint}")
    baseline_responses = generate_responses(baseline_checkpoint, prompts)

    # Load teacher for judging (mlx-lm)
    print(f"[win_rate] Loading teacher judge: {TEACHER_MODEL}")
    try:
        from mlx_lm import load
        judge_model, judge_tokenizer = load(TEACHER_MODEL)
    except ImportError:
        print("[win_rate] WARNING: mlx-lm not available. Returning dummy win rate.")
        return {"win_rate": 0.0, "tie_rate": 0.0, "loss_rate": 0.0, "n_comparisons": 0}

    wins = 0
    ties = 0
    losses = 0

    for i, (prompt, cond_resp, base_resp) in enumerate(
        zip(prompts, condition_responses, baseline_responses)
    ):
        if i % 20 == 0:
            print(f"  Judging {i}/{len(prompts)}...")

        verdict = judge_pairwise(
            prompt=prompt,
            response_a=cond_resp,
            response_b=base_resp,
            model=judge_model,
            tokenizer=judge_tokenizer,
        )

        if verdict == "A":
            wins += 1
        elif verdict == "B":
            losses += 1
        else:
            ties += 1

    n = len(prompts)
    results = {
        "win_rate": wins / n if n > 0 else 0.0,
        "tie_rate": ties / n if n > 0 else 0.0,
        "loss_rate": losses / n if n > 0 else 0.0,
        "n_comparisons": n,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }

    print(
        f"[win_rate] Win: {wins} ({results['win_rate']:.1%}) | "
        f"Tie: {ties} ({results['tie_rate']:.1%}) | "
        f"Loss: {losses} ({results['loss_rate']:.1%})"
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Compute win rate vs baseline (teacher-as-judge)")
    parser.add_argument("--condition", required=True, help="Condition name (for checkpoint path)")
    parser.add_argument(
        "--condition-checkpoint",
        default=None,
        help="Explicit path to condition merged checkpoint",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/baseline",
        help="Path to baseline model checkpoint",
    )
    parser.add_argument("--data", default="data/alpaca.jsonl", help="Alpaca JSONL path")
    parser.add_argument("--n-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--fast", action="store_true", help="Use only 50 prompts")
    args = parser.parse_args()

    cond_ckpt = args.condition_checkpoint or f"checkpoints/{args.condition}_merged"

    results = compute_win_rate(
        condition_checkpoint=cond_ckpt,
        baseline_checkpoint=args.baseline_checkpoint,
        data_path=args.data,
        n_prompts=args.n_prompts,
        fast=args.fast,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
