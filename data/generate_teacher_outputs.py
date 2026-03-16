"""
Generate teacher outputs using mlx-lm (Apple Silicon only).

This is the ONLY file that uses mlx-lm. It:
1. Loads mlx-community/Qwen3.5-9B-OptiQ-4bit via mlx-lm
2. Runs inference on all Alpaca prompts
3. Extracts soft labels (top-k logits) and token_ids
4. Saves to data/teacher_outputs.jsonl

The teacher model is never loaded during training — training scripts load from disk.

Usage:
    python data/generate_teacher_outputs.py [--fast] [--input data/alpaca.jsonl] [--output data/teacher_outputs.jsonl]
"""

import argparse
import json
import os
from typing import Optional


TEACHER_MODEL = "mlx-community/Qwen3.5-9B-OptiQ-4bit"
TOP_K_LOGITS = 50  # Store top-50 token logits as soft labels


def generate_teacher_outputs(
    input_path: str = "data/alpaca.jsonl",
    output_path: str = "data/teacher_outputs.jsonl",
    fast: bool = False,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = TOP_K_LOGITS,
    num_records: int = None,
    start_from: int = 0,
) -> str:
    """
    Generate teacher soft labels for all prompts in input_path.

    Args:
        input_path: Path to Alpaca JSONL (from prepare_alpaca.py).
        output_path: Path to save teacher outputs.
        fast: If True, process only 10K samples.
        max_new_tokens: Max tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Number of top logits to store as soft labels.

    Returns:
        Path to the saved file.
    """
    # Import mlx-lm here — only this file uses it
    try:
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        raise ImportError(
            "mlx-lm is required for teacher output generation. "
            "Install with: pip install mlx-lm>=0.18.0\n"
            f"Original error: {e}"
        )

    print(f"Loading teacher model: {TEACHER_MODEL}")
    model, tokenizer = load(TEACHER_MODEL)
    print("Teacher model loaded.")

    # Load prompts
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if num_records is not None:
        records = records[start_from:start_from + num_records]
    elif fast:
        records = records[start_from:start_from + 10_000]
    else:
        records = records[start_from:]

    print(f"Processing {len(records)} prompts (max_new_tokens={max_new_tokens})")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    written = 0
    file_mode = "a" if start_from > 0 else "w"
    with open(output_path, file_mode, encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            if i % 500 == 0:
                print(f"  [{i}/{len(records)}] generating teacher outputs...")

            prompt_text = record["prompt"]

            # Generate token-by-token using stream_generate (mlx-lm >= 0.18)
            # Each GenerationResponse has .token (int) and .logprobs (mx.array of log-probs)
            token_ids = []
            soft_labels = []  # list of {token_id: logit} dicts (top-k per step)

            try:
                sampler = make_sampler(temp=temperature)
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt_text,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                ):
                    token_ids.append(int(response.token))

                    # response.logprobs is an mx.array of log-probs over vocab
                    if response.logprobs is not None:
                        lp = response.logprobs.tolist()
                        indexed = sorted(enumerate(lp), key=lambda x: x[1], reverse=True)[:top_k]
                        soft_labels.append({int(idx): float(val) for idx, val in indexed})
                    else:
                        soft_labels.append({})

                    if response.finish_reason is not None:
                        break

            except Exception as e:
                print(f"  Warning: generation failed for record {i}: {e}")
                token_ids = []
                soft_labels = []

            output_record = {
                "prompt": prompt_text,
                "instruction": record.get("instruction", ""),
                "input": record.get("input", ""),
                "reference_output": record.get("output", ""),
                "token_ids": token_ids,
                "soft_labels": soft_labels,
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} teacher outputs to {output_path}")
    return output_path


def judge_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    model=None,
    tokenizer=None,
    teacher_model_id: str = TEACHER_MODEL,
) -> str:
    """
    Use the teacher model to judge which of two responses is better.
    Returns "A", "B", or "tie".

    This function is also used by eval/win_rate.py.
    """
    try:
        from mlx_lm import load, generate
    except ImportError:
        raise ImportError("mlx-lm required for pairwise judging.")

    if model is None or tokenizer is None:
        model, tokenizer = load(teacher_model_id)

    judge_prompt = (
        f"You are an impartial judge evaluating two AI assistant responses.\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Which response is better? Answer with exactly one word: A, B, or tie."
    )

    result = generate(model, tokenizer, prompt=judge_prompt, max_tokens=5)
    result = result.strip().upper()

    if result.startswith("A"):
        return "A"
    elif result.startswith("B"):
        return "B"
    else:
        return "tie"


def main():
    parser = argparse.ArgumentParser(description="Generate teacher outputs via mlx-lm")
    parser.add_argument("--input", default="data/alpaca.jsonl", help="Input Alpaca JSONL")
    parser.add_argument("--output", default="data/teacher_outputs.jsonl", help="Output JSONL path")
    parser.add_argument("--fast", action="store_true", help="Process only 10K samples")
    parser.add_argument("--num-records", type=int, default=None, help="Process exactly N records (overrides --fast)")
    parser.add_argument("--start-from", type=int, default=0, help="Skip first N records (resume from offset)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=TOP_K_LOGITS, help="Top-k logits to store")
    args = parser.parse_args()

    generate_teacher_outputs(
        input_path=args.input,
        output_path=args.output,
        fast=args.fast,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        num_records=args.num_records,
        start_from=args.start_from,
    )


if __name__ == "__main__":
    main()
