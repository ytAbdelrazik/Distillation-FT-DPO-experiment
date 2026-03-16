"""
ROUGE-L and BERTScore F1 evaluation on CNN/DailyMail summarization subset.

Usage:
    python eval/compute_metrics.py --checkpoint checkpoints/lora_sft_merged
"""

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_SUMMARIZATION_SAMPLES = 500


def load_cnn_dailymail(n_samples: int = N_SUMMARIZATION_SAMPLES) -> list[dict]:
    """Load CNN/DailyMail summarization examples."""
    from datasets import load_dataset

    print("[compute_metrics] Loading CNN/DailyMail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    dataset = dataset.select(range(min(n_samples, len(dataset))))

    records = []
    for ex in dataset:
        records.append({
            "article": ex["article"],
            "highlights": ex["highlights"],
            "prompt": (
                f"Summarize the following article in 2-3 sentences:\n\n"
                f"{ex['article'][:1500]}\n\nSummary:"
            ),
        })
    print(f"[compute_metrics] Loaded {len(records)} CNN/DailyMail samples.")
    return records


def generate_summaries(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int = 128,
    batch_size: int = 4,
) -> list[str]:
    """Generate summaries from a HuggingFace model."""
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

    summaries = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, out in enumerate(output_ids):
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = out[prompt_len:]
            summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
            summaries.append(summary.strip())

        if i % (batch_size * 5) == 0:
            print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} summaries...")

    return summaries


def compute_rouge_bertscore(
    model_path: str,
    n_samples: int = N_SUMMARIZATION_SAMPLES,
    batch_size: int = 4,
    fast: bool = False,
) -> dict:
    """
    Compute ROUGE-L and BERTScore F1 on CNN/DailyMail.

    Args:
        model_path: Path to merged HuggingFace model.
        n_samples: Number of CNN/DM test samples.
        batch_size: Inference batch size.
        fast: Use 100 samples only.

    Returns:
        Dict with rouge_l and bertscore_f1.
    """
    from rouge_score import rouge_scorer
    import bert_score

    if fast:
        n_samples = 100

    records = load_cnn_dailymail(n_samples=n_samples)
    prompts = [r["prompt"] for r in records]
    references = [r["highlights"] for r in records]

    print(f"[compute_metrics] Generating summaries from {model_path}")
    predictions = generate_summaries(model_path, prompts, batch_size=batch_size)

    # ROUGE-L
    print("[compute_metrics] Computing ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for ref, pred in zip(references, predictions)]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
    print(f"[compute_metrics] ROUGE-L: {avg_rouge_l:.4f}")

    # BERTScore
    print("[compute_metrics] Computing BERTScore F1...")
    P, R, F1 = bert_score.score(
        predictions,
        references,
        lang="en",
        verbose=False,
        batch_size=batch_size,
    )
    avg_bertscore_f1 = F1.mean().item()
    print(f"[compute_metrics] BERTScore F1: {avg_bertscore_f1:.4f}")

    return {
        "rouge_l": avg_rouge_l,
        "bertscore_f1": avg_bertscore_f1,
        "n_samples": len(records),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE-L and BERTScore on CNN/DailyMail")
    parser.add_argument("--checkpoint", required=True, help="Merged model checkpoint path")
    parser.add_argument("--n-samples", type=int, default=N_SUMMARIZATION_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fast", action="store_true", help="Use 100 samples only")
    args = parser.parse_args()

    results = compute_rouge_bertscore(
        model_path=args.checkpoint,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        fast=args.fast,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
