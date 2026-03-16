"""
Perplexity evaluation on a held-out set of Alpaca prompts.

Computes perplexity on 1K held-out Alpaca prompts using the given model checkpoint.

Usage:
    python eval/perplexity.py --checkpoint checkpoints/lora_sft_merged --data data/alpaca.jsonl
"""

import argparse
import json
import math
import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.base_trainer import get_device


HELD_OUT_SIZE = 1000


class HeldOutDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 512, n_samples: int = HELD_OUT_SIZE):
        # Use last N samples as held-out set
        with open(data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        records = records[-n_samples:]  # tail for held-out

        self.examples = []
        for record in records:
            text = record["prompt"]
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_perplexity(
    model_path: str,
    data_path: str = "data/alpaca.jsonl",
    max_seq_length: int = 512,
    batch_size: int = 4,
    n_samples: int = HELD_OUT_SIZE,
) -> float:
    """
    Compute perplexity of a model on held-out Alpaca prompts.

    Args:
        model_path: Path to merged HuggingFace model checkpoint.
        data_path: Path to Alpaca JSONL.
        max_seq_length: Max tokenized length.
        batch_size: Batch size for inference.
        n_samples: Number of held-out samples to evaluate.

    Returns:
        Perplexity score.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = get_device()
    print(f"[perplexity] Device: {device}")

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

    dataset = HeldOutDataset(data_path, tokenizer, max_seq_length=max_seq_length, n_samples=n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Loss is mean NLL over non-ignored tokens
            n_tokens = (labels != -100).sum().item()
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(avg_nll)
    print(f"[perplexity] PPL = {perplexity:.2f} over {total_tokens} tokens")
    return perplexity


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on held-out Alpaca set")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path (merged)")
    parser.add_argument("--data", default="data/alpaca.jsonl", help="Alpaca JSONL path")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=HELD_OUT_SIZE)
    args = parser.parse_args()

    ppl = compute_perplexity(
        model_path=args.checkpoint,
        data_path=args.data,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
