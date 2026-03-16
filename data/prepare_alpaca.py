"""
Prepare Alpaca dataset: download yahma/alpaca-cleaned and save formatted prompts to disk.
Usage:
    python data/prepare_alpaca.py [--fast] [--output data/alpaca.jsonl]
"""

import argparse
import json
import os


def format_prompt(example: dict) -> str:
    """Format an Alpaca example into an instruction-following prompt."""
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if input_text:
        prompt = (
            f"Below is an instruction that describes a task, paired with an input that provides "
            f"further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"Below is an instruction that describes a task. Write a response that appropriately "
            f"completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return prompt


def prepare_alpaca(output_path: str = "data/alpaca.jsonl", fast: bool = False) -> str:
    """
    Download and format the Alpaca dataset.

    Args:
        output_path: Path to save the formatted JSONL file.
        fast: If True, limit to 10K samples for quick iteration.

    Returns:
        Path to the saved file.
    """
    from datasets import load_dataset

    print("Loading yahma/alpaca-cleaned dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    if fast:
        dataset = dataset.select(range(min(10_000, len(dataset))))
        print(f"Fast mode: using {len(dataset)} samples")
    else:
        print(f"Full dataset: {len(dataset)} samples")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    records = []
    for example in dataset:
        record = {
            "instruction": example.get("instruction", "").strip(),
            "input": example.get("input", "").strip(),
            "output": example.get("output", "").strip(),
            "prompt": format_prompt(example),
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset")
    parser.add_argument("--output", default="data/alpaca.jsonl", help="Output JSONL path")
    parser.add_argument("--fast", action="store_true", help="Use only 10K samples")
    args = parser.parse_args()

    prepare_alpaca(output_path=args.output, fast=args.fast)


if __name__ == "__main__":
    main()
