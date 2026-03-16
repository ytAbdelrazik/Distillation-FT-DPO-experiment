"""
Prepare HH-RLHF preference dataset for DPO training.

Downloads Anthropic/hh-rlhf and formats chosen/rejected pairs
into a unified format saved to data/hh_rlhf.jsonl.

Usage:
    python data/prepare_hh_rlhf.py [--fast] [--output data/hh_rlhf.jsonl]
"""

import argparse
import json
import os
import re


def extract_last_exchange(text: str) -> tuple[str, str]:
    """
    Extract the last (prompt, response) pair from an HH-RLHF conversation.

    HH-RLHF format:
        \n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...
    """
    # Split on Human/Assistant turns
    human_turns = re.split(r"\n\nHuman: ", text)
    # Take last human turn
    if len(human_turns) < 2:
        # Fallback: use entire text as prompt
        return text.strip(), ""

    last_human = human_turns[-1]
    parts = re.split(r"\n\nAssistant: ", last_human, maxsplit=1)

    if len(parts) == 2:
        prompt = parts[0].strip()
        response = parts[1].strip()
    else:
        prompt = last_human.strip()
        response = ""

    return prompt, response


def format_full_context(text: str) -> str:
    """Keep the full conversation context as the prompt, stripping only the final assistant turn."""
    # Find the last \n\nAssistant: and remove everything after it
    idx = text.rfind("\n\nAssistant: ")
    if idx != -1:
        return text[:idx].strip()
    return text.strip()


def prepare_hh_rlhf(
    output_path: str = "data/hh_rlhf.jsonl",
    fast: bool = False,
    split: str = "train",
) -> str:
    """
    Download and format the HH-RLHF dataset.

    Args:
        output_path: Path to save the formatted JSONL file.
        fast: If True, limit to 10K samples.
        split: Dataset split ('train' or 'test').

    Returns:
        Path to the saved file.
    """
    from datasets import load_dataset

    print(f"Loading Anthropic/hh-rlhf ({split} split)...")
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)

    if fast:
        dataset = dataset.select(range(min(10_000, len(dataset))))
        print(f"Fast mode: using {len(dataset)} samples")
    else:
        print(f"Full dataset: {len(dataset)} samples")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    records = []
    skipped = 0

    for example in dataset:
        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")

        if not chosen_text or not rejected_text:
            skipped += 1
            continue

        # Extract prompt (full context minus final assistant turn)
        prompt = format_full_context(chosen_text)

        # Extract responses
        _, chosen_response = extract_last_exchange(chosen_text)
        _, rejected_response = extract_last_exchange(rejected_text)

        if not chosen_response or not rejected_response:
            skipped += 1
            continue

        record = {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} records to {output_path} (skipped {skipped})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF preference dataset")
    parser.add_argument("--output", default="data/hh_rlhf.jsonl", help="Output JSONL path")
    parser.add_argument("--fast", action="store_true", help="Use only 10K samples")
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Dataset split")
    args = parser.parse_args()

    prepare_hh_rlhf(output_path=args.output, fast=args.fast, split=args.split)


if __name__ == "__main__":
    main()
