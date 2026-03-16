"""
Benchmark evaluation using lm-evaluation-harness.

Runs HellaSwag, TruthfulQA, and MMLU on a given model checkpoint.

Usage:
    python eval/benchmarks.py --checkpoint checkpoints/lora_sft_merged [--tasks hellaswag,truthfulqa,mmlu]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional


SUPPORTED_TASKS = ["hellaswag", "truthfulqa_mc2", "mmlu"]
TASK_DISPLAY_NAMES = {
    "hellaswag": "HellaSwag",
    "truthfulqa_mc2": "TruthfulQA",
    "mmlu": "MMLU",
}


def run_lm_eval(
    model_path: str,
    tasks: list[str] = None,
    num_fewshot: int = 0,
    batch_size: int = 4,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run lm-evaluation-harness on a model checkpoint.

    Args:
        model_path: Path to merged HuggingFace model.
        tasks: List of task names. Defaults to ['hellaswag', 'truthfulqa_mc2', 'mmlu'].
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size for eval.
        device: Compute device ('cuda', 'mps', 'cpu'). Auto-detected if None.
        output_dir: Directory to save raw results JSON.

    Returns:
        Dict mapping task_name -> accuracy.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train.base_trainer import get_device

    if tasks is None:
        tasks = SUPPORTED_TASKS

    if device is None:
        dev = get_device()
        device = str(dev)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    task_str = ",".join(tasks)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float32",
        "--tasks", task_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", output_dir or tempfile.mkdtemp(),
        "--log_samples",
    ]

    print(f"[benchmarks] Running lm_eval on: {task_str}")
    print(f"[benchmarks] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[benchmarks] lm_eval failed:\n{e.stderr[-2000:]}")
        # Return zeros so eval doesn't crash
        return {task: 0.0 for task in tasks}

    # Try to parse results from output directory
    metrics = {}
    if output_dir:
        for fname in os.listdir(output_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(output_dir, fname)) as f:
                        data = json.load(f)
                    results = data.get("results", {})
                    for task, task_results in results.items():
                        acc = (
                            task_results.get("acc,none")
                            or task_results.get("acc_norm,none")
                            or task_results.get("mc2,none")
                            or 0.0
                        )
                        metrics[task] = float(acc)
                    break
                except (json.JSONDecodeError, KeyError):
                    pass

    # Parse from stdout as fallback
    if not metrics:
        metrics = _parse_stdout(result.stdout, tasks)

    print(f"[benchmarks] Results: {metrics}")
    return metrics


def _parse_stdout(stdout: str, tasks: list[str]) -> dict:
    """Parse accuracy values from lm_eval stdout."""
    metrics = {}
    lines = stdout.splitlines()
    for line in lines:
        for task in tasks:
            if task in line and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                for i, part in enumerate(parts):
                    try:
                        val = float(part)
                        if 0.0 <= val <= 1.0:
                            metrics[task] = val
                            break
                    except ValueError:
                        pass
    return metrics


def run_benchmarks(
    model_path: str,
    tasks: list[str] = None,
    batch_size: int = 4,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Run all benchmark evaluations.

    Returns:
        Dict with keys: hellaswag_acc, truthfulqa_acc, mmlu_acc
    """
    if tasks is None:
        tasks = SUPPORTED_TASKS

    raw = run_lm_eval(
        model_path=model_path,
        tasks=tasks,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    return {
        "hellaswag_acc": raw.get("hellaswag", 0.0),
        "truthfulqa_acc": raw.get("truthfulqa_mc2", 0.0),
        "mmlu_acc": raw.get("mmlu", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evals (HellaSwag, TruthfulQA, MMLU)")
    parser.add_argument("--checkpoint", required=True, help="Merged model checkpoint path")
    parser.add_argument(
        "--tasks",
        default="hellaswag,truthfulqa_mc2,mmlu",
        help="Comma-separated list of tasks",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", default=None, help="Directory for raw lm_eval output")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = run_benchmarks(
        model_path=args.checkpoint,
        tasks=tasks,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
