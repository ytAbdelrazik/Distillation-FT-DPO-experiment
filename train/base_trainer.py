"""
Shared trainer utilities for AlignBench.

Provides:
- get_device(): auto-detect MPS / CUDA / CPU
- load_config(): parse YAML config
- setup_wandb(): initialize W&B run
- save_checkpoint(): save model + tokenizer
- get_cosine_schedule_with_warmup(): LR scheduler
- set_seed(): reproducibility helper
"""

import os
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Device detection — no hardcoded "cuda" anywhere
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(config: dict, *keys, default=None):
    """Safely get a nested config value with a default fallback."""
    obj = config
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def setup_wandb(
    project: str,
    run_name: str,
    config: dict,
    disabled: bool = False,
) -> Any:
    """Initialize a W&B run. Returns the run object."""
    try:
        import wandb
        mode = "disabled" if disabled else "online"
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
            mode=mode,
        )
        return run
    except ImportError:
        print("W&B not installed — logging disabled.")
        return None


def log_metrics(run, metrics: dict, step: Optional[int] = None) -> None:
    """Log a metrics dict to W&B if available."""
    if run is not None:
        try:
            run.log(metrics, step=step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    step: Optional[int] = None,
) -> None:
    """Save model and tokenizer to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if step is not None:
        print(f"  Checkpoint saved to {output_dir} (step {step})")
    else:
        print(f"  Checkpoint saved to {output_dir}")


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------

def get_peak_memory_mb(device: torch.device) -> float:
    """Return peak allocated memory in MB for the given device."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    elif device.type == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

class TrainingTimer:
    """Simple wall-clock timer for training runs."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed

    def elapsed_seconds(self) -> float:
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed


# ---------------------------------------------------------------------------
# LoRA rank ablation helper
# ---------------------------------------------------------------------------

def run_rank_ablation(
    train_fn,
    base_kwargs: dict,
    ranks: list[int] = [4, 8, 16, 32, 64],
    output_dir: str = "results/rank_ablation",
) -> dict:
    """
    Run a training function across multiple LoRA ranks and collect results.

    Args:
        train_fn: Callable that accepts rank as a kwarg and returns metrics dict.
        base_kwargs: Base kwargs passed to train_fn for every rank.
        ranks: List of LoRA ranks to test.
        output_dir: Directory to save per-rank results.

    Returns:
        Dict mapping rank -> metrics.
    """
    import json

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for r in ranks:
        print(f"\n--- Rank ablation: r={r} ---")
        kwargs = {**base_kwargs, "lora_r": r}
        metrics = train_fn(**kwargs)
        results[r] = metrics

        result_path = os.path.join(output_dir, f"rank_{r}.json")
        with open(result_path, "w") as f:
            json.dump({"rank": r, "metrics": metrics}, f, indent=2)
        print(f"  r={r} results saved to {result_path}")

    return results
