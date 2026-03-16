"""
C2: LoRA supervised fine-tuning on Alpaca dataset.

Trains Qwen/Qwen3-0.6B with PEFT LoRA adapters on MPS / CUDA / CPU.
Teacher model is NEVER loaded here — distillation loads from disk only.

Usage:
    python train/lora_sft.py --config configs/lora_sft.yaml [--seed 42] [--fast]
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.base_trainer import (
    get_device,
    get_config_value,
    get_cosine_schedule_with_warmup,
    get_peak_memory_mb,
    load_config,
    log_metrics,
    save_checkpoint,
    set_seed,
    setup_wandb,
    TrainingTimer,
)

CONDITION_NAME = "lora_sft"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AlpacaDataset(Dataset):
    """Tokenized Alpaca instruction dataset for SFT."""

    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 512, fast: bool = False):
        with open(data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        if fast:
            records = records[:10_000]

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


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_lora_sft(
    config_path: str = "configs/lora_sft.yaml",
    seed: int = 42,
    fast: bool = False,
    lora_r: int = None,  # Override rank (for ablation)
    no_wandb: bool = False,
) -> dict:
    """
    Train LoRA SFT (C2).

    Args:
        config_path: Path to YAML config.
        seed: Random seed.
        fast: If True, limit to 10K samples.
        lora_r: Override LoRA rank (for rank ablation).
        no_wandb: Disable W&B logging.

    Returns:
        Dict with final training metrics.
    """
    config = load_config(config_path)
    set_seed(seed)

    device = get_device()
    print(f"[lora_sft] Device: {device}")

    # Config values
    student_model_id = get_config_value(config, "model", "student", default="Qwen/Qwen3-0.6B")
    r = lora_r or get_config_value(config, "lora", "r", default=16)
    lora_alpha = get_config_value(config, "lora", "alpha", default=32)
    lora_dropout = get_config_value(config, "lora", "dropout", default=0.05)
    target_modules = get_config_value(config, "lora", "target_modules", default=["q_proj", "v_proj"])

    epochs = get_config_value(config, "training", "epochs", default=3)
    batch_size = get_config_value(config, "training", "batch_size", default=4)
    grad_accum = get_config_value(config, "training", "gradient_accumulation_steps", default=8)
    lr = get_config_value(config, "training", "learning_rate", default=2e-4)
    warmup_ratio = get_config_value(config, "training", "warmup_ratio", default=0.03)
    max_seq_length = get_config_value(config, "training", "max_seq_length", default=512)

    wandb_project = get_config_value(config, "logging", "wandb_project", default="alignbench")
    save_steps = get_config_value(config, "logging", "save_steps", default=500)
    eval_steps = get_config_value(config, "logging", "eval_steps", default=500)

    output_dir = f"checkpoints/{CONDITION_NAME}"
    data_path = "data/alpaca.jsonl"

    # W&B
    run = setup_wandb(
        project=wandb_project,
        run_name=f"{CONDITION_NAME}_seed{seed}_r{r}",
        config={"condition": CONDITION_NAME, "seed": seed, "lora_r": r, **config},
        disabled=no_wandb,
    )

    # Load tokenizer + model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading tokenizer: {student_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {student_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        student_model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Dataset + DataLoader
    print(f"Loading dataset from {data_path}")
    dataset = AlpacaDataset(data_path, tokenizer, max_seq_length=max_seq_length, fast=fast)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(dataloader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop
    timer = TrainingTimer()
    timer.start()

    global_step = 0
    optimizer.zero_grad()
    final_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Labels are input_ids shifted (standard causal LM)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(
                        f"  Epoch {epoch+1}/{epochs} | Step {global_step} | "
                        f"Loss {avg_loss:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    log_metrics(run, {
                        "train/loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + batch_idx / len(dataloader),
                    }, step=global_step)

                if global_step % save_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir, step=global_step)

        final_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} done. Avg loss: {final_loss:.4f}")

    # Final save
    save_checkpoint(model, tokenizer, output_dir)

    elapsed = timer.stop()
    peak_mem = get_peak_memory_mb(device)

    metrics = {
        "condition": CONDITION_NAME,
        "seed": seed,
        "lora_r": r,
        "final_loss": final_loss,
        "training_time_sec": elapsed,
        "peak_memory_mb": peak_mem,
    }

    log_metrics(run, {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    if run is not None:
        run.finish()

    print(f"\n[lora_sft] Training complete in {elapsed:.1f}s | Peak memory: {peak_mem:.1f}MB")
    return metrics


# ---------------------------------------------------------------------------
# Rank ablation entry point
# ---------------------------------------------------------------------------

def run_rank_ablation_lora(config_path: str = "configs/lora_sft.yaml", seed: int = 42):
    """Run LoRA rank ablation across r ∈ {4, 8, 16, 32, 64}."""
    from train.base_trainer import run_rank_ablation

    def _train(**kwargs):
        return train_lora_sft(config_path=config_path, **kwargs)

    return run_rank_ablation(
        train_fn=_train,
        base_kwargs={"seed": seed, "no_wandb": False},
        ranks=[4, 8, 16, 32, 64],
        output_dir="results/rank_ablation_lora_sft",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C2: LoRA SFT training")
    parser.add_argument("--config", default="configs/lora_sft.yaml", help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use 10K samples only")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--rank-ablation", action="store_true", help="Run rank ablation")
    args = parser.parse_args()

    if args.rank_ablation:
        run_rank_ablation_lora(config_path=args.config, seed=args.seed)
    else:
        train_lora_sft(
            config_path=args.config,
            seed=args.seed,
            fast=args.fast,
            lora_r=args.lora_r,
            no_wandb=args.no_wandb,
        )


if __name__ == "__main__":
    main()
