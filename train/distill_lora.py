"""
C4: Combined knowledge distillation + LoRA fine-tuning.

Combines KD loss (from pre-generated teacher soft labels) with LoRA adapters.
Teacher model is NEVER loaded here — loads from data/teacher_outputs.jsonl only.

Loss = alpha * KL(student || teacher) + (1 - alpha) * CrossEntropy(student, hard_labels)

Usage:
    python train/distill_lora.py --config configs/distill_lora.yaml [--seed 42] [--fast]
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
    run_rank_ablation,
)
from train.distillation import DistillationDataset, collate_fn, distillation_loss

CONDITION_NAME = "distill_lora"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_distill_lora(
    config_path: str = "configs/distill_lora.yaml",
    seed: int = 42,
    fast: bool = False,
    lora_r: int = None,
    no_wandb: bool = False,
) -> dict:
    """
    Train distillation + LoRA (C4).

    Teacher outputs loaded from disk — teacher model is never loaded here.

    Args:
        config_path: Path to YAML config.
        seed: Random seed.
        fast: Limit to 10K samples.
        lora_r: Override LoRA rank (for rank ablation).
        no_wandb: Disable W&B logging.

    Returns:
        Dict with final training metrics.
    """
    config = load_config(config_path)
    set_seed(seed)

    device = get_device()
    print(f"[distill_lora] Device: {device}")

    student_model_id = get_config_value(config, "model", "student", default="Qwen/Qwen3-0.6B")
    teacher_outputs_path = get_config_value(
        config, "model", "teacher_outputs", default="data/teacher_outputs.jsonl"
    )

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

    temperature = get_config_value(config, "distillation", "temperature", default=4.0)
    alpha = get_config_value(config, "distillation", "alpha", default=0.7)

    wandb_project = get_config_value(config, "logging", "wandb_project", default="alignbench")
    save_steps = get_config_value(config, "logging", "save_steps", default=500)

    output_dir = f"checkpoints/{CONDITION_NAME}"

    run = setup_wandb(
        project=wandb_project,
        run_name=f"{CONDITION_NAME}_seed{seed}_r{r}",
        config={"condition": CONDITION_NAME, "seed": seed, "lora_r": r, **config},
        disabled=no_wandb,
    )

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

    vocab_size = model.config.vocab_size

    # Dataset — teacher soft labels from disk, never loads teacher model
    print(f"Loading teacher outputs from {teacher_outputs_path}")
    dataset = DistillationDataset(
        teacher_outputs_path,
        tokenizer,
        max_seq_length=max_seq_length,
        fast=fast,
        vocab_size=vocab_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(dataloader) // grad_accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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
            teacher_soft_labels = batch["teacher_soft_labels"]

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = distillation_loss(
                student_logits=outputs.logits,
                teacher_soft_labels=teacher_soft_labels,
                labels=labels,
                temperature=temperature,
                alpha=alpha,
                vocab_size=vocab_size,
            )

            (loss / grad_accum).backward()
            epoch_loss += loss.item()
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
                        f"Loss {avg_loss:.4f}"
                    )
                    log_metrics(run, {
                        "train/loss": avg_loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % save_steps == 0:
                    save_checkpoint(model, tokenizer, output_dir, step=global_step)

        final_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} done. Avg loss: {final_loss:.4f}")

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

    print(f"\n[distill_lora] Training complete in {elapsed:.1f}s | Peak memory: {peak_mem:.1f}MB")
    return metrics


# ---------------------------------------------------------------------------
# Rank ablation entry point
# ---------------------------------------------------------------------------

def run_rank_ablation_distill_lora(config_path: str = "configs/distill_lora.yaml", seed: int = 42):
    """Run LoRA rank ablation for C4 across r ∈ {4, 8, 16, 32, 64}."""

    def _train(**kwargs):
        return train_distill_lora(config_path=config_path, **kwargs)

    return run_rank_ablation(
        train_fn=_train,
        base_kwargs={"seed": seed, "no_wandb": False},
        ranks=[4, 8, 16, 32, 64],
        output_dir="results/rank_ablation_distill_lora",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C4: Distillation + LoRA training")
    parser.add_argument("--config", default="configs/distill_lora.yaml", help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use 10K samples only")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--rank-ablation", action="store_true", help="Run rank ablation")
    args = parser.parse_args()

    if args.rank_ablation:
        run_rank_ablation_distill_lora(config_path=args.config, seed=args.seed)
    else:
        train_distill_lora(
            config_path=args.config,
            seed=args.seed,
            fast=args.fast,
            lora_r=args.lora_r,
            no_wandb=args.no_wandb,
        )


if __name__ == "__main__":
    main()
