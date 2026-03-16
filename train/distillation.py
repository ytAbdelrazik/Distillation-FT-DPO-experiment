"""
C3: Knowledge distillation from pre-generated teacher soft labels.

Loads teacher soft labels from data/teacher_outputs.jsonl (written by
data/generate_teacher_outputs.py). The teacher model is NEVER loaded here.

Loss = alpha * KL(student || teacher) + (1 - alpha) * CrossEntropy(student, hard_labels)

Usage:
    python train/distillation.py --config configs/distillation.yaml [--seed 42] [--fast]
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
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

CONDITION_NAME = "distillation"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DistillationDataset(Dataset):
    """
    Dataset that pairs tokenized prompts with teacher soft labels loaded from disk.
    Teacher model is never loaded — only the pre-generated JSONL.
    """

    def __init__(
        self,
        teacher_outputs_path: str,
        tokenizer,
        max_seq_length: int = 512,
        fast: bool = False,
        vocab_size: Optional[int] = None,
    ):
        with open(teacher_outputs_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        if fast:
            records = records[:10_000]

        self.examples = []
        self.vocab_size = vocab_size

        for record in records:
            prompt_text = record["prompt"]
            token_ids = record.get("token_ids", [])
            soft_labels_list = record.get("soft_labels", [])  # list of {tok_id: logit}

            if not token_ids:
                continue

            # Tokenize the prompt
            encoded = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "teacher_token_ids": token_ids[:max_seq_length],
                "teacher_soft_labels": soft_labels_list[:max_seq_length],
                "reference_output": record.get("reference_output", ""),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "teacher_token_ids": ex["teacher_token_ids"],
            "teacher_soft_labels": ex["teacher_soft_labels"],
        }


def collate_fn(batch):
    """Custom collate to handle variable-length teacher soft labels."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    teacher_token_ids = [b["teacher_token_ids"] for b in batch]
    teacher_soft_labels = [b["teacher_soft_labels"] for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_token_ids": teacher_token_ids,
        "teacher_soft_labels": teacher_soft_labels,
    }


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_soft_labels: list,  # list of dicts {token_id: logit}
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    vocab_size: int = 32000,
) -> torch.Tensor:
    """
    Compute mixed distillation + cross-entropy loss.

    Loss = alpha * KL(student_soft || teacher_soft) + (1 - alpha) * CE(student, hard_labels)

    Args:
        student_logits: [B, seq_len, vocab_size]
        teacher_soft_labels: list[list[dict]] — B x seq_len x {token_id: logit}
        labels: [B, seq_len] hard token labels (-100 for ignored positions)
        temperature: Soft label temperature
        alpha: Weight on KL loss
        vocab_size: Vocabulary size

    Returns:
        Scalar loss tensor.
    """
    B, seq_len, V = student_logits.shape
    device = student_logits.device

    # Hard label cross-entropy loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        ignore_index=-100,
    )

    # Build teacher soft distribution tensor from sparse dicts
    # Shape: [B, seq_len, V]
    teacher_logits = torch.zeros(B, seq_len, V, device=device)
    for b_idx, seq_soft_labels in enumerate(teacher_soft_labels):
        for t_idx, tok_logits in enumerate(seq_soft_labels):
            if t_idx >= seq_len:
                break
            for tok_id_str, logit_val in tok_logits.items():
                tok_id = int(tok_id_str)
                if tok_id < V:
                    teacher_logits[b_idx, t_idx, tok_id] = float(logit_val)

    # Temperature-scaled softmax
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum over vocab, mean over valid positions
    kl = F.kl_div(student_soft, teacher_soft, reduction="none").sum(-1)

    # Mask padding positions (where labels == -100)
    mask = (labels != -100).float()
    kl_loss = (kl * mask).sum() / mask.sum().clamp(min=1)

    # Scale KL by T^2 (standard practice)
    kl_loss = kl_loss * (temperature ** 2)

    total_loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    return total_loss


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_distillation(
    config_path: str = "configs/distillation.yaml",
    seed: int = 42,
    fast: bool = False,
    no_wandb: bool = False,
) -> dict:
    """
    Train knowledge distillation model (C3).

    Teacher outputs loaded from disk — teacher model is never loaded here.

    Args:
        config_path: Path to YAML config.
        seed: Random seed.
        fast: Limit to 10K samples.
        no_wandb: Disable W&B logging.

    Returns:
        Dict with final training metrics.
    """
    config = load_config(config_path)
    set_seed(seed)

    device = get_device()
    print(f"[distillation] Device: {device}")

    student_model_id = get_config_value(config, "model", "student", default="Qwen/Qwen3-0.6B")
    teacher_outputs_path = get_config_value(
        config, "model", "teacher_outputs", default="data/teacher_outputs.jsonl"
    )

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
        run_name=f"{CONDITION_NAME}_seed{seed}",
        config={"condition": CONDITION_NAME, "seed": seed, **config},
        disabled=no_wandb,
    )

    from transformers import AutoTokenizer, AutoModelForCausalLM

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
    model = model.to(device)

    vocab_size = model.config.vocab_size

    # Dataset — loads teacher soft labels from disk
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
        "final_loss": final_loss,
        "training_time_sec": elapsed,
        "peak_memory_mb": peak_mem,
    }

    log_metrics(run, {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    if run is not None:
        run.finish()

    print(f"\n[distillation] Training complete in {elapsed:.1f}s | Peak memory: {peak_mem:.1f}MB")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C3: Knowledge distillation training")
    parser.add_argument("--config", default="configs/distillation.yaml", help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use 10K samples only")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    train_distillation(
        config_path=args.config,
        seed=args.seed,
        fast=args.fast,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
