"""
Bradley-Terry reward model trained on HH-RLHF chosen/rejected pairs.

Architecture: Qwen/Qwen3-0.6B backbone with a linear scalar head on the
last hidden state. Trained with ranking loss (pure PyTorch, runs on MPS).

Saves to checkpoints/reward_model/
Reports Kendall's tau on held-out eval set.

Usage:
    python reward_model/train_rm.py [--config configs/lora_sft.yaml] [--seed 42] [--fast]
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
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
    set_seed,
    setup_wandb,
    TrainingTimer,
)


# ---------------------------------------------------------------------------
# Reward Model architecture
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Reward model: student backbone + linear scalar head.

    Inputs: tokenized (prompt + response) pairs.
    Output: scalar reward score.
    """

    def __init__(self, backbone_model_id: str = "Qwen/Qwen3-0.6B"):
        super().__init__()
        from transformers import AutoModelForCausalLM

        # Load backbone (student architecture, separate instance)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        # Scalar reward head
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]

        Returns:
            rewards: [B] scalar reward per sequence
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Use last hidden state at final non-padding token
        hidden = outputs.hidden_states[-1]  # [B, seq_len, H]

        # Gather last non-padding position per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_size = hidden.size(0)
        last_hidden = hidden[
            torch.arange(batch_size, device=hidden.device),
            seq_lengths.clamp(min=0),
        ]  # [B, H]

        rewards = self.reward_head(last_hidden).squeeze(-1)  # [B]
        return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    """HH-RLHF dataset formatted as (chosen, rejected) tokenized pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        fast: bool = False,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        if fast:
            records = records[:10_000]

        self.examples = []
        for record in records:
            prompt = record["prompt"]
            chosen = record["chosen"]
            rejected = record["rejected"]

            chosen_text = prompt + "\n\nAssistant: " + chosen
            rejected_text = prompt + "\n\nAssistant: " + rejected

            chosen_enc = tokenizer(
                chosen_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                rejected_text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            self.examples.append({
                "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Bradley-Terry ranking loss
# ---------------------------------------------------------------------------

def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry ranking loss:
        L = -E[log σ(r_chosen - r_rejected)]

    Args:
        chosen_rewards: [B] scalar rewards for chosen responses
        rejected_rewards: [B] scalar rewards for rejected responses

    Returns:
        Scalar loss.
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


# ---------------------------------------------------------------------------
# Kendall's tau evaluation
# ---------------------------------------------------------------------------

def compute_kendall_tau(model: RewardModel, dataloader: DataLoader, device: torch.device) -> float:
    """Compute Kendall's tau on a held-out eval set."""
    from scipy.stats import kendalltau

    model.eval()
    all_chosen = []
    all_rejected = []

    with torch.no_grad():
        for batch in dataloader:
            chosen_rewards = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            ).cpu().tolist()
            rejected_rewards = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            ).cpu().tolist()
            all_chosen.extend(chosen_rewards)
            all_rejected.extend(rejected_rewards)

    # Binary ranking: 1 if chosen > rejected, 0 otherwise
    predicted = [1 if c > r else 0 for c, r in zip(all_chosen, all_rejected)]
    actual = [1] * len(predicted)  # Ground truth: chosen is always preferred

    tau, _ = kendalltau(predicted, actual)
    return float(tau)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_reward_model(
    data_path: str = "data/hh_rlhf.jsonl",
    backbone_model_id: str = "Qwen/Qwen3-0.6B",
    output_dir: str = "checkpoints/reward_model",
    epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 1e-4,
    warmup_ratio: float = 0.03,
    max_seq_length: int = 512,
    seed: int = 42,
    fast: bool = False,
    wandb_project: str = "alignbench",
    no_wandb: bool = False,
) -> dict:
    """
    Train Bradley-Terry reward model on HH-RLHF data.

    Args:
        data_path: Path to HH-RLHF JSONL.
        backbone_model_id: HuggingFace model ID for the student backbone.
        output_dir: Where to save the reward model checkpoint.
        epochs: Number of training epochs.
        batch_size: Per-device batch size.
        grad_accum: Gradient accumulation steps.
        lr: Learning rate.
        warmup_ratio: LR warmup ratio.
        max_seq_length: Max tokenized sequence length.
        seed: Random seed.
        fast: Limit to 10K samples.
        wandb_project: W&B project name.
        no_wandb: Disable W&B logging.

    Returns:
        Dict with final metrics including Kendall's tau.
    """
    set_seed(seed)
    device = get_device()
    print(f"[reward_model] Device: {device}")

    run = setup_wandb(
        project=wandb_project,
        run_name=f"reward_model_seed{seed}",
        config={
            "backbone": backbone_model_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
        disabled=no_wandb,
    )

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {backbone_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(backbone_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building reward model...")
    model = RewardModel(backbone_model_id=backbone_model_id)
    model = model.to(device)

    print(f"Loading HH-RLHF data from {data_path}")
    dataset = PairwiseDataset(data_path, tokenizer, max_seq_length=max_seq_length, fast=fast)

    # Train/eval split
    n = len(dataset)
    split_idx = int(0.9 * n)
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, n))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * epochs
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

        for batch_idx, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(chosen_ids, chosen_mask)
            rejected_rewards = model(rejected_ids, rejected_mask)

            loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
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
                    log_metrics(run, {"train/loss": avg_loss}, step=global_step)

        final_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} done. Avg loss: {final_loss:.4f}")

        # Eval: Kendall's tau
        tau = compute_kendall_tau(model, eval_loader, device)
        print(f"Epoch {epoch+1} Kendall's tau: {tau:.4f}")
        log_metrics(run, {"eval/kendall_tau": tau, "eval/loss": final_loss}, step=global_step)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "reward_model.pt"))
    tokenizer.save_pretrained(output_dir)

    # Save config for later loading
    rm_config = {
        "backbone_model_id": backbone_model_id,
        "hidden_size": model.backbone.config.hidden_size,
    }
    with open(os.path.join(output_dir, "rm_config.json"), "w") as f:
        json.dump(rm_config, f, indent=2)

    print(f"Reward model saved to {output_dir}")

    elapsed = timer.stop()
    peak_mem = get_peak_memory_mb(device)
    final_tau = compute_kendall_tau(model, eval_loader, device)

    metrics = {
        "condition": "reward_model",
        "seed": seed,
        "final_loss": final_loss,
        "kendall_tau": final_tau,
        "training_time_sec": elapsed,
        "peak_memory_mb": peak_mem,
    }

    log_metrics(run, {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    if run is not None:
        run.finish()

    print(f"\n[reward_model] Done in {elapsed:.1f}s | Kendall's tau: {final_tau:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Bradley-Terry reward model")
    parser.add_argument("--data", default="data/hh_rlhf.jsonl", help="HH-RLHF JSONL path")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B", help="Backbone model ID")
    parser.add_argument("--output", default="checkpoints/reward_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use 10K samples only")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    train_reward_model(
        data_path=args.data,
        backbone_model_id=args.backbone,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        fast=args.fast,
        no_wandb=args.no_wandb,
    )


if __name__ == "__main__":
    main()
