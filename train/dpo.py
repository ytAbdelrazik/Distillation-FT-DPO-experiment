"""
DPO alignment training — used for both C5 (distill_dpo) and C6 (lora_sft_dpo).

C5: Starts from checkpoints/distillation, applies DPO on HH-RLHF
C6: Starts from checkpoints/lora_sft, applies DPO on HH-RLHF (control)

Uses trl DPOTrainer with LoRA adapters. Logs reward margin per step to W&B.

Usage:
    python train/dpo.py --condition distill_dpo --config configs/dpo.yaml [--seed 42] [--fast]
    python train/dpo.py --condition lora_sft_dpo --config configs/dpo.yaml [--seed 42] [--fast]
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.base_trainer import (
    get_device,
    get_config_value,
    get_peak_memory_mb,
    load_config,
    log_metrics,
    save_checkpoint,
    set_seed,
    setup_wandb,
    TrainingTimer,
    run_rank_ablation,
)

# Maps condition name to its source checkpoint
CONDITION_SOURCE = {
    "distill_dpo": "checkpoints/distillation",
    "lora_sft_dpo": "checkpoints/lora_sft",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HHRLHFDataset(Dataset):
    """HH-RLHF dataset formatted for DPOTrainer."""

    def __init__(self, data_path: str, fast: bool = False):
        with open(data_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        if fast:
            records = records[:10_000]

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_dpo(
    condition: str,
    config_path: str = "configs/dpo.yaml",
    seed: int = 42,
    fast: bool = False,
    lora_r: int = None,
    no_wandb: bool = False,
) -> dict:
    """
    DPO fine-tuning starting from a pre-trained checkpoint.

    Args:
        condition: 'distill_dpo' (C5) or 'lora_sft_dpo' (C6).
        config_path: Path to YAML config.
        seed: Random seed.
        fast: Limit to 10K samples.
        lora_r: Override LoRA rank.
        no_wandb: Disable W&B logging.

    Returns:
        Dict with final training metrics.
    """
    if condition not in CONDITION_SOURCE:
        raise ValueError(f"condition must be one of {list(CONDITION_SOURCE.keys())}, got {condition!r}")

    config = load_config(config_path)
    set_seed(seed)

    device = get_device()
    print(f"[{condition}] Device: {device}")

    source_checkpoint = CONDITION_SOURCE[condition]
    student_model_id = get_config_value(config, "model", "student", default="Qwen/Qwen3-0.6B")

    r = lora_r or get_config_value(config, "lora", "r", default=16)
    lora_alpha = get_config_value(config, "lora", "alpha", default=32)
    lora_dropout = get_config_value(config, "lora", "dropout", default=0.05)
    target_modules = get_config_value(config, "lora", "target_modules", default=["q_proj", "v_proj"])

    epochs = get_config_value(config, "training", "epochs", default=1)
    batch_size = get_config_value(config, "training", "batch_size", default=4)
    grad_accum = get_config_value(config, "training", "gradient_accumulation_steps", default=8)
    lr = get_config_value(config, "training", "learning_rate", default=5e-5)
    warmup_ratio = get_config_value(config, "training", "warmup_ratio", default=0.03)
    max_seq_length = get_config_value(config, "training", "max_seq_length", default=512)

    beta = get_config_value(config, "dpo", "beta", default=0.1)
    max_prompt_length = get_config_value(config, "dpo", "max_prompt_length", default=256)

    wandb_project = get_config_value(config, "logging", "wandb_project", default="alignbench")
    save_steps = get_config_value(config, "logging", "save_steps", default=200)
    eval_steps = get_config_value(config, "logging", "eval_steps", default=200)

    output_dir = f"checkpoints/{condition}"
    data_path = "data/hh_rlhf.jsonl"

    run = setup_wandb(
        project=wandb_project,
        run_name=f"{condition}_seed{seed}",
        config={"condition": condition, "seed": seed, "source": source_checkpoint, **config},
        disabled=no_wandb,
    )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    print(f"Loading tokenizer from source checkpoint: {source_checkpoint}")
    # Try loading tokenizer from source checkpoint first, fall back to base model
    try:
        tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(student_model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from source checkpoint: {source_checkpoint}")
    # Load the source checkpoint (may be a merged model or PEFT adapter)
    try:
        # Try as a regular HuggingFace model (merged or full fine-tuned)
        model = AutoModelForCausalLM.from_pretrained(
            source_checkpoint,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    except Exception:
        # Fall back to loading base model + PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, source_checkpoint)
        model = model.merge_and_unload()

    # Apply fresh LoRA adapters for DPO
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

    # Reference model (frozen copy of source)
    ref_model = AutoModelForCausalLM.from_pretrained(
        source_checkpoint,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    for param in ref_model.parameters():
        param.requires_grad = False

    from trl import DPOTrainer, DPOConfig

    print(f"Loading HH-RLHF data from {data_path}")
    dataset = HHRLHFDataset(data_path, fast=fast)

    # Split into train/eval
    split_idx = int(0.95 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

    timer = TrainingTimer()
    timer.start()

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        logging_steps=50,
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        beta=beta,
        seed=seed,
        report_to="wandb" if not no_wandb else "none",
        run_name=f"{condition}_seed{seed}",
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
        use_mps_device=(device.type == "mps"),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Custom callback to log reward margin
    class RewardMarginCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                chosen_reward = logs.get("train/rewards/chosen", None)
                rejected_reward = logs.get("train/rewards/rejected", None)
                if chosen_reward is not None and rejected_reward is not None:
                    margin = chosen_reward - rejected_reward
                    log_metrics(run, {"train/reward_margin": margin}, step=state.global_step)

    trainer.add_callback(RewardMarginCallback())

    print(f"Starting DPO training ({condition})...")
    train_result = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = timer.stop()
    peak_mem = get_peak_memory_mb(device)

    metrics = {
        "condition": condition,
        "seed": seed,
        "lora_r": r,
        "final_loss": train_result.training_loss,
        "training_time_sec": elapsed,
        "peak_memory_mb": peak_mem,
    }

    log_metrics(run, {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    if run is not None:
        run.finish()

    print(f"\n[{condition}] Training complete in {elapsed:.1f}s | Peak memory: {peak_mem:.1f}MB")
    return metrics


# ---------------------------------------------------------------------------
# Rank ablation entry point
# ---------------------------------------------------------------------------

def run_rank_ablation_dpo(
    condition: str,
    config_path: str = "configs/dpo.yaml",
    seed: int = 42,
):
    """Run DPO LoRA rank ablation across r ∈ {4, 8, 16, 32, 64}."""

    def _train(**kwargs):
        return train_dpo(condition=condition, config_path=config_path, **kwargs)

    return run_rank_ablation(
        train_fn=_train,
        base_kwargs={"seed": seed, "no_wandb": False},
        ranks=[4, 8, 16, 32, 64],
        output_dir=f"results/rank_ablation_{condition}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C5/C6: DPO alignment training")
    parser.add_argument(
        "--condition",
        required=True,
        choices=["distill_dpo", "lora_sft_dpo"],
        help="Which DPO condition to train",
    )
    parser.add_argument("--config", default="configs/dpo.yaml", help="Config YAML path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fast", action="store_true", help="Use 10K samples only")
    parser.add_argument("--lora-r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--rank-ablation", action="store_true", help="Run rank ablation")
    args = parser.parse_args()

    if args.rank_ablation:
        run_rank_ablation_dpo(condition=args.condition, config_path=args.config, seed=args.seed)
    else:
        train_dpo(
            condition=args.condition,
            config_path=args.config,
            seed=args.seed,
            fast=args.fast,
            lora_r=args.lora_r,
            no_wandb=args.no_wandb,
        )


if __name__ == "__main__":
    main()
