"""
AlignBench comparison dashboard.

Loads all results/*.json files and produces:
  - Comparison table (markdown + CSV) with all conditions × all metrics
  - Bar charts comparing each metric across conditions
  - Radar chart showing overall profile of each condition
  - Best condition per metric highlighted in green
  - Summary: which condition wins overall, which is most compute-efficient

Usage:
    python compare/dashboard.py [--results-dir results] [--output-dir results/plots]
"""

import argparse
import glob
import json
import math
import os
import sys


METRICS = [
    ("perplexity", "Perplexity", False),          # lower is better
    ("hellaswag_acc", "HellaSwag Acc", True),      # higher is better
    ("truthfulqa_acc", "TruthfulQA Acc", True),
    ("mmlu_acc", "MMLU Acc", True),
    ("rouge_l", "ROUGE-L", True),
    ("bertscore_f1", "BERTScore F1", True),
    ("win_rate", "Win Rate", True),
    ("training_time_sec", "Train Time (s)", False),
    ("peak_memory_mb", "Peak Mem (MB)", False),
]

CONDITION_ORDER = [
    "baseline",
    "lora_sft",
    "distillation",
    "distill_lora",
    "distill_dpo",
    "lora_sft_dpo",
]

CONDITION_LABELS = {
    "baseline": "C1: Baseline",
    "lora_sft": "C2: LoRA SFT",
    "distillation": "C3: Distillation",
    "distill_lora": "C4: Distill+LoRA",
    "distill_dpo": "C5: Distill+DPO",
    "lora_sft_dpo": "C6: LoRA+DPO",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str = "results") -> dict:
    """Load all condition result JSON files."""
    all_results = {}

    for condition in CONDITION_ORDER:
        path = os.path.join(results_dir, f"{condition}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            all_results[condition] = data
        else:
            print(f"[dashboard] Missing results for: {condition}")

    if not all_results:
        # Try loading from all_results.json
        combined_path = os.path.join(results_dir, "all_results.json")
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                all_results = json.load(f)
            print(f"[dashboard] Loaded from {combined_path}")

    return all_results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def build_table(all_results: dict) -> tuple[list[list], list[str]]:
    """Build a 2D table of [condition × metric] values."""
    metric_keys = [m[0] for m in METRICS]
    metric_labels = [m[1] for m in METRICS]
    higher_is_better = {m[0]: m[2] for m in METRICS}

    rows = []
    for cond in CONDITION_ORDER:
        if cond not in all_results:
            continue
        data = all_results[cond]
        row = [CONDITION_LABELS.get(cond, cond)]
        for key in metric_keys:
            val = data.get(key)
            row.append(val)
        rows.append(row)

    return rows, ["Condition"] + metric_labels


def find_best_per_metric(rows: list[list]) -> dict:
    """Find which row index has the best value for each metric column."""
    best = {}
    for col_idx, (key, _, higher) in enumerate(METRICS):
        col = col_idx + 1  # skip condition name col
        values = [(r_idx, r[col]) for r_idx, r in enumerate(rows) if r[col] is not None]
        if not values:
            continue
        if higher:
            best_idx = max(values, key=lambda x: x[1])[0]
        else:
            best_idx = min(values, key=lambda x: x[1])[0]
        best[col] = best_idx
    return best


def format_val(val, key: str) -> str:
    """Format a metric value for display."""
    if val is None:
        return "N/A"
    if key in ("perplexity",):
        return f"{val:.2f}"
    if key in ("training_time_sec",):
        return f"{val:.1f}"
    if key in ("peak_memory_mb",):
        return f"{val:.0f}"
    if key in ("win_rate", "hellaswag_acc", "truthfulqa_acc", "mmlu_acc"):
        return f"{val:.1%}"
    if key in ("rouge_l", "bertscore_f1"):
        return f"{val:.4f}"
    return str(val)


def save_markdown_table(
    rows: list[list],
    headers: list[str],
    best_per_col: dict,
    output_path: str,
) -> None:
    """Save comparison table as markdown."""
    metric_keys = [m[0] for m in METRICS]

    lines = []
    lines.append("# AlignBench — Results Comparison\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r_idx, row in enumerate(rows):
        cells = []
        for c_idx, val in enumerate(row):
            if c_idx == 0:
                cells.append(str(val))
                continue
            key = metric_keys[c_idx - 1]
            formatted = format_val(val, key)
            if best_per_col.get(c_idx) == r_idx:
                formatted = f"**{formatted}**"  # bold = best
            cells.append(formatted)
        lines.append("| " + " | ".join(cells) + " |")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[dashboard] Markdown table saved: {output_path}")


def save_csv_table(rows: list[list], headers: list[str], output_path: str) -> None:
    """Save comparison table as CSV."""
    import csv

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([v if v is not None else "" for v in row])
    print(f"[dashboard] CSV table saved: {output_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar_charts(all_results: dict, output_dir: str) -> None:
    """Produce one bar chart per metric across all conditions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    os.makedirs(output_dir, exist_ok=True)

    conditions = [c for c in CONDITION_ORDER if c in all_results]
    labels = [CONDITION_LABELS.get(c, c) for c in conditions]

    for key, display_name, higher_is_better in METRICS:
        values = [all_results[c].get(key) for c in conditions]

        # Skip if all None
        if all(v is None for v in values):
            continue

        # Replace None with 0 for plotting
        plot_values = [v if v is not None else 0.0 for v in values]
        colors = []
        best_idx = None

        if higher_is_better:
            valid = [(i, v) for i, v in enumerate(plot_values) if values[i] is not None]
            if valid:
                best_idx = max(valid, key=lambda x: x[1])[0]
        else:
            valid = [(i, v) for i, v in enumerate(plot_values) if values[i] is not None]
            if valid:
                best_idx = min(valid, key=lambda x: x[1])[0]

        for i in range(len(conditions)):
            if i == best_idx:
                colors.append("#2ca02c")  # green = best
            else:
                colors.append(COLORS[i % len(COLORS)])

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, plot_values, color=colors, edgecolor="white", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    format_val(val, key),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_title(f"{display_name} by Condition", fontsize=14, fontweight="bold")
        ax.set_ylabel(display_name)
        ax.set_xticklabels(labels, rotation=15, ha="right")

        arrow = "↓ lower is better" if not higher_is_better else "↑ higher is better"
        ax.text(0.98, 0.98, arrow, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="gray")

        plt.tight_layout()
        fname = os.path.join(output_dir, f"bar_{key}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[dashboard] Bar chart saved: {fname}")


def plot_radar_chart(all_results: dict, output_dir: str) -> None:
    """Produce a radar chart comparing all conditions across key metrics."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Radar metrics: exclude time/memory, normalize to [0, 1]
    radar_metrics = [
        ("perplexity", "Perplexity\n(inv.)", False),
        ("hellaswag_acc", "HellaSwag", True),
        ("truthfulqa_acc", "TruthfulQA", True),
        ("mmlu_acc", "MMLU", True),
        ("rouge_l", "ROUGE-L", True),
        ("bertscore_f1", "BERTScore", True),
        ("win_rate", "Win Rate", True),
    ]

    conditions = [c for c in CONDITION_ORDER if c in all_results]
    labels_list = [CONDITION_LABELS.get(c, c) for c in conditions]
    n_metrics = len(radar_metrics)

    # Normalize all metrics to [0, 1]
    normalized = {c: [] for c in conditions}

    for key, _, higher in radar_metrics:
        vals = {c: all_results[c].get(key) for c in conditions}
        valid_vals = [v for v in vals.values() if v is not None]

        if not valid_vals:
            for c in conditions:
                normalized[c].append(0.0)
            continue

        min_v = min(valid_vals)
        max_v = max(valid_vals)
        rng = max_v - min_v if max_v != min_v else 1.0

        for c in conditions:
            v = vals[c]
            if v is None:
                normalized[c].append(0.0)
            else:
                norm = (v - min_v) / rng
                if not higher:
                    norm = 1.0 - norm  # invert for "lower is better"
                normalized[c].append(norm)

    # Angles for radar
    angles = [n / float(n_metrics) * 2 * math.pi for n in range(n_metrics)]
    angles += angles[:1]  # close the polygon

    metric_labels = [m[1] for m in radar_metrics]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (c, label) in enumerate(zip(conditions, labels_list)):
        values = normalized[c] + normalized[c][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("AlignBench Condition Profiles", size=14, fontweight="bold", pad=20)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[dashboard] Radar chart saved: {fname}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_overall_winner(all_results: dict, rows: list[list], best_per_col: dict) -> None:
    """Print which condition wins overall and which is most compute-efficient."""
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    # Count wins per condition
    win_counts = {c: 0 for c in CONDITION_ORDER if c in all_results}
    conditions_present = [c for c in CONDITION_ORDER if c in all_results]

    for col_idx, best_row_idx in best_per_col.items():
        if best_row_idx < len(rows):
            cond_label = rows[best_row_idx][0]
            # Find condition by label
            for c, label in CONDITION_LABELS.items():
                if label == cond_label and c in win_counts:
                    win_counts[c] += 1

    print("\nMetric wins per condition:")
    for c in CONDITION_ORDER:
        if c in win_counts:
            print(f"  {CONDITION_LABELS.get(c, c)}: {win_counts[c]} wins")

    if win_counts:
        overall_winner = max(win_counts, key=win_counts.get)
        print(f"\n  Overall winner: {CONDITION_LABELS.get(overall_winner, overall_winner)}")

    # Most compute-efficient: best perplexity (or ROUGE-L) per unit of training time
    efficiency = {}
    for c in conditions_present:
        data = all_results[c]
        ppl = data.get("perplexity")
        time_s = data.get("training_time_sec")
        if ppl and time_s and time_s > 0:
            # Lower perplexity = better; lower time = better
            # Efficiency = 1 / (ppl * time_s) — higher is more efficient
            efficiency[c] = 1.0 / (ppl * time_s)

    if efficiency:
        most_efficient = max(efficiency, key=efficiency.get)
        print(f"  Most compute-efficient: {CONDITION_LABELS.get(most_efficient, most_efficient)}")

    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dashboard(
    results_dir: str = "results",
    output_dir: str = "results/plots",
) -> None:
    """Generate the full comparison dashboard."""
    all_results = load_results(results_dir)

    if not all_results:
        print("[dashboard] No results found. Run eval/run_eval.py first.")
        return

    print(f"[dashboard] Loaded results for: {list(all_results.keys())}")

    # Build table
    rows, headers = build_table(all_results)
    best_per_col = find_best_per_metric(rows)

    # Save markdown + CSV tables
    table_dir = results_dir
    save_markdown_table(rows, headers, best_per_col, os.path.join(table_dir, "comparison_table.md"))
    save_csv_table(rows, headers, os.path.join(table_dir, "comparison_table.csv"))

    # Bar charts
    plot_bar_charts(all_results, output_dir)

    # Radar chart
    plot_radar_chart(all_results, output_dir)

    # Summary
    compute_overall_winner(all_results, rows, best_per_col)

    print(f"\n[dashboard] Dashboard complete. Plots in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="AlignBench comparison dashboard")
    parser.add_argument("--results-dir", default="results", help="Directory with result JSON files")
    parser.add_argument("--output-dir", default="results/plots", help="Directory for output plots")
    args = parser.parse_args()

    build_dashboard(results_dir=args.results_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
