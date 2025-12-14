import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
N_PERMS = 50_000   # Permutations for p-value calculation
N_BOOTS = 10_000   # Bootstrap iterations for CI
ALPHA = 0.05       # Significance threshold
CI_LEVEL = 0.95    # Confidence Interval level

@dataclass
class EvalStat:
    task: str
    metric: str
    n: int
    base_mean: float
    ckpt_mean: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    is_significant: bool

def load_paired_data(tsv_path: Path) -> pd.DataFrame:
    """Loads TSV, cleans numeric data, and pivots to paired (base vs ckpt) format."""
    try:
        df = pd.read_csv(tsv_path, sep="\t")
    except FileNotFoundError:
        print(f"❌ File not found: {tsv_path}")
        sys.exit(1)

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df.dropna(subset=["Value"], inplace=True)

    # Average rollouts if multiple per example, then pivot
    pivot = df.pivot_table(
        index=["Task", "Metric", "Example_ID"],
        columns="Model_Version",
        values="Value",
        aggfunc="mean"
    ).reset_index()

    # Drop unpaired examples
    pivot.dropna(subset=["base", "ckpt"], inplace=True)
    pivot["diff"] = pivot["ckpt"] - pivot["base"]
    
    return pivot

def compute_stats(diffs: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes p-value (approx. exact permutation test) and 95% CI (bootstrap).
    Returns: (p_value, ci_low, ci_high)
    """
    n = len(diffs)
    if n < 2: 
        return 1.0, 0.0, 0.0
    
    obs_mean = np.mean(diffs)

    # 1. Vectorized Paired Permutation Test
    # Matrix of random signs (-1 or 1): shape (N_PERMS, n_samples)
    signs = np.random.choice([-1, 1], size=(N_PERMS, n))
    perm_means = np.sum(diffs * signs, axis=1) / n
    p_val = (np.sum(np.abs(perm_means) >= np.abs(obs_mean)) + 1) / (N_PERMS + 1)

    # 2. Vectorized Bootstrap CI
    indices = np.random.randint(0, n, size=(N_BOOTS, n))
    boot_means = np.mean(diffs[indices], axis=1)
    ci_low = np.percentile(boot_means, (1 - CI_LEVEL) / 2 * 100)
    ci_high = np.percentile(boot_means, (1 + CI_LEVEL) / 2 * 100)

    return p_val, ci_low, ci_high

def plot_results(stats: List[EvalStat], output_path: Path):
    """Generates a horizontal bar chart of Deltas with Confidence Intervals."""
    if not stats: return

    # Sort by Delta magnitude for readability
    stats_sorted = sorted(stats, key=lambda x: x.delta, reverse=True)
    
    tasks = [f"{s.task}\n({s.metric})" for s in stats_sorted]
    deltas = [s.delta for s in stats_sorted]
    errors = [
        [s.delta - s.ci_low for s in stats_sorted],  # Lower error
        [s.ci_high - s.delta for s in stats_sorted]  # Upper error
    ]
    
    # Color coding
    colors = []
    for s in stats_sorted:
        if not s.is_significant: colors.append("#95a5a6")  # Grey (NS)
        elif s.delta > 0:        colors.append("#27ae60")  # Green (Good)
        else:                    colors.append("#c0392b")  # Red (Bad)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, max(4, len(stats) * 0.6)))
    
    y_pos = np.arange(len(stats))
    ax.barh(y_pos, deltas, xerr=errors, align='center', color=colors, alpha=0.8, capsize=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Delta (Checkpoint - Base)')
    ax.set_title(f'Performance Delta with {int(CI_LEVEL*100)}% CI')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evals.tsv", type=Path)
    parser.add_argument("--output_dir", default="results", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Processing {args.input}...")
    df = load_paired_data(args.input)
    
    stats_list = []
    
    # Console Table Header
    row_fmt = "{:<25} | {:<12} | {:>4} | {:>7} | {:>7} | {:>7} | {:>9} | {:<15}"
    print("-" * 105)
    print(row_fmt.format("TASK", "METRIC", "N", "BASE", "CKPT", "DELTA", "P-VAL", "95% CI"))
    print("-" * 105)

    for (task, metric), group in df.groupby(["Task", "Metric"]):
        diffs = group["diff"].values
        
        # Optimization: Skip stats if identical
        if np.allclose(diffs, 0):
            p_val, low, high = 1.0, 0.0, 0.0
        else:
            p_val, low, high = compute_stats(diffs)

        base_mean = group["base"].mean()
        ckpt_mean = group["ckpt"].mean()
        delta = ckpt_mean - base_mean
        is_sig = p_val < ALPHA

        stat = EvalStat(
            task=task, metric=metric, n=len(diffs),
            base_mean=base_mean, ckpt_mean=ckpt_mean, delta=delta,
            p_value=p_val, ci_low=low, ci_high=high, is_significant=is_sig
        )
        stats_list.append(stat)

        # Console Output
        sig_icon = " "
        if is_sig: sig_icon = "🟢" if delta > 0 else "🔴"
        
        p_str = "<.0001" if p_val < 0.0001 else f"{p_val:.4f}"
        print(row_fmt.format(
            task[:25], metric[:12], len(diffs),
            f"{base_mean:.3f}", f"{ckpt_mean:.3f}", f"{delta:+.3f}",
            p_str, f"[{low:+.2f}, {high:+.2f}] {sig_icon}"
        ))

    print("-" * 105)

    # 1. Save JSON
    with open(args.output_dir / "analysis_stats.json", "w") as f:
        json.dump([asdict(s) for s in stats_list], f, indent=2)

    # 2. Save Markdown
    with open(args.output_dir / "analysis_summary.md", "w") as f:
        f.write("# 🧪 Evaluation Report\n\n")
        f.write(f"| Task | Metric | Base | Ckpt | Delta (95% CI) | P-Value |\n|---|---|---|---|---|---|\n")
        for s in stats_list:
            bold = "**" if s.is_significant else ""
            f.write(f"| {s.task} | {s.metric} | {s.base_mean:.3f} | {s.ckpt_mean:.3f} | "
                    f"{bold}{s.delta:+.3f}{bold} <br> [{s.ci_low:.3f}, {s.ci_high:.3f}] | "
                    f"{s.p_value:.4f} |\n")

    # 3. Generate Plot
    plot_results(stats_list, args.output_dir / "analysis_plot.png")
    print(f"✅ Analysis complete. Artifacts saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
