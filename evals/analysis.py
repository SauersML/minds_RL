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

# Professional display names for task IDs
TASK_DISPLAY_NAMES = {
    "sanity": "Capability Retention",
    "math": "Confidence and Accuracy", 
    "calibration": "Math Brier Score",
    "ghost": "Latent Encoding",
    "ghost_ood": "Latent Encoding (OOD)",
    "scheming": "Alignment Integrity",
}

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

# Environment display name mapping (matches TASK_DISPLAY_NAMES where applicable)
ENV_DISPLAY_NAMES = {
    "self_prediction": "Self-Prediction",
    "ghost_trace": "Latent Encoding", 
    "entropy_intuition": "Entropy Intuition",
    "gradient_prophet": "Gradient Prophet",
    "default": "Default",
    "all": "Combined (All)",
}

def plot_rewards_over_time(metrics_path: Path, output_path: Path, window: int = 20):
    """Plots rolling average rewards over training steps from metrics.jsonl."""
    if not metrics_path.exists():
        print(f"⚠️  Metrics file not found: {metrics_path}")
        return
    
    # Load metrics
    df = pd.read_json(metrics_path, lines=True)
    if "step" not in df.columns:
        print("⚠️  No 'step' column in metrics file")
        return
    
    # Extract reward columns
    reward_cols = [c for c in df.columns if "/reward/total" in c and "env/" in c]
    if not reward_cols:
        print("⚠️  No reward columns found in metrics")
        return
    
    # Modern color palette  
    COLORS = {
        'bg': '#1a1a2e',
        'card': '#16213e',
        'text': '#e2e8f0',
        'grid': '#2d3748',
    }
    LINE_COLORS = ['#00d26a', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a855f7', '#f97316']
    
    # Set up figure with dark theme
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif',
        'font.size': 10
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['card'])
    
    # Plot each environment's reward
    for i, col in enumerate(sorted(reward_cols)):
        # Extract env name from column like "env/ghost_trace/reward/total"
        parts = col.split("/")
        env_name = parts[1] if len(parts) > 1 else col
        display_name = ENV_DISPLAY_NAMES.get(env_name, env_name)
        
        # Skip 'all' if we have individual envs (to avoid clutter)
        if env_name == "all" and len(reward_cols) > 1:
            continue
        
        # Compute rolling average
        series = df[col].dropna()
        if len(series) < window:
            rolling = series
        else:
            rolling = series.rolling(window=window, min_periods=1).mean()
        
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.plot(df.loc[rolling.index, "step"], rolling, 
                label=display_name, color=color, linewidth=2, alpha=0.9)
    
    # Style
    ax.set_xlabel("Training Step", fontsize=11, fontweight='medium')
    ax.set_ylabel("Reward (Rolling Avg)", fontsize=11, fontweight='medium')
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['grid'])
    
    ax.legend(loc='best', framealpha=0.9, facecolor=COLORS['card'], 
              edgecolor=COLORS['grid'], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"📈 Saved rewards plot to {output_path}")


def plot_kde_distributions(df: pd.DataFrame, output_path: Path, task_filter: str = "reward"):
    """Plot KDE distributions comparing base vs checkpoint for each task."""
    from scipy import stats
    
    # Modern color palette
    COLORS = {
        'bg': '#1a1a2e',
        'card': '#16213e', 
        'base': '#ff6b6b',
        'ckpt': '#4ecdc4',
        'text': '#e2e8f0',
        'muted': '#718096',
        'grid': '#2d3748'
    }
    
    # Filter to reward metric
    df_filtered = df[df["Metric"] == task_filter] if task_filter else df
    tasks = df_filtered["Task"].unique()
    
    if len(tasks) == 0:
        print("⚠️  No data for KDE plot")
        return
    
    # Set up figure
    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(10, 3 * n_tasks), squeeze=False)
    fig.patch.set_facecolor(COLORS['bg'])
    
    for idx, task in enumerate(tasks):
        ax = axes[idx, 0]
        ax.set_facecolor(COLORS['card'])
        
        task_data = df_filtered[df_filtered["Task"] == task]
        base_vals = task_data["base"].values
        ckpt_vals = task_data["ckpt"].values
        
        # Plot KDEs
        if len(base_vals) > 2:
            kde_base = stats.gaussian_kde(base_vals)
            kde_ckpt = stats.gaussian_kde(ckpt_vals)
            
            x_min = min(base_vals.min(), ckpt_vals.min())
            x_max = max(base_vals.max(), ckpt_vals.max())
            x_range = x_max - x_min
            x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)
            
            ax.fill_between(x, kde_base(x), alpha=0.4, color=COLORS['base'], label='Base')
            ax.fill_between(x, kde_ckpt(x), alpha=0.4, color=COLORS['ckpt'], label='Checkpoint')
            ax.plot(x, kde_base(x), color=COLORS['base'], linewidth=2)
            ax.plot(x, kde_ckpt(x), color=COLORS['ckpt'], linewidth=2)
        
        # Style
        display_name = TASK_DISPLAY_NAMES.get(task, task)
        ax.set_ylabel('Density', color=COLORS['text'], fontsize=10)
        ax.set_xlabel('Reward', color=COLORS['text'], fontsize=10)
        ax.set_title(display_name, color=COLORS['text'], fontsize=12, fontweight='bold')
        ax.tick_params(colors=COLORS['text'])
        ax.legend(framealpha=0.9, facecolor=COLORS['card'], edgecolor=COLORS['grid'], fontsize=9)
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"📊 Saved KDE distributions to {output_path}")


def plot_paired_violin(df: pd.DataFrame, output_path: Path, task_filter: str = "reward"):
    """Plot paired violin plots with connecting lines between observations."""
    
    # Modern color palette
    COLORS = {
        'bg': '#1a1a2e',
        'card': '#16213e', 
        'base': '#ff6b6b',
        'ckpt': '#4ecdc4',
        'text': '#e2e8f0',
        'muted': '#718096',
        'grid': '#2d3748',
        'connect': '#a0aec0'
    }
    
    # Filter to reward metric
    df_filtered = df[df["Metric"] == task_filter] if task_filter else df
    tasks = df_filtered["Task"].unique()
    
    if len(tasks) == 0:
        print("⚠️  No data for violin plot")
        return
    
    # Set up figure
    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 6), squeeze=False)
    fig.patch.set_facecolor(COLORS['bg'])
    
    for idx, task in enumerate(tasks):
        ax = axes[0, idx]
        ax.set_facecolor(COLORS['card'])
        
        task_data = df_filtered[df_filtered["Task"] == task]
        base_vals = task_data["base"].values
        ckpt_vals = task_data["ckpt"].values
        
        # Create violin plots
        parts_base = ax.violinplot([base_vals], positions=[0], showmeans=True, showmedians=False)
        parts_ckpt = ax.violinplot([ckpt_vals], positions=[1], showmeans=True, showmedians=False)
        
        # Style violins
        for pc in parts_base['bodies']:
            pc.set_facecolor(COLORS['base'])
            pc.set_alpha(0.6)
            pc.set_edgecolor(COLORS['base'])
        for pc in parts_ckpt['bodies']:
            pc.set_facecolor(COLORS['ckpt'])
            pc.set_alpha(0.6)
            pc.set_edgecolor(COLORS['ckpt'])
        
        # Style mean lines
        for partname in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            if partname in parts_base:
                parts_base[partname].set_color(COLORS['base'])
                parts_base[partname].set_linewidth(2)
            if partname in parts_ckpt:
                parts_ckpt[partname].set_color(COLORS['ckpt'])
                parts_ckpt[partname].set_linewidth(2)
        
        # Draw connecting lines between paired observations
        for i in range(len(base_vals)):
            # Add small jitter for visibility
            jitter = np.random.uniform(-0.05, 0.05)
            ax.plot([0 + jitter, 1 + jitter], [base_vals[i], ckpt_vals[i]], 
                    color=COLORS['connect'], alpha=0.3, linewidth=0.8, zorder=1)
        
        # Overlay scatter points
        jitter_base = np.random.uniform(-0.08, 0.08, len(base_vals))
        jitter_ckpt = np.random.uniform(-0.08, 0.08, len(ckpt_vals))
        ax.scatter(np.zeros(len(base_vals)) + jitter_base, base_vals, 
                   color=COLORS['base'], alpha=0.7, s=15, zorder=2, edgecolors='white', linewidths=0.5)
        ax.scatter(np.ones(len(ckpt_vals)) + jitter_ckpt, ckpt_vals, 
                   color=COLORS['ckpt'], alpha=0.7, s=15, zorder=2, edgecolors='white', linewidths=0.5)
        
        # Style
        display_name = TASK_DISPLAY_NAMES.get(task, task)
        ax.set_title(display_name, color=COLORS['text'], fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Base', 'Checkpoint'], color=COLORS['text'], fontsize=10)
        ax.set_ylabel('Reward', color=COLORS['text'], fontsize=10)
        ax.tick_params(colors=COLORS['text'])
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
        
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"🎻 Saved paired violin plot to {output_path}")

def plot_results(stats: List[EvalStat], output_path: Path):
    """Generates a professional horizontal bar chart of Deltas with Confidence Intervals."""
    if not stats: return

    # Filter to only show the primary "reward" metric (skip wrapped, format_reward, etc.)
    stats_filtered = [s for s in stats if s.metric == "reward"]
    if not stats_filtered:
        # Fallback: if no "reward" metric, use all
        stats_filtered = stats
    
    # Sort by Delta magnitude for readability
    stats_sorted = sorted(stats_filtered, key=lambda x: x.delta, reverse=True)
    
    # Just show task names, no metric in parentheses
    tasks = [TASK_DISPLAY_NAMES.get(s.task, s.task) for s in stats_sorted]
    deltas = [s.delta for s in stats_sorted]
    errors = [
        [s.delta - s.ci_low for s in stats_sorted],  # Lower error
        [s.ci_high - s.delta for s in stats_sorted]  # Upper error
    ]
    
    # Modern color palette
    COLORS = {
        'bg': '#1a1a2e',
        'card': '#16213e', 
        'green': '#00d26a',
        'red': '#ff6b6b',
        'grey': '#4a5568',
        'text': '#e2e8f0',
        'muted': '#718096',
        'grid': '#2d3748'
    }
    
    # Color coding for bars
    colors = []
    for s in stats_sorted:
        if not s.is_significant: colors.append(COLORS['grey'])
        elif s.delta > 0:        colors.append(COLORS['green'])
        else:                    colors.append(COLORS['red'])

    # Set up figure with dark theme
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif',
        'font.size': 10
    })
    
    fig, ax = plt.subplots(figsize=(12, max(5, len(stats_sorted) * 0.7)))
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_facecolor(COLORS['card'])
    
    y_pos = np.arange(len(stats_sorted))
    bars = ax.barh(y_pos, deltas, xerr=errors, align='center', color=colors, 
                   alpha=0.9, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'alpha': 0.7})
    
    # Style axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=11)
    ax.axvline(0, color=COLORS['text'], linewidth=1.5, linestyle='-', alpha=0.3)
    ax.set_xlabel('Δ Reward (Checkpoint − Base)', fontsize=12, fontweight='medium')
    
    # Add subtle grid
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['grid'])
    
    # Annotate bars with p-values
    x_min = min(s.ci_low for s in stats_sorted)
    x_max = max(s.ci_high for s in stats_sorted)
    x_range = max(x_max - x_min, 0.1)  # Avoid division by zero
    padding = x_range * 0.35
    
    for i, s in enumerate(stats_sorted):
        # Format p-value with significance stars
        if s.p_value < 0.001:
            p_str = "p<.001 ★★★"
        elif s.p_value < 0.01:
            p_str = f"p={s.p_value:.3f} ★★"
        elif s.p_value < 0.05:
            p_str = f"p={s.p_value:.3f} ★"
        else:
            p_str = f"p={s.p_value:.2f}"
        
        # Position label to the right of the error bar
        x_pos = s.ci_high + x_range * 0.03
        text_color = COLORS['green'] if s.is_significant else COLORS['muted']
        ax.annotate(p_str, (x_pos, i), va='center', fontsize=9, 
                    color=text_color, fontweight='medium' if s.is_significant else 'normal')
    
    ax.set_xlim(x_min - padding * 0.3, x_max + padding)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
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
        is_sig = bool(p_val < ALPHA)

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
            display_name = TASK_DISPLAY_NAMES.get(s.task, s.task)
            f.write(f"| {display_name} | {s.metric} | {s.base_mean:.3f} | {s.ckpt_mean:.3f} | "
                    f"{bold}{s.delta:+.3f}{bold} <br> [{s.ci_low:.3f}, {s.ci_high:.3f}] | "
                    f"{s.p_value:.4f} |\n")

    # 3. Generate Plots
    plot_results(stats_list, args.output_dir / "analysis_plot.png")
    
    # 4. Generate KDE distribution plots
    plot_kde_distributions(df, args.output_dir / "kde_distributions.png")
    
    # 5. Generate paired violin plots
    plot_paired_violin(df, args.output_dir / "paired_violin.png")
    
    # 6. Auto-find metrics.jsonl in outputs/ for rewards-over-time plot
    script_dir = Path(__file__).parent.parent  # Go up from evals/ to repo root
    metrics_files = list(script_dir.glob("outputs/**/metrics.jsonl"))
    if metrics_files:
        # Use the most recently modified one
        metrics_path = max(metrics_files, key=lambda p: p.stat().st_mtime)
        plot_rewards_over_time(metrics_path, args.output_dir / "rewards_over_time.png")
    
    print(f"✅ Analysis complete. Artifacts saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
