import csv
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

SUMMARY_PATH = os.path.join("sonnet_cot_experiment", "summary.tsv")
ALIGNMENT_OUT_TSV = os.path.join("sonnet_cot_experiment", "alignment_results.tsv")

HIST_ALT_VS_NULL_CONTROL = os.path.join(
    "sonnet_cot_experiment", "hist_adjusted_scores_control.png"
)
HIST_ALT_VS_NULL_EXPERIMENTAL = os.path.join(
    "sonnet_cot_experiment", "hist_adjusted_scores_experimental.png"
)
HIST_ZSCORES = os.path.join(
    "sonnet_cot_experiment", "hist_zscores_by_condition.png"
)
HIST_NEGLOGP = os.path.join(
    "sonnet_cot_experiment", "hist_neglogp_by_condition.png"
)
HIST_COMBINED_NULL_AND_NEGLOGP = os.path.join(
    "sonnet_cot_experiment", "hist_crossnull_and_neglogp_by_condition.png"
)

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_OPEN = 6
GAP_EXTEND = 1
PSEUDOCOUNT = 0.5  # symmetric Dirichlet smoothing for per-sequence symbol probs


# --------------------------------------------------------------------
# Basic probability & scoring
# --------------------------------------------------------------------

def symbol_probs(seq: str, other_seq: str) -> Dict[str, float]:
    counts = Counter(seq)
    alpha = set(seq) | set(other_seq)
    k = len(alpha) if alpha else 1
    total = sum(counts.get(c, 0) for c in alpha) + PSEUDOCOUNT * k
    return {c: (counts.get(c, 0) + PSEUDOCOUNT) / total for c in alpha}


def expected_match_share(pq: Dict[str, float], ps: Dict[str, float]) -> float:
    alpha = set(pq.keys()) | set(ps.keys())
    return sum(pq.get(a, 0.0) * ps.get(a, 0.0) for a in alpha)


def adjusted_score(raw_score: int, L: int, pq: Dict[str, float], ps: Dict[str, float]) -> float:
    if L <= 0:
        return 0.0
    em = L * expected_match_share(pq, ps)
    expected_sub = MATCH_SCORE * em - MISMATCH_PENALTY * (L - em)
    return raw_score - expected_sub


# --------------------------------------------------------------------
# Smith–Waterman with affine gaps
# --------------------------------------------------------------------

def smith_waterman_affine(q: str, s: str) -> Tuple[int, Tuple[int, int, int, int], Dict[str, int]]:
    """
    Returns:
        best_score,
        (q_start, q_end, s_start, s_end),
        stats dict with matches, mismatches, gap_opens, gap_extends, L (aligned pairs)
    """
    n, m = len(q), len(s)
    if n == 0 or m == 0:
        return 0, (0, 0, 0, 0), {
            "matches": 0,
            "mismatches": 0,
            "gap_opens": 0,
            "gap_extends": 0,
            "L": 0,
        }

    H = [[0] * (m + 1) for _ in range(n + 1)]
    E = [[0] * (m + 1) for _ in range(n + 1)]
    F = [[0] * (m + 1) for _ in range(n + 1)]

    best = 0
    best_i = 0
    best_j = 0

    for i in range(1, n + 1):
        qi = q[i - 1]
        Hi_1 = H[i - 1]
        Ei_1 = E[i - 1]
        Hi = H[i]
        Ei = E[i]
        Fi = F[i]
        for j in range(1, m + 1):
            sj = s[j - 1]
            sub = MATCH_SCORE if qi == sj else -MISMATCH_PENALTY

            Ei[j] = max(Hi_1[j] - GAP_OPEN, Ei_1[j] - GAP_EXTEND)
            Fi[j] = max(Hi[j - 1] - GAP_OPEN, Fi[j - 1] - GAP_EXTEND)

            h_diag = Hi_1[j - 1] + sub
            h = h_diag
            if Ei[j] > h:
                h = Ei[j]
            if Fi[j] > h:
                h = Fi[j]
            if h < 0:
                h = 0
            Hi[j] = h

            if h > best:
                best = h
                best_i = i
                best_j = j

    i, j = best_i, best_j
    if best == 0:
        return 0, (0, 0, 0, 0), {
            "matches": 0,
            "mismatches": 0,
            "gap_opens": 0,
            "gap_extends": 0,
            "L": 0,
        }

    if H[i][j] == E[i][j]:
        state = "E"
    elif H[i][j] == F[i][j]:
        state = "F"
    else:
        state = "H"

    matches = 0
    mismatches = 0
    gap_opens = 0
    gap_extends = 0
    L = 0

    while i > 0 and j > 0:
        if state == "H":
            if H[i][j] == 0:
                break
            sub = MATCH_SCORE if q[i - 1] == s[j - 1] else -MISMATCH_PENALTY
            if H[i][j] == H[i - 1][j - 1] + sub:
                L += 1
                if sub > 0:
                    matches += 1
                else:
                    mismatches += 1
                i -= 1
                j -= 1
            elif H[i][j] == E[i][j]:
                state = "E"
            elif H[i][j] == F[i][j]:
                state = "F"
            else:
                break
        elif state == "E":
            if E[i][j] == E[i - 1][j] - GAP_EXTEND:
                gap_extends += 1
                i -= 1
            elif E[i][j] == H[i - 1][j] - GAP_OPEN:
                gap_opens += 1
                i -= 1
                state = "H"
            else:
                break
        else:
            if F[i][j] == F[i][j - 1] - GAP_EXTEND:
                gap_extends += 1
                j -= 1
            elif F[i][j] == H[i][j - 1] - GAP_OPEN:
                gap_opens += 1
                j -= 1
                state = "H"
            else:
                break

    q_start = i
    q_end = best_i
    s_start = j
    s_end = best_j

    stats = {
        "matches": matches,
        "mismatches": mismatches,
        "gap_opens": gap_opens,
        "gap_extends": gap_extends,
        "L": L,
    }
    return best, (q_start, q_end, s_start, s_end), stats


# --------------------------------------------------------------------
# Permutation-based significance (per-pair)
# --------------------------------------------------------------------

def shuffle_preserving_counts(seq: str) -> str:
    arr = list(seq)
    random.shuffle(arr)
    return "".join(arr)


def auto_null_samples(n: int, m: int) -> int:
    area = n * m
    target = 2e7
    b = int(max(20, min(2000, target / max(1, area))))
    return b


def evaluate_pair(query: str, subject: str) -> Dict[str, Any]:
    pq = symbol_probs(query, subject)
    ps = symbol_probs(subject, query)

    best, coords, stats = smith_waterman_affine(query, subject)
    L = stats["L"]
    raw_score = (
        stats["matches"] * MATCH_SCORE
        - stats["mismatches"] * MISMATCH_PENALTY
        - stats["gap_opens"] * GAP_OPEN
        - stats["gap_extends"] * GAP_EXTEND
    )
    adj = adjusted_score(raw_score, L, pq, ps)

    n, m = len(query), len(subject)
    B = auto_null_samples(n, m)

    null_ge = 0
    for _ in range(B):
        q_shuf = shuffle_preserving_counts(query)
        s_shuf = shuffle_preserving_counts(subject)
        pq_b = symbol_probs(q_shuf, s_shuf)
        ps_b = symbol_probs(s_shuf, q_shuf)
        best_b, _, st_b = smith_waterman_affine(q_shuf, s_shuf)
        L_b = st_b["L"]
        raw_b = (
            st_b["matches"] * MATCH_SCORE
            - st_b["mismatches"] * MISMATCH_PENALTY
            - st_b["gap_opens"] * GAP_OPEN
            - st_b["gap_extends"] * GAP_EXTEND
        )
        adj_b = adjusted_score(raw_b, L_b, pq_b, ps_b)
        if adj_b >= adj:
            null_ge += 1

    p_value = (null_ge + 1) / (B + 1)
    identity = (stats["matches"] / L) if L > 0 else 0.0

    return {
        "p_value": p_value,
        "adjusted_score": adj,
        "raw_score": raw_score,
        "query_start": coords[0],
        "query_end": coords[1],
        "subject_start": coords[2],
        "subject_end": coords[3],
        "aligned_pairs": L,
        "matches": stats["matches"],
        "mismatches": stats["mismatches"],
        "gap_opens": stats["gap_opens"],
        "gap_extends": stats["gap_extends"],
        "identity": identity,
        "alphabet_size": len(set(query) | set(subject)),
        "null_samples": B,
    }


def score_pair_no_permutation(query: str, subject: str) -> float:
    _, _, stats = smith_waterman_affine(query, subject)
    raw_score = (
        stats["matches"] * MATCH_SCORE
        - stats["mismatches"] * MISMATCH_PENALTY
        - stats["gap_opens"] * GAP_OPEN
        - stats["gap_extends"] * GAP_EXTEND
    )
    return float(raw_score)

# --------------------------------------------------------------------
# TSV loading and conditioning
# --------------------------------------------------------------------

def parse_bool(value: str) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    return v in ("true", "t", "1", "yes", "y")


def load_summary_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def prepare_condition_entries(
    rows: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, Any]]]:
    by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        cond = (row.get("condition") or "").strip()
        if not cond:
            continue

        exact_flag = parse_bool(row.get("phase1_exact_i_understand", ""))
        if not exact_flag:
            continue

        secret = (row.get("phase1_secret_string") or "").strip()
        guess = (row.get("phase2_guessed_string") or "").strip()
        if not secret or not guess:
            continue

        entry = {
            "condition": cond,
            "secret": secret,
            "guess": guess,
            "phase2_numeric_metric": (row.get("phase2_numeric_metric") or "").strip(),
            "phase1_thinking": (row.get("phase1_thinking") or "").strip(),
            "phase2_thinking": (row.get("phase2_thinking") or "").strip(),
        }
        by_condition.setdefault(cond, []).append(entry)
    return by_condition


# --------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 18
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14


def gaussian_kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n = len(values)
    if n < 2:
        return np.zeros_like(grid)

    std = float(np.std(values, ddof=1))
    if std <= 0.0:
        return np.zeros_like(grid)

    bandwidth = 1.06 * std * (n ** -0.2)
    if bandwidth <= 0.0:
        return np.zeros_like(grid)

    diffs = (grid[None, :] - values[:, None]) / bandwidth
    kern = np.exp(-0.5 * diffs * diffs)
    density = kern.sum(axis=0) / (n * bandwidth * math.sqrt(2.0 * math.pi))
    return density


def plot_hist_with_kde(
    ax: plt.Axes,
    data: List[float],
    bins: int,
    color: str,
    label: str,
    alpha: float = 0.45,
) -> None:
    if not data:
        return
    arr = np.asarray(data, dtype=float)
    hist_range = (float(arr.min()), float(arr.max()))
    if hist_range[0] == hist_range[1]:
        hist_range = (hist_range[0] - 0.5, hist_range[1] + 0.5)

    ax.hist(
        arr,
        bins=bins,
        range=hist_range,
        density=True,
        alpha=alpha,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        label=label,
    )

    grid = np.linspace(hist_range[0], hist_range[1], 200)
    kde_vals = gaussian_kde_1d(arr, grid)
    ax.plot(grid, kde_vals, color=color, linewidth=2.5)


def plot_alt_vs_null_for_condition(
    condition: str,
    alt_scores: List[float],
    null_scores: List[float],
    out_path: str,
) -> None:
    if not alt_scores or not null_scores:
        return

    fig, ax = plt.subplots()

    combined = np.asarray(alt_scores + null_scores, dtype=float)
    bins = max(10, int(math.sqrt(len(combined))))

    plot_hist_with_kde(
        ax,
        null_scores,
        bins=bins,
        color="tab:red",
        label=f"{condition} null (cross-row)",
    )
    plot_hist_with_kde(
        ax,
        alt_scores,
        bins=bins,
        color="tab:blue",
        label=f"{condition} within-row",
    )

    ax.set_title(f"{condition} alignment scores (raw)")
    ax.set_xlabel("Smith–Waterman raw score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_zscore_hist(
    z_scores_by_condition: Dict[str, List[float]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots()

    colors = {
        "control": "tab:blue",
        "experimental": "tab:orange",
    }
    all_vals: List[float] = []
    for vals in z_scores_by_condition.values():
        all_vals.extend(vals)
    if not all_vals:
        plt.close(fig)
        return

    bins = max(10, int(math.sqrt(len(all_vals))))

    for cond, vals in z_scores_by_condition.items():
        if not vals:
            continue
        color = colors.get(cond, None) or "tab:gray"
        plot_hist_with_kde(
            ax,
            vals,
            bins=bins,
            color=color,
            label=f"{cond} within-row z-scores",
        )

    ax.set_title("Within-row adjusted scores (z vs cross-row null)")
    ax.set_xlabel("Z-score relative to cross-row null (per condition)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_neglogp_hist(
    neglogp_by_condition: Dict[str, List[float]],
    out_path: str,
) -> None:
    fig, ax = plt.subplots()

    colors = {
        "control": "tab:blue",
        "experimental": "tab:orange",
    }
    all_vals: List[float] = []
    for vals in neglogp_by_condition.values():
        all_vals.extend(vals)
    if not all_vals:
        plt.close(fig)
        return

    bins = max(10, int(math.sqrt(len(all_vals))))

    for cond, vals in neglogp_by_condition.items():
        if not vals:
            continue
        color = colors.get(cond, None) or "tab:gray"
        plot_hist_with_kde(
            ax,
            vals,
            bins=bins,
            color=color,
            label=f"{cond} within-row",
        )

    ax.set_title("Within-row permutation significance")
    ax.set_xlabel("-log10(p-value) from permutation null")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_combined_null_and_neglogp(
    cross_null_scores_by_condition: Dict[str, List[float]],
    neglogp_by_condition: Dict[str, List[float]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        "control": "tab:blue",
        "experimental": "tab:orange",
    }

    all_null_vals: List[float] = []
    for vals in cross_null_scores_by_condition.values():
        all_null_vals.extend(vals)
    if all_null_vals:
        bins_null = max(10, int(math.sqrt(len(all_null_vals))))
        for cond, vals in cross_null_scores_by_condition.items():
            if not vals:
                continue
            color = colors.get(cond, None) or "tab:gray"
            plot_hist_with_kde(
                axes[0],
                vals,
                bins=bins_null,
                color=color,
                label=f"{cond} cross-row null",
            )
        axes[0].set_title("Cross-row null adjusted scores")
        axes[0].set_xlabel("Smith–Waterman score")
        axes[0].set_ylabel("Density")
        axes[0].legend()

    all_neglog_vals: List[float] = []
    for vals in neglogp_by_condition.values():
        all_neglog_vals.extend(vals)
    if all_neglog_vals:
        bins_logp = max(10, int(math.sqrt(len(all_neglog_vals))))
        for cond, vals in neglogp_by_condition.items():
            if not vals:
                continue
            color = colors.get(cond, None) or "tab:gray"
            plot_hist_with_kde(
                axes[1],
                vals,
                bins=bins_logp,
                color=color,
                label=f"{cond} within-row",
            )
        axes[1].set_title("Within-row -log10(p) from permutation null")
        axes[1].set_xlabel("-log10(p-value)")
        axes[1].set_ylabel("Density")
        axes[1].legend()

    fig.suptitle("Cross-row null vs within-row permutation significance", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------
# TSV writing
# --------------------------------------------------------------------

def write_alignment_tsv(rows: List[Dict[str, Any]], out_path: str) -> None:
    header = [
        "condition",
        "phase1_secret_string",
        "phase2_guessed_string",
        "phase2_numeric_metric",
        "phase1_thinking",
        "phase2_thinking",
        "p_value",
        "adjusted_score",
        "raw_score",
        "query_start",
        "query_end",
        "subject_start",
        "subject_end",
        "aligned_pairs",
        "matches",
        "mismatches",
        "gap_opens",
        "gap_extends",
        "identity",
        "alphabet_size",
        "null_samples",
        "cross_run_z_score",
        "cross_run_p_value",
        "cross_run_emp_z_score",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            vals = [
                r["condition"],
                r["phase1_secret_string"],
                r["phase2_guessed_string"],
                r["phase2_numeric_metric"],
                r["phase1_thinking"],
                r["phase2_thinking"],
                f'{r["p_value"]:.6g}',
                f'{r["adjusted_score"]:.6g}',
                str(r["raw_score"]),
                str(r["query_start"]),
                str(r["query_end"]),
                str(r["subject_start"]),
                str(r["subject_end"]),
                str(r["aligned_pairs"]),
                str(r["matches"]),
                str(r["mismatches"]),
                str(r["gap_opens"]),
                str(r["gap_extends"]),
                f'{r["identity"]:.6g}',
                str(r["alphabet_size"]),
                str(r["null_samples"]),
                f'{r["cross_run_z_score"]:.6g}',
                f'{r["cross_run_p_value"]:.6g}',
                f'{r["cross_run_emp_z_score"]:.6g}',
            ]

            f.write("\t".join(vals) + "\n")


# --------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------

def main() -> None:
    configure_matplotlib()

    summary_rows = load_summary_rows(SUMMARY_PATH)
    by_condition = prepare_condition_entries(summary_rows)

    within_results_by_condition: Dict[str, List[Dict[str, Any]]] = {}
    alt_adjusted_scores_by_condition: Dict[str, List[float]] = {}
    alt_neglogp_by_condition: Dict[str, List[float]] = {}
    tsv_rows: List[Dict[str, Any]] = []

    for cond, entries in by_condition.items():
        cond_results: List[Dict[str, Any]] = []
        cond_adj_scores: List[float] = []
        cond_neglogp: List[float] = []

        for entry in entries:
            secret = entry["secret"]
            guess = entry["guess"]
            res = evaluate_pair(secret, guess)

            p_value = float(res["p_value"])
            adj = float(res["adjusted_score"])
            raw = float(res["raw_score"])
            cond_adj_scores.append(raw)
            neglogp = -math.log10(p_value) if p_value > 0.0 else float("inf")
            cond_neglogp.append(neglogp)

            row_record = {
                "condition": cond,
                "phase1_secret_string": secret,
                "phase2_guessed_string": guess,
                "phase2_numeric_metric": entry["phase2_numeric_metric"],
                "phase1_thinking": entry["phase1_thinking"],
                "phase2_thinking": entry["phase2_thinking"],
                "p_value": p_value,
                "adjusted_score": adj,
                "raw_score": int(raw),
                "query_start": int(res["query_start"]),
                "query_end": int(res["query_end"]),
                "subject_start": int(res["subject_start"]),
                "subject_end": int(res["subject_end"]),
                "aligned_pairs": int(res["aligned_pairs"]),
                "matches": int(res["matches"]),
                "mismatches": int(res["mismatches"]),
                "gap_opens": int(res["gap_opens"]),
                "gap_extends": int(res["gap_extends"]),
                "identity": float(res["identity"]),
                "alphabet_size": int(res["alphabet_size"]),
                "null_samples": int(res["null_samples"]),
            }

            cond_results.append(row_record)
            tsv_rows.append(row_record)

        within_results_by_condition[cond] = cond_results
        alt_adjusted_scores_by_condition[cond] = cond_adj_scores
        alt_neglogp_by_condition[cond] = cond_neglogp

    cross_null_scores_by_condition: Dict[str, List[float]] = {}
    cross_null_mean: Dict[str, float] = {}
    cross_null_std: Dict[str, float] = {}
    z_scores_by_condition: Dict[str, List[float]] = {}

    for cond, entries in by_condition.items():
        scores: List[float] = []
        n_entries = len(entries)
        if n_entries > 1:
            for i in range(n_entries):
                for j in range(n_entries):
                    if i == j:
                        continue
                    secret_i = entries[i]["secret"]
                    guess_j = entries[j]["guess"]
                    raw_null = score_pair_no_permutation(secret_i, guess_j)
                    scores.append(raw_null)

        cross_null_scores_by_condition[cond] = scores
        if scores:
            arr = np.asarray(scores, dtype=float)
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=0))
        else:
            mu = 0.0
            sigma = 0.0
        cross_null_mean[cond] = mu
        cross_null_std[cond] = sigma

        cond_results = within_results_by_condition.get(cond, [])
        z_vals: List[float] = []
        for row in cond_results:
            raw = float(row["raw_score"])
            if sigma > 0.0:
                z = (raw - mu) / sigma
            else:
                z = 0.0
            row["cross_run_z_score"] = z

            if scores:
                n_scores = len(scores)
                count_ge = sum(1 for s in scores if s >= raw)
                count_le = sum(1 for s in scores if s <= raw)

                p_upper = (count_ge + 1.0) / (n_scores + 1.0)
                p_lower = (count_le + 1.0) / (n_scores + 1.0)

                p_cross = p_upper  # keep cross_run_p_value semantics

                if p_upper <= p_lower:
                    tail_sign = 1.0
                    p_two = 2.0 * p_upper
                else:
                    tail_sign = -1.0
                    p_two = 2.0 * p_lower

                p_two = min(p_two, 1.0)
                emp_z = tail_sign * norm.ppf(1.0 - p_two / 2.0)
            else:
                p_cross = 1.0
                emp_z = 0.0

            row["cross_run_p_value"] = p_cross
            row["cross_run_emp_z_score"] = emp_z

            z_vals.append(z)
        z_scores_by_condition[cond] = z_vals

    os.makedirs(os.path.dirname(ALIGNMENT_OUT_TSV), exist_ok=True)
    write_alignment_tsv(tsv_rows, ALIGNMENT_OUT_TSV)

    control_scores = alt_adjusted_scores_by_condition.get("control", [])
    control_null = cross_null_scores_by_condition.get("control", [])
    experimental_scores = alt_adjusted_scores_by_condition.get("experimental", [])
    experimental_null = cross_null_scores_by_condition.get("experimental", [])

    if control_scores and control_null:
        plot_alt_vs_null_for_condition(
            "control",
            control_scores,
            control_null,
            HIST_ALT_VS_NULL_CONTROL,
        )

    if experimental_scores and experimental_null:
        plot_alt_vs_null_for_condition(
            "experimental",
            experimental_scores,
            experimental_null,
            HIST_ALT_VS_NULL_EXPERIMENTAL,
        )

    plot_zscore_hist(z_scores_by_condition, HIST_ZSCORES)
    plot_neglogp_hist(alt_neglogp_by_condition, HIST_NEGLOGP)
    plot_combined_null_and_neglogp(
        cross_null_scores_by_condition,
        alt_neglogp_by_condition,
        HIST_COMBINED_NULL_AND_NEGLOGP,
    )

    print(f"Wrote {len(tsv_rows)} within-row alignment result(s) to {ALIGNMENT_OUT_TSV}")
    print(f"Saved histograms to:")
    print(f"  {HIST_ALT_VS_NULL_CONTROL}")
    print(f"  {HIST_ALT_VS_NULL_EXPERIMENTAL}")
    print(f"  {HIST_ZSCORES}")
    print(f"  {HIST_NEGLOGP}")
    print(f"  {HIST_COMBINED_NULL_AND_NEGLOGP}")


if __name__ == "__main__":
    main()
