import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

# Input files from Phase 1
RUNS_TSV = os.path.join("sonnet_cot_experiment", "runs.tsv")
SCORE_MATRIX_TSV = os.path.join("sonnet_cot_experiment", "score_matrix.tsv")

# Output files from this script
PER_RUN_LEAK_TSV = os.path.join("sonnet_cot_experiment", "per_run_leak_scores.tsv")
PERM_SUMMARY_TSV = os.path.join("sonnet_cot_experiment", "permutation_summary.tsv")

# Permutation settings
DERANGEMENT_PERMS = 1_000_000
RNG_SEED_DERANGE = 12345

LABEL_PERMS = 100_000
RNG_SEED_LABEL = 98765

def load_runs(runs_path: str) -> pd.DataFrame:
    df = pd.read_csv(runs_path, sep="\t")
    df = df.sort_values("run_index").reset_index(drop=True)
    return df


def load_score_matrix(score_path: str, n_runs: int) -> np.ndarray:
    df = pd.read_csv(score_path, sep="\t")
    S = np.full((n_runs, n_runs), np.nan, dtype=float)

    for row in df.itertuples(index=False):
        i = int(row.secret_run_index)
        j = int(row.guess_run_index)
        S[i, j] = float(row.score)

    return S


def build_baselines(S: np.ndarray, conditions: np.ndarray) -> list:
    n = S.shape[0]
    baselines = []
    for i in range(n):
        row = S[i, :]
        mask = np.isfinite(row)
        if i < mask.size:
            mask[i] = False

        cond_i = conditions[i]
        same_cond = (conditions == cond_i)

        mask = mask & same_cond
        baseline_i = row[mask]
        baselines.append(baseline_i)
    return baselines

def precompute_pairwise_pznl(S: np.ndarray, baselines: list) -> tuple:
    n = S.shape[0]
    P = np.full((n, n), np.nan, dtype=float)
    Z = np.full((n, n), np.nan, dtype=float)
    NL = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        F_i = baselines[i]
        m = F_i.size
        if m == 0:
            continue
        row = S[i, :]
        for j in range(n):
            s_ij = float(row[j])
            if not np.isfinite(s_ij):
                continue
            count_ge = int(np.sum(F_i >= s_ij))
            p = (count_ge + 1.0) / (m + 1.0)
            P[i, j] = p
            Z[i, j] = float(stats.norm.isf(p))
            NL[i, j] = float(-np.log10(p))
    return P, Z, NL


def compute_per_run_observed(
    S: np.ndarray,
    baselines: list,
    P: np.ndarray,
    Z: np.ndarray,
    NL: np.ndarray,
) -> dict:
    n = S.shape[0]
    diag_scores = np.empty(n, dtype=float)
    p_obs = np.empty(n, dtype=float)
    p_obs_low = np.empty(n, dtype=float)
    p_obs_two = np.empty(n, dtype=float)
    z_obs = np.empty(n, dtype=float)
    neglog10p_obs = np.empty(n, dtype=float)
    baseline_sizes = np.empty(n, dtype=int)

    for i in range(n):
        diag_scores[i] = float(S[i, i])
        F_i = baselines[i]
        baseline_sizes[i] = int(F_i.size)

        p = float(P[i, i])
        z = float(Z[i, i])
        nl = float(NL[i, i])

        if not np.isfinite(p) or not np.isfinite(z) or not np.isfinite(nl):
            p_obs[i] = np.nan
            p_obs_low[i] = np.nan
            p_obs_two[i] = np.nan
            z_obs[i] = np.nan
            neglog10p_obs[i] = np.nan
        else:
            p_obs[i] = p
            z_obs[i] = z
            neglog10p_obs[i] = nl

            if baseline_sizes[i] > 0 and np.isfinite(diag_scores[i]):
                count_le = int(np.sum(F_i <= diag_scores[i]))
                p_low = (count_le + 1.0) / (baseline_sizes[i] + 1.0)
            else:
                p_low = np.nan

            p_obs_low[i] = p_low

            if np.isfinite(p_low):
                p_two = 2.0 * min(p, p_low)
                if p_two > 1.0:
                    p_two = 1.0
                p_obs_two[i] = p_two
            else:
                p_obs_two[i] = np.nan

    result = {
        "diag_scores": diag_scores,
        "p_obs": p_obs,
        "p_obs_low": p_obs_low,
        "p_obs_two": p_obs_two,
        "z_obs": z_obs,
        "neglog10p_obs": neglog10p_obs,
        "baseline_sizes": baseline_sizes,
    }
    return result

def _resolve_condition_labels(unique_conditions: np.ndarray) -> tuple:
    labels = list(unique_conditions)
    if len(labels) != 2:
        raise ValueError("_resolve_condition_labels expects exactly two conditions")

    # Always treat 'control' as control, and the other label as experimental
    if "control" in labels:
        labels.remove("control")
        exp_label = labels[0]
        return exp_label, "control"

    # Fallback: just treat the first as "experimental", second as "control"
    return labels[0], labels[1]


def _condition_stat(
    values: np.ndarray,
    conditions: np.ndarray,
    stat_fn,
) -> dict:
    vals = np.asarray(values, dtype=float)
    cond = np.asarray(conditions)

    mask = np.isfinite(vals)
    if not np.any(mask):
        raise ValueError("No finite values available for statistic computation")

    vals = vals[mask]
    cond = cond[mask]

    unique = np.unique(cond)
    if unique.size != 2:
        raise ValueError("_condition_stat assumes exactly two conditions")

    exp_label, ctrl_label = _resolve_condition_labels(unique)
    exp_mask = (cond == exp_label)
    ctrl_mask = (cond == ctrl_label)

    if not np.any(exp_mask) or not np.any(ctrl_mask):
        raise ValueError("Need at least one run per condition for statistic computation")

    overall = float(stat_fn(vals))
    exp_stat = float(stat_fn(vals[exp_mask]))
    ctrl_stat = float(stat_fn(vals[ctrl_mask]))

    return {
        "overall": overall,
        "exp": exp_stat,
        "ctrl": ctrl_stat,
        "delta": exp_stat - ctrl_stat,
    }


def mean_neglog10p_stat(neglog10p_vals: np.ndarray, conditions: np.ndarray) -> dict:
    return _condition_stat(neglog10p_vals, conditions, np.mean)


def max_z_stat(z_vals: np.ndarray, conditions: np.ndarray) -> dict:
    return _condition_stat(z_vals, conditions, np.max)


def skew_z_stat(z_vals: np.ndarray, conditions: np.ndarray) -> dict:
    return _condition_stat(z_vals, conditions, lambda arr: stats.skew(arr, bias=False))


DEFAULT_STAT_SPECS = {
    "mean_neglog10p": {"fn": mean_neglog10p_stat, "value_key": "neglog10p"},
    "max_z": {"fn": max_z_stat, "value_key": "z"},
    "skew_z": {"fn": skew_z_stat, "value_key": "z"},
}


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm

def blockwise_derangement(groups: list, rng: np.random.Generator, n: int) -> np.ndarray:
    perm = np.empty(n, dtype=int)
    for g in groups:
        k = g.size
        if k == 0:
            continue
        if k == 1:
            # no nontrivial derangement in a singleton group; leave it fixed
            perm[g[0]] = g[0]
            continue

        permuted = rng.permutation(g)
        while np.any(permuted == g):
            permuted = rng.permutation(g)
        perm[g] = permuted
    return perm

def compute_per_run_from_perm(
    P: np.ndarray,
    Z: np.ndarray,
    NL: np.ndarray,
    perm: np.ndarray,
) -> dict:
    n = P.shape[0]
    idx = np.arange(n, dtype=int)

    p_perm = P[idx, perm]
    z_perm = Z[idx, perm]
    neglog10p_perm = NL[idx, perm]

    return {
        "p_perm": p_perm,
        "z_perm": z_perm,
        "neglog10p_perm": neglog10p_perm,
    }


def _permutation_chunk_worker(args) -> dict:
    (
        P,
        Z,
        NL,
        n_perms_chunk,
        rng_seed,
        z_obs,
        groups,
        conditions,
        stat_specs,
    ) = args

    n = P.shape[0]
    rng = np.random.default_rng(rng_seed)
    stat_names = list(stat_specs.keys())
    chunk_values = {}
    for name in stat_names:
        chunk_values[f"{name}_overall"] = np.empty(n_perms_chunk, dtype=float)
        chunk_values[f"{name}_delta"] = np.empty(n_perms_chunk, dtype=float)
    counts_ge = np.zeros(n, dtype=np.int64)      # high-side counts (z_perm >= z_obs)
    counts_le = np.zeros(n, dtype=np.int64)      # low-side counts  (z_perm <= z_obs)

    for b in range(n_perms_chunk):
        perm = blockwise_derangement(groups, rng, n)

        per_run_perm = compute_per_run_from_perm(P, Z, NL, perm)
        value_lookup = {
            "p": per_run_perm["p_perm"],
            "z": per_run_perm["z_perm"],
            "neglog10p": per_run_perm["neglog10p_perm"],
        }

        for stat_name, spec in stat_specs.items():
            value_key = spec["value_key"]
            if value_key not in value_lookup:
                raise KeyError(f"Unknown value_key '{value_key}' for statistic '{stat_name}'")
            res = spec["fn"](value_lookup[value_key], conditions)
            chunk_values[f"{stat_name}_overall"][b] = res["overall"]
            chunk_values[f"{stat_name}_delta"][b] = res["delta"]

        z_perm = per_run_perm["z_perm"]
        mask = np.isfinite(z_obs) & np.isfinite(z_perm)
        counts_ge[mask] += (z_perm[mask] >= z_obs[mask])
        counts_le[mask] += (z_perm[mask] <= z_obs[mask])
        # absolute tail counts not needed once both one-sided tails are tracked

    chunk_values["counts_ge_z"] = counts_ge
    chunk_values["counts_le_z"] = counts_le
    return chunk_values

def calibrate_permutations(
    P: np.ndarray,
    Z: np.ndarray,
    NL: np.ndarray,
    p_obs: np.ndarray,
    z_obs: np.ndarray,
    neglog10p_obs: np.ndarray,
    n_perms: int,
    rng_seed: int,
    groups: list,
    conditions: np.ndarray,
    stat_specs: dict,
) -> dict:
    stat_names = list(stat_specs.keys())

    mask_effective = (
        np.isfinite(p_obs) & np.isfinite(z_obs) & np.isfinite(neglog10p_obs)
    )
    if not np.any(mask_effective):
        raise ValueError("No valid per-run values for global statistics")
    n_effective = int(np.sum(mask_effective))

    obs_stat_struct = {}
    value_lookup_obs = {
        "p": p_obs,
        "z": z_obs,
        "neglog10p": neglog10p_obs,
    }
    for stat_name, spec in stat_specs.items():
        value_key = spec["value_key"]
        if value_key not in value_lookup_obs:
            raise KeyError(f"Unknown value_key '{value_key}' for statistic '{stat_name}'")
        obs_stat_struct[stat_name] = spec["fn"](value_lookup_obs[value_key], conditions)

    n_workers = os.cpu_count() or 1
    if n_workers < 1:
        n_workers = 1

    target_chunks = max(1, n_workers * 4)
    base_chunk_size = max(1, n_perms // target_chunks)
    if base_chunk_size <= 0:
        base_chunk_size = 1

    chunk_sizes = []
    total = 0
    while total < n_perms:
        remaining = n_perms - total
        size = min(base_chunk_size, remaining)
        chunk_sizes.append(size)
        total += size

    args_list = []
    for idx, size in enumerate(chunk_sizes):
        chunk_seed = rng_seed + idx
        args_list.append(
            (
                P,
                Z,
                NL,
                size,
                chunk_seed,
                z_obs,
                groups,
                conditions,
                stat_specs,
            )
        )

    chunk_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_permutation_chunk_worker, a) for a in args_list]
        for fut in futures:
            chunk_results.append(fut.result())

    null_values = {}
    for name in stat_names:
        overall_key = f"{name}_overall"
        delta_key = f"{name}_delta"
        null_values[overall_key] = np.concatenate([cr[overall_key] for cr in chunk_results], axis=0)
        null_values[delta_key] = np.concatenate([cr[delta_key] for cr in chunk_results], axis=0)

    n = P.shape[0]
    total_counts_ge = np.zeros(n, dtype=np.int64)
    total_counts_le = np.zeros(n, dtype=np.int64)
    for cr in chunk_results:
        total_counts_ge += cr["counts_ge_z"]
        total_counts_le += cr["counts_le_z"]

    summary = {}
    for name in stat_names:
        res_obs = obs_stat_struct[name]
        obs_overall = res_obs["overall"]
        obs_delta = res_obs["delta"]
        null_overall = null_values[f"{name}_overall"]
        null_delta = null_values[f"{name}_delta"]

        count_ge = int(np.sum(null_overall >= obs_overall))
        p_perm_overall = (count_ge + 1.0) / (n_perms + 1.0)
        null_mean = float(np.mean(null_overall))
        null_std = float(np.std(null_overall, ddof=0))
        z_shift = (obs_overall - null_mean) / null_std if null_std > 0.0 else np.nan

        count_ge_delta = int(np.sum(null_delta >= obs_delta))
        p_perm_delta = (count_ge_delta + 1.0) / (n_perms + 1.0)
        delta_null_mean = float(np.mean(null_delta))
        delta_null_std = float(np.std(null_delta, ddof=0))

        summary[name] = {
            "obs": obs_overall,
            "obs_overall": obs_overall,
            "obs_exp": res_obs["exp"],
            "obs_ctrl": res_obs["ctrl"],
            "obs_delta": obs_delta,
            "null_mean": null_mean,
            "null_std": null_std,
            "delta_null_mean": delta_null_mean,
            "delta_null_std": delta_null_std,
            "z_shift_vs_null": z_shift,
            "p_perm": p_perm_overall,
            "p_perm_overall": p_perm_overall,
            "p_perm_delta": p_perm_delta,
        }

    summary["n_effective"] = n_effective

    z_obs_arr = np.asarray(z_obs, dtype=float)
    per_run_p_perm_high = np.full(n, np.nan, dtype=float)
    per_run_p_perm_low = np.full(n, np.nan, dtype=float)
    per_run_p_perm_two_sided = np.full(n, np.nan, dtype=float)
    per_run_z_perm = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(z_obs_arr)
    per_run_p_perm_high[valid] = (total_counts_ge[valid] + 1.0) / (n_perms + 1.0)
    per_run_p_perm_low[valid] = (total_counts_le[valid] + 1.0) / (n_perms + 1.0)

    both = np.vstack(
        [
            per_run_p_perm_high[valid],
            per_run_p_perm_low[valid],
        ]
    )
    per_run_p_perm_two_sided[valid] = np.minimum(1.0, 2.0 * np.min(both, axis=0))
    per_run_z_perm[valid] = stats.norm.isf(per_run_p_perm_high[valid])

    summary["per_run_p_perm_high"] = per_run_p_perm_high
    summary["per_run_p_perm_low"] = per_run_p_perm_low
    summary["per_run_p_perm_two_sided"] = per_run_p_perm_two_sided
    summary["per_run_z_perm_calibrated"] = per_run_z_perm
    summary["per_run_p_perm"] = per_run_p_perm_high

    return summary

def write_per_run_leak_scores(
    runs_df: pd.DataFrame,
    diag_scores: np.ndarray,
    p_obs: np.ndarray,
    z_obs: np.ndarray,
    neglog10p_obs: np.ndarray,
    baseline_sizes: np.ndarray,
    p_obs_low: np.ndarray,
    p_obs_two: np.ndarray,
    derange_p_perm_high: np.ndarray,
    derange_p_perm_low: np.ndarray,
    derange_p_perm_two_sided: np.ndarray,
    derange_z_perm: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(PER_RUN_LEAK_TSV), exist_ok=True)
    with open(PER_RUN_LEAK_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "run_index",
            "condition",
            "diag_score",
            "baseline_size",
            "empirical_p",
            "empirical_p_low",
            "empirical_p_two_sided",
            "z_from_p",
            "neglog10_p",
            "perm_p_derangements_high",
            "perm_p_derangements_low",
            "perm_p_derangements_two_sided",
            "z_perm_calibrated_derangements",
        ]
        writer.writerow(header)

        for idx, row in runs_df.iterrows():
            run_index = int(row["run_index"])
            cond = str(row["condition"])

            writer.writerow(
                [
                    run_index,
                    cond,
                    float(diag_scores[idx]),
                    int(baseline_sizes[idx]),
                    float(p_obs[idx]),
                    float(p_obs_low[idx]),
                    float(p_obs_two[idx]),
                    float(z_obs[idx]),
                    float(neglog10p_obs[idx]),
                    float(derange_p_perm_high[idx]),
                    float(derange_p_perm_low[idx]),
                    float(derange_p_perm_two_sided[idx]),
                    float(derange_z_perm[idx]),
                ]
            )

def write_perm_summary(summary: dict, scheme_name: str = "derangements") -> None:
    os.makedirs(os.path.dirname(PERM_SUMMARY_TSV), exist_ok=True)
    stat_names = list(DEFAULT_STAT_SPECS.keys())

    with open(PERM_SUMMARY_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "statistic",
            "scheme",
            "obs",
            "null_mean",
            "null_std",
            "z_shift_vs_null",
            "p_perm",
            "n_effective",
        ]
        writer.writerow(header)

        n_eff = summary["n_effective"]
        for stat in stat_names:
            s = summary[stat]
            writer.writerow(
                [
                    stat,
                    scheme_name,
                    s["obs"],
                    s["null_mean"],
                    s["null_std"],
                    s["z_shift_vs_null"],
                    s["p_perm"],
                    n_eff,
                ]
            )

def print_interpretation(
    summary: dict,
    n_runs: int,
    n_perms: int,
) -> None:
    print()
    print(
        "Global leak diagnostics based on derangement-calibrated "
        "per-secret statistics"
    )
    print("----------------------------------------------------------------")
    print(
        f"Number of runs used (n_effective): "
        f"{summary['n_effective']} of {n_runs}"
    )
    print(f"Derangement permutations   : {n_perms}")
    print()

    preferred_order = ["max_z", "skew_z", "mean_neglog10p"]
    stat_names = [name for name in preferred_order if name in DEFAULT_STAT_SPECS]

    for name in stat_names:
        print(f"Statistic: {name}")
        stats_dict = summary[name]
        print("  Derangement null (overall statistic across all runs):")
        print(f"    obs       = {stats_dict['obs_overall']:.4f}")
        print(
            f"    null mean = {stats_dict['null_mean']:.4f}, std = {stats_dict['null_std']:.4f}"
        )
        print(
            f"    shift     = {stats_dict['z_shift_vs_null']:.3f} sd, "
            f"p_perm_overall = {stats_dict['p_perm_overall']:.4g}"
        )
        print(
            f"    experimental = {stats_dict['obs_exp']:.4f}, "
            f"control = {stats_dict['obs_ctrl']:.4f}"
        )
        print("  Derangement null (experimental vs control difference):")
        print(
            f"    Δstat (exp - ctrl) = {stats_dict['obs_delta']:.4f}"
        )
        print(
            f"    null mean = {stats_dict['delta_null_mean']:.4f}, "
            f"std = {stats_dict['delta_null_std']:.4f}"
        )
        print(
            f"    p_perm_delta = {stats_dict['p_perm_delta']:.4g}"
        )
        print()

def _binary_kl(x: float, pi: float) -> float:
    eps = 1e-12
    x = min(max(x, eps), 1.0 - eps)
    pi = min(max(pi, eps), 1.0 - eps)
    return x * np.log(x / pi) + (1.0 - x) * np.log((1.0 - x) / (1.0 - pi))


def _bj_one_sample(p_values: np.ndarray, max_p: float = 0.5) -> float:
    p_values = np.asarray(p_values, dtype=float)
    mask = np.isfinite(p_values)
    p = p_values[mask]
    n = p.size
    if n == 0:
        return 0.0

    order = np.argsort(p)
    p_sorted = p[order]

    bj_max = 0.0
    for k in range(1, n + 1):
        p_k = p_sorted[k - 1]
        if p_k > max_p:
            break

        x = k / n
        bj_val = n * _binary_kl(x, float(p_k))
        if bj_val > bj_max:
            bj_max = bj_val

    return bj_max


def _bj_scan(p_values: np.ndarray, is_group_A: np.ndarray, max_p: float = 0.5) -> float:
    p_values = np.asarray(p_values, dtype=float)
    is_group_A = np.asarray(is_group_A, dtype=bool)

    mask = np.isfinite(p_values)
    if not np.any(mask):
        return 0.0

    p_vals = p_values[mask]
    gA = is_group_A[mask]

    if p_vals.size == 0:
        return 0.0

    order = np.argsort(p_vals)
    p_sorted = p_vals[order]
    gA_sorted = gA[order]

    pi = float(gA_sorted.mean())
    if pi <= 0.0 or pi >= 1.0:
        return 0.0

    bj_max = 0.0
    n_A_cum = 0
    n_cum = 0

    for k in range(p_sorted.size):
        p_k = p_sorted[k]
        if p_k > max_p:
            break

        n_cum += 1
        if gA_sorted[k]:
            n_A_cum += 1

        x_t = n_A_cum / n_cum
        if x_t <= pi:
            continue

        bj_val = n_cum * _binary_kl(x_t, pi)
        if bj_val > bj_max:
            bj_max = bj_val

    return bj_max


def run_bj_condition_enrichment(
    p_left: np.ndarray,
    conditions: np.ndarray,
    n_perms: int,
    rng_seed: int,
) -> dict:
    p_left = np.asarray(p_left, dtype=float)
    cond = np.asarray(conditions)

    mask = np.isfinite(p_left) & (
        (cond == "control") | (cond == "experimental")
    )
    if not np.any(mask):
        raise ValueError("No valid runs for BJ condition enrichment")

    p_vals = p_left[mask]
    cond_vals = cond[mask]

    is_ctrl = cond_vals == "control"
    bj_obs = _bj_scan(p_vals, is_ctrl, max_p=0.5)

    rng = np.random.default_rng(rng_seed)
    bj_null = np.empty(n_perms, dtype=float)

    for b in range(n_perms):
        perm_labels = rng.permutation(is_ctrl)
        bj_null[b] = _bj_scan(p_vals, perm_labels, max_p=0.5)

    finite = np.isfinite(bj_null)
    if not np.any(finite):
        p_perm = np.nan
        null_mean = np.nan
        null_std = np.nan
    else:
        null_arr = bj_null[finite]
        p_perm = (1.0 + np.sum(null_arr >= bj_obs)) / (1.0 + null_arr.size)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr, ddof=0))

    return {
        "bj_obs": bj_obs,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_perm": p_perm,
    }


def run_bj_overall(
    p_left: np.ndarray,
    n_perms: int,
    rng_seed: int,
    max_p: float = 0.5,
) -> dict:
    p_left = np.asarray(p_left, dtype=float)
    mask = np.isfinite(p_left)
    if not np.any(mask):
        raise ValueError("No valid runs for BJ overall test")

    p_vals = p_left[mask]
    n_used = p_vals.size

    bj_obs = _bj_one_sample(p_vals, max_p=max_p)

    rng = np.random.default_rng(rng_seed)
    bj_null = np.empty(n_perms, dtype=float)

    for b in range(n_perms):
        u = rng.random(n_used)
        bj_null[b] = _bj_one_sample(u, max_p=max_p)

    finite = np.isfinite(bj_null)
    if not np.any(finite):
        p_perm = np.nan
        null_mean = np.nan
        null_std = np.nan
    else:
        null_arr = bj_null[finite]
        p_perm = (1.0 + np.sum(null_arr >= bj_obs)) / (1.0 + null_arr.size)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr, ddof=0))

    return {
        "bj_obs": bj_obs,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_perm": p_perm,
        "n_used": n_used,
    }


def run_analysis_for_pair(
    runs_df_full: pd.DataFrame,
    S_full: np.ndarray,
    include_conditions: tuple,
    scheme_name: str,
) -> None:
    # Select only runs with the requested conditions (e.g. control + exp_phase1)
    mask = runs_df_full["condition"].isin(include_conditions)
    subset_runs = runs_df_full[mask].copy().reset_index(drop=True)
    n_runs = subset_runs.shape[0]
    print(f"\n=== {scheme_name}: using {n_runs} runs ===")

    if n_runs == 0:
        print("No runs for this comparison; skipping.")
        return

    unique_subset_conditions = subset_runs["condition"].astype(str).unique()
    if unique_subset_conditions.size != 2:
        print(
            "This comparison requires exactly two conditions; "
            f"found {unique_subset_conditions.size}. Skipping."
        )
        return

    # Map to the appropriate block of the score matrix
    indices = np.where(mask.to_numpy())[0]
    S = S_full[np.ix_(indices, indices)]
    print(f"Score matrix subset shape: {S.shape}")

    print("Building per-secret baselines F_i from off-diagonal same-condition scores")
    conditions = subset_runs["condition"].astype(str).to_numpy()
    baselines = build_baselines(S, conditions)

    print("Defining within-condition groups for permutation null")
    unique_conditions = sorted(set(conditions))
    groups = [np.where(conditions == c)[0] for c in unique_conditions]

    print("Precomputing per-pair empirical p, z, and -log10 p")
    P, Z, NL = precompute_pairwise_pznl(S, baselines)

    print("Computing observed per-run leak scores (p_i, z_i, -log10 p_i)")
    obs_per_run = compute_per_run_observed(S, baselines, P, Z, NL)
    diag_scores = obs_per_run["diag_scores"]
    p_obs = obs_per_run["p_obs"]
    p_obs_low = obs_per_run["p_obs_low"]
    p_obs_two = obs_per_run["p_obs_two"]
    z_obs = obs_per_run["z_obs"]
    neglog10p_obs = obs_per_run["neglog10p_obs"]
    baseline_sizes = obs_per_run["baseline_sizes"]

    print("Calibrating permutation null (derangements only, within condition)")
    derange_summary = calibrate_permutations(
        P=P,
        Z=Z,
        NL=NL,
        p_obs=p_obs,
        z_obs=z_obs,
        neglog10p_obs=neglog10p_obs,
        n_perms=DERANGEMENT_PERMS,
        rng_seed=RNG_SEED_DERANGE,
        groups=groups,
        conditions=conditions,
        stat_specs=DEFAULT_STAT_SPECS,
    )

    # Use separate output files per comparison
    global PER_RUN_LEAK_TSV, PERM_SUMMARY_TSV
    PER_RUN_LEAK_TSV = os.path.join(
        "sonnet_cot_experiment",
        f"per_run_leak_scores_{scheme_name}.tsv",
    )
    PERM_SUMMARY_TSV = os.path.join(
        "sonnet_cot_experiment",
        f"permutation_summary_{scheme_name}.tsv",
    )

    print(f"Writing per-run leak scores to {PER_RUN_LEAK_TSV}")
    write_per_run_leak_scores(
        subset_runs,
        diag_scores,
        p_obs,
        z_obs,
        neglog10p_obs,
        baseline_sizes,
        p_obs_low,
        p_obs_two,
        derange_summary["per_run_p_perm_high"],
        derange_summary["per_run_p_perm_low"],
        derange_summary["per_run_p_perm_two_sided"],
        derange_summary["per_run_z_perm_calibrated"],
    )

    print(f"Writing permutation summary to {PERM_SUMMARY_TSV}")
    write_perm_summary(derange_summary, scheme_name)

    print_interpretation(
        summary=derange_summary,
        n_runs=n_runs,
        n_perms=DERANGEMENT_PERMS,
    )

    print(
        "\nRunning Berk–Jones overall test on permutation-calibrated leak p-values"
    )
    bj_overall = run_bj_overall(
        p_left=derange_summary["per_run_p_perm_high"],
        n_perms=LABEL_PERMS,
        rng_seed=RNG_SEED_LABEL,
    )

    print("Berk–Jones overall enrichment on permutation-calibrated leak p-values")
    print(
        "  BJ statistic (overall enrichment of small leak p-values): "
        f"{bj_overall['bj_obs']:.4f}"
    )
    print(
        "  Null mean ± std: "
        f"{bj_overall['null_mean']:.4f} ± {bj_overall['null_std']:.4f}"
    )
    print(
        "  Permutation p-value: "
        f"{bj_overall['p_perm']:.5f}"
    )
    print(
        f"  Number of runs used: {bj_overall['n_used']}"
    )


def main() -> None:
    print(f"Loading runs from {RUNS_TSV}")
    runs_df_full = load_runs(RUNS_TSV)
    n_runs_full = runs_df_full.shape[0]
    print(f"Found {n_runs_full} runs total")

    print(f"Loading score matrix from {SCORE_MATRIX_TSV}")
    S_full = load_score_matrix(SCORE_MATRIX_TSV, n_runs_full)
    print(f"Full score matrix shape: {S_full.shape}")

    # First analysis: control vs experimental_phase1
    run_analysis_for_pair(
        runs_df_full,
        S_full,
        include_conditions=("control", "experimental_phase1"),
        scheme_name="control_vs_experimental_phase1",
    )

    # Second analysis: control vs experimental_phase2
    run_analysis_for_pair(
        runs_df_full,
        S_full,
        include_conditions=("control", "experimental_phase2"),
        scheme_name="control_vs_experimental_phase2",
    )

if __name__ == "__main__":
    main()
