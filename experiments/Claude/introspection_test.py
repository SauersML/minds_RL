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
ALL_PERMS = 1 # Disabled for speed
RNG_SEED_DERANGE = 12345
RNG_SEED_ALLPERM = 67890

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
    z_obs = np.empty(n, dtype=float)
    neglog10p_obs = np.empty(n, dtype=float)
    baseline_sizes = np.empty(n, dtype=int)

    for i in range(n):
        diag_scores[i] = float(S[i, i])
        baseline_sizes[i] = int(baselines[i].size)

        p = float(P[i, i])
        z = float(Z[i, i])
        nl = float(NL[i, i])

        if not np.isfinite(p) or not np.isfinite(z) or not np.isfinite(nl):
            p_obs[i] = np.nan
            z_obs[i] = np.nan
            neglog10p_obs[i] = np.nan
        else:
            p_obs[i] = p
            z_obs[i] = z
            neglog10p_obs[i] = nl

    result = {
        "diag_scores": diag_scores,
        "p_obs": p_obs,
        "z_obs": z_obs,
        "neglog10p_obs": neglog10p_obs,
        "baseline_sizes": baseline_sizes,
    }
    return result

def compute_global_stats(
    p_vals: np.ndarray,
    z_vals: np.ndarray,
    neglog10p_vals: np.ndarray,
) -> dict:
    p = np.asarray(p_vals, dtype=float)
    z = np.asarray(z_vals, dtype=float)
    nl = np.asarray(neglog10p_vals, dtype=float)

    mask = np.isfinite(p) & np.isfinite(z) & np.isfinite(nl)
    if not np.any(mask):
        raise ValueError("No valid per-run values for global statistics")

    p_valid = p[mask]
    z_valid = z[mask]
    nl_valid = nl[mask]
    
    mean_neglog10p = float(np.mean(nl_valid))
    max_z = float(np.max(z_valid))
    skew_z = float(stats.skew(z_valid, bias=False))
    
    stats_dict = {
        "mean_neglog10p": mean_neglog10p,
        "max_z": max_z,
        "skew_z": skew_z,
        "n_effective": int(z_valid.size),
    }

    return stats_dict


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm

def blockwise_permutation(groups: list, rng: np.random.Generator, n: int) -> np.ndarray:
    perm = np.empty(n, dtype=int)
    for g in groups:
        if g.size == 0:
            continue
        perm[g] = rng.permutation(g)
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
        scheme,
        n_perms_chunk,
        rng_seed,
        z_obs,
        groups,
    ) = args

    n = P.shape[0]
    rng = np.random.default_rng(rng_seed)
    stat_names = ["mean_neglog10p", "max_z", "skew_z"]
    chunk_values = {name: np.empty(n_perms_chunk, dtype=float) for name in stat_names}
    counts_ge = np.zeros(n, dtype=np.int64)
    counts_ge_abs = np.zeros(n, dtype=np.int64)

    for b in range(n_perms_chunk):
        if scheme == "derangements":
            perm = blockwise_derangement(groups, rng, n)
        elif scheme == "all_permutations":
            perm = blockwise_permutation(groups, rng, n)
        else:
            raise ValueError("Unknown scheme: " + scheme)

        per_run_perm = compute_per_run_from_perm(P, Z, NL, perm)
        stats_perm = compute_global_stats(
            per_run_perm["p_perm"],
            per_run_perm["z_perm"],
            per_run_perm["neglog10p_perm"],
        )

        for name in stat_names:
            chunk_values[name][b] = stats_perm[name]

        z_perm = per_run_perm["z_perm"]
        mask = np.isfinite(z_obs) & np.isfinite(z_perm)
        counts_ge[mask] += (z_perm[mask] >= z_obs[mask])
        counts_ge_abs[mask] += (np.abs(z_perm[mask]) >= np.abs(z_obs[mask]))

    chunk_values["counts_ge_z"] = counts_ge
    chunk_values["counts_ge_z_abs"] = counts_ge_abs
    return chunk_values

def calibrate_permutations(
    P: np.ndarray,
    Z: np.ndarray,
    NL: np.ndarray,
    p_obs: np.ndarray,
    z_obs: np.ndarray,
    neglog10p_obs: np.ndarray,
    scheme: str,
    n_perms: int,
    rng_seed: int,
    groups: list,
) -> dict:
    obs_stats = compute_global_stats(p_obs, z_obs, neglog10p_obs)
    stat_names = ["mean_neglog10p", "max_z", "skew_z"]

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
                scheme,
                size,
                chunk_seed,
                z_obs,
                groups,
            )
        )

    chunk_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_permutation_chunk_worker, a) for a in args_list]
        for fut in futures:
            chunk_results.append(fut.result())

    null_values = {}
    for name in stat_names:
        parts = [cr[name] for cr in chunk_results]
        null_values[name] = np.concatenate(parts, axis=0)

    n = P.shape[0]
    total_counts_ge = np.zeros(n, dtype=np.int64)
    total_counts_ge_abs = np.zeros(n, dtype=np.int64)
    for cr in chunk_results:
        total_counts_ge += cr["counts_ge_z"]
        total_counts_ge_abs += cr["counts_ge_z_abs"]

    summary = {}
    for name in stat_names:
        obs = obs_stats[name]
        null_arr = null_values[name]
        count_ge = int(np.sum(null_arr >= obs))
        p_perm = (count_ge + 1.0) / (n_perms + 1.0)
        null_mean = float(np.mean(null_arr))
        null_std = float(np.std(null_arr, ddof=0))
        z_shift = (obs - null_mean) / null_std if null_std > 0.0 else np.nan

        summary[name] = {
            "obs": obs,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_shift_vs_null": z_shift,
            "p_perm": p_perm,
        }

    summary["n_effective"] = obs_stats["n_effective"]

    z_obs_arr = np.asarray(z_obs, dtype=float)
    per_run_p_perm = np.full(n, np.nan, dtype=float)
    per_run_p_perm_two_sided = np.full(n, np.nan, dtype=float)
    per_run_z_perm = np.full(n, np.nan, dtype=float)

    valid = np.isfinite(z_obs_arr)
    per_run_p_perm[valid] = (total_counts_ge[valid] + 1.0) / (n_perms + 1.0)
    per_run_p_perm_two_sided[valid] = (total_counts_ge_abs[valid] + 1.0) / (n_perms + 1.0)
    per_run_z_perm[valid] = stats.norm.isf(per_run_p_perm[valid])

    summary["per_run_p_perm"] = per_run_p_perm
    summary["per_run_p_perm_two_sided"] = per_run_p_perm_two_sided
    summary["per_run_z_perm_calibrated"] = per_run_z_perm

    return summary

def write_per_run_leak_scores(
    runs_df: pd.DataFrame,
    diag_scores: np.ndarray,
    p_obs: np.ndarray,
    z_obs: np.ndarray,
    neglog10p_obs: np.ndarray,
    baseline_sizes: np.ndarray,
    derange_p_perm: np.ndarray,
    derange_p_perm_two_sided: np.ndarray,
    derange_z_perm: np.ndarray,
    allperm_p_perm: np.ndarray,
    allperm_z_perm: np.ndarray,
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
            "z_from_p",
            "neglog10_p",
            "perm_p_derangements",
            "perm_p_derangements_two_sided",
            "z_perm_calibrated_derangements",
            "perm_p_all_permutations",
            "z_perm_calibrated_all_permutations",
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
                    float(z_obs[idx]),
                    float(neglog10p_obs[idx]),
                    float(derange_p_perm[idx]),
                    float(derange_p_perm_two_sided[idx]),
                    float(derange_z_perm[idx]),
                    float(allperm_p_perm[idx]),
                    float(allperm_z_perm[idx]),
                ]
            )

def write_perm_summary(
    derange_summary: dict,
    allperm_summary: dict,
) -> None:
    os.makedirs(os.path.dirname(PERM_SUMMARY_TSV), exist_ok=True)
    stat_names = ["mean_neglog10p", "max_z", "skew_z",]

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

        for scheme_name, summary in [
            ("derangements", derange_summary),
            ("all_permutations", allperm_summary),
        ]:
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
    derange_summary: dict,
    allperm_summary: dict,
    n_runs: int,
    n_derange: int,
    n_allperm: int,
) -> None:
    print()
    print("Global leak diagnostics based on per-secret empirical p-values")
    print("----------------------------------------------------------------")
    print(
        f"Number of runs used (n_effective): "
        f"{derange_summary['n_effective']} of {n_runs}"
    )
    print(f"Derangement permutations   : {n_derange}")
    print(f"All-permutations permutations: {n_allperm}")
    print()

    stat_names = ["max_z", "skew_z", "mean_neglog10p"]

    for name in stat_names:
        s_d = derange_summary[name]
        s_a = allperm_summary[name]

        print(f"Statistic: {name}")
        print("  Derangements null:")
        print(f"    obs       = {s_d['obs']:.4f}")
        print(f"    null mean = {s_d['null_mean']:.4f}, std = {s_d['null_std']:.4f}")
        print(
            f"    shift     = {s_d['z_shift_vs_null']:.3f} sd, "
            f"p_perm = {s_d['p_perm']:.4g}"
        )

        print("  All-permutations null:")
        print(f"    obs       = {s_a['obs']:.4f}")
        print(f"    null mean = {s_a['null_mean']:.4f}, std = {s_a['null_std']:.4f}")
        print(
            f"    shift     = {s_a['z_shift_vs_null']:.3f} sd, "
            f"p_perm = {s_a['p_perm']:.4g}"
        )
        print()

def _binary_kl(x: float, pi: float) -> float:
    eps = 1e-12
    x = min(max(x, eps), 1.0 - eps)
    pi = min(max(pi, eps), 1.0 - eps)
    return x * np.log(x / pi) + (1.0 - x) * np.log((1.0 - x) / (1.0 - pi))


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


def run_condition_difference_tests(
    Z: np.ndarray,
    z_obs: np.ndarray,
    conditions: np.ndarray,
    groups: list,
    n_perms: int,
    rng_seed: int,
) -> dict:
    # z_obs is the observed per-run z (e.g. z_obs from compute_per_run_observed)
    z_obs = np.asarray(z_obs, dtype=float)
    cond = np.asarray(conditions)

    unique = np.unique(cond)
    if unique.size != 2:
        raise ValueError(f"Expected exactly two conditions, got {unique}")

    exp_mask = (cond == "experimental")
    ctrl_mask = (cond == "control")

    valid = np.isfinite(z_obs) & (exp_mask | ctrl_mask)
    if not np.any(valid):
        raise ValueError("No valid z_obs values for condition skew tests")

    z_obs_valid = z_obs[valid]
    cond_valid = cond[valid]

    exp_mask_valid = (cond_valid == "experimental")
    ctrl_mask_valid = (cond_valid == "control")

    if not np.any(exp_mask_valid) or not np.any(ctrl_mask_valid):
        raise ValueError("Need at least one experimental and one control run for tests")

    # Observed skew within each condition (labels fixed)
    z_exp_obs = z_obs_valid[exp_mask_valid]
    z_ctrl_obs = z_obs_valid[ctrl_mask_valid]

    skew_exp_obs = float(stats.skew(z_exp_obs, bias=False))
    skew_ctrl_obs = float(stats.skew(z_ctrl_obs, bias=False))
    delta_skew_obs = skew_exp_obs - skew_ctrl_obs

    # Build null by deranging guesses within condition (groups),
    # then recomputing per-condition skew from z_perm = Z[i, perm[i]].
    rng = np.random.default_rng(rng_seed)
    n = Z.shape[0]
    idx = np.arange(n, dtype=int)

    skew_exp_null = np.empty(n_perms, dtype=float)
    skew_ctrl_null = np.empty(n_perms, dtype=float)
    delta_skew_null = np.empty(n_perms, dtype=float)

    for b in range(n_perms):
        # permutes guess indices within each condition group
        perm = blockwise_derangement(groups, rng, n)

        # per-run z under this deranged pairing
        z_perm_full = Z[idx, perm]

        z_perm_valid = z_perm_full[valid]
        z_exp_perm = z_perm_valid[exp_mask_valid]
        z_ctrl_perm = z_perm_valid[ctrl_mask_valid]

        skew_e = float(stats.skew(z_exp_perm, bias=False))
        skew_c = float(stats.skew(z_ctrl_perm, bias=False))

        skew_exp_null[b] = skew_e
        skew_ctrl_null[b] = skew_c
        delta_skew_null[b] = skew_e - skew_c

    p_exp_two_sided = (1.0 + np.sum(np.abs(skew_exp_null)  >= abs(skew_exp_obs)))  / (1.0 + n_perms)
    p_ctrl_two_sided = (1.0 + np.sum(np.abs(skew_ctrl_null) >= abs(skew_ctrl_obs))) / (1.0 + n_perms)
    p_delta_one_sided = (1.0 + np.sum(delta_skew_null >= delta_skew_obs))          / (1.0 + n_perms)

    return {
        "skew_exp_obs": skew_exp_obs,
        "skew_exp_p_two_sided": p_exp_two_sided,
        "skew_ctrl_obs": skew_ctrl_obs,
        "skew_ctrl_p_two_sided": p_ctrl_two_sided,
        "delta_skew_obs": delta_skew_obs,
        "delta_skew_p_one_sided": p_delta_one_sided,
    }

def main() -> None:
    print(f"Loading runs from {RUNS_TSV}")
    runs_df = load_runs(RUNS_TSV)
    n_runs = runs_df.shape[0]
    print(f"Found {n_runs} runs")

    print(f"Loading score matrix from {SCORE_MATRIX_TSV}")
    S = load_score_matrix(SCORE_MATRIX_TSV, n_runs)
    print(f"Score matrix shape: {S.shape}")

    print("Building per-secret baselines F_i from off-diagonal same-condition scores")
    conditions = runs_df["condition"].astype(str).to_numpy()
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
        scheme="derangements",
        n_perms=DERANGEMENT_PERMS,
        rng_seed=RNG_SEED_DERANGE,
        groups=groups,
    )
    
    print("Calibrating permutation null (all permutations, within condition)")
    allperm_summary = calibrate_permutations(
        P=P,
        Z=Z,
        NL=NL,
        p_obs=p_obs,
        z_obs=z_obs,
        neglog10p_obs=neglog10p_obs,
        scheme="all_permutations",
        n_perms=ALL_PERMS,
        rng_seed=RNG_SEED_ALLPERM,
        groups=groups,
    )

    print(f"Writing per-run leak scores to {PER_RUN_LEAK_TSV}")
    write_per_run_leak_scores(
        runs_df,
        diag_scores,
        p_obs,
        z_obs,
        neglog10p_obs,
        baseline_sizes,
        derange_summary["per_run_p_perm"],
        derange_summary["per_run_p_perm_two_sided"],
        derange_summary["per_run_z_perm_calibrated"],
        allperm_summary["per_run_p_perm"],
        allperm_summary["per_run_z_perm_calibrated"],
    )

    print(f"Writing permutation summary to {PERM_SUMMARY_TSV}")
    write_perm_summary(derange_summary, allperm_summary)

    print_interpretation(
        derange_summary=derange_summary,
        allperm_summary=allperm_summary,
        n_runs=n_runs,
        n_derange=DERANGEMENT_PERMS,
        n_allperm=ALL_PERMS,
    )

    print("Running experimental vs control skew tests using within-condition derangement null")
    cond_tests = run_condition_difference_tests(
        Z=Z,
        z_obs=z_obs,
        conditions=conditions,
        groups=groups,
        n_perms=LABEL_PERMS,
        rng_seed=RNG_SEED_LABEL,
    )

    print()
    print("Condition comparison tests based on skew(z_perm_calibrated_derangements)")
    print("-----------------------------------------------------------------------")
    print(
        f"Skew(z) in experimental runs: {cond_tests['skew_exp_obs']:.4f}, "
        f"p_perm (two-sided by |skew|) = {cond_tests['skew_exp_p_two_sided']:.5f}"
    )
    print(
        f"Skew(z) in control runs: {cond_tests['skew_ctrl_obs']:.4f}, "
        f"p_perm (two-sided by |skew|) = {cond_tests['skew_ctrl_p_two_sided']:.5f}"
    )
    print(
        "One-sided test that experimental skews more positive and control more negative:"
    )
    print(
        f"  Δskew (experimental - control) = {cond_tests['delta_skew_obs']:.4f}, "
        f"p_perm = {cond_tests['delta_skew_p_one_sided']:.5f}"
    )

if __name__ == "__main__":
    main()
