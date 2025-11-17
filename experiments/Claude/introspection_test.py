import os

import numpy as np
import pandas as pd
from scipy import stats


def load_z_scores(tsv_path: str, column: str = "score") -> np.ndarray:
    df = pd.read_csv(tsv_path, sep="\t")
    z = pd.to_numeric(df[column], errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    return z.to_numpy()


def one_in_string(p: float) -> str:
    if p <= 0.0:
        return "less than 1 in (simulation resolution)"
    inv = 1.0 / p
    if inv >= 1e9:
        return f"about 1 in {inv:.2e}"
    return f"about 1 in {inv:,.1f}"


def analytic_extreme_p(z_vals: np.ndarray) -> dict:
    z = np.asarray(z_vals, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        raise ValueError("No valid score values found")

    z_max = float(np.max(z))
    # Right-tail single-run probability under N(0,1)
    p_single = float(stats.norm.sf(z_max))
    # Global probability that max of n i.i.d. N(0,1) exceeds z_max
    p_global = 1.0 - (1.0 - p_single) ** n

    return {
        "n": n,
        "z_max": z_max,
        "p_single": p_single,
        "p_global": p_global,
    }


def monte_carlo_extreme_p(
    z_vals: np.ndarray,
    n_sim: int,
    p_global_analytic: float,
    rng_seed: int = 12345,
) -> dict:
    z = np.asarray(z_vals, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        raise ValueError("No valid score values found")

    z_max = float(np.max(z))
    rng = np.random.default_rng(rng_seed)

    count_ge = 0
    for _ in range(n_sim):
        sim = rng.standard_normal(size=n)
        if np.max(sim) >= z_max:
            count_ge += 1

    p_mc = (count_ge + 1.0) / (n_sim + 1.0)
    expected_exceed = n_sim * p_global_analytic

    return {
        "n": n,
        "z_max": z_max,
        "n_sim": n_sim,
        "count_ge": count_ge,
        "p_mc": p_mc,
        "expected_exceed": expected_exceed,
    }


def extreme_value_test(z_vals: np.ndarray, n_sim: int = 1_000_000) -> None:
    analytic = analytic_extreme_p(z_vals)
    mc = monte_carlo_extreme_p(z_vals, n_sim=n_sim, p_global_analytic=analytic["p_global"])

    print("Extreme-value test for max score under N(0,1) null")
    print("-------------------------------------------------------------")
    print(f"Number of runs (n):                      {analytic['n']}")
    print(f"Maximum observed cross_run_z:            {analytic['z_max']:.6f}")
    print()
    print("Analytic calculation (independent N(0,1) z-scores):")
    print(f"  Single-run tail P(Z >= z_max):         {analytic['p_single']:.6e}")
    print(
        f"  Global P(max Z >= z_max):              {analytic['p_global']:.6e} "
        f"({one_in_string(analytic['p_global'])})"
    )
    print()
    print("Monte Carlo check (max of n standard normals):")
    print(f"  Simulations (N_SIM):                   {mc['n_sim']}")
    print(f"  Actual exceedances (sim max >= z_max): {mc['count_ge']}")
    print(
        f"  Expected exceedances under analytic p: {mc['expected_exceed']:.2f} "
        f"(= N_SIM * analytic_global_p)"
    )
    print(
        f"  Empirical P(max Z >= z_max):           {mc['p_mc']:.6e} "
        f"({one_in_string(mc['p_mc'])})"
    )


def test_positive_asymmetry(
    z_vals: np.ndarray,
    n_perm: int = 1_000_000,
    rng_seed: int = 123,
) -> None:
    z = np.asarray(z_vals, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        print("No finite z-scores available.")
        return

    mean_z = float(z.mean())
    std_z = float(z.std(ddof=0))
    if std_z == 0.0:
        print(f"All z-scores are identical (n={n}), cannot test asymmetry.")
        return

    # Use library skewness for the observed statistic
    skew_obs = float(stats.skew(z, bias=False))

    abs_z = np.abs(z)
    rng = np.random.default_rng(rng_seed)

    skew_perm = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        signs = rng.integers(0, 2, size=n, dtype=int) * 2 - 1
        z_perm = abs_z * signs
        skew_perm[b] = float(stats.skew(z_perm, bias=False))

    p_one_sided = (np.sum(skew_perm >= skew_obs) + 1.0) / (n_perm + 1.0)

    frac_pos = float((z > 0).mean())
    frac_neg = float((z < 0).mean())

    print("Positive-asymmetry sign-flip test on cross-run z-scores")
    print(f"  n runs                      : {n}")
    print(f"  mean(z)                     : {mean_z:.4f}")
    print(f"  std(z)                      : {std_z:.4f}")
    print(f"  fraction z > 0              : {frac_pos:.3f}")
    print(f"  fraction z < 0              : {frac_neg:.3f}")
    print(f"  observed skewness           : {skew_obs:.4f}")
    print(f"  sign-flip permutations      : {n_perm}")
    print(f"  one-sided p (skew > 0)      : {p_one_sided:.3g}")


def run_tail_enrichment_tests(
    z_vals: np.ndarray,
    n_mc: int = 4000,
    rng_seed: int = 12345,
) -> None:
    """
    One-sided higher-criticism tests for right and left tails of a z-score sample,
    with Monte Carlo calibration under i.i.d. N(0,1) null.
    """
    z = np.asarray(z_vals, dtype=float).ravel()
    z = z[~np.isnan(z)]
    n = z.size
    if n == 0:
        print("No valid z-scores provided.")
        return

    eps = 1e-16

    def one_sided_pvals(z_arr: np.ndarray, tail: str) -> np.ndarray:
        if tail == "right":
            p = stats.norm.sf(z_arr)  # P(Z >= z)
        elif tail == "left":
            p = stats.norm.cdf(z_arr)  # P(Z <= z)
        else:
            raise ValueError("tail must be 'right' or 'left'")
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1.0 - eps)
        p.sort()
        return p

    def compute_hc(p_sorted: np.ndarray) -> float:
        n_local = p_sorted.size
        if n_local == 0:
            return 0.0
        idx = np.arange(1, n_local + 1, dtype=float)
        lower = 1.0 / n_local
        upper = 0.5
        mask = (p_sorted >= lower) & (p_sorted <= upper)
        ps = p_sorted[mask]
        ks = idx[mask]
        if ps.size == 0:
            return 0.0
        x = ks / n_local
        numer = x - ps
        denom = np.sqrt(ps * (1.0 - ps))
        hc_vals = np.sqrt(n_local) * numer / denom
        return float(np.max(hc_vals))

    p_right = one_sided_pvals(z, "right")
    p_left = one_sided_pvals(z, "left")

    hc_right_obs = compute_hc(p_right)
    hc_left_obs = compute_hc(p_left)

    rng = np.random.default_rng(rng_seed)
    hc_right_null = np.empty(n_mc, dtype=float)
    hc_left_null = np.empty(n_mc, dtype=float)

    for i in range(n_mc):
        z_sim = rng.standard_normal(n)
        p_r_sim = one_sided_pvals(z_sim, "right")
        p_l_sim = one_sided_pvals(z_sim, "left")
        hc_right_null[i] = compute_hc(p_r_sim)
        hc_left_null[i] = compute_hc(p_l_sim)

    hc_right_p = (1.0 + np.sum(hc_right_null >= hc_right_obs)) / (n_mc + 1.0)
    hc_left_p = (1.0 + np.sum(hc_left_null >= hc_left_obs)) / (n_mc + 1.0)

    print(f"Sample size (z-scores): {n}")
    print("Right-tail higher-criticism (enrichment of large positive z):")
    print(f"  HC statistic = {hc_right_obs:.4f}")
    print(f"  Monte Carlo p-value = {hc_right_p:.4g}")
    print("Left-tail higher-criticism (enrichment of large negative z):")
    print(f"  HC statistic = {hc_left_obs:.4f}")
    print(f"  Monte Carlo p-value = {hc_left_p:.4g}")


def main() -> None:
    tsv_path = os.path.join("sonnet_cot_experiment", "alignment_results.tsv")

    z_vals = load_z_scores(tsv_path)

    run_tail_enrichment_tests(z_vals, n_mc=4000, rng_seed=12345)
    test_positive_asymmetry(z_vals, n_perm=1_000_000, rng_seed=123)
    extreme_value_test(z_vals, n_sim=1_000_000)


if __name__ == "__main__":
    main()
