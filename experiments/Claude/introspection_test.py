import math
import os

import numpy as np
import pandas as pd

TSV_PATH = os.path.join("sonnet_cot_experiment", "alignment_results.tsv")
N_SIM = 100000


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_z_scores(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep="\t")
    z = pd.to_numeric(df["cross_run_z_score"], errors="coerce")
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    return z.to_numpy()


def analytic_extreme_p(z_vals: np.ndarray) -> dict:
    n = len(z_vals)
    if n == 0:
        raise ValueError("No valid cross_run_z_score values found")

    z_max = float(np.max(z_vals))
    p_single = 1.0 - normal_cdf(z_max)
    p_global = 1.0 - (1.0 - p_single) ** n

    return {
        "n": n,
        "z_max": z_max,
        "p_single": p_single,
        "p_global": p_global,
    }


def monte_carlo_extreme_p(z_vals: np.ndarray, n_sim: int, p_global_analytic: float) -> dict:
    n = len(z_vals)
    if n == 0:
        raise ValueError("No valid cross_run_z_score values found")

    z_max = float(np.max(z_vals))
    rng = np.random.default_rng(12345)

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


def one_in_string(p: float) -> str:
    if p <= 0.0:
        return "less than 1 in (simulation resolution)"
    inv = 1.0 / p
    if inv >= 1e9:
        return f"about 1 in {inv:.2e}"
    return f"about 1 in {inv:,.1f}"

def test_positive_asymmetry(z_vals: np.ndarray) -> None:
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

    x = (z - mean_z) / std_z
    skew_obs = float(np.mean(x**3))

    abs_z = np.abs(z)
    B = 20000
    rng = np.random.default_rng(123)

    def skewness(arr: np.ndarray) -> float:
        m = float(arr.mean())
        s = float(arr.std(ddof=0))
        if s == 0.0:
            return 0.0
        x_loc = (arr - m) / s
        return float(np.mean(x_loc**3))

    skew_perm = np.empty(B, dtype=float)
    for b in range(B):
        signs = rng.integers(0, 2, size=n, dtype=int) * 2 - 1
        z_perm = abs_z * signs
        skew_perm[b] = skewness(z_perm)

    p_one_sided = (np.sum(skew_perm >= skew_obs) + 1.0) / (B + 1.0)

    frac_pos = float((z > 0).mean())
    frac_neg = float((z < 0).mean())

    print("Positive-asymmetry sign-flip test on cross-run z-scores")
    print(f"  n runs                      : {n}")
    print(f"  mean(z)                     : {mean_z:.4f}")
    print(f"  std(z)                      : {std_z:.4f}")
    print(f"  fraction z > 0              : {frac_pos:.3f}")
    print(f"  fraction z < 0              : {frac_neg:.3f}")
    print(f"  observed skewness           : {skew_obs:.4f}")
    print(f"  sign-flip permutations      : {B}")
    print(f"  one-sided p (skew > 0)      : {p_one_sided:.3g}")

def main() -> None:
    z_vals = load_z_scores(TSV_PATH)

    test_positive_asymmetry(z_vals)

    analytic = analytic_extreme_p(z_vals)
    mc = monte_carlo_extreme_p(z_vals, N_SIM, analytic["p_global"])

    print("Extreme-value test for max cross_run_z_score under N(0,1) null")
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


if __name__ == "__main__":
    main()
