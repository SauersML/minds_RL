import csv
import os
from typing import Any, Dict, List, Tuple

SUMMARY_TSV = os.path.join("sonnet_cot_experiment", "summary.tsv")
RUNS_TSV = os.path.join("sonnet_cot_experiment", "runs.tsv")
SCORE_MATRIX_TSV = os.path.join("sonnet_cot_experiment", "score_matrix.tsv")

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_OPEN = 6
GAP_EXTEND = 1


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


def select_runs(summary_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    run_index = 0

    for row in summary_rows:
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

        record = {
            "run_index": run_index,
            "condition": cond,
            "secret": secret,
            "guess": guess,
        }
        selected.append(record)
        run_index += 1

    return selected


def smith_waterman_affine(
    q: str, s: str
) -> Tuple[int, Tuple[int, int, int, int], Dict[str, int]]:
    n, m = len(q), len(s)
    if n == 0 or m == 0:
        stats_zero = {
            "matches": 0,
            "mismatches": 0,
            "gap_opens": 0,
            "gap_extends": 0,
            "L": 0,
        }
        return 0, (0, 0, 0, 0), stats_zero

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

    i = best_i
    j = best_j
    if best == 0:
        stats_zero = {
            "matches": 0,
            "mismatches": 0,
            "gap_opens": 0,
            "gap_extends": 0,
            "L": 0,
        }
        return 0, (0, 0, 0, 0), stats_zero

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


def write_runs_tsv(runs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(RUNS_TSV), exist_ok=True)
    with open(RUNS_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "run_index",
            "condition",
            "secret_sequence",
            "guess_sequence",
            "within_run_score",
        ]
        writer.writerow(header)

        for r in runs:
            secret = r["secret"]
            guess = r["guess"]
            score, _, _ = smith_waterman_affine(secret, guess)
            writer.writerow(
                [
                    r["run_index"],
                    r["condition"],
                    secret,
                    guess,
                    score,
                ]
            )


def write_score_matrix_tsv(runs: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(SCORE_MATRIX_TSV), exist_ok=True)
    with open(SCORE_MATRIX_TSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = [
            "secret_run_index",
            "guess_run_index",
            "score",
            "aligned_pairs",
            "matches",
            "mismatches",
            "gap_opens",
            "gap_extends",
            "query_start",
            "query_end",
            "subject_start",
            "subject_end",
        ]
        writer.writerow(header)

        for r_i in runs:
            i = r_i["run_index"]
            secret = r_i["secret"]
            for r_j in runs:
                j = r_j["run_index"]
                guess = r_j["guess"]
                score, coords, stats = smith_waterman_affine(secret, guess)
                writer.writerow(
                    [
                        i,
                        j,
                        score,
                        stats["L"],
                        stats["matches"],
                        stats["mismatches"],
                        stats["gap_opens"],
                        stats["gap_extends"],
                        coords[0],
                        coords[1],
                        coords[2],
                        coords[3],
                    ]
                )

def main() -> None:
    summary_rows = load_summary_rows(SUMMARY_TSV)
    print(f"Loaded {len(summary_rows)} rows from {SUMMARY_TSV}")

    runs = select_runs(summary_rows)
    print(f"Selected {len(runs)} runs after filtering")

    if not runs:
        print("No usable runs found; nothing to write.")
        return

    write_runs_tsv(runs)
    print(f"Wrote runs.tsv with {len(runs)} rows to {RUNS_TSV}")

    write_score_matrix_tsv(runs)
    print(
        f"Wrote score_matrix.tsv with {len(runs) * len(runs)} rows to {SCORE_MATRIX_TSV}"
    )


if __name__ == "__main__":
    main()
