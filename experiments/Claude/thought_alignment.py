import argparse
import math
import random
from collections import Counter
from typing import List, Tuple, Dict, Any

# Composition-aware Smith–Waterman (local) alignment with affine gaps and
# nonparametric (permutation) significance. Works for arbitrary alphabets.
# Inputs:
#   - Positional: query subject    (evaluated once)
#   - Optional: one or more --tsv files with 'query' and 'subject' columns
# Output:
#   - A SINGLE TSV file 'alignment_results.tsv' in the current directory
#     containing one row per pair from all provided inputs (CLI + TSVs).

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_OPEN = 6
GAP_EXTEND = 1
PSEUDOCOUNT = 0.5  # symmetric Dirichlet smoothing for per-sequence symbol probs

# ---------- basic probability & scoring ----------

def symbol_probs(seq: str, other_seq: str) -> Dict[str, float]:
    """Per-sequence unigram probabilities over the UNION alphabet, with smoothing."""
    counts = Counter(seq)
    alpha = set(seq) | set(other_seq)
    k = len(alpha) if alpha else 1
    total = sum(counts.get(c, 0) for c in alpha) + PSEUDOCOUNT * k
    return {c: (counts.get(c, 0) + PSEUDOCOUNT) / total for c in alpha}

def expected_match_share(pq: Dict[str, float], ps: Dict[str, float]) -> float:
    alpha = set(pq.keys()) | set(ps.keys())
    return sum(pq.get(a, 0.0) * ps.get(a, 0.0) for a in alpha)

def adjusted_score(raw_score: int, L: int, pq: Dict[str, float], ps: Dict[str, float]) -> float:
    """Center the substitution component by its composition-expected value; leave gaps as-is."""
    if L <= 0:
        return 0.0
    em = L * expected_match_share(pq, ps)
    expected_sub = MATCH_SCORE * em - MISMATCH_PENALTY * (L - em)
    return raw_score - expected_sub

# ---------- Smith–Waterman with affine gaps ----------

def smith_waterman_affine(q: str, s: str) -> Tuple[int, Tuple[int,int,int,int], Dict[str,int]]:
    """
    Returns:
        best_score,
        (q_start, q_end, s_start, s_end),
        stats dict with matches, mismatches, gap_opens, gap_extends, L (aligned pairs)
    """
    n, m = len(q), len(s)
    if n == 0 or m == 0:
        return 0, (0, 0, 0, 0), {"matches": 0, "mismatches": 0, "gap_opens": 0, "gap_extends": 0, "L": 0}

    # Matrices: H=best, E=gap in query (up moves), F=gap in subject (left moves)
    H = [[0]*(m+1) for _ in range(n+1)]
    E = [[0]*(m+1) for _ in range(n+1)]
    F = [[0]*(m+1) for _ in range(n+1)]

    best = 0
    best_i = 0
    best_j = 0

    for i in range(1, n+1):
        qi = q[i-1]
        Hi_1 = H[i-1]
        Ei_1 = E[i-1]
        Hi = H[i]
        Ei = E[i]
        Fi = F[i]
        for j in range(1, m+1):
            sj = s[j-1]
            sub = MATCH_SCORE if qi == sj else -MISMATCH_PENALTY

            # affine recurrences
            Ei[j] = max(Hi_1[j] - GAP_OPEN, Ei_1[j] - GAP_EXTEND)
            Fi[j] = max(Hi[j-1] - GAP_OPEN, Fi[j-1] - GAP_EXTEND)

            h_diag = Hi_1[j-1] + sub
            h = h_diag
            if Ei[j] > h: h = Ei[j]
            if Fi[j] > h: h = Fi[j]
            if h < 0: h = 0
            Hi[j] = h

            if h > best:
                best = h
                best_i = i
                best_j = j

    # Traceback
    i, j = best_i, best_j
    if best == 0:
        return 0, (0, 0, 0, 0), {"matches": 0, "mismatches": 0, "gap_opens": 0, "gap_extends": 0, "L": 0}

    # Determine starting state
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
            sub = MATCH_SCORE if q[i-1] == s[j-1] else -MISMATCH_PENALTY
            if H[i][j] == H[i-1][j-1] + sub:
                L += 1
                if sub > 0: matches += 1
                else: mismatches += 1
                i -= 1
                j -= 1
            elif H[i][j] == E[i][j]:
                state = "E"
            elif H[i][j] == F[i][j]:
                state = "F"
            else:
                break
        elif state == "E":
            # gap in query => move up
            if E[i][j] == E[i-1][j] - GAP_EXTEND:
                gap_extends += 1
                i -= 1
            elif E[i][j] == H[i-1][j] - GAP_OPEN:
                gap_opens += 1
                i -= 1
                state = "H"
            else:
                break
        else:  # state == "F"
            # gap in subject => move left
            if F[i][j] == F[i][j-1] - GAP_EXTEND:
                gap_extends += 1
                j -= 1
            elif F[i][j] == H[i][j-1] - GAP_OPEN:
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

# ---------- significance via permutation null ----------

def shuffle_preserving_counts(seq: str) -> str:
    arr = list(seq)
    random.shuffle(arr)
    return "".join(arr)

def auto_null_samples(n: int, m: int) -> int:
    # Keep total DP work roughly bounded; affine is costlier than linear
    area = n * m
    target = 2e7  # ~20 million cell updates budget
    b = int(max(20, min(2000, target / max(1, area))))
    return b

# ---------- per-pair evaluation ----------

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

# ---------- TSV I/O ----------

def read_tsv_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return pairs
        cols = [c.strip().lower() for c in header.rstrip("\n").split("\t")]
        try:
            qi = cols.index("query")
            si = cols.index("subject")
        except ValueError:
            raise SystemExit(f"ERROR: TSV '{path}' must have columns 'query' and 'subject' (tab-separated, header required).")
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if qi >= len(parts) or si >= len(parts):
                continue
            q = parts[qi]
            s = parts[si]
            pairs.append((q, s))
    return pairs

def write_results_tsv(rows: List[Dict[str, Any]], out_path: str = "alignment_results.tsv") -> None:
    header = [
        "source",
        "index",
        "query",
        "subject",
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
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            vals = [
                r["source"],
                str(r["index"]),
                r["query"],
                r["subject"],
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
            ]
            f.write("\t".join(vals) + "\n")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Pairwise composition-aware local alignment with affine gaps and permutation significance."
    )
    parser.add_argument("query", nargs="?", default=None, help="Query sequence (optional if --tsv is used)")
    parser.add_argument("subject", nargs="?", default=None, help="Subject sequence (optional if --tsv is used)")
    parser.add_argument("--tsv", nargs="*", default=[], help="One or more TSV files with 'query' and 'subject' columns")
    args = parser.parse_args()

    all_jobs: List[Tuple[str, int, str, str]] = []  # (source, index, q, s)

    # CLI pair
    if args.query is not None and args.subject is not None:
        all_jobs.append(("CLI", 1, args.query, args.subject))

    # TSV files
    for path in args.tsv:
        pairs = read_tsv_pairs(path)
        for idx, (q, s) in enumerate(pairs, start=1):
            all_jobs.append((path, idx, q, s))

    if not all_jobs:
        raise SystemExit("Provide (query subject) and/or one or more --tsv files with 'query' and 'subject' columns.")

    results_rows: List[Dict[str, Any]] = []
    for source, idx, q, s in all_jobs:
        res = evaluate_pair(q, s)
        row = {
            "source": source,
            "index": idx,
            "query": q,
            "subject": s,
            **res,
        }
        results_rows.append(row)

    out_path = "alignment_results.tsv"
    write_results_tsv(results_rows, out_path)
    print(f"Wrote {len(results_rows)} result(s) to {out_path}")

if __name__ == "__main__":
    main()
