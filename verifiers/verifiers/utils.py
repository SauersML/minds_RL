from __future__ import annotations

from typing import Dict, Tuple

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_OPEN = 6
GAP_EXTEND = 1


def smith_waterman_affine(q: str, s: str) -> Tuple[int, Tuple[int, int, int, int], Dict[str, int]]:
    """Compute local alignment using Smith-Waterman with affine gaps.

    This implementation is adapted for lightweight reuse across environments.
    It returns the best alignment score, alignment boundaries, and statistics
    about the alignment path.
    """

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
            h = max(h_diag, Ei[j], Fi[j])
            Hi[j] = h if h > 0 else 0

            if Hi[j] > best:
                best = Hi[j]
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

    state = "H"
    if H[i][j] == E[i][j]:
        state = "E"
    elif H[i][j] == F[i][j]:
        state = "F"

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


__all__ = ["smith_waterman_affine"]
