from __future__ import annotations

from typing import Dict, Tuple

MATCH_SCORE = 2
MISMATCH_PENALTY = 1
GAP_OPEN = 6
GAP_EXTEND = 1


def smith_waterman_affine(q: str, s: str) -> Tuple[int, Tuple[int, int, int, int], Dict[str, int]]:
    """Return affine-gap Smith-Waterman score and statistics.

    This is a lightweight copy extracted from the previous custom_utils utilities
    so reward functions that rely on fuzzy string matching can continue to import
    it without pulling in the deleted trainer code.
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
            h = max(h_diag, Ei[j], Fi[j], 0)
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

    matches = mismatches = gap_opens = gap_extends = 0
    while i > 0 and j > 0 and H[i][j] > 0:
        if H[i][j] == H[i - 1][j - 1] + (MATCH_SCORE if q[i - 1] == s[j - 1] else -MISMATCH_PENALTY):
            if q[i - 1] == s[j - 1]:
                matches += 1
            else:
                mismatches += 1
            i -= 1
            j -= 1
        elif H[i][j] == E[i][j]:
            gap_opens += 1
            i -= 1
        elif H[i][j] == F[i][j]:
            gap_extends += 1
            j -= 1
        else:
            break

    stats = {
        "matches": matches,
        "mismatches": mismatches,
        "gap_opens": gap_opens,
        "gap_extends": gap_extends,
        "L": matches + mismatches + gap_opens + gap_extends,
    }
    return best, (i, best_i, j, best_j), stats


__all__ = ["smith_waterman_affine"]
