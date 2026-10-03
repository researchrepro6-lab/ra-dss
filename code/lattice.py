"""
lattice.py -- exact evaluation of Q_t(A) for EVERY A in 2^E in O(k * n * 2^n).

For the oriented (or plain) pairwise separation structure with unit pair
weights, each positive-negative pair (p, n) is characterised by its separation
pattern s(p, n) in {0,1}^E, s_e = 1 iff parameter e meets the demand (p, n).
Grouping pairs by pattern gives counts c_s, and

    Q^{phi}(A) = sum_s c_s * phi(|s & A|).

For phi = min(., k) we need, for every A, the distribution of |s & A| truncated
at k.  Write g_A(t) = sum_s c_s t^{|s & A|}.  Since |s & A| = sum_i s_i A_i,
g = K^{(x) n} c with the 2x2 kernel K[A_i, s_i] = t^{A_i s_i}; the Kronecker
product is applied one coordinate at a time (a Yates / fast-zeta style pass),
with polynomial coefficients kept up to degree k and the top degree saturating.
Total cost O(k n 2^n) instead of O(2^n |P||N|).

Also provided: the separable-switching distance transform used by the exact
multi-period dynamic program (Theorem 5.3 in the paper).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def pattern_counts(pos_bits: np.ndarray, neg_bits: np.ndarray,
                   n: int) -> np.ndarray:
    """c_s = number of (p, n) pairs with separation pattern s.

    pos_bits : (P,) int codes of the positive objects' 'meets' vectors a_p
    neg_bits : (N,) int codes of the negative objects' vectors b_n
    Oriented separation: s = a_p & b_n.
    """
    up, cp = np.unique(pos_bits, return_counts=True)
    un, cn = np.unique(neg_bits, return_counts=True)
    S = (up[:, None] & un[None, :]).ravel()
    W = (cp[:, None] * cn[None, :]).ravel().astype(np.float64)
    return np.bincount(S, weights=W, minlength=1 << n)


def intersection_profile(c: np.ndarray, n: int, kmax: int) -> np.ndarray:
    """H[j, A] = sum_s c_s [ min(|s & A|, kmax) == j ],  j = 0..kmax."""
    N = 1 << n
    H = np.zeros((kmax + 1, N), dtype=np.float64)
    H[0] = c
    idx = np.arange(N)
    for i in range(n):
        bit = 1 << i
        lo = idx[(idx & bit) == 0]
        hi = lo | bit
        # before this pass, position lo/hi hold s_i = 0 / 1 (A_i undetermined)
        a0 = H[:, lo] + H[:, hi]            # A_i = 0: nothing shifts
        sh = np.zeros_like(H[:, hi])        # A_i = 1: s_i = 1 part shifts +1
        sh[1:] = H[:-1, hi]
        sh[kmax] += H[kmax, hi]             # saturate the top degree
        a1 = H[:, lo] + sh
        H[:, lo] = a0
        H[:, hi] = a1
    return H


def q_lattice_fast(c: np.ndarray, n: int, phi: Callable[[np.ndarray], np.ndarray],
                   kmax: Optional[int] = None) -> np.ndarray:
    """Q(A) for all A.  phi must be constant beyond kmax (e.g. min(., k))."""
    if kmax is None:
        kmax = n
    H = intersection_profile(c, n, kmax)
    w = phi(np.arange(kmax + 1)).astype(np.float64)
    return w @ H


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.int64)
    c = np.zeros_like(x)
    while np.any(x):
        c += x & 1
        x >>= 1
    return c


def mask_costs(cost: Sequence[float], n: int) -> np.ndarray:
    """c(A) for every mask; inf if A contains an unavailable parameter."""
    N = 1 << n
    out = np.zeros(N)
    idx = np.arange(N)
    for i in range(n):
        out = out + np.where((idx >> i) & 1, cost[i], 0.0)
    return out


def to_mask(A: Sequence[int]) -> int:
    m = 0
    for e in A:
        m |= 1 << int(e)
    return m


def from_mask(m: int, n: int) -> list:
    return [e for e in range(n) if (m >> e) & 1]


# ---------------------------------------------------------------------------
# exact multi-period DP via the separable distance transform
# ---------------------------------------------------------------------------
def distance_transform(V: np.ndarray, n: int, lam: float,
                       kplus: Sequence[float], kminus: Sequence[float]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """G(A) = max_{A'} V(A') - lam * kappa(A', A) with separable kappa.

    kappa(A', A) = sum_{e in A \\ A'} kplus[e] + sum_{e in A' \\ A} kminus[e].
    Computed by one relaxation pass per coordinate (exact because kappa is a
    sum of per-coordinate terms).  Returns G and the arg-max mask.
    """
    N = 1 << n
    G = V.copy()
    arg = np.arange(N)
    idx = np.arange(N)
    for i in range(n):
        bit = 1 << i
        lo = idx[(idx & bit) == 0]
        hi = lo | bit
        # target A has bit 0 and source A' has bit 1 -> deactivation of e_i
        cand_lo = G[hi] - lam * kminus[i]
        # target has bit 1, source has bit 0 -> activation of e_i
        cand_hi = G[lo] - lam * kplus[i]
        new_lo = np.where(cand_lo > G[lo], cand_lo, G[lo])
        arg_lo = np.where(cand_lo > G[lo], arg[hi], arg[lo])
        new_hi = np.where(cand_hi > G[hi], cand_hi, G[hi])
        arg_hi = np.where(cand_hi > G[hi], arg[lo], arg[hi])
        G[lo], G[hi] = new_lo, new_hi
        arg[lo], arg[hi] = arg_lo, arg_hi
    return G, arg


def exact_multiperiod(Qlat: Sequence[np.ndarray], feas: Sequence[np.ndarray],
                      n: int, lam: float,
                      kplus: Optional[Sequence[float]] = None,
                      kminus: Optional[Sequence[float]] = None,
                      init_mask: int = 0) -> Tuple[list, float]:
    """Exact optimum of J over all budget-feasible policies.

    Qlat[t] : Q_t over all masks; feas[t] : boolean feasibility of each mask.
    Complexity O(|T| n 2^n).
    """
    if kplus is None:
        kplus = [1.0] * n
    if kminus is None:
        kminus = [1.0] * n
    T = len(Qlat)
    N = 1 << n
    NEG = -np.inf
    # initial configuration as a point mass
    V0 = np.full(N, NEG)
    V0[init_mask] = 0.0
    G, arg = distance_transform(V0, n, lam, kplus, kminus)
    back = []
    V = np.where(feas[0], Qlat[0] + G, NEG)
    back.append(arg)
    for t in range(1, T):
        G, arg = distance_transform(V, n, lam, kplus, kminus)
        back.append(arg)
        V = np.where(feas[t], Qlat[t] + G, NEG)
    last = int(np.argmax(V))
    val = float(V[last])
    pol = [0] * T
    pol[T - 1] = last
    for t in range(T - 1, 0, -1):
        pol[t - 1] = int(back[t][pol[t]])
    return [from_mask(m, n) for m in pol], val


def switch_cost_masks(pol: Sequence[Sequence[int]], init: Sequence[int] = ()):
    prev = set(init)
    tot = 0
    for A in pol:
        tot += len(prev ^ set(A))
        prev = set(A)
    return tot
