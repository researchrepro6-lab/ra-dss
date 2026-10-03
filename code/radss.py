"""
radss.py -- Resource-Aware Dynamic Soft Sets: core structures and algorithms.

Implements:
  * RA-DSS slice / multi-slice containers
  * weighted discernibility functional Q_t
  * budget-feasible family utilities (for the non-collapse lemma)
  * cost-benefit greedy with singleton guard  (Alg. 1)
  * partial-enumeration greedy                (Alg. 2, Sviridenko-style)
  * exact solver by enumeration / DP over candidate pools (Alg. 3)
  * incremental warm-start greedy             (Alg. 4)

All objects are plain Python/NumPy so that every claim in the paper can be
checked by brute force on small instances.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

INF = float("inf")


# ----------------------------------------------------------------------------
# Slice: one time index t of an RA-DSS
# ----------------------------------------------------------------------------
@dataclass
class Slice:
    """One time-slice of an RA-DSS.

    n_obj   : |U|
    F       : dict e -> boolean array of length n_obj  (indicator of F_t(e))
    cost    : dict e -> positive float (INF encodes unavailability)
    budget  : B_t
    W       : symmetric (n_obj, n_obj) nonneg pair-weight matrix, zero diagonal
    """

    n_obj: int
    F: Dict[int, np.ndarray]
    cost: Dict[int, float]
    budget: float
    W: np.ndarray

    def __post_init__(self):
        assert self.W.shape == (self.n_obj, self.n_obj)
        assert np.allclose(self.W, self.W.T), "pair weights must be symmetric"
        assert np.allclose(np.diag(self.W), 0.0), "diagonal weights must vanish"
        self._params = sorted(self.F.keys())
        # separating-pair mask P_t(e): boolean (n_obj, n_obj), True iff e splits the pair
        self._P: Dict[int, np.ndarray] = {}
        for e, ind in self.F.items():
            ind = ind.astype(bool)
            sep = ind[:, None] ^ ind[None, :]
            np.fill_diagonal(sep, False)
            self._P[e] = sep

    # -- parameters ---------------------------------------------------------
    @property
    def params(self) -> List[int]:
        return list(self._params)

    def available(self) -> List[int]:
        return [e for e in self._params if self.cost[e] < INF]

    def affordable(self) -> List[int]:
        """D_t = {e : c_t(e) <= B_t} -- the individually affordable parameters."""
        return [e for e in self._params if self.cost[e] <= self.budget]

    # -- objective ----------------------------------------------------------
    def sep_mask(self, A: Iterable[int]) -> np.ndarray:
        """Union of separating-pair masks over A."""
        m = np.zeros((self.n_obj, self.n_obj), dtype=bool)
        for e in A:
            m |= self._P[e]
        return m

    def Q(self, A: Iterable[int]) -> float:
        """Weighted discernibility Q_t(A) = w( union_{e in A} P_t(e) ).

        Each unordered pair counted once, hence the 0.5 factor on the full matrix.
        """
        m = self.sep_mask(A)
        return 0.5 * float(np.sum(self.W[m]))

    def Q_total(self) -> float:
        return self.Q(self.available())

    def marginal(self, e: int, A: Iterable[int]) -> float:
        """Q_t(A + e) - Q_t(A), computed as w(P(e) \\ union_A P)."""
        cur = self.sep_mask(A)
        gain_mask = self._P[e] & ~cur
        return 0.5 * float(np.sum(self.W[gain_mask]))

    # -- feasibility --------------------------------------------------------
    def cost_of(self, A: Iterable[int]) -> float:
        return float(sum(self.cost[e] for e in A))

    def feasible(self, A: Iterable[int]) -> bool:
        return self.cost_of(A) <= self.budget + 1e-12

    def feasible_family(self) -> List[FrozenSet[int]]:
        """All budget-feasible subsets. Exponential -- small instances only."""
        av = self.available()
        out = []
        for r in range(len(av) + 1):
            for comb in itertools.combinations(av, r):
                if self.cost_of(comb) <= self.budget + 1e-12:
                    out.append(frozenset(comb))
        return out


# ----------------------------------------------------------------------------
# Non-collapse diagnostics (Lemma 3.5)
# ----------------------------------------------------------------------------
def family_is_power_set(family: Sequence[FrozenSet[int]]) -> bool:
    """True iff `family` equals P(A') for some set A' (i.e. is union-closed
    and downward-closed with a unique maximum)."""
    fam = set(family)
    if not fam:
        return False
    union_all = frozenset().union(*fam)
    if union_all not in fam:
        return False
    # P(A') has exactly 2^|A'| members
    return len(fam) == 2 ** len(union_all)


def budget_binds_on_affordable(sl: Slice) -> bool:
    """True iff sum_{e in D_t} c_t(e) > B_t, with D_t the affordable set."""
    return sl.cost_of(sl.affordable()) > sl.budget + 1e-12


# ----------------------------------------------------------------------------
# Algorithm 1: cost-benefit greedy with singleton guard
# ----------------------------------------------------------------------------
def greedy_density(sl: Slice, trace: bool = False):
    """Cost-benefit (density) greedy. Returns (set, trace).

    trace entries: (step, chosen, density, marginal, cached_densities)
    """
    A: List[int] = []
    remaining = set(sl.available())
    spent = 0.0
    tr = []
    step = 0
    while True:
        best, best_rho, best_gain = None, -1.0, 0.0
        densities = {}
        for e in sorted(remaining):
            if spent + sl.cost[e] > sl.budget + 1e-12:
                continue
            g = sl.marginal(e, A)
            rho = g / sl.cost[e]
            densities[e] = rho
            if rho > best_rho + 1e-15:
                best, best_rho, best_gain = e, rho, g
        if best is None or best_gain <= 1e-15:
            break
        A.append(best)
        remaining.discard(best)
        spent += sl.cost[best]
        step += 1
        if trace:
            tr.append((step, best, best_rho, best_gain, dict(densities)))
    return set(A), tr


def best_singleton(sl: Slice) -> Tuple[set, float]:
    best, bestv = set(), 0.0
    for e in sl.affordable():
        v = sl.Q([e])
        if v > bestv:
            best, bestv = {e}, v
    return best, bestv


def greedy_guarded(sl: Slice) -> set:
    """Alg. 1: better of (cost-benefit greedy, best affordable singleton)."""
    g, _ = greedy_density(sl)
    s, sv = best_singleton(sl)
    return g if sl.Q(g) >= sv else s


def greedy_partial_enum(sl: Slice, kappa: int = 3) -> set:
    """Alg. 2: enumerate all feasible seeds of size <= kappa, extend each by
    density greedy, return the best. Sviridenko-style."""
    av = sl.available()
    best, bestv = set(), 0.0
    for r in range(0, kappa + 1):
        for seed in itertools.combinations(av, r):
            if sl.cost_of(seed) > sl.budget + 1e-12:
                continue
            A = list(seed)
            spent = sl.cost_of(seed)
            remaining = set(av) - set(seed)
            while True:
                bb, brho, bg = None, -1.0, 0.0
                for e in sorted(remaining):
                    if spent + sl.cost[e] > sl.budget + 1e-12:
                        continue
                    g = sl.marginal(e, A)
                    rho = g / sl.cost[e]
                    if rho > brho + 1e-15:
                        bb, brho, bg = e, rho, g
                if bb is None or bg <= 1e-15:
                    break
                A.append(bb)
                remaining.discard(bb)
                spent += sl.cost[bb]
            v = sl.Q(A)
            if v > bestv:
                best, bestv = set(A), v
    return best


def greedy_cost_blind(sl: Slice) -> set:
    """Greedy on raw marginal gain, ignoring cost; stops when infeasible."""
    A: List[int] = []
    remaining = set(sl.available())
    spent = 0.0
    while True:
        best, bg = None, 0.0
        for e in sorted(remaining):
            if spent + sl.cost[e] > sl.budget + 1e-12:
                continue
            g = sl.marginal(e, A)
            if g > bg + 1e-15:
                best, bg = e, g
        if best is None:
            break
        A.append(best)
        remaining.discard(best)
        spent += sl.cost[best]
    return set(A)


def exact_best(sl: Slice) -> Tuple[set, float]:
    """Brute-force optimum over the feasible family. Small |E| only."""
    best, bestv = set(), -1.0
    for S in sl.feasible_family():
        v = sl.Q(S)
        if v > bestv:
            best, bestv = set(S), v
    return best, bestv


# ----------------------------------------------------------------------------
# Multi-slice with switching costs
# ----------------------------------------------------------------------------
@dataclass
class RADSS:
    slices: List[Slice]
    kappa_on: Dict[int, float] = field(default_factory=dict)   # activation cost
    kappa_off: Dict[int, float] = field(default_factory=dict)  # deactivation cost
    lam: float = 0.0                                           # switching weight

    def switch_cost(self, A_prev: Iterable[int], A_cur: Iterable[int]) -> float:
        A_prev, A_cur = set(A_prev), set(A_cur)
        on = sum(self.kappa_on.get(e, 1.0) for e in A_cur - A_prev)
        off = sum(self.kappa_off.get(e, 1.0) for e in A_prev - A_cur)
        return on + off

    def objective(self, policy: Sequence[Iterable[int]],
                  A0: Optional[Iterable[int]] = None) -> float:
        """sum_t Q_t(A_t) - lam * sum_t kappa(A_{t-1}, A_t)."""
        prev = set() if A0 is None else set(A0)
        tot = 0.0
        for sl, A in zip(self.slices, policy):
            tot += sl.Q(A) - self.lam * self.switch_cost(prev, A)
            prev = set(A)
        return tot

    def quality_only(self, policy: Sequence[Iterable[int]]) -> float:
        return float(sum(sl.Q(A) for sl, A in zip(self.slices, policy)))

    def total_switches(self, policy: Sequence[Iterable[int]],
                       A0: Optional[Iterable[int]] = None) -> float:
        prev = set() if A0 is None else set(A0)
        tot = 0.0
        for A in policy:
            tot += self.switch_cost(prev, A)
            prev = set(A)
        return tot

    # -- policies ----------------------------------------------------------
    def myopic(self, alg=greedy_guarded) -> List[set]:
        return [alg(sl) for sl in self.slices]

    def persistent(self, alg=greedy_guarded) -> List[set]:
        """One set, chosen to be feasible in every slice, reused throughout.

        Uses a pooled slice whose cost is the max over slices (so the chosen
        set is feasible everywhere) and whose weights/F come from slice 0.
        """
        # candidate pool: the myopic solutions plus their union-prefixes
        cands = self._candidate_pool(extra_persistent=False)
        common = [S for S in cands if all(sl.feasible(S) for sl in self.slices)]
        if not common:
            common = [frozenset()]
        best, bestv = frozenset(), -INF
        for S in common:
            v = self.objective([S] * len(self.slices))
            if v > bestv:
                best, bestv = S, v
        return [set(best)] * len(self.slices)

    def _candidate_pool(self, extra_persistent: bool = True) -> List[FrozenSet[int]]:
        """Union over slices of: greedy prefixes, guarded greedy, singletons, {}."""
        pool = {frozenset()}
        for sl in self.slices:
            _, tr = greedy_density(sl)
            pref: List[int] = []
            for (_, e, _, _, _) in tr:
                pref.append(e)
                pool.add(frozenset(pref))
            pool.add(frozenset(greedy_guarded(sl)))
            for e in sl.affordable():
                pool.add(frozenset([e]))
        return sorted(pool, key=lambda s: (len(s), sorted(s)))

    def dp_restricted(self, pool: Optional[Sequence[FrozenSet[int]]] = None,
                      A0: Optional[Iterable[int]] = None
                      ) -> Tuple[List[set], float]:
        """Alg. 3: exact DP over a restricted candidate pool.

        Returns (policy, objective). Optimal among policies with A_t in pool
        and A_t feasible at t.
        """
        if pool is None:
            pool = self._candidate_pool()
        pool = [frozenset(S) for S in pool]
        Tn = len(self.slices)
        # per-slice feasible candidates + their quality
        cand: List[List[FrozenSet[int]]] = []
        qual: List[Dict[FrozenSet[int], float]] = []
        for sl in self.slices:
            cs = [S for S in pool if sl.feasible(S)]
            if not cs:
                cs = [frozenset()]
            cand.append(cs)
            qual.append({S: sl.Q(S) for S in cs})

        start = frozenset() if A0 is None else frozenset(A0)
        # value[t][S]
        value: List[Dict[FrozenSet[int], float]] = [dict() for _ in range(Tn)]
        back: List[Dict[FrozenSet[int], Optional[FrozenSet[int]]]] = [dict() for _ in range(Tn)]
        for S in cand[0]:
            value[0][S] = qual[0][S] - self.lam * self.switch_cost(start, S)
            back[0][S] = None
        for t in range(1, Tn):
            for S in cand[t]:
                bestv, bestp = -INF, None
                qs = qual[t][S]
                for P in cand[t - 1]:
                    v = value[t - 1][P] + qs - self.lam * self.switch_cost(P, S)
                    if v > bestv:
                        bestv, bestp = v, P
                value[t][S] = bestv
                back[t][S] = bestp
        # recover
        Slast, bestv = None, -INF
        for S, v in value[Tn - 1].items():
            if v > bestv:
                Slast, bestv = S, v
        policy = [set()] * Tn
        S = Slast
        for t in range(Tn - 1, -1, -1):
            policy[t] = set(S)
            S = back[t][S]
        return policy, bestv

    def dp_exact_full(self, A0: Optional[Iterable[int]] = None
                      ) -> Tuple[List[set], float]:
        """Exact DP over the full feasible family. Exponential; validation only."""
        pool = set()
        for sl in self.slices:
            pool |= set(sl.feasible_family())
        return self.dp_restricted(sorted(pool, key=lambda s: (len(s), sorted(s))), A0)


# ----------------------------------------------------------------------------
# Algorithm 4: incremental warm-start greedy  (Theorem 5.6)
# ----------------------------------------------------------------------------
def incremental_greedy(sl_new: Slice,
                       prev_trace,
                       delta: Iterable[int]):
    """Warm-start the density greedy at slice t+1 from the trace at slice t.

    Returns (A_new, j_star, n_marginal_evals).

    The trace of slice t caches, for every step i, the density of every
    candidate considered. Elements outside `delta` have unchanged densities
    *given the same prefix*, so at step i we only recompute densities for
    e in delta and compare against the cached values; the first step whose
    argmax changes is j_star. Steps 1..j_star-1 provably reproduce the old
    choices (Lemma: prefix determinism).
    """
    delta = set(delta)
    A: List[int] = []
    spent = 0.0
    evals = 0
    j_star = None

    for (step, chosen, rho_old, gain_old, densities_old) in prev_trace:
        # affordability of the old candidate set may have shifted if costs of
        # already-selected elements changed; detect that first.
        if any(e in delta for e in A):
            j_star = step
            break
        # recompute only the changed parameters
        best, best_rho = chosen, None
        # old cached density for `chosen` is valid only if chosen not in delta
        if chosen in delta:
            j_star = step
            break
        if spent + sl_new.cost[chosen] > sl_new.budget + 1e-12:
            j_star = step
            break
        best_rho = densities_old.get(chosen, None)
        if best_rho is None:
            j_star = step
            break
        changed_beats = False
        for f in sorted(delta):
            if f in A:
                continue
            if spent + sl_new.cost[f] > sl_new.budget + 1e-12:
                continue
            g = sl_new.marginal(f, A)
            evals += 1
            if g / sl_new.cost[f] > best_rho + 1e-15:
                changed_beats = True
                break
        if changed_beats:
            j_star = step
            break
        # safe to reuse the old choice
        A.append(chosen)
        spent += sl_new.cost[chosen]

    if j_star is None:
        j_star = len(prev_trace) + 1

    # recompute from j_star onwards
    remaining = set(sl_new.available()) - set(A)
    while True:
        best, brho, bg = None, -1.0, 0.0
        for e in sorted(remaining):
            if spent + sl_new.cost[e] > sl_new.budget + 1e-12:
                continue
            g = sl_new.marginal(e, A)
            evals += 1
            rho = g / sl_new.cost[e]
            if rho > brho + 1e-15:
                best, brho, bg = e, rho, g
        if best is None or bg <= 1e-15:
            break
        A.append(best)
        remaining.discard(best)
        spent += sl_new.cost[best]
    return set(A), j_star, evals


def full_greedy_evals(sl: Slice) -> int:
    """Number of marginal evaluations performed by the from-scratch greedy."""
    A: List[int] = []
    remaining = set(sl.available())
    spent = 0.0
    evals = 0
    while True:
        best, brho, bg = None, -1.0, 0.0
        for e in sorted(remaining):
            if spent + sl.cost[e] > sl.budget + 1e-12:
                continue
            g = sl.marginal(e, A)
            evals += 1
            rho = g / sl.cost[e]
            if rho > brho + 1e-15:
                best, brho, bg = e, rho, g
        if best is None or bg <= 1e-15:
            break
        A.append(best)
        remaining.discard(best)
        spent += sl.cost[best]
    return evals


# ----------------------------------------------------------------------------
# helpers for building random test instances
# ----------------------------------------------------------------------------
def random_slice(rng: np.random.Generator, n_obj: int, n_par: int,
                 budget_frac: float = 0.4,
                 cost_lo: float = 1.0, cost_hi: float = 10.0,
                 p_lo: float = 0.15, p_hi: float = 0.85,
                 weight_mode: str = "binary",
                 n_unavail: int = 0) -> Slice:
    F = {}
    for e in range(n_par):
        p = rng.uniform(p_lo, p_hi)
        F[e] = (rng.random(n_obj) < p)
    cost = {e: float(np.round(rng.uniform(cost_lo, cost_hi), 2)) for e in range(n_par)}
    for e in rng.choice(n_par, size=min(n_unavail, n_par), replace=False):
        cost[int(e)] = INF
    if weight_mode == "binary":
        lab = rng.integers(0, 2, size=n_obj)
        W = (lab[:, None] != lab[None, :]).astype(float)
    elif weight_mode == "uniform":
        W = np.ones((n_obj, n_obj))
    else:
        W = rng.random((n_obj, n_obj))
        W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    avail_costs = [c for c in cost.values() if c < INF]
    budget = float(np.round(budget_frac * sum(avail_costs), 2))
    return Slice(n_obj=n_obj, F=F, cost=cost, budget=budget, W=W)
