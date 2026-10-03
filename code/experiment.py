"""
experiment.py -- case-study utilities used by experiment3.py (warm-start
re-selection, job 'inc') and a compact single-replicate grid used during development.

Selection is performed INSIDE each cross-validation fold on training-fold data
only; the held-out fold scores the downstream classifier, whose ridge strength
is itself chosen by an inner cross-validation on the training fold. Costs and
availability are known a priori and are not estimated.

Outputs (results/):
  main_grid.csv        budget sweep x method x replicate x fold
  selection_map.json   which tests each method picks in each window
  lambda_sweep.csv     switching-weight sweep
  incremental.csv      warm-start greedy evaluation counts
  approx_gap.csv       greedy vs exact optimum on the real slices
  headroom.csv         per-window optimum vs best reusable fixed panel
  saturation.csv       Q/Q(E) against redundancy level k
  summary.json         headline numbers quoted in the text
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casestudy import (AVAIL, COST, FULL_PANEL_COST, K_RED, KEYS, L2_GRID,
                       N_PAR, N_WIN, TESTS, Cohort, FastSlice, _greedy_core,
                       auroc, availability_matrix, build_design,
                       candidate_pool, cost_matrix, fit_logreg, fit_logreg_cv,
                       kfold, make_cohort, phi_min, predict,
                       sel_cost_blind_greedy, sel_density_greedy, sel_dp,
                       sel_exact, sel_guarded_greedy, sel_partial_enum,
                       sel_random, sel_static_pooled, switch_cost)
from radss import INF

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
os.makedirs(RESULTS, exist_ok=True)

N_PATIENTS = 3000
N_REP = 4
N_FOLD = 5
OUTCOME_NOISE = 2.8
BUDGET_FRACS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20,
                0.25, 0.30, 0.40, 0.60, 0.80, 1.00]
GAMMA_DEFAULT = 0.20   # switching price in units of mean single-parameter quality
N_RAND_DRAW = 6

METHODS = ["Full panel", "Random (budget)", "Cost-blind greedy",
           "Static reduct", "RA-DSS myopic", "RA-DSS myopic+enum",
           "RA-DSS DP"]


def make_slices(coh: Cohort, tr: np.ndarray, budget: float,
                sub_seed: int = 0) -> List[FastSlice]:
    C = cost_matrix()
    return [FastSlice(coh.X, coh.y, tr, w, C, budget, sub_seed=sub_seed)
            for w in range(N_WIN)]


def realised_cost(policy: Sequence[Sequence[int]]) -> float:
    return float(sum(COST[e] for A in policy for e in A))


def evaluate(coh: Cohort, tr: np.ndarray, te: np.ndarray,
             policy: Sequence[Sequence[int]], seed: int = 0) -> float:
    Dtr = build_design(coh.X[tr], policy)
    Dte = build_design(coh.X[te], policy)
    beta = fit_logreg_cv(Dtr, coh.y[tr], seed=seed)
    return auroc(coh.y[te], predict(beta, Dte))


# ---------------------------------------------------------------------------
def run_main_grid():
    rows: List[dict] = []
    selmap: Dict[str, dict] = {}
    t0 = time.time()
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        folds = kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
        for bf in BUDGET_FRACS:
            budget = bf * FULL_PANEL_COST
            for fi, (tr, te) in enumerate(folds):
                slices = make_slices(coh, tr, budget, sub_seed=rep)
                qmax = sum(sl.Q_max() for sl in slices)
                rng = np.random.default_rng(9000 + 31 * rep + fi)

                pols: Dict[str, List[List[int]]] = {
                    "Full panel": [sorted(sl.available()) for sl in slices],
                    "Cost-blind greedy": [sel_cost_blind_greedy(sl) for sl in slices],
                    "Static reduct": sel_static_pooled(slices),
                    "RA-DSS myopic": [sel_guarded_greedy(sl) for sl in slices],
                    "RA-DSS myopic+enum": [sel_partial_enum(sl, kappa=1) for sl in slices],
                }
                singles = [sl.Q([e]) for sl in slices for e in sl.available()]
                qbar = float(np.mean(singles)) if singles else 1.0
                pols["RA-DSS DP"] = sel_dp(slices, lam=GAMMA_DEFAULT * qbar)[0]

                draws = [sel_random(slices[w], rng, N_RAND_DRAW)
                         for w in range(N_WIN)]
                r_a, r_c, r_n, r_q = [], [], [], []
                for d in range(N_RAND_DRAW):
                    pol = [draws[w][d] for w in range(N_WIN)]
                    r_a.append(evaluate(coh, tr, te, pol, seed=fi))
                    r_c.append(realised_cost(pol))
                    r_n.append(sum(len(a) for a in pol))
                    r_q.append(sum(sl.Q(pol[w]) for w, sl in enumerate(slices)))

                for m in METHODS:
                    if m == "Random (budget)":
                        rows.append(dict(
                            rep=rep, fold=fi, budget_frac=bf, budget=budget,
                            method=m, auroc=float(np.mean(r_a)),
                            cost=float(np.mean(r_c)),
                            n_tests=float(np.mean(r_n)),
                            switches=float("nan"),
                            q_frac=float(np.mean(r_q)) / qmax if qmax else float("nan")))
                        continue
                    pol = pols[m]
                    qtot = sum(sl.Q(pol[w]) for w, sl in enumerate(slices))
                    rows.append(dict(
                        rep=rep, fold=fi, budget_frac=bf, budget=budget,
                        method=m, auroc=evaluate(coh, tr, te, pol, seed=fi),
                        cost=realised_cost(pol),
                        n_tests=sum(len(x) for x in pol),
                        switches=sum(switch_cost(pol[w - 1] if w else [], pol[w])
                                     for w in range(N_WIN)),
                        q_frac=qtot / qmax if qmax else float("nan")))
                if rep == 0 and fi == 0 and abs(bf - 0.10) < 1e-12:
                    for m in METHODS:
                        if m == "Random (budget)":
                            continue
                        selmap[m] = {f"t{w}": [KEYS[e] for e in pols[m][w]]
                                     for w in range(N_WIN)}
            print(f"  rep {rep}  budget {bf:.3f}   ({time.time()-t0:.0f}s)",
                  flush=True)
    return rows, selmap


# ---------------------------------------------------------------------------
def q_lattice(sl: FastSlice) -> np.ndarray:
    """Q_t(S) for every S in 2^E, indexed by bitmask.

    Depth-first traversal maintaining the separation-multiplicity matrix D
    incrementally, so each of the 2^{|E|} nodes costs one matrix add and one
    reduction rather than a rebuild. Q does not depend on the budget, so this
    lattice is computed once per (replicate, window) and reused across the
    whole budget sweep.
    """
    n = N_PAR
    out = np.zeros(1 << n, dtype=np.float64)
    D0 = np.zeros((sl.P, sl.N), dtype=np.int16)

    def rec(idx: int, mask: int, D: np.ndarray):
        if idx == n:
            return
        for e in range(idx, n):
            D2 = D + sl.M[e]
            m2 = mask | (1 << e)
            out[m2] = sl.Q_from_D(D2)
            rec(e + 1, m2, D2)

    rec(0, 0, D0)
    return out


def _masks_cost() -> np.ndarray:
    c = np.zeros(1 << N_PAR)
    for m in range(1 << N_PAR):
        s = 0.0
        mm = m
        while mm:
            b = mm & -mm
            s += COST[b.bit_length() - 1]
            mm ^= b
        c[m] = s
    return c


def run_headroom():
    """Maximum achievable gain from temporal adaptation: sum of per-window
    optima against the best single panel reusable in every window."""
    rows = []
    mcost = _masks_cost()
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        tr = np.arange(N_PATIENTS)
        slices = make_slices(coh, tr, 1e9, sub_seed=rep)
        lat = [q_lattice(sl) for sl in slices]
        # availability mask per window
        availmask = []
        for sl in slices:
            bad = 0
            for e in range(N_PAR):
                if sl.cost[e] == INF:
                    bad |= (1 << e)
            availmask.append(bad)
        allmask = np.arange(1 << N_PAR)
        pooled = np.sum(lat, axis=0)
        for bf in BUDGET_FRACS:
            B = bf * FULL_PANEL_COST
            feas_cost = mcost <= B + 1e-9
            per, qsum = [], 0.0
            for w, sl in enumerate(slices):
                ok = feas_cost & ((allmask & availmask[w]) == 0)
                v = np.where(ok, lat[w], -np.inf)
                b = int(np.argmax(v))
                per.append(b)
                qsum += float(v[b])
            okall = feas_cost.copy()
            for w in range(N_WIN):
                okall &= ((allmask & availmask[w]) == 0)
            vf = np.where(okall, pooled, -np.inf)
            bfix = int(np.argmax(vf))
            bv = float(vf[bfix])

            def keys(m):
                return ",".join(KEYS[e] for e in range(N_PAR) if m >> e & 1)

            rows.append(dict(rep=rep, budget_frac=bf,
                             q_adaptive=qsum, q_fixed=bv,
                             gain_pct=100.0 * (qsum - bv) / bv if bv > 0 else 0.0,
                             n_distinct=len(set(per)),
                             fixed_panel=keys(bfix).replace(",", "|"),
                             adaptive_panels=" ; ".join(keys(m) for m in per)))
        print(f"  headroom rep {rep} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
def run_approx_gap():
    """Greedy variants against the exact optimum on the real slices.

    The exact optimum is read off the cached objective lattice (see
    q_lattice); the greedy variants are re-run at every budget level.
    """
    rows = []
    mcost = _masks_cost()
    allmask = np.arange(1 << N_PAR)
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        tr = np.arange(N_PATIENTS)
        base = make_slices(coh, tr, 1e9, sub_seed=rep)
        lat = [q_lattice(sl) for sl in base]
        availmask = []
        for sl in base:
            bad = 0
            for e in range(N_PAR):
                if sl.cost[e] == INF:
                    bad |= (1 << e)
            availmask.append(bad)
        for bf in BUDGET_FRACS:
            B = bf * FULL_PANEL_COST
            feas = mcost <= B + 1e-9
            slices = make_slices(coh, tr, B, sub_seed=rep)
            for w, sl in enumerate(slices):
                ok = feas & ((allmask & availmask[w]) == 0)
                opt = float(np.max(np.where(ok, lat[w], -np.inf)))
                if opt <= 0:
                    continue
                rows.append(dict(
                    rep=rep, budget_frac=bf, window=w, opt=opt,
                    r_density=sl.Q(sel_density_greedy(sl)) / opt,
                    r_guarded=sl.Q(sel_guarded_greedy(sl)) / opt,
                    r_enum1=sl.Q(sel_partial_enum(sl, kappa=1)) / opt,
                    r_enum2=sl.Q(sel_partial_enum(sl, kappa=2)) / opt,
                    r_blind=sl.Q(sel_cost_blind_greedy(sl)) / opt))
        print(f"  approx gap rep {rep} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
GAMMAS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def run_lambda_sweep():
    """Switching-weight sweep, scale-free.

    The raw weight lambda is not comparable across instances because Q is an
    extensive quantity. We therefore sweep gamma = lambda / Qbar, where Qbar is
    the mean single-parameter quality over the horizon: gamma is then the price
    of one panel change expressed in units of what one parameter contributes.
    """
    rows = []
    bf = 0.10
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        folds = kfold(N_PATIENTS, N_FOLD, seed=500 + rep)
        for fi, (tr, te) in enumerate(folds):
            slices = make_slices(coh, tr, bf * FULL_PANEL_COST, sub_seed=rep)
            pool = candidate_pool(slices)
            qmax = sum(sl.Q_max() for sl in slices)
            singles = [sl.Q([e]) for sl in slices for e in sl.available()]
            qbar = float(np.mean(singles)) if singles else 1.0
            for gam in GAMMAS:
                lam = gam * qbar
                pol, obj = sel_dp(slices, lam=lam, pool=pool)
                rows.append(dict(
                    rep=rep, fold=fi, gamma=gam, lam=lam, qbar=qbar,
                    auroc=evaluate(coh, tr, te, pol, seed=fi),
                    switches=sum(switch_cost(pol[w - 1] if w else [], pol[w])
                                 for w in range(N_WIN)),
                    n_distinct=len(set(tuple(p) for p in pol)),
                    q_frac=sum(sl.Q(pol[w]) for w, sl in enumerate(slices)) / qmax,
                    cost=realised_cost(pol),
                    n_tests=sum(len(x) for x in pol)))
        print(f"  lambda sweep rep {rep} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
def incremental_fast(sl: FastSlice, prev_trace, delta):
    """Warm-start greedy (Alg. 4).

    Evaluation accounting: a *parameter-gain evaluation* is the computation of
    one marginal gain Q(A+e)-Q(A) for one parameter e. Both this routine and
    full_greedy_evals below compute and charge exactly that, one parameter at a
    time, so the two counts are directly comparable. (Charging only |Delta|
    while evaluating the verified prefix in batch would understate the warm
    start's work.)
    """
    delta = set(delta)
    A: List[int] = []
    spent = 0.0
    evals = 0
    j_star = None
    for (step, chosen, rho_old, gain_old, dens_old) in prev_trace:
        if any(e in delta for e in A) or chosen in delta:
            j_star = step
            break
        if spent + sl.cost[chosen] > sl.budget + 1e-9:
            j_star = step
            break
        base = dens_old.get(chosen)
        if base is None:
            j_star = step
            break
        beaten = False
        for f in sorted(delta):
            if f in A or sl.cost[f] == INF:
                continue
            if spent + sl.cost[f] > sl.budget + 1e-9:
                continue
            g = sl.marginal(f, A)          # one genuine gain evaluation
            evals += 1
            if g / sl.cost[f] > base + 1e-9:
                beaten = True
                break
        if beaten:
            j_star = step
            break
        A.append(chosen)
        spent += sl.cost[chosen]
    if j_star is None:
        j_star = len(prev_trace) + 1
    # complete from scratch beyond the verified prefix
    tail, ev_tail = _greedy_count(sl, seed=A, spent=spent)
    return sorted(tail), j_star, evals + ev_tail


def _greedy_count(sl: FastSlice, seed: Sequence[int] = (), spent: float = 0.0):
    """Density-greedy loop charging one evaluation per parameter considered."""
    A = list(seed)
    rem = set(sl.available()) - set(A)
    evals = 0
    while rem:
        best, brho, bg = None, -1.0, 0.0
        for e in sorted(rem):
            if spent + sl.cost[e] > sl.budget + 1e-9:
                continue
            g = sl.marginal(e, A)
            evals += 1
            rho = g / sl.cost[e]
            if rho > brho + 1e-12:
                best, brho, bg = e, rho, g
        if best is None or bg <= 1e-12:
            break
        A.append(best)
        rem.discard(best)
        spent += sl.cost[best]
    return A, evals


def full_greedy_evals(sl: FastSlice) -> Tuple[List[int], int]:
    A, ev = _greedy_count(sl)
    return sorted(A), ev


def run_incremental():
    """Two re-selection regimes.

    mode='cost-only': window w's data with window (w-1)'s availability, so
      Q_{t+1}=Q_t and only the cost vector moves. This is exactly the setting of
      Theorem 5.6 and the setting in which its guarantee applies.
    mode='true': the actual consecutive windows t=w-1 -> t=w of the case study,
      in which the regime shift moves F_t as well, so Theorem 5.6's hypothesis
      Q_{t+1}=Q_t FAILS. The warm start is then a heuristic and we report how
      often it still reproduces the from-scratch selection.
    """
    rows = []
    C = cost_matrix()
    Av = availability_matrix()
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        tr = np.arange(N_PATIENTS)
        for bf in BUDGET_FRACS:
            budget = bf * FULL_PANEL_COST
            for w in range(1, N_WIN):
                delta = {j for j in range(N_PAR) if Av[w - 1, j] != Av[w, j]}
                sl_new = FastSlice(coh.X, coh.y, tr, w, C, budget, sub_seed=rep)

                # (a) cost-only perturbation: Theorem 5.6 hypotheses hold
                sl_prev = FastSlice(coh.X, coh.y, tr, w, C, budget, sub_seed=rep)
                sl_prev.cost = {j: (COST[j] if Av[w - 1, j] else INF)
                                for j in range(N_PAR)}
                _, tr_a = sel_density_greedy(sl_prev, trace=True)
                inc_a, js_a, ev_a = incremental_fast(sl_new, tr_a, delta)

                # (b) the true window-to-window transition: Q also moves
                sl_true = FastSlice(coh.X, coh.y, tr, w - 1, C, budget,
                                    sub_seed=rep)
                _, tr_b = sel_density_greedy(sl_true, trace=True)
                inc_b, js_b, ev_b = incremental_fast(sl_new, tr_b, delta)

                ref, ev_full = full_greedy_evals(sl_new)
                rows.append(dict(rep=rep, budget_frac=bf, window=w,
                                 delta_size=len(delta),
                                 j_star=js_a, evals_inc=ev_a,
                                 evals_full=ev_full, agree=int(inc_a == ref),
                                 j_star_true=js_b, evals_inc_true=ev_b,
                                 agree_true=int(inc_b == ref)))
        print(f"  incremental rep {rep} done", flush=True)
    return rows


# ---------------------------------------------------------------------------
def run_saturation():
    rows = []
    C = cost_matrix()
    for rep in range(N_REP):
        coh = make_cohort(n=N_PATIENTS, seed=100 + rep,
                          outcome_noise=OUTCOME_NOISE)
        tr = np.arange(N_PATIENTS)
        rng = np.random.default_rng(3000 + rep)
        for k in (1, 2, 3, 4):
            sl = FastSlice(coh.X, coh.y, tr, 0, C, 1e9,
                           phi=phi_min(k), sub_seed=rep)
            qmax = sl.Q(sl.available())
            for size in (1, 2, 3, 4, 5, 6, 8, 10, 13):
                vals = []
                for _ in range(20):
                    A = list(rng.choice(sl.available(),
                                        size=min(size, len(sl.available())),
                                        replace=False))
                    vals.append(sl.Q(A) / qmax)
                rows.append(dict(rep=rep, k=k, size=size,
                                 mean=float(np.mean(vals)),
                                 lo=float(np.min(vals)),
                                 hi=float(np.max(vals)),
                                 spread=float(np.max(vals) - np.min(vals))))
    return rows


# ---------------------------------------------------------------------------
def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            out = []
            for k in keys:
                v = r[k]
                if isinstance(v, float) and math.isnan(v):
                    out.append("nan")
                elif isinstance(v, str) and ("," in v):
                    out.append('"' + v + '"')
                else:
                    out.append(str(v))
            f.write(",".join(out) + "\n")
    print(f"  wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "main"):
        print("=== main grid ===", flush=True)
        rows, selmap = run_main_grid()
        write_csv(os.path.join(RESULTS, "main_grid.csv"), rows)
        with open(os.path.join(RESULTS, "selection_map.json"), "w") as f:
            json.dump(selmap, f, indent=2)
    if which in ("all", "headroom"):
        print("=== headroom ===", flush=True)
        write_csv(os.path.join(RESULTS, "headroom.csv"), run_headroom())
    if which in ("all", "gap"):
        print("=== approximation gap ===", flush=True)
        write_csv(os.path.join(RESULTS, "approx_gap.csv"), run_approx_gap())
    if which in ("all", "lambda"):
        print("=== lambda sweep ===", flush=True)
        write_csv(os.path.join(RESULTS, "lambda_sweep.csv"), run_lambda_sweep())
    if which in ("all", "inc"):
        print("=== incremental ===", flush=True)
        write_csv(os.path.join(RESULTS, "incremental.csv"), run_incremental())
    if which in ("all", "sat"):
        print("=== saturation ===", flush=True)
        write_csv(os.path.join(RESULTS, "saturation.csv"), run_saturation())
    print("done")
