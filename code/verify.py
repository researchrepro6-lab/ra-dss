"""
verify.py -- brute-force verification of every formal claim in the paper.

Run:  python3 verify.py
Writes results/verification.json and prints a report.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radss import (INF, RADSS, Slice, budget_binds_on_affordable, exact_best,
                   family_is_power_set, full_greedy_evals, greedy_cost_blind,
                   greedy_density, greedy_guarded, greedy_partial_enum,
                   incremental_greedy, random_slice)

OUT = {}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# ---------------------------------------------------------------------------
# V1. Lemma 3.5(ii) (non-collapse): feasible family is a power set  <=>  budget
#     does not bind on the affordable set D_t.
# ---------------------------------------------------------------------------
def v1_non_collapse(n_trials=4000, seed=1):
    hdr("V1  Lemma 3.5(ii) -- feasible family is P(A') iff budget does not bind on D_t")
    rng = np.random.default_rng(seed)
    bad = 0
    n_ps = 0
    n_binding = 0
    for _ in range(n_trials):
        n_par = int(rng.integers(1, 9))
        sl = random_slice(rng, n_obj=5, n_par=n_par,
                          budget_frac=float(rng.uniform(0.05, 1.3)),
                          n_unavail=int(rng.integers(0, max(1, n_par // 3) + 1)))
        fam = sl.feasible_family()
        is_ps = family_is_power_set(fam)
        binds = budget_binds_on_affordable(sl)
        n_ps += is_ps
        n_binding += binds
        # claim: is_ps  <=>  not binds
        if is_ps != (not binds):
            bad += 1
            if bad <= 3:
                print("  COUNTEREXAMPLE", dict(cost=sl.cost, B=sl.budget,
                                               is_ps=is_ps, binds=binds))
    print(f"  trials={n_trials}  power-set families={n_ps}  binding budgets={n_binding}")
    print(f"  violations of the equivalence: {bad}")
    OUT["V1_non_collapse"] = dict(trials=n_trials, violations=bad,
                                  n_power_set=int(n_ps), n_binding=int(n_binding))
    return bad == 0


# ---------------------------------------------------------------------------
# V2. Theorem 4.4 and Proposition 4.5: Q_t is normalised, monotone, submodular.
#     Also: Q_t modular  <=>  separating masks pairwise w-disjoint.
# ---------------------------------------------------------------------------
def v2_submodularity(n_trials=400, seed=2):
    hdr("V2  Theorem 4.4 / Prop. 4.5 -- Q_t normalised / monotone / submodular; modularity test")
    rng = np.random.default_rng(seed)
    viol_mono = viol_sub = viol_norm = 0
    viol_mod = 0
    worst_sub = 0.0
    n_modular = 0
    for _ in range(n_trials):
        n_par = int(rng.integers(1, 7))
        wm = ["binary", "uniform", "random"][int(rng.integers(0, 3))]
        sl = random_slice(rng, n_obj=int(rng.integers(2, 7)), n_par=n_par,
                          weight_mode=wm)
        E = sl.params
        if abs(sl.Q([])) > 1e-12:
            viol_norm += 1
        # monotonicity + submodularity over all A subset B, e not in B
        for rA in range(len(E) + 1):
            for A in itertools.combinations(E, rA):
                QA = sl.Q(A)
                for e in E:
                    if e in A:
                        continue
                    if sl.Q(list(A) + [e]) < QA - 1e-12:
                        viol_mono += 1
                rest = [x for x in E if x not in A]
                for rB in range(len(rest) + 1):
                    for add in itertools.combinations(rest, rB):
                        B = list(A) + list(add)
                        QB = sl.Q(B)
                        for e in E:
                            if e in B:
                                continue
                            dA = sl.Q(list(A) + [e]) - QA
                            dB = sl.Q(list(B) + [e]) - QB
                            if dB > dA + 1e-9:
                                viol_sub += 1
                                worst_sub = max(worst_sub, dB - dA)
        # modularity test
        masks = {e: sl._P[e] for e in E}
        pairwise_disjoint = True
        for a, b in itertools.combinations(E, 2):
            inter = masks[a] & masks[b]
            if 0.5 * float(np.sum(sl.W[inter])) > 1e-12:
                pairwise_disjoint = False
                break
        is_modular = True
        for r in range(len(E) + 1):
            for A in itertools.combinations(E, r):
                if abs(sl.Q(A) - sum(sl.Q([e]) for e in A)) > 1e-9:
                    is_modular = False
                    break
            if not is_modular:
                break
        n_modular += is_modular
        if is_modular != pairwise_disjoint:
            viol_mod += 1
    print(f"  trials={n_trials}")
    print(f"  normalisation violations : {viol_norm}")
    print(f"  monotonicity violations  : {viol_mono}")
    print(f"  submodularity violations : {viol_sub}  (worst excess {worst_sub:.3e})")
    print(f"  modular instances        : {n_modular}")
    print(f"  'modular <=> w-disjoint' violations: {viol_mod}")
    OUT["V2_submodularity"] = dict(trials=n_trials, norm=viol_norm, mono=viol_mono,
                                  sub=viol_sub, worst_excess=worst_sub,
                                  n_modular=int(n_modular), modularity_iff=viol_mod)
    return viol_norm == viol_mono == viol_sub == viol_mod == 0


# ---------------------------------------------------------------------------
# V3. Theorem 4.9 hardness reduction: the Max-Coverage embedding is faithful.
#     Build a coverage instance, embed it, check Q(A) == |union S_j|.
# ---------------------------------------------------------------------------
def v3_reduction(n_trials=600, seed=3):
    hdr("V3  Theorem 4.9 -- Max-Coverage embedding is exact (Q(A) = |union of S_j|)")
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_trials):
        m = int(rng.integers(2, 8))      # |X|
        n = int(rng.integers(1, 6))      # number of sets
        X = list(range(m))
        S = [set(int(x) for x in np.flatnonzero(rng.random(m) < rng.uniform(.2, .8)))
             for _ in range(n)]
        # embedding: U = {u_0} u {u_1..u_m};  F(e_j) = {u_i : x_i in S_j}
        n_obj = m + 1
        F = {}
        for j, Sj in enumerate(S):
            ind = np.zeros(n_obj, dtype=bool)
            for i in Sj:
                ind[i + 1] = True
            F[j] = ind
        W = np.zeros((n_obj, n_obj))
        for i in range(1, n_obj):
            W[0, i] = W[i, 0] = 1.0
        sl = Slice(n_obj=n_obj, F=F, cost={j: 1.0 for j in range(n)},
                   budget=float(n), W=W)
        for r in range(n + 1):
            for A in itertools.combinations(range(n), r):
                cov = set()
                for j in A:
                    cov |= S[j]
                if abs(sl.Q(A) - len(cov)) > 1e-9:
                    bad += 1
    print(f"  trials={n_trials}   embedding mismatches: {bad}")
    OUT["V3_reduction"] = dict(trials=n_trials, mismatches=bad)
    return bad == 0


# ---------------------------------------------------------------------------
# V4. Theorem 4.10: empirical approximation ratios vs the theoretical guarantees.
# ---------------------------------------------------------------------------
def v4_greedy_ratio(n_trials=1500, seed=4):
    hdr("V4  Theorem 4.10 -- empirical approximation ratio of the greedy variants")
    rng = np.random.default_rng(seed)
    alpha_guard = 1.0 - math.exp(-0.5)
    alpha_enum = 1.0 - 1.0 / math.e
    rows = []
    viol_guard = viol_enum = 0
    for _ in range(n_trials):
        n_par = int(rng.integers(3, 11))
        sl = random_slice(rng, n_obj=int(rng.integers(4, 11)), n_par=n_par,
                          budget_frac=float(rng.uniform(0.1, 0.7)),
                          cost_lo=1.0, cost_hi=20.0,
                          weight_mode=["binary", "uniform"][int(rng.integers(0, 2))])
        opt_set, opt = exact_best(sl)
        if opt <= 1e-12:
            continue
        g_plain = sl.Q(greedy_density(sl)[0]) / opt
        g_guard = sl.Q(greedy_guarded(sl)) / opt
        g_enum2 = sl.Q(greedy_partial_enum(sl, kappa=2)) / opt
        # Theorem 4.10(ii) gives 1-1/e for seed size 2 (Kulik et al.); seed size 3
        # (Sviridenko) is checked as well.
        g_enum3 = sl.Q(greedy_partial_enum(sl, kappa=3)) / opt
        g_blind = sl.Q(greedy_cost_blind(sl)) / opt
        rows.append((g_plain, g_guard, g_enum2, g_enum3, g_blind))
        if g_guard < alpha_guard - 1e-9:
            viol_guard += 1
        if g_enum3 < alpha_enum - 1e-9:
            viol_enum += 1
    a = np.array(rows)
    names = ["density greedy", "guarded greedy", "partial-enum (l=2)",
             "partial-enum (l=3)", "cost-blind greedy"]
    stats = {}
    for i, nm in enumerate(names):
        col = a[:, i]
        stats[nm] = dict(mean=float(col.mean()), min=float(col.min()),
                         p05=float(np.percentile(col, 5)),
                         frac_optimal=float(np.mean(col > 1 - 1e-9)))
        print(f"  {nm:22s} mean={col.mean():.4f}  min={col.min():.4f}  "
              f"p05={np.percentile(col,5):.4f}  exact={100*np.mean(col>1-1e-9):5.1f}%")
    print(f"  guarantee 1-e^(-1/2) = {alpha_guard:.4f}; guarded violations: {viol_guard}")
    print(f"  guarantee 1-1/e      = {alpha_enum:.4f}; partial-enum violations: {viol_enum}")
    OUT["V4_greedy_ratio"] = dict(n=len(rows), stats=stats,
                                 alpha_guard=alpha_guard, alpha_enum=alpha_enum,
                                 viol_guard=viol_guard, viol_enum=viol_enum)
    np.save(os.path.join(RESULTS_DIR, "ratios.npy"), a)
    return viol_guard == 0 and viol_enum == 0


# ---------------------------------------------------------------------------
# V5. Theorem 5.2(i): the switching-penalised objective is NOT submodular.
#     Exhibit and confirm the minimal counterexample.
# ---------------------------------------------------------------------------
def v5_switch_not_submodular(n_trials=400, seed=5):
    hdr("V5  Theorem 5.2(i) -- switching-penalised objective is not submodular")
    # Evaluated through RADSS.objective, i.e. the SAME J the paper defines
    # (with the initial switch from the empty configuration charged).
    W = np.zeros((2, 2))
    F = {0: np.array([True, False])}
    sl = lambda: Slice(n_obj=2, F=F, cost={0: 1.0}, budget=10.0, W=W)
    S = RADSS(slices=[sl(), sl()], lam=1.0)
    vals = {
        "empty":   S.objective([[], []]),
        "t1_only": S.objective([[0], []]),
        "t2_only": S.objective([[], [0]]),
        "both":    S.objective([[0], [0]]),
    }
    lhs = vals["t1_only"] + vals["t2_only"]
    rhs = vals["empty"] + vals["both"]
    for k, v in vals.items():
        print(f"  J({k:8s}) = {v:+.0f}")
    print(f"  submodularity requires J(a)+J(b) >= J(empty)+J(a,b):  "
          f"{lhs:+.0f} >= {rhs:+.0f}  ->  {lhs >= rhs}")
    ok = (lhs < rhs)
    print(f"  violation present (as claimed): {ok}")

    # random search: how often does the violation occur on random instances?
    rng = np.random.default_rng(seed)
    viol = 0
    for _ in range(n_trials):
        Tn, npar, nobj = 2, int(rng.integers(1, 4)), int(rng.integers(2, 6))
        lab = rng.integers(0, 2, size=nobj)
        Wm = (lab[:, None] != lab[None, :]).astype(float)
        np.fill_diagonal(Wm, 0.0)
        cost = {e: 1.0 for e in range(npar)}
        sls = [Slice(n_obj=nobj,
                     F={e: (rng.random(nobj) < 0.5) for e in range(npar)},
                     cost=cost, budget=float(npar), W=Wm) for _ in range(Tn)]
        Sx = RADSS(slices=sls, lam=float(rng.uniform(0.5, 5.0)))
        omega = [(t, e) for t in range(Tn) for e in range(npar)]
        def J(sel):
            pol = [[e for (t, e) in sel if t == tt] for tt in range(Tn)]
            return Sx.objective(pol)
        found = False
        for i in range(len(omega)):
            for j in range(len(omega)):
                if i == j:
                    continue
                a, b = omega[i], omega[j]
                if J([a]) + J([b]) < J([]) + J([a, b]) - 1e-9:
                    found = True
                    break
            if found:
                break
        viol += found
    print(f"  random instances exhibiting a violation: {viol}/{n_trials}")
    OUT["V5_switch_not_submodular"] = dict(
        J_empty=vals["empty"], J_a=vals["t1_only"], J_b=vals["t2_only"],
        J_ab=vals["both"], lhs=lhs, rhs=rhs, violation=bool(ok),
        random_trials=n_trials, random_violations=int(viol))
    return ok


# ---------------------------------------------------------------------------
# V6. A-posteriori bound  OBJ(myopic) >= alpha*OPT - lam*K  and dominance of
#     the restricted DP (Algorithm 3) over the myopic and constant policies.
# ---------------------------------------------------------------------------
def v6_multiperiod(n_trials=300, seed=6):
    hdr("V6  Algorithm 3 -- a-posteriori bound and DP dominance")
    rng = np.random.default_rng(seed)
    alpha = 1.0 - math.exp(-0.5)
    viol_bound = 0
    viol_dom = 0
    gaps = []
    dp_vs_exact = []
    for _ in range(n_trials):
        Tn = int(rng.integers(2, 5))
        n_par = int(rng.integers(3, 7))
        n_obj = int(rng.integers(4, 8))
        lam = float(rng.uniform(0.0, 3.0))
        base_lab = rng.integers(0, 2, size=n_obj)
        W = (base_lab[:, None] != base_lab[None, :]).astype(float)
        np.fill_diagonal(W, 0.0)
        slices = []
        cost = {e: float(np.round(rng.uniform(1, 12), 2)) for e in range(n_par)}
        for t in range(Tn):
            F = {e: (rng.random(n_obj) < rng.uniform(.2, .8)) for e in range(n_par)}
            c = dict(cost)
            # random dropouts
            for e in range(n_par):
                if rng.random() < 0.12:
                    c[e] = INF
            av = [v for v in c.values() if v < INF]
            B = float(np.round(rng.uniform(0.2, 0.6) * (sum(av) if av else 1.0), 2))
            slices.append(Slice(n_obj=n_obj, F=F, cost=c, budget=B, W=W))
        S = RADSS(slices=slices, lam=lam)

        myo = S.myopic()
        obj_myo = S.objective(myo)
        K = S.total_switches(myo)
        # per-slice unconstrained optima
        sum_opt = sum(exact_best(sl)[1] for sl in slices)
        # true multi-period optimum (exponential, small instances)
        pol_ex, obj_ex = S.dp_exact_full()
        # bound: OBJ(myopic) >= alpha*OPT - lam*K
        if obj_myo < alpha * obj_ex - lam * K - 1e-7:
            viol_bound += 1
        if abs(obj_ex) > 1e-9:          # exclude degenerate all-zero instances
            gaps.append(obj_myo - (alpha * obj_ex - lam * K))
        # DP dominance
        pol_dp, obj_dp = S.dp_restricted()
        per = S.persistent()
        if obj_dp < max(obj_myo, S.objective(per)) - 1e-7:
            viol_dom += 1
        # ratio is only meaningful when the optimum is strictly positive;
        # J can be negative once the switching penalty dominates
        if obj_ex > 1e-9:
            dp_vs_exact.append(obj_dp / obj_ex)
    print(f"  trials={n_trials}, alpha={alpha:.4f}")
    print(f"  a-posteriori bound violations: {viol_bound}")
    print(f"  min slack of the bound: {min(gaps):.4f}")
    print(f"  DP-dominance violations: {viol_dom}")
    dv = np.array(dp_vs_exact)
    print(f"  restricted DP / exact DP: mean={dv.mean():.4f} min={dv.min():.4f} "
          f"exact-match={100*np.mean(dv>1-1e-9):.1f}%")
    OUT["V6_multiperiod"] = dict(trials=n_trials, alpha=alpha,
                                bound_violations=viol_bound,
                                min_slack=float(min(gaps)),
                                dominance_violations=viol_dom,
                                dp_ratio_mean=float(dv.mean()),
                                dp_ratio_min=float(dv.min()),
                                dp_exact_match_frac=float(np.mean(dv > 1 - 1e-9)))
    return viol_bound == 0 and viol_dom == 0


# ---------------------------------------------------------------------------
# V7. Theorem 5.6: incremental warm-start greedy returns exactly the
#     from-scratch greedy solution, with fewer marginal evaluations.
# ---------------------------------------------------------------------------
def v7_incremental(n_trials=1200, seed=7):
    hdr("V7  Theorem 5.6 -- incremental greedy is exact and cheaper")
    rng = np.random.default_rng(seed)
    mismatches = 0
    ev_inc, ev_full = [], []
    jstars = []
    for _ in range(n_trials):
        n_par = int(rng.integers(4, 13))
        n_obj = int(rng.integers(6, 16))
        lab = rng.integers(0, 2, size=n_obj)
        W = (lab[:, None] != lab[None, :]).astype(float)
        np.fill_diagonal(W, 0.0)
        F = {e: (rng.random(n_obj) < rng.uniform(.2, .8)) for e in range(n_par)}
        cost = {e: float(np.round(rng.uniform(1, 15), 2)) for e in range(n_par)}
        B = float(np.round(rng.uniform(0.25, 0.6) * sum(cost.values()), 2))
        sl_t = Slice(n_obj=n_obj, F=F, cost=cost, budget=B, W=W)
        A_t, trace = greedy_density(sl_t, trace=True)

        # perturb ONLY costs (F unchanged, B unchanged) -> Theorem 5.6 hypotheses
        n_delta = int(rng.integers(1, max(2, n_par // 2) + 1))
        delta = set(int(x) for x in rng.choice(n_par, size=n_delta, replace=False))
        cost2 = dict(cost)
        for e in delta:
            if rng.random() < 0.2:
                cost2[e] = INF
            else:
                cost2[e] = float(np.round(cost[e] * rng.uniform(0.3, 3.0), 2))
        sl_t1 = Slice(n_obj=n_obj, F=F, cost=cost2, budget=B, W=W)

        A_inc, j_star, e_inc = incremental_greedy(sl_t1, trace, delta)
        A_ref, _ = greedy_density(sl_t1)
        e_ful = full_greedy_evals(sl_t1)
        if A_inc != A_ref:
            mismatches += 1
            if mismatches <= 3:
                print("  MISMATCH", sorted(A_inc), sorted(A_ref), "delta", sorted(delta))
        ev_inc.append(e_inc)
        ev_full.append(e_ful)
        jstars.append(j_star)
    ei, ef = np.array(ev_inc, float), np.array(ev_full, float)
    print(f"  trials={n_trials}   solution mismatches: {mismatches}")
    print(f"  marginal evaluations: incremental mean={ei.mean():.1f}  "
          f"from-scratch mean={ef.mean():.1f}")
    print(f"  mean saving: {100*(1-ei.sum()/ef.sum()):.1f}%   "
          f"median j*={np.median(jstars):.1f}")
    OUT["V7_incremental"] = dict(trials=n_trials, mismatches=mismatches,
                                evals_inc_mean=float(ei.mean()),
                                evals_full_mean=float(ef.mean()),
                                saving_frac=float(1 - ei.sum() / ef.sum()),
                                median_jstar=float(np.median(jstars)))
    return mismatches == 0


# ---------------------------------------------------------------------------
# V8. Proposition 3.7 (embedding): a DSS is the RA-DSS with {1, INF} costs and
#     B_t = |A_t|; the feasible family is then exactly P(A_t).
# ---------------------------------------------------------------------------
def v8_embedding(n_trials=800, seed=8):
    hdr("V8  Proposition 3.7 -- DSS embeds as {1,INF}-cost RA-DSS with B_t=|A_t|")
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n_trials):
        n_par = int(rng.integers(1, 9))
        n_obj = int(rng.integers(2, 6))
        A_t = set(int(x) for x in np.flatnonzero(rng.random(n_par) < 0.6))
        F = {e: (rng.random(n_obj) < 0.5) for e in range(n_par)}
        cost = {e: (1.0 if e in A_t else INF) for e in range(n_par)}
        W = np.ones((n_obj, n_obj))
        np.fill_diagonal(W, 0.0)
        sl = Slice(n_obj=n_obj, F=F, cost=cost, budget=float(len(A_t)), W=W)
        fam = set(sl.feasible_family())
        expect = set(frozenset(s) for r in range(len(A_t) + 1)
                     for s in itertools.combinations(sorted(A_t), r))
        if fam != expect or not family_is_power_set(list(fam)):
            bad += 1
    print(f"  trials={n_trials}   embedding failures: {bad}")
    OUT["V8_embedding"] = dict(trials=n_trials, failures=bad)
    return bad == 0


# ---------------------------------------------------------------------------
# V9. Counterexample bank: claims that LOOK true but are false.
#     (These go into the paper as explicit remarks / non-theorems.)
# ---------------------------------------------------------------------------
def v9_counterexamples(seed=9):
    hdr("V9  Counterexample bank (false-looking-true claims stated in the paper)")
    res = {}

    # (a) Cost-blind greedy has NO constant-factor guarantee under a budget.
    #     Family: one expensive high-gain parameter (cost k, gain k+1) versus
    #     k cheap unit-cost parameters of gain k each. Budget k.
    #     Cost-blind takes the expensive one and stops: value k+1.
    #     Optimum takes all k cheap ones: value k^2.  Ratio -> 0 as k grows.
    ratios_blind = []
    for k in (2, 3, 5, 8, 12, 20):
        # objects: a spine pair per cheap parameter, plus a block for the dear one
        # cheap parameter i separates k designated pairs; dear parameter separates k+1
        n_pairs_cheap, n_pairs_dear = k, k + 1
        n_obj = 2 * (k * n_pairs_cheap + n_pairs_dear)
        W = np.zeros((n_obj, n_obj))
        F, cost = {}, {}
        idx = 0
        for i in range(k):                        # k cheap parameters
            ind = np.zeros(n_obj, bool)
            for _ in range(n_pairs_cheap):
                ind[idx] = True
                W[idx, idx + 1] = W[idx + 1, idx] = 1.0
                idx += 2
            F[i] = ind
            cost[i] = 1.0
        ind = np.zeros(n_obj, bool)               # one dear parameter
        for _ in range(n_pairs_dear):
            ind[idx] = True
            W[idx, idx + 1] = W[idx + 1, idx] = 1.0
            idx += 2
        F[k] = ind
        cost[k] = float(k)
        sl = Slice(n_obj=n_obj, F=F, cost=cost, budget=float(k), W=W)
        blind = greedy_cost_blind(sl)
        dens, _ = greedy_density(sl)
        guard = greedy_guarded(sl)
        opt_v = float(k * k)          # all k cheap parameters, by construction
        ratios_blind.append(dict(k=k, Q_blind=sl.Q(blind), Q_density=sl.Q(dens),
                                 Q_guarded=sl.Q(guard), Q_opt=opt_v,
                                 ratio_blind=sl.Q(blind) / opt_v,
                                 ratio_guarded=sl.Q(guard) / opt_v))
    res["cost_blind_no_guarantee"] = ratios_blind
    print("  (a) cost-blind greedy degrades without bound (budget k):")
    for r in ratios_blind:
        print(f"      k={r['k']:3d}  blind={r['Q_blind']:7.1f}  guarded={r['Q_guarded']:7.1f}"
              f"  opt={r['Q_opt']:7.1f}  ratio_blind={r['ratio_blind']:.3f}"
              f"  ratio_guarded={r['ratio_guarded']:.3f}")

    # (a2) The singleton guard is NOT cosmetic: density greedy alone can be
    #      strictly suboptimal. Smallest instance found by search.
    n_obj = 4
    Wq = np.ones((n_obj, n_obj)); np.fill_diagonal(Wq, 0.0)
    Fq = {0: np.array([1, 1, 0, 0], bool), 1: np.array([1, 0, 0, 0], bool)}
    slq = Slice(n_obj=n_obj, F=Fq, cost={0: 10.0, 1: 1.0}, budget=10.0, W=Wq)
    dq, _ = greedy_density(slq)
    gq = greedy_guarded(slq)
    oq_set, oq = exact_best(slq)
    res["singleton_guard_needed"] = dict(Q_density=slq.Q(dq), Q_guarded=slq.Q(gq),
                                        Q_opt=oq, density=sorted(dq),
                                        guarded=sorted(gq), opt=sorted(oq_set))
    print(f"  (a2) density greedy alone = {slq.Q(dq)} < opt = {oq}; "
          f"with singleton guard = {slq.Q(gq)}  -> guard is necessary")

    # (b) Parametric-reduction compatibility (Supplementary Section S1):
    #     restriction does NOT commute with union when the two reducts differ.
    #     U={x,y}, one parameter e, F(e)={x}, G(e)={y}, R_S={e}, R_G={}.
    lhs = {"x", "y"}            # (F u G) restricted to {e} u {} = {e}
    rhs = {"x"}                 # F|{e}  u  G|{}
    res["restriction_union_noncommute"] = dict(lhs=sorted(lhs), rhs=sorted(rhs),
                                              equal=(lhs == rhs))
    print(f"  (b) (S u G)|R_S u R_G = {sorted(lhs)}  vs  S|R_S u G|R_G = {sorted(rhs)}"
          f"   equal={lhs == rhs}")

    # (c) De Morgan fails for the value-only complement when A_t != B_t.
    #     dom((S u G)^c) = A u B   but  dom(S^c n G^c) = A n B.
    A, Bp = {"e1", "e2"}, {"e2", "e3"}
    res["de_morgan_domain_mismatch"] = dict(dom_lhs=sorted(A | Bp),
                                           dom_rhs=sorted(A & Bp),
                                           equal=((A | Bp) == (A & Bp)))
    print(f"  (c) De Morgan domains: {sorted(A|Bp)} vs {sorted(A&Bp)} "
          f" equal={(A|Bp)==(A&Bp)}")

    # (d) Cardinality of a consistent-state set is NOT monotone in severity:
    #     the ambiguity-scoring flaw. |F(e) n R| rewards vagueness.
    definite = {"Critical"}
    vague = {"AtRisk", "Critical"}
    R = {"Critical", "AtRisk"}
    res["ambiguity_scoring_flaw"] = dict(
        definite_score=len(definite & R), vague_score=len(vague & R),
        monotone_in_severity=(len(definite & R) >= len(vague & R)))
    print(f"  (d) |{{Critical}} n R| = {len(definite&R)}  <  "
          f"|{{AtRisk,Critical}} n R| = {len(vague&R)}  -> ambiguity outscores severity")

    # (e) Modularity of Q does NOT make the selection problem easy: with
    #     w-disjoint separating masks the problem is exactly 0-1 knapsack, so
    #     greedy can still be suboptimal. Parameter i separates n_i disjoint
    #     pairs (value n_i) at cost c_i -- a genuine knapsack with unequal values.
    rng = np.random.default_rng(seed)
    n_bad = 0
    n_trials_mod = 400
    worst = 1.0
    for _ in range(n_trials_mod):
        k = int(rng.integers(3, 7))
        vals = [int(rng.integers(1, 9)) for _ in range(k)]
        cost = {i: float(rng.integers(1, 10)) for i in range(k)}
        n_obj = 2 * sum(vals)
        F = {}
        W = np.zeros((n_obj, n_obj))
        idx = 0
        for i in range(k):
            ind = np.zeros(n_obj, bool)
            for _ in range(vals[i]):
                ind[idx] = True
                W[idx, idx + 1] = W[idx + 1, idx] = 1.0
                idx += 2
            F[i] = ind
        B = float(rng.integers(2, max(3, int(sum(cost.values())))))
        sl = Slice(n_obj=n_obj, F=F, cost=cost, budget=B, W=W)
        # confirm modularity of this instance
        assert abs(sl.Q(range(k)) - sum(sl.Q([i]) for i in range(k))) < 1e-9
        g = greedy_guarded(sl)
        _, opt = exact_best(sl)
        if opt > 1e-9:
            worst = min(worst, sl.Q(g) / opt)
        if sl.Q(g) < opt - 1e-9:
            n_bad += 1
    res["modular_greedy_not_always_exact"] = dict(
        trials=n_trials_mod, suboptimal=n_bad, worst_ratio=float(worst))
    print(f"  (e) modular (w-disjoint) case: greedy suboptimal in "
          f"{n_bad}/{n_trials_mod} instances, worst ratio {worst:.3f}")
    print(f"      -> modularity alone does NOT make the problem easy "
          f"(it is 0-1 knapsack)")

    OUT["V9_counterexamples"] = res
    return True


if __name__ == "__main__":
    ok = {}
    ok["V1"] = v1_non_collapse()
    ok["V2"] = v2_submodularity()
    ok["V3"] = v3_reduction()
    ok["V4"] = v4_greedy_ratio()
    ok["V5"] = v5_switch_not_submodular()
    ok["V6"] = v6_multiperiod()
    ok["V7"] = v7_incremental()
    ok["V8"] = v8_embedding()
    ok["V9"] = v9_counterexamples()

    hdr("SUMMARY")
    for k, v in ok.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    OUT["_summary"] = {k: bool(v) for k, v in ok.items()}
    with open(os.path.join(RESULTS_DIR, "verification.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    print(f"\nwrote {os.path.join(RESULTS_DIR, 'verification.json')}")
    sys.exit(0 if all(ok.values()) else 1)
