"""
scaling.py -- wall-clock behaviour on large synthetic instances.

Separation structures with |E| in {100, 200, 500} parameters and 20,000 demands
(each parameter meets each demand independently with its own probability
p_e ~ U(0.01, 0.10)), phi = min(., 3), lognormal costs, budget = 5% of the
total cost.  We time:
  plain     density greedy, one marginal evaluation per candidate per step
  lazy      lazy (accelerated) density greedy of Minoux (1978)
  warm      warm start (Alg. 4) after |Delta| in {1, 5, 20} cost changes,
            Q unchanged (the hypothesis of Theorem 5.6), against plain and lazy
            greedy recomputed from scratch on the new costs
All three return the same selection when ties are broken identically; the
script checks this and reports mismatches.
"""
from __future__ import annotations

import heapq
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "v2")
os.makedirs(OUT, exist_ok=True)
K = 3


class Inst:
    def __init__(self, n, n_dem, seed):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.01, 0.10, n)
        self.members = [np.flatnonzero(rng.random(n_dem) < p[e]) for e in range(n)]
        self.cost = np.exp(rng.normal(0.0, 0.75, n))
        self.n, self.n_dem = n, n_dem
        self.B = 0.05 * self.cost.sum()


def gain(inst, D, e):
    return float(np.count_nonzero(D[inst.members[e]] < K))


def plain_greedy(inst, cost, trace=False):
    D = np.zeros(inst.n_dem, dtype=np.int16)
    sel, spent, ev, tr = [], 0.0, 0, []
    rem = set(range(inst.n))
    while True:
        best, bd, bg = -1, -1.0, 0.0
        for e in sorted(rem):
            if spent + cost[e] > inst.B:
                continue
            g = gain(inst, D, e)
            ev += 1
            d = g / cost[e]
            if d > bd + 1e-12:
                best, bd, bg = e, d, g
        if best < 0 or bg <= 0:
            break
        tr.append((best, bd))
        sel.append(best)
        rem.discard(best)
        spent += cost[best]
        D[inst.members[best]] += 1
    return (sel, ev, tr) if trace else (sel, ev)


def lazy_greedy(inst, cost):
    D = np.zeros(inst.n_dem, dtype=np.int16)
    sel, spent, ev = [], 0.0, 0
    heap = []
    for e in range(inst.n):
        g = gain(inst, D, e)
        ev += 1
        heapq.heappush(heap, (-g / cost[e], e, 0))
    it = 0
    while heap:
        negd, e, stamp = heapq.heappop(heap)
        if spent + cost[e] > inst.B:
            continue                      # unaffordable now, and forever
        if stamp == it:
            if -negd <= 0:
                break
            sel.append(e)
            spent += cost[e]
            D[inst.members[e]] += 1
            it += 1
            continue
        g = gain(inst, D, e)
        ev += 1
        heapq.heappush(heap, (-g / cost[e], e, it))
    return sel, ev


def warm_start(inst, cost_new, trace, delta):
    D = np.zeros(inst.n_dem, dtype=np.int16)
    sel, spent, ev = [], 0.0, 0
    delta = set(delta)
    for (e, rho) in trace:
        if e in delta or spent + cost_new[e] > inst.B:
            break
        beaten = False
        for f in sorted(delta):
            if f in sel or spent + cost_new[f] > inst.B:
                continue
            ev += 1
            if gain(inst, D, f) / cost_new[f] > rho + 1e-12:
                beaten = True
                break
        if beaten:
            break
        sel.append(e)
        spent += cost_new[e]
        D[inst.members[e]] += 1
    # complete by plain greedy from the verified prefix
    rem = set(range(inst.n)) - set(sel)
    while True:
        best, bd, bg = -1, -1.0, 0.0
        for e in sorted(rem):
            if spent + cost_new[e] > inst.B:
                continue
            g = gain(inst, D, e)
            ev += 1
            d = g / cost_new[e]
            if d > bd + 1e-12:
                best, bd, bg = e, d, g
        if best < 0 or bg <= 0:
            break
        sel.append(best)
        rem.discard(best)
        spent += cost_new[best]
        D[inst.members[best]] += 1
    return sel, ev


def run(sizes=(100, 200, 500), n_dem=20000, reps=5, deltas=(1, 5, 20)):
    rows = []
    for n in sizes:
        for r in range(reps):
            inst = Inst(n, n_dem, seed=1000 * n + r)
            t = time.perf_counter(); s_p, ev_p, tr = plain_greedy(inst, inst.cost, trace=True); tp = time.perf_counter() - t
            t = time.perf_counter(); s_l, ev_l = lazy_greedy(inst, inst.cost); tl = time.perf_counter() - t
            rows.append(dict(kind="scratch", n=n, rep=r, delta=0,
                             t_plain=tp, ev_plain=ev_p, t_lazy=tl, ev_lazy=ev_l,
                             same=int(sorted(s_p) == sorted(s_l)), k_sel=len(s_p)))
            rng = np.random.default_rng(7 + r + n)
            # perturb costs of parameters NOT selected (the regime the theorem
            # rewards) and, separately, of arbitrary parameters
            for d in deltas:
                for mode in ("unselected", "random"):
                    pool = [e for e in range(n) if e not in s_p] if mode == "unselected" else list(range(n))
                    Dl = list(rng.choice(pool, size=min(d, len(pool)), replace=False))
                    cn = inst.cost.copy()
                    cn[Dl] *= np.exp(rng.normal(0.0, 0.5, len(Dl)))
                    t = time.perf_counter(); s_w, ev_w = warm_start(inst, cn, tr, Dl); tw = time.perf_counter() - t
                    t = time.perf_counter(); s_f, ev_f = plain_greedy(inst, cn); tf = time.perf_counter() - t
                    t = time.perf_counter(); s_z, ev_z = lazy_greedy(inst, cn); tz = time.perf_counter() - t
                    rows.append(dict(kind=f"warm-{mode}", n=n, rep=r, delta=d,
                                     t_warm=tw, ev_warm=ev_w, t_plain=tf,
                                     ev_plain=ev_f, t_lazy=tz, ev_lazy=ev_z,
                                     same=int(s_w == s_f)))
            print(f"n={n} rep={r} plain {tp:.2f}s lazy {tl:.2f}s", flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    with open(os.path.join(OUT, "scaling.pkl"), "wb") as f:
        pickle.dump(rows, f)
    print("scaling done")
