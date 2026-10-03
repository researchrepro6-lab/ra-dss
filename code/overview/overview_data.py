"""overview_data.py -- selection frequencies (static reduct vs RA-DSS DP-exact, 10% budget)
and the case-study availability matrix used in panel 5 and panel 2 of Fig. 1.
Reads results/v2/rep_*.pkl; writes code/overview/sel.json."""
import glob, json, os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import casestudy as CS

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "results", "v2", "rep_*.pkl"))):
    rows += pickle.load(open(f, "rb"))["sel"]
order = sorted(range(CS.N_PAR), key=lambda j: CS.COST[j])
out = {"keys": [CS.KEYS[j] for j in order], "cost": [CS.COST[j] for j in order],
       "phase": [CS.TESTS[j]["phase"] for j in order],
       "avail": CS.availability_matrix()[:, order].T.astype(int).tolist()}
for m in ["Static reduct", "RA-DSS DP-exact"]:
    F = np.zeros((CS.N_PAR, CS.N_WIN)); n = 0
    for r in rows:
        if r["method"] != m:
            continue
        n += 1
        for w, mk in enumerate(r["masks"]):
            for j in range(CS.N_PAR):
                F[j, w] += (mk >> j) & 1
    out[m] = (F[order] / n).round(3).tolist()
json.dump(out, open(os.path.join(HERE, "sel.json"), "w"))
print("wrote sel.json")
