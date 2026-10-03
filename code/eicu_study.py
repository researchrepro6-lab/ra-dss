"""
eicu_study.py -- validation of the selection methods on real intensive-care
data: the eICU Collaborative Research Database Demo (v2.0.1, PhysioNet, Open
Data Commons Open Database License v1.0).

Data (not redistributed here; download the three files listed in
data/eicu_demo/README.md into data/eicu_demo/):
  patient.csv.gz, lab.csv.gz, microLab.csv.gz

Cohort.  First ICU unit stay of each hospital stay (unitvisitnumber = 1) with a
unit length of stay of at least 24 h and a recorded hospital discharge status.
Outcome: in-hospital death (hospitaldischargestatus = 'Expired').  Because the
horizon is the first 24 h and every included stay lasts at least 24 h, the
outcome is not determined inside the horizon.

Readings.  Six four-hour windows from unit admission (lab result offset in
[0, 1440) minutes).  For assay e and window t, X[i, t, e] = 1 when the assay was
resulted in the window and at least one of its components lay outside the
conventional adult reference limits in LIMITS below; 0 when it was normal or
not measured.  Panel assays (CBC, BMP, CMP, ABG) are abnormal when any
component is; CMP is 'measured' when any of its hepatic components is, and is
then abnormal also when a BMP component in the same window is.  The blood
culture is abnormal when a blood culture taken in the window grew an organism.
D-dimer and procalcitonin are not recorded in the demo database and are treated
as unavailable in every window; every other assay is available in every window
at its fee-schedule price.  The reference limits are assumptions of the study
(ASM), not observations.

Evaluation.  Repeated 5-fold cross-validation (N_REPEAT repeats) with folds
grouped by patient (uniquepid), the downstream ridge-logistic model and all
selection code of the simulated study.  For each repeat the out-of-fold
predictions are pooled and scored; analysis2.py reports the mean over repeats
and 95% intervals from a patient-level bootstrap of the whole procedure.

Usage: python eicu_study.py [n_repeat] [n_proc]   -> results/v2/eicu_XXX.pkl
       (n_proc = 1 runs without multiprocessing, e.g. when started from IDLE)

Without the three data files the script stops with instructions.  The paper's
eICU numbers (Table 13) do not need them: analysis2.py rebuilds Table 13 from
results/v2/eicu_summary.json.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtools import n_proc_arg, note_existing, run_all
import casestudy as CS
import engine as E
from experiment2 import GAMMA, OUT
from experiment4 import static_exact
from experiment5 import ss_reduct_window

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "eicu_demo")
INF = float("inf")
N_WIN, NP = CS.N_WIN, CS.N_PAR
KEY = {k: i for i, k in enumerate(CS.KEYS)}
BUDGETS_E = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.00]
N_FOLD = 5

# conventional adult reference limits (low, high); None = no limit on that side
LIMITS = {
    "PT - INR": (None, 1.2), "PT": (None, 13.5),
    "WBC x 1000": (4.0, 11.0), "platelets x 1000": (150.0, 450.0),
    "sodium": (135.0, 145.0), "potassium": (3.5, 5.0), "chloride": (98.0, 106.0),
    "bicarbonate": (22.0, 29.0), "BUN": (None, 20.0), "creatinine": (None, 1.2),
    "glucose": (70.0, 180.0), "calcium": (8.5, 10.5),
    "albumin": (3.5, None), "total bilirubin": (None, 1.2), "AST (SGOT)": (None, 40.0),
    "ALT (SGPT)": (None, 40.0), "alkaline phos.": (None, 120.0), "total protein": (6.0, 8.3),
    "lactate": (None, 2.0),
    "pH": (7.35, 7.45), "paO2": (80.0, None), "paCO2": (35.0, 45.0),
    "phosphate": (2.5, 4.5), "CRP": (None, 1.0), "magnesium": (1.7, 2.2),
    "troponin - I": (None, 0.04), "troponin - T": (None, 0.01), "BNP": (None, 100.0),
}
HGB_LOW = {"Female": 12.0, "Male": 13.0}          # WHO anaemia thresholds
ASSAY_OF = {
    "PT - INR": "PT", "PT": "PT",
    "WBC x 1000": "CBC", "Hgb": "CBC", "platelets x 1000": "CBC",
    "sodium": "BMP", "potassium": "BMP", "chloride": "BMP", "bicarbonate": "BMP",
    "BUN": "BMP", "creatinine": "BMP", "glucose": "BMP", "calcium": "BMP",
    "albumin": "CMPH", "total bilirubin": "CMPH", "AST (SGOT)": "CMPH",
    "ALT (SGPT)": "CMPH", "alkaline phos.": "CMPH", "total protein": "CMPH",
    "lactate": "LAC", "pH": "ABG", "paO2": "ABG", "paCO2": "ABG",
    "phosphate": "PHOS", "CRP": "CRP", "magnesium": "MG",
    "troponin - I": "TROP", "troponin - T": "TROP", "BNP": "BNP",
}


FILES = ("patient", "lab", "microLab")
LINK = "https://physionet.org/files/eicu-crd-demo/2.0.1/{}.csv.gz?download"


def data_file(name, data_dir=DATA):
    """Path of <name>.csv.gz, or of an unzipped <name>.csv; None if absent.
    File names are matched case-insensitively."""
    if not os.path.isdir(data_dir):
        return None
    present = {f.lower(): f for f in os.listdir(data_dir)}
    for cand in (name + ".csv.gz", name + ".csv"):
        if cand.lower() in present:
            return os.path.join(data_dir, present[cand.lower()])
    return None


def check_data(data_dir=DATA):
    missing = [n for n in FILES if data_file(n, data_dir) is None]
    if not missing:
        return
    folder = os.path.normpath(os.path.abspath(data_dir))
    lines = [
        "",
        "The eICU demo data files are not in the bundle (licence: ODbL); download them first.",
        f"Missing: {', '.join(n + '.csv.gz' for n in missing)}",
        f"Put the files in: {folder}",
        "Download links (free, no PhysioNet account needed):",
        *[f"  {LINK.format(n)}" for n in FILES],
        "Keep the names patient.csv.gz, lab.csv.gz and microLab.csv.gz (unzipped .csv files also work).",
        "See data/eicu_demo/README.md for the SHA-256 checksums.",
        "",
        "Not needed to reproduce the paper: 'python analysis2.py' rebuilds Table 13",
        "from results/v2/eicu_summary.json.",
    ]
    sys.exit("\n".join(lines))


def load_cohort(data_dir=DATA):
    check_data(data_dir)
    p = pd.read_csv(data_file("patient", data_dir))
    lab = pd.read_csv(data_file("lab", data_dir))
    mic = pd.read_csv(data_file("microLab", data_dir))
    c = p[(p.unitvisitnumber == 1) & (p.unitdischargeoffset >= 1440) &
          p.hospitaldischargestatus.notna()].copy()
    c = c.sort_values("patientunitstayid").reset_index(drop=True)
    idx = {s: i for i, s in enumerate(c.patientunitstayid)}
    n = len(c)
    y = (c.hospitaldischargestatus == "Expired").astype(int).values
    sex = dict(zip(c.patientunitstayid, c.gender))
    L = lab[lab.patientunitstayid.isin(idx) & (lab.labresultoffset >= 0) &
            (lab.labresultoffset < 1440) & lab.labname.isin(list(ASSAY_OF) + ["Hgb"]) &
            lab.labresult.notna()].copy()
    L["w"] = (L.labresultoffset // 240).astype(int)
    lo = L.labname.map(lambda s: LIMITS.get(s, (None, None))[0])
    hi = L.labname.map(lambda s: LIMITS.get(s, (None, None))[1])
    hgb = L.labname == "Hgb"
    lo = lo.where(~hgb, L.patientunitstayid.map(lambda s: HGB_LOW.get(sex.get(s), 12.0)))
    v = L.labresult.astype(float)
    L["abn"] = ((lo.notna() & (v < lo.astype(float))) | (hi.notna() & (v > hi.astype(float))))
    L["assay"] = L.labname.map(ASSAY_OF)
    meas = np.zeros((n, N_WIN, NP), dtype=bool)
    X = np.zeros((n, N_WIN, NP), dtype=np.uint8)
    g = L.groupby(["patientunitstayid", "w", "assay"]).abn.any().reset_index()
    bmp_abn = np.zeros((n, N_WIN), dtype=bool)
    for r in g.itertuples(index=False):
        i = idx[r.patientunitstayid]
        if r.assay == "CMPH":
            j = KEY["CMP"]
        else:
            j = KEY[r.assay]
        meas[i, r.w, j] = True
        X[i, r.w, j] = max(X[i, r.w, j], int(r.abn))
        if r.assay == "BMP" and r.abn:
            bmp_abn[i, r.w] = True
    cmp_j = KEY["CMP"]
    X[:, :, cmp_j] = np.where(meas[:, :, cmp_j] & bmp_abn, 1, X[:, :, cmp_j])
    B = mic[mic.patientunitstayid.isin(idx) & mic.culturesite.str.startswith("Blood") &
            (mic.culturetakenoffset >= 0) & (mic.culturetakenoffset < 1440)]
    for r in B.itertuples(index=False):
        i, w = idx[r.patientunitstayid], int(r.culturetakenoffset // 240)
        meas[i, w, KEY["CULT"]] = True
        if isinstance(r.organism, str) and r.organism.strip().lower() != "no growth":
            X[i, w, KEY["CULT"]] = 1
    groups = c.uniquepid.values
    return SimpleNamespace(X=X, y=y, meas=meas, groups=groups, n=n,
                           n_patients=int(c.uniquepid.nunique()),
                           n_hosp=int(c.hospitalid.nunique()))


def eicu_cost_matrix():
    C = np.empty((N_WIN, NP))
    for w in range(N_WIN):
        for j in range(NP):
            C[w, j] = CS.COST[j]
    C[:, KEY["DDIM"]] = INF
    C[:, KEY["PCT"]] = INF
    return C


def grouped_folds(groups, k, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.flatnonzero(f != j), np.flatnonzero(f == j)) for j in range(k)]


POLICIES = ["Full panel", "Random (budget)", "Static reduct", "Static exact", "SS-reduct",
            "TCS-reduct", "L1-cost", "NB-VOI greedy", "RA-DSS myopic",
            "RA-DSS exact/window", "RA-DSS DP-exact"]


def run_repeat(rep):
    path = os.path.join(OUT, f"eicu_{rep:03d}.pkl")
    if os.path.exists(path):
        return path
    t0 = time.time()
    coh = load_cohort()
    C = eicu_cost_matrix()
    folds = grouped_folds(coh.groups, N_FOLD, seed=700 + rep)
    rows, oof, sel = [], {}, []
    for fi, (tr, te) in enumerate(folds):
        sls = E.build_slices(coh, tr, C, sub_seed=rep)
        sig = [s.sigma for s in sls]
        tsl = E.build_slices(coh, te, C, sigmas=sig, subsample=False)
        qmax_te = sum(s.qmax() for s in tsl)
        lam = GAMMA * E.qbar(sls)
        Xtr, ytr = coh.X[tr], coh.y[tr]
        cache = {}
        rng = np.random.default_rng(9100 + 31 * rep + fi)
        for bf in BUDGETS_E:
            B = bf * CS.FULL_PANEL_COST
            pols = {
                "Full panel": [s.avail_mask for s in sls],
                "Random (budget)": [E.random_policy(s, B, rng) for s in sls],
                "Static reduct": E.static_pooled(sls, B),
                "Static exact": static_exact(sls, B),
                "SS-reduct": [ss_reduct_window(Xtr[:, w, :].astype(np.int8), C[w], B)
                              for w in range(N_WIN)],
                "TCS-reduct": [E.tcs_reduct_window(Xtr, ytr, w, C[w], B) for w in range(N_WIN)],
                "L1-cost": E.l1_cost(Xtr, ytr, C, B),
                "NB-VOI greedy": E.nb_voi(Xtr, ytr, C, B),
                "RA-DSS myopic": [E.guarded(s, B) for s in sls],
                "RA-DSS exact/window": [E.exact_window(s, B) for s in sls],
                "RA-DSS DP-exact": E.dp_exact(sls, B, lam)[0],
            }
            for m, pol in pols.items():
                key = tuple(pol)
                if key not in cache:
                    cache[key] = E.evaluate(coh, tr, te, pol, seed=fi)
                auc, pred = cache[key]
                oof.setdefault((bf, m), np.zeros(coh.n))[te] = pred
                rows.append(dict(rep=rep, fold=fi, budget_frac=bf, method=m, auroc=auc,
                                 cost=E.realised_cost(pol, C),
                                 q_test=sum(s.L[p] for s, p in zip(tsl, pol)) / qmax_te,
                                 switches=E.n_switches(pol),
                                 n_distinct=len(set(pol)),
                                 n_tests=int(sum(E.POP[p] for p in pol))))
                if abs(bf - 0.10) < 1e-12 and m in ("Static exact", "RA-DSS DP-exact",
                                                    "RA-DSS exact/window", "Static reduct"):
                    sel.append(dict(rep=rep, fold=fi, method=m, masks=list(pol)))
    res = dict(rows=rows, oof=oof, sel=sel, seconds=time.time() - t0)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    print(f"eicu repeat {rep} done in {time.time() - t0:.0f}s", flush=True)
    return path


def describe():
    coh = load_cohort()
    return dict(n=coh.n, deaths=int(coh.y.sum()), n_patients=coh.n_patients, n_hosp=coh.n_hosp,
                meas_rate=coh.meas.mean(axis=0), abn_rate=coh.X.mean(axis=0),
                y=coh.y, groups=coh.groups)


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_proc = n_proc_arg(2)   # one process when started from IDLE (runtools.py)
    check_data()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eicu_cohort.pkl"), "wb") as f:
        pickle.dump(describe(), f)
    note_existing(OUT, "eicu", n_rep)
    run_all(run_repeat, range(n_rep), n_proc)
    print("all done; now run: python analysis2.py")
