# Reproducibility bundle

**Resource-aware dynamic soft sets for budget-constrained laboratory test selection in intensive care monitoring**

This bundle holds everything needed to reproduce the results in the manuscript and its Supplementary Material:

- the input data;
- the code that simulates the cohorts and runs every experiment;
- the raw results of the runs reported in the paper;
- the script that turns those results into every table, figure and number in the text.

## Contents

| path | contents |
|---|---|
| `data/assay_prices_and_design.csv` | The 14 assays with CPT codes and Medicare Clinical Laboratory Fee Schedule national payment rates (April 2025 quarterly release, file 25CLABQ2; rates effective 1 January 2025). The CSV also gives each assay's clinical phase, salience and availability windows. Each quantity is marked observed (OBS) or assumed (ASM), as in Table 4 of the paper. |
| `data/README.md` | Data dictionary, and a statement on the simulated cohort. |
| `data/eicu_demo/README.md` | How to obtain the eICU demo files for the real-data study (not redistributed), with their checksums and licence. |
| `code/` | All source code (Python 3). |
| `results/v2/rep_000.pkl … rep_049.pkl` | Raw output of the main experiment: 50 cohort replicates × 5 folds × 15 budgets × 14 policies (the wrapper is run for budgets up to 40%). |
| `results/v2/ext_000.pkl … ext_049.pkl` | Raw output of `experiment4.py`: the exact static panel and the endpoint-weighted exact program, on the same cohorts, folds and subsamples as the main experiment (paired replicate by replicate), with the learned window weights. |
| `results/v2/ssr_*.pkl`, `grd_*.pkl`, `scale_mp.pkl` | Raw output of `experiment5.py` (soft-set parameter reduction, 50 replicates), `experiment6.py` (graded readings, 20 replicates) and `scale_mp.py` (multi-period methods at n = 20, 50, 100). |
| `results/v2/eicu_summary.json` | Aggregate results of the real-data study (Section 7.6, Table 13). The per-stay outputs are not redistributed. |
| `results/v2/*.pkl` (other) | Raw output of the secondary experiments: `sat`, `headroom`, `stat`, `price`, `corr`, `sens`, `inc`, `certificates`, `scaling`, `dp_timing`. |
| `results/verification.json`, `results/v2/verification2.json` | Output of the brute-force verification suites (Supplementary Table S1), including the graded-separation check V17 of Proposition 4.6. |
| `results/main_grid.csv`, `results/selection_map.json`, `results/headroom.csv` | Output of an earlier, smaller run of `experiment.py`. No table, figure or number in the paper uses them; they are kept for completeness. |
| `results/v2/summary2.json`, `results/v2/paired_diffs.csv` | Headline statistics and per-replicate paired differences, written by `analysis2.py`. |
| `outputs/` | Reference outputs: `numbers2.tex` (every number quoted in the text), `tables/` (table bodies) and `figures/` (the 12 figures and the graphical abstract as PNG). Figs. 2–12 (300 dpi) are written by `analysis2.py`; Fig. 1 and `graphical_abstract.png` (2656 × 1062 px, the 531 × 1328 ratio the journal asks for) by `code/overview/`, which reads its numbers from `numbers2.tex`. |
| `requirements.txt` | Python package versions used. |
| `run_all.sh` | Runs the full pipeline from scratch. |
| `LICENSE` | MIT License for the code. |
| `SHA256SUMS.txt` | Checksums of every other file in the bundle; check with `sha256sum -c SHA256SUMS.txt`. |

## Requirements

- Python 3.11. Tested with numpy 2.4.4, scipy 1.17.1, pandas 3.0.2 and matplotlib 3.10.9. No other third-party packages are needed.
- Install with `pip install -r requirements.txt`.
- On Windows, type `python` (or `py`) where this README says `python3`; `run_all.sh` needs a Bash shell such as Git Bash or WSL, but each Python step can be run on its own.
- Every script also runs from IDLE (Run > Run Module) with its default arguments. Started from IDLE, the scripts use one process instead of two (slower, same results); `experiment3.py` without an argument runs all seven secondary experiments. Scripts that find their result files already in `results/v2/` say so and keep them; delete those files to recompute.
- The figures use matplotlib's bundled Computer Modern font (`cmr10`). No LaTeX installation is needed to make them.
- Fig. 1 (study overview) and the graphical abstract are rendered from HTML: they need Node.js 18 or later, the npm packages in `code/overview/package.json` (`npm install` there) and Chromium via Playwright (`npx playwright install chromium`). To use a Chromium that is already installed, set `CHROMIUM_PATH` to its executable.

## Quick check (about 1 minute)

To regenerate every table, Figs. 2–12 and every number from the enclosed raw results, run:

```bash
cd code
python3 analysis2.py
```

This overwrites `outputs/` (Fig. 1 and the graphical abstract excepted), `results/v2/summary2.json` and `results/v2/paired_diffs.csv`, and also writes a PDF copy of each figure. The regenerated `outputs/numbers2.tex` is identical to the enclosed copy, and it is the file the manuscript reads.

## Full reproduction

To rerun every experiment, run `bash run_all.sh`. It performs these steps:

```bash
cd code
export OMP_NUM_THREADS=1
python3 verify.py              # brute-force checks of the framework, single-period
                               # theory, Theorem 5.2(i), the restricted DP and
                               # the warm start                                   -> results/verification.json
python3 verify2.py             # brute-force checks of the exact DPs, certificates,
                               # surrogate and data-dependent bounds, lattice,
                               # graded separation (V17)                           -> results/v2/verification2.json
python3 experiment2.py 50 2    # main experiment, 50 replicates on 2 processes    -> results/v2/rep_*.pkl
python3 experiment4.py 50 2    # exact static panel, endpoint-weighted program    -> results/v2/ext_*.pkl
python3 experiment5.py 50 2    # soft-set parameter reduction baseline            -> results/v2/ssr_*.pkl
python3 experiment6.py 20 2    # graded (fuzzy) readings                          -> results/v2/grd_*.pkl
python3 scale_mp.py            # multi-period methods at n = 20, 50, 100          -> results/v2/scale_mp.pkl
python3 eicu_study.py 20 2     # real eICU demo stays (needs data/eicu_demo/)     -> results/v2/eicu_*.pkl
bash run_secondary.sh          # all secondary experiments                        -> results/v2/*.pkl
python3 analysis2.py           # tables, figures and numbers                      -> outputs/
cd overview && python3 overview_data.py && npm install && node build.js   # Fig. 1, graphical abstract
```

Approximate running times on two cores of a 2026 cloud machine:

| step | time |
|---|---|
| Main experiment | 3–7 min per replicate (about 4 min on average), 1.5–2.5 h in total |
| Exact static panel and endpoint-weighted program (`experiment4.py`) | about 13 s per replicate, 5 min in total |
| Soft-set reduction baseline (`experiment5.py`) | about 13 s per replicate, 6 min in total |
| Graded readings (`experiment6.py`) | about 85 s per replicate, 15 min in total |
| Multi-period scale study (`scale_mp.py`) | about 5 min |
| Real-data study (`eicu_study.py`, 20 repeats) | about 5 min |
| Routine-or-STAT, price and correlation experiments | 6–13 min each |
| One-factor-at-a-time sensitivity | about 13 min |
| Everything else | under 2 min each |

The second argument of `experiment2.py` and `experiment3.py` sets the number of processes.

Every random quantity is seeded. Cohort replicate *r* uses seed `100 + r`, and fold splits and subsamples are seeded from *r*. A rerun therefore reproduces the enclosed raw results. Timing measurements are the exception: `scaling.py` and `dp_timing.py` report wall-clock times, which depend on the machine. Their selections and optimal values are deterministic. As a check, rerunning replicate 0 of the main experiment from scratch reproduced the enclosed `rep_000.pkl` exactly, apart from its recorded wall-clock time.

## Code map

| file | role in the paper |
|---|---|
| `casestudy.py` | Case-study definition: assays and prices, availability, the time-varying loadings, the generative model of the cohort (Section 6.2), the redundancy-aware objective, and the downstream ridge-logistic model and AUROC. |
| `radss.py` | Core RA-DSS structures and reference algorithms: the greedy of Algorithm 1, partial enumeration (Algorithm 2), the restricted DP (Algorithm 3) and warm start (Algorithm 4). The brute-force verification suite uses it. |
| `lattice.py` | Pattern transform that evaluates Q_t on all 2^n subsets in O(k n 2^n) time, and the exact multi-period DP by separable distance transform (Theorem 5.3). |
| `bounds.py` | Switching-aware Lagrangian certificate (Theorem 5.4) and the depleting-budget DP (Proposition 5.5). |
| `engine.py` | Lattice-based engine and the selection policies of Section 6.4: RA-DSS variants, static reduct, TCS-reduct, ℓ1-cost, wrapper, NB-VOI, GA and the others. Also DeLong's test and the data-dependent bound. |
| `experiment2.py` | Main experiment: Sections 7.1–7.3, Tables 5–6, Figs. 7–8, 9(b) and 10(a, b), and Supplementary Table S2. |
| `experiment4.py` | The exact static panel (best constant order set at the true prices) and the endpoint-weighted exact program (Sections 6.4 and 7.1, Tables 5–6 and S2, Fig. 7). It recomputes the static reduct as a check that its pairing with `experiment2.py` is exact. |
| `experiment5.py` | Soft-set parameter reduction baseline, SS-reduct (Section 6.4; Tables 5, 6 and S2). |
| `experiment6.py` | Graded (fuzzy) readings: graded cohort that thresholds to the crisp one, graded objective of Proposition 4.6, crisp versus graded selection (Section 7.4.3, Table 9). |
| `scale_mp.py` | Restricted program and switching-aware certificate at n = 20, 50, 100, with the exact optimum at n = 20 (Section 7.5.2, Table 12). |
| `eicu_study.py` | Real-data validation on the eICU Collaborative Research Database Demo: cohort, readings, reference limits, patient-grouped repeated cross-validation (Section 7.6, Table 13). Needs the files in `data/eicu_demo/`. |
| `experiment3.py` | Secondary experiments: saturation (`sat`), headroom, routine-or-STAT prices (`stat`), price structures (`price`), correlated cohort (`corr`), one-factor-at-a-time sensitivity (`sens`) and warm-start re-selection (`inc`); without an argument it runs all of them (`python3 experiment3.py help` lists them). They cover Fig. 2 (Section 4.3.1), Fig. 9(a) (Section 7.2.2), Tables 7, 8 and 10 and Figs. 11–12 (Section 7.4), and the re-selection results (Section 7.5.1). |
| `experiment.py` | Case-study utilities; `experiment3.py inc` uses its warm-start routines. |
| `certificates.py` | Tightness of the three a-posteriori upper bounds (Section 7.3, Fig. 10(c)). |
| `scaling.py` | Wall-clock scaling to \|E\| = 500 (Section 7.5.1, Table 11). |
| `dp_timing.py` | Timing of the exact DP and the objective lattice (Fig. 4). |
| `verify.py`, `verify2.py` | Brute-force checks of every claim that admits finite checking (Supplementary Table S1). `python3 verify2.py v17` reruns only the graded-separation check and merges it into `verification2.json`. |
| `analysis2.py`, `figs_extra.py` | Statistics, tables, Figs. 2–12 and the `numbers2.tex` macros. |
| `overview/` | Fig. 1, the study overview, and the graphical abstract: `overview_data.py` extracts the selection frequencies and availability they show, and `build.js` renders both, taking every number from `outputs/numbers2.tex` (Tabler icons, MIT licence; KaTeX; Inter font). |
| `run_secondary.sh` | Runs all secondary experiments in order. |
| `runtools.py` | Shared helper: number of processes (one when started from IDLE), serial or parallel execution, and the notice about result files that already exist. |

## Data statement

The main case study uses no patient data. The real-data study uses the openly available eICU Collaborative Research Database Demo (see `data/eicu_demo/README.md`), which is not redistributed here. The acquisition costs are public Medicare fee-schedule payment rates. The patient cohort is simulated from the generative model documented in Section 6.2 of the paper and in `casestudy.make_cohort`, so the cohorts are recreated from their seeds rather than stored. The prices are defined in `casestudy.py`; the CSV in `data/` is a documented copy of them for reference.

## License

The code in this bundle is released under the MIT License; see `LICENSE`. The fee-schedule prices in `data/` are public information from the Centers for Medicare & Medicaid Services.

## Citation

If you use this code, please cite the paper *Resource-aware dynamic soft sets for budget-constrained laboratory test selection in intensive care monitoring*. Full bibliographic details will be added here after publication.
