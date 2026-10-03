# Data

## assay_prices_and_design.csv

| column | meaning |
|---|---|
| `key` | short name used in the code and in the paper |
| `cpt_code` | CPT code of the assay |
| `assay` | assay name |
| `price_usd` | Medicare Clinical Laboratory Fee Schedule national payment rate in US$, April 2025 quarterly release (file 25CLABQ2), effective 1 January 2025. All 14 rates were checked against the CMS file `CLFS 2025 Q2V1.csv` on 30 September 2026 (unmodified HCPCS lines). |
| `price_status` | `OBS` (observed) |
| `phase` | clinical phase in which the assay is informative: `early` or `late` |
| `phase_status` | `ASM` (assumed) |
| `salience` | scale factor s_e of the assay's loading beta_{e,t} = s_e * l_t |
| `salience_status` | `ASM` (assumed) |
| `available_windows` | space-separated four-hour windows t0–t5 in which the assay can be resulted; in the other windows the cost is infinite |
| `availability_status` | `ASM` (assumed) |

The observed/assumed labels are those of Table 4 of the paper. The same values are defined in `code/casestudy.py`, which is what the experiments read.

## Patient cohort

No patient data are used. Each cohort of 3,000 admissions over six four-hour windows is simulated by
`casestudy.make_cohort(n=3000, seed=100 + r, outcome_noise=2.8)` for replicate `r`. The model is
documented in Section 6.2 of the paper. Cohorts are recreated deterministically from their seeds and
are therefore not stored.
