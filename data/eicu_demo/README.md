# eICU Collaborative Research Database Demo (not included)

The real-data study (`code/eicu_study.py`, Section 7.6 of the paper) uses the
eICU Collaborative Research Database Demo, version 2.0.1:

> Johnson, A., Pollard, T., Badawi, O., & Raffa, J. (2021). eICU Collaborative
> Research Database Demo (version 2.0.1). PhysioNet. https://doi.org/10.13026/4mxk-na84

It is openly available (no credentialing) under the Open Data Commons Open
Database License v1.0 and is **not redistributed** in this bundle. Nor are the
patient-level outputs derived from it (per-stay predictions and outcomes); the
bundle keeps only the aggregate summary `results/v2/eicu_summary.json`, from
which `analysis2.py` rebuilds Table 13 and the numbers quoted in the text.

To rerun the study, download these three files into this folder:

| file | link | SHA-256 of the copy used |
|---|---|---|
| `patient.csv.gz` | https://physionet.org/files/eicu-crd-demo/2.0.1/patient.csv.gz?download | `15e1eb52169828e3cfff3b67132a087298de4d0365bbb31bb294f6043bffd474` |
| `lab.csv.gz` | https://physionet.org/files/eicu-crd-demo/2.0.1/lab.csv.gz?download | `e06852850abeb65211bc22ac43f17121ba41b14656fac862396449af84d8273a` |
| `microLab.csv.gz` | https://physionet.org/files/eicu-crd-demo/2.0.1/microLab.csv.gz?download | `0250a23d70033f27e1954ced50935cefedaaf3fddd5a5cf5bda2d5a6927eb94d` |

Keep the file names as above (`patient.csv.gz`, `lab.csv.gz`, `microLab.csv.gz`);
unzipped `.csv` files with the same names also work. If a file is missing,
`eicu_study.py` stops and prints these instructions.

Then run, from the `code` folder:

```bash
python eicu_study.py 20 2     # 20 repeats on 2 processes (about 5 minutes)
python analysis2.py           # rebuilds Table 13 and the numbers in the text
```

On Windows, run these commands in Command Prompt or PowerShell. The script can
also be started from IDLE (Run > Run Module): it then uses the defaults, 20
repeats on one process (about 10 minutes).

These files are not needed to reproduce the paper: without them,
`analysis2.py` rebuilds Table 13 from `results/v2/eicu_summary.json`.

Cohort, readings and the reference limits used to call a result abnormal are
defined at the top of `code/eicu_study.py`. The limits are assumptions of the
study (ASM), not observations.
