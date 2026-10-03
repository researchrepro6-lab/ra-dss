#!/bin/bash
# Full reproduction of every result in the paper (see README.md for running times).
set -e
cd "$(dirname "$0")/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NPROC=${NPROC:-2}
python3 verify.py
python3 verify2.py
python3 experiment2.py 50 "$NPROC"
python3 experiment4.py 50 "$NPROC"
python3 experiment5.py 50 "$NPROC"
python3 experiment6.py 20 "$NPROC"
python3 scale_mp.py
if [ -f ../data/eicu_demo/lab.csv.gz ]; then python3 eicu_study.py 20 "$NPROC"; else echo "eICU demo files not found; using results/v2/eicu_summary.json"; fi
bash run_secondary.sh
python3 analysis2.py
(cd overview && python3 overview_data.py && npm install --silent && node build.js)
echo "All results regenerated in ../results and ../outputs"
