#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for j in sat headroom; do python3 experiment3.py $j 2; done
python3 certificates.py 20
for j in stat price corr sens; do python3 experiment3.py $j 2; done
python3 experiment3.py inc
python3 scaling.py
python3 dp_timing.py            # exact-DP and lattice timing (Fig. 4)
echo SECONDARY_ALL_DONE
