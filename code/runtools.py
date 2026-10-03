"""
runtools.py -- how the experiment scripts spread replicates over processes.

Every script takes the number of processes as an optional argument.  Without
it they use two processes from a terminal and one process when started from
IDLE (Run > Run Module): IDLE passes no arguments, does not show the output of
worker processes, and handles multiprocessing poorly on Windows.  With one
process the replicates run one after another in the current process; the
results are identical, only slower.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool


def in_idle() -> bool:
    return "idlelib" in sys.modules


def n_proc_arg(pos: int = 2, default: int = 2) -> int:
    """Number of processes from sys.argv[pos], else the default (1 in IDLE)."""
    if len(sys.argv) > pos:
        return int(sys.argv[pos])
    return 1 if in_idle() else default


def run_all(fn, items, n_proc: int, ordered: bool = False):
    """Apply fn to every item, serially when n_proc <= 1; returns the results
    (in input order when ordered=True)."""
    items = list(items)
    if n_proc <= 1:
        return [fn(i) for i in items]
    with Pool(n_proc) as pool:
        if ordered:
            return pool.map(fn, items)
        return list(pool.imap_unordered(fn, items))


def note_existing(out_dir: str, prefix: str, n_rep: int) -> None:
    """Replicate files already in out_dir are kept and not recomputed (the runs
    are resumable).  Say so, so that a fast 'all done' is not a surprise."""
    import os
    have = [r for r in range(n_rep)
            if os.path.exists(os.path.join(out_dir, f"{prefix}_{r:03d}.pkl"))]
    if not have:
        return
    if len(have) == n_rep:
        print(f"All {n_rep} result files {prefix}_000.pkl ... {prefix}_{n_rep - 1:03d}.pkl are already "
              f"in {os.path.normpath(out_dir)}; nothing to recompute.\n"
              f"To rerun from scratch, delete or move those files first.", flush=True)
    else:
        print(f"{len(have)} of {n_rep} result files {prefix}_*.pkl already exist and are kept; "
              f"computing the other {n_rep - len(have)}.", flush=True)
