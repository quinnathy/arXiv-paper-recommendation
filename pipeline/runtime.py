"""Runtime guards for thread-heavy native libraries.

This project does not require multi-threaded BLAS/OpenMP for correctness.
For stability across mixed Python distributions (e.g., conda/homebrew/uv),
we can force single-thread execution to avoid OpenMP initialization crashes.
"""

from __future__ import annotations

import os


def configure_single_thread_runtime() -> None:
    """Constrain common numerical runtimes to one thread.

    Must run before importing heavy numerical libraries where possible.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

