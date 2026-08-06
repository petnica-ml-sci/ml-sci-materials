"""Provided helper library for the Honest Error Bars project.

You do not need to edit anything in this package. Your work happens in
`notebooks/`.

    data     the toy problem and the three domain tracks
    models   an MLP, the Gaussian output head, and a training loop
    metrics  scoring functions whose implementation is fiddly enough to provide
    plots    the figures the report needs
    store    saving and loading results between notebooks

Importing `bdl` caps PyTorch at four CPU threads. The models here are small, and
for small tensors the cost of synchronising many threads is larger than the
arithmetic being split between them: on a 16-core machine the default setting
trains about 50 times slower. Override with `BDL_NUM_THREADS` if you want to
measure this yourself.
"""

from __future__ import annotations

import os

import torch

_DEFAULT_THREADS = 4


def _configure_threads() -> None:
    requested = os.environ.get("BDL_NUM_THREADS")
    try:
        n = int(requested) if requested else min(_DEFAULT_THREADS, torch.get_num_threads())
    except ValueError:
        n = _DEFAULT_THREADS
    torch.set_num_threads(max(1, n))


_configure_threads()

__all__ = ["data", "metrics", "models", "plots", "store"]
