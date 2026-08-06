# Honest Error Bars — starter code

The assignment is in [`../ASSIGNMENT.md`](../ASSIGNMENT.md). This file is about
getting the code running.

## Setup

You need [uv](https://docs.astral.sh/uv/). That is the only prerequisite; it
installs the right Python and every dependency itself.

```bash
cd starter
uv sync                  # ~1 minute: CPU-only PyTorch, JupyterLab, everything else
uv sync --extra med      # optional: MedMNIST data, only for Track C
uv run jupyter lab
```

Then open `notebooks/01_deterministic.ipynb` and work through the notebooks in
order.

**Data.** Nothing under `data/` is committed. Each track downloads what it needs the
first time you use it, so you can ignore this. If you are about to work offline,
fetch it first:

```bash
uv run python -m bdl.data          # tracks A and B, ~150 KB
uv run python -m bdl.data --all    # everything, including Track C images
```

Downloads are checked against pinned SHA-256 hashes. A mismatch is an error rather
than a warning: a mirror that quietly changed its contents would give you results
that disagree with everyone else's.

## Layout

```
notebooks/
  01_deterministic.ipynb    the baseline, and how to measure an error bar
  02_mc_dropout.ipynb       MC Dropout, and the uncertainty decomposition
  03_ensemble.ipynb         deep ensembles
  04_bbb.ipynb              Bayes by Backprop
  05_compare.ipynb          the table, the figures, the discussion
  06_bonus_vi.ipynb         bonus: is your variational inference right?
  07_bonus_active.ipynb     bonus: active learning, or a shift sweep
  08_bonus_gpu.ipynb        bonus: CNN scale-up, and score matching

bdl/                        PROVIDED. You do not need to edit any of it.
  data.py       toy1d and the three tracks, with downloading
  models.py     the MLP, the Gaussian output head, the training loop
  metrics.py    NLL, coverage, AUROC, the results table
  plots.py      every figure the report needs
  store.py      saving results between notebooks

results/                    written by the notebooks
data/                       downloaded on first use, never committed
```

Everything you write is inside a notebook, marked like this:

```python
# ---- TODO ----------------------------------------------------------------
```

## How the notebooks fit together

Each notebook runs on its own and none of them imports from another. They pass
results through `results/` instead: notebooks 01 to 04 each save a metrics row and,
for `toy1d`, the curves needed to draw a figure, and notebook 05 loads all of it.

Two consequences worth knowing. Figures can be redrawn without retraining anything.
And a small function you wrote in an earlier notebook appears again, already
written, in the setup cell of the later ones, so that each notebook stands alone.

To regenerate everything from scratch:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-5]*.ipynb
```

## Two things that will save you time

**Run the check cells.** They compare your code against answers known in closed
form: the analytic Gaussian KL, the exact Bayesian-linear-regression posterior, the
variance of a Gaussian mixture computed by brute force. A check cell either prints
`OK` or stops the notebook, so you can tell whether you are right without
submitting anything.

**Restart the kernel and run from the top when something looks wrong.** Cells run
out of order are the most common cause of results that make no sense, and they are
invisible in the saved output.

## Do not raise the thread count

Importing `bdl` caps PyTorch at 4 threads. The models here are small, and with the
default setting on a many-core machine thread synchronisation costs far more than
the arithmetic: in testing, training was roughly 50 times slower. Use
`BDL_NUM_THREADS` if you want to measure that yourself.
