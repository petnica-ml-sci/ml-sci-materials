# Bayesian Deep Learning Methods

### Comparing Bayesian deep learning methods on your own data

---

## What this project is about

A trained network will give you a confident prediction for an input unlike anything
it was trained on. For a scientist, knowing the uncertainty of output is as crucial as output itself. 

We are testing four methods: a deterministic baseline with a
heteroscedastic likelihood, MC Dropout, a deep ensemble, and Bayes by Backprop
written from the ELBO.

You will find that none of them is "honest" all the time, and they all have pros and cons. 
You should use the notes from Andrija's lectures (see the discord) to obtain insight into the theoretical 
(and implementation) details on this four techniques.

Below is some Claude-generated code to help you kick off the project:

---

## Setup

```bash
cd starter
uv sync                  # installs Python, CPU PyTorch and JupyterLab, ~1 min
uv sync --extra med      # optional: MedMNIST data, only needed for Track C
uv run jupyter lab       # then open notebooks/01_deterministic.ipynb
```

Datasets download on first use. If you will be working offline, fetch them first:

```bash
uv run python -m bdl.data --all
```

You work entirely in `notebooks/`. Nothing in `bdl/` needs editing: it holds the
data loaders, the network, the training loop, the scoring functions that are fiddly
to implement, and the figures.

---

## The basic task: five notebooks, and testing them on a toy-model

Work through them in order. Each one states what you write, ends with a "done when"
list, and contains check cells that compare your code against answers known in
closed form. A check cell either prints `OK` or stops the notebook.

| notebook | you write | 
|---|---|
| `01_deterministic.ipynb` | `calibration_error` | 
| `02_mc_dropout.ipynb` | `predict_mc_dropout`, `decompose_variance` | 
| `03_ensemble.ipynb` | `fit_ensemble`, `predict_ensemble` | 
| `04_bbb.ipynb` | `BayesLinear`, `elbo_loss`, `predict_bbb` |
| `05_compare.ipynb` | nothing; the table, the figures and the discussion | 

### Notebook 01 — the baseline, and how to measure honesty

The network predicts two numbers, a mean `mu(x)` and a log-variance
`log sigma^2(x)`, and trains on the Gaussian negative log-likelihood instead of
squared error. That gives "aleatoric" uncertainty: the noise the model believes is in
the measurement. It gives no "epistemic" uncertainty at all, because there is one
weight vector and nothing for it to disagree with.

You write the regression calibration error: the average discrepancy between a nominal
credible level and the coverage actually observed.

*Done when:* the check cell prints `OK` and you can explain what this is about. Do not go further 
until we are all clear what has been done here.

### Notebook 02 — MC Dropout, and the decomposition

Train with dropout (we did not talk about this too much but it is a common feature of NNs), 
then leave dropout on at test time and push each input through
`T` times. The `T` answers are approximate posterior samples, at no extra training
cost.

Then the split the rest of the project depends on:

```
Var[y] = E_W[sigma^2(x)]  +  Var_W[mu(x)]
         aleatoric            epistemic
```

Average the per-sample noise variances for the first term; take the variance of the
per-sample means for the second. This is an identity, so your total must equal the
sum to machine precision.

Common mistake: calling `model.train()` instead of `enable_dropout(model)`. On this
MLP the two behave the same, but with batch normalisation the first one makes your
prediction depend on which other test points share the batch.

Track C also needs the classification version, the entropy decomposition, and a
binned confidence calibration error. Section 2.8 covers both; skip it on Tracks A
and B.

*Done when:* both check cells print `OK` and the epistemic band is visibly wider
inside the data gap.

### Notebook 03 — deep ensembles

Train `M = 5` networks from different random initialisations; the spread between
them is the epistemic term. No variational family, no KL, nothing to derive, and it
is repeatedly the strongest baseline in the literature.

Common mistake: varying the batch order but not the initialisation. Section 3.4
measures what that costs, so you can see the difference rather than take it on
trust.

*Done when:* the check cell confirms five distinct members with non-zero
disagreement.

### Notebook 04 — Bayes by Backprop

The one built from the mathematics. Every weight becomes a Gaussian with a learned
mean and standard deviation, and the objective is

```
L = E_q[log p(D | W)] - KL(q(W) || p(W))
```

maximised by sampling `W = mu + sigma * eps` with `eps ~ N(0, I)`, which keeps the
whole thing differentiable in `mu` and `sigma`. Four pieces:

1. `BayesLinear.__init__` — register `mu` and `log sigma` for weights and biases.
2. `BayesLinear.forward` — sample with the reparameterization trick. Sampling with
   `torch.normal(mu, sigma)` instead breaks the gradient path and `sigma` never
   trains; there is a check for exactly that.
3. `BayesLinear.kl_divergence` — the closed-form Gaussian KL, checked against a
   Monte Carlo estimate, so a missing factor of two will be caught.
4. `elbo_loss` — the KL is a per-dataset term, not a per-example one. The base loss
   is already a batch mean, so add `model_kl(model) / n_train`. Not
   `/ batch_size`, not unscaled. Get this wrong and nothing crashes: too much
   weight collapses the posterior onto the prior, too little gives you an expensive
   deterministic network.

Also pass `weight_decay=0` when fitting. Weight decay is a Gaussian prior on the
weights and your KL term already contains one.

*Done when:* all four check cells print `OK`, in particular the last one, which
points your variational inference at Bayesian linear regression, where the
posterior is known exactly. If that passes, your ELBO, KL and scaling are all
correct together.

### Notebook 05 — the comparison

Loads what the first four notebooks saved and produces the table and the figures the
report needs. Nothing to implement. It lists four questions to answer; these all
show up in a correct run, and if your numbers disagree, that is worth investigating
and worth reporting.

- The baseline has an OOD-detection AUROC of exactly 0.500 and a shifted NLL in the
  hundreds. Why are those the same fact?
- Deep ensembles usually get the best RMSE while their 95% intervals cover less than
  95% of the test data. Best accuracy and worst honesty, in one row.
- Bayes by Backprop fits worst and is often the best calibrated. Explain the trade.
- The winner is not the same on every track, or in every column.

---

## Your own application

Pick one in notebook 01 and keep it for the whole project. The notebooks, the
metrics and the rubric are the same for all three; the subject matter changes, the
workload does not. Note that two problems are regression (inferring a number / set of numbers) and 
one problem is classification. If you are unsure about what changes are needed for classificaion, focus
on one of the first two.

| |  A — astrophysics | B — chemistry |  C — biology / imaging |
|---|---|---|---|
| **Data** | Pantheon+ Type Ia supernovae, 1624 objects, 5 features | ESOL aqueous solubility, 1128 molecules, 6 descriptors | MedMNIST blood cells, 8 classes, 28×28 RGB |
| **Predict** | peak apparent brightness (magnitudes) | log solubility (mol/L) | cell type |
| **Shift** | supernovae more distant than any in training (z > 0.5) | molecules heavier than any in training | degraded acquisition (sensor noise) |
| **Task** | regression | regression | classification |

Every notebook also uses **`toy1d`**, a 1-D problem with a deliberate hole in the
training data and noise that grows with `x`. It is where a broken implementation is
obvious at a glance, so debug there first.

---

## Bonus notebooks — at least one required

About 4 hours each. Only the best one is graded.

### `06_bonus_vi.ipynb` — is your variational inference actually right?

Both parts use Bayesian linear regression, where the exact posterior is known, so
every claim is checkable.

**(a) Mean-field against full covariance.** Fit both to the same posterior and sweep
the correlation between features. Mean-field's posterior width falls away from the
truth as correlation grows, and it matches the closed-form prediction
`sqrt(1 - R^2)` rather than the exact width. This turns the standard warning about
mean-field variational inference into a number.

Run the control first: the full-covariance fit must recover the exact answer to
within a percent. If it does not, you have an optimisation problem rather than a
statistics one, and nothing the mean-field fit tells you can be trusted. At 2000
optimisation steps both families come out about 13% too wide.

**(b) Score function against reparameterization.** Implement both gradient
estimators for the same ELBO and measure their variance against the Monte Carlo
budget. Both are unbiased, so verify that their means agree, but one has a few
hundred times the variance of the other. Say what that means for the number of
samples you would need.

### `07_bonus_active.ipynb` — uncertainty that pays for itself

Start from 20 labels, repeatedly label the points with the highest *epistemic*
uncertainty, and compare against labelling at random. Five seeds, learning curves
with error bands.

Epistemic and not total: aleatoric uncertainty is irreducible, so labelling a point
that is uncertain only because it is noisy teaches the model nothing.

Be ready for the answer to be that random wins. On `toy1d` it does, reproducibly,
and explaining why is a better report than a narrow win. Never report a single-seed
learning curve: strategy differences are routinely smaller than seed-to-seed
spread, and the notebook prints a warning when that is the case.

The alternative in the same notebook is a shift sweep: sweep the corruption
strength on the test inputs and plot NLL, calibration and OOD detection against it.
Report where each method's error bars stop meaning anything.

### `08_bonus_gpu.ipynb` — scale-up, and a model with no likelihood

Wants a GPU for part (a); Colab is fine. Comment out the `[tool.uv.sources]` block
in `pyproject.toml` first, because it pins CPU-only wheels. `SMOKE = True` runs the
whole code path on a laptop.

**(a) Scale-up.** CNN versions of the methods on Track C under increasing
corruption. The question is not which method wins but whether the *ordering* you
found with MLPs survives, and if not, why.

**(b) Score matching.** Denoising score matching with a noise-conditioned score
network, plus annealed Langevin sampling, on a 2-D density. A model that never
represents a density at all, only its gradient. Then discuss what "uncertainty"
means without an explicit likelihood.

---

## The open-ended component — required

State **one research question of your own**, form a hypothesis, run a controlled
experiment, and report what happened, including when it contradicts you.

This is graded on experimental design and honesty, not on the result. A clean
experiment that refutes your hypothesis scores higher than a vague one that
confirms it.

Some starting points, though your own question is better:

- Bayes by Backprop with `hidden=(32, 32)` is better calibrated than with
  `(64, 64)`. Why would a smaller network give better variational inference?
- At matched compute, is a 5-member ensemble better than 5× more MC Dropout samples?
- How sensitive is everything to `prior_std`? Is there a value at which Bayes by
  Backprop stops underfitting without diverging?
- The heteroscedastic head can absorb epistemic uncertainty into `sigma(x)`.
  Notebook 04 shows this happening on `toy1d`; can you measure it on your track?
- Notebook 07 acquires on raw epistemic variance. The information-theoretic score is
  `0.5 * log(1 + v / sigma^2)`, which discounts noisy points. Does it beat random?
- Is the ranking of methods stable across seeds, or did you report noise?
- (Track C) `load_far_ood()` gives images from a different imaging modality. Most
  methods score *below* 0.5 AUROC on it, meaning they are more confident on data
  they have never seen. Explain it.

---

## What to hand in

1. **The notebooks**, with their outputs, including the check cells. `results/`
   regenerates by running `01` to `05` in order, or in one command:

   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/0[1-5]*.ipynb
   ```

2. **A report, at most 5 pages:**
   - the comparison table for `toy1d` and for your track;
   - at least three figures, discussed rather than merely included;
   - the derivation behind one thing you implemented, in your own notation: the
     Gaussian KL, the `1 / n_train` scaling, or the variance decomposition;
   - your open-ended experiment: question, hypothesis, design, result;
   - a closing paragraph: **"what I would not trust this model for."**

3. Optionally, a 5-minute lightning talk.

Grading is in [`RUBRIC.md`](RUBRIC.md). The short version: correctness is checked by
the check cells, and everything above that is about whether you understood what the
numbers were telling you.

---

## If you get stuck

- **Restart the kernel and run from the top.** Notebook state that is out of order
  is the most common cause of results that make no sense.
- **Debug on `toy1d` and look at the band figure.** Almost every bug is visible
  there: a band that does not widen in the gap, a mean that ignores the data, an
  aleatoric band that swallows the plot.
- **Epistemic uncertainty exactly zero for a stochastic method?** Your samples are
  identical: dropout never got switched back on, or every ensemble member shares an
  initialisation.
- **Bayes by Backprop predicting a flat line with enormous error bars?** The KL is
  overweighted. Check the `/ n_train`.
- **Do not raise the thread count.** Importing `bdl` caps PyTorch at 4 threads on
  purpose. These models are small, and on a many-core machine the default is about
  50× slower.
