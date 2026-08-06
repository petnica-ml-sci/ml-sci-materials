# Grading rubric — Honest Error Bars

**Total: 100 points.** Roughly 15 hours of work.

More than a third of the marks are for interpretation and honesty rather than for
working code. Correctness is checked mechanically by the check cells in the
notebooks. Once your code is right, what distinguishes submissions is whether you
understood what the numbers were telling you, including when they said something
inconvenient.

**How correctness is graded.** Submitted notebooks are re-executed:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/0[1-5]*.ipynb --output-dir /tmp/check
```

A failed check cell stops execution, so a submission either runs clean or it does
not. Submit your notebooks **with their outputs saved**.

---

## 1. Core correctness — 35 points

| | Points |
|---|---|
| `calibration_error`, correct for your own task type (notebook 01) | 5 |
| `predict_mc_dropout`: dropout active at test time, `T` samples returned (notebook 02) | 5 |
| `decompose_variance`, an exact identity; plus `decompose_entropy` on Track C (notebook 02) | 5 |
| `fit_ensemble`: `M` members with genuinely different initialisations (notebook 03) | 5 |
| `BayesLinear`: reparameterized sampling, and `log sigma` receiving gradient (notebook 04) | 6 |
| `kl_divergence`: the closed-form Gaussian KL, correct to a factor (notebook 04) | 5 |
| `elbo_loss`: the KL scaled by `1 / n_train`, and `weight_decay=0` when fitting (notebook 04) | 4 |

**Partial credit:** code that runs and produces sensible uncertainty but fails one
edge case in a check cell loses at most half of that row. A cell left raising
`NotImplementedError` scores zero for its row.

The check cells are the reference. If every one prints `OK`, this section is close
to full marks. There is no credit for passing a check cell by special-casing its
inputs.

---

## 2. Evaluation quality — 20 points

| | Points |
|---|---|
| Comparison table produced for **both** `toy1d` and your chosen track | 4 |
| The three required figures present, legible and captioned | 5 |
| The four questions in notebook 05 answered from your own numbers | 5 |
| Metrics used correctly in the discussion: NLL, RMSE, ECE and AUROC are not treated as interchangeable | 6 |

Common ways to lose points here:

- Reading the ECE of a model whose accuracy is poor as good news. A model that
  predicts the marginal distribution everywhere is almost perfectly calibrated and
  useless.
- Reporting `ood_auroc = 0.500` for the baseline as a result rather than
  recognising it as arithmetic: with zero epistemic uncertainty every point has the
  same score, so every pair is a tie.
- Comparing epistemic uncertainty *magnitudes* across methods. They are not on a
  common scale. Compare shapes, rankings and ratios.

---

## 3. Bonus notebook — 15 points

At least one is required. Only the best one is graded; extra ones can offset losses
elsewhere up to the 100-point cap.

| | Points |
|---|---|
| The implementation is correct and its check cells pass | 7 |
| The experiment is run properly: enough seeds, a control, a stated budget | 4 |
| The result is interpreted, not just plotted | 4 |

Bonus-specific expectations:

- **Notebook 06 part (a)** must report the full-covariance control before drawing
  any conclusion about mean-field. Without it the measured shrinkage could be
  under-convergence, and at 2000 optimisation steps it partly is.
- **Notebook 06 part (b)** must check that both estimators have the same mean. A
  variance comparison between estimators of different quantities means nothing.
- **Notebook 07** must include the random baseline and at least five seeds, and must
  compare the strategy gap against the seed-to-seed spread.
- **Notebook 08** must state which parts ran on a GPU and at what scale.

---

## 4. Open-ended experiment — 20 points

| | Points |
|---|---|
| A specific, answerable question, not "I tried some hyperparameters" | 4 |
| An explicit hypothesis, stated before the result | 3 |
| Sound design: one variable at a time, a control, multiple seeds where the effect is small | 7 |
| Honest interpretation, including limits and negative results | 6 |

Full marks are available for a refuted hypothesis. They are not available for a
confirmed one that was never at risk, or for a claim the evidence does not support.
If your effect is smaller than your error bars, the correct conclusion is that the
experiment could not detect a difference. Write that and you keep the points.

Ways to lose points: changing three things at once; reporting a single seed for a
noisy quantity; concluding "method X is better" from one track; quietly dropping an
experiment that did not work.

---

## 5. Reproducibility and clarity — 10 points

| | Points |
|---|---|
| The notebooks run top to bottom on a fresh kernel, with outputs saved | 4 |
| Seeds and splits unchanged; the split fingerprints match the reference | 3 |
| Cells are readable and consistent with the surrounding style; no dead experiments left in | 3 |

The split fingerprint printed in notebooks 01 and 05 is a cheap integrity check. If
yours differs from everyone else's on the same track, you are not evaluating on the
same data and your table is not comparable. Find out why before submitting.

---

## What an excellent submission looks like

Not "all four methods worked and deep ensembles won". Rather:

> All check cells pass. The comparison shows deep ensembles with the best RMSE on
> Track A but 95% intervals covering 93% of the test set, while Bayes by Backprop
> fits visibly worse and covers 96%. I traced the difference to the heteroscedastic
> head: under weight noise Bayes by Backprop cannot fit precisely, so it inflates
> `sigma(x)`, and part of what it books as aleatoric uncertainty is really
> epistemic. Figure 3 shows the estimated `sigma(x)` against the known truth on
> `toy1d`, where it overestimates roughly threefold.
>
> I hypothesised that uncertainty-based acquisition would beat random sampling. It
> did not: random was better by 0.24 nats against a seed spread of 0.15, and all
> five paired per-seed differences favoured random. The acquisition function
> concentrates on the edges of the input range, which on this problem is also where
> the observation noise is largest, so it spends its budget on points that are
> irreducibly noisy rather than informative.
>
> I would not trust any of these models for a supernova beyond z = 0.5. Every method
> reports a shifted NLL at least an order of magnitude worse than its
> in-distribution value, and the ensemble, which wins every in-distribution metric,
> is among the worst of them out there.

Specific claims, traced to evidence, with a refuted hypothesis reported as plainly
as a confirmed one.
