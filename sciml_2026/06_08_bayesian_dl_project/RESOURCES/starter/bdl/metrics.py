"""Scoring functions.

Provided: the ones whose implementation is fiddly (a log-sum-exp over a mixture,
a bisection of a mixture CDF, a rank-sum with correct tie handling). You write
the ones that carry the ideas -- the uncertainty decomposition and the calibration
error -- in the notebooks.

Predictions are plain tensors throughout, with the sample axis first:

    regression      mu, sigma  of shape [S, N]
    classification  probs      of shape [S, N, K]

S is the number of sampled networks. It is 1 for the deterministic baseline, T
for MC Dropout, M for an ensemble. Keeping that axis until the last moment is
what lets you separate the two kinds of uncertainty.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# --------------------------------------------------------------------------
# Accuracy-style metrics
# --------------------------------------------------------------------------


def rmse(mu: torch.Tensor, y: torch.Tensor) -> float:
    """Root mean squared error of the predictive mean. `mu` is [S, N]."""
    return float(torch.sqrt(((mu.mean(0) - y.reshape(-1)) ** 2).mean()))


def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Accuracy of the mean predicted class distribution. `probs` is [S, N, K]."""
    return float((probs.mean(0).argmax(-1) == y.reshape(-1)).float().mean())


def predictive_nll(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> float:
    """Negative log predictive density for regression, averaged over test points.

    This is the headline number for probabilistic prediction: unlike RMSE it
    punishes a model for being confidently wrong, and unlike ECE it also punishes
    a model for being wrong in the first place.

    The predictive distribution is a mixture of the S sampled Gaussians, not a
    single Gaussian:

        p(y | x) = (1/S) sum_s N(y; mu_s, sigma_s^2)

    so the log density needs a log-sum-exp. Averaging the log densities instead
    would compute the NLL of nothing in particular; by Jensen it is an upper bound
    on this one, so it would penalise every method that has a sample disagreeing
    with the rest.
    """
    y = y.reshape(1, -1)
    log_p = (
        -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2
    )  # [S, N]
    log_mix = torch.logsumexp(log_p, dim=0) - math.log(mu.shape[0])
    return float(-log_mix.mean())


def predictive_nll_probs(probs: torch.Tensor, y: torch.Tensor) -> float:
    """Negative log predictive probability for classification.

    A mixture of categorical distributions is itself categorical, so here
    averaging the probabilities *is* the mixture and no log-sum-exp is needed.
    """
    p = probs.mean(0)  # [N, K]
    idx = y.reshape(-1)
    return float(-torch.log(p[torch.arange(len(idx)), idx].clamp_min(1e-12)).mean())


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def interval_coverage(
    mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor, level: float = 0.95
) -> float:
    """Fraction of targets inside the central `level` predictive interval.

    A well-calibrated model returns about `level`. Below it the model is
    overconfident (error bars too tight); above it, underconfident.

    The interval is found by bisecting the mixture CDF, so this is exact for a
    mixture of Gaussians rather than assuming a single Gaussian.
    """
    lo_q, hi_q = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    lo = _mixture_quantile(mu, sigma, lo_q)
    hi = _mixture_quantile(mu, sigma, hi_q)
    y = y.reshape(-1)
    return float(((y >= lo) & (y <= hi)).float().mean())


def _mixture_quantile(
    mu: torch.Tensor, sigma: torch.Tensor, q: float, iters: int = 60
) -> torch.Tensor:
    """Quantile of the Gaussian-mixture predictive, per test point, by bisection."""
    lo = (mu - 12 * sigma).min(dim=0).values
    hi = (mu + 12 * sigma).max(dim=0).values
    normal = torch.distributions.Normal(0.0, 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = normal.cdf((mid.unsqueeze(0) - mu) / sigma).mean(dim=0)
        too_low = cdf < q
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Out-of-distribution detection
# --------------------------------------------------------------------------


def auroc(scores_id: torch.Tensor, scores_ood: torch.Tensor) -> float:
    """Area under the ROC curve for separating shifted points from test points.

    Scores must be higher for more uncertain. 0.5 is chance: the uncertainty
    carries no information about whether the input is shifted. 1.0 means every
    shifted point scores above every test point.

    Computed via the rank-sum (Mann-Whitney U) identity, which handles ties by
    averaging them. Note what that means for a score that is constant: every pair
    is a tie and the answer is exactly 0.5.
    """
    s_id = np.asarray(scores_id.detach().cpu()).ravel()
    s_ood = np.asarray(scores_ood.detach().cpu()).ravel()
    n_id, n_ood = len(s_id), len(s_ood)
    if n_id == 0 or n_ood == 0:
        return float("nan")
    combined = np.concatenate([s_id, s_ood])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # average ranks within ties
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    rank_sum_ood = ranks[n_id:].sum()
    return float((rank_sum_ood - n_ood * (n_ood + 1) / 2) / (n_id * n_ood))


# --------------------------------------------------------------------------
# One metrics row
# --------------------------------------------------------------------------


def evaluate(
    pred_id,
    y_id: torch.Tensor,
    pred_ood,
    y_ood: torch.Tensor | None,
    *,
    task: str,
    decompose,
    calibration,
) -> dict[str, float]:
    """Every core metric for one method, as a flat dict.

    Used from notebook 02 onwards so that all four methods are scored identically.
    It needs the two functions you wrote yourself:

        decompose     your decompose_variance (regression) or decompose_entropy
                      (classification) from notebook 02
        calibration   your calibration_error from notebook 01

    `pred_id` and `pred_ood` are `(mu, sigma)` for regression, `(probs,)` for
    classification, with the sample axis first.

        row = evaluate((mu, sd), ds.y_test, (mu_ood, sd_ood), ds.y_ood,
                       task=ds.task, decompose=decompose_variance,
                       calibration=calibration_error)
    """
    out: dict[str, float] = {}

    if task == "regression":
        mu, sigma = pred_id
        mu_ood, sigma_ood = pred_ood
        out["nll"] = predictive_nll(mu, sigma, y_id)
        out["rmse"] = rmse(mu, y_id)
        out["ece"] = calibration(mu, sigma, y_id)
        out["coverage@95"] = interval_coverage(mu, sigma, y_id, 0.95)
        _, ale, epi = decompose(mu, sigma)
        _, _, epi_ood = decompose(mu_ood, sigma_ood)
        if y_ood is not None:
            out["nll_ood"] = predictive_nll(mu_ood, sigma_ood, y_ood)
    else:
        (probs,) = pred_id
        (probs_ood,) = pred_ood
        out["nll"] = predictive_nll_probs(probs, y_id)
        out["accuracy"] = accuracy(probs, y_id)
        out["ece"] = calibration(probs, y_id)
        _, ale, epi = decompose(probs)
        _, _, epi_ood = decompose(probs_ood)

    out["aleatoric_id"] = float(ale.mean())
    out["epistemic_id"] = float(epi.mean())
    out["epistemic_ood"] = float(epi_ood.mean())
    # How much does the model's uncertainty about its own weights grow when it
    # leaves the training distribution? A value near 1 means it did not notice.
    out["epistemic_ratio"] = out["epistemic_ood"] / max(out["epistemic_id"], 1e-12)
    out["ood_auroc"] = auroc(epi, epi_ood)
    return out


# --------------------------------------------------------------------------
# The results table
# --------------------------------------------------------------------------

METRIC_ORDER = [
    "nll",
    "rmse",
    "accuracy",
    "ece",
    "coverage@95",
    "aleatoric_id",
    "epistemic_id",
    "epistemic_ood",
    "epistemic_ratio",
    "ood_auroc",
    "nll_ood",
    "train_s",
]


def results_table(results: dict[str, dict[str, float]]) -> str:
    """Render {method: metrics} as a fixed-width table for the report."""
    cols = [m for m in METRIC_ORDER if any(m in r for r in results.values())]
    w = max((len(k) for k in results), default=6) + 2
    head = "method".ljust(w) + "".join(c.rjust(17) for c in cols)
    lines = [head, "-" * len(head)]
    for name, res in results.items():
        row = name.ljust(w)
        for c in cols:
            row += (f"{res[c]:.4f}" if c in res else "-").rjust(17)
        lines.append(row)
    return "\n".join(lines)
