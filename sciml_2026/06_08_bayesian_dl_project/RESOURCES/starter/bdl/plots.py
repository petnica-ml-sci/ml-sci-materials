"""Figures.

Provided. Restyle them if you like, but do not write plotting code from scratch:
your time is better spent on the models.

Every function takes plain arrays, not model objects, so you pass in whatever you
computed yourself. The three figures the core task asks for come from:

    plot_bands             the predictive bands, all methods side by side
    plot_reliability       the calibration diagram
    plot_uncertainty_vs_x  epistemic uncertainty against distance from the data
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import Dataset, toy1d_noise_std, toy1d_truth

ALEATORIC_COLOR = "#f0a202"
EPISTEMIC_COLOR = "#3f7cac"
DATA_COLOR = "#20232a"


def _np(a) -> np.ndarray:
    """Accept a tensor or an array, return a flat numpy array."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a).ravel()


# --------------------------------------------------------------------------
# 1-D predictive bands
# --------------------------------------------------------------------------


def plot_band(
    ax: plt.Axes,
    x: torch.Tensor | np.ndarray,
    mean: torch.Tensor | np.ndarray,
    sd_aleatoric: torch.Tensor | np.ndarray,
    sd_total: torch.Tensor | np.ndarray,
    ds: Dataset | None = None,
    title: str = "",
    show_truth: bool = True,
    n_std: float = 2.0,
) -> None:
    """Mean prediction with its two error bars, on one set of axes.

    The inner band is aleatoric only: the noise the model thinks is in the
    measurement. The outer band adds the epistemic part: the model's uncertainty
    about its own weights. All three arguments are per-x-point standard
    deviations, not variances.

    What to look for:

      * inside the training gap the outer band should grow while the inner one
        stays put, because the model has no data there;
      * on the right-hand side, where the true noise is larger, the inner band
        should widen, because that noise is irreducible;
      * beyond the training range both should grow.

    A band of roughly constant width everywhere has told you nothing.
    """
    x = _np(x)
    mean, sd_ale, sd_tot = _np(mean), _np(sd_aleatoric), _np(sd_total)

    ax.fill_between(
        x,
        mean - n_std * sd_tot,
        mean + n_std * sd_tot,
        color=EPISTEMIC_COLOR,
        alpha=0.30,
        lw=0,
        label=f"±{n_std:g}σ total",
    )
    ax.fill_between(
        x,
        mean - n_std * sd_ale,
        mean + n_std * sd_ale,
        color=ALEATORIC_COLOR,
        alpha=0.55,
        lw=0,
        label=f"±{n_std:g}σ aleatoric",
    )
    ax.plot(x, mean, color=EPISTEMIC_COLOR, lw=2.0, label="predictive mean")

    if show_truth:
        ax.plot(x, toy1d_truth(x), "--", color="k", lw=1.2, alpha=0.7, label="true function")

    if ds is not None:
        ax.plot(
            _np(ds.x_train),
            _np(ds.y_train),
            ".",
            color=DATA_COLOR,
            ms=4,
            alpha=0.55,
            label="training data",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-4.5, 4.5)


def plot_bands(
    panels: dict[str, dict[str, np.ndarray]],
    x: torch.Tensor | np.ndarray,
    ds: Dataset,
    path: str | Path | None = None,
    suptitle: str = "Predictive uncertainty on toy1d",
) -> plt.Figure:
    """Several methods' predictive bands side by side. Required figure 1.

    `panels` maps a method name to a dict with keys `mean`, `sd_aleatoric` and
    `sd_total`, exactly the arguments of `plot_band`.
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), sharey=True, squeeze=False)
    for ax, (name, p) in zip(axes[0], panels.items(), strict=True):
        plot_band(ax, x, p["mean"], p["sd_aleatoric"], p["sd_total"], ds, title=name)
    axes[0][-1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return _save(fig, path)


def plot_uncertainty_vs_x(
    curves: dict[str, np.ndarray],
    x: torch.Tensor | np.ndarray,
    ds: Dataset,
    path: str | Path | None = None,
    gap: tuple[float, float] = (-0.5, 1.5),
    x_range: tuple[float, float] = (-3.0, 3.0),
) -> plt.Figure:
    """Epistemic standard deviation against x, with the gaps shaded. Required figure 3.

    `curves` maps a method name to its per-x epistemic standard deviation.

    This is the quantitative version of "does the model know what it does not
    know". The shaded regions have no training data, and an honest method peaks
    there. Compare the shape across methods rather than the absolute scale: the
    methods are not on a common scale and do not need to be.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = _np(x)
    for name, epi_std in curves.items():
        ax.plot(x, _np(epi_std), lw=1.8, label=name)

    for lo, hi in [(x.min(), x_range[0]), gap, (x_range[1], x.max())]:
        ax.axvspan(lo, hi, color="0.85", zorder=0)
    ax.plot(
        _np(ds.x_train),
        np.zeros(len(ds.x_train)),
        "|",
        color=DATA_COLOR,
        ms=8,
        alpha=0.5,
        label="training inputs",
    )
    ax.set_xlabel("x   (shaded = no training data)")
    ax.set_ylabel("epistemic std")
    ax.set_yscale("log")
    ax.set_title("Does uncertainty grow where the data stops?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_aleatoric_recovery(
    curves: dict[str, np.ndarray],
    x: torch.Tensor | np.ndarray,
    path: str | Path | None = None,
) -> plt.Figure:
    """Estimated against true observation noise on toy1d.

    `curves` maps a method name to its per-x aleatoric standard deviation. The
    true sigma(x) is known for this dataset, so you can check whether the
    heteroscedastic head learned it or absorbed something else into it.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = _np(x)
    ax.plot(x, toy1d_noise_std(x), "k--", lw=2, label="true σ(x)")
    for name, sd_ale in curves.items():
        ax.plot(x, _np(sd_ale), lw=1.6, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("aleatoric std")
    ax.set_title("Is the estimated observation noise right?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Calibration and OOD detection
# --------------------------------------------------------------------------


def plot_reliability(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: str | Path | None = None,
    task: str = "regression",
) -> plt.Figure:
    """Reliability diagram. Required figure 2.

    `curves` maps a method name to (x, y) arrays:

        regression      nominal credible level, empirical coverage
        classification  mean predicted confidence per bin, accuracy per bin

    The diagonal is perfect calibration. Below it means overconfident: the model
    claims more certainty than it has earned.
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="perfect calibration")

    for name, (xs, ys) in curves.items():
        ax.plot(_np(xs), _np(ys), "o-", ms=4, lw=1.6, label=name)

    ax.set_xlabel("predicted confidence" if task == "classification" else "nominal credible level")
    ax.set_ylabel("observed accuracy" if task == "classification" else "empirical coverage")
    ax.set_title("Reliability diagram\n(below the diagonal = overconfident)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


def confidence_bins(
    probs: torch.Tensor, y: torch.Tensor, n_bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Mean confidence and accuracy per bin, for a classification reliability curve.

    `probs` is [S, N, K]. Returns (confidence, accuracy) for the non-empty bins,
    ready to hand to `plot_reliability`.
    """
    p = probs.mean(0)
    conf, hat = p.max(dim=-1)
    correct = (hat == y.reshape(-1)).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if bool(m.any()):
            xs.append(float(conf[m].mean()))
            ys.append(float(correct[m].mean()))
    return np.array(xs), np.array(ys)


def plot_ood_histogram(
    scores_id: torch.Tensor | np.ndarray,
    scores_ood: torch.Tensor | np.ndarray,
    path: str | Path | None = None,
    name: str = "",
) -> plt.Figure:
    """Epistemic score on test inputs against shifted inputs.

    The AUROC in your results table is a one-number summary of this picture.
    Overlapping histograms mean the model cannot tell it has left the training
    distribution.
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    s_id, s_ood = _np(scores_id), _np(scores_ood)
    bins = np.histogram_bin_edges(np.concatenate([s_id, s_ood]), bins=40)
    ax.hist(s_id, bins=bins, alpha=0.6, label="test (in-distribution)", color=EPISTEMIC_COLOR)
    ax.hist(s_ood, bins=bins, alpha=0.6, label="shifted", color=ALEATORIC_COLOR)
    ax.set_xlabel("epistemic uncertainty")
    ax.set_ylabel("count")
    ax.set_title(f"Does the model notice it left the training data? {name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------
# Training and bonus figures
# --------------------------------------------------------------------------


def plot_loss_history(
    histories: dict[str, list[float]], path: str | Path | None = None
) -> plt.Figure:
    """Training curves. A debugging aid: a flat ELBO means trouble."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, h in histories.items():
        ax.plot(h, lw=1.5, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (negative ELBO where applicable)")
    ax.set_yscale("symlog")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_learning_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str | Path | None = None,
    xlabel: str = "number of labelled points",
    ylabel: str = "test NLL",
    title: str = "Active learning",
) -> plt.Figure:
    """Learning curves with error bands, for the active-learning bonus.

    `curves` maps a strategy name to (x, mean, std) arrays. The std should be
    across seeds: a single-seed active learning curve says almost nothing, because
    the difference between acquisition strategies is usually smaller than the
    seed-to-seed spread.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, (x, mean, std) in curves.items():
        ax.plot(x, mean, "o-", ms=4, lw=1.8, label=name)
        ax.fill_between(x, mean - std, mean + std, alpha=0.20, lw=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_shift_sweep(
    results: dict[str, dict[str, list[float]]],
    strengths: list[float],
    path: str | Path | None = None,
    metrics: tuple[str, ...] = ("nll", "ece", "ood_auroc"),
) -> plt.Figure:
    """Metric against corruption strength, one panel per metric."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        values = []
        for name, series in results.items():
            if metric in series:
                ax.plot(strengths, series[metric], "o-", ms=4, lw=1.7, label=name)
                values += list(series[metric])
        ax.set_xlabel("shift strength (training std)")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        # NLL under shift spans several orders of magnitude; on a linear axis
        # every point but the last sits flat against the bottom.
        finite = [v for v in values if v > 0 and np.isfinite(v)]
        if finite and max(finite) / min(finite) > 50:
            ax.set_yscale("log")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    """Save the figure if a path was given, and return it so a notebook shows it."""
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plots] wrote {path}")
    return fig
