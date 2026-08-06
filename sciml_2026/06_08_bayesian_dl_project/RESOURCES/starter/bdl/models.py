"""The network, the output head, and the training loop.

Provided. Read `fit`: in notebook 04 you pass it your own loss function, and the
`n_train` argument it hands to that loss is the thing most people get wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Dataset

# A loss takes the model, a minibatch, and the size of the FULL training set, and
# returns a scalar to minimise. The last argument matters only for losses with a
# per-dataset term (the KL in the ELBO); the others ignore it.
LossFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, int], torch.Tensor]


# --------------------------------------------------------------------------
# The network
# --------------------------------------------------------------------------


class MLP(nn.Module):
    """A multilayer perceptron with optional dropout.

    For regression use `n_out=2`: the two outputs are the predicted mean and the
    predicted log-variance of the observation noise (see `gaussian_head`). For
    classification use `n_out=n_classes` and read the outputs as logits.

    `dropout` is applied after every hidden activation. Dropout is active only in
    training mode; turning it back on at test time is the trick behind MC Dropout,
    and `enable_dropout` below is how you do it.

    `layer` lets you swap `nn.Linear` for something else. Notebook 04 passes its
    own Bayesian layer here.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: tuple[int, ...] = (64, 64),
        dropout: float = 0.0,
        layer: type[nn.Module] = nn.Linear,
    ) -> None:
        super().__init__()
        self.n_in, self.n_out, self.dropout_p = n_in, n_out, dropout
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [layer(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(layer(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    ds: Dataset,
    hidden: tuple[int, ...] = (64, 64),
    dropout: float = 0.0,
    layer: type[nn.Module] = nn.Linear,
) -> nn.Module:
    """Build the right-shaped MLP for a track.

    Regression gets two outputs (mean and log-variance); classification gets one
    logit per class.
    """
    n_out = 2 if ds.task == "regression" else ds.n_outputs
    return MLP(ds.n_features, n_out, hidden=hidden, dropout=dropout, layer=layer)


# --------------------------------------------------------------------------
# Output head
# --------------------------------------------------------------------------

# Keeps sigma in a sane range. Without this a network can drive the log-variance
# to -inf on an easy point and produce NaN gradients.
LOG_VAR_MIN, LOG_VAR_MAX = -8.0, 4.0


def gaussian_head(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a [N, 2] regression output into (mean, std).

    The second output is a log-variance, so `sigma = exp(0.5 * log_var)`.
    Predicting the log-variance rather than the variance keeps sigma positive
    without a constraint. This is the standard heteroscedastic parameterisation.
    """
    mu = out[..., 0]
    log_var = out[..., 1].clamp(LOG_VAR_MIN, LOG_VAR_MAX)
    return mu, torch.exp(0.5 * log_var)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def gaussian_nll_loss(
    model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, n_train: int
) -> torch.Tensor:
    """Negative log-likelihood of a heteroscedastic Gaussian, averaged over the batch.

        -log p(y | x, W) = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)

    Compare this with MSE, which is what you get if sigma is held constant.
    Letting the network predict sigma is what produces an aleatoric uncertainty
    estimate.
    """
    mu, sigma = gaussian_head(model(xb))
    y = yb.reshape(mu.shape)
    return (0.5 * torch.log(2 * torch.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)).mean()


def cross_entropy_loss(
    model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, n_train: int
) -> torch.Tensor:
    """Categorical NLL, averaged over the batch."""
    return F.cross_entropy(model(xb), yb.reshape(-1))


def default_loss(ds: Dataset) -> LossFn:
    """The right loss for a track: Gaussian NLL for regression, cross-entropy otherwise."""
    return gaussian_nll_loss if ds.task == "regression" else cross_entropy_loss


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _batches(
    x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = len(x)
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield x[idx], y[idx]


def fit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_fn: LossFn,
    epochs: int = 400,
    batch_size: int = 128,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
    device: torch.device | str = "cpu",
) -> list[float]:
    """Train `model` by minimising `loss_fn`. Returns the per-epoch loss history.

    `loss_fn` is called as `loss_fn(model, xb, yb, n_train)`, where `n_train` is
    the size of the whole training set, not of the batch. A loss with a
    per-dataset term rather than a per-example term needs that number to weight it
    correctly; the KL in the ELBO is the case in point (notebook 04).

    `weight_decay` is L2 regularisation, which is the same thing as a Gaussian
    prior on the weights, so training with it gives a MAP estimate. It is the
    right default for the deterministic, dropout and ensemble methods. Set it to
    0 for Bayes by Backprop: the prior is already in the loss there, and applying
    it twice counts it twice.
    """
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_train = len(x)

    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        total, n_batches = 0.0, 0
        for xb, yb in _batches(x, y, batch_size, generator):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, xb, yb, n_train)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        history.append(total / max(n_batches, 1))
        if verbose and (epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  loss {history[-1]: .4f}")
    model.eval()
    return history


# Per-track training settings, so nobody has to tune anything by hand and the
# tracks cost about the same. Two things drive them:
#   * dataset size. Track C has 6000 points and 47 minibatches per epoch, so it
#     needs far fewer epochs than toy1d's two, and lr=1e-2 diverges on its 2352
#     inputs.
#   * Bayes by Backprop always needs more steps than the point-estimate methods,
#     because the gradient signal for the posterior widths is much weaker, and it
#     needs a smaller learning rate because the sampled-weight objective is noisy.
# Track A additionally needs a narrower initial posterior: its target is sharply
# determined (residual scatter is a few percent of the target's spread), and at
# the usual exp(-5) the initial weight noise swamps the signal, so the network
# never fits.
TRACK_HPARAMS: dict[str, dict[str, dict[str, float]]] = {
    "toy1d": {"_default": {"epochs": 400, "lr": 1e-2}, "bbb": {"epochs": 3000, "lr": 5e-3}},
    "a": {
        "_default": {"epochs": 400, "lr": 1e-2},
        "bbb": {"epochs": 3000, "lr": 5e-3, "init_log_sigma": -7.0},
    },
    "b": {"_default": {"epochs": 400, "lr": 1e-2}, "bbb": {"epochs": 3000, "lr": 5e-3}},
    "c": {
        "_default": {"epochs": 40, "lr": 1e-3},
        "bbb": {"epochs": 80, "lr": 1e-3},
    },
}


def track_hparams(track: str, method: str) -> dict[str, float]:
    """Training settings for one method on one track.

    Returns a dict with at least `epochs` and `lr`; Bayes by Backprop also gets
    `init_log_sigma` where the track needs a specific value.

        hp = track_hparams("A", "ensemble")
        fit(model, ..., epochs=hp["epochs"], lr=hp["lr"])
    """
    per_track = TRACK_HPARAMS[track.strip().lower()]
    out = dict(per_track["_default"])
    out.update(per_track.get(method, {}))
    return out


def enable_dropout(model: nn.Module) -> None:
    """Put only the dropout layers back into training mode.

    This is the MC Dropout trick. Note what it does not do: it leaves batch-norm
    and everything else in eval mode. `model.train()` would also make batch-norm
    use batch statistics, which corrupts predictions for reasons that have nothing
    to do with Bayesian inference.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int) -> None:
    """Seed torch. Call this before building a model if you want reproducibility."""
    torch.manual_seed(seed)
