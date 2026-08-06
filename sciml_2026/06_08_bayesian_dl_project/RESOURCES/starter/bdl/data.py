"""Datasets: the toy problem and the three domain tracks.

Provided. Read `make_toy1d` and `load_track` so you know what your models are
being fed.

Every track returns the same `Dataset` object, so nothing else in the project
needs to know which track you picked:

    ds = load_track("A")
    ds.x_train, ds.y_train      # training data
    ds.x_test,  ds.y_test       # test data, same distribution as training
    ds.x_ood,   ds.y_ood        # shifted data
    ds.task                     # "regression" or "classification"

Nothing under `data/` is committed. Each track downloads what it needs the first
time you use it and checks it against a pinned SHA-256. To fetch everything up
front, before working offline:

    uv run python -m bdl.data --all

The shifted split is never a random split. In every track it is a specific,
scientifically meaningful shift:

    Track A   supernovae more distant than any in the training set
    Track B   molecules heavier than any in the training set
    Track C   the same images with sensor noise added

A model that is honest about its uncertainty should become less confident on
`x_ood`. Most models do not.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# One seed for the data splits, shared by everyone, so results are comparable
# across submissions. Do not change this.
SPLIT_SEED = 20260807


@dataclass
class Dataset:
    """A track's data, standardised and converted to tensors.

    x_train, y_train : training inputs and targets
    x_test,  y_test  : test inputs and targets, same distribution as training
    x_ood,   y_ood   : shifted inputs and targets (y is for reporting only,
                       never for fitting)
    task             : "regression" or "classification"
    n_features       : input dimensionality (flattened, for images)
    n_outputs        : 1 for regression, n_classes for classification
    y_mean, y_std    : standardisation constants for the regression target, so
                       predictions can be converted back to physical units
    name             : human-readable track name
    x_stats          : (mean, std) used to standardise the inputs, so extra data
                       can be put on the same scale later
    """

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    x_ood: torch.Tensor
    y_ood: torch.Tensor | None
    task: str
    n_features: int
    n_outputs: int
    y_mean: float
    y_std: float
    name: str
    x_stats: tuple[np.ndarray, np.ndarray] | None = None

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Dataset({self.name}, task={self.task}, "
            f"train={tuple(self.x_train.shape)}, test={tuple(self.x_test.shape)}, "
            f"ood={tuple(self.x_ood.shape)}, n_outputs={self.n_outputs})"
        )


# --------------------------------------------------------------------------
# The toy problem: 1-D regression with a gap
# --------------------------------------------------------------------------


def toy1d_truth(x: np.ndarray) -> np.ndarray:
    """The true mean function."""
    return np.sin(2.0 * x) + 0.3 * x


def toy1d_noise_std(x: np.ndarray) -> np.ndarray:
    """The true observation-noise standard deviation, which depends on x.

    This is aleatoric uncertainty: scatter in the measurement itself. It grows
    towards the right of the domain, as it would for an instrument with a
    constant relative error. More data does not remove it.
    """
    return 0.05 + 0.15 * np.clip(x + 3.0, 0.0, None) / 6.0


def make_toy1d(
    n_train: int = 200,
    n_test: int = 400,
    gap: tuple[float, float] = (-0.5, 1.5),
    x_range: tuple[float, float] = (-3.0, 3.0),
    x_eval_range: tuple[float, float] = (-5.0, 5.0),
    seed: int = SPLIT_SEED,
) -> Dataset:
    """1-D regression with heteroscedastic noise and a hole in the data.

    Two things are built in, and a good uncertainty method reacts to both:

    * inside `gap` there is no training data, so a model should be uncertain
      there because it was told nothing (epistemic);
    * the noise level grows with x, so a model should be uncertain on the right
      even where it has plenty of data (aleatoric).

    Splits: `x_test` comes from the same distribution as `x_train` (gap
    excluded), so test metrics measure what they claim to. `x_ood` is the gap
    plus the regions beyond the training range. Use `toy1d_grid()` for plotting;
    it is a dense grid, not a data split.
    """
    rng = np.random.default_rng(seed)

    def sample_outside_gap(n: int) -> np.ndarray:
        a = rng.uniform(x_range[0], x_range[1], size=int(n * 2.0))
        return np.sort(a[(a < gap[0]) | (a > gap[1])][:n])

    def observe(a: np.ndarray) -> np.ndarray:
        return toy1d_truth(a) + rng.normal(0.0, toy1d_noise_std(a))

    x = sample_outside_gap(n_train)
    y = observe(x)

    # The test set is drawn from the same distribution as the training set, gap
    # excluded. If the gap were included here, the headline test metrics would
    # be measuring extrapolation and the two failure modes could not be told
    # apart.
    x_test = sample_outside_gap(n_test)
    y_test = observe(x_test)

    # The shifted set is everything the training data did not cover: the gap in
    # the middle (interpolation into a hole) and the regions beyond the training
    # range on either side (extrapolation).
    n_ood = n_test // 2
    x_ood = np.concatenate(
        [
            np.linspace(x_eval_range[0], x_range[0], n_ood // 3),
            np.linspace(gap[0], gap[1], n_ood // 3),
            np.linspace(x_range[1], x_eval_range[1], n_ood // 3),
        ]
    )
    y_ood = observe(x_ood)

    def col(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32).reshape(-1, 1)

    return Dataset(
        x_train=col(x),
        y_train=col(y),
        x_test=col(x_test),
        y_test=col(y_test),
        x_ood=col(x_ood),
        y_ood=col(y_ood),
        task="regression",
        n_features=1,
        n_outputs=1,
        y_mean=0.0,  # toy1d is left in its natural units
        y_std=1.0,
        name="toy1d",
    )


def toy1d_grid(
    x_eval_range: tuple[float, float] = (-5.0, 5.0), n: int = 500
) -> torch.Tensor:
    """A dense input grid for plotting predictive bands. Shape [n, 1]."""
    return torch.linspace(x_eval_range[0], x_eval_range[1], n).reshape(-1, 1)


# --------------------------------------------------------------------------
# Track A -- astrophysics: Type Ia supernovae as standard candles
# --------------------------------------------------------------------------

#: Redshift above which a supernova is treated as shifted.
TRACK_A_Z_SPLIT = 0.5

TRACK_A_FEATURES = ["zHD", "x1", "c", "HOST_LOGMASS", "MWEBV"]
TRACK_A_TARGET = "mB"


def _load_pantheon():
    """Read the Pantheon+ release table (space-delimited, 1701 SNe x 47 columns)."""
    import pandas as pd

    ensure_tabular()  # downloads and checks on first use, no-op afterwards
    return pd.read_csv(DATA_DIR / "pantheon.dat", sep=r"\s+")


def _make_track_a(seed: int) -> Dataset:
    """Pantheon+ Type Ia supernovae: how bright is a supernova, and how far away?

    A Type Ia supernova is close to a standard candle: they all explode with
    roughly the same intrinsic brightness, so how bright one looks tells you how
    far away it is. Correcting the residual differences between them is what this
    dataset is about.

    Target
        `mB`, the peak apparent brightness in the B band, in magnitudes.
        Magnitudes run backwards: a bigger number means fainter and further.

    Features
        `zHD`           redshift, the primary distance indicator
        `x1`            light-curve stretch; slower-declining supernovae are
                        intrinsically brighter
        `c`             colour; redder supernovae appear fainter
        `HOST_LOGMASS`  stellar mass of the host galaxy, which correlates with
                        brightness
        `MWEBV`         dust extinction in our own galaxy along the line of sight

    Learning `mB` from these is the Tripp standardisation, the relation that
    turned supernovae into a distance ladder. The stretch and colour terms cut
    the residual scatter on this split by about 46%.

    Three of the 47 columns are the answer and are excluded: `x0` (the target by
    construction, `mB = -2.5 log10(x0) + const`), `m_b_corr` (the target with the
    corrections already applied, correlation 0.996) and `MU_SH0ES` (the distance
    modulus derived from it, correlation 0.996). Using any of them would be
    target leakage. Note also that `m_b_corr` is already stretch- and
    colour-corrected, so predicting it would make `x1` and `c` redundant.

    The 77 Cepheid calibrators (`IS_CALIBRATOR`) are dropped: their distances are
    known independently rather than inferred, so they do not belong in a sample
    where distance is what you are trying to learn.

    Shifted split: the most distant supernovae, `zHD > 0.5`. Nearby supernovae
    constrain the brightness-distance relation well; whether it still holds
    further out is the question that led to the discovery of dark energy. A model
    trained only on nearby supernovae has to continue a logarithmic curve into
    territory it has never seen, and it will do so confidently.

    One caveat: no cosmologist estimates distances by fitting a neural network to
    `mB`. A real analysis fits a cosmological model and uses the covariance
    matrix of correlated systematics that ships with this release, which we
    ignore. The subject here is what happens to an uncertainty estimate under
    extrapolation, not the astronomy.
    """
    df = _load_pantheon()
    df = df[df["IS_CALIBRATOR"] <= 0]  # keep only distance-inferred supernovae

    x = df[TRACK_A_FEATURES].to_numpy(dtype=np.float64)
    y = df[TRACK_A_TARGET].to_numpy(dtype=np.float64)
    z = df["zHD"].to_numpy(dtype=np.float64)

    ood_mask = z > TRACK_A_Z_SPLIT  # the ~13% most distant supernovae
    x_id, y_id = x[~ood_mask], y[~ood_mask]
    x_ood, y_ood = x[ood_mask], y[ood_mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_id))
    n_train = int(0.8 * len(x_id))
    tr, te = perm[:n_train], perm[n_train:]

    return _standardise_regression(
        x_id[tr], y_id[tr], x_id[te], y_id[te], x_ood, y_ood, name="A:pantheon"
    )


# --------------------------------------------------------------------------
# Track B -- chemistry: aqueous solubility (ESOL / Delaney)
# --------------------------------------------------------------------------


def _make_track_b(seed: int) -> Dataset:
    """ESOL aqueous solubility: 1128 molecules, 6 precomputed descriptors.

    Target: measured log solubility (log mol/L).

    The raw file also contains a column called "ESOL predicted log solubility",
    which is another model's prediction of the target. Using it as a feature
    would be target leakage, so it is dropped here.

    Shifted split: the heaviest molecules. Descriptor-based solubility models
    degrade outside the molecular-weight range they were fitted on, and a chemist
    asking a model about a large novel compound is exactly the case where an
    honest error bar matters.
    """
    import pandas as pd

    ensure_tabular()
    df = pd.read_csv(DATA_DIR / "esol.csv")
    feature_names = [
        "Minimum Degree",
        "Molecular Weight",
        "Number of H-Bond Donors",
        "Number of Rings",
        "Number of Rotatable Bonds",
        "Polar Surface Area",
    ]
    x = df[feature_names].to_numpy(dtype=np.float64)
    y = df["measured log solubility in mols per litre"].to_numpy(dtype=np.float64)

    mw = df["Molecular Weight"].to_numpy(dtype=np.float64)
    ood_mask = mw > np.quantile(mw, 0.92)

    x_id, y_id = x[~ood_mask], y[~ood_mask]
    x_ood, y_ood = x[ood_mask], y[ood_mask]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_id))
    n_train = int(0.8 * len(x_id))
    tr, te = perm[:n_train], perm[n_train:]

    return _standardise_regression(
        x_id[tr], y_id[tr], x_id[te], y_id[te], x_ood, y_ood, name="B:esol"
    )


# --------------------------------------------------------------------------
# Track C -- biology / imaging
# --------------------------------------------------------------------------

#: Standard deviation of the sensor noise that defines Track C's shifted set.
CORRUPTION_STD = 0.5


def _make_track_c(seed: int, n_train: int = 6000, n_test: int = 2000) -> Dataset:
    """Small-image classification with a degraded-acquisition shift.

    Preferred data: MedMNIST `bloodmnist`, eight classes of peripheral blood
    cell, 28x28 RGB, a ~35 MB download cached under `data/medmnist/`. If
    `medmnist` is not installed it falls back to FashionMNIST. Every metric
    behaves the same either way, so no part of the assignment depends on which
    you get.

    The shifted set is the same test images with Gaussian sensor noise added
    (`CORRUPTION_STD`). This is the imaging analogue of Track A's more distant
    supernovae: the subject matter is unchanged, but the measurement is degraded
    in a way the training set never showed. It also has a strength knob,
    `corrupt()` below, which the shift sweep in the bonus notebook uses.

    A harder shift is available via `load_far_ood()`: images from a different
    MedMNIST modality. Most methods score below chance on it, which is a real
    phenomenon rather than a bug in your code, and it makes good material for the
    open-ended part of the report.

    Images are flattened and standardised with training statistics, so the MLP in
    `models.py` consumes them directly.
    """
    try:
        x_tr, y_tr, x_te, y_te, name = _load_bloodmnist(seed, n_train, n_test)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
        print(f"[data] medmnist unavailable ({type(exc).__name__}); falling back to FashionMNIST")
        x_tr, y_tr, x_te, y_te, name = _load_fashion(seed, n_train, n_test)

    rng = np.random.default_rng(seed + 1)
    x_ood = np.clip(x_te + rng.normal(0, CORRUPTION_STD, x_te.shape), 0.0, 1.0).astype(np.float32)
    return _as_classification(x_tr, y_tr, x_te, y_te, x_ood, name=f"{name}+noise")


def _load_bloodmnist(seed: int, n_train: int, n_test: int):
    import medmnist
    from medmnist import INFO

    root = DATA_DIR / "medmnist"
    root.mkdir(parents=True, exist_ok=True)  # medmnist will not create this itself

    def load(flag: str, split: str) -> tuple[np.ndarray, np.ndarray]:
        cls = getattr(medmnist, INFO[flag]["python_class"])
        d = cls(split=split, download=True, root=str(root))
        imgs = d.imgs.astype(np.float32) / 255.0
        return imgs.reshape(len(imgs), -1), d.labels.reshape(-1).astype(np.int64)

    x_tr, y_tr = load("bloodmnist", "train")
    x_te, y_te = load("bloodmnist", "test")
    rng = np.random.default_rng(seed)
    x_tr, y_tr = _subsample(x_tr, y_tr, n_train, rng)
    x_te, y_te = _subsample(x_te, y_te, n_test, rng)
    return x_tr, y_tr, x_te, y_te, "C:bloodmnist"


def _load_fashion(seed: int, n_train: int, n_test: int):
    from torchvision import datasets

    root = str(DATA_DIR / "torchvision")

    def load(train: bool) -> tuple[np.ndarray, np.ndarray]:
        d = datasets.FashionMNIST(root=root, train=train, download=True)
        imgs = d.data.numpy().astype(np.float32) / 255.0
        return imgs.reshape(len(imgs), -1), d.targets.numpy().astype(np.int64)

    x_tr, y_tr = load(True)
    x_te, y_te = load(False)
    rng = np.random.default_rng(seed)
    x_tr, y_tr = _subsample(x_tr, y_tr, n_train, rng)
    x_te, y_te = _subsample(x_te, y_te, n_test, rng)
    return x_tr, y_tr, x_te, y_te, "C:fashionmnist"


def load_far_ood(ds: Dataset, n: int = 2000, seed: int = SPLIT_SEED) -> torch.Tensor:
    """A second, harder shifted set for Track C: a different imaging modality.

    Returns dermatoscopic skin-lesion images (or MNIST digits, matching whichever
    fallback `load_track("C")` used), standardised with the same statistics as the
    track. Optional, for the open-ended part of the report.

    Most methods assign these images lower epistemic uncertainty than genuine
    test images, giving an OOD AUROC below 0.5. That is not a bug: these images
    are smooth and low-contrast, so they sit closer to the mean input than real
    blood smears do, and every member of an ensemble confidently agrees on the
    same wrong answer.
    """
    if not ds.name.startswith("C:"):
        raise ValueError("load_far_ood is only defined for Track C")
    rng = np.random.default_rng(seed)
    if "bloodmnist" in ds.name:
        import medmnist
        from medmnist import INFO

        cls = getattr(medmnist, INFO["dermamnist"]["python_class"])
        d = cls(split="test", download=True, root=str(DATA_DIR / "medmnist"))
        x = d.imgs.astype(np.float32).reshape(len(d.imgs), -1) / 255.0
    else:
        from torchvision import datasets

        d = datasets.MNIST(root=str(DATA_DIR / "torchvision"), train=False, download=True)
        x = d.data.numpy().astype(np.float32).reshape(len(d.data), -1) / 255.0
    x, _ = _subsample(x, np.zeros(len(x), dtype=np.int64), n, rng)
    return torch.tensor((x - ds.x_stats[0]) / ds.x_stats[1], dtype=torch.float32)


def corrupt(x: torch.Tensor, strength: float, seed: int = 0) -> torch.Tensor:
    """Add Gaussian noise of the given strength, in the dataset's own units.

    Works for every track: inputs are standardised, so `strength` is measured in
    training standard deviations. `strength=0` returns the input unchanged.
    """
    if strength <= 0:
        return x
    g = torch.Generator().manual_seed(seed)
    return x + strength * torch.randn(x.shape, generator=g)


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def _subsample(
    x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if n >= len(x):
        return x, y
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx], y[idx]


def _standardise_regression(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    name: str,
) -> Dataset:
    """Standardise using training statistics only.

    Using test or shifted statistics here would leak information and would partly
    hide the distribution shift you are trying to measure.
    """
    x_mean, x_std = x_tr.mean(0), x_tr.std(0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())

    def sx(a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - x_mean) / x_std, dtype=torch.float32)

    def sy(a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - y_mean) / y_std, dtype=torch.float32).reshape(-1, 1)

    return Dataset(
        x_train=sx(x_tr),
        y_train=sy(y_tr),
        x_test=sx(x_te),
        y_test=sy(y_te),
        x_ood=sx(x_ood),
        y_ood=sy(y_ood),
        task="regression",
        n_features=x_tr.shape[1],
        n_outputs=1,
        y_mean=y_mean,
        y_std=y_std,
        name=name,
        x_stats=(x_mean, x_std),
    )


def _as_classification(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    x_ood: np.ndarray,
    name: str,
) -> Dataset:
    """Standardise pixels with training statistics and package as a Dataset.

    Standardising matters more than it looks: on raw [0, 1] pixels the learning
    rate that works for the tabular tracks diverges, and Track C would need its
    own hyperparameters for no good reason.
    """
    x_mean, x_std = x_tr.mean(0), x_tr.std(0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    n_classes = int(y_tr.max()) + 1

    def sx(a: np.ndarray) -> torch.Tensor:
        return torch.tensor((a - x_mean) / x_std, dtype=torch.float32)

    return Dataset(
        x_train=sx(x_tr),
        y_train=torch.tensor(y_tr, dtype=torch.long),
        x_test=sx(x_te),
        y_test=torch.tensor(y_te, dtype=torch.long),
        x_ood=sx(x_ood),
        y_ood=None,
        task="classification",
        n_features=x_tr.shape[1],
        n_outputs=n_classes,
        y_mean=0.0,
        y_std=1.0,
        name=name,
        x_stats=(x_mean, x_std),
    )


def load_track(track: str, seed: int = SPLIT_SEED) -> Dataset:
    """Load one of the tracks: "A" (supernovae), "B" (esol), "C" (images), "toy1d"."""
    track = track.strip().upper()
    if track == "TOY1D":
        return make_toy1d(seed=seed)
    if track == "A":
        return _make_track_a(seed)
    if track == "B":
        return _make_track_b(seed)
    if track == "C":
        return _make_track_c(seed)
    raise ValueError(f"unknown track {track!r}; expected one of A, B, C, toy1d")


def split_fingerprint(ds: Dataset) -> str:
    """A short hash of the split. Print it and compare with your classmates'.

    Two people who report different fingerprints for the same track are not
    comparing like with like, usually because a seed was changed.
    """
    h = hashlib.sha256()
    for arr in (ds.x_train, ds.y_train, ds.x_test, ds.y_test, ds.x_ood):
        h.update(np.ascontiguousarray(arr.numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------

#: filename -> (url, sha256, description)
TABULAR_SOURCES: dict[str, tuple[str, str, str]] = {
    "pantheon.dat": (
        "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease"
        "/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
        "1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8",
        "Track A -- Pantheon+ Type Ia supernovae (1701 SNe)",
    ),
    "esol.csv": (
        "https://raw.githubusercontent.com/deepchem/deepchem"
        "/master/datasets/delaney-processed.csv",
        "8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
        "Track B -- ESOL aqueous solubility (1128 molecules)",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str, url: str, expected: str, description: str, force: bool) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name

    if path.exists() and not force:
        actual = _sha256(path)
        if actual == expected:
            return path
        raise RuntimeError(
            f"{path} exists but its checksum is wrong.\n"
            f"  expected {expected}\n  found    {actual}\n"
            "Delete the file and re-run, or use --force to overwrite it."
        )

    print(f"[download] {name}: {description}")
    print(f"[download]   from {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except Exception as exc:  # noqa: BLE001 - network errors need a readable message
        raise RuntimeError(
            f"could not download {name} ({type(exc).__name__}: {exc}).\n"
            f"If you have no network access, obtain {name} another way and place it "
            f"in {DATA_DIR}."
        ) from exc

    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"{name} downloaded but the checksum does not match.\n"
            f"  expected {expected}\n  found    {actual}\n"
            "The upstream mirror has changed. Do not use this file: your results "
            "would not be comparable with anyone else's. Tell the course organisers."
        )

    path.write_bytes(payload)
    print(f"[download]   wrote {path} ({len(payload) // 1024} KB, checksum OK)")
    return path


def ensure_tabular(force: bool = False) -> None:
    """Make sure the Track A and Track B tables are present and intact.

    Called automatically by `load_track("A")` and `load_track("B")`, so the tracks
    work on a fresh clone with no manual step.
    """
    for name, (url, sha, description) in TABULAR_SOURCES.items():
        _fetch(name, url, sha, description, force)


def ensure_images(force: bool = False) -> None:
    """Pre-fetch Track C's images by loading the track once.

    MedMNIST (~53 MB) if the `med` extra is installed, otherwise FashionMNIST and
    MNIST via torchvision (~146 MB). Both libraries handle their own caching.
    """
    ds = load_track("C")
    print(f"[download] Track C ready: {ds.name}")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Download the datasets.")
    ap.add_argument("--all", action="store_true", help="also fetch Track C images (large)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    try:
        ensure_tabular(force=args.force)
        print("[download] tabular tracks (A, B) ready")
        if args.all:
            ensure_images(force=args.force)
    except RuntimeError as exc:
        print(f"\n[download] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("[download] done")


if __name__ == "__main__":
    _main()
