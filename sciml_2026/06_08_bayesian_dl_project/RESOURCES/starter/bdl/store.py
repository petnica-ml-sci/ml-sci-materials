"""Saving results between notebooks.

Provided. Each method notebook ends by calling `save_run`, and notebook 05 reads
everything back with `load_runs`. That is the only way the notebooks share
anything: they pass data through `results/`, never code, so each notebook runs on
its own and figures can be redrawn without retraining.

    save_run("toy1d", "mc_dropout", metrics, mean=mean, sd_total=sd_total)
    runs = load_runs("toy1d")
    runs["mc_dropout"]["metrics"]["nll"]
    runs["mc_dropout"]["arrays"]["mean"]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#: Order used for table rows and figure panels, so every notebook agrees.
METHOD_ORDER = ["deterministic", "mc_dropout", "ensemble", "bbb"]


def _as_numpy(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def run_dir(track: str, results_dir: str | Path | None = None) -> Path:
    """Directory holding one track's results. Created if missing."""
    base = Path(results_dir) if results_dir is not None else RESULTS_DIR
    path = base / track.lower()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run(
    track: str,
    method: str,
    metrics: dict[str, float],
    results_dir: str | Path | None = None,
    **arrays,
) -> Path:
    """Write one method's metrics (JSON) and any arrays (npz) for one track.

    Floats are cast with `float()` so numpy scalars do not break JSON.
    """
    path = run_dir(track, results_dir)
    clean = {k: (float(v) if v is not None else None) for k, v in metrics.items()}
    (path / f"{method}.json").write_text(json.dumps(clean, indent=2))
    if arrays:
        np.savez(path / f"{method}.npz", **{k: _as_numpy(v) for k, v in arrays.items()})
    print(f"[store] wrote {path / method}.json" + (" + .npz" if arrays else ""))
    return path


def load_run(
    track: str, method: str, results_dir: str | Path | None = None
) -> dict[str, dict]:
    """Read back one method's results: {"metrics": {...}, "arrays": {...}}."""
    path = run_dir(track, results_dir)
    json_path = path / f"{method}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found. Run the {method} notebook for track {track} first."
        )
    out: dict[str, dict] = {"metrics": json.loads(json_path.read_text()), "arrays": {}}
    npz_path = path / f"{method}.npz"
    if npz_path.exists():
        with np.load(npz_path) as data:
            out["arrays"] = {k: data[k] for k in data.files}
    return out


def load_runs(track: str, results_dir: str | Path | None = None) -> dict[str, dict]:
    """Read back every saved method for one track, in `METHOD_ORDER`."""
    path = run_dir(track, results_dir)
    found = sorted(p.stem for p in path.glob("*.json"))
    ordered = [m for m in METHOD_ORDER if m in found] + [
        m for m in found if m not in METHOD_ORDER
    ]
    return {m: load_run(track, m, results_dir) for m in ordered}
